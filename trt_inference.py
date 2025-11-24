#!/usr/bin/env python3
"""
Task 18, Phase 3: TensorRT Inference Module (Re-created)
- Contains the core logic for running inference with a TensorRT engine.
- Encapsulates model loading, pre-processing, inference, and post-processing.
- Loads class names dynamically from a file to ensure synchronization with the model.
"""

import tensorrt as trt
import numpy as np
import cv2
import pycuda.autoinit
import pycuda.driver as cuda
from pathlib import Path

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTDetector:
    def __init__(self, engine_path):
        self.engine_path = Path(engine_path)
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        
        # Load class names FIRST, as they are needed for buffer allocation logic
        self.class_names = self._load_class_names()
        
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        
        # Get model metadata and input shape
        self.input_shape = self.engine.get_binding_shape(0)
        self.batch_size = self.input_shape[0]

    def _load_class_names(self):
        """Loads class names from a file named class_names.txt in the same directory as the engine."""
        class_names_path = self.engine_path.parent / "class_names.txt"
        print(f"[INFO] Loading class names from: {class_names_path}")
        try:
            with open(class_names_path, "r") as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        except FileNotFoundError:
            print(f"[ERROR] '{class_names_path}' not found. Cannot determine class names.")
            raise SystemExit("Aborting: Missing class_names.txt file.")

    def _load_engine(self):
        print(f"[INFO] Loading TensorRT engine from: {self.engine_path}")
        if not self.engine_path.exists():
            print(f"[ERROR] Engine file not found at '{self.engine_path}'.")
            raise SystemExit("Aborting: Please generate the .engine file first.")
        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        inputs, outputs, bindings, stream = [], [], [], cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        return inputs, outputs, bindings, stream
        
    def detect(self, image):
        # Preprocess
        input_image, ratio, dwdh = self._preprocess(image)
        
        # Copy input data to device
        np.copyto(self.inputs[0]['host'], input_image.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # Copy output data from device
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()

        # Postprocess
        # THIS IS THE CRITICAL FIX: The output shape is calculated dynamically
        # using the number of classes we loaded from the file.
        output_shape = (self.batch_size, -1, len(self.class_names) + 4) # 4 for box coords
        output_data = self.outputs[0]['host'].reshape(output_shape)
        
        detections = self._postprocess(output_data, ratio, dwdh)
        return detections

    def _preprocess(self, img):
        input_w, input_h = self.input_shape[3], self.input_shape[2]
        img_h, img_w, _ = img.shape

        # Letterbox
        r = min(input_w / img_w, input_h / img_h)
        new_unpad = (int(round(img_w * r)), int(round(img_h * r)))
        dw, dh = (input_w - new_unpad[0]) / 2, (input_h - new_unpad[1]) / 2
        
        if (img_w, img_h) != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # HWC to CHW, BGR to RGB, normalize
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        return img, r, (dw, dh)

    def _postprocess(self, output, ratio, dwdh):
        """
        Post-processes with step-by-step logging to debug data transformations.
        """
        print("\n--- POST-PROCESSING DEBUG ---")
        
        # Step 1: Initial output from model
        print(f"1. Initial output shape: {output.shape}")
        # Expected: (1, 10, 8400)

        # Step 2: Squeeze to remove batch dimension
        output = np.squeeze(output)
        print(f"2. Shape after squeeze: {output.shape}")
        # Expected: (10, 8400)

        # Step 3: Transpose to get predictions per row
        # Each row will now represent one of the 8400 predictions
        output = output.T
        print(f"3. Shape after transpose: {output.shape}")
        # Expected: (8400, 10)

        # Step 4: Extract scores and filter
        # The last 6 columns are class probabilities
        class_probs = output[:, 4:]
        # The score for each prediction is the highest probability among its classes
        max_scores = np.max(class_probs, axis=1)
        
        conf_threshold = 0.25
        mask = max_scores > conf_threshold
        
        filtered_output = output[mask]
        filtered_scores = max_scores[mask]
        
        print(f"4. Found {output.shape[0]} total predictions.")
        print(f"   - Found {filtered_output.shape[0]} predictions above conf_threshold {conf_threshold}")

        if filtered_output.shape[0] == 0:
            print("--- END DEBUG ---")
            return []

        # Step 5: Get class IDs for filtered predictions
        # This is where the error occurred. Let's inspect the data fed into it.
        filtered_class_probs = filtered_output[:, 4:]
        class_ids = np.argmax(filtered_class_probs, axis=1)
        
        print(f"5. Shape of class probabilities array for argmax: {filtered_class_probs.shape}")
        # Expected: (num_filtered_predictions, 6)
        print(f"   - Min class ID found: {np.min(class_ids)}, Max class ID found: {np.max(class_ids)}")

        # If max class ID > 5, there is a fundamental logic error.
        if np.max(class_ids) > (len(self.class_names) - 1):
            print(f"   - ❌ CRITICAL ERROR: Found a class ID ({np.max(class_ids)}) greater than the max allowed index ({len(self.class_names) - 1}).")
            print("     This indicates the model's output does not match the expected 6 classes.")
            print("--- END DEBUG ---")
            # Return empty to prevent crash, but the root cause is a model mismatch.
            return []

        # Step 6: Process boxes for NMS
        boxes_raw = filtered_output[:, :4]
        boxes_rescaled = self.rescale_boxes(boxes_raw, ratio, dwdh)

        boxes_for_nms = []
        for box in boxes_rescaled:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            boxes_for_nms.append([int(x1), int(y1), int(width), int(height)])
        
        print(f"6. Prepared {len(boxes_for_nms)} boxes for NMS.")

        # Step 7: Apply NMS
        nms_threshold = 0.5
        indices = cv2.dnn.NMSBoxes(boxes_for_nms, filtered_scores.tolist(), conf_threshold, nms_threshold)
        print(f"7. NMS applied. Found {len(indices)} final detections.")
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                final_box = boxes_for_nms[i]
                final_box[2] += final_box[0] # Convert w,h back to x2,y2 for drawing
                final_box[3] += final_box[1]

                detections.append({
                    'box': final_box,
                    'confidence': filtered_scores[i],
                    'class_name': self.class_names[class_ids[i]]
                })
        
        print("--- END DEBUG ---")
        return detections

    def rescale_boxes(self, boxes, ratio, dwdh):
        """Helper function to rescale letterboxed boxes to original image space."""
        dw, dh = dwdh
        # Rescale centers and dimensions
        boxes[:, 0] = (boxes[:, 0] - dw) / ratio
        boxes[:, 1] = (boxes[:, 1] - dh) / ratio
        boxes[:, 2] = boxes[:, 2] / ratio
        boxes[:, 3] = boxes[:, 3] / ratio
        
        # Convert from (center_x, center_y, width, height) to (x1, y1, x2, y2)
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        return np.stack([x1, y1, x2, y2], axis=1)
