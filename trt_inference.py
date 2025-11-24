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
        Post-processes the raw output from the YOLOv8 model.
        This version includes refined NMS logic for better filtering.
        """
        output = np.squeeze(output)

        # Filter out detections with a low "objectness" score
        conf_threshold = 0.25  # This is the initial confidence threshold
        output = output[output[:, 4] > conf_threshold]

        boxes = []
        scores = []
        class_ids = []

        for row in output:
            # For each detection, the format is:
            # [cx, cy, w, h, obj_conf, class_prob_1, class_prob_2, ...]
            obj_conf = row[4]
            class_probs = row[5:]
            class_id = np.argmax(class_probs)
            class_conf = class_probs[class_id]
            
            # The final confidence score is a product of objectness and class probability
            final_conf = obj_conf * class_conf

            # We perform a second filter based on this more accurate final confidence
            if final_conf < conf_threshold:
                continue

            scores.append(float(final_conf)) # Use final confidence for NMS
            class_ids.append(int(class_id))

            # Rescale box coordinates from the letterboxed image space back to the original image space
            xc, yc, w, h = row[:4]
            x1 = (xc - w / 2 - dwdh[0]) / ratio
            y1 = (yc - h / 2 - dwdh[1]) / ratio
            width = w / ratio
            height = h / ratio
            
            # The box format required by cv2.dnn.NMSBoxes is (x_top_left, y_top_left, width, height)
            boxes.append([int(x1), int(y1), int(width), int(height)])

        # Apply Non-Maximum Suppression
        nms_threshold = 0.5  # IoU threshold for NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                # Get the box coordinates after NMS
                x, y, w, h = boxes[i]
                final_box = [x, y, x + w, y + h] # Convert back to (x1, y1, x2, y2) for drawing

                detections.append({
                    'box': final_box,
                    'confidence': scores[i],
                    'class_name': self.class_names[class_ids[i]]
                })
        
        return detections
