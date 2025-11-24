#!/usr/bin/env python3
"""
Task 18, Phase 3: TensorRT Inference Module
- Contains the core logic for running inference with a TensorRT engine.
- Encapsulates model loading, pre-processing, inference, and post-processing.
"""

import tensorrt as trt
import numpy as np
import cv2
import pycuda.autoinit
import pycuda.driver as cuda

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTDetector:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        
        # Get model metadata and input shape
        self.meta = self.engine.get_binding_name(0) # A way to get metadata if needed, though often not used in TRT directly
        self.input_shape = self.engine.get_binding_shape(0)
        self.batch_size = self.input_shape[0]
        self.class_names = {0: 'fail', 1: 'pagan', 2: 'valid', 3: 'NOK', 4: 'breaks', 5: 'damage', 6: 'drops', 7: 'normal', 8: 'shift'} # Placeholder, will need a better way if this changes

    def _load_engine(self):
        print(f"[INFO] Loading TensorRT engine from: {self.engine_path}")
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
        output_data = self.outputs[0]['host'].reshape(self.batch_size, -1, len(self.class_names) + 4)
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
        output = np.squeeze(output).T
        boxes, scores, class_ids = [], [], []
        
        # Transpose and filter
        conf_threshold = 0.25
        output = output[output[:, 4] > conf_threshold]

        for row in output:
            class_id = np.argmax(row[4:])
            prob = row[4 + class_id]
            
            xc, yc, w, h = row[:4]
            x1 = (xc - w / 2 - dwdh[0]) / ratio
            y1 = (yc - h / 2 - dwdh[1]) / ratio
            x2 = (xc + w / 2 - dwdh[0]) / ratio
            y2 = (yc + h / 2 - dwdh[1]) / ratio
            
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
            scores.append(float(prob))
            class_ids.append(int(class_id))
            
        # NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, 0.7)
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append({
                    'class_name': self.class_names[class_ids[i]],
                    'confidence': scores[i],
                    'box': boxes[i]
                })
        return detections
