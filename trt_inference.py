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
import threading

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Thread lock for CUDA operations (CUDA context is not thread-safe)
# All CUDA operations must be serialized to avoid "invalid resource handle" errors
_cuda_lock = threading.Lock()

class TRTDetector:
    def __init__(self, engine_path):
        self.engine_path = Path(engine_path)
        
        # pycuda.autoinit already creates CUDA context automatically
        # Don't create another context manually to avoid context stack issues
        # Just verify CUDA is available
        try:
            # Get device info without creating new context
            device = cuda.Device(0)
            device_name = device.name()
            print(f"[INFO] CUDA device detected: {device_name}")
        except Exception as e:
            print(f"[ERROR] Failed to detect CUDA device: {e}")
            raise RuntimeError(f"CUDA device not available: {e}. Please check CUDA installation.")
        
        self.engine = self._load_engine()
        # Store engine but create execution context lazily per-thread
        # TensorRT execution context is not thread-safe - each thread needs its own context
        self._context = None
        self._context_lock = threading.Lock()
        
        # Load class names FIRST, as they are needed for buffer allocation logic
        self.class_names = self._load_class_names()
        
        # Allocate buffers (host and device memory - shared, but access serialized)
        try:
            self.inputs, self.outputs, self.bindings = self._allocate_buffers()
        except Exception as e:
            print(f"[ERROR] Failed to allocate CUDA buffers: {e}")
            raise RuntimeError(f"Buffer allocation failed: {e}")
        
        # Get model metadata and input shape (needed for preprocessing)
        self.input_shape = self.engine.get_binding_shape(0)
        self.batch_size = self.input_shape[0]
        
        # Get output binding shape for debugging
        for i in range(self.engine.num_bindings):
            if not self.engine.binding_is_input(i):
                self.output_shape = self.engine.get_binding_shape(i)
                print(f"[DEBUG] Output binding shape: {self.output_shape}")
                print(f"[DEBUG] Number of classes loaded: {len(self.class_names)}")
                print(f"[DEBUG] Class names: {self.class_names}")
                break
    
    def _get_context_and_stream(self):
        """
        Get or create execution context and CUDA stream for current thread.
        Both TensorRT execution context and CUDA stream are not thread-safe.
        Each thread needs its own context and stream.
        """
        thread_id = threading.current_thread().ident
        if not hasattr(self, '_thread_contexts'):
            self._thread_contexts = {}
        if not hasattr(self, '_thread_streams'):
            self._thread_streams = {}
        
        if thread_id not in self._thread_contexts:
            # Create new execution context and stream for this thread
            with self._context_lock:
                # Double-check after acquiring lock
                if thread_id not in self._thread_contexts:
                    print(f"[DEBUG] Creating TensorRT execution context and CUDA stream for thread {thread_id}")
                    # Create execution context
                    context = self.engine.create_execution_context()
                    # Create CUDA stream for this thread
                    stream = cuda.Stream()
                    self._thread_contexts[thread_id] = context
                    self._thread_streams[thread_id] = stream
        
        return self._thread_contexts[thread_id], self._thread_streams[thread_id]

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
        """
        Allocate CUDA buffers (host and device memory).
        Note: CUDA stream is created per-thread, not here.
        """
        inputs, outputs, bindings = [], [], []
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
        return inputs, outputs, bindings
        
    def detect(self, image):
        """
        Detect objects in image using TensorRT engine.
        
        Args:
            image: Input image (numpy array, BGR format)
        
        Returns:
            List of detection dictionaries with 'box', 'confidence', 'class_name'
        """
        # Validate input
        if image is None or image.size == 0:
            print("[ERROR] Invalid input image")
            return []
        
        try:
            # Preprocess
            input_image, ratio, dwdh = self._preprocess(image)
        except Exception as e:
            print(f"[ERROR] Preprocessing failed: {e}")
            return []
        
        # CRITICAL: Get thread-specific execution context and CUDA stream
        # Both TensorRT execution context and CUDA stream are NOT thread-safe
        # Each thread needs its own context and stream
        context, stream = self._get_context_and_stream()
        
        # CRITICAL: Serialize CUDA operations with lock to avoid multi-threading issues
        # CUDA context is not thread-safe - all operations must be serialized
        with _cuda_lock:
            # Copy input data to device using thread-specific stream
            try:
                np.copyto(self.inputs[0]['host'], input_image.ravel())
                cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], stream)
            except Exception as e:
                print(f"[ERROR] Failed to copy input to device: {e}")
                return []

            # Run inference using thread-specific context and stream
            try:
                success = context.execute_async_v2(bindings=self.bindings, stream_handle=stream.handle)
                if not success:
                    print("[ERROR] TensorRT inference execution returned False")
                    return []
            except Exception as e:
                print(f"[ERROR] TensorRT inference error: {e}")
                print("=" * 60)
                print("[ERROR] TensorRT execution error detected!")
                print("[ERROR] This usually means:")
                print("  - CUDA context/stream conflict in multi-threaded environment")
                print("  - Or engine was built on a different device")
                print("=" * 60)
                print("[INFO] If test_with_images.py works but GUI doesn't:")
                print("  - This is a CUDA threading issue - using per-thread context and stream")
                print("[INFO] If both fail, rebuild engine:")
                print("  cd shipping")
                print("  ./rebuild_engine.sh")
                print("=" * 60)
                return []

            # Copy output data from device using thread-specific stream
            try:
                cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], stream)
                stream.synchronize()
            except Exception as e:
                print(f"[ERROR] Failed to copy output from device: {e}")
                return []

        # Postprocess
        # Use the actual output shape from the engine instead of assuming
        output_data = self.outputs[0]['host'].reshape(self.output_shape)
        
        # Debug: Print output shape
        print(f"[DEBUG] Raw output shape: {output_data.shape}")
        print(f"[DEBUG] Output data dtype: {output_data.dtype}")
        print(f"[DEBUG] Output data min/max: {output_data.min():.4f} / {output_data.max():.4f}")
        
        # Validate output - check if all zeros (likely inference failure)
        if np.all(output_data == 0):
            print("[WARNING] Output is all zeros - inference may have failed!")
            print("[WARNING] This could indicate:")
            print("  - Engine was built on a different device")
            print("  - CUDA context initialization issue")
            print("  - Engine corruption")
            print("[INFO] Try rebuilding the engine on this device: python3 trt_converter.py")
            return []
        
        # Check if output has reasonable values (not all zeros or NaNs)
        if np.any(np.isnan(output_data)) or np.any(np.isinf(output_data)):
            print("[WARNING] Output contains NaN or Inf values - inference may have failed!")
            return []
        
        try:
            detections = self._postprocess(output_data, ratio, dwdh)
        except Exception as e:
            print(f"[ERROR] Postprocessing failed: {e}")
            return []
        
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
        Final, corrected post-processing function based on standard YOLOv8 ONNX examples.
        This version correctly handles the transposed output and NMS logic.
        """
        # Debug: Print output shape before processing
        print(f"[DEBUG] _postprocess input shape: {output.shape}")
        
        # Handle different output shapes from TensorRT
        # Common formats: (1, num_attributes, num_predictions) or (1, num_predictions, num_attributes)
        if len(output.shape) == 3:
            # Remove batch dimension and transpose if needed
            # Output is typically (1, num_attributes, num_predictions) or (1, num_predictions, num_attributes)
            squeezed = np.squeeze(output, axis=0)  # Remove batch dimension
            print(f"[DEBUG] After squeeze: {squeezed.shape}")
            
            # Determine if we need to transpose
            # If shape is (num_attributes, num_predictions), transpose to (num_predictions, num_attributes)
            # If shape is (num_predictions, num_attributes), use as is
            if squeezed.shape[0] < squeezed.shape[1]:
                # Shape is (num_attributes, num_predictions), need to transpose
                predictions = squeezed.T
                print(f"[DEBUG] Transposed to: {predictions.shape}")
            else:
                # Shape is (num_predictions, num_attributes), use as is
                predictions = squeezed
                print(f"[DEBUG] Using as is: {predictions.shape}")
        else:
            predictions = output
            print(f"[DEBUG] Using output directly: {predictions.shape}")

        # Determine number of classes from predictions shape
        # Format should be: [x, y, w, h, class_1, class_2, ..., class_n]
        num_attributes = predictions.shape[1]
        num_classes = num_attributes - 4  # Subtract 4 for box coordinates
        
        print(f"[DEBUG] Number of attributes per prediction: {num_attributes}")
        print(f"[DEBUG] Inferred number of classes: {num_classes}")
        print(f"[DEBUG] Actual number of class names: {len(self.class_names)}")
        
        # Validate class count
        if num_classes != len(self.class_names):
            print(f"[WARNING] Mismatch: Model has {num_classes} classes but class_names.txt has {len(self.class_names)} classes")
            print(f"[WARNING] Using min({num_classes}, {len(self.class_names)}) = {min(num_classes, len(self.class_names))} classes")
            num_classes = min(num_classes, len(self.class_names))

        # Filter out predictions with confidence lower than threshold.
        # In this format, the confidence score of a prediction is the highest score
        # among its class probabilities.
        conf_threshold = 0.25
        
        # Get the scores for all classes for all predictions.
        class_probs = predictions[:, 4:4+num_classes]
        print(f"[DEBUG] Class probabilities shape: {class_probs.shape}")
        print(f"[DEBUG] Class probabilities min/max: {class_probs.min():.4f} / {class_probs.max():.4f}")
        
        # Get the max score for each prediction.
        max_scores = np.max(class_probs, axis=1)
        print(f"[DEBUG] Max scores shape: {max_scores.shape}")
        print(f"[DEBUG] Max scores min/max: {max_scores.min():.4f} / {max_scores.max():.4f}")
        print(f"[DEBUG] Predictions above threshold: {np.sum(max_scores > conf_threshold)}")
        
        # Filter all predictions with a score lower than the threshold.
        mask = max_scores > conf_threshold
        predictions = predictions[mask]
        max_scores = max_scores[mask]

        if predictions.shape[0] == 0:
            print("[DEBUG] No predictions above confidence threshold")
            return []

        # Get the class IDs for the filtered predictions.
        class_ids = np.argmax(predictions[:, 4:4+num_classes], axis=1)
        print(f"[DEBUG] Class IDs shape: {class_ids.shape}")
        print(f"[DEBUG] Class IDs min/max: {class_ids.min()} / {class_ids.max()}")
        print(f"[DEBUG] Class IDs sample: {class_ids[:10] if len(class_ids) >= 10 else class_ids}")
        
        # Validate class IDs are within range
        invalid_mask = (class_ids >= len(self.class_names)) | (class_ids < 0)
        if np.any(invalid_mask):
            print(f"[ERROR] Invalid class IDs found: {class_ids[invalid_mask]}")
            print(f"[ERROR] Valid range: 0 to {len(self.class_names) - 1}")
            # Filter out invalid class IDs
            valid_mask = ~invalid_mask
            predictions = predictions[valid_mask]
            max_scores = max_scores[valid_mask]
            class_ids = class_ids[valid_mask]
            print(f"[DEBUG] After filtering invalid class IDs: {len(predictions)} predictions")
        
        if predictions.shape[0] == 0:
            return []
        
        # Rescale the box coordinates to the original image space.
        boxes_raw = predictions[:, :4]
        boxes_rescaled = self.rescale_boxes(boxes_raw, ratio, dwdh)

        # Apply Non-Maximum Suppression to filter out overlapping boxes.
        nms_threshold = 0.5
        # Convert boxes_rescaled (x1, y1, x2, y2) to (x, y, width, height) for NMSBoxes
        boxes_for_nms = []
        for box in boxes_rescaled:
            x1, y1, x2, y2 = box
            boxes_for_nms.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
        
        indices = cv2.dnn.NMSBoxes(boxes_for_nms, max_scores.tolist(), conf_threshold, nms_threshold)
        
        print(f"[DEBUG] NMS indices: {indices}")
        print(f"[DEBUG] Number of detections after NMS: {len(indices) if len(indices) > 0 else 0}")
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                # Validate class_id before accessing class_names
                if class_ids[i] >= len(self.class_names) or class_ids[i] < 0:
                    print(f"[ERROR] Invalid class_id {class_ids[i]} at index {i}, skipping")
                    continue
                    
                # Get the final box in (x1, y1, x2, y2) format
                x1, y1, x2, y2 = boxes_rescaled[i]
                detections.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(max_scores[i]),
                    'class_name': self.class_names[class_ids[i]]
                })
        
        print(f"[DEBUG] Final detections: {len(detections)}")
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
