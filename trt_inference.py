#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
            print("[INFO] CUDA device detected: {}".format(device_name))
        except Exception as e:
            print("[ERROR] Failed to detect CUDA device: {}".format(e))
            raise RuntimeError("CUDA device not available: {}. Please check CUDA installation.".format(e))
        
        self.engine = self._load_engine()
        # Store engine but create execution context lazily per-thread
        # TensorRT execution context is not thread-safe - each thread needs its own context
        self._context = None
        self._context_lock = threading.Lock()
        
        # Load class names FIRST, as they are needed for buffer allocation logic
        self.class_names = self._load_class_names()
        
        # NOTE: Buffers are NOT allocated here anymore
        # Buffers must be allocated per-thread in the same CUDA context as the thread
        # CUDA device memory pointers are only valid within the same CUDA context
        # Buffers will be allocated lazily per-thread in _get_context_and_stream()
        
        # Get model metadata and input shape (needed for preprocessing)
        self.input_shape = self.engine.get_binding_shape(0)
        self.batch_size = self.input_shape[0]
        
        # Get output binding shape for debugging
        for i in range(self.engine.num_bindings):
            if not self.engine.binding_is_input(i):
                self.output_shape = self.engine.get_binding_shape(i)
                print("[DEBUG] Output binding shape: {}".format(self.output_shape))
                print("[DEBUG] Number of classes loaded: {}".format(len(self.class_names)))
                print("[DEBUG] Class names: {}".format(self.class_names))
                break
    
    def _get_context_and_stream(self):
        """
        Get or create CUDA context, TensorRT execution context, and CUDA stream for current thread.
        
        CRITICAL: Main thread should REUSE pycuda.autoinit context instead of creating new one.
        Other threads must create and push their own CUDA context.
        
        Pattern:
        - Main thread: Reuse context from pycuda.autoinit (already created on import)
        - Other threads: Create new CUDA context: cuda.Device(0).make_context()
        - Context is automatically pushed when created
        - Keep context active for thread lifetime
        """
        thread_id = threading.current_thread().ident
        main_thread_id = threading.main_thread().ident
        is_main_thread = (thread_id == main_thread_id)
        
        # Initialize dictionaries if needed
        if not hasattr(self, '_thread_cuda_contexts'):
            self._thread_cuda_contexts = {}  # CUDA contexts per thread
        if not hasattr(self, '_thread_contexts'):
            self._thread_contexts = {}  # TensorRT execution contexts per thread
        if not hasattr(self, '_thread_streams'):
            self._thread_streams = {}  # CUDA streams per thread
        if not hasattr(self, '_thread_buffers'):
            self._thread_buffers = {}  # CUDA buffers per thread (inputs, outputs, bindings)
        
        if thread_id not in self._thread_cuda_contexts:
            # First time this thread calls detect() - need to setup CUDA context
            with self._context_lock:
                # Double-check after acquiring lock
                if thread_id not in self._thread_cuda_contexts:
                    if is_main_thread:
                        print("[DEBUG] Main thread detected - using pycuda.autoinit context (no new context needed)")
                        # CRITICAL: For main thread, pycuda.autoinit already created and pushed a context on import
                        # Based on NVIDIA forums best practices: DO NOT create a new context for main thread
                        # The context from pycuda.autoinit is already active - just proceed with buffer allocation
                        # Store None as marker to indicate we're using pycuda.autoinit context
                        # This prevents us from trying to pop/detach it later (which would cause "context stack not empty" error)
                        self._thread_cuda_contexts[thread_id] = None  # Marker: using pycuda.autoinit
                        print("[DEBUG] Using pycuda.autoinit context for main thread (context already active, no creation needed)")
                    else:
                        print("[DEBUG] Creating CUDA context, TensorRT execution context, CUDA stream, and buffers for thread {}".format(thread_id))
                        # CRITICAL: Create CUDA context for non-main thread
                        try:
                            cuda.init()  # Ensure CUDA is initialized
                            device = cuda.Device(0)
                            cuda_context = device.make_context()
                            # Context is automatically pushed when created
                            self._thread_cuda_contexts[thread_id] = cuda_context
                            print("[DEBUG] CUDA context created and pushed for thread {}".format(thread_id))
                        except Exception as e:
                            print("[ERROR] Failed to create CUDA context for thread {}: {}".format(thread_id, e))
                            raise RuntimeError("CUDA context creation failed: {}".format(e))
                    
                    # CRITICAL: Allocate buffers in THIS thread's CUDA context
                    # CUDA device memory pointers are only valid within the same CUDA context
                    try:
                        inputs, outputs, bindings = self._allocate_buffers()
                        self._thread_buffers[thread_id] = {
                            'inputs': inputs,
                            'outputs': outputs,
                            'bindings': bindings
                        }
                        print("[DEBUG] CUDA buffers allocated for thread {}".format(thread_id))
                    except Exception as e:
                        print("[ERROR] Failed to allocate CUDA buffers for thread {}: {}".format(thread_id, e))
                        # Cleanup CUDA context if buffer allocation fails
                        # CRITICAL: Don't pop pycuda.autoinit context for main thread (it's None)
                        if not is_main_thread and 'cuda_context' in locals() and cuda_context is not None:
                            try:
                                cuda_context.pop()
                                cuda_context.detach()
                            except:
                                pass
                        raise RuntimeError("CUDA buffer allocation failed: {}".format(e))
                    
                    # Create TensorRT execution context (requires active CUDA context)
                    try:
                        trt_context = self.engine.create_execution_context()
                        self._thread_contexts[thread_id] = trt_context
                    except Exception as e:
                        print("[ERROR] Failed to create TensorRT execution context for thread {}: {}".format(thread_id, e))
                        # Cleanup CUDA context if TRT context creation fails
                        # CRITICAL: Don't pop pycuda.autoinit context for main thread (it's None)
                        if not is_main_thread and 'cuda_context' in locals() and cuda_context is not None:
                            try:
                                cuda_context.pop()
                                cuda_context.detach()
                            except:
                                pass
                        raise RuntimeError("TensorRT execution context creation failed: {}".format(e))
                    
                    # Create CUDA stream (requires active CUDA context)
                    try:
                        stream = cuda.Stream()
                        self._thread_streams[thread_id] = stream
                    except Exception as e:
                        print("[ERROR] Failed to create CUDA stream for thread {}: {}".format(thread_id, e))
                        # Cleanup if stream creation fails
                        # CRITICAL: Don't pop pycuda.autoinit context for main thread (it's None)
                        if not is_main_thread and 'cuda_context' in locals() and cuda_context is not None:
                            try:
                                cuda_context.pop()
                                cuda_context.detach()
                            except:
                                pass
                        raise RuntimeError("CUDA stream creation failed: {}".format(e))
        
        return self._thread_contexts[thread_id], self._thread_streams[thread_id], self._thread_buffers[thread_id]

    def _load_class_names(self):
        """Loads class names from a file named class_names.txt in the same directory as the engine."""
        class_names_path = self.engine_path.parent / "class_names.txt"
        print("[INFO] Loading class names from: {}".format(class_names_path))
        try:
            with open(class_names_path, "r") as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        except FileNotFoundError:
            print("[ERROR] '{}' not found. Cannot determine class names.".format(class_names_path))
            raise SystemExit("Aborting: Missing class_names.txt file.")

    def _load_engine(self):
        print("[INFO] Loading TensorRT engine from: {}".format(self.engine_path))
        if not self.engine_path.exists():
            print("[ERROR] Engine file not found at '{}'.".format(self.engine_path))
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
            print("[ERROR] Preprocessing failed: {}".format(e))
            return []
        
        # CRITICAL: Get thread-specific execution context, CUDA stream, and buffers
        # All of these must be per-thread because:
        # - TensorRT execution context is NOT thread-safe
        # - CUDA stream is NOT thread-safe
        # - CUDA device memory buffers are only valid within the same CUDA context
        context, stream, buffers = self._get_context_and_stream()
        inputs = buffers['inputs']
        outputs = buffers['outputs']
        bindings = buffers['bindings']
        
        # CRITICAL: Ensure CUDA context is active before use
        # For main thread: pycuda.autoinit context is already active (no push needed)
        # For other threads: push their context to make it active
        thread_id = threading.current_thread().ident
        cuda_context = self._thread_cuda_contexts.get(thread_id)
        if cuda_context is not None:  # None means using pycuda.autoinit (already active)
            try:
                # Push context for non-main threads (main thread uses pycuda.autoinit, already active)
                cuda_context.push()
            except RuntimeError:
                # Context already active - this is fine
                pass
            except Exception as e:
                print("[WARN] Could not ensure CUDA context is active: {}".format(e))
                # Continue anyway - context might still be active
        # If cuda_context is None, we're using pycuda.autoinit context (already active, no action needed)
        
        # CRITICAL: Serialize CUDA operations with lock to avoid multi-threading issues
        # CUDA context is not thread-safe - all operations must be serialized
        with _cuda_lock:
            # Copy input data to device using thread-specific stream and buffers
            try:
                np.copyto(inputs[0]['host'], input_image.ravel())
                cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
            except Exception as e:
                print("[ERROR] Failed to copy input to device: {}".format(e))
                return []

            # Run inference using thread-specific context, stream, and bindings
            try:
                success = context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                if not success:
                    print("[ERROR] TensorRT inference execution returned False")
                    return []
            except Exception as e:
                print("[ERROR] TensorRT inference error: {}".format(e))
                print("=" * 60)
                print("[ERROR] TensorRT execution error detected!")
                print("[ERROR] This usually means:")
                print("  - CUDA context/stream/buffer conflict in multi-threaded environment")
                print("  - Or engine was built on a different device")
                print("=" * 60)
                print("[INFO] If test_with_images.py works but GUI doesn't:")
                print("  - This is a CUDA threading issue - using per-thread context, stream, and buffers")
                print("[INFO] If both fail, rebuild engine:")
                print("  cd shipping")
                print("  ./rebuild_engine.sh")
                print("=" * 60)
                return []

            # Copy output data from device using thread-specific stream and buffers
            try:
                cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
                stream.synchronize()
            except Exception as e:
                print("[ERROR] Failed to copy output from device: {}".format(e))
                return []

        # Postprocess
        # Use the actual output shape from the engine instead of assuming
        output_data = outputs[0]['host'].reshape(self.output_shape)
        
        # Debug: Print output shape
        print("[DEBUG] Raw output shape: {}".format(output_data.shape))
        print("[DEBUG] Output data dtype: {}".format(output_data.dtype))
        print("[DEBUG] Output data min/max: {:.4f} / {:.4f}".format(output_data.min(), output_data.max()))
        
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
            print("[ERROR] Postprocessing failed: {}".format(e))
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
        print("[DEBUG] _postprocess input shape: {}".format(output.shape))
        
        # Handle different output shapes from TensorRT
        # Common formats: (1, num_attributes, num_predictions) or (1, num_predictions, num_attributes)
        if len(output.shape) == 3:
            # Remove batch dimension and transpose if needed
            # Output is typically (1, num_attributes, num_predictions) or (1, num_predictions, num_attributes)
            squeezed = np.squeeze(output, axis=0)  # Remove batch dimension
            print("[DEBUG] After squeeze: {}".format(squeezed.shape))
            
            # Determine if we need to transpose
            # If shape is (num_attributes, num_predictions), transpose to (num_predictions, num_attributes)
            # If shape is (num_predictions, num_attributes), use as is
            if squeezed.shape[0] < squeezed.shape[1]:
                # Shape is (num_attributes, num_predictions), need to transpose
                predictions = squeezed.T
                print("[DEBUG] Transposed to: {}".format(predictions.shape))
            else:
                # Shape is (num_predictions, num_attributes), use as is
                predictions = squeezed
                print("[DEBUG] Using as is: {}".format(predictions.shape))
        else:
            predictions = output
            print("[DEBUG] Using output directly: {}".format(predictions.shape))

        # Determine number of classes from predictions shape
        # Format should be: [x, y, w, h, class_1, class_2, ..., class_n]
        num_attributes = predictions.shape[1]
        num_classes = num_attributes - 4  # Subtract 4 for box coordinates
        
        print("[DEBUG] Number of attributes per prediction: {}".format(num_attributes))
        print("[DEBUG] Inferred number of classes: {}".format(num_classes))
        print("[DEBUG] Actual number of class names: {}".format(len(self.class_names)))
        
        # Validate class count
        if num_classes != len(self.class_names):
            print("[WARNING] Mismatch: Model has {} classes but class_names.txt has {} classes".format(num_classes, len(self.class_names)))
            print("[WARNING] Using min({}, {}) = {} classes".format(num_classes, len(self.class_names), min(num_classes, len(self.class_names))))
            num_classes = min(num_classes, len(self.class_names))

        # Filter out predictions with confidence lower than threshold.
        # In this format, the confidence score of a prediction is the highest score
        # among its class probabilities.
        conf_threshold = 0.25
        
        # Get the scores for all classes for all predictions.
        class_probs = predictions[:, 4:4+num_classes]
        print("[DEBUG] Class probabilities shape: {}".format(class_probs.shape))
        print("[DEBUG] Class probabilities min/max: {:.4f} / {:.4f}".format(class_probs.min(), class_probs.max()))
        
        # Get the max score for each prediction.
        max_scores = np.max(class_probs, axis=1)
        print("[DEBUG] Max scores shape: {}".format(max_scores.shape))
        print("[DEBUG] Max scores min/max: {:.4f} / {:.4f}".format(max_scores.min(), max_scores.max()))
        print("[DEBUG] Predictions above threshold: {}".format(np.sum(max_scores > conf_threshold)))
        
        # Filter all predictions with a score lower than the threshold.
        mask = max_scores > conf_threshold
        predictions = predictions[mask]
        max_scores = max_scores[mask]

        if predictions.shape[0] == 0:
            print("[DEBUG] No predictions above confidence threshold")
            return []

        # Get the class IDs for the filtered predictions.
        class_ids = np.argmax(predictions[:, 4:4+num_classes], axis=1)
        print("[DEBUG] Class IDs shape: {}".format(class_ids.shape))
        print("[DEBUG] Class IDs min/max: {} / {}".format(class_ids.min(), class_ids.max()))
        print("[DEBUG] Class IDs sample: {}".format(class_ids[:10] if len(class_ids) >= 10 else class_ids))
        
        # Validate class IDs are within range
        invalid_mask = (class_ids >= len(self.class_names)) | (class_ids < 0)
        if np.any(invalid_mask):
            print("[ERROR] Invalid class IDs found: {}".format(class_ids[invalid_mask]))
            print("[ERROR] Valid range: 0 to {}".format(len(self.class_names) - 1))
            # Filter out invalid class IDs
            valid_mask = ~invalid_mask
            predictions = predictions[valid_mask]
            max_scores = max_scores[valid_mask]
            class_ids = class_ids[valid_mask]
            print("[DEBUG] After filtering invalid class IDs: {} predictions".format(len(predictions)))
        
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
        
        print("[DEBUG] NMS indices: {}".format(indices))
        print("[DEBUG] Number of detections after NMS: {}".format(len(indices) if len(indices) > 0 else 0))
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                # Validate class_id before accessing class_names
                if class_ids[i] >= len(self.class_names) or class_ids[i] < 0:
                    print("[ERROR] Invalid class_id {} at index {}, skipping".format(class_ids[i], i))
                    continue
                    
                # Get the final box in (x1, y1, x2, y2) format
                x1, y1, x2, y2 = boxes_rescaled[i]
                detections.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(max_scores[i]),
                    'class_name': self.class_names[class_ids[i]]
                })
        
        print("[DEBUG] Final detections: {}".format(len(detections)))
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
