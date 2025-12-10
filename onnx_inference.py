#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom ONNX Runtime inference module for YOLOv8
Matches TensorRT preprocessing exactly for consistency
"""
import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path

class ONNXDetector:
    """ONNX Runtime detector matching TensorRT preprocessing"""
    
    def __init__(self, onnx_path, class_names_path=None):
        """
        Initialize ONNX detector.
        
        Args:
            onnx_path: Path to ONNX model file
            class_names_path: Path to class_names.txt file
        """
        self.onnx_path = Path(onnx_path)
        self.class_names_path = Path(class_names_path) if class_names_path else None
        
        # Load ONNX model
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, H, W]
        
        # Load class names
        self.class_names = self._load_class_names()
        
        print(f"[INFO] ONNX model loaded: {self.onnx_path}")
        print(f"[INFO] Input shape: {self.input_shape}")
        print(f"[INFO] Loaded {len(self.class_names)} class names")
        
    def _load_class_names(self):
        """Load class names from file"""
        if self.class_names_path and self.class_names_path.exists():
            with open(self.class_names_path, 'r') as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        return []
    
    def _preprocess(self, img):
        """
        Preprocess image for ONNX inference - matches TensorRT preprocessing exactly.
        
        Args:
            img: Input image (numpy array, 3-channel grayscale format)
        
        Returns:
            (preprocessed_image, ratio, (dw, dh))
        """
        input_w, input_h = self.input_shape[3], self.input_shape[2]
        
        # Validate input
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] != 3:
            raise ValueError(f"Expected 3-channel input, got {img.shape[2]} channels")
        
        img_h, img_w, _ = img.shape

        # Letterbox resize to 416x256 maintaining aspect ratio (matches TensorRT)
        r = min(input_w / img_w, input_h / img_h)
        new_unpad = (int(round(img_w * r)), int(round(img_h * r)))
        dw, dh = (input_w - new_unpad[0]) / 2, (input_h - new_unpad[1]) / 2
        
        if (img_w, img_h) != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # HWC to CHW, BGR to RGB, normalize (matches TensorRT)
        img = img.transpose((2, 0, 1))[::-1]  # HWC→CHW, BGR→RGB
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        
        return img, r, (dw, dh)
    
    def _postprocess(self, output, ratio, dwdh, conf_threshold=0.25, nms_threshold=0.5):
        """
        Post-process ONNX output - matches TensorRT postprocessing.
        
        Args:
            output: Model output array
            ratio: Resize ratio from preprocessing
            dwdh: (dw, dh) padding offsets
            conf_threshold: Confidence threshold
            nms_threshold: NMS IoU threshold
        
        Returns:
            List of detection dicts with 'box', 'confidence', 'class_name'
        """
        # Handle output shape (same as TensorRT)
        if len(output.shape) == 3:
            squeezed = np.squeeze(output, axis=0)
            if squeezed.shape[0] < squeezed.shape[1]:
                predictions = squeezed.T
            else:
                predictions = squeezed
        else:
            predictions = output
        
        # Determine number of classes
        num_attributes = predictions.shape[1]
        num_classes = num_attributes - 4
        
        if num_classes != len(self.class_names):
            num_classes = min(num_classes, len(self.class_names))
        
        # Filter by confidence - check each class separately (like Ultralytics)
        class_probs = predictions[:, 4:4+num_classes]
        
        # Get max score per prediction
        max_scores = np.max(class_probs, axis=1)
        
        # Get class IDs for all predictions
        class_ids = np.argmax(class_probs, axis=1)
        
        # Filter: keep predictions where the selected class probability > threshold
        # This matches Ultralytics behavior: check the class-specific probability, not just max
        selected_class_probs = class_probs[np.arange(len(class_probs)), class_ids]
        mask = selected_class_probs > conf_threshold
        
        predictions = predictions[mask]
        max_scores = selected_class_probs[mask]
        class_ids = class_ids[mask]
        
        if predictions.shape[0] == 0:
            return []
        
        # Validate class IDs
        valid_mask = (class_ids >= 0) & (class_ids < len(self.class_names))
        predictions = predictions[valid_mask]
        max_scores = max_scores[valid_mask]
        class_ids = class_ids[valid_mask]
        
        if predictions.shape[0] == 0:
            return []
        
        # Rescale boxes (matches TensorRT)
        boxes_raw = predictions[:, :4]
        boxes_rescaled = self.rescale_boxes(boxes_raw, ratio, dwdh)
        
        # Apply class-specific NMS (like Ultralytics with agnostic_nms=False)
        # This is the default behavior - NMS is applied per class separately
        all_indices = []
        
        for cls_id in range(num_classes):
            # Get predictions for this class
            cls_mask = class_ids == cls_id
            if np.sum(cls_mask) == 0:
                continue
            
            cls_boxes = boxes_rescaled[cls_mask]
            cls_scores = max_scores[cls_mask]
            cls_indices_original = np.where(cls_mask)[0]
            
            # Convert to (x, y, width, height) format for NMSBoxes
            boxes_for_nms = []
            for box in cls_boxes:
                x1, y1, x2, y2 = box
                boxes_for_nms.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
            
            # Apply NMS for this class
            cls_nms_indices = cv2.dnn.NMSBoxes(boxes_for_nms, cls_scores.tolist(), conf_threshold, nms_threshold)
            
            if len(cls_nms_indices) > 0:
                # Map back to original indices
                for idx in cls_nms_indices.flatten():
                    all_indices.append(cls_indices_original[idx])
        
        # Sort by confidence (highest first)
        if len(all_indices) > 0:
            all_indices = np.array(all_indices)
            all_scores = max_scores[all_indices]
            sorted_order = np.argsort(all_scores)[::-1]
            all_indices = all_indices[sorted_order]
        
        detections = []
        for i in all_indices:
            if class_ids[i] >= len(self.class_names) or class_ids[i] < 0:
                continue
            x1, y1, x2, y2 = boxes_rescaled[i]
            
            # Keep float precision (like Ultralytics) - don't round to int
            # Clipping will be handled by caller if needed
            detections.append({
                'box': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(max_scores[i]),
                'class_name': self.class_names[class_ids[i]] if class_ids[i] < len(self.class_names) else f'class_{class_ids[i]}'
            })
        
        return detections
    
    def rescale_boxes(self, boxes, ratio, dwdh):
        """Rescale boxes from letterboxed coordinates to original image - matches TensorRT"""
        dw, dh = dwdh
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
    
    def detect(self, image):
        """
        Detect objects in image using ONNX Runtime.
        
        Args:
            image: Input image (numpy array, 3-channel grayscale format)
        
        Returns:
            List of detection dicts with 'box', 'confidence', 'class_name'
        """
        if image is None or image.size == 0:
            return []
        
        try:
            # Preprocess
            input_image, ratio, dwdh = self._preprocess(image)
            input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
            
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: input_image})
            output = outputs[0]
            
            # Post-process
            detections = self._postprocess(output, ratio, dwdh)
            
            return detections
        except Exception as e:
            print(f"[ERROR] ONNX inference failed: {e}")
            import traceback
            traceback.print_exc()
            return []

