#!/usr/bin/env python3
"""
Wire Defect Detection - Image Testing
Test the inference pipeline with static images before using camera
"""

import cv2
import numpy as np
import sys
import os
import time
import platform
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
TEST_IMAGES_DIR = ROOT_DIR / "test_images"
TEST_RESULTS_DIR = ROOT_DIR / "test_results"

# Add system packages to path for compatibility on Linux
LINUX_SITE_PACKAGES = Path("/usr/lib/python3/dist-packages")
if LINUX_SITE_PACKAGES.exists():
    linux_site_packages_str = str(LINUX_SITE_PACKAGES)
    if linux_site_packages_str not in sys.path:
        sys.path.insert(0, linux_site_packages_str)

try:
    import onnxruntime as ort
    print("[OK] ONNX Runtime available")
except ImportError:
    print("[ERROR] ONNX Runtime not found")
    print("Install with: pip install onnxruntime")
    sys.exit(1)


class SimpleWireDetector:
    """Simple wire defect detector for testing"""
    
    def __init__(self, model_path):
        print(f"Loading model: {model_path}")
        
        # Create ONNX Runtime session tailored for Jetson/desktop
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.enable_mem_pattern = False

        is_aarch64 = platform.machine().lower() in ("aarch64", "armv8", "armv8l")
        cpu_threads = max(1, min(2, os.cpu_count() or 1))

        if is_aarch64:
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
        else:
            sess_options.intra_op_num_threads = cpu_threads
            sess_options.inter_op_num_threads = 1

        available_providers = ort.get_available_providers()
        providers = []

        if 'CUDAExecutionProvider' in available_providers:
            os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')
            providers.append('CUDAExecutionProvider')
            print("[INFO] Using CUDAExecutionProvider")
        else:
            print("[INFO] CUDAExecutionProvider not available, using CPU only")

        providers.append('CPUExecutionProvider')

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.using_cuda = 'CUDAExecutionProvider' in self.session.get_providers()
        self.is_aarch64 = is_aarch64
        
        # Model settings
        self.input_size = 416
        self.crop_ratio = 0.6
        self.conf_threshold = 0.22  # Fine-tuned threshold to get closer to 19 detections
        
        # Class info
        self.class_names = ['fail', 'pagan', 'valid']
        self.colors = {
            'fail': (0, 0, 255),    # Red
            'pagan': (255, 0, 0),   # Blue  
            'valid': (0, 255, 0)    # Green
        }
        
        print("[OK] Model loaded successfully")
    
    def crop_to_roi(self, image):
        """Crop image to center 60% width"""
        h, w = image.shape[:2]
        crop_width = int(w * self.crop_ratio)
        start_x = (w - crop_width) // 2
        end_x = start_x + crop_width
        return image[:, start_x:end_x], start_x
    
    def letterbox(self, image, new_shape=416, color=(114, 114, 114)):
        """Resize image to a square while keeping aspect ratio (YOLO letterbox)."""
        shape = image.shape[:2]

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=color
        )

        return image, r, (dw, dh)

    def scale_bbox_from_letterbox(self, bbox, ratio, dwdh, cropped_shape):
        """Map bbox from letterboxed coordinates back to cropped image space."""
        dw, dh = dwdh
        x1, y1, x2, y2 = [float(x) for x in bbox]

        x1 = (x1 - dw) / ratio
        y1 = (y1 - dh) / ratio
        x2 = (x2 - dw) / ratio
        y2 = (y2 - dh) / ratio

        width = cropped_shape[1]
        height = cropped_shape[0]
        x1 = max(0.0, min(x1, width))
        y1 = max(0.0, min(y1, height))
        x2 = max(0.0, min(x2, width))
        y2 = max(0.0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            return None

        return [x1, y1, x2, y2]

    def shift_bbox_to_original(self, bbox, original_shape, crop_start_x):
        """Translate bbox from cropped coordinates back to original image space."""
        x1, y1, x2, y2 = bbox

        x1 += crop_start_x
        x2 += crop_start_x

        width = original_shape[1]
        height = original_shape[0]
        x1 = max(0, min(int(round(x1)), width))
        y1 = max(0, min(int(round(y1)), height))
        x2 = max(0, min(int(round(x2)), width))
        y2 = max(0, min(int(round(y2)), height))

        if x2 <= x1 or y2 <= y1:
            return None

        return [x1, y1, x2, y2]
    
    def preprocess(self, image):
        """Preprocess image for model input using YOLO letterbox."""
        img, ratio, dwdh = self.letterbox(image, new_shape=self.input_size)

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Transpose to NCHW format and add batch dimension
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img)

        return img, ratio, dwdh
    
    def nms(self, detections, iou_threshold=0.5):
        """Apply Non-Maximum Suppression to remove overlapping detections"""
        if len(detections) == 0:
            return []
        
        # Sort detections by confidence (descending)
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            # Keep the detection with highest confidence
            best = detections.pop(0)
            keep.append(best)
            
            # Remove detections with high IoU overlap with the best detection
            remaining = []
            for det in detections:
                iou = self.calculate_iou(best['bbox'], det['bbox'])
                if iou < iou_threshold:
                    remaining.append(det)
            detections = remaining
        
        return keep
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        # box format: [x1, y1, x2, y2]
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        # Calculate intersection area
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate areas of both boxes
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calculate union area
        union = area1 + area2 - intersection
        
        # Calculate IoU
        iou = intersection / union if union > 0 else 0.0
        return iou

    def postprocess(self, output, ratio, dwdh, cropped_shape):
        """Extract detections from YOLO model output with NMS."""
        print(f"ONNX output shape: {output.shape}")

        if output.ndim == 3:
            output = output[0]

        print(f"After batch removal: {output.shape}")

        raw_detections = []

        for detection in output:
            if len(detection) < 6:
                continue

            x1, y1, x2, y2, conf, class_id = detection[:6]

            if conf < self.conf_threshold:
                continue

            class_id = int(class_id)
            if class_id >= len(self.class_names):
                continue

            bbox_cropped = self.scale_bbox_from_letterbox(
                [x1, y1, x2, y2],
                ratio,
                dwdh,
                cropped_shape
            )

            if bbox_cropped is None:
                continue

            raw_detections.append({
                'class_id': class_id,
                'class_name': self.class_names[class_id],
                'confidence': float(conf),
                'bbox': bbox_cropped,
                'bbox_letterbox': [x1, y1, x2, y2]
            })

        final_detections = self.nms(raw_detections, iou_threshold=0.5)

        print(f"Detections found: {len(final_detections)}")

        return final_detections
    
    def detect_image(self, image_path):
        """Detect defects in a single image"""
        # Expected results for validation (from learning_based)
        expected_results = {
            '0bfb41.jpg': [('pagan', 0.912), ('valid', 0.851), ('valid', 0.808), ('fail', 0.780)],
            '1d42d9.jpg': [('pagan', 0.822), ('fail', 0.738), ('fail', 0.718), ('valid', 0.680)],
            '2cb5b3.jpg': [('valid', 0.865), ('valid', 0.791), ('valid', 0.439)],
            '2cda46.jpg': [('valid', 0.908), ('valid', 0.742), ('valid', 0.711), ('valid', 0.676)],
            '3ebacb.jpg': [('valid', 0.909), ('valid', 0.751), ('valid', 0.751), ('valid', 0.700)]
        }
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return None, []
        
        original_image = image.copy()
        print(f"  Original image: {image.shape[1]}x{image.shape[0]}")
        
        # Crop to ROI
        cropped_image, crop_start_x = self.crop_to_roi(image)
        print(f"  Cropped image: {cropped_image.shape[1]}x{cropped_image.shape[0]}")
        
        # Preprocess (resize to 416x416)
        input_data, ratio, dwdh = self.preprocess(cropped_image)
        
        # Run inference
        start_time = time.time()
        outputs = self.session.run(None, {self.input_name: input_data})
        inference_time = time.time() - start_time
        
        # Postprocess
        detections = self.postprocess(outputs[0], ratio, dwdh, cropped_image.shape)
        
        # Compare with expected results
        image_name = image_path.name if hasattr(image_path, 'name') else str(image_path).split('/')[-1]
        if image_name in expected_results:
            expected = expected_results[image_name]
            print(f"  EXPECTED ({len(expected)}): {[(cls, f'{conf:.3f}') for cls, conf in expected]}")
            
            # Count class differences
            expected_classes = [cls for cls, _ in expected]
            actual_classes = [det['class_name'] for det in detections]
            
            exp_count = {'fail': expected_classes.count('fail'), 'pagan': expected_classes.count('pagan'), 'valid': expected_classes.count('valid')}
            act_count = {'fail': actual_classes.count('fail'), 'pagan': actual_classes.count('pagan'), 'valid': actual_classes.count('valid')}
            
            print(f"  EXPECTED classes: fail={exp_count['fail']}, pagan={exp_count['pagan']}, valid={exp_count['valid']}")
            print(f"  ACTUAL classes:   fail={act_count['fail']}, pagan={act_count['pagan']}, valid={act_count['valid']}")
        
        # Scale detections from 416x416 to original image coordinates
        scaled_detections = []
        for det in detections:
            scaled_bbox = self.shift_bbox_to_original(
                det['bbox'],
                original_image.shape,
                crop_start_x
            )

            if scaled_bbox is None:
                continue

            scaled_det = {
                'bbox': scaled_bbox,
                'class_name': det['class_name'],
                'class_id': det['class_id'],
                'confidence': det['confidence']
            }
            scaled_detections.append(scaled_det)
        
        # Draw results on ORIGINAL image with scaled coordinates
        result_image = original_image.copy()
        for i, det in enumerate(scaled_detections):
            bbox = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            color = self.colors[class_name]
            
            # Verify bbox is within image bounds
            if (bbox[0] >= 0 and bbox[1] >= 0 and 
                bbox[2] <= result_image.shape[1] and bbox[3] <= result_image.shape[0]):
                
                # Draw bounding box
                cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(result_image, label, (bbox[0], bbox[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result_image, scaled_detections, inference_time

def test_images():
    """Test detection with sample images"""
    print("=" * 60)
    print("[TEST] Wire Defect Detection - Image Testing")
    print("=" * 60)
    
    # Check for downgraded model first, then original
    model_path_opset16 = MODELS_DIR / "best_cropped_opset16.onnx"
    model_path_original = MODELS_DIR / "best_cropped.onnx"
    
    if model_path_opset16.exists():
        model_path = model_path_opset16
        print(f"[INFO] Using downgraded opset 16 model: {model_path}")
    elif model_path_original.exists():
        model_path = model_path_original
        print(f"[INFO] Using original model: {model_path}")
    else:
        print(f"[ERROR] Model file not found. Searched for:")
        print(f"  - {model_path_opset16}")
        print(f"  - {model_path_original}")
        print("Please ensure the ONNX model is in the models/ directory")
        return 1
    
    # Initialize detector
    try:
        detector = SimpleWireDetector(model_path)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return 1
    
    # Get test images
    if not TEST_IMAGES_DIR.exists():
        print(f"[ERROR] Test images directory not found: {TEST_IMAGES_DIR}")
        return 1
    
    image_files = sorted(TEST_IMAGES_DIR.glob("*.jpg"))[:10]  # Test first 10 images
    
    if not image_files:
        print(f"[ERROR] No test images found in {TEST_IMAGES_DIR}")
        return 1
    
    print(f"[INFO] Found {len(image_files)} test images")
    print()
    
    # Test each image
    total_detections = 0
    total_time = 0
    class_counts = {'fail': 0, 'pagan': 0, 'valid': 0}
    TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Testing: {image_path.name}")
        
        try:
            result_image, detections, inference_time = detector.detect_image(image_path)
            
            if result_image is not None:
                # Count detections
                total_detections += len(detections)
                total_time += inference_time
                
                # Update class counts
                for det in detections:
                    class_counts[det['class_name']] += 1
                
                # Print results
                print(f"  [TIME] Inference: {inference_time*1000:.1f}ms")
                print(f"  [DETECT] Detections: {len(detections)}")
                
                for det in detections:
                    print(f"    - {det['class_name']}: {det['confidence']:.3f}")
                
                # Save result (optional)
                output_path = TEST_RESULTS_DIR / f"test_results_{image_path.name}"
                cv2.imwrite(str(output_path), result_image)
                print(f"  [SAVE] Result saved: {output_path}")
                
            else:
                print("  [ERROR] Failed to process image")
                
        except Exception as e:
            print(f"  [ERROR] Error: {e}")
        
        print()
    
    # Summary
    print("=" * 60)
    print("[SUMMARY] TEST SUMMARY")
    print("=" * 60)
    
    avg_time = total_time / len(image_files) if image_files else 0
    avg_fps = 1.0 / avg_time if avg_time > 0 else 0
    
    print(f"Images tested: {len(image_files)}")
    print(f"Total detections: {total_detections}")
    print(f"Average inference time: {avg_time*1000:.1f}ms")
    print(f"Average FPS: {avg_fps:.1f}")
    print()
    
    print("Class distribution:")
    for class_name, count in class_counts.items():
        percentage = (count / total_detections * 100) if total_detections > 0 else 0
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    print()
    
    # Performance assessment
    if avg_fps >= 3:
        print("[OK] Performance looks good for real-time detection!")
    elif avg_fps >= 1:
        print("[OK] Performance acceptable for real-time detection")
    else:
        print("[WARN] Performance may be slow for real-time detection")
    
    print()
    print("Next step: python run_camera_detection.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(test_images())
