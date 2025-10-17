#!/usr/bin/env python3
"""
Debug script to check model output format and fix visualization issues
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Add system packages to path
sys.path.insert(0, '/usr/lib/python3/dist-packages')

try:
    import onnxruntime as ort
    print("✅ ONNX Runtime loaded")
except ImportError:
    print("❌ ONNX Runtime not found")
    sys.exit(1)

def debug_model_output():
    """Debug model output format"""
    print("=" * 60)
    print("🔍 Debugging Model Output Format")
    print("=" * 60)
    
    # Check model
    model_path = "models/best_cropped.onnx"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Load model
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    print("Model input/output info:")
    for input_info in session.get_inputs():
        print(f"  Input: {input_info.name}, Shape: {input_info.shape}")
    
    for output_info in session.get_outputs():
        print(f"  Output: {output_info.name}, Shape: {output_info.shape}")
    
    # Test with dummy data
    print("\nTesting with dummy data...")
    dummy_input = np.random.rand(1, 3, 416, 416).astype(np.float32)
    
    outputs = session.run(None, {'images': dummy_input})
    
    print(f"Number of outputs: {len(outputs)}")
    for i, output in enumerate(outputs):
        print(f"Output {i}: shape={output.shape}, dtype={output.dtype}")
        print(f"  Min: {output.min():.4f}, Max: {output.max():.4f}")
        print(f"  Sample values: {output[0, :5] if len(output.shape) > 1 else output[:5]}")
    
    return outputs[0]

def debug_real_image():
    """Debug with a real test image"""
    print("\n" + "=" * 60)
    print("🖼️  Debugging with Real Image")
    print("=" * 60)
    
    # Find a test image
    test_dir = Path("test_images")
    if not test_dir.exists():
        print("❌ No test_images directory")
        return
    
    image_files = list(test_dir.glob("*.jpg"))
    if not image_files:
        print("❌ No test images found")
        return
    
    image_path = image_files[0]
    print(f"Testing with: {image_path.name}")
    
    # Load and preprocess image
    image = cv2.imread(str(image_path))
    if image is None:
        print("❌ Failed to load image")
        return
    
    print(f"Original image shape: {image.shape}")
    
    # Crop to ROI (60% center)
    h, w = image.shape[:2]
    crop_width = int(w * 0.6)
    start_x = (w - crop_width) // 2
    end_x = start_x + crop_width
    cropped = image[:, start_x:end_x]
    
    print(f"Cropped image shape: {cropped.shape}")
    
    # Resize to 416x416
    resized = cv2.resize(cropped, (416, 416))
    print(f"Resized image shape: {resized.shape}")
    
    # Convert BGR to RGB and normalize
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    
    # Transpose to NCHW
    nchw = np.transpose(normalized, (2, 0, 1))
    batch_input = np.expand_dims(nchw, axis=0)
    
    print(f"Model input shape: {batch_input.shape}")
    
    # Run inference
    session = ort.InferenceSession("models/best_cropped.onnx", providers=['CPUExecutionProvider'])
    outputs = session.run(None, {'images': batch_input})
    
    output = outputs[0]
    print(f"Model output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    
    # Analyze output
    if len(output.shape) == 3:
        output = output[0]  # Remove batch dimension
    
    print(f"After removing batch: {output.shape}")
    
    # Check for valid detections
    valid_detections = 0
    confidences = []
    
    for i, detection in enumerate(output):
        if len(detection) >= 6:
            conf = detection[4]
            confidences.append(conf)
            if conf > 0.1:  # Lower threshold for debugging
                valid_detections += 1
                print(f"Detection {i}: conf={conf:.4f}, class={int(detection[5])}, bbox={detection[:4]}")
    
    print(f"Total detections: {len(output)}")
    print(f"Valid detections (>0.1): {valid_detections}")
    print(f"Confidence range: {min(confidences):.4f} - {max(confidences):.4f}")
    
    # Try different confidence thresholds
    print("\nTrying different confidence thresholds:")
    for threshold in [0.01, 0.05, 0.1, 0.25, 0.5]:
        count = sum(1 for d in output if len(d) >= 6 and d[4] > threshold)
        print(f"  Threshold {threshold}: {count} detections")

def create_fixed_test_script():
    """Create a fixed version of the test script"""
    print("\n" + "=" * 60)
    print("🔧 Creating Fixed Test Script")
    print("=" * 60)
    
    fixed_script = '''#!/usr/bin/env python3
"""
Fixed Wire Defect Detection - Image Testing
Debugged version with proper model output handling
"""

import cv2
import numpy as np
import sys
import os
import time
from pathlib import Path

# Add system packages to path
sys.path.insert(0, '/usr/lib/python3/dist-packages')

try:
    import onnxruntime as ort
except ImportError:
    print("❌ ONNX Runtime not found")
    sys.exit(1)

class FixedWireDetector:
    """Fixed wire defect detector"""
    
    def __init__(self, model_path):
        print(f"Loading model: {model_path}")
        
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Model settings
        self.input_size = 416
        self.crop_ratio = 0.6
        self.conf_threshold = 0.1  # Lower threshold for debugging
        
        # Class info
        self.class_names = ['fail', 'pagan', 'valid']
        self.colors = {
            'fail': (0, 0, 255),    # Red
            'pagan': (255, 0, 0),   # Blue  
            'valid': (0, 255, 0)    # Green
        }
        
        print("✅ Model loaded")
    
    def crop_to_roi(self, image):
        """Crop image to center 60% width"""
        h, w = image.shape[:2]
        crop_width = int(w * self.crop_ratio)
        start_x = (w - crop_width) // 2
        end_x = start_x + crop_width
        return image[:, start_x:end_x]
    
    def preprocess(self, image):
        """Preprocess image for model input"""
        img = cv2.resize(image, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img
    
    def postprocess(self, output):
        """Extract detections with debugging"""
        detections = []
        
        print(f"Raw output shape: {output.shape}")
        
        if len(output.shape) == 3:
            output = output[0]
        
        print(f"Processed output shape: {output.shape}")
        
        for i, detection in enumerate(output):
            if len(detection) >= 6:
                conf = detection[4]
                class_id = int(detection[5])
                
                if conf > self.conf_threshold:
                    # Convert normalized coordinates
                    x_center, y_center, width, height = detection[:4]
                    x1 = int((x_center - width/2) * self.input_size)
                    y1 = int((y_center - height/2) * self.input_size)
                    x2 = int((x_center + width/2) * self.input_size)
                    y2 = int((y_center + height/2) * self.input_size)
                    
                    detections.append({
                        'class_id': class_id,
                        'class_name': self.class_names[class_id],
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
                    
                    print(f"Detection {i}: {self.class_names[class_id]} conf={conf:.3f} bbox=[{x1},{y1},{x2},{y2}]")
        
        return detections
    
    def detect_image(self, image_path):
        """Detect defects in image with debugging"""
        print(f"\\nProcessing: {image_path.name}")
        
        image = cv2.imread(str(image_path))
        if image is None:
            return None, []
        
        # Crop to ROI
        cropped_image = self.crop_to_roi(image)
        
        # Preprocess
        input_data = self.preprocess(cropped_image)
        
        # Run inference
        outputs = self.session.run(None, {'images': input_data})
        detections = self.postprocess(outputs[0])
        
        # Draw results
        result_image = cropped_image.copy()
        for det in detections:
            bbox = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            color = self.colors[class_name]
            
            # Draw bounding box
            cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(result_image, label, (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result_image, detections

def main():
    """Test with debugging"""
    print("🔧 Fixed Wire Defect Detection - Debug Mode")
    
    model_path = "models/best_cropped.onnx"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return 1
    
    detector = FixedWireDetector(model_path)
    
    test_dir = Path("test_images")
    image_files = list(test_dir.glob("*.jpg"))[:3]  # Test first 3 images
    
    for image_path in image_files:
        result_image, detections = detector.detect_image(image_path)
        
        if result_image is not None:
            output_path = f"debug_result_{image_path.name}"
            cv2.imwrite(output_path, result_image)
            print(f"✅ Saved: {output_path} with {len(detections)} detections")
        else:
            print(f"❌ Failed to process {image_path.name}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''
    
    with open("debug_test_fixed.py", "w") as f:
        f.write(fixed_script)
    
    print("✅ Created debug_test_fixed.py")
    print("Run: python debug_test_fixed.py")

def main():
    """Main debug function"""
    print("🔍 Wire Defect Detection - Debug Tool")
    print("This will help identify why bounding boxes are not showing")
    
    # Debug model output format
    debug_model_output()
    
    # Debug with real image
    debug_real_image()
    
    # Create fixed test script
    create_fixed_test_script()
    
    print("\n" + "=" * 60)
    print("🎯 Next Steps:")
    print("1. Run: python debug_test_fixed.py")
    print("2. Check the debug output for model format issues")
    print("3. Adjust confidence threshold if needed")
    print("4. Verify bounding box coordinates are valid")
    print("=" * 60)

if __name__ == "__main__":
    main()
