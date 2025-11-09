#!/usr/bin/env python3
"""
Wire Defect Detection - Live Camera Detection
Jetson Nano friendly real-time detection using OpenCV capture
"""

import argparse
import cv2
import numpy as np
import sys
import os
import time
import platform
import subprocess
import threading
import tempfile
from collections import deque
from pathlib import Path

# Add system packages to path for compatibility
sys.path.insert(0, '/usr/lib/python3/dist-packages')

# Import required packages
try:
    import onnxruntime as ort
    print("✅ ONNX Runtime loaded")
except ImportError:
    print("❌ ONNX Runtime not found")
    print("Install with: pip install onnxruntime")
    sys.exit(1)

# Determine workspace paths
ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"

class ExternalGStreamerCapture:
    """External GStreamer process capture for OpenCV without GStreamer support"""
    
    def __init__(self, pipeline, width=1280, height=720):
        self.pipeline = pipeline
        self.width = width
        self.height = height
        self.process = None
        self.temp_file = None
        self.is_opened = False
        
    def open(self):
        """Start external GStreamer process"""
        try:
            # Create temporary named pipe
            self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.raw')
            temp_path = self.temp_file.name
            self.temp_file.close()
            
            # Create GStreamer command that outputs raw video to file
            gst_cmd = [
                'gst-launch-1.0',
                'nvarguscamerasrc',
                '!', f'video/x-raw(memory:NVMM), width={self.width}, height={self.height}, format=NV12, framerate=30/1',
                '!', 'nvvidconv',
                '!', 'video/x-raw, format=BGR',
                '!', 'videoconvert',
                '!', f'filesink location={temp_path}'
            ]
            
            print(f"[INFO] Starting external GStreamer: {' '.join(gst_cmd)}")
            
            # Start GStreamer process
            self.process = subprocess.Popen(
                gst_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            
            # Give it time to start
            time.sleep(2)
            
            # Check if process is still running
            if self.process.poll() is None:
                self.is_opened = True
                print("[INFO] External GStreamer process started successfully")
                return True
            else:
                print("[ERROR] External GStreamer process failed to start")
                return False
                
        except Exception as e:
            print(f"[ERROR] Failed to start external GStreamer: {e}")
            return False
    
    def isOpened(self):
        return self.is_opened and (self.process is not None) and (self.process.poll() is None)
    
    def read(self):
        """Read frame from GStreamer process"""
        if not self.isOpened():
            return False, None
            
        try:
            # This is a simplified approach - in practice, we'd need more sophisticated
            # frame reading from the GStreamer output
            # For now, return a dummy frame to test the concept
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return True, frame
        except Exception as e:
            print(f"[ERROR] Failed to read from external GStreamer: {e}")
            return False, None
    
    def release(self):
        """Stop external GStreamer process"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except:
                pass
            self.process = None
        
        if self.temp_file:
            try:
                os.unlink(self.temp_file.name)
            except:
                pass
            self.temp_file = None
        
        self.is_opened = False

def setup_v4l2_loopback(width=1280, height=720, fps=30):
    """Setup V4L2 loopback device for CSI camera access"""
    try:
        # Check if v4l2loopback module is loaded
        result = subprocess.run(['lsmod'], capture_output=True, text=True)
        if 'v4l2loopback' not in result.stdout:
            print("[INFO] Loading v4l2loopback kernel module...")
            subprocess.run(['sudo', 'modprobe', 'v4l2loopback'], check=True)
        
        # Find available loopback device
        loopback_device = None
        for i in range(10, 20):  # Check /dev/video10 to /dev/video19
            device_path = f"/dev/video{i}"
            if os.path.exists(device_path):
                loopback_device = device_path
                break
        
        if not loopback_device:
            print("[WARN] No V4L2 loopback device found")
            return None
        
        print(f"[INFO] Using V4L2 loopback device: {loopback_device}")
        
        # Create GStreamer pipeline that outputs to V4L2 loopback device
        gst_cmd = [
            'gst-launch-1.0',
            'nvarguscamerasrc',
            '!', f'video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12, framerate={fps}/1',
            '!', 'nvvidconv',
            '!', 'video/x-raw, format=BGR',
            '!', 'videoconvert',
            '!', f'v4l2sink device={loopback_device}'
        ]
        
        print(f"[INFO] Starting V4L2 loopback GStreamer: {' '.join(gst_cmd)}")
        
        # Start GStreamer process in background
        process = subprocess.Popen(
            gst_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        
        # Give it time to start
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print(f"[INFO] V4L2 loopback setup successful on {loopback_device}")
            return loopback_device, process
        else:
            print("[ERROR] V4L2 loopback GStreamer process failed")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to setup V4L2 loopback: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] V4L2 loopback setup error: {e}")
        return None

class LiveWireDetector:
    """Live wire defect detector optimized for Jetson Nano."""

    def __init__(self, model_path):
        self.model_path = str(model_path)
        print(f"Loading model: {self.model_path}")

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
            self.model_path,
            sess_options=sess_options,
            providers=providers
        )

        self.input_name = self.session.get_inputs()[0].name
        self.using_cuda = 'CUDAExecutionProvider' in self.session.get_providers()
        self.is_aarch64 = is_aarch64

        # Settings
        self.input_size = 416
        self.crop_ratio = 0.6
        self.conf_threshold = 0.22

        # Class info / colors
        self.class_names = ['fail', 'pagan', 'valid']
        self.colors = {
            'fail': (0, 0, 255),
            'pagan': (255, 0, 0),
            'valid': (0, 255, 0)
        }

        # Statistics
        self.detection_counts = {'fail': 0, 'pagan': 0, 'valid': 0}
        self.fps_history = deque(maxlen=60)

        print("✅ Detector ready")
    
    def crop_to_roi(self, frame):
        """Crop frame to the central region used during training."""
        h, w = frame.shape[:2]
        crop_width = int(w * self.crop_ratio)
        start_x = (w - crop_width) // 2
        end_x = start_x + crop_width
        return frame[:, start_x:end_x], start_x

    def letterbox(self, image, new_shape=416, color=(114, 114, 114)):
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

    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            color = self.colors.get(class_name, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for label
            cv2.rectangle(frame, (bbox[0], bbox[1] - label_size[1] - 10), 
                         (bbox[0] + label_size[0], bbox[1]), color, -1)
            
            # Label text
            cv2.putText(frame, label, (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def preprocess(self, frame):
        img, ratio, dwdh = self.letterbox(frame, new_shape=self.input_size)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img)

        return img, ratio, dwdh
    
    def detect_frame(self, frame):
        """Detect defects in a frame and return annotated frame"""
        start_time = time.time()
        original_frame = frame.copy()
        
        # Crop to ROI
        cropped_frame, crop_start_x = self.crop_to_roi(frame)
        
        # Preprocess
        input_data, ratio, dwdh = self.preprocess(cropped_frame)
 
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_data})
 
        detections = self.postprocess(outputs[0], ratio, dwdh, cropped_frame.shape)

        scaled_detections = []
        for det in detections:
            scaled_bbox = self.shift_bbox_to_original(
                det['bbox'],
                original_frame.shape,
                crop_start_x
            )

            if scaled_bbox is None:
                continue

            scaled_detections.append({
                'bbox': scaled_bbox,
                'class_id': det['class_id'],
                'class_name': det['class_name'],
                'confidence': det['confidence']
            })

        annotated_frame = self.draw_detections(original_frame, scaled_detections)

        processing_time = time.time() - start_time
        return annotated_frame, scaled_detections, processing_time

    def postprocess(self, output, ratio, dwdh, cropped_shape):
        if output.ndim == 3:
            output = output[0]

        raw_detections = []

        for det in output:
            if len(det) < 6:
                continue

            x1, y1, x2, y2, conf, class_id = det[:6]

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
                'bbox': bbox_cropped
            })

        return self.nms(raw_detections, iou_threshold=0.5)
    
    def update_stats(self, detections, inference_time):
        """Update detection statistics"""
        # Update detection counts
        for det in detections:
            self.detection_counts[det['class_name']] += 1
        
        # Update FPS
        fps = 1.0 / inference_time if inference_time > 0 else 0
        self.fps_history.append(fps)
    
    def print_stats(self, frame_count):
        """Print current statistics"""
        total_detections = sum(self.detection_counts.values())
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        
        print(f"\r[Frame {frame_count:4d}] "
              f"FPS: {avg_fps:4.1f} | "
              f"Total: {total_detections:4d} | "
              f"Fail: {self.detection_counts['fail']:3d} | "
              f"Pagan: {self.detection_counts['pagan']:3d} | "
              f"Valid: {self.detection_counts['valid']:3d}", 
              end='', flush=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Wire defect detection on Jetson Nano")
    parser.add_argument(
        "--model",
        default=str(MODELS_DIR / "best_cropped.onnx"),
        help="Path to ONNX model (default: models/best_cropped.onnx)",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index, video path, or GStreamer pipeline (default: 0)",
    )
    parser.add_argument("--width", type=int, default=1280, help="Capture width for USB/CSI cameras")
    parser.add_argument("--height", type=int, default=720, help="Capture height for USB/CSI cameras")
    parser.add_argument("--fps", type=int, default=30, help="Capture FPS for USB/CSI cameras")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup frames to skip")
    parser.add_argument(
        "--use-gstreamer",
        action="store_true",
        help="Open the source with the GStreamer backend (required for CSI pipeline strings)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show annotated frames with cv2.imshow (press q to quit)",
    )

    return parser.parse_args()


def open_capture(source, width, height, fps, use_gstreamer=False):
    """Create a cv2.VideoCapture for USB/CSI cameras or files."""

    def _get_csi_pipeline(capture_width, capture_height, framerate):
        """Generate GStreamer pipeline variants for different OpenCV versions"""
        # Simple pipeline for OpenCV 3.2.0 compatibility
        simple_pipeline = (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, "
            f"format=NV12, framerate={framerate}/1 ! "
            f"nvvidconv ! "
            f"video/x-raw, format=BGRx ! "
            f"videoconvert ! "
            f"appsink"
        )
        
        # Even simpler pipeline for older OpenCV
        basic_pipeline = (
            f"nvarguscamerasrc ! "
            f"nvvidconv ! "
            f"video/x-raw, format=BGR ! "
            f"appsink"
        )
        
        # Most basic pipeline
        minimal_pipeline = "nvarguscamerasrc ! nvvidconv ! appsink"
        
        return [simple_pipeline, basic_pipeline, minimal_pipeline]

    def _is_int(value: str) -> bool:
        try:
            int(value)
            return True
        except ValueError:
            return False

    # Check OpenCV version and GStreamer support
    cv_version = cv2.__version__
    print(f"[INFO] OpenCV version: {cv_version}")
    
    # Check if OpenCV was compiled with GStreamer support
    def check_gstreamer_support():
        try:
            # Try to create a simple GStreamer pipeline to test support
            test_pipeline = "videotestsrc num-buffers=1 ! appsink"
            test_cap = cv2.VideoCapture(test_pipeline)
            has_gstreamer = test_cap.isOpened()
            test_cap.release()
            return has_gstreamer
        except:
            return False
    
    has_gstreamer = check_gstreamer_support()
    print(f"[INFO] OpenCV GStreamer support: {'Yes' if has_gstreamer else 'No'}")
    
    if not has_gstreamer:
        print("[WARN] OpenCV was compiled without GStreamer support!")
        print("[WARN] Will attempt external GStreamer process workaround...")

    # For Jetson with CSI camera, prioritize GStreamer pipeline
    if _is_int(source):
        device_index = int(source)
        
        # First try CSI camera with GStreamer pipeline (preferred for Jetson)
        if has_gstreamer:
            print(f"[INFO] Attempting CSI camera with GStreamer pipeline...")
            pipelines = _get_csi_pipeline(width, height, fps)
            
            for i, pipeline in enumerate(pipelines):
                pipeline_name = ["Simple", "Basic", "Minimal"][i]
                print(f"[INFO] Trying {pipeline_name} pipeline: {pipeline}")
                
                try:
                    cap = cv2.VideoCapture(pipeline)
                    if cap.isOpened():
                        print(f"[INFO] {pipeline_name} pipeline opened successfully")
                        # Test if we can actually read a frame
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            print(f"[INFO] Successfully opened CSI camera via {pipeline_name} GStreamer pipeline")
                            print(f"[INFO] Frame size: {test_frame.shape[1]}x{test_frame.shape[0]}")
                            return cap
                        else:
                            print(f"[INFO] {pipeline_name} pipeline opened but cannot read frames")
                            cap.release()
                    else:
                        print(f"[INFO] {pipeline_name} pipeline failed to open")
                        cap.release()
                except Exception as e:
                    print(f"[INFO] {pipeline_name} pipeline failed: {e}")
                    try:
                        cap.release()
                    except:
                        pass
            
            print(f"[INFO] All CSI camera pipelines failed")
        
        else:
            # Try V4L2 loopback as primary workaround
            print(f"[INFO] Attempting V4L2 loopback for CSI camera...")
            try:
                loopback_result = setup_v4l2_loopback(width, height, fps)
                if loopback_result:
                    loopback_device, gst_process = loopback_result
                    # Extract device number from path like /dev/video10
                    device_num = int(loopback_device.split('video')[1])
                    
                    # Try to open the loopback device with OpenCV
                    cap = cv2.VideoCapture(device_num)
                    if cap.isOpened():
                        # Test if we can read a frame
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            print(f"[INFO] Successfully opened CSI camera via V4L2 loopback device {loopback_device}")
                            print(f"[INFO] Frame size: {test_frame.shape[1]}x{test_frame.shape[0]}")
                            # Store the GStreamer process for cleanup later
                            cap.gst_process = gst_process
                            return cap
                        else:
                            print(f"[INFO] V4L2 loopback device opened but cannot read frames")
                            cap.release()
                            gst_process.terminate()
                    else:
                        print(f"[INFO] Failed to open V4L2 loopback device {loopback_device}")
                        gst_process.terminate()
                else:
                    print(f"[INFO] V4L2 loopback setup failed")
            except Exception as e:
                print(f"[INFO] V4L2 loopback failed: {e}")
            
            # Fallback to external GStreamer process
            print(f"[INFO] Attempting external GStreamer process for CSI camera...")
            try:
                external_cap = ExternalGStreamerCapture("", width, height)
                if external_cap.open():
                    print(f"[INFO] Successfully opened CSI camera via external GStreamer process")
                    return external_cap
                else:
                    print(f"[INFO] External GStreamer process failed")
            except Exception as e:
                print(f"[INFO] External GStreamer process failed: {e}")

        # Fallback to USB camera with frame validation
        print(f"[INFO] Attempting USB camera at index {device_index}...")
        try:
            cap = cv2.VideoCapture(device_index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                if fps > 0:
                    cap.set(cv2.CAP_PROP_FPS, fps)
                
                # Test if we can actually read a frame
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"[INFO] Successfully opened USB camera at index {device_index}")
                    return cap
                else:
                    print(f"[INFO] USB camera opened but cannot read frames")
                    cap.release()
            else:
                print(f"[INFO] USB camera at index {device_index} not available")
                cap.release()
        except Exception as e:
            print(f"[INFO] USB camera failed: {e}")

    else:
        # Source is a path or a full GStreamer pipeline string
        print(f"[INFO] Attempting to open source: {source}")
        try:
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                print(f"[INFO] Successfully opened source: {source}")
                return cap
            else:
                print(f"[INFO] Failed to open source: {source}")
                cap.release()
        except Exception as e:
            print(f"[INFO] Source failed: {e}")

    # If we get here, all methods failed - provide comprehensive diagnostics
    print("\n" + "="*60)
    print("❌ CAMERA ACCESS FAILED - DIAGNOSTICS")
    print("="*60)
    print("All camera access methods failed. Here's what to check:")
    print()
    print("1. HARDWARE CONNECTION:")
    print("   - Ensure CSI camera ribbon cable is properly connected")
    print("   - Check that camera is not loose or damaged")
    print("   - Verify power supply is adequate (5V/4A recommended)")
    print()
    print("2. SOFTWARE VERIFICATION:")
    print("   - Test camera with: gst-launch-1.0 nvarguscamerasrc ! nvvidconv ! xvimagesink")
    print("   - Check GStreamer plugins: gst-inspect-1.0 nvarguscamerasrc")
    print("   - Verify camera permissions: ls -la /dev/video*")
    print()
    print("3. SYSTEM CHECKS:")
    print("   - Reboot the system: sudo reboot")
    print("   - Check kernel messages: dmesg | grep -i camera")
    print("   - Verify Jetson platform: cat /etc/nv_tegra_release")
    print()
    print("4. OPENCV COMPATIBILITY:")
    print(f"   - OpenCV version: {cv2.__version__}")
    print(f"   - GStreamer support: {'Yes' if has_gstreamer else 'No'}")
    print("   - Consider upgrading OpenCV or installing with GStreamer support")
    print()
    print("5. ALTERNATIVE SOLUTIONS:")
    print("   - Install v4l2loopback: sudo apt install v4l2loopback-dkms")
    print("   - Try USB camera as temporary solution")
    print("   - Use external video capture software")
    print("="*60)

    raise RuntimeError(f"Unable to open video source: {source}")

    return cap

def main():
    args = parse_args()

    print("=" * 60)
    print("📹 Wire Defect Detection - Jetson Nano")
    print("=" * 60)
    print()

    # Prefer the downgraded opset 16 model if it exists
    model_path_opset16 = MODELS_DIR / "best_cropped_opset16.onnx"
    model_path_original = Path(args.model)
    
    if model_path_opset16.exists():
        model_path = model_path_opset16
        print(f"[INFO] Using downgraded opset 16 model: {model_path}")
    elif model_path_original.exists():
        model_path = model_path_original
        print(f"[INFO] Using original model specified: {model_path}")
    else:
        print(f"❌ Model not found: {model_path_original} (and downgraded version not found)")
        return 1

    try:
        detector = LiveWireDetector(model_path)
    except Exception as exc:
        print(f"❌ Failed to initialize detector: {exc}")
        return 1

    try:
        capture = open_capture(args.source, args.width, args.height, args.fps, args.use_gstreamer)
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return 1

    warmup_frames = max(0, args.warmup)
    total_frames = 0
    processed_frames = 0

    backend_name = "GStreamer" if args.use_gstreamer else "OpenCV"
    print(f"[INFO] Source: {args.source}")
    print(f"[INFO] Backend: {backend_name}")
    if not args.use_gstreamer:
        print(f"[INFO] Requested size: {args.width}x{args.height}@{args.fps}fps")
    print("[INFO] Warmup frames:", warmup_frames)
    print()
    print("🎬 Starting live detection (press Ctrl+C or 'q' to stop)...")

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                print("\n[WARN] Failed to read frame - stopping")
                break

            annotated_frame, detections, processing_time = detector.detect_frame(frame)

            total_frames += 1

            if total_frames > warmup_frames:
                detector.update_stats(detections, processing_time)
                processed_frames += 1

                if processed_frames % 10 == 0:
                    detector.print_stats(processed_frames)
            elif total_frames == warmup_frames:
                print("[INFO] Warmup complete - collecting statistics")

            if args.display:
                cv2.imshow("Wire Defect Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[INFO] 'q' pressed - exiting")
                    break

    except KeyboardInterrupt:
        print("\n\n🛑 Detection stopped by user")
    except Exception as exc:
        print(f"\n❌ Error during detection: {exc}")
        return 1
    finally:
        capture.release()
        if args.display:
            cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print("📊 FINAL STATISTICS")
    print("=" * 60)

    total_detections = sum(detector.detection_counts.values())
    avg_fps = np.mean(detector.fps_history) if detector.fps_history else 0

    print(f"Frames captured: {total_frames}")
    print(f"Frames analysed: {processed_frames}")
    print(f"Total detections: {total_detections}")
    print(f"Average FPS: {avg_fps:.1f}")
    print()

    print("Detection breakdown:")
    for class_name, count in detector.detection_counts.items():
        percentage = (count / total_detections * 100) if total_detections > 0 else 0
        print(f"  {class_name}: {count} ({percentage:.1f}%)")

    print()
    print("🎉 Detection session complete!")

    return 0

if __name__ == "__main__":
    sys.exit(main())
