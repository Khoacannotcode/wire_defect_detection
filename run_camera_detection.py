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

# Camera tuning parameters (matching simple_recorder.py successful settings)
CAMERA_EXPOSURE_TIME = 250000  # microseconds
CAMERA_ANALOG_GAIN = 7.0
ISP_DIGITAL_GAIN = 1.0
TNR_MODE = 0               # 0=Off, 1=Fast, 2=HighQuality
TNR_STRENGTH = 0.4         # -1 (auto) to 1
EE_MODE = 0                # 0=Off, 1=Fast, 2=HighQuality
EE_STRENGTH = 0.2          # -1 (auto) to 1
EXPOSURE_COMPENSATION = -2.0
SENSOR_MODE = 5            # 1280x720 @120fps

# Combine all settings into a GStreamer-compatible property string
CAMERA_PROPERTY_STRING = (
    'sensor-mode={} '
    'exposuretimerange="{} {}" '
    'gainrange="{} {}" '
    'ispdigitalgainrange="{} {}" '
    'tnr-mode={} tnr-strength={} '
    'ee-mode={} ee-strength={} '
    'exposurecompensation={}'
).format(
    SENSOR_MODE,
    CAMERA_EXPOSURE_TIME, CAMERA_EXPOSURE_TIME,
    CAMERA_ANALOG_GAIN, CAMERA_ANALOG_GAIN,
    ISP_DIGITAL_GAIN, ISP_DIGITAL_GAIN,
    TNR_MODE, TNR_STRENGTH,
    EE_MODE, EE_STRENGTH,
    EXPOSURE_COMPENSATION
)

# Optional PyGObject / GStreamer bindings
GST_AVAILABLE = False
try:
    import gi  # type: ignore

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst  # type: ignore

    if not Gst.is_initialized():
        Gst.init(None)
    GST_AVAILABLE = True
except (ImportError, ValueError):
    GST_AVAILABLE = False

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

# Import visualization standards
try:
    # Try to import from learning_based directory
    sys.path.insert(0, str(Path(__file__).parent.parent / 'learning_based'))
    from visualization_standards import get_class_color
    VISUALIZATION_STANDARDS_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: visualization_standards.py not found, using default colors")
    VISUALIZATION_STANDARDS_AVAILABLE = False
    def get_class_color(class_name):
        """Fallback color function"""
        color_map = {
            'NOK': (0, 165, 255), 'breaks': (0, 255, 255), 'damage': (0, 0, 255),
            'drops': (128, 128, 128), 'normal': (0, 128, 0), 'shift': (255, 0, 0)
        }
        return color_map.get(class_name, (128, 128, 128))

# Determine workspace paths
ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"

class UDPGStreamerCapture:
    """UDP-based GStreamer capture for OpenCV without GStreamer support"""
    
    def __init__(self, width=1280, height=720, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.gst_process = None
        self.opencv_cap = None
        self.is_opened = False
        self.udp_port = 5000
        self.exposure_time = CAMERA_EXPOSURE_TIME
        self.analog_gain = CAMERA_ANALOG_GAIN
        
    def open(self):
        """Start UDP GStreamer server and OpenCV UDP client"""
        try:
            # Create GStreamer UDP server command using the working pipeline components
            gst_server_cmd = [
                'gst-launch-1.0',
                f'nvarguscamerasrc {CAMERA_PROPERTY_STRING}',
                '!', f'video/x-raw(memory:NVMM), width={self.width}, height={self.height}, format=NV12, framerate={self.fps}/1',
                '!', 'nvvidconv',
                '!', 'video/x-raw, format=BGR',
                '!', 'videoconvert',
                '!', 'x264enc tune=zerolatency bitrate=2000 speed-preset=superfast',
                '!', 'rtph264pay',
                '!', f'udpsink host=127.0.0.1 port={self.udp_port}'
            ]
            
            print(f"[INFO] Starting UDP GStreamer server: {' '.join(gst_server_cmd)}")
            
            # Start GStreamer UDP server process
            self.gst_process = subprocess.Popen(
                gst_server_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            
            # Give GStreamer time to start
            time.sleep(3)
            
            # Check if GStreamer process is still running
            if self.gst_process.poll() is not None:
                print("[ERROR] GStreamer UDP server failed to start")
                return False
            
            # Try to connect OpenCV to UDP stream (if GStreamer support exists)
            udp_pipeline = f"udpsrc port={self.udp_port} ! application/x-rtp,encoding-name=H264,payload=96 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
            
            print(f"[INFO] Connecting OpenCV to UDP stream...")
            self.opencv_cap = cv2.VideoCapture(udp_pipeline)
            
            if self.opencv_cap and self.opencv_cap.isOpened():
                # Test if we can read a frame
                time.sleep(2)  # Give more time for connection
                ret, test_frame = self.opencv_cap.read()
                if ret and test_frame is not None:
                    print(f"[INFO] UDP streaming working! Frame size: {test_frame.shape[1]}x{test_frame.shape[0]}")
                    self.is_opened = True
                    return True
                else:
                    print("[INFO] UDP stream connected but no frames received")
            else:
                print("[INFO] OpenCV cannot connect to UDP stream (expected without GStreamer support)")
            
            # If OpenCV connection fails, we still keep GStreamer running for potential manual testing
            print("[INFO] GStreamer UDP server is running on port 5000")
            print("[INFO] You can test with: gst-launch-1.0 udpsrc port=5000 ! application/x-rtp,encoding-name=H264,payload=96 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! xvimagesink")
            
            return False
                
        except Exception as e:
            print(f"[ERROR] Failed to start UDP GStreamer: {e}")
            return False
    
    def isOpened(self):
        return self.is_opened and (self.opencv_cap is not None) and self.opencv_cap.isOpened()
    
    def read(self):
        """Read frame from UDP stream"""
        if not self.isOpened():
            return False, None
            
        try:
            return self.opencv_cap.read()
        except Exception as e:
            print(f"[ERROR] Failed to read from UDP stream: {e}")
            return False, None
    
    def release(self):
        """Stop UDP GStreamer server and OpenCV client"""
        if self.opencv_cap:
            try:
                self.opencv_cap.release()
            except:
                pass
            self.opencv_cap = None
        
        if self.gst_process:
            try:
                self.gst_process.terminate()
                self.gst_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.gst_process.kill()
            except:
                pass
            self.gst_process = None
        
        self.is_opened = False


class GStreamerCSICapture:
    """Pure GStreamer CSI capture using PyGObject when OpenCV lacks GStreamer."""

    def __init__(self, width=1280, height=720, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.appsink = None
        self.is_opened = False
        self.bus = None

    def _build_pipeline(self):
        return (
            f"nvarguscamerasrc name=camerasrc {CAMERA_PROPERTY_STRING} ! "
            f"video/x-raw(memory:NVMM), width={self.width}, height={self.height}, format=NV12, framerate={self.fps}/1 ! "
            "nvvidconv ! "
            "video/x-raw, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink name=csiappsink emit-signals=false max-buffers=1 drop=true sync=false"
        )

    def open(self):
        if not GST_AVAILABLE:
            print("[WARN] PyGObject GStreamer bindings missing - cannot use CSI fallback")
            return False

        if self.is_opened:
            return True

        try:
            description = self._build_pipeline()
            self.pipeline = Gst.parse_launch(description)
            self.appsink = self.pipeline.get_by_name("csiappsink")
            if self.appsink is None:
                print("[ERROR] GStreamer pipeline missing appsink element")
                self.release()
                return False

            # Camera properties are already set via CAMERA_PROPERTY_STRING in pipeline string
            # No need for manual set_property() calls which can cause conflicts
            # The pipeline string format is the correct way to set nvarguscamerasrc properties
            source = self.pipeline.get_by_name("camerasrc")
            if source:
                print(f"[INFO] Camera properties set via pipeline string: exposure={CAMERA_EXPOSURE_TIME}, gain={CAMERA_ANALOG_GAIN}, "
                      f"ISP_digital_gain={ISP_DIGITAL_GAIN}, TNR={TNR_MODE}/{TNR_STRENGTH}, "
                      f"EE={EE_MODE}/{EE_STRENGTH}, exposure_compensation={EXPOSURE_COMPENSATION}, "
                      f"sensor_mode={SENSOR_MODE}")

            self.appsink.set_property("emit-signals", False)
            self.appsink.set_property("max-buffers", 1)
            self.appsink.set_property("drop", True)
            self.appsink.set_property("sync", False)

            self.bus = self.pipeline.get_bus()

            state_change = self.pipeline.set_state(Gst.State.PLAYING)
            if state_change == Gst.StateChangeReturn.FAILURE:
                print("[ERROR] Failed to start GStreamer pipeline")
                self.release()
                return False

            if self.bus:
                msg = self.bus.timed_pop_filtered(
                    2 * Gst.SECOND,
                    Gst.MessageType.ERROR | Gst.MessageType.EOS,
                )
                if msg:
                    err, debug = msg.parse_error()
                    print(f"[ERROR] GStreamer pipeline error: {err} ({debug})")
                    self.release()
                    return False

            self.is_opened = True
            print("[INFO] Using PyGObject GStreamer capture for CSI camera")
            return True
        except Exception as exc:
            print(f"[ERROR] Failed to initialize GStreamer capture: {exc}")
            self.release()
            return False

    def isOpened(self):
        return self.is_opened

    def read(self):
        if not self.is_opened or self.appsink is None:
            return False, None

        try:
            sample = self.appsink.emit("try-pull-sample", Gst.SECOND)
            if sample is None:
                return False, None

            buffer = sample.get_buffer()
            caps = sample.get_caps()
            structure = caps.get_structure(0)
            width = structure.get_value('width')
            height = structure.get_value('height')

            success, map_info = buffer.map(Gst.MapFlags.READ)
            if not success:
                return False, None

            frame = np.frombuffer(map_info.data, dtype=np.uint8)
            frame = frame.reshape((height, width, 3)).copy()
            buffer.unmap(map_info)
            return True, frame
        except Exception as exc:
            print(f"[ERROR] GStreamer read failed: {exc}")
            return False, None

    def release(self):
        if self.pipeline:
            try:
                self.pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
        self.pipeline = None
        self.appsink = None
        self.bus = None
        self.is_opened = False


def setup_v4l2_loopback(width=1280, height=720, fps=30):
    """Setup V4L2 loopback device for CSI camera access"""
    try:
        # Check if v4l2loopback module is loaded (Python 3.5+ compatible)
        result = subprocess.run(['lsmod'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode('utf-8') if result.stdout else ""
        if 'v4l2loopback' not in output:
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
            f'nvarguscamerasrc {CAMERA_PROPERTY_STRING}',
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

        # Detect input size from model automatically
        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) >= 3 and input_shape[2] is not None:
            self.input_size = int(input_shape[2])  # e.g., [1, 3, 640, 640] -> 640
        else:
            # Fallback: default to 640 for 6-class model
            self.input_size = 640
            print("[WARN] Could not detect input size from model, defaulting to 640")
        
        print(f"[INFO] Detected model input size: {self.input_size}x{self.input_size}")
        self.crop_height = 80
        self.crop_width_ratio = 0.6
        self.conf_threshold = 0.3  # Default threshold (increased from 0.25 - typical YOLOv8 threshold after sigmoid)
        self.roi_color = (0, 255, 255)

        # Load class names dynamically
        self.class_names = self._load_class_names()
        
        # Build color map using visualization standards
        self.colors = {}
        for class_name in self.class_names:
            self.colors[class_name] = get_class_color(class_name)
        
        # Defect classes (exclude "normal" from visualization)
        self.defect_classes = [cls for cls in self.class_names if cls != 'normal']
        
        # Per-class thresholds (default to global threshold)
        self.class_thresholds = {cls: self.conf_threshold for cls in self.class_names}
        
        # Statistics - initialize for all classes
        self.detection_counts = {cls: 0 for cls in self.class_names}
        self.fps_history = deque(maxlen=60)

        print(f"✅ Detector ready - Classes: {self.class_names}")
        print(f"   Defect classes (for visualization): {self.defect_classes}")
    
    def _load_class_names(self):
        """Load class names from config file or default to 6-class model classes"""
        # Try to load from config file first
        config_file = Path(__file__).parent / 'config.json'
        if config_file.exists():
            try:
                import json
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    if 'classes' in config:
                        print(f"[INFO] Loaded classes from config.json: {config['classes']}")
                        return config['classes']
            except Exception as e:
                print(f"[WARN] Failed to load classes from config.json: {e}")
        
        # Try to load from classes.txt in learning_based
        classes_file = Path(__file__).parent.parent / 'learning_based' / 'labels' / 'classes.txt'
        if classes_file.exists():
            try:
                with open(classes_file, 'r', encoding='utf-8') as f:
                    class_names = [line.strip() for line in f if line.strip()]
                if class_names:
                    print(f"[INFO] Loaded classes from {classes_file}: {class_names}")
                    return class_names
            except Exception as e:
                print(f"[WARN] Failed to load classes from {classes_file}: {e}")
        
        # Default to 6-class model classes
        default_classes = ['NOK', 'breaks', 'damage', 'drops', 'normal', 'shift']
        print(f"[INFO] Using default 6-class model classes: {default_classes}")
        return default_classes
    
    def crop_to_roi(self, frame):
        """Crop frame to the central ROI (vertical band + side trim) used during training."""
        h, w = frame.shape[:2]

        # Vertical crop (keep center band)
        crop_height = min(self.crop_height, h)
        start_y = max((h - crop_height) // 2, 0)
        end_y = start_y + crop_height
        vertical_cropped = frame[start_y:end_y, :]

        # Horizontal crop (keep center width portion)
        crop_width = int(vertical_cropped.shape[1] * self.crop_width_ratio)
        crop_width = max(1, min(crop_width, vertical_cropped.shape[1]))
        start_x = max((vertical_cropped.shape[1] - crop_width) // 2, 0)
        end_x = start_x + crop_width

        roi = vertical_cropped[:, start_x:end_x]

        roi_info = {
            "top": start_y,
            "left": start_x,
            "height": roi.shape[0],
            "width": roi.shape[1],
        }

        return roi, roi_info

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

    def clip_bbox_to_roi(self, bbox, roi):
        """Clip bbox (ROI coordinates) so it stays within ROI bounds."""
        x1, y1, x2, y2 = bbox
        x1 = max(0.0, min(x1, roi["width"]))
        x2 = max(0.0, min(x2, roi["width"]))
        y1 = max(0.0, min(y1, roi["height"]))
        y2 = max(0.0, min(y2, roi["height"]))

        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    def shift_bbox_to_original(self, bbox, original_shape, roi):
        x1, y1, x2, y2 = bbox

        width = original_shape[1]
        height = original_shape[0]

        roi_left = roi["left"]
        roi_top = roi["top"]
        roi_right = roi_left + roi["width"]
        roi_bottom = roi_top + roi["height"]

        x1 = int(round(x1 + roi_left))
        x2 = int(round(x2 + roi_left))
        y1 = int(round(y1 + roi_top))
        y2 = int(round(y2 + roi_top))

        x1 = max(roi_left, min(x1, roi_right))
        x2 = max(roi_left, min(x2, roi_right))
        y1 = max(roi_top, min(y1, roi_bottom))
        y2 = max(roi_top, min(y2, roi_bottom))

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

    def draw_roi(self, frame, roi):
        """Draw ROI rectangle on frame."""
        if not roi:
            return frame

        top = roi["top"]
        left = roi["left"]
        bottom = top + roi["height"]
        right = left + roi["width"]

        cv2.rectangle(frame, (left, top), (right, bottom), self.roi_color, 1, lineType=cv2.LINE_AA)
        cv2.putText(
            frame,
            "ROI",
            (left + 8, max(15, top - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.roi_color,
            1,
        )
        return frame

    def draw_detections(self, frame, detections, roi=None):
        """
        Draw ROI outline plus minimal bounding boxes (colored rectangles only, no text).
        Only displays defect classes (excludes 'normal').
        Optimized for FPS performance.
        """
        annotated = frame
        if roi:
            annotated = self.draw_roi(annotated, roi)

        # Filter detections: only show defect classes (exclude 'normal')
        defect_detections = [
            det for det in detections 
            if det['class_name'] in self.defect_classes
        ]

        # Draw minimal bboxes: colored rectangles only, no text labels
        for detection in defect_detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            
            # Get color from visualization standards
            color = self.colors.get(class_name, (128, 128, 128))
            
            # Draw minimal bounding box (rectangle only, no text)
            # Using thickness 2 for visibility
            cv2.rectangle(annotated, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         color, 2)
        
        return annotated
    
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
        cropped_frame, roi = self.crop_to_roi(frame)
        
        # Preprocess
        input_data, ratio, dwdh = self.preprocess(cropped_frame)
 
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_data})
        
        # Postprocess with smart logging (no verbose for real-time)
        detections = self.postprocess(
            outputs[0], ratio, dwdh, cropped_frame.shape,
            verbose=False, image_name=None
        )

        scaled_detections = []
        for det in detections:
            clipped = self.clip_bbox_to_roi(det['bbox'], roi)
            if clipped is None:
                continue

            scaled_bbox = self.shift_bbox_to_original(clipped, original_frame.shape, roi)

            if scaled_bbox is None:
                continue

            scaled_detections.append({
                'bbox': scaled_bbox,
                'class_id': det['class_id'],
                'class_name': det['class_name'],
                'confidence': det['confidence']
            })

        # Draw detections (only defect classes, minimal bbox)
        annotated_frame = self.draw_detections(original_frame, scaled_detections, roi=roi)

        processing_time = time.time() - start_time
        return annotated_frame, scaled_detections, processing_time

    def postprocess(self, output, ratio, dwdh, cropped_shape, verbose=False, image_name=None):
        """Extract detections from YOLOv8 ONNX model output.
        
        YOLOv8 ONNX output format: (batch, features, anchors) = (1, 13, 8400)
        Format: [x_center, y_center, w, h, objectness, class_0, ..., class_5, ...]
        Need to transpose to (8400, 13) to iterate over anchors.
        
        Args:
            output: Model output array
            ratio: Letterbox ratio
            dwdh: Letterbox padding
            cropped_shape: Cropped image shape
            verbose: If True, print detailed debug info (default: False)
            image_name: Optional image name for logging (default: None)
        
        Returns:
            List of detections after NMS
        """
        if output.ndim == 3:
            output = output[0]  # Remove batch dimension: (13, 8400)

        # Transpose: (13, 8400) -> (8400, 13)
        # Now each row is an anchor with 13 features
        output = output.transpose(1, 0)  # (8400, 13)

        # Statistics collection (not printed during loop)
        raw_detections = []
        skipped_count = {'len<13': 0, 'threshold': 0, 'bbox_invalid': 0}
        confidence_values = []
        objectness_values = []
        class_counts = {name: 0 for name in self.class_names}

        for anchor in output:
            if len(anchor) < 13:
                skipped_count['len<13'] += 1
                continue

            # Parse YOLOv8 format: [x_center, y_center, w, h, objectness, class_0, ..., class_5, ...]
            x_center = float(anchor[0])
            y_center = float(anchor[1])
            w = float(anchor[2])
            h = float(anchor[3])
            
            # Apply sigmoid activation: YOLOv8 ONNX outputs raw logits
            objectness_logit = float(anchor[4])
            class_scores_logits = anchor[5:11].astype(np.float32)  # 6 classes (indices 5-10)
            
            # Sigmoid: 1 / (1 + exp(-x))
            objectness = 1.0 / (1.0 + np.exp(-objectness_logit))
            
            # Objectness pre-filter: Filter low objectness anchors before class score calculation
            # This dramatically reduces false positives
            OBJECTNESS_THRESHOLD = 0.5
            if objectness < OBJECTNESS_THRESHOLD:
                skipped_count['threshold'] += 1
                continue
            
            class_scores = 1.0 / (1.0 + np.exp(-class_scores_logits))

            # Calculate confidence and class_id
            max_class_score = float(np.max(class_scores))
            conf = objectness * max_class_score
            class_id = int(np.argmax(class_scores))

            # Use per-class threshold if available, otherwise use global threshold
            class_name = self.class_names[class_id]
            threshold = self.class_thresholds.get(class_name, self.conf_threshold)

            # Collect statistics
            objectness_values.append(objectness)

            # Verbose debug: Print first 3 detections only
            if verbose and len(raw_detections) < 3:
                print(f"[DEBUG] Anchor: x_center={x_center:.2f}, y_center={y_center:.2f}, w={w:.2f}, h={h:.2f}, "
                      f"objectness={objectness:.4f}, max_class_score={max_class_score:.4f}, "
                      f"conf={conf:.4f}, class={class_name}, threshold={threshold:.4f}")

            if conf < threshold:
                skipped_count['threshold'] += 1
                continue

            # Convert center format to xyxy
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2

            # Scale bbox from letterbox to cropped image
            bbox_cropped = self.scale_bbox_from_letterbox(
                [x1, y1, x2, y2],
                ratio,
                dwdh,
                cropped_shape
            )

            if bbox_cropped is None:
                skipped_count['bbox_invalid'] += 1
                continue

            # Collect statistics for valid detections
            confidence_values.append(conf)
            class_counts[class_name] += 1

            raw_detections.append({
                'class_id': class_id,
                'class_name': class_name,
                'confidence': float(conf),
                'bbox': bbox_cropped
            })

        # Apply NMS
        final_detections = self.nms(raw_detections, iou_threshold=0.5)

        # Smart logging: One-line summary per image
        if image_name:
            conf_min = min(confidence_values) if confidence_values else 0.0
            conf_max = max(confidence_values) if confidence_values else 0.0
            conf_mean = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
            obj_min = min(objectness_values) if objectness_values else 0.0
            obj_max = max(objectness_values) if objectness_values else 0.0
            obj_mean = sum(objectness_values) / len(objectness_values) if objectness_values else 0.0
            
            print(f"[IMAGE] {image_name}: anchors={len(output)}, after_threshold={len(raw_detections)}, "
                  f"after_nms={len(final_detections)}, conf_range=[{conf_min:.2f}-{conf_max:.2f}], "
                  f"obj_range=[{obj_min:.2f}-{obj_max:.2f}]")

        return final_detections
    
    def set_class_threshold(self, class_name: str, threshold: float):
        """Set threshold for a specific class"""
        if class_name in self.class_names:
            self.class_thresholds[class_name] = threshold
    
    def set_class_thresholds(self, thresholds: dict):
        """Set thresholds for multiple classes"""
        for class_name, threshold in thresholds.items():
            if class_name in self.class_names:
                self.class_thresholds[class_name] = threshold
    
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
        
        # Build stats string dynamically for all classes
        stats_parts = [f"FPS: {avg_fps:4.1f}", f"Total: {total_detections:4d}"]
        for class_name in self.class_names:
            count = self.detection_counts.get(class_name, 0)
            stats_parts.append(f"{class_name}: {count:3d}")
        
        print(f"\r[Frame {frame_count:4d}] " + " | ".join(stats_parts), 
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
        """Generate simple working GStreamer pipeline based on user's confirmed working command"""
        # User confirmed working: gst-launch-1.0 nvarguscamerasrc ! nvvidconv ! xvimagesink
        # We just replace xvimagesink with appsink for OpenCV
        source_block = f"nvarguscamerasrc {CAMERA_PROPERTY_STRING}"

        # Primary pipeline - exactly like user's working command but with appsink
        working_pipeline = f"{source_block} ! nvvidconv ! appsink"
        
        # Alternative with basic format specification
        basic_pipeline = (
            f"{source_block} ! "
            f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, "
            f"format=NV12, framerate={framerate}/1 ! "
            f"nvvidconv ! appsink"
        )
        
        # Most detailed pipeline (if system OpenCV supports it)
        detailed_pipeline = (
            f"{source_block} ! "
            f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, "
            f"format=NV12, framerate={framerate}/1 ! "
            f"nvvidconv ! "
            f"video/x-raw, format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! appsink"
        )
        
        return [working_pipeline, basic_pipeline, detailed_pipeline]

    def _is_int(value: str) -> bool:
        try:
            int(value)
            return True
        except ValueError:
            return False

    # Check OpenCV version and GStreamer support
    cv_version = cv2.__version__
    print(f"[INFO] OpenCV version: {cv_version}")
    print(f"[INFO] Desired ROI: center 80px high × {int(width * 0.6)}px wide (overlay enabled)")
    
    # Verify camera settings are loaded correctly (matching simple_recorder.py)
    print(f"[INFO] Camera settings (from constants): exposure={CAMERA_EXPOSURE_TIME}, gain={CAMERA_ANALOG_GAIN}, "
          f"ISP_digital_gain={ISP_DIGITAL_GAIN}, TNR={TNR_MODE}/{TNR_STRENGTH}, "
          f"EE={EE_MODE}/{EE_STRENGTH}, exposure_compensation={EXPOSURE_COMPENSATION}, "
          f"sensor_mode={SENSOR_MODE}")
    print(f"[INFO] Camera property string: {CAMERA_PROPERTY_STRING}")
    
    # Check OpenCV source and GStreamer support
    def check_opencv_source():
        """Check if using system OpenCV vs pip OpenCV and GStreamer support"""
        print("[INFO] Checking OpenCV installation source...")
        
        # Get OpenCV file location
        opencv_path = cv2.__file__
        print(f"[INFO] OpenCV loaded from: {opencv_path}")
        
        # Check if it's system OpenCV or pip OpenCV
        if '/usr/lib/python3/dist-packages' in opencv_path:
            print("[INFO] Using SYSTEM OpenCV (Good - likely has GStreamer support)")
            is_system_opencv = True
        elif 'site-packages' in opencv_path:
            print("[WARN] Using PIP OpenCV (May lack GStreamer support)")
            is_system_opencv = False
        else:
            print("[INFO] OpenCV source unclear")
            is_system_opencv = False
        
        # Get detailed build information
        try:
            build_info = cv2.getBuildInformation()
            
            # Check for GStreamer in build info
            if 'GStreamer:' in build_info:
                gstreamer_line = [line for line in build_info.split('\n') if 'GStreamer:' in line]
                if gstreamer_line:
                    print(f"[INFO] {gstreamer_line[0].strip()}")
                    has_gstreamer = 'YES' in gstreamer_line[0]
                else:
                    has_gstreamer = False
            else:
                has_gstreamer = False
                
            # Also check for NVIDIA-specific components
            if 'CUDA:' in build_info:
                cuda_line = [line for line in build_info.split('\n') if 'CUDA:' in line]
                if cuda_line:
                    print(f"[INFO] {cuda_line[0].strip()}")
                    
        except Exception as e:
            print(f"[WARN] Could not get build information: {e}")
            has_gstreamer = False
        
        return is_system_opencv, has_gstreamer
    
    # Test GStreamer support practically
    def test_gstreamer_support():
        try:
            # Try to create a simple GStreamer pipeline to test support
            test_pipeline = "videotestsrc num-buffers=1 ! appsink"
            test_cap = cv2.VideoCapture(test_pipeline)
            has_gstreamer = test_cap.isOpened()
            test_cap.release()
            return has_gstreamer
        except:
            return False
    
    is_system_opencv, has_gstreamer_build = check_opencv_source()
    has_gstreamer_test = test_gstreamer_support()
    
    print(f"[INFO] GStreamer in build info: {'Yes' if has_gstreamer_build else 'No'}")
    print(f"[INFO] GStreamer test result: {'Yes' if has_gstreamer_test else 'No'}")
    
    has_gstreamer = has_gstreamer_build and has_gstreamer_test
    
    if not has_gstreamer:
        if not is_system_opencv:
            print("[CRITICAL] Using pip OpenCV without GStreamer support!")
            print("[SOLUTION] Need to use system OpenCV with GStreamer support")
            print()
            print("=" * 60)
            print("🔧 OPENCV FIX REQUIRED")
            print("=" * 60)
            print("To fix OpenCV and enable CSI camera support:")
            print()
            print("1. Remove pip-installed OpenCV:")
            print("   pip uninstall opencv-python opencv-contrib-python opencv-python-headless")
            print()
            print("2. Ensure system OpenCV is available:")
            print("   sudo apt update")
            print("   sudo apt install python3-opencv")
            print()
            print("3. Verify system OpenCV path is in venv:")
            print("   Check if venv/lib/python3.x/site-packages/opencv-system.pth exists")
            print("   Content should be: /usr/lib/python3/dist-packages")
            print()
            print("4. Restart the script after fixing")
            print("=" * 60)
            print()
            
            # Try to provide automatic fix
            try:
                import site
                venv_site_packages = site.getsitepackages()[0] if site.getsitepackages() else None
                if venv_site_packages:
                    pth_file = os.path.join(venv_site_packages, "opencv-system.pth")
                    system_opencv_path = "/usr/lib/python3/dist-packages"
                    
                    if not os.path.exists(pth_file):
                        print(f"[INFO] Creating opencv-system.pth file: {pth_file}")
                        try:
                            with open(pth_file, 'w', encoding='utf-8') as f:
                                f.write(system_opencv_path + '\n')
                            print("[INFO] Created opencv-system.pth - please restart the script")
                        except Exception as e:
                            print(f"[WARN] Could not create pth file: {e}")
                    else:
                        print(f"[INFO] opencv-system.pth already exists: {pth_file}")
                        
            except Exception as e:
                print(f"[WARN] Could not auto-fix OpenCV path: {e}")
                
        else:
            print("[WARN] System OpenCV found but GStreamer test failed!")
            print("[SOLUTION] May need to reinstall GStreamer packages")
            print()
            print("Try: sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good")

    # For Jetson with CSI camera, prioritize GStreamer pipeline
    if _is_int(source):
        device_index = int(source)

        if not has_gstreamer:
            if GST_AVAILABLE:
                print("[INFO] OpenCV build lacks GStreamer - trying PyGObject CSI capture")
                gst_capture = GStreamerCSICapture(width, height, fps)
                if gst_capture.open():
                    return gst_capture
                else:
                    print("[INFO] PyGObject CSI capture failed, trying additional fallbacks")
            else:
                print("[WARN] python3-gi not installed; cannot use PyGObject CSI fallback")
        
        # First try CSI camera with GStreamer pipeline (preferred for Jetson when available)
        if has_gstreamer:
            print(f"[INFO] Attempting CSI camera with GStreamer pipeline...")
            pipelines = _get_csi_pipeline(width, height, fps)
            
            for i, pipeline in enumerate(pipelines):
                pipeline_name = ["Working (Simple)", "Basic (With Format)", "Detailed (Full)"][i]
                print(f"[INFO] Trying {pipeline_name} pipeline: {pipeline}")
                
                try:
                    cap = cv2.VideoCapture(pipeline)
                    if cap.isOpened():
                        print(f"[INFO] {pipeline_name} pipeline opened successfully")
                        # Test if we can actually read a frame
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            print(f"[INFO] ✅ SUCCESS! CSI camera working via {pipeline_name} pipeline")
                            print(f"[INFO] Frame size: {test_frame.shape[1]}x{test_frame.shape[0]}")
                            print(f"[INFO] This matches your working gst-launch command!")
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
            
            # Fallback to UDP GStreamer streaming
            print(f"[INFO] Attempting UDP GStreamer streaming for CSI camera...")
            try:
                udp_cap = UDPGStreamerCapture(width, height, fps)
                if udp_cap.open():
                    print(f"[INFO] Successfully opened CSI camera via UDP GStreamer streaming")
                    return udp_cap
                else:
                    print(f"[INFO] UDP GStreamer streaming setup completed but OpenCV connection failed")
                    print(f"[INFO] This is expected with OpenCV 3.2.0 without GStreamer support")
                    # Keep the process running for manual testing
                    return udp_cap  # Return it anyway so user can test manually
            except Exception as e:
                print(f"[INFO] UDP GStreamer streaming failed: {e}")

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
    print("   - Verify PyGObject bindings: python3 -c \"import gi; gi.require_version('Gst','1.0')\"")
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
    print("   - Consider upgrading OpenCV or rely on the PyGObject fallback added in this repo")
    print()
    print("5. ALTERNATIVE SOLUTIONS:")
    print("   - Install v4l2loopback: sudo apt install v4l2loopback-dkms")
    print("   - Try USB camera as temporary solution")
    print("   - Use external video capture software")
    print()
    print("6. MANUAL TESTING (if UDP server started):")
    print("   - Test UDP stream: gst-launch-1.0 udpsrc port=5000 ! application/x-rtp,encoding-name=H264,payload=96 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! xvimagesink")
    print("   - Check if port 5000 is active: netstat -an | grep 5000")
    print("   - Kill any remaining processes: pkill -f gst-launch")
    print("="*60)

    raise RuntimeError(f"Unable to open video source: {source}")

    return cap

def main():
    args = parse_args()

    print("=" * 60)
    print("📹 Wire Defect Detection - Jetson Nano")
    print("=" * 60)
    print()

    # Use the model specified in args (default: best_cropped.onnx)
    model_path = Path(args.model)
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return 1
    
    print(f"[INFO] Using model: {model_path}")

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
