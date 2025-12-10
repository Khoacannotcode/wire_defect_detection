#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wire Defect Detection - Desktop GUI Application
Tkinter-based GUI for real-time defect detection on Jetson Nano
Layout: TD (Top-Down) - Video display on top, Control panel below
Live video cropped to show only ROI region
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import json
import time
from datetime import datetime
from pathlib import Path
import sys
import os
import logging

# Add parent directory to path for imports
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

# Import from run_camera_detection
from run_camera_detection import LiveWireDetector, open_capture, MODELS_DIR

# Import defect logger
from defect_logger import DefectLogger

# Add system packages to path for compatibility
sys.path.insert(0, '/usr/lib/python3/dist-packages')

# Configure logging based on DEBUG environment variable
DEBUG_MODE = os.getenv('DEBUG', '0') in ('1', 'true', 'True', 'TRUE')
if DEBUG_MODE:
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
else:
    logging.basicConfig(
        level=logging.WARNING,  # Only WARNING and ERROR in production
        format='[%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

logger = logging.getLogger(__name__)


class DetectionGUI:
    """Desktop GUI application for wire defect detection"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Wire Defect Detection - Real-time Monitor")
        # Increased height by 40%: 768 * 1.4 = 1075.2 ≈ 1080
        self.root.geometry("1024x1080")
        
        # State variables
        self.detector = None
        self.capture = None
        self.capture_thread = None
        self.is_capturing = False
        self.current_frame = None
        self.current_detections = []
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.last_log_update = time.time()
        self.log_update_interval = 1.0  # Update log display every 1 second
        self.log_widgets = {}  # Store log widgets for reuse (avoid destroy/create)
        self.legend_items = {}  # Store legend items for highlighting
        self.roi_aspect_ratio = None  # Store ROI aspect ratio
        self.pending_gui_updates = 0  # Track pending GUI updates to prevent queue buildup
        self.session_duration_update_timer = None  # Timer for session duration updates
        self.last_alert_update = time.time()
        self.alert_update_interval = 0.2  # Update alerts every 200ms (5 FPS for alerts)
        self.last_legend_update = time.time()
        self.legend_update_interval = 0.3
        
        # Session alarm animation state
        self.alarm_animation_active = False
        self.alarm_animation_state = 0
        self.alarm_animation_timer = None  # Update legend highlighting every 300ms
        
        # Config
        self.config_file = ROOT_DIR / 'config.json'
        self.config = self.load_config()
        
        # Threshold sliders (will be initialized after detector)
        self.threshold_sliders = {}
        self.threshold_labels = {}
        self.save_thresholds_timer = None  # Timer ID for debounced save
        
        # Frame counter for logging
        self.frame_number = 0
        
        # Initialize defect logger
        self.defect_logger = DefectLogger()
        
        # Initialize detector
        self.init_detector()
        
        # Setup GUI
        self.setup_gui()
        
        # Start camera capture
        self.start_capture()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def load_config(self):
        """Load configuration from config.json"""
        # Load model paths from model_config.json
        model_config_path = ROOT_DIR / 'model_config.json'
        onnx_path = None
        engine_path = None
        if model_config_path.exists():
            try:
                with open(model_config_path, 'r') as f:
                    model_config = json.load(f)
                    onnx_path = str(ROOT_DIR / model_config['onnx_model_path'])
                    engine_path = str(ROOT_DIR / model_config['tensorrt_engine_path'])
            except Exception as e:
                logger.warning(f"Failed to load model_config.json: {e}")
        
        # Prefer engine path if available, otherwise use onnx path
        default_model_path = engine_path or onnx_path or str(MODELS_DIR / 'best_v3_416x256.engine')
        
        default_config = {
            'model_path': default_model_path,
            'camera_source': '0',
            'camera_width': 1280,
            'camera_height': 720,
            'camera_fps': 30,
            'use_gstreamer': True,  # Default to True for Jetson (proven to work)
            'display_width': 800,
            'display_height': 600,
            'thresholds': {}  # Per-class thresholds (will be populated after detector init)
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # Remove model_path from user_config - it should always come from model_config.json
                    user_config.pop('model_path', None)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning("Failed to load config.json: {}, using defaults".format(e))
        else:
            # Create default config file
            self.save_config(default_config)
        
        # Force model_path from model_config.json (never allow config.json to override it)
        default_config['model_path'] = default_model_path
        
        return default_config
    
    def save_config(self, config=None):
        """Save configuration to config.json (model_path is excluded - it comes from model_config.json)"""
        if config is None:
            config = self.config
        
        # Create a copy without model_path (model_path should only be in model_config.json)
        config_to_save = {k: v for k, v in config.items() if k != 'model_path'}
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("Failed to save config.json: {}".format(e))
            # Keep critical error print for production mode visibility
            if not DEBUG_MODE:
                print("[ERROR] Failed to save config.json: {}".format(e))
    
    def init_detector(self):
        """Initialize LiveWireDetector"""
        # Load from model_config.json if available
        model_config_path = ROOT_DIR / 'model_config.json'
        default_model_path = str(MODELS_DIR / 'best_v3_416x256.engine')
        if model_config_path.exists():
            try:
                with open(model_config_path, 'r') as f:
                    model_config = json.load(f)
                    # Prefer engine path, fallback to onnx path
                    if 'tensorrt_engine_path' in model_config:
                        default_model_path = str(ROOT_DIR / model_config['tensorrt_engine_path'])
                    elif 'onnx_model_path' in model_config:
                        default_model_path = str(ROOT_DIR / model_config['onnx_model_path'])
            except Exception as e:
                logger.warning(f"Failed to load model_config.json: {e}")
        
        model_path = Path(self.config.get('model_path', default_model_path))
        
        if not model_path.exists():
            messagebox.showerror("Error", "Model not found: {}".format(model_path))
            return
        
        try:
            self.detector = LiveWireDetector(model_path)
            logger.info("Detector initialized successfully")
            
            # Load per-class thresholds from config
            self.load_thresholds()
        except Exception as e:
            messagebox.showerror("Error", "Failed to initialize detector: {}".format(e))
            logger.error("Detector initialization failed: {}".format(e))
            # Keep critical error print for production mode visibility
            if not DEBUG_MODE:
                print("[ERROR] Detector initialization failed: {}".format(e))
    
    def setup_gui(self):
        """Setup GUI layout - TD (Top-Down) layout"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # ============================================
        # TOP SECTION: Video Display
        # ============================================
        video_frame = ttk.LabelFrame(main_frame, text="Live Video (ROI Cropped)", padding="5")
        video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))  # Don't fill vertically - size to content
        main_frame.columnconfigure(0, weight=1)  # Only column expands, not row
        
        # Video display label - will size to ROI image only (no large 16:9 frame, no white space)
        self.video_label = ttk.Label(video_frame, text="Initializing camera...", 
                                     background="black", foreground="white")
        self.video_label.pack()  # Don't fill/expand - will size to ROI image only
        
        # ============================================
        # BOTTOM SECTION: Control Panel
        # ============================================
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="5")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Status display
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.fps_label = ttk.Label(status_frame, text="FPS: 0.0")
        self.fps_label.pack(side=tk.LEFT, padx=5)
        
        # Visual alert indicator (red/green)
        self.defect_alert_label = ttk.Label(status_frame, text="●", font=("Arial", 16), 
                                           foreground="green")
        self.defect_alert_label.pack(side=tk.LEFT, padx=5)
        
        # Defect status display
        self.defect_status_label = ttk.Label(status_frame, text="Status: OK", 
                                            foreground="green", font=("Arial", 9, "bold"))
        self.defect_status_label.pack(side=tk.LEFT, padx=5)
        
        self.detection_label = ttk.Label(status_frame, text="Detections: 0")
        self.detection_label.pack(side=tk.LEFT, padx=5)
        
        # Defect classes display (active classes in current frame)
        self.active_classes_label = ttk.Label(status_frame, text="", 
                                             foreground="gray", font=("Arial", 8))
        self.active_classes_label.pack(side=tk.LEFT, padx=5)
        
        # Class color legend with active defect highlighting (merged with statistics)
        legend_frame = ttk.LabelFrame(control_frame, text="Defect Classes", padding="5")
        legend_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Create frame for color swatches (visual, human-friendly)
        self.legend_container = ttk.Frame(legend_frame)
        self.legend_container.pack(fill=tk.X)
        
        # Threshold controls section
        threshold_frame = ttk.LabelFrame(control_frame, text="Per-Class Threshold Controls", padding="5")
        threshold_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Create threshold sliders container
        self.threshold_container = ttk.Frame(threshold_frame)
        self.threshold_container.pack(fill=tk.X)
        
        # Session controls (placed directly above Session Timer Monitor)
        session_frame = ttk.Frame(control_frame)
        session_frame.pack(fill=tk.X, pady=(5, 5))
        
        ttk.Label(session_frame, text="Session:").pack(side=tk.LEFT, padx=(0, 5))
        self.start_btn = ttk.Button(session_frame, text="Start Session", 
                                    command=self.start_session, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=2)
        self.stop_btn = ttk.Button(session_frame, text="Stop Session", 
                                   command=self.stop_session, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        
        self.session_status_label = ttk.Label(session_frame, text="Status: Stopped", 
                                              foreground="gray")
        self.session_status_label.pack(side=tk.LEFT, padx=10)
        
        # Store session start time for duration calculation
        self.session_start_time = None
        
        # Session Timer Monitor section
        log_frame = ttk.LabelFrame(control_frame, text="Session Timer Monitor", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))  # Allow expansion
        
        # ============================================
        # Session Timer Frame (inside Session Timer Monitor)
        # ============================================
        timer_section_frame = ttk.Frame(log_frame)
        timer_section_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Timer header frame with "View Rule" button
        timer_header_frame = ttk.Frame(timer_section_frame)
        timer_header_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(timer_header_frame, text="Session Timer:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 5))
        
        # "View Rule" label with hover effect
        self.view_rule_label = ttk.Label(timer_header_frame, text="ℹ View Rule", 
                                         foreground="blue", cursor="hand2",
                                         font=("Arial", 8, "underline"))
        self.view_rule_label.pack(side=tk.RIGHT, padx=5)
        
        # Bind hover events for highlight effect
        self.view_rule_label.bind("<Enter>", self._on_view_rule_enter)
        self.view_rule_label.bind("<Leave>", self._on_view_rule_leave)
        self.view_rule_label.bind("<Button-1>", self._show_session_timer_rule)
        
        # Session duration display
        timer_content_frame = ttk.Frame(timer_section_frame)
        timer_content_frame.pack(fill=tk.X)
        
        self.session_duration_label = ttk.Label(timer_content_frame, text="Duration: 0s", 
                                                foreground="gray", font=("Arial", 9))
        self.session_duration_label.pack(side=tk.LEFT, padx=5)
        
        # ============================================
        # Session Alarm Frame (inside Session Timer Monitor)
        # ============================================
        alarm_section_frame = ttk.Frame(log_frame)
        alarm_section_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Alarm header frame with "View Rule" button
        alarm_header_frame = ttk.Frame(alarm_section_frame)
        alarm_header_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(alarm_header_frame, text="Session Alarm:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 5))
        
        # "View Rule" label with hover effect
        self.alarm_view_rule_label = ttk.Label(alarm_header_frame, text="ℹ View Rule", 
                                               foreground="blue", cursor="hand2",
                                               font=("Arial", 8, "underline"))
        self.alarm_view_rule_label.pack(side=tk.RIGHT, padx=5)
        
        # Bind hover events for highlight effect
        self.alarm_view_rule_label.bind("<Enter>", self._on_alarm_view_rule_enter)
        self.alarm_view_rule_label.bind("<Leave>", self._on_alarm_view_rule_leave)
        self.alarm_view_rule_label.bind("<Button-1>", self._show_alarm_rule)
        
        # Alarm content frame
        alarm_content_frame = ttk.Frame(alarm_section_frame)
        alarm_content_frame.pack(fill=tk.X)
        
        # Alarm icon with animation (using "!" instead of emoji for Python 3.6 compatibility)
        self.alarm_icon_label = ttk.Label(alarm_content_frame, text="!", font=("Arial", 20, "bold"))
        self.alarm_icon_label.pack(side=tk.LEFT, padx=2)
        
        # Alarm description (only shown when alarm is active)
        self.alarm_text_label = ttk.Label(alarm_content_frame, text="", 
                                          foreground="red", font=("Arial", 9, "bold"))
        self.alarm_text_label.pack(side=tk.LEFT, padx=5)
        
        # ============================================
        # Log Display (inside Session Timer Monitor)
        # ============================================
        # Create container for log display (no scrollbar - full content visible)
        self.log_container = ttk.Frame(log_frame)
        self.log_container.pack(fill=tk.BOTH, expand=True)  # Allow expansion
        
        # Log labels (will be created/updated dynamically)
        self.log_labels = {}
        
        # Log file path display with open button
        log_path_frame = ttk.Frame(control_frame)
        log_path_frame.pack(pady=(5, 0))
        
        self.log_path_label = ttk.Label(log_path_frame, text="Log file: Not saved", 
                                       foreground="gray", font=("Arial", 8))
        self.log_path_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.open_log_btn = ttk.Button(log_path_frame, text="Open Log", 
                                      command=self.open_log_file, state=tk.DISABLED)
        self.open_log_btn.pack(side=tk.LEFT)
        
        # Store current log path
        self.current_log_path = None
        
        # Update legend and thresholds if detector is available
        if self.detector:
            self.update_legend()
            self.setup_threshold_sliders()
        
        # Initialize log display
        self.update_log_display()
    
    def update_legend(self):
        """Update class color legend with visual color swatches (human-friendly)"""
        if not self.detector:
            return
        
        # Clear existing legend widgets
        for widget in self.legend_container.winfo_children():
            widget.destroy()
        self.legend_items.clear()
        
        # Create color swatches for each defect class
        for class_name in self.detector.defect_classes:
            color = self.detector.colors.get(class_name, (128, 128, 128))
            # Convert BGR to RGB for display
            color_rgb = (color[2], color[1], color[0])
            hex_color = "#{:02x}{:02x}{:02x}".format(color_rgb[0], color_rgb[1], color_rgb[2])
            
            # Create frame for each color item
            item_frame = ttk.Frame(self.legend_container)
            item_frame.pack(side=tk.LEFT, padx=5, pady=2)
            
            # Create color swatch (visual color box)
            color_canvas = tk.Canvas(item_frame, width=30, height=20, highlightthickness=1, 
                                    highlightbackground="gray", borderwidth=0)
            color_canvas.pack(side=tk.LEFT, padx=(0, 5))
            color_canvas.create_rectangle(2, 2, 28, 18, fill=hex_color, outline="gray", width=1)
            
            # Create label with class name
            class_label = ttk.Label(item_frame, text=class_name, font=("Arial", 9))
            class_label.pack(side=tk.LEFT)
            
            # Store for highlighting
            self.legend_items[class_name] = {
                'frame': item_frame,
                'label': class_label,
                'canvas': color_canvas
            }
    
    def setup_threshold_sliders(self):
        """Setup threshold sliders for each defect class"""
        if not self.detector:
            return
        
        # Clear existing sliders
        for widget in self.threshold_container.winfo_children():
            widget.destroy()
        self.threshold_sliders.clear()
        self.threshold_labels.clear()
        
        # Create slider for each defect class
        for class_name in self.detector.defect_classes:
            # Get color for this class
            color = self.detector.colors.get(class_name, (128, 128, 128))
            color_rgb = (color[2], color[1], color[0])
            hex_color = "#{:02x}{:02x}{:02x}".format(color_rgb[0], color_rgb[1], color_rgb[2])
            
            # Create frame for each slider
            slider_frame = ttk.Frame(self.threshold_container)
            slider_frame.pack(fill=tk.X, pady=2)
            
            # Color indicator (small colored box)
            color_indicator = tk.Canvas(slider_frame, width=20, height=20, highlightthickness=1,
                                       highlightbackground="gray", borderwidth=0)
            color_indicator.pack(side=tk.LEFT, padx=(0, 5))
            color_indicator.create_rectangle(2, 2, 18, 18, fill=hex_color, outline="gray", width=1)
            
            # Class name label
            class_label = ttk.Label(slider_frame, text="{}:".format(class_name), width=10, anchor=tk.W)
            class_label.pack(side=tk.LEFT, padx=(0, 5))
            
            # Slider (range 0.0 to 1.0, resolution 0.01)
            slider = ttk.Scale(slider_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                              length=200, command=lambda val, name=class_name: self.on_threshold_change(name, val))
            slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            
            # Bind click event to jump slider to clicked position
            def on_slider_click(event, scale=slider, name=class_name):
                """Handle click on slider track - jump to clicked position"""
                # Calculate value based on click position
                # Get slider width and position
                scale.update_idletasks()
                scale_width = scale.winfo_width()
                if scale_width <= 1:
                    return  # Slider not yet sized
                
                # Get click position relative to slider
                click_x = event.x
                
                # Calculate value: 0.0 at left edge, 1.0 at right edge
                value = click_x / scale_width
                value = max(0.0, min(1.0, value))  # Clamp to range
                
                # Set slider value (this will trigger command callback)
                scale.set(value)
            
            slider.bind("<Button-1>", on_slider_click)
            
            # Value label (shows current threshold)
            value_label = ttk.Label(slider_frame, text="0.25", width=6, anchor=tk.E)
            value_label.pack(side=tk.LEFT, padx=(5, 0))
            
            # Store references
            self.threshold_sliders[class_name] = slider
            self.threshold_labels[class_name] = value_label
            
            # Set initial value from config or default
            threshold = self.config.get('thresholds', {}).get(class_name, 0.25)
            threshold = max(0.0, min(1.0, float(threshold)))  # Validate range
            slider.set(threshold)
            value_label.config(text="{:.2f}".format(threshold))
    
    def load_thresholds(self):
        """Load per-class thresholds from config and apply to detector"""
        if not self.detector:
            return
        
        thresholds = self.config.get('thresholds', {})
        thresholds_dict = {}
        
        # Load thresholds for each defect class
        for class_name in self.detector.defect_classes:
            threshold = thresholds.get(class_name, 0.25)
            threshold = max(0.0, min(1.0, float(threshold)))  # Validate range
            thresholds_dict[class_name] = threshold
        
        # Apply to detector
        if thresholds_dict:
            self.detector.set_class_thresholds(thresholds_dict)
            logger.info("Loaded thresholds from config: {}".format(thresholds_dict))
    
    def on_threshold_change(self, class_name, value):
        """Callback when threshold slider changes"""
        try:
            threshold = float(value)
            threshold = max(0.0, min(1.0, threshold))  # Ensure in range
            
            # Update detector in real-time
            if self.detector:
                self.detector.set_class_threshold(class_name, threshold)
            
            # Update value label
            if class_name in self.threshold_labels:
                self.threshold_labels[class_name].config(text="{:.2f}".format(threshold))
            
            # Cancel previous save timer if exists
            if self.save_thresholds_timer is not None:
                self.root.after_cancel(self.save_thresholds_timer)
            
            # Save to config (debounced - only save after user stops dragging for 500ms)
            self.save_thresholds_timer = self.root.after(500, self._debounced_save_thresholds)
            
        except Exception as e:
            logger.error("Failed to update threshold for {}: {}".format(class_name, e))
            # Keep critical error print for production mode visibility
            if not DEBUG_MODE:
                print("[ERROR] Failed to update threshold for {}: {}".format(class_name, e))
    
    def _debounced_save_thresholds(self):
        """Debounced save thresholds (called after user stops dragging)"""
        self.save_thresholds_timer = None  # Clear timer ID
        self.save_thresholds()
    
    def save_thresholds(self):
        """Save current thresholds to config.json"""
        if not self.detector:
            return
        
        # Collect current threshold values from sliders
        thresholds = {}
        for class_name in self.detector.defect_classes:
            if class_name in self.threshold_sliders:
                threshold = self.threshold_sliders[class_name].get()
                thresholds[class_name] = float(threshold)
        
        # Update config
        self.config['thresholds'] = thresholds
        
        # Save to file (silent - no console spam)
        self.save_config()
        # Only print if debug mode (removed to reduce console spam)
    
    def start_capture(self):
        """Start camera capture in separate thread"""
        if self.is_capturing:
            return
        
        source = self.config.get('camera_source', '0')
        width = self.config.get('camera_width', 1280)
        height = self.config.get('camera_height', 720)
        fps = self.config.get('camera_fps', 30)
        use_gstreamer = self.config.get('use_gstreamer', True)  # Default True for Jetson (proven to work)
        
        logger.debug("Camera config: source={}, width={}, height={}, fps={}, use_gstreamer={}".format(source, width, height, fps, use_gstreamer))
        
        # CRITICAL: Release any existing camera capture first
        # This prevents "Failed to create CaptureSession" errors
        if self.capture is not None:
            logger.info("Releasing existing camera capture...")
            try:
                self.is_capturing = False  # Stop capture loop first
                if self.capture_thread and self.capture_thread.is_alive():
                    # Wait for capture thread to finish (with timeout)
                    self.capture_thread.join(timeout=1.0)
                self.capture.release()
                import time
                time.sleep(0.5)  # Wait for release to complete
                logger.info("Camera released successfully")
            except Exception as e:
                logger.warning("Error releasing camera: {}".format(e))
            finally:
                self.capture = None
        
        # CRITICAL: Kill any processes that might be holding the camera
        # This helps when previous runs didn't clean up properly
        import subprocess
        try:
            # Check for processes using camera
            result = subprocess.Popen(['lsof', '/dev/video0'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            stdout, stderr = result.communicate(timeout=2)
            if result.returncode == 0 and stdout:
                logger.warning("Camera /dev/video0 is in use. Attempting to free it...")
                # Try to kill common camera processes (but not our own Python process)
                try:
                    subprocess.Popen(['pkill', '-f', 'nvarguscamerasrc'], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate(timeout=1)
                    subprocess.Popen(['pkill', '-f', 'gst-launch'], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate(timeout=1)
                    import time
                    time.sleep(1.0)  # Wait for processes to terminate
                    logger.info("Attempted to free camera resources")
                except:
                    pass
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            # lsof/pkill not available or timeout - continue anyway
            pass
        
        try:
            self.capture = open_capture(source, width, height, fps, use_gstreamer)
            if not self.capture or not self.capture.isOpened():
                error_msg = (
                    "Failed to open camera source: {}\n\n"
                    "Possible solutions:\n"
                    "1. Check camera is connected\n"
                    "2. Try different camera source (0, 1, 2...)\n"
                    "3. Check camera permissions\n"
                    "4. Restart the application"
                ).format(source)
                messagebox.showerror("Camera Error", error_msg)
                return
            
            # Verify camera can actually read frames (with multiple attempts)
            import time
            ret = False
            test_frame = None
            start_time = time.time()
            for attempt in range(10):  # Try up to 10 times
                ret, test_frame = self.capture.read()
                if ret and test_frame is not None:
                    break
                time.sleep(0.1)
                if time.time() - start_time > 3.0:  # 3 second timeout
                    break
            
            if not ret or test_frame is None:
                error_msg = (
                    "Camera opened but cannot read frames\n\n"
                    "Possible solutions:\n"
                    "1. Camera may be in use by another application\n"
                    "2. Try different camera source\n"
                    "3. Check camera connection\n"
                    "4. Try disabling GStreamer in config.json (set use_gstreamer: false)"
                )
                messagebox.showerror("Camera Error", error_msg)
                try:
                    self.capture.release()
                except:
                    pass
                self.capture = None
                return
            
            self.is_capturing = True
            self.current_frame_raw = None  # Initialize frame buffer
            
            # Start camera capture thread (only reads frames, no detection)
            self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
            self.capture_thread.start()
            
            # Start frame processing in MAIN THREAD (detection runs here)
            self.process_frame()
            
            # Enable session buttons
            self.start_btn.config(state=tk.NORMAL)
            
            logger.info("Camera capture started successfully (detection runs in main thread)")
        except Exception as e:
            error_msg = "Failed to start camera: {}\n\nCheck camera connection and try again.".format(e)
            messagebox.showerror("Camera Error", error_msg)
            logger.error("Camera start failed: {}".format(e))
            # Keep critical error print for production mode visibility
            if not DEBUG_MODE:
                print("[ERROR] Camera start failed: {}".format(e))
    
    def capture_loop(self):
        """
        Camera capture loop running in separate thread.
        CRITICAL: Only reads frames here. Detection runs in main thread to avoid CUDA context issues.
        """
        import time
        while self.is_capturing:
            if self.capture is None:
                break
            
            try:
                ret, frame = self.capture.read()
                if not ret or frame is None:
                    logger.warning("Failed to read frame")
                    time.sleep(0.1)
                    continue
                
                # Store raw frame for main thread to process
                # Main thread will run detection (single-threaded TensorRT inference)
                self.current_frame_raw = frame
                
                # Small delay to prevent overwhelming main thread
                time.sleep(0.01)  # ~100 FPS max capture rate
            except Exception as e:
                logger.error("Error in capture loop: {}".format(e))
                # Keep critical error print for production mode visibility
                if not DEBUG_MODE:
                    print("[ERROR] Error in capture loop: {}".format(e))
                time.sleep(0.1)
                continue
        
        # Cleanup when loop exits
        logger.info("Capture loop exiting, releasing camera...")
        if self.capture is not None:
            try:
                self.capture.release()
            except:
                pass
            self.capture = None
    
    def process_frame(self):
        """
        Process frame and run detection in MAIN THREAD.
        This ensures TensorRT inference runs in the same thread as CUDA context initialization.
        Called periodically via Tkinter's after() callback.
        """
        if not self.is_capturing or not hasattr(self, 'current_frame_raw') or self.current_frame_raw is None:
            # Schedule next check
            self.root.after(33, self.process_frame)  # ~30 FPS
            return
        
        frame = self.current_frame_raw
        self.current_frame_raw = None  # Clear to prevent reprocessing
        
        # CRITICAL: Always crop to ROI for display (as it was done before)
        # This ensures only the ROI region is shown in GUI, not the full 16:9 frame
        # ROI: 768x80 (very wide and short rectangle matching training data)
        if self.detector:
            roi_frame, roi_info = self.detector.crop_to_roi_for_display(frame)
        else:
            # If detector not available, still crop using default ratio and strip height
            h, w = frame.shape[:2]
            crop_ratio = 0.6  # Default ROI ratio (60% center width)
            strip_height = 80  # Default strip height (80px center strip)
            crop_width = int(w * crop_ratio)
            start_x = (w - crop_width) // 2
            y_top = (h - strip_height) // 2
            y_bottom = y_top + strip_height
            roi_frame = frame[y_top:y_bottom, start_x:start_x + crop_width]
            roi_info = {
                'start_x': start_x,
                'end_x': start_x + crop_width,
                'y_top': y_top,
                'y_bottom': y_bottom,
                'width': crop_width,
                'height': strip_height
            }
        
        # Run detection in MAIN THREAD (single-threaded TensorRT inference)
        if self.detector:
            try:
                # Run detection on full frame (detector handles ROI cropping internally)
                annotated_frame, detections, processing_time = self.detector.detect_frame(frame)
                
                # CRITICAL: Crop annotated frame to ROI for display
                # The annotated_frame contains detections drawn on full frame,
                # but GUI should only show ROI region (as it was before)
                annotated_roi, _ = self.detector.crop_to_roi_for_display(annotated_frame)
                
                # Update detector statistics (detection counts, FPS history)
                self.detector.update_stats(detections, processing_time)
                
                # Update FPS
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_update >= 1.0:
                    self.fps = self.frame_count / (current_time - self.last_fps_update)
                    self.frame_count = 0
                    self.last_fps_update = current_time
                
                # Log defects if session is active
                if self.defect_logger.session_active:
                    self.defect_logger.log_defects(self.frame_number, detections)
                
                # Increment frame number
                self.frame_number += 1
                
                # Store cropped ROI frame for GUI update (always cropped, never full frame)
                self.current_frame = annotated_roi
                self.current_detections = detections
            except Exception as e:
                logger.error("Detection failed: {}".format(e))
                if DEBUG_MODE:
                    import traceback
                    traceback.print_exc()
                # Keep critical error print for production mode visibility
                if not DEBUG_MODE:
                    print("[ERROR] Detection failed: {}".format(e))
                # On error, still use cropped ROI frame (not full frame)
                self.current_frame = roi_frame
                self.current_detections = []
        else:
            # No detector: use cropped ROI frame (not full frame)
            self.current_frame = roi_frame
            self.current_detections = []
            # Still increment frame number
            self.frame_number += 1
        
        # Update GUI (already in main thread)
        self.update_display()
        
        # Schedule next frame processing
        self.root.after(33, self.process_frame)  # ~30 FPS
    
    def _safe_update_display(self):
        """Wrapper for update_display with exception handling and pending counter"""
        try:
            self.update_display()
        except Exception as e:
            logger.error("Display update exception: {}".format(e))
            # Keep critical error print for production mode visibility
            if not DEBUG_MODE:
                print("[ERROR] Display update exception: {}".format(e))
        finally:
            self.pending_gui_updates = max(0, self.pending_gui_updates - 1)
    
    def update_display(self):
        """Update video display (called from main thread)"""
        if self.current_frame is None:
            return
        
        try:
            # Get video label size (will be set to fill available space)
            # Resize ROI frame to fill entire video display widget (no black borders)
            # Get the actual size of the video label widget
            self.video_label.update_idletasks()
            label_width = self.video_label.winfo_width()
            label_height = self.video_label.winfo_height()
            
            # If label not yet sized, use config defaults
            if label_width <= 1 or label_height <= 1:
                label_width = self.config.get('display_width', 800)
                label_height = self.config.get('display_height', 600)
            
            # Get ROI frame dimensions
            frame_height, frame_width = self.current_frame.shape[:2]
            frame_aspect = frame_width / frame_height if frame_height > 0 else 1.0
            
            # Store ROI aspect ratio
            self.roi_aspect_ratio = frame_aspect
            
            # User requirement: Only show ROI area compactly, no large 16:9 frame
            # Strategy: Resize ROI to reasonable display size maintaining aspect ratio
            # Widget will size to image (no large frame, just compact ROI display)
            
            # Get available width from video_frame (parent of video_label)
            video_frame_widget = self.video_label.master
            video_frame_widget.update_idletasks()
            available_width = video_frame_widget.winfo_width() - 10  # Account for padding
            
            # If video_frame not yet sized, use config defaults
            if available_width <= 1:
                available_width = self.config.get('display_width', 800)
            
            # Calculate resize to fill available width, maintain ROI aspect ratio
            display_width = available_width
            display_height = int(available_width / frame_aspect)
            
            # Resize ROI maintaining aspect ratio
            resized = cv2.resize(self.current_frame, (display_width, display_height), 
                                interpolation=cv2.INTER_LINEAR)
            
            # Use resized ROI directly - widget will size to image (compact, no large frame)
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(image=pil_image)
            
            # Update label - label will size to image (no black bars)
            self.video_label.config(image=photo, text="")
            self.video_label.image = photo  # Keep a reference
            
            # Update status labels (always update - lightweight)
            self.fps_label.config(text="FPS: {:.1f}".format(self.fps))
            self.detection_label.config(text="Detections: {}".format(len(self.current_detections)))
            
            # Update visual alerts periodically (not every frame for performance)
            current_time = time.time()
            if current_time - self.last_alert_update >= self.alert_update_interval:
                self.update_defect_alerts()
                self.update_session_alarm()  # Update session alarm
                self.last_alert_update = current_time
            
            # Update log display periodically (not every frame for performance)
            if current_time - self.last_log_update >= self.log_update_interval:
                self.update_log_display()
                self.last_log_update = current_time
            
            # Update legend highlighting periodically (not every frame for performance)
            if current_time - self.last_legend_update >= self.legend_update_interval:
                if self.detector and self.legend_items:
                    # Get active defect classes from current detections
                    active_defects = set()
                    for det in self.current_detections:
                        if det['class_name'] in self.detector.defect_classes:
                            active_defects.add(det['class_name'])
                    
                    # Highlight active defects in legend
                    for class_name, item_data in self.legend_items.items():
                        label = item_data['label']
                        canvas = item_data['canvas']
                        
                        if class_name in active_defects:
                            # Highlight: bold text, thicker border
                            label.config(font=("Arial", 9, "bold"), foreground="red")
                            canvas.config(highlightthickness=2, highlightbackground="red")
                        else:
                            # Normal: regular text, normal border
                            label.config(font=("Arial", 9), foreground="black")
                            canvas.config(highlightthickness=1, highlightbackground="gray")
                self.last_legend_update = current_time
        
        except Exception as e:
            logger.error("Display update failed: {}".format(e))
            # Keep critical error print for production mode visibility
            if not DEBUG_MODE:
                print("[ERROR] Display update failed: {}".format(e))
    
    def update_defect_alerts(self):
        """Update visual alerts and defect status display"""
        try:
            # Filter defect detections (exclude 'normal')
            defect_detections = [
                det for det in self.current_detections 
                if self.detector and det.get('class_name') in self.detector.defect_classes
            ]
            
            # Get active defect classes
            active_classes = set()
            for det in defect_detections:
                active_classes.add(det['class_name'])
            
            # Update visual alert (red when defects, green when OK)
            if defect_detections:
                self.defect_alert_label.config(foreground="red")
                self.defect_status_label.config(
                    text="Status: DEFECT ({})".format(len(defect_detections)),
                    foreground="red"
                )
            else:
                self.defect_alert_label.config(foreground="green")
                self.defect_status_label.config(
                    text="Status: OK",
                    foreground="green"
                )
            
            # Update active classes display
            if active_classes:
                classes_str = ", ".join(sorted(active_classes))
                self.active_classes_label.config(
                    text="Active: {}".format(classes_str),
                    foreground="red"
                )
            else:
                self.active_classes_label.config(
                    text="",
                    foreground="gray"
                )
        except Exception as e:
            logger.error("Defect alerts update failed: {}".format(e))
            # Keep critical error print for production mode visibility
            if not DEBUG_MODE:
                print("[ERROR] Defect alerts update failed: {}".format(e))
    
    def _animate_alarm_icon(self):
        """Simple but eye-catching animation for alarm icon"""
        if not self.alarm_animation_active:
            return
        
        # Toggle between normal and bold/larger
        if self.alarm_animation_state == 0:
            self.alarm_icon_label.config(font=("Arial", 24, "bold"), foreground="red")
            self.alarm_animation_state = 1
        else:
            self.alarm_icon_label.config(font=("Arial", 20, "bold"), foreground="darkred")
            self.alarm_animation_state = 0
        
        # Schedule next animation (every 500ms for visible effect)
        self.alarm_animation_timer = self.root.after(500, self._animate_alarm_icon)
    
    def update_session_alarm(self):
        """Update session alarm display based on active cluster conditions"""
        try:
            alarm_info = self.defect_logger.get_active_alarm()
            
            if alarm_info:
                # Show alarm
                self.alarm_text_label.config(
                    text=alarm_info['message'],
                    foreground="red"
                )
                
                # Start animation if not already active
                if not self.alarm_animation_active:
                    self.alarm_animation_active = True
                    self.alarm_animation_state = 0
                    self._animate_alarm_icon()
                
                # TODO: Future - trigger sound alarm here
                # self._trigger_sound_alarm()
            else:
                # Clear alarm
                self.alarm_text_label.config(text="")
                
                # Stop animation
                if self.alarm_animation_active:
                    self.alarm_animation_active = False
                    if self.alarm_animation_timer is not None:
                        self.root.after_cancel(self.alarm_animation_timer)
                        self.alarm_animation_timer = None
                    self.alarm_icon_label.config(font=("Arial", 20, "bold"), foreground="gray")  # Reset to normal (inactive)
                    self.alarm_animation_state = 0
                    
        except Exception as e:
            logger.error("Session alarm update failed: {}".format(e))
            # Keep critical error print for production mode visibility
            if not DEBUG_MODE:
                print("[ERROR] Session alarm update failed: {}".format(e))
    
    def start_session(self):
        """Start detection session"""
        # Reset frame number for new session
        self.frame_number = 0
        self.defect_logger.start_session()
        self.session_start_time = time.time()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.session_status_label.config(text="Status: Active", foreground="green")
        self.log_path_label.config(text="Log file: Session active...", foreground="green")
        self.open_log_btn.config(state=tk.DISABLED)  # Disable open button during session
        logger.info("Session started")
        self.update_log_display()
        self.update_session_duration()  # Start duration updates
    
    def stop_session(self):
        """Stop detection session"""
        # Stop duration updates
        if self.session_duration_update_timer:
            self.root.after_cancel(self.session_duration_update_timer)
            self.session_duration_update_timer = None
        
        self.defect_logger.stop_session()  # Set session_active = False
        
        # Calculate final stripe duration (first_stripe → last_stripe)
        stripe_timing = self.defect_logger.get_stripe_timing_info()
        
        if stripe_timing and stripe_timing['duration'] > 0:
            duration = stripe_timing['duration']
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            
            if minutes > 0:
                duration_text = "{}m {}s ({:.1f}min)".format(minutes, seconds, duration/60)
            else:
                duration_text = "{:.1f}s".format(duration)
            
            # Hiển thị thông tin chi tiết với timestamps
            start_time_str = datetime.fromtimestamp(stripe_timing['start_time']).strftime("%H:%M:%S")
            end_time_str = datetime.fromtimestamp(stripe_timing['end_time']).strftime("%H:%M:%S")
            
            self.session_duration_label.config(
                text="Duration: {} ({} → {})".format(
                    duration_text,
                    start_time_str,
                    end_time_str
                ),
                foreground="gray"
            )
        else:
            # Không có stripe nào
            self.session_duration_label.config(
                text="Duration: 0s (no stripes)",
                foreground="gray"
            )
        
        self.session_start_time = None
        
        # Save log file
        log_path = self.defect_logger.save_session_log()
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.session_status_label.config(text="Status: Stopped", foreground="gray")
        
        # Display log file path and enable open button
        if log_path:
            self.current_log_path = log_path
            self.log_path_label.config(
                text="Log file: {}".format(log_path.name),
                foreground="blue"
            )
            self.open_log_btn.config(state=tk.NORMAL)  # Enable open button
            logger.info("Session log saved: {}".format(log_path))
        else:
            self.current_log_path = None
            self.log_path_label.config(
                text="Log file: Failed to save",
                foreground="red"
            )
            self.open_log_btn.config(state=tk.DISABLED)
        
        logger.info("Session stopped")
        self.update_log_display()
    
    def update_session_duration(self):
        """Update session duration display - shows stripe duration"""
        if not self.defect_logger.session_active:
            self.session_duration_label.config(text="Duration: 0s", foreground="gray")
            return
        
        # Lấy stripe duration từ logger
        stripe_duration = self.defect_logger.get_stripe_duration()
        
        if stripe_duration is None:
            # Chưa có stripe nào
            self.session_duration_label.config(
                text="Duration: Waiting for stripe...",
                foreground="gray"
            )
        else:
            # Hiển thị duration từ first stripe → current time (vì session đang active)
            minutes = int(stripe_duration // 60)
            seconds = int(stripe_duration % 60)
            
            if minutes > 0:
                text = "Duration: {}m {}s".format(minutes, seconds)
            else:
                text = "Duration: {}s".format(seconds)
            
            self.session_duration_label.config(
                text=text,
                foreground="green"  # Active - đang chạy
            )
        
        # Schedule next update
        self.session_duration_update_timer = self.root.after(1000, self.update_session_duration)
    
    def _on_view_rule_enter(self, event):
        """Highlight effect when hovering over View Rule"""
        self.view_rule_label.config(foreground="darkblue", font=("Arial", 8, "bold", "underline"))
    
    def _on_view_rule_leave(self, event):
        """Reset highlight when leaving View Rule"""
        self.view_rule_label.config(foreground="blue", font=("Arial", 8, "underline"))
    
    def _on_alarm_view_rule_enter(self, event):
        """Highlight effect when hovering over Alarm View Rule"""
        self.alarm_view_rule_label.config(foreground="darkblue", font=("Arial", 8, "bold", "underline"))
    
    def _on_alarm_view_rule_leave(self, event):
        """Reset highlight when leaving Alarm View Rule"""
        self.alarm_view_rule_label.config(foreground="blue", font=("Arial", 8, "underline"))
    
    def _show_session_timer_rule(self, event):
        """Show popup window with session timer rule explanation"""
        rule_text = """Session Timer Rule

The session timer measures the duration from the first stripe appearance to the last stripe appearance.

• Stripe Definition: Any detection (normal or defect class) counts as a "stripe"

• Timer Behavior:
  - Start Session: Timer waits for the first stripe to appear
  - First Stripe: Timer begins counting from this moment
  - Active Period: Timer continues counting while session is active
  - Stop Session: Timer shows final duration (first stripe → last stripe)

• Display Format:
  - Active: "Duration: Xm Ys" (updates every second)
  - Waiting: "Duration: Waiting for stripe..."
  - Final: "Duration: Xm Ys (HH:MM:SS → HH:MM:SS)"

• Purpose:
  Start/Stop Session buttons serve as ROI markers for when new wire is loaded into camera view. The timer measures actual wire processing time, not user interaction time."""
        
        # Create popup window
        popup = tk.Toplevel(self.root)
        popup.title("Session Timer Rule")
        popup.geometry("500x400")
        popup.resizable(False, False)
        
        # Make popup modal (focus on it)
        popup.transient(self.root)
        
        # CRITICAL: Update window first to make it viewable before grab_set()
        popup.update_idletasks()
        
        # Center popup on screen
        x = (popup.winfo_screenwidth() // 2) - (popup.winfo_width() // 2)
        y = (popup.winfo_screenheight() // 2) - (popup.winfo_height() // 2)
        popup.geometry(f"+{x}+{y}")
        
        # Update again after centering, then set grab
        popup.update_idletasks()
        popup.grab_set()
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(popup, padding="10")
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Arial", 10),
                             padx=10, pady=10, relief=tk.FLAT, bg="#f5f5f5")
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scrollbar.set)
        
        # Insert rule text
        text_widget.insert("1.0", rule_text)
        text_widget.config(state=tk.DISABLED)  # Make read-only
        
        # Close button
        button_frame = ttk.Frame(popup, padding="10")
        button_frame.pack(fill=tk.X)
        
        close_btn = ttk.Button(button_frame, text="Close", command=popup.destroy)
        close_btn.pack()
        
        # Focus on close button
        close_btn.focus_set()
        popup.bind("<Return>", lambda e: popup.destroy())
        popup.bind("<Escape>", lambda e: popup.destroy())
    
    def _show_alarm_rule(self, event):
        """Show popup window with session alarm rule explanation"""
        rule_text = """Session Alarm Rules

The session alarm triggers visual alerts when critical defect conditions are detected during an active session.

• Activation: Alarm only works when session timer is active (session started)

• Alarm Conditions:

  Damage:
  - Triggers when 1 or more damage defects appear in a cluster
  - Cluster duration must exceed 0.2 seconds
  - Example: 1 damage defect detected for >0.2s → ALARM

  Shift/Drop/Break:
  - Triggers when total 5 or more defects (any combination) from these classes appear in a cluster
  - Cluster duration must exceed 0.5 seconds
  - Example: 3 shift + 2 breaks = 5 total defects for >0.5s → ALARM

• Alarm Behavior:
  - Alarm appears immediately when conditions are met
  - Alarm clears immediately when conditions are no longer met
  - Icon animates (size/color changes) when alarm is active
  - Alarm message shows which condition triggered

• Exclusions:
  - NOK class defects are not counted in alarm conditions
  - Normal class detections do not trigger alarms
  - Only defect classes (damage, shift, drops, breaks) are considered

• Purpose:
  Provides real-time critical defect notifications to help operators respond quickly to quality issues."""
        
        # Create popup window
        popup = tk.Toplevel(self.root)
        popup.title("Session Alarm Rules")
        popup.geometry("600x450")
        popup.resizable(False, False)
        
        # Make popup modal (focus on it)
        popup.transient(self.root)
        
        # CRITICAL: Update window first to make it viewable before grab_set()
        popup.update_idletasks()
        
        # Center popup on screen
        x = (popup.winfo_screenwidth() // 2) - (popup.winfo_width() // 2)
        y = (popup.winfo_screenheight() // 2) - (popup.winfo_height() // 2)
        popup.geometry(f"+{x}+{y}")
        
        # Update again after centering, then set grab
        popup.update_idletasks()
        popup.grab_set()
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(popup, padding="10")
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Arial", 10),
                             padx=10, pady=10, relief=tk.FLAT, bg="#f5f5f5")
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scrollbar.set)
        
        # Insert rule text
        text_widget.insert("1.0", rule_text)
        text_widget.config(state=tk.DISABLED)  # Make read-only
        
        # Close button
        button_frame = ttk.Frame(popup, padding="10")
        button_frame.pack(fill=tk.X)
        
        close_btn = ttk.Button(button_frame, text="Close", command=popup.destroy)
        close_btn.pack()
        
        # Focus on close button
        close_btn.focus_set()
        popup.bind("<Return>", lambda e: popup.destroy())
        popup.bind("<Escape>", lambda e: popup.destroy())
    
    def open_log_file(self):
        """Open log file in default system application"""
        if not self.current_log_path or not self.current_log_path.exists():
            messagebox.showerror("Error", "Log file not found or not saved yet")
            return
        
        try:
            import platform
            import subprocess
            
            system = platform.system()
            if system == "Windows":
                os.startfile(str(self.current_log_path))
            elif system == "Darwin":  # macOS
                subprocess.run(["open", str(self.current_log_path)])
            else:  # Linux
                subprocess.run(["xdg-open", str(self.current_log_path)])
            
            logger.info("Opened log file: {}".format(self.current_log_path))
        except Exception as e:
            messagebox.showerror("Error", "Failed to open log file: {}".format(e))
            logger.error("Failed to open log file: {}".format(e))
            # Keep critical error print for production mode visibility
            if not DEBUG_MODE:
                print("[ERROR] Failed to open log file: {}".format(e))
    
    def on_closing(self):
        """Handle window close event"""
        logger.info("Closing application, cleaning up resources...")
        
        # Stop capture loop first
        self.is_capturing = False
        
        # Wait for capture thread to finish (with timeout)
        if self.capture_thread and self.capture_thread.is_alive():
            logger.info("Waiting for capture thread to finish...")
            self.capture_thread.join(timeout=2.0)
        
        # Release camera with proper cleanup
        if self.capture is not None:
            logger.info("Releasing camera...")
            try:
                self.capture.release()
                import time
                time.sleep(0.3)  # Give camera time to release
                logger.info("Camera released")
            except Exception as e:
                logger.warning("Error releasing camera: {}".format(e))
            finally:
                self.capture = None
        
        # Stop session if active
        if self.defect_logger.session_active:
            self.defect_logger.stop_session()
            self.defect_logger.save_session_log()
        
        # Save thresholds and config
        self.save_thresholds()
        self.save_config()
        
        logger.info("Cleanup complete, closing window...")
        self.root.destroy()
    
    def update_log_display(self):
        """Update real-time log display with current statistics (full content, no scroll)"""
        if not hasattr(self, 'log_container'):
            return
        
        try:
            # Check if container still exists (might be destroyed)
            try:
                self.log_container.winfo_exists()
            except:
                return  # Container destroyed, skip update
            
            # Get session statistics
            stats = self.defect_logger.get_session_stats()
            
            # Reuse widgets instead of destroy/create to avoid memory churn
            widget_keys = [
                'header', 'session_id', 'frames', 'summary_header', 'total_defects', 'total_clusters',
                'cluster_header', 'cluster_duration', 'cluster_frames', 'cluster_defects', 'cluster_classes',
                'timing_header', 'timing_duration', 'timing_frames',
                'class_header'
            ]
            
            # Clear widgets that are no longer needed
            existing_widgets = set(self.log_widgets.keys())
            needed_widgets = set(widget_keys)
            for key in existing_widgets - needed_widgets:
                if key in self.log_widgets:
                    try:
                        self.log_widgets[key].destroy()
                    except:
                        pass
                    del self.log_widgets[key]
            
            row = 0
            
            # Session status header (reuse widget if exists)
            header_text = "=== SESSION ACTIVE ===" if stats['session_active'] else "=== SESSION STOPPED ==="
            header_color = "blue" if stats['session_active'] else "gray"
            
            if 'header' not in self.log_widgets:
                self.log_widgets['header'] = ttk.Label(self.log_container, 
                                                      font=("Courier", 9, "bold"))
                self.log_widgets['header'].grid(row=row, column=0, sticky=tk.W, pady=(0, 2))
            self.log_widgets['header'].config(text=header_text, foreground=header_color)
            row += 1
            
            # Session ID (reuse widget)
            if stats['session_id']:
                if 'session_id' not in self.log_widgets:
                    self.log_widgets['session_id'] = ttk.Label(self.log_container, 
                                                              font=("Courier", 9))
                    self.log_widgets['session_id'].grid(row=row, column=0, sticky=tk.W, padx=(10, 0))
                self.log_widgets['session_id'].config(text="Session ID: {}".format(stats['session_id']))
                row += 1
            elif 'session_id' in self.log_widgets:
                self.log_widgets['session_id'].grid_remove()
            
            # Frames processed (reuse widget)
            if 'frames' not in self.log_widgets:
                self.log_widgets['frames'] = ttk.Label(self.log_container, font=("Courier", 9))
                self.log_widgets['frames'].grid(row=row, column=0, sticky=tk.W, padx=(10, 0), pady=(0, 5))
            self.log_widgets['frames'].config(text="Frames Processed: {}".format(stats['frames_processed']))
            self.log_widgets['frames'].grid()
            row += 1
            
            # Defect summary section (reuse widgets)
            if 'summary_header' not in self.log_widgets:
                self.log_widgets['summary_header'] = ttk.Label(self.log_container, 
                                                               font=("Courier", 9, "bold"), 
                                                               foreground="darkgreen")
                self.log_widgets['summary_header'].grid(row=row, column=0, sticky=tk.W, pady=(5, 2))
            self.log_widgets['summary_header'].config(text="--- Defect Summary ---")
            row += 1
            
            if 'total_defects' not in self.log_widgets:
                self.log_widgets['total_defects'] = ttk.Label(self.log_container, font=("Courier", 9))
                self.log_widgets['total_defects'].grid(row=row, column=0, sticky=tk.W, padx=(10, 0))
            self.log_widgets['total_defects'].config(text="Total Defects: {}".format(stats['total_defects']))
            row += 1
            
            if 'total_clusters' not in self.log_widgets:
                self.log_widgets['total_clusters'] = ttk.Label(self.log_container, font=("Courier", 9))
                self.log_widgets['total_clusters'].grid(row=row, column=0, sticky=tk.W, padx=(10, 0), pady=(0, 5))
            self.log_widgets['total_clusters'].config(text="Total Clusters: {}".format(stats['total_clusters']))
            row += 1
            
            # Active cluster section (reuse widgets)
            if 'cluster_header' not in self.log_widgets:
                self.log_widgets['cluster_header'] = ttk.Label(self.log_container, 
                                                              font=("Courier", 9, "bold"), 
                                                              foreground="darkgreen")
                self.log_widgets['cluster_header'].grid(row=row, column=0, sticky=tk.W, pady=(5, 2))
            self.log_widgets['cluster_header'].config(text="--- Active Cluster ---")
            row += 1
            
            if stats['active_cluster']:
                cluster = stats['active_cluster']
                # Hide "no cluster" message if exists
                if 'no_cluster' in self.log_widgets:
                    self.log_widgets['no_cluster'].grid_remove()
                
                # Create/reuse cluster widgets
                cluster_widgets = ['cluster_duration', 'cluster_frames', 'cluster_defects', 'cluster_classes']
                for i, key in enumerate(cluster_widgets):
                    if key not in self.log_widgets:
                        self.log_widgets[key] = ttk.Label(self.log_container, font=("Courier", 9))
                    self.log_widgets[key].grid(row=row, column=0, sticky=tk.W, padx=(10, 0))
                    row += 1
                
                # Update widget text
                self.log_widgets['cluster_duration'].config(text="Duration: {:.2f}s".format(cluster['duration']))
                self.log_widgets['cluster_frames'].config(text="Frames: {}".format(cluster['frame_count']))
                self.log_widgets['cluster_defects'].config(text="Defects: {}".format(cluster['defect_count']))
                classes_str = ', '.join(cluster['classes'].keys()) if cluster['classes'] else 'None'
                self.log_widgets['cluster_classes'].config(text="Classes: {}".format(classes_str))
            else:
                # Hide cluster detail widgets
                for key in ['cluster_duration', 'cluster_frames', 'cluster_defects', 'cluster_classes']:
                    if key in self.log_widgets:
                        self.log_widgets[key].grid_remove()
                
                # Show "no cluster" message
                if 'no_cluster' not in self.log_widgets:
                    self.log_widgets['no_cluster'] = ttk.Label(self.log_container, 
                                                              text="No active cluster",
                                                              font=("Courier", 9))
                    self.log_widgets['no_cluster'].grid(row=row, column=0, sticky=tk.W, padx=(10, 0), pady=(0, 5))
                else:
                    self.log_widgets['no_cluster'].grid(row=row, column=0, sticky=tk.W, padx=(10, 0), pady=(0, 5))
                row += 1
            
            # Stripe timing section (reuse widgets)
            if 'timing_header' not in self.log_widgets:
                self.log_widgets['timing_header'] = ttk.Label(self.log_container, 
                                                             font=("Courier", 9, "bold"), 
                                                             foreground="darkgreen")
                self.log_widgets['timing_header'].grid(row=row, column=0, sticky=tk.W, pady=(5, 2))
            self.log_widgets['timing_header'].config(text="--- Stripe Timing ---")
            row += 1
            
            if stats['stripe_timing']:
                timing = stats['stripe_timing']
                if 'timing_duration' not in self.log_widgets:
                    self.log_widgets['timing_duration'] = ttk.Label(self.log_container, font=("Courier", 9))
                    self.log_widgets['timing_duration'].grid(row=row, column=0, sticky=tk.W, padx=(10, 0))
                self.log_widgets['timing_duration'].config(text="Duration: {:.3f}s".format(timing['duration']))
                self.log_widgets['timing_duration'].grid()
                row += 1
                
                if 'timing_frames' not in self.log_widgets:
                    self.log_widgets['timing_frames'] = ttk.Label(self.log_container, font=("Courier", 9))
                    self.log_widgets['timing_frames'].grid(row=row, column=0, sticky=tk.W, padx=(10, 0), pady=(0, 5))
                self.log_widgets['timing_frames'].config(text="Frames: {} → {}".format(timing['start_frame'], timing['end_frame']))
                self.log_widgets['timing_frames'].grid()
                row += 1
            else:
                for key in ['timing_duration', 'timing_frames']:
                    if key in self.log_widgets:
                        self.log_widgets[key].grid_remove()
            
            # Per-class counts section (simplified - only show if needed)
            # Note: Per-class counts can be many, so we'll keep it simple
            if 'class_header' not in self.log_widgets:
                self.log_widgets['class_header'] = ttk.Label(self.log_container, 
                                                            font=("Courier", 9, "bold"), 
                                                            foreground="darkgreen")
                self.log_widgets['class_header'].grid(row=row, column=0, sticky=tk.W, pady=(5, 2))
            
            if stats['class_counts']:
                class_summary = ", ".join(["{}:{}".format(k, v) for k, v in sorted(stats['class_counts'].items(), 
                                                                        key=lambda x: x[1], reverse=True)[:5]])
                self.log_widgets['class_header'].config(text="--- Per-Class Counts: {} ---".format(class_summary))
                self.log_widgets['class_header'].grid()
            else:
                self.log_widgets['class_header'].config(text="--- Per-Class Counts: No defects ---")
                self.log_widgets['class_header'].grid()
            
        except Exception as e:
            logger.error("Failed to update log display: {}".format(e))
            if DEBUG_MODE:
                import traceback
                traceback.print_exc()
            # Keep critical error print for production mode visibility
            if not DEBUG_MODE:
                print("[ERROR] Failed to update log display: {}".format(e))


def main():
    """Main entry point"""
    root = tk.Tk()
    app = DetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

