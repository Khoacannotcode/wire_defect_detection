#!/usr/bin/env python3
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
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

# Import from run_camera_detection
from run_camera_detection import LiveWireDetector, open_capture, MODELS_DIR

# Import defect logger
from defect_logger import DefectLogger

# Add system packages to path for compatibility
sys.path.insert(0, '/usr/lib/python3/dist-packages')


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
        self.log_update_interval = 1.0  # Update log display every 1 second (reduced frequency)
        self.log_widgets = {}  # Store log widgets for reuse (avoid destroy/create)
        self.legend_items = {}  # Store legend items for highlighting
        self.roi_aspect_ratio = None  # Store ROI aspect ratio
        self.pending_gui_updates = 0  # Track pending GUI updates to prevent queue buildup
        
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
        default_config = {
            'model_path': str(MODELS_DIR / 'best_cropped.onnx'),
            'camera_source': '0',
            'camera_width': 1280,
            'camera_height': 720,
            'camera_fps': 30,
            'use_gstreamer': False,
            'display_width': 800,
            'display_height': 600,
            'thresholds': {}  # Per-class thresholds (will be populated after detector init)
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"[WARN] Failed to load config.json: {e}, using defaults")
        else:
            # Create default config file
            self.save_config(default_config)
        
        return default_config
    
    def save_config(self, config=None):
        """Save configuration to config.json"""
        if config is None:
            config = self.config
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[ERROR] Failed to save config.json: {e}")
    
    def init_detector(self):
        """Initialize LiveWireDetector"""
        model_path = Path(self.config.get('model_path', str(MODELS_DIR / 'best_cropped.onnx')))
        
        # Prefer opset16 model if exists
        model_path_opset16 = MODELS_DIR / "best_cropped_opset16.onnx"
        if model_path_opset16.exists():
            model_path = model_path_opset16
        
        if not model_path.exists():
            messagebox.showerror("Error", f"Model not found: {model_path}")
            return
        
        try:
            self.detector = LiveWireDetector(model_path)
            print("[INFO] Detector initialized successfully")
            
            # Load per-class thresholds from config
            self.load_thresholds()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize detector: {e}")
            print(f"[ERROR] Detector initialization failed: {e}")
    
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
        
        # Session controls
        session_frame = ttk.Frame(control_frame)
        session_frame.pack(fill=tk.X, pady=(0, 5))
        
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
        
        # Status display
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.fps_label = ttk.Label(status_frame, text="FPS: 0.0")
        self.fps_label.pack(side=tk.LEFT, padx=5)
        
        self.detection_label = ttk.Label(status_frame, text="Detections: 0")
        self.detection_label.pack(side=tk.LEFT, padx=5)
        
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
        
        # Real-time Log/Statistics section
        log_frame = ttk.LabelFrame(control_frame, text="Real-time Log & Statistics", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))  # Allow expansion
        
        # Create container for log display (no scrollbar - full content visible)
        self.log_container = ttk.Frame(log_frame)
        self.log_container.pack(fill=tk.BOTH, expand=True)  # Allow expansion
        
        # Log labels (will be created/updated dynamically)
        self.log_labels = {}
        
        # Log file path display
        self.log_path_label = ttk.Label(control_frame, text="Log file: Not saved", 
                                       foreground="gray", font=("Arial", 8))
        self.log_path_label.pack(pady=(5, 0))
        
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
            hex_color = f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}"
            
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
            hex_color = f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}"
            
            # Create frame for each slider
            slider_frame = ttk.Frame(self.threshold_container)
            slider_frame.pack(fill=tk.X, pady=2)
            
            # Color indicator (small colored box)
            color_indicator = tk.Canvas(slider_frame, width=20, height=20, highlightthickness=1,
                                       highlightbackground="gray", borderwidth=0)
            color_indicator.pack(side=tk.LEFT, padx=(0, 5))
            color_indicator.create_rectangle(2, 2, 18, 18, fill=hex_color, outline="gray", width=1)
            
            # Class name label
            class_label = ttk.Label(slider_frame, text=f"{class_name}:", width=10, anchor=tk.W)
            class_label.pack(side=tk.LEFT, padx=(0, 5))
            
            # Slider (range 0.0 to 1.0, resolution 0.01)
            slider = ttk.Scale(slider_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                              length=200, command=lambda val, name=class_name: self.on_threshold_change(name, val))
            slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            
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
            value_label.config(text=f"{threshold:.2f}")
    
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
            print(f"[INFO] Loaded thresholds from config: {thresholds_dict}")
    
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
                self.threshold_labels[class_name].config(text=f"{threshold:.2f}")
            
            # Cancel previous save timer if exists
            if self.save_thresholds_timer is not None:
                self.root.after_cancel(self.save_thresholds_timer)
            
            # Save to config (debounced - only save after user stops dragging for 500ms)
            self.save_thresholds_timer = self.root.after(500, self._debounced_save_thresholds)
            
        except Exception as e:
            print(f"[ERROR] Failed to update threshold for {class_name}: {e}")
    
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
        use_gstreamer = self.config.get('use_gstreamer', False)
        
        try:
            self.capture = open_capture(source, width, height, fps, use_gstreamer)
            if not self.capture or not self.capture.isOpened():
                messagebox.showerror("Error", "Failed to open camera")
                return
            
            self.is_capturing = True
            self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
            self.capture_thread.start()
            
            # Enable session buttons
            self.start_btn.config(state=tk.NORMAL)
            
            print("[INFO] Camera capture started")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {e}")
            print(f"[ERROR] Camera start failed: {e}")
    
    def capture_loop(self):
        """Camera capture loop running in separate thread"""
        while self.is_capturing:
            if self.capture is None:
                break
            
            ret, frame = self.capture.read()
            if not ret or frame is None:
                print("[WARN] Failed to read frame")
                time.sleep(0.1)
                continue
            
            # Crop to ROI for display
            if self.detector:
                roi_frame, roi_info = self.detector.crop_to_roi(frame)
            else:
                roi_frame = frame
                roi_info = None
            
            # Run detection if detector is available
            if self.detector:
                try:
                    annotated_frame, detections, processing_time = self.detector.detect_frame(frame)
                    # Crop annotated frame to ROI for display
                    annotated_roi, _ = self.detector.crop_to_roi(annotated_frame)
                    
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
                    
                    # Store for GUI update
                    self.current_frame = annotated_roi
                    self.current_detections = detections
                except Exception as e:
                    print(f"[ERROR] Detection failed: {e}")
                    self.current_frame = roi_frame
                    self.current_detections = []
            else:
                self.current_frame = roi_frame
                self.current_detections = []
                # Still increment frame number
                self.frame_number += 1
            
            # Update GUI from main thread (throttle to prevent queue buildup)
            # Only schedule if not too many pending updates
            if self.pending_gui_updates < 2:
                self.pending_gui_updates += 1
                self.root.after(0, lambda: self._safe_update_display())
            
            # Control frame rate (target ~15-30 FPS)
            time.sleep(0.033)  # ~30 FPS max
    
    def _safe_update_display(self):
        """Wrapper for update_display with exception handling and pending counter"""
        try:
            self.update_display()
        except Exception as e:
            print(f"[ERROR] Display update exception: {e}")
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
            
            # Update status labels
            self.fps_label.config(text=f"FPS: {self.fps:.1f}")
            self.detection_label.config(text=f"Detections: {len(self.current_detections)}")
            
            # Update log display periodically (not every frame for performance)
            current_time = time.time()
            if current_time - self.last_log_update >= self.log_update_interval:
                self.update_log_display()
                self.last_log_update = current_time
            
            # Update legend highlighting for active defects
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
        
        except Exception as e:
            print(f"[ERROR] Display update failed: {e}")
    
    def start_session(self):
        """Start detection session"""
        # Reset frame number for new session
        self.frame_number = 0
        self.defect_logger.start_session()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.session_status_label.config(text="Status: Active", foreground="green")
        self.log_path_label.config(text="Log file: Session active...", foreground="green")
        print("[INFO] Session started")
        self.update_log_display()
    
    def stop_session(self):
        """Stop detection session"""
        self.defect_logger.stop_session()
        
        # Save log file
        log_path = self.defect_logger.save_session_log()
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.session_status_label.config(text="Status: Stopped", foreground="gray")
        
        # Display log file path
        if log_path:
            self.log_path_label.config(
                text=f"Log file: {log_path.name} (saved to {log_path.parent})",
                foreground="blue"
            )
            print(f"[INFO] Session log saved: {log_path}")
        else:
            self.log_path_label.config(
                text="Log file: Failed to save",
                foreground="red"
            )
        
        print("[INFO] Session stopped")
        self.update_log_display()
    
    def on_closing(self):
        """Handle window close event"""
        self.is_capturing = False
        
        if self.capture:
            try:
                self.capture.release()
            except:
                pass
        
        # Stop session if active
        if self.defect_logger.session_active:
            self.defect_logger.stop_session()
            self.defect_logger.save_session_log()
        
        # Save thresholds and config
        self.save_thresholds()
        self.save_config()
        
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
                self.log_widgets['session_id'].config(text=f"Session ID: {stats['session_id']}")
                row += 1
            elif 'session_id' in self.log_widgets:
                self.log_widgets['session_id'].grid_remove()
            
            # Frames processed (reuse widget)
            if 'frames' not in self.log_widgets:
                self.log_widgets['frames'] = ttk.Label(self.log_container, font=("Courier", 9))
                self.log_widgets['frames'].grid(row=row, column=0, sticky=tk.W, padx=(10, 0), pady=(0, 5))
            self.log_widgets['frames'].config(text=f"Frames Processed: {stats['frames_processed']}")
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
            self.log_widgets['total_defects'].config(text=f"Total Defects: {stats['total_defects']}")
            row += 1
            
            if 'total_clusters' not in self.log_widgets:
                self.log_widgets['total_clusters'] = ttk.Label(self.log_container, font=("Courier", 9))
                self.log_widgets['total_clusters'].grid(row=row, column=0, sticky=tk.W, padx=(10, 0), pady=(0, 5))
            self.log_widgets['total_clusters'].config(text=f"Total Clusters: {stats['total_clusters']}")
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
                self.log_widgets['cluster_duration'].config(text=f"Duration: {cluster['duration']:.2f}s")
                self.log_widgets['cluster_frames'].config(text=f"Frames: {cluster['frame_count']}")
                self.log_widgets['cluster_defects'].config(text=f"Defects: {cluster['defect_count']}")
                classes_str = ', '.join(cluster['classes'].keys()) if cluster['classes'] else 'None'
                self.log_widgets['cluster_classes'].config(text=f"Classes: {classes_str}")
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
                self.log_widgets['timing_duration'].config(text=f"Duration: {timing['duration']:.3f}s")
                self.log_widgets['timing_duration'].grid()
                row += 1
                
                if 'timing_frames' not in self.log_widgets:
                    self.log_widgets['timing_frames'] = ttk.Label(self.log_container, font=("Courier", 9))
                    self.log_widgets['timing_frames'].grid(row=row, column=0, sticky=tk.W, padx=(10, 0), pady=(0, 5))
                self.log_widgets['timing_frames'].config(text=f"Frames: {timing['start_frame']} → {timing['end_frame']}")
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
                class_summary = ", ".join([f"{k}:{v}" for k, v in sorted(stats['class_counts'].items(), 
                                                                        key=lambda x: x[1], reverse=True)[:5]])
                self.log_widgets['class_header'].config(text=f"--- Per-Class Counts: {class_summary} ---")
                self.log_widgets['class_header'].grid()
            else:
                self.log_widgets['class_header'].config(text="--- Per-Class Counts: No defects ---")
                self.log_widgets['class_header'].grid()
            
        except Exception as e:
            print(f"[ERROR] Failed to update log display: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = DetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

