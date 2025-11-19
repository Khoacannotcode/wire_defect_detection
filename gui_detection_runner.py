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

# Add system packages to path for compatibility
sys.path.insert(0, '/usr/lib/python3/dist-packages')


class DetectionGUI:
    """Desktop GUI application for wire defect detection"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Wire Defect Detection - Real-time Monitor")
        self.root.geometry("1024x768")
        
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
        
        # Config
        self.config_file = ROOT_DIR / 'config.json'
        self.config = self.load_config()
        
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
            'display_height': 600
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
        video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))
        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # Video display label
        self.video_label = ttk.Label(video_frame, text="Initializing camera...", 
                                     background="black", foreground="white")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
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
        
        # Statistics panel (placeholder for future tasks)
        stats_frame = ttk.LabelFrame(control_frame, text="Statistics", padding="5")
        stats_frame.pack(fill=tk.BOTH, expand=True)
        
        self.stats_text = tk.Text(stats_frame, height=4, wrap=tk.WORD, state=tk.DISABLED)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        # Class color legend with visual color swatches
        legend_frame = ttk.LabelFrame(control_frame, text="Class Colors", padding="5")
        legend_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Create frame for color swatches (visual, human-friendly)
        self.legend_container = ttk.Frame(legend_frame)
        self.legend_container.pack(fill=tk.X)
        
        # Update legend if detector is available
        if self.detector:
            self.update_legend()
    
    def update_legend(self):
        """Update class color legend with visual color swatches (human-friendly)"""
        if not self.detector:
            return
        
        # Clear existing legend widgets
        for widget in self.legend_container.winfo_children():
            widget.destroy()
        
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
                    
                    # Update FPS
                    self.frame_count += 1
                    current_time = time.time()
                    if current_time - self.last_fps_update >= 1.0:
                        self.fps = self.frame_count / (current_time - self.last_fps_update)
                        self.frame_count = 0
                        self.last_fps_update = current_time
                    
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
            
            # Update GUI from main thread
            self.root.after(0, self.update_display)
            
            # Control frame rate (target ~15-30 FPS)
            time.sleep(0.033)  # ~30 FPS max
    
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
            
            # Resize ROI frame to fill entire label (stretch to fit, no black borders)
            resized = cv2.resize(self.current_frame, (label_width, label_height), 
                                interpolation=cv2.INTER_LINEAR)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(image=pil_image)
            
            # Update label
            self.video_label.config(image=photo, text="")
            self.video_label.image = photo  # Keep a reference
            
            # Update status labels
            self.fps_label.config(text=f"FPS: {self.fps:.1f}")
            self.detection_label.config(text=f"Detections: {len(self.current_detections)}")
            
            # Update statistics
            if self.detector:
                stats_lines = []
                stats_lines.append(f"Total detections: {sum(self.detector.detection_counts.values())}")
                for class_name in self.detector.defect_classes:
                    count = self.detector.detection_counts.get(class_name, 0)
                    stats_lines.append(f"{class_name}: {count}")
                
                self.stats_text.config(state=tk.NORMAL)
                self.stats_text.delete(1.0, tk.END)
                self.stats_text.insert(1.0, "\n".join(stats_lines))
                self.stats_text.config(state=tk.DISABLED)
        
        except Exception as e:
            print(f"[ERROR] Display update failed: {e}")
    
    def start_session(self):
        """Start detection session"""
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.session_status_label.config(text="Status: Active", foreground="green")
        print("[INFO] Session started")
    
    def stop_session(self):
        """Stop detection session"""
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.session_status_label.config(text="Status: Stopped", foreground="gray")
        print("[INFO] Session stopped")
    
    def on_closing(self):
        """Handle window close event"""
        self.is_capturing = False
        
        if self.capture:
            try:
                self.capture.release()
            except:
                pass
        
        # Save config
        self.save_config()
        
        self.root.destroy()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = DetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

