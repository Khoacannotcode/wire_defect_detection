#!/usr/bin/env python3
"""
GUI Detection Runner for Jetson Nano
Desktop GUI application for real-time wire defect detection with threshold tuning,
defect logging, and session management.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
from pathlib import Path
import json
from datetime import datetime
from collections import deque

# Import detector from run_camera_detection
from run_camera_detection import LiveWireDetector, open_capture, MODELS_DIR

# Import defect logger (will be created in Task 5)
try:
    from defect_logger import DefectLogger
except ImportError:
    # Placeholder until Task 5
    DefectLogger = None


class DetectionGUI:
    """Main GUI application for wire defect detection"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Wire Defect Detection - Jetson Nano")
        self.root.geometry("1280x800")
        
        # State variables
        self.detector = None
        self.capture = None
        self.is_running = False
        self.is_session_active = False
        self.current_frame = None
        self.fps_deque = deque(maxlen=30)
        
        # Thresholds for each defect class (will be loaded from config)
        self.thresholds = {}
        
        # Load configuration
        self.config = self.load_config()
        
        # Initialize detector
        self.init_detector()
        
        # Setup GUI
        self.setup_gui()
        
        # Start camera capture thread
        self.capture_thread = None
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def load_config(self):
        """Load configuration from config.json or create default"""
        config_file = Path(__file__).parent / 'config.json'
        
        default_config = {
            'model_path': str(MODELS_DIR / 'best_cropped.onnx'),
            'camera_source': '0',
            'camera_width': 1280,
            'camera_height': 720,
            'camera_fps': 30,
            'use_gstreamer': False,
            'thresholds': {
                'NOK': 0.25,
                'breaks': 0.25,
                'damage': 0.25,
                'drops': 0.25,
                'shift': 0.25
            },
            'log_directory': 'logs'
        }
        
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # Merge with defaults
                    default_config.update(user_config)
                    # Ensure thresholds are present
                    if 'thresholds' not in default_config:
                        default_config['thresholds'] = default_config['thresholds']
                    else:
                        default_config['thresholds'].update(default_config['thresholds'])
            except Exception as e:
                print(f"[WARN] Failed to load config.json: {e}, using defaults")
        
        # Save config (to create file if it doesn't exist)
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to save config.json: {e}")
        
        return default_config
    
    def save_config(self):
        """Save current configuration to config.json"""
        config_file = Path(__file__).parent / 'config.json'
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"[ERROR] Failed to save config: {e}")
    
    def init_detector(self):
        """Initialize the detector"""
        try:
            model_path = Path(self.config['model_path'])
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            self.detector = LiveWireDetector(model_path)
            
            # Update thresholds from config
            thresholds_dict = {}
            for class_name in self.detector.defect_classes:
                if class_name in self.config['thresholds']:
                    threshold = self.config['thresholds'][class_name]
                    self.thresholds[class_name] = threshold
                    thresholds_dict[class_name] = threshold
                else:
                    self.thresholds[class_name] = 0.25
                    thresholds_dict[class_name] = 0.25
            
            # Set all thresholds at once
            if thresholds_dict:
                self.detector.set_class_thresholds(thresholds_dict)
            
            print("[INFO] Detector initialized successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize detector: {e}")
            self.root.quit()
    
    def setup_gui(self):
        """Setup the GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left side: Video display
        self.setup_video_display(main_frame)
        
        # Right side: Control panel
        self.setup_control_panel(main_frame)
    
    def setup_video_display(self, parent):
        """Setup video display area"""
        video_frame = ttk.LabelFrame(parent, text="Live Detection", padding="5")
        video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Video canvas
        self.video_label = tk.Label(video_frame, bg='black', width=960, height=540)
        self.video_label.pack(expand=True, fill=tk.BOTH)
        
        # Status bar below video
        status_frame = ttk.Frame(video_frame)
        status_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.fps_label = ttk.Label(status_frame, text="FPS: 0.0")
        self.fps_label.pack(side=tk.LEFT, padx=5)
        
        self.detection_count_label = ttk.Label(status_frame, text="Detections: 0")
        self.detection_count_label.pack(side=tk.LEFT, padx=5)
        
        self.session_status_label = ttk.Label(status_frame, text="Session: Stopped", 
                                              foreground="red")
        self.session_status_label.pack(side=tk.LEFT, padx=5)
    
    def setup_control_panel(self, parent):
        """Setup control panel with sliders and buttons"""
        control_frame = ttk.LabelFrame(parent, text="Controls", padding="10")
        control_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Session controls
        session_frame = ttk.LabelFrame(control_frame, text="Session Management", padding="5")
        session_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.start_button = ttk.Button(session_frame, text="Start Session", 
                                       command=self.start_session, width=20)
        self.start_button.pack(pady=5)
        
        self.stop_button = ttk.Button(session_frame, text="Stop Session", 
                                      command=self.stop_session, width=20, state=tk.DISABLED)
        self.stop_button.pack(pady=5)
        
        # Threshold controls
        threshold_frame = ttk.LabelFrame(control_frame, text="Threshold Tuning", padding="5")
        threshold_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.threshold_sliders = {}
        self.threshold_labels = {}
        
        for class_name in self.detector.defect_classes:
            # Create frame for each class
            class_frame = ttk.Frame(threshold_frame)
            class_frame.pack(fill=tk.X, pady=2)
            
            # Class name label with color indicator
            color = self.detector.colors.get(class_name, (128, 128, 128))
            color_hex = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"  # BGR to RGB hex
            
            class_label = ttk.Label(class_frame, text=f"{class_name}:", width=10)
            class_label.pack(side=tk.LEFT, padx=5)
            
            # Color indicator
            color_label = tk.Label(class_frame, bg=color_hex, width=3, height=1)
            color_label.pack(side=tk.LEFT, padx=2)
            
            # Slider
            threshold = self.thresholds.get(class_name, 0.25)
            slider = ttk.Scale(class_frame, from_=0.0, to=1.0, 
                              value=threshold, orient=tk.HORIZONTAL,
                              command=lambda val, cls=class_name: self.on_threshold_change(cls, val))
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            # Value label
            value_label = ttk.Label(class_frame, text=f"{threshold:.2f}", width=5)
            value_label.pack(side=tk.LEFT, padx=5)
            
            self.threshold_sliders[class_name] = slider
            self.threshold_labels[class_name] = value_label
        
        # Class color legend
        legend_frame = ttk.LabelFrame(control_frame, text="Class Colors", padding="5")
        legend_frame.pack(fill=tk.X, pady=(0, 10))
        
        for class_name in self.detector.defect_classes:
            legend_item = ttk.Frame(legend_frame)
            legend_item.pack(fill=tk.X, pady=1)
            
            color = self.detector.colors.get(class_name, (128, 128, 128))
            color_hex = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"
            
            tk.Label(legend_item, bg=color_hex, width=3, height=1).pack(side=tk.LEFT, padx=2)
            ttk.Label(legend_item, text=class_name).pack(side=tk.LEFT, padx=5)
        
        # Detection status
        status_frame = ttk.LabelFrame(control_frame, text="Detection Status", padding="5")
        status_frame.pack(fill=tk.X)
        
        self.defect_status_label = ttk.Label(status_frame, text="No defects detected", 
                                             foreground="green")
        self.defect_status_label.pack(pady=5)
        
        # Statistics
        self.stats_text = tk.Text(status_frame, height=8, width=30, wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        self.stats_text.insert('1.0', "Statistics will appear here...")
        self.stats_text.config(state=tk.DISABLED)
    
    def on_threshold_change(self, class_name, value):
        """Handle threshold slider change"""
        threshold = float(value)
        self.threshold_labels[class_name].config(text=f"{threshold:.2f}")
        
        # Update detector per-class threshold
        if self.detector:
            self.detector.set_class_threshold(class_name, threshold)
        
        # Save to config
        self.config['thresholds'][class_name] = threshold
        self.save_config()
    
    def start_session(self):
        """Start a new wire session"""
        if not self.is_running:
            messagebox.showwarning("Warning", "Please start detection first")
            return
        
        if self.is_session_active:
            messagebox.showinfo("Info", "Session already active")
            return
        
        self.is_session_active = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.session_status_label.config(text="Session: Active", foreground="green")
        
        # Initialize defect logger if available
        if DefectLogger:
            self.defect_logger = DefectLogger()
            self.defect_logger.start_session()
        
        print("[INFO] Session started")
    
    def stop_session(self):
        """Stop current wire session"""
        if not self.is_session_active:
            return
        
        self.is_session_active = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.session_status_label.config(text="Session: Stopped", foreground="red")
        
        # Save session log if logger available
        if DefectLogger and hasattr(self, 'defect_logger'):
            self.defect_logger.stop_session()
            log_path = self.defect_logger.save_session_log()
            if log_path:
                messagebox.showinfo("Session Logged", f"Session log saved to:\n{log_path}")
        
        print("[INFO] Session stopped")
    
    def start_detection(self):
        """Start detection loop"""
        if self.is_running:
            return
        
        try:
            # Open camera
            source = self.config['camera_source']
            width = self.config['camera_width']
            height = self.config['camera_height']
            fps = self.config['camera_fps']
            use_gstreamer = self.config.get('use_gstreamer', False)
            
            self.capture = open_capture(source, width, height, fps, use_gstreamer)
            
            self.is_running = True
            self.capture_thread = threading.Thread(target=self.detection_loop, daemon=True)
            self.capture_thread.start()
            
            print("[INFO] Detection started")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start detection: {e}")
            self.is_running = False
    
    def stop_detection(self):
        """Stop detection loop"""
        self.is_running = False
        if self.capture:
            self.capture.release()
            self.capture = None
        
        # Stop session if active
        if self.is_session_active:
            self.stop_session()
        
        print("[INFO] Detection stopped")
    
    def detection_loop(self):
        """Main detection loop running in separate thread"""
        frame_count = 0
        
        while self.is_running and self.capture:
            ret, frame = self.capture.read()
            if not ret:
                break
            
            # Run detection
            start_time = time.time()
            annotated_frame, detections, processing_time = self.detector.detect_frame(frame)
            
            # Update statistics
            self.detector.update_stats(detections, processing_time)
            
            # Update FPS
            fps = 1.0 / processing_time if processing_time > 0 else 0
            self.fps_deque.append(fps)
            
            # Log defects and normal stripes if session active
            if self.is_session_active and DefectLogger and hasattr(self, 'defect_logger'):
                # Log all detections (including normal for stripe timing)
                self.defect_logger.log_defects(frame_count, detections)
            
            # Convert frame for display
            self.current_frame = annotated_frame.copy()
            
            # Update GUI (must be done in main thread)
            self.root.after(0, self.update_display, annotated_frame, detections, fps, frame_count)
            
            frame_count += 1
            
            # Small delay to prevent overwhelming the GUI
            time.sleep(0.01)
    
    def update_display(self, frame, detections, fps, frame_count):
        """Update GUI display (called from main thread)"""
        # Update video display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (960, 540))
        img = Image.fromarray(frame_resized)
        img_tk = ImageTk.PhotoImage(image=img)
        
        self.video_label.config(image=img_tk)
        self.video_label.image = img_tk  # Keep a reference
        
        # Update FPS
        avg_fps = np.mean(self.fps_deque) if self.fps_deque else 0
        self.fps_label.config(text=f"FPS: {avg_fps:.1f}")
        
        # Update detection count
        defect_count = len([d for d in detections if d['class_name'] in self.detector.defect_classes])
        self.detection_count_label.config(text=f"Detections: {defect_count}")
        
        # Update defect status
        if defect_count > 0:
            self.defect_status_label.config(text=f"⚠ Defects detected: {defect_count}", 
                                           foreground="red")
        else:
            self.defect_status_label.config(text="✓ No defects", foreground="green")
        
        # Update statistics
        self.update_statistics()
    
    def update_statistics(self):
        """Update statistics display"""
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete('1.0', tk.END)
        
        stats_lines = ["Detection Statistics:\n"]
        stats_lines.append(f"Total detections: {sum(self.detector.detection_counts.values())}\n\n")
        
        for class_name in self.detector.class_names:
            count = self.detector.detection_counts.get(class_name, 0)
            stats_lines.append(f"{class_name}: {count}\n")
        
        if self.is_session_active and hasattr(self, 'defect_logger'):
            stats_lines.append("\n--- Session Info ---\n")
            # Add session stats from logger if available
            stats_lines.append("Session active\n")
        
        self.stats_text.insert('1.0', ''.join(stats_lines))
        self.stats_text.config(state=tk.DISABLED)
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_detection()
        self.root.quit()
        self.root.destroy()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = DetectionGUI(root)
    
    # Start detection automatically
    app.start_detection()
    
    # Run GUI main loop
    root.mainloop()


if __name__ == "__main__":
    main()

