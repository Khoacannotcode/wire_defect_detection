#!/usr/bin/env python3
"""
Defect Logger Module
Tracks defect clusters and stripe timing for wire sessions.
Logs in human-friendly text format.
"""

from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional


class DefectLogger:
    """Logger for tracking defects and stripe timing during wire sessions"""
    
    def __init__(self, log_directory: str = "logs/defect_sessions"):
        """
        Initialize defect logger
        
        Args:
            log_directory: Directory to save session logs
        """
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Session state
        self.session_start_time = None
        self.session_end_time = None
        self.is_active = False
        
        # Tracking data
        self.defect_clusters = []  # List of defect cluster dicts
        self.current_cluster = None  # Current active cluster
        
        # Stripe timing
        self.first_normal_stripe_frame = None
        self.last_normal_stripe_frame = None
        self.first_normal_stripe_time = None
        self.last_normal_stripe_time = None
        
        # Per-class tracking
        self.first_defect_frame = {}  # First frame each defect class appeared
        self.last_defect_frame = {}  # Last frame each defect class appeared
        self.first_defect_time = {}  # First time each defect class appeared
        self.last_defect_time = {}  # Last time each defect class appeared
        
        # Frame counter
        self.frame_count = 0
        
        # Detection counts per class
        self.class_counts = defaultdict(int)
    
    def start_session(self):
        """Start a new wire session"""
        if self.is_active:
            return
        
        self.session_start_time = datetime.now()
        self.is_active = True
        
        # Reset tracking data
        self.defect_clusters = []
        self.current_cluster = None
        self.first_normal_stripe_frame = None
        self.last_normal_stripe_frame = None
        self.first_normal_stripe_time = None
        self.last_normal_stripe_time = None
        self.first_defect_frame = {}
        self.last_defect_frame = {}
        self.first_defect_time = {}
        self.last_defect_time = {}
        self.frame_count = 0
        self.class_counts = defaultdict(int)
        
        print(f"[INFO] Defect logger: Session started at {self.session_start_time}")
    
    def stop_session(self):
        """Stop current wire session"""
        if not self.is_active:
            return
        
        self.session_end_time = datetime.now()
        self.is_active = False
        
        # Close any active cluster
        if self.current_cluster:
            self._close_cluster()
        
        print(f"[INFO] Defect logger: Session stopped at {self.session_end_time}")
    
    def log_defects(self, frame_number: int, detections: List[Dict]):
        """
        Log defects detected in a frame
        
        Args:
            frame_number: Current frame number
            detections: List of detection dicts with 'class_name', 'confidence', etc.
        """
        if not self.is_active:
            return
        
        self.frame_count = frame_number
        current_time = datetime.now()
        
        # Track normal stripes
        normal_detections = [d for d in detections if d.get('class_name') == 'normal']
        if normal_detections:
            if self.first_normal_stripe_frame is None:
                self.first_normal_stripe_frame = frame_number
                self.first_normal_stripe_time = current_time
            self.last_normal_stripe_frame = frame_number
            self.last_normal_stripe_time = current_time
        
        # Track defect detections
        defect_detections = [d for d in detections if d.get('class_name') != 'normal']
        if not defect_detections:
            # No defects in this frame - close current cluster if exists
            if self.current_cluster:
                self._close_cluster()
            return
        
        # Update class counts
        for det in defect_detections:
            class_name = det.get('class_name')
            self.class_counts[class_name] += 1
            
            # Track first/last appearance
            if class_name not in self.first_defect_frame:
                self.first_defect_frame[class_name] = frame_number
                self.first_defect_time[class_name] = current_time
            self.last_defect_frame[class_name] = frame_number
            self.last_defect_time[class_name] = current_time
        
        # Handle defect cluster
        if self.current_cluster is None:
            # Start new cluster
            self.current_cluster = {
                'start_frame': frame_number,
                'end_frame': frame_number,
                'start_time': current_time,
                'end_time': current_time,
                'defects': defaultdict(int),
                'defect_list': []
            }
        else:
            # Extend current cluster
            self.current_cluster['end_frame'] = frame_number
            self.current_cluster['end_time'] = current_time
        
        # Add defects to cluster
        for det in defect_detections:
            class_name = det.get('class_name')
            self.current_cluster['defects'][class_name] += 1
            self.current_cluster['defect_list'].append({
                'class': class_name,
                'confidence': det.get('confidence', 0.0),
                'frame': frame_number
            })
    
    def _close_cluster(self):
        """Close current defect cluster and add to clusters list"""
        if self.current_cluster:
            self.defect_clusters.append(self.current_cluster.copy())
            self.current_cluster = None
    
    def save_session_log(self) -> Optional[Path]:
        """
        Save session log to human-friendly text file
        
        Returns:
            Path to saved log file, or None if error
        """
        if not self.session_start_time:
            return None
        
        # Generate filename with timestamp
        timestamp = self.session_start_time.strftime("%Y%m%d_%H%M%S")
        log_file = self.log_directory / f"{timestamp}_session.txt"
        
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                self._write_log_header(f)
                self._write_stripe_timing(f)
                self._write_defect_clusters(f)
                self._write_summary(f)
            
            print(f"[INFO] Session log saved to: {log_file}")
            return log_file
        except Exception as e:
            print(f"[ERROR] Failed to save session log: {e}")
            return None
    
    def _write_log_header(self, f):
        """Write log header"""
        f.write("Wire Session Log\n")
        f.write("=" * 50 + "\n")
        f.write(f"Session Start: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if self.session_end_time:
            f.write(f"Session End:   {self.session_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            duration = self.session_end_time - self.session_start_time
            minutes = int(duration.total_seconds() // 60)
            seconds = int(duration.total_seconds() % 60)
            f.write(f"Duration:      {minutes} minutes {seconds} seconds\n")
        else:
            f.write("Session End:   (Not ended)\n")
        
        f.write("\n")
    
    def _write_stripe_timing(self, f):
        """Write stripe timing information"""
        f.write("Stripe Timing:\n")
        
        if self.first_normal_stripe_time:
            f.write(f"- First stripe detected:  {self.first_normal_stripe_time.strftime('%Y-%m-%d %H:%M:%S')} (normal)\n")
            f.write(f"  Frame: {self.first_normal_stripe_frame}\n")
        else:
            f.write("- First stripe detected:  (Not detected)\n")
        
        if self.last_normal_stripe_time:
            f.write(f"- Last stripe detected:   {self.last_normal_stripe_time.strftime('%Y-%m-%d %H:%M:%S')} (normal)\n")
            f.write(f"  Frame: {self.last_normal_stripe_frame}\n")
        else:
            f.write("- Last stripe detected:   (Not detected)\n")
        
        if self.first_normal_stripe_time and self.last_normal_stripe_time:
            duration = self.last_normal_stripe_time - self.first_normal_stripe_time
            minutes = int(duration.total_seconds() // 60)
            seconds = int(duration.total_seconds % 60)
            f.write(f"- Stripe duration:        {minutes} minutes {seconds} seconds\n")
        
        f.write("\n")
    
    def _write_defect_clusters(self, f):
        """Write defect cluster information"""
        f.write("Defect Clusters:\n")
        
        if not self.defect_clusters:
            f.write("(No defect clusters detected)\n\n")
            return
        
        for i, cluster in enumerate(self.defect_clusters, 1):
            start_time = cluster['start_time'].strftime('%H:%M:%S')
            end_time = cluster['end_time'].strftime('%H:%M:%S')
            duration = cluster['end_time'] - cluster['start_time']
            duration_sec = int(duration.total_seconds())
            
            f.write(f"[{i}] {start_time} - {end_time} ({duration_sec} seconds)\n")
            f.write(f"    Frame range: {cluster['start_frame']}-{cluster['end_frame']}\n")
            
            # Group defects by class
            defect_summary = []
            for class_name, count in cluster['defects'].items():
                defect_summary.append(f"{class_name} ({count})")
            
            f.write(f"    Defects: {', '.join(defect_summary)}\n")
            f.write("\n")
    
    def _write_summary(self, f):
        """Write summary statistics"""
        f.write("Summary:\n")
        f.write(f"- Total defect clusters: {len(self.defect_clusters)}\n")
        
        total_defects = sum(self.class_counts.values())
        f.write(f"- Total defects detected: {total_defects}\n")
        
        if self.class_counts:
            f.write("- Classes: ")
            class_summary = [f"{cls} ({count})" for cls, count in self.class_counts.items()]
            f.write(", ".join(class_summary))
            f.write("\n")
        
        # Per-class first/last appearance
        if self.first_defect_time:
            f.write("\nDefect Class Timing:\n")
            for class_name in sorted(self.first_defect_time.keys()):
                first_time = self.first_defect_time[class_name].strftime('%H:%M:%S')
                first_frame = self.first_defect_frame.get(class_name, 'N/A')
                last_time = self.last_defect_time[class_name].strftime('%H:%M:%S')
                last_frame = self.last_defect_frame.get(class_name, 'N/A')
                f.write(f"- {class_name}:\n")
                f.write(f"  First: {first_time} (frame {first_frame})\n")
                f.write(f"  Last:  {last_time} (frame {last_frame})\n")

