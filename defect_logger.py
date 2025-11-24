#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wire Defect Detection - Defect Logger Module
Tracks defect clusters, normal stripes, and generates human-friendly log files
"""

import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional


class DefectCluster:
    """Represents a cluster of consecutive defect detections"""
    
    def __init__(self, start_frame: int, start_time: float, first_detection: Dict):
        self.start_frame = start_frame
        self.start_time = start_time
        self.end_frame = start_frame
        self.end_time = start_time
        self.detections = [first_detection]
        self.class_counts = defaultdict(int)
        self.class_counts[first_detection['class_name']] += 1
    
    def add_detection(self, frame: int, timestamp: float, detection: Dict):
        """Add a detection to this cluster"""
        self.end_frame = frame
        self.end_time = timestamp
        self.detections.append(detection)
        self.class_counts[detection['class_name']] += 1
    
    def get_duration(self) -> float:
        """Get cluster duration in seconds"""
        return self.end_time - self.start_time
    
    def get_frame_count(self) -> int:
        """Get number of frames in cluster"""
        return self.end_frame - self.start_frame + 1


class DefectLogger:
    """Logs defects with cluster tracking and human-friendly text format"""
    
    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize DefectLogger
        
        Args:
            log_dir: Directory to save log files (default: shipping/logs/)
        """
        if log_dir is None:
            log_dir = Path(__file__).parent / 'logs'
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Session state
        self.session_active = False
        self.session_start_time = None
        self.session_start_frame = None
        self.session_end_time = None
        self.session_end_frame = None
        self.session_id = None
        
        # Frame tracking
        self.current_frame = 0
        self.last_defect_frame = -1  # Last frame with defects
        self.cluster_gap_threshold = 15  # Frames without defects to close cluster (≈0.5s at 30 FPS)
        
        # Defect clusters
        self.active_cluster: Optional[DefectCluster] = None
        self.closed_clusters: List[DefectCluster] = []
        
        # Normal stripe tracking (for stripe timing)
        self.first_normal_frame = None
        self.first_normal_time = None
        self.last_normal_frame = None
        self.last_normal_time = None
        
        # Per-class tracking
        self.class_first_appearance: Dict[str, Dict] = {}  # {class_name: {frame, time}}
        self.class_last_appearance: Dict[str, Dict] = {}   # {class_name: {frame, time}}
        self.class_detection_counts: Dict[str, int] = defaultdict(int)
        
        # All detections (for summary)
        self.all_detections: List[Dict] = []
    
    def start_session(self):
        """Start a new logging session"""
        if self.session_active:
            print("[WARN] Session already active, stopping previous session first")
            self.stop_session()
        
        self.session_active = True
        self.session_start_time = time.time()
        self.session_start_frame = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Reset all tracking
        self.current_frame = 0
        self.last_defect_frame = -1
        self.active_cluster = None
        self.closed_clusters = []
        self.first_normal_frame = None
        self.first_normal_time = None
        self.last_normal_frame = None
        self.last_normal_time = None
        self.class_first_appearance = {}
        self.class_last_appearance = {}
        self.class_detection_counts = defaultdict(int)
        self.all_detections = []
        
        print("[INFO] Defect logging session started: {}".format(self.session_id))
    
    def stop_session(self):
        """Stop the current logging session"""
        if not self.session_active:
            print("[WARN] No active session to stop")
            return
        
        self.session_active = False
        self.session_end_time = time.time()
        self.session_end_frame = self.current_frame
        
        # Close any active cluster
        if self.active_cluster:
            self.closed_clusters.append(self.active_cluster)
            self.active_cluster = None
        
        print("[INFO] Defect logging session stopped: {}".format(self.session_id))
    
    def log_defects(self, frame_number: int, detections: List[Dict]):
        """
        Log defects for a frame
        
        Args:
            frame_number: Current frame number
            detections: List of detection dicts with keys: class_name, confidence, bbox
        """
        if not self.session_active:
            return
        
        self.current_frame = frame_number
        timestamp = time.time()
        
        # Filter detections (only defect classes, exclude 'normal')
        defect_detections = [
            det for det in detections 
            if det.get('class_name') != 'normal'
        ]
        normal_detections = [
            det for det in detections 
            if det.get('class_name') == 'normal'
        ]
        
        # Track normal stripes (for stripe timing)
        if normal_detections:
            if self.first_normal_frame is None:
                self.first_normal_frame = frame_number
                self.first_normal_time = timestamp
            self.last_normal_frame = frame_number
            self.last_normal_time = timestamp
        
        # Track defects
        if defect_detections:
            self.last_defect_frame = frame_number
            
            # Update per-class tracking
            for det in defect_detections:
                class_name = det['class_name']
                self.class_detection_counts[class_name] += 1
                
                # Track first appearance
                if class_name not in self.class_first_appearance:
                    self.class_first_appearance[class_name] = {
                        'frame': frame_number,
                        'time': timestamp
                    }
                
                # Track last appearance
                self.class_last_appearance[class_name] = {
                    'frame': frame_number,
                    'time': timestamp
                }
            
            # Store all detections
            self.all_detections.extend(defect_detections)
            
            # Handle cluster tracking
            if self.active_cluster is None:
                # Start new cluster
                self.active_cluster = DefectCluster(
                    frame_number, timestamp, defect_detections[0]
                )
                # Add remaining detections to cluster
                for det in defect_detections[1:]:
                    self.active_cluster.add_detection(frame_number, timestamp, det)
            else:
                # Add to existing cluster
                for det in defect_detections:
                    self.active_cluster.add_detection(frame_number, timestamp, det)
        else:
            # No defects in this frame
            if self.active_cluster is not None:
                # Check if we should close the cluster (gap threshold)
                gap_frames = frame_number - self.last_defect_frame
                if gap_frames >= self.cluster_gap_threshold:
                    # Close cluster
                    self.closed_clusters.append(self.active_cluster)
                    self.active_cluster = None
    
    def get_stripe_timing(self) -> Optional[Dict]:
        """
        Get stripe timing information (first normal to last normal)
        
        Returns:
            Dict with keys: start_time, end_time, duration, start_frame, end_frame
            or None if no normal stripes detected
        """
        if self.first_normal_time is None or self.last_normal_time is None:
            return None
        
        return {
            'start_time': self.first_normal_time,
            'end_time': self.last_normal_time,
            'duration': self.last_normal_time - self.first_normal_time,
            'start_frame': self.first_normal_frame,
            'end_frame': self.last_normal_frame
        }
    
    def save_session_log(self) -> Optional[Path]:
        """
        Save session log to file in human-friendly text format
        
        Returns:
            Path to saved log file, or None if no session or error
        """
        if not self.session_start_time:
            print("[WARN] No session data to save")
            return None
        
        # Generate log file path
        log_filename = "session_{}.log".format(self.session_id)
        log_path = self.log_dir / log_filename
        
        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                self._write_log_header(f)
                self._write_session_info(f)
                self._write_stripe_timing(f)
                self._write_defect_summary(f)
                self._write_defect_clusters(f)
                self._write_per_class_stats(f)
                self._write_log_footer(f)
            
            print("[INFO] Session log saved: {}".format(log_path))
            return log_path
        except Exception as e:
            print("[ERROR] Failed to save session log: {}".format(e))
            return None
    
    def _write_log_header(self, f):
        """Write log file header"""
        f.write("=" * 80 + "\n")
        f.write("WIRE DEFECT DETECTION - SESSION LOG\n")
        f.write("=" * 80 + "\n")
        f.write("Generated: {}\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        f.write("Session ID: {}\n".format(self.session_id))
        f.write("\n")
    
    def _write_session_info(self, f):
        """Write session information"""
        f.write("-" * 80 + "\n")
        f.write("SESSION INFORMATION\n")
        f.write("-" * 80 + "\n")
        
        start_dt = datetime.fromtimestamp(self.session_start_time)
        f.write("Start Time: {}\n".format(start_dt.strftime('%Y-%m-%d %H:%M:%S')))
        
        if self.session_end_time:
            end_dt = datetime.fromtimestamp(self.session_end_time)
            duration = self.session_end_time - self.session_start_time
            f.write("End Time:   {}\n".format(end_dt.strftime('%Y-%m-%d %H:%M:%S')))
            f.write("Duration:   {:.2f} seconds ({:.2f} minutes)\n".format(duration, duration/60))
        else:
            f.write("End Time:   Session still active\n")
        
        f.write("Frames Processed: {}\n".format(self.current_frame))
        f.write("\n")
    
    def _write_stripe_timing(self, f):
        """Write stripe timing information"""
        f.write("-" * 80 + "\n")
        f.write("STRIPE TIMING\n")
        f.write("-" * 80 + "\n")
        
        timing = self.get_stripe_timing()
        if timing:
            start_dt = datetime.fromtimestamp(timing['start_time'])
            end_dt = datetime.fromtimestamp(timing['end_time'])
            
            f.write("First Normal Stripe:\n")
            f.write("  Frame: {}\n".format(timing['start_frame']))
            f.write("  Time:  {}\n".format(start_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]))
            f.write("\n")
            f.write("Last Normal Stripe:\n")
            f.write("  Frame: {}\n".format(timing['end_frame']))
            f.write("  Time:  {}\n".format(end_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]))
            f.write("\n")
            f.write("Stripe Duration: {:.3f} seconds\n".format(timing['duration']))
        else:
            f.write("No normal stripes detected in this session.\n")
        
        f.write("\n")
    
    def _write_defect_summary(self, f):
        """Write defect summary statistics"""
        f.write("-" * 80 + "\n")
        f.write("DEFECT SUMMARY\n")
        f.write("-" * 80 + "\n")
        
        total_defects = len(self.all_detections)
        total_clusters = len(self.closed_clusters)
        active_cluster_info = ""
        if self.active_cluster:
            total_clusters += 1
            active_cluster_info = " (1 active, {} closed)".format(len(self.closed_clusters))
        
        f.write("Total Defects Detected: {}\n".format(total_defects))
        f.write("Total Defect Clusters: {}{}\n".format(total_clusters, active_cluster_info))
        f.write("\n")
    
    def _write_defect_clusters(self, f):
        """Write detailed defect cluster information"""
        f.write("-" * 80 + "\n")
        f.write("DEFECT CLUSTERS\n")
        f.write("-" * 80 + "\n")
        
        if not self.closed_clusters and not self.active_cluster:
            f.write("No defect clusters detected in this session.\n")
            f.write("\n")
            return
        
        # Write closed clusters
        for i, cluster in enumerate(self.closed_clusters, 1):
            start_dt = datetime.fromtimestamp(cluster.start_time)
            end_dt = datetime.fromtimestamp(cluster.end_time)
            
            f.write("\nCluster {} (Closed):\n".format(i))
            f.write("  Start: Frame {} at {}\n".format(cluster.start_frame, start_dt.strftime('%H:%M:%S.%f')[:-3]))
            f.write("  End:   Frame {} at {}\n".format(cluster.end_frame, end_dt.strftime('%H:%M:%S.%f')[:-3]))
            f.write("  Duration: {:.3f} seconds ({} frames)\n".format(cluster.get_duration(), cluster.get_frame_count()))
            f.write("  Defects: {}\n".format(len(cluster.detections)))
            f.write("  Classes: {}\n".format(dict(cluster.class_counts)))
        
        # Write active cluster if exists
        if self.active_cluster:
            start_dt = datetime.fromtimestamp(self.active_cluster.start_time)
            f.write("\nCluster {} (Active):\n".format(len(self.closed_clusters) + 1))
            f.write("  Start: Frame {} at {}\n".format(self.active_cluster.start_frame, start_dt.strftime('%H:%M:%S.%f')[:-3]))
            f.write("  End:   Frame {} (ongoing)\n".format(self.active_cluster.end_frame))
            f.write("  Duration: {:.3f} seconds ({} frames)\n".format(self.active_cluster.get_duration(), self.active_cluster.get_frame_count()))
            f.write("  Defects: {}\n".format(len(self.active_cluster.detections)))
            f.write("  Classes: {}\n".format(dict(self.active_cluster.class_counts)))
        
        f.write("\n")
    
    def _write_per_class_stats(self, f):
        """Write per-class statistics"""
        f.write("-" * 80 + "\n")
        f.write("PER-CLASS STATISTICS\n")
        f.write("-" * 80 + "\n")
        
        if not self.class_detection_counts:
            f.write("No defects detected in this session.\n")
            f.write("\n")
            return
        
        # Sort by detection count (descending)
        sorted_classes = sorted(
            self.class_detection_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for class_name, count in sorted_classes:
            f.write("\n{}:\n".format(class_name))
            f.write("  Total Detections: {}\n".format(count))
            
            if class_name in self.class_first_appearance:
                first = self.class_first_appearance[class_name]
                first_dt = datetime.fromtimestamp(first['time'])
                f.write("  First Appearance: Frame {} at {}\n".format(first['frame'], first_dt.strftime('%H:%M:%S.%f')[:-3]))
            
            if class_name in self.class_last_appearance:
                last = self.class_last_appearance[class_name]
                last_dt = datetime.fromtimestamp(last['time'])
                f.write("  Last Appearance:  Frame {} at {}\n".format(last['frame'], last_dt.strftime('%H:%M:%S.%f')[:-3]))
        
        f.write("\n")
    
    def _write_log_footer(self, f):
        """Write log file footer"""
        f.write("=" * 80 + "\n")
        f.write("END OF SESSION LOG\n")
        f.write("=" * 80 + "\n")
    
    def get_active_cluster_info(self) -> Optional[Dict]:
        """
        Get information about the currently active cluster
        
        Returns:
            Dict with cluster info or None if no active cluster
        """
        if self.active_cluster is None:
            return None
        
        return {
            'start_frame': self.active_cluster.start_frame,
            'start_time': self.active_cluster.start_time,
            'end_frame': self.active_cluster.end_frame,
            'end_time': self.active_cluster.end_time,
            'duration': self.active_cluster.get_duration(),
            'frame_count': self.active_cluster.get_frame_count(),
            'defect_count': len(self.active_cluster.detections),
            'classes': dict(self.active_cluster.class_counts)
        }
    
    def get_session_stats(self) -> Dict:
        """
        Get current session statistics
        
        Returns:
            Dict with session statistics
        """
        return {
            'session_active': self.session_active,
            'session_id': self.session_id,
            'frames_processed': self.current_frame,
            'total_defects': len(self.all_detections),
            'total_clusters': len(self.closed_clusters) + (1 if self.active_cluster else 0),
            'active_cluster': self.get_active_cluster_info(),
            'stripe_timing': self.get_stripe_timing(),
            'class_counts': dict(self.class_detection_counts)
        }

