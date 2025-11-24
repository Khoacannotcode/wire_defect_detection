#!/usr/bin/env python3
"""
Visualization standards for bounding boxes
Standardized colors and drawing functions for consistent visualization across all scripts
"""

import cv2
import numpy as np

def get_class_color(class_name):
    """
    Get color for class according to visualization standards
    Returns: BGR color tuple
    
    Color standards:
    - NOK: Orange
    - breaks: Yellow
    - damage: Red
    - drops: Gray
    - normal: Dark green
    - shift: Blue
    """
    color_map = {
        'NOK': (0, 165, 255),      # Orange (BGR)
        'breaks': (0, 255, 255),   # Yellow (BGR)
        'damage': (0, 0, 255),     # Red (BGR)
        'drops': (128, 128, 128),  # Gray (BGR)
        'normal': (0, 128, 0),      # Dark green (BGR)
        'shift': (255, 0, 0),       # Blue (BGR)
        # Legacy classes (backward compatibility)
        'fail': (0, 0, 255),       # Red
        'pagan': (255, 0, 0),      # Blue
        'valid': (0, 255, 0),      # Green
    }
    return color_map.get(class_name, (128, 128, 128))  # Default gray

def draw_bbox_with_standards(image, x1, y1, x2, y2, class_name, label_text=None):
    """
    Draw bounding box with visualization standards:
    - Square corners (no rounded corners) - cv2.rectangle already has square corners
    - Border: 100% opacity color
    - Fill overlay: 30% opacity color (transparent, see-through)
    
    Args:
        image: Image to draw on (will be modified)
        x1, y1, x2, y2: Bounding box coordinates
        class_name: Class name for color lookup
        label_text: Optional label text to display above bbox
    
    Returns:
        Modified image
    """
    # Get color for class
    color = get_class_color(class_name)
    
    # Create overlay for transparent fill
    overlay = image.copy()
    
    # Draw fill overlay (will be blended with 30% opacity)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)  # -1 = filled
    
    # Blend overlay with original image (30% opacity for fill)
    alpha = 0.3  # 30% opacity
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    # Draw border with 100% opacity (solid color, thickness 2)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Draw label if provided
    if label_text:
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        # Label background (solid color, 100% opacity)
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        # Label text
        cv2.putText(image, label_text, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return image

def draw_multiple_bboxes(image, bboxes_info):
    """
    Draw multiple bounding boxes efficiently on image
    
    Args:
        image: Image to draw on (will be modified)
        bboxes_info: List of dicts with keys: 'x1', 'y1', 'x2', 'y2', 'class_name', 'label' (optional)
    
    Returns:
        Modified image
    """
    # Create overlay for all fills
    overlay = image.copy()
    
    # First pass: Draw all fill overlays
    for bbox_info in bboxes_info:
        x1 = bbox_info['x1']
        y1 = bbox_info['y1']
        x2 = bbox_info['x2']
        y2 = bbox_info['y2']
        class_name = bbox_info['class_name']
        color = get_class_color(class_name)
        
        # Draw fill overlay
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    
    # Blend overlay with original image (30% opacity for fill)
    alpha = 0.3  # 30% opacity
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    # Second pass: Draw borders and labels (100% opacity)
    for bbox_info in bboxes_info:
        x1 = bbox_info['x1']
        y1 = bbox_info['y1']
        x2 = bbox_info['x2']
        y2 = bbox_info['y2']
        class_name = bbox_info['class_name']
        label_text = bbox_info.get('label', None)
        color = get_class_color(class_name)
        
        # Draw border with 100% opacity
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label if provided
        if label_text:
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(image, label_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return image

