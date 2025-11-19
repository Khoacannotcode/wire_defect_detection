# Wire Defect Detection - GUI Usage Guide

## Quick Start

### Running the GUI Application on Linux/Jetson Nano

```bash
cd shipping
chmod +x run_gui.sh
./run_gui.sh
```

The script will:

1. Activate the virtual environment automatically
2. Start the GUI application
3. Display any errors if the application fails

**Note:** If you see "Virtual environment not found", run `setup_environment.sh` first to create the virtual environment.

## GUI Overview

The GUI application provides real-time wire defect detection with the following features:

### Main Components

1. **Live Video Display** (Top Section)

   - Shows real-time video feed with ROI (Region of Interest) cropped view
   - Displays defect bounding boxes (colored rectangles) when defects are detected
   - Only defect classes are shown (normal class is filtered out)
2. **Control Panel** (Bottom Section)

   - Session management controls
   - Real-time statistics and alerts
   - Threshold controls
   - Log display

## Using the GUI

### 1. Session Management

**Start Session:**

- Click "Start Session" button to begin logging defects
- Session status changes to "Active" (green)
- Session duration starts counting
- Defect logging begins automatically

**Stop Session:**

- Click "Stop Session" button to end the session
- Session log file is automatically saved to `shipping/logs/session_YYYYMMDD_HHMMSS.log`
- Log file path is displayed below
- Click "Open Log" button to view the log file in your default text editor

**Session Duration:**

- Displays real-time session duration (updates every second)
- Format: "Duration: Xm Ys" or "Duration: Xs"
- Shows final duration when session stops

### 2. Visual Alerts

**Defect Status Indicator:**

- **Green dot (●) + "Status: OK"**: No defects detected in current frame
- **Red dot (●) + "Status: DEFECT (X)"**: Defects detected (X = number of defects)

**Active Classes Display:**

- Shows defect classes currently detected in the frame
- Format: "Active: class1, class2, ..."
- Only appears when defects are present

### 3. Threshold Controls

**Per-Class Threshold Sliders:**

- Each defect class (NOK, breaks, damage, drops, shift) has its own threshold slider
- Range: 0.0 to 1.0
- Current value displayed next to slider
- Color indicator shows class color

**How to Use:**

1. Adjust slider to change detection threshold for that class
2. Changes apply immediately (no restart needed)
3. Thresholds are automatically saved to `config.json` after 500ms of no changes
4. Thresholds are loaded automatically when GUI starts

**Tips:**

- Lower threshold = more sensitive (more detections, may include false positives)
- Higher threshold = less sensitive (fewer detections, may miss some defects)
- Default: 0.25 for all classes

### 4. Real-time Log & Statistics

**Session Information:**

- Session ID (timestamp-based)
- Frames processed
- Session status (Active/Stopped)

**Defect Summary:**

- Total defects detected in session
- Total defect clusters (groups of consecutive defects)

**Active Cluster:**

- Shows currently active defect cluster (if any)
- Displays: Duration, Frames, Defects count, Classes involved
- Updates in real-time

**Stripe Timing:**

- First normal stripe appearance (frame + time)
- Last normal stripe appearance (frame + time)
- Duration between first and last normal stripe
- Used for calculating wire processing timing

**Per-Class Counts:**

- Summary of detection counts per class (top 5 classes)
- Format: "class1:count1, class2:count2, ..."

### 5. Defect Classes Legend

**Color Swatches:**

- Visual color indicators for each defect class
- Active defects are highlighted (bold text, red border)
- Inactive defects show normal appearance

## Performance Tips

**FPS Optimization:**

- GUI is optimized for 15-30 FPS on Jetson Nano
- Log display updates every 1 second (not every frame)
- Visual alerts update every 200ms
- Legend highlighting updates every 300ms

**If FPS is still low:**

- Check camera connection and settings
- Verify model file exists and is correct
- Check system resources (CPU/GPU usage)
- Consider reducing video resolution in config.json

## Configuration

**config.json:**

- Automatically created in directory
- Contains: model path, camera settings, display settings, thresholds
- Can be edited manually if needed
- Changes are saved automatically

## Log Files

**Location:** `logs/session_YYYYMMDD_HHMMSS.log`

**Format:** Human-friendly text format, includes:

- Session information (start/end time, duration)
- Stripe timing (first/last normal stripe, duration)
- Defect summary (total defects, clusters)
- Defect clusters (detailed information for each cluster)
- Per-class statistics (counts, first/last appearance)

**Viewing Logs:**

- Click "Open Log" button after stopping a session
- Or manually open files from `logs/` directory
- Logs are plain text files, readable in any text editor

## Troubleshooting

**GUI doesn't start:**

- Check virtual environment is activated
- Verify Python dependencies are installed
- Check camera is connected and accessible
- Verify model file exists at specified path

**No video display:**

- Check camera connection
- Verify camera permissions
- Check camera source in config.json (default: '0')
- Try different camera source numbers if multiple cameras

**Low FPS:**

- System automatically optimizes, but if still low:
  - Check camera resolution settings
  - Verify GPU/CUDA is available and working
  - Check system resources
  - Reduce update frequencies in code if needed

**Defects not detected:**

- Adjust threshold sliders (lower = more sensitive)
- Check if model file is correct (6-class model)
- Verify camera is focused on wire area
- Check ROI (Region of Interest) is correct

**Session log not saved:**

- Check write permissions in `logs/` directory
- Verify disk space is available
- Check for error messages in terminal

## Keyboard Shortcuts

- **Close Window**: Click X button
- All operations are mouse-based (no keyboard shortcuts needed)

## Best Practices

1. **Start Session** before beginning wire inspection
2. **Stop Session** after inspection is complete
3. **Review Log File** to analyze defect patterns
4. **Adjust Thresholds** based on detection accuracy needs
5. **Monitor FPS** - should be 15-30 FPS for smooth operation
6. **Check Active Clusters** to see current defect status

## Notes

- GUI window size: 1024x1080 (optimized for log display)
- All updates are optimized for performance
- Log files are timestamped to prevent overwriting
- Config file is automatically saved when closing GUI
