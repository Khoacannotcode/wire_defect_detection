# -*- coding: utf-8 -*-
import cv2
import time
from datetime import datetime

# Camera tuning parameters
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

# Build the GStreamer pipeline
gst_pipeline = (
    'nvarguscamerasrc {} ! '
    'video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=120/1 ! '
    'nvvidconv ! video/x-raw, format=BGRx ! '
    'videoconvert ! video/x-raw, format=BGR ! '
    'appsink max-buffers=1 drop=true'
).format(CAMERA_PROPERTY_STRING)

# Open camera
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("❌ Could not open camera")
    exit()

# Prepare output video writer
output_filename = "wire_recording_{}.avi".format(datetime.now().strftime('%Y%m%d_%H%M%S'))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_filename, fourcc, 30.0, (1280, 80))  # recording cropped region

# Cropping configuration: center strip
frame_height = 720
strip_height = 80
y_top = (frame_height - strip_height) // 2
y_bottom = y_top + strip_height

record_seconds = 60
start_time = time.time()
print("🎥 Recording started: {}".format(output_filename))

while time.time() - start_time < record_seconds:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Stream ended or error.")
        break

    cropped = frame[y_top:y_bottom, :]
    out.write(cropped)
    cv2.imshow("Wire Preview", cropped)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to stop early
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Recording complete.")


