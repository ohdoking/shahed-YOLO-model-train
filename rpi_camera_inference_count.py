import cv2
from ultralytics import YOLO
from picamera2 import Picamera2 # 💡 New Import
import numpy as np # 💡 New Import

def main():
    # --- Picamera2 Setup ---
    picam2 = Picamera2()
    # Configure for video/preview, set resolution for speed/processing
    config = picam2.create_video_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    # -----------------------

    model = YOLO('/home/dokeunoh/Desktop/project/shahed-YOLO-model-train/best_tmp.pt')

    # Removed: cap = cv2.VideoCapture(0)
    # Removed: if not cap.isOpened(): ...

    while True:
        # 💡 Capture the frame as a NumPy array (RGB)
        frame = picam2.capture_array()
        
        # Convert RGB image (from Picamera2) to BGR (expected by OpenCV/YOLO)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Removed: ret, frame = cap.read() # frame_bgr is the frame now
        
        # YOLO Processing uses the BGR frame
        results = model(frame_bgr, conf=0.35)

        # Get detected boxes, confidences, and class IDs
        boxes = results[0].boxes
        count = len(boxes)

        # Draw bounding boxes and labels on frame
        annotated_frame = results[0].plot()
        
        # Put text of count on top-left corner
        text = f"Objects detected: {count}"
        cv2.putText(annotated_frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show frame
        cv2.imshow('YOLOv8 RPi Camera', annotated_frame)

        # Print count in terminal
        print(text)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Use picam2.stop() instead of cap.release()
    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()