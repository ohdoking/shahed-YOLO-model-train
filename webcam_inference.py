import cv2
from ultralytics import YOLO

def main():
    # Load your trained YOLOv8 model (change path if needed)
    model = YOLO('/Users/dokeunoh/DevOh/project/private/project/hackerton/edth-2025/shahed-model-train/best_tmp.pt')  # or 'yolov8n.pt' for pretrained

    # Open webcam (0 is default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Run YOLOv8 inference on frame (returns a list of results)
        results = model(frame, conf=0.35, iou=0.5)

        # Render results on frame (bounding boxes, labels)
        annotated_frame = results[0].plot()
        boxes = results[0].boxes
        count = len(boxes)
         # Put text of count on top-left corner
        text = f"Objects detected: {count}"
        cv2.putText(annotated_frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('YOLOv8 Webcam', annotated_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
