import cv2
from ultralytics import YOLO

def main():
    model = YOLO('/home/dokeunoh/Desktop/project/shahed-YOLO-model-train/best_tmp.pt')  # path to your trained model

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        results = model(frame)

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

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
