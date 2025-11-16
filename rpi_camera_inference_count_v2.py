import cv2
from picamera2 import Picamera2
import numpy as np
# use tflite-runtime 
import tflite_runtime.interpreter as tflite
import time

# --- Configure environment ---
# Path to the INT8 TFLite model file
TFLITE_MODEL_PATH = '/home/dokeunoh/Desktop/project/shahed-YOLO-model-train/best_tmp_saved_model/best_tmp_int8.tflite'
# Confidence threshold for object detection
CONFIDENCE_THRESHOLD = 0.35
# -----------------

# --- 💡 TFLite Model Post-Processing Function (Essential Implementation Required) ---
# This function interprets the TFLite model's output tensor into YOLO format (boxes, scores, classes).
# It must be implemented according to the specific output structure of your YOLOv8 TFLite model.
# (Reference: e.g., https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tflite_utils.py)
def post_process_yolo_tflite(output_data, frame_w, frame_h, conf_thres):
    """
    Converts the raw output data from the TFLite model into bounding boxes, scores, and class IDs.
    """
    # 🚨 Actual YOLOv8 TFLite post-processing logic must be implemented here. 
    # Object detection results will not be correct without proper implementation.

    # Temporary return values (Must be replaced upon implementation)
    boxes = np.array([[100, 100, 200, 200]]) # xmin, ymin, xmax, ymax
    scores = np.array([0.95])
    class_ids = np.array([0])
    
    return boxes, scores, class_ids

# --- Main function ---
def main():
    # 1. Picamera2 Setup
    picam2 = Picamera2()
    # Configure capture resolution (640x480)
    config = picam2.create_video_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    # 2. Load and Initialize TFLite Model
    try:
        interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"TFLite model loading or tensor allocation failed: {e}")
        return

    # Get model input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Standard input shape for YOLOv8 TFLite model (usually [1, 640, 640, 3] or [1, 416, 416, 3])
    input_shape = input_details[0]['shape'] 
    
    # Captured frame dimensions
    frame_w, frame_h = 640, 480
    
    last_time = time.time()

    while True:
        # 3. Frame Capture and Preprocessing
        frame = picam2.capture_array()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        annotated_frame = frame_rgb.copy() # Copy for visualization

        # A. Resizing: 640x480 -> TFLite input size (e.g., 640x640)
        input_w, input_h = input_shape[1], input_shape[2]
        resized_frame = cv2.resize(frame_rgb, (input_w, input_h)) 

        # B. Data Type and Shape Conversion: (W, H, 3) -> (1, W, H, 3) [UINT8]
        # Quantized models typically require np.uint8.
        input_tensor = np.expand_dims(resized_frame, axis=0) 
        input_tensor = input_tensor.astype(input_details[0]['dtype']) 
        
        # 4. Execute TFLite Inference
        start_inference_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        
        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        end_inference_time = time.time()
        
        # 5. Post-processing and Visualization
        # 💡 Interpret TFLite Output (The most critical part)
        boxes, scores, class_ids = post_process_yolo_tflite(
            output_data, frame_w, frame_h, CONFIDENCE_THRESHOLD
        )
        
        count = len(boxes)
        
        # 6. Visualization (Example logic)
        for box, score, class_id in zip(boxes, scores, class_ids):
            if score >= CONFIDENCE_THRESHOLD:
                xmin, ymin, xmax, ymax = box.astype(int)
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                
                # Display label and score
                label = f"ID:{class_id} {score:.2f}"
                cv2.putText(annotated_frame, label, (xmin, ymin - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 7. Print Performance and Information
        current_time = time.time()
        interval = (current_time - last_time) * 1000 # ms (Frame processing time)
        inference_ms = (end_inference_time - start_inference_time) * 1000 # ms (Model inference time)
        last_time = current_time
        
        text_count = f"Objects: {count}"
        text_perf = f"Interval: {interval:.1f}ms / Inference: {inference_ms:.1f}ms"
        
        cv2.putText(annotated_frame, text_count, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, text_perf, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('YOLOv8 TFLite RPi Camera', annotated_frame)
        print(text_perf)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()