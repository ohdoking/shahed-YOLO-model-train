import cv2
import numpy as np
import zmq
import tflite_runtime.interpreter as tflite
import time

TFLITE_MODEL_PATH = '/home/dokeunoh/Desktop/project/shahed-YOLO-model-train/best_tmp_saved_model/best_tmp_int8.tflite'
CONFIDENCE_THRESHOLD = 0.35

def post_process_yolo_tflite(output_data, frame_w, frame_h, conf_thres):
    # Replace with your actual post-processing
    boxes = np.array([[100, 100, 200, 200]])
    scores = np.array([0.95])
    class_ids = np.array([0])
    return boxes, scores, class_ids

def main():
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5555")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    frame_w, frame_h = 640, 480
    last_time = time.time()

    while True:
        jpeg_bytes = socket.recv()
        npimg = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame_rgb = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Resize frame for model input
        input_w, input_h = input_shape[1], input_shape[2]
        resized_frame = cv2.resize(frame_rgb, (input_w, input_h))

        input_tensor = np.expand_dims(resized_frame, axis=0).astype(input_details[0]['dtype'])

        start_inference_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        end_inference_time = time.time()

        boxes, scores, class_ids = post_process_yolo_tflite(output_data, frame_w, frame_h, CONFIDENCE_THRESHOLD)
        count = len(boxes)

        annotated_frame = frame_rgb.copy()
        for box, score, class_id in zip(boxes, scores, class_ids):
            if score >= CONFIDENCE_THRESHOLD:
                xmin, ymin, xmax, ymax = box.astype(int)
                cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                label = f"ID:{class_id} {score:.2f}"
                cv2.putText(annotated_frame, label, (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        current_time = time.time()
        interval = (current_time - last_time) * 1000
        inference_ms = (end_inference_time - start_inference_time) * 1000
        last_time = current_time

        text_count = f"Objects: {count}"
        text_perf = f"Interval: {interval:.1f}ms / Inference: {inference_ms:.1f}ms"

        cv2.putText(annotated_frame, text_count, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, text_perf, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('YOLOv8 TFLite RPi Camera', annotated_frame)
        print(text_perf)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
