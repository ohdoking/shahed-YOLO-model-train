import cv2
from picamera2 import Picamera2
import zmq
import time

def main():
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5555")

    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    try:
        while True:
            frame = picam2.capture_array()
            _, jpeg = cv2.imencode('.jpg', frame)
            socket.send(jpeg.tobytes())
            time.sleep(1)  # ~30 FPS
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()

if __name__ == "__main__":
    main()
