from ultralytics import YOLO

def train():
    # Load a small pre-trained YOLOv8 model
    model = YOLO('/Users/dokeunoh/DevOh/project/private/project/hackerton/edth-2025/shahed-model-train/train_shahed_yolov8n.pt')

    # Train the model
    model.train(
        data='data.yaml',  # path to your data.yaml
        epochs=50,         # number of training epochs
        imgsz=640,         # image size
        device='mps'       # use Mac M1 GPU acceleration (change to 'cpu' if issues)
    )

if __name__ == "__main__":
    train()