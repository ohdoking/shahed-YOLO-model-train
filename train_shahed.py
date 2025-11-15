from ultralytics import YOLO

def train():
    # Load a small pre-trained YOLOv8 model
    model = YOLO('/Users/dokeunoh/DevOh/project/private/project/hackerton/edth-2025/shahed-model-train/train_shahed_yolov8n.pt')

    # Train the model
    model.train(
        data='data.yaml',  # path to your data.yaml
        epochs=50,         # number of training epochs
        imgsz=640,         # image size
        batch= 8,
        device='mps',       # use Mac M1 GPU acceleration (change to 'cpu' if issues)
        
        # --- Optimizer & Learning Rate ---
        optimizer='Adam',        # Optimizer: Use Adam (fast convergence, robust)
        lr0=0.0005,               # Initial Learning Rate (Lower for Adam)
        lrf=0.01,                # Final Learning Rate (1% of lr0)
        # momentum=0.937,          # Momentum (Standard)

        # --- Regularization & Stopping ---
        weight_decay=0.0005,     # Regularization: L2 weight decay
        patience=75,             # Early Stopping: Stop if no improvement after 50 epochs
        
        # --- Data Augmentation ---
        # flipud=0.5,              # Augmentation: Flip up/down 50% of the time
        # hsv_h=0.015              # Augmentation: Hue adjustment strength
        # --- Data Augmentation Adjustment ---
        # mosaic= 0.5,           # Reduced mosaic (from 1.0) to prevent excessive distortion of very small targets.
        # degrees= 5.0,          # Small rotation range (drones should remain generally upright).
        # scale= 0.7,            # Higher scaling factor for robust detection at various distances.
        # hyp="hyp.yaml",

        augment=True,
        auto_augment="randaugment",
        degrees=10,
        scale=0.5,
        translate=0.1,
        fliplr=0.5,
        mosaic=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # --- Other Settings ---
        name= 'Shahed136_fine-tunned_model-extra-data'
    )

if __name__ == "__main__":
    train()