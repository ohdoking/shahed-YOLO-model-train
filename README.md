# Shahed YOLO Model Train

- YOLO Model Training and Quantization for Raspberry Pi (RPI)

This process involves two main phases: training your object detection model using YOLOv8 (PyTorch) and then quantizing and exporting it to the TFLite format so it can run efficiently on the Raspberry Pi's CPU.

## Key Steps

1. **Data Labeling**  
   Labeled image data using the Roboflow platform and downloaded the dataset formatted for YOLOv8.

2. **data.yaml Configuration**  
   Created and configured the `data.yaml` file containing paths to training/validation data and class information, tailored for YOLOv8 training.

3. **YOLOv8 Pretrained Model Fine-tuning**  
   Fine-tuned a pretrained YOLOv8 model using the labeled data from Roboflow.

4. **INT8 Quantization**  
   Performed INT8 quantization to reduce model size and improve inference speed.

## How to work

### Phase 1: Training the YOLOv8 Model

#### 1.1 Data Preparation(Label Training Image)
#### 1.2 Model Training(YOLOv8n)

### Phase 2:Quantization and Export to TFLite

#### 2.1 Export to ONNX (Intermediate Step)
#### 2.2 SavedModel Conversion (TensorFlow Format)
#### 2.3 TFLite Conversion and INT8 Quantization

### Phase 3: Deployment on Raspberry Pi

#### 3.1 Installation and Inference

## Project structure

```
project/
 ├── train/
 │    └── images/
 │    └── labels/
 ├── valid/
 │    └── images/
 │    └── labels/
 ├── test/
 │    └── images/
 │    └── labels/
 └── data.yaml
 ```



## How to install

```
python3 -m venv venv
source venv/bin/activate
```

```
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install ultralytics opencv-python
pip install picamera2 numpy
pip install onnx2tf==1.7.9 tensorflow tf_keras onnx-graphsurgeon sng4onnx

pip install tflite-runtime
```

## How to run 
### with code

```
python train_shahed.py
```

### command line

```
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640 batch=16 device=mps
```

### convert to tflite

```
deactivate

python3.11 -m venv venv_311_final
source venv_311_final/bin/activate


pip install torch tensorflow==2.15.0 ultralytics
pip install tf_keras==2.15.0 onnx2tf onnx-graphsurgeon
yolo export model=best_tmp.pt format=tflite int8 data=data.yaml

```


## Reference
- https://universe.roboflow.com/uav-taebs/shahed136-detect-thn08/dataset/2
- https://universe.roboflow.com/shahed136/shahed136-detect/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true 
- https://universe.roboflow.com/bbokyeong/shahed-136-chbsm/dataset/1
- https://www.kaggle.com/datasets/banderastepan/drone-detection
- https://universe.roboflow.com/aircraft-ol8hx/shahed-136-ohjzr
- https://docs.ultralytics.com/modes/train/#arguments
