# Shahed YOLO Model Train

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


## Reference
- https://universe.roboflow.com/uav-taebs/shahed136-detect-thn08/dataset/2
- https://universe.roboflow.com/shahed136/shahed136-detect/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true# 
