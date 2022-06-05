# fruit counting
This is the official implementation of the paper "Cascade-SORT: A Robust Fruit Counting Approach Using Multiple Features Cascade Matching".

At present, we only support YOLO in Darknet format. More functions will be added in future versions, to be continued...

## Dependencies
- python 3.8
- numpy 1.18.5
- scipy 1.5.0
- opencv-python 4.4.0.44
- opencv-contrib-python 4.4.0.44
- scikit learn 0.23.1

## Quick Start
0. Check all dependencies installed

1. Clone this repository
```
git clone git@github.com:ZQPei/deep_sort_pytorch.git
```

2. Set configuration, revise "yaml/apple.yaml", if the code is run on the custom videos and models
```python
YOLO:
  CFG: "cfg/apple.cfg"
  WEIGHT: "checkpoints/apple_best.weights"
  CLASS_NAMES: "cfg/apple.names"
  SCORE_THRESH: 0.5
  NMS_THRESH: 0.5

TRACK:
  MODE: "cascade"
  VIDEO_DIR: "video/apple.mp4"
  SAVE_DIR: "results/apple.mp4"
```

3. Run demo
```
python main.py
```