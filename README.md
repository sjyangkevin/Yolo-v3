# YoloV3: You Only Look Once - TensorFlow 2  

# Table of Content

1. <strong> Pre-trained Model </strong>
2. <strong> Run Prediction on Images and Videos </strong>  
3. <strong> Run Training on Custom Datasets </strong>
4. <strong> Reference </strong>
5. <strong> Demo </strong>

# Pre-trained Model

| Datasets | Files | Input Size |
| :--- | :----: | ---: |
| COCO 2017 | [weights.h5](https://github.com/bubbliiiing/yolo3-tf2/releases/download/v1.0/yolo_weights.h5) | 416 x 416 |

# Run Prediction on Images and Videos
1. Download an unzip the weight file, and put it under the path `src/data/yolov3/`, and modify `src/config.py` as necessary
2. In `predict.py`, modify the variable `mode` to either `"predict"` or `"video"` for running prediction on images and videos respectively.
If it is running on video, then the `video_path`, `video_save_path`, and `video_fps` need to be set.
3. Open command line or terminal, and go to the `/src` directory, and type in and run `python predict.py`

# Run Training on Custom Datasets
1. The data loader of the model requires the data to be processed first into the Pascal VOC input format. Specifically, the inputs should
be two files: `train.txt` and `valid.txt`, each file should satisfy the following format for each record:   
`path_to_image, x_min, y_min, x_max, y_max, class_id, x_min, y_min, x_max, y_max, class_id, ...`

2. After the `train.txt` and `valid.txt` generated, go to the `src/config.py` to modify `class_path`, `train_annotation_path`, and `valid_annotation_path`.
 You need to generate a <strong>class file</strong> that each row is a `class_name` that matches to a `class_id`. Here is
 a sample of the class file: `src/data/voc/voc_classes.txt`.

3. Then, open command line or terminal, go to `/src` and run `python train.py` to start training.

4. If you would like to modify the hyperparameters or the train setting, open the `/src/train.py` and modify the file.

# Reference

YOLOV3：You Only Look Once目标检测模型在Tensorflow2当中的实现: [bubbliiiing/yolo3-tf2](https://github.com/bubbliiiing/yolo3-tf2)  
Yolo v3 Object Detection in Tensorflow: [HEARTKILLA](https://www.kaggle.com/aruchomu/yolo-v3-object-detection-in-tensorflow)  
UAVDT (Unmanned Aerial Vehicle Benchmark Object Detection and Tracking): [UAVDT](https://paperswithcode.com/dataset/uavdt)

# Demo

* Experiment with Highway Traffic Car (Type) Detection with the pre-trained weight from COCO17  
<img src="https://github.com/sjyangkevin/Yolo-v3/blob/main/src/data/images/highway_traffic.png" alt="Highway Traffic Car Type" style="width:800px;height:450px;" />

* Experiment with Car (Type) Detection during night-time with the pre-trained weight from COCO17 (on CPU)  
<img src="https://github.com/sjyangkevin/Yolo-v3/blob/main/src/data/images/night_time_m1101.gif" alt="Night-time Car Detection" style="width:800px;height:450px;" />
