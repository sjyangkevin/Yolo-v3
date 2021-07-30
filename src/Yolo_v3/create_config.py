from configparser import ConfigParser

config = ConfigParser()

config['model'] = {
    'batch_norm_decay': 0.9,
    'batch_norm_epsilon': 1e-05,
    'leaky_relu_alpha': 0.1,
    'model_size': 416,
    'n_classes': 80,
    'training': False,
    'data_format': 'channels_last',
    'weight_path': '../../../yolov3_weights.h5' 
}

config['yolo'] = {
    'anchor_dir': '../../data/Yolo_v3/anchors.txt',
    'class_name_dir': '../../data/coco.names',
    'font_dir': '../../data/futur.ttf',
    'output_dir': '../../data/Yolo_v3/outputs',
    'max_output_size': 25,
    'iou_threshold': 0.5,
    'confidence_threshold': 0.5
}

with open('./config.ini', 'w') as f:
    config.write(f)