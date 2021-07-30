import tensorflow as tf
from configparser import ConfigParser
import argparse
from yolo import yolo_layer, yolo_predict
from utils import load_anchors, load_class_names, load_images, draw_boxes

parser = ConfigParser()
parser.read('config.ini')

argparser = argparse.ArgumentParser(description='Yolo v3 Object Detection')
argparser.add_argument('--input', type=str, help='Input Image')

class Yolo_v3:
    def __init__(self):
        self.n_classes = parser.getint('model', 'n_classes')
        self.anchors = load_anchors(parser.get('yolo', 'anchor_dir'))
        self.n_anchors = len(self.anchors) // 3
        self.max_output_size = parser.getint('yolo', 'max_output_size')
        self.iou_threshold = parser.getfloat('yolo', 'iou_threshold')
        self.confidence_threshold = parser.getfloat('yolo', 'confidence_threshold')
        self.class_names = load_class_names(parser.get('yolo', 'class_name_dir'))
        self.output_dir = parser.get('yolo', 'output_dir')
        self.model_size = parser.getint('model', 'model_size')
        self.training = parser.getboolean('model', 'training')
        self.data_format = parser.get('model', 'data_format')
        self.weight_path = parser.get('model', 'weight_path')

    def predict(self, inputs):
        
        model = self.yolo_model(
            self.n_classes, 
            self.n_anchors, 
            (self.model_size, self.model_size), 
            self.training, 
            self.data_format
        )
        model.load_weights(self.weight_path)

        images = load_images(
            [inputs], 
            model_size=(self.model_size, self.model_size)
        )

        images /= 255.

        images = tf.constant(images)
        
        preds = model(images)
        preds = yolo_predict(
            preds, 
            self.n_classes, 
            self.anchors, 
            (self.model_size, self.model_size),
            self.max_output_size,
            self.iou_threshold,
            self.confidence_threshold,
            self.data_format
        )

        draw_boxes(
            [inputs], 
            preds, 
            self.class_names, 
            (self.model_size, self.model_size)
        )

    def yolo_model(self, n_classes, n_anchors, model_size, training, data_format):
        inputs = tf.keras.layers.Input(shape=(model_size[0], model_size[1], 3))
        
        f1, f2, f3 = yolo_layer(inputs, n_classes, n_anchors, training, data_format)
        
        return tf.keras.models.Model(inputs=inputs, outputs=[f1, f2, f3])

if __name__ == '__main__':
    Yolov3 = Yolo_v3()
    args = argparser.parse_args()
    input_image = args.input

    Yolov3.predict(input_image)

