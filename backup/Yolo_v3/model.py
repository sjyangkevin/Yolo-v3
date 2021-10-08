import os
from six import exec_
import tensorflow as tf
from configparser import ConfigParser
import argparse
import cv2
import numpy as np
from timeit import default_timer as timer
from yolo import yolo_layer, yolo_predict
from utils import load_anchors, load_class_names, load_images, draw_boxes
# from utils import draw_boxes_on_frame, load_single_frame

parser = ConfigParser()
parser.read('config.ini')

argparser = argparse.ArgumentParser(description='Yolo v3 Object Detection')
argparser.add_argument('--input_dir', type=str, help='Input Image')
# argparser.add_argument('--save_output', type=bool, help='whether to save output')

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
        self.model = self.create_model()

    # def predict_video(self, video_path):
    #     vid = cv2.VideoCapture(video_path)
    #     if not vid.isOpened():
    #         raise IOError("Couldn't open webcam or video")
        
    #     video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    #     video_fps = vid.get(cv2.CAP_PROP_FPS)
    #     video_size = (
    #         int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #         int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     )
        
    #     save_output = True if self.output_dir != "" else False

    #     if save_output:
    #         print("TYPE:", type(self.output_dir), type(video_FourCC), type(video_fps), type(video_size))
    #         out = cv2.VideoWriter(self.output_dir, video_FourCC, video_fps, video_size)

    #     accum_time = 0
    #     curr_fps = 0
    #     fps = "FPS: ??"
    #     prev_time = timer()

    #     while True:
    #         _, frame = vid.read()
    #         inp = load_single_frame(
    #             frame, 
    #             model_size=(self.model_size, self.model_size)
    #         )
    #         _inp = inp / 255.
    #         _inp = tf.constant(_inp)
    #         preds = self.model(_inp)
    #         boxes_dicts = yolo_predict(
    #             preds, 
    #             self.n_classes, 
    #             self.anchors, 
    #             (self.model_size, self.model_size),
    #             self.max_output_size,
    #             self.iou_threshold,
    #             self.confidence_threshold,
    #             self.data_format
    #         )
    #         image = draw_boxes_on_frame(
    #         frame, 
    #         boxes_dicts, 
    #         self.class_names, 
    #         (self.model_size, self.model_size)
    #         )
    #         result = np.asarray(image)
    #         curr_time = timer()
    #         exec_time = curr_time - prev_time
    #         prev_time = curr_time
    #         accum_time += exec_time
    #         curr_fps = curr_fps + 1
    #         if accum_time > 1:
    #             accum_time = accum_time - 1
    #             fps = "FPS: " + str(curr_fps)
    #             curr_fps = 0
    #         cv2.namedWindow("detect from video", cv2.WINDOW_NORMAL)
    #         cv2.imshow("result", result)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #         out.write(result)

    
    def create_model(self):
        model = self.yolo_model(
            self.n_classes, 
            self.n_anchors, 
            (self.model_size, self.model_size), 
            self.training, 
            self.data_format
        )
        model.load_weights(self.weight_path)
        return model

    def predict(self, inputs):

        images = load_images(
            inputs, 
            model_size=(self.model_size, self.model_size)
        )

        images /= 255.
        
        preds = self.model(images)
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

        _ = draw_boxes(
            inputs, 
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

    # inputs = args.input_dir
    inputs = os.listdir(args.input_dir)
    inputs = [os.path.join(args.input_dir, img_dir) for img_dir in inputs]
    Yolov3.predict(inputs)
    # Yolov3.predict_video(inputs)

