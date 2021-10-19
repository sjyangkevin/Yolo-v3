import os
import time
import colorsys

import numpy as np
import tensorflow as tf
from PIL import ImageDraw, ImageFont
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

from model.yolov3 import yolo_body
from utils.utils import convert_color, get_anchors, get_classes, preprocess_input, resize_image, decode_box
import config as cfg

class YOLO(object):
    _defaults = {
        "weights_path": cfg.weights_path,
        "classes_path": cfg.classes_path,
        "anchors_path": cfg.anchors_path,
        "anchors_mask": cfg.anchors_mask,
        "input_shape" : cfg.input_shape,
        "confidence"  : cfg.confidence_score,
        "nms_iou"     : cfg.nms_iou,
        "max_boxes"   : cfg.max_boxes,
        "is_padded"   : cfg.is_padded
    }

    @classmethod
    def get_defaults(cls, attr):
        if attr in cls._defaults:
            return cls._defaults[attr]
        else:
            return "unrecognized attribute name '" + attr + "'"
    
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors     = get_anchors(self.anchors_path)

        hsv_tuples  = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.create_model()
    
    def create_model(self):
        weights_path = os.path.expanduser(self.weights_path)
        assert weights_path.endswith('.h5'), 'keras model or weights must be a .h5 file.'

        self.yolo_model = yolo_body(
            (self.input_shape[0], self.input_shape[1], cfg.num_channels), 
            self.num_classes, 
            len(self.anchors_mask), 
            training    = False, 
            data_format = cfg.data_format
        )

        self.yolo_model.load_weights(self.weights_path)
        print('{} model, anchors, and classes loaded.'.format(weights_path))

        self.input_image_shape = Input([2, ], batch_size=1)
        inputs  = [*self.yolo_model.output, self.input_image_shape]
        outputs = Lambda(
            decode_box,
            output_shape = (1, ),
            name         = 'yolo_eval',
            arguments = {
                'anchors'       : self.anchors,
                'num_classes'   : self.num_classes,
                'input_shape'   : self.input_shape,
                'anchors_mask'  : self.anchors_mask,
                'confidence'    : self.confidence,
                'nms_iou'       : self.nms_iou,
                'max_boxes'     : self.max_boxes,
                'is_padded'     : self.is_padded
            }
        )(inputs)

        self.yolo_model = Model([self.yolo_model.input, self.input_image_shape], outputs)
    
    @tf.function
    def get_pred(self, image_data, input_image_shape):
        out_boxes, out_scores, out_classes = self.yolo_model([image_data, input_image_shape])
        return out_boxes, out_scores, out_classes
    
    def detect_image(self, image):
        image             = convert_color(image)
        image_data        = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.is_padded)
        image_data        = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)

        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape) 
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font      = ImageFont.truetype(font=cfg.font_path, size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[int(c)]
            box             = out_boxes[i]
            score           = out_scores[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        
        return image

    def get_FPS(self, image, test_interval):
        image      = convert_color(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.is_padded)
        image_data = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape) 

        t1 = time.time()
        for _ in range(test_interval):
            out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape) 
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time




# if __name__ == "__main__":
#     from PIL import Image

#     yolo = YOLO()

#     while True:
#         img = input('Input image file')
#         try:
#             image = Image.open(img)
#         except:
#             print('open error, try again')
#             continue
#         else:
#             r_image = yolo.detect_image(image)
#             r_image.show()