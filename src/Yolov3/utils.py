
#================================================================
#
#   File name         : utils.py
#   Author            : Shijin Yang
#   Created date      : 2021-08-11
#   Website           : https://github.com/sjyangkevin/ODLib
#   Modified based on : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Original Author   : PyLessons
#
#================================================================

from multiprocessing import Process, Queue, Pipe
import cv2
import time
import random
import colorsys
import numpy as np
import tensorflow as tf
from Yolov3.config import *
from tensorflow.python.saved_model import tag_constants

def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def image_preprocess(image, target_size, gt_boxes=None):
    ih, iw    = target_size
    h,  w, _  = image.shape

    # if not the min one is chosen, then one dimension could be larger
    # than the model input size
    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    # return a new array of given shape (ih, iw) - model input size
    # filled with fill_value
    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    # normalize the image
    image_paded = image_paded / 255.

    if gt_boxes is None:
        # no ground truth box, for prediction - just the processed image
        return image_paded

    else:
        # when training, scale the ground truth box to image location
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes