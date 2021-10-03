from Yolov3.yolo import create_model, compute_loss, decode
from Yolov3.config import *
from Yolov3.utils import *
from Yolov3.dataset import Dataset
import numpy as np
import tensorflow as tf
import os

if __name__ == "__main__":

    n_classes = len(load_class_names(CLASS_NAME_FILE))
    anchors = load_anchors(ANCHOR_FILE)
    n_anchors = len(anchors) // 3
    input_size = MODEL_INPUT_SIZE
    training = True
    data_format = DATA_FORMAT

    model = create_model(n_classes, n_anchors, input_size, training, data_format)
    model.load_weights(MODEL_WEIGHTS_DIR)

    trainset = Dataset('train')

    print(len(trainset))

    images, labels = iter(next(trainset))

    print(images.shape)
    for label in labels:
        for l in label:
            print(l.shape)

    conv_tensors = model(images, training=True)

    for route in conv_tensors:
        print(route.shape)
    
    giou_loss = 0
    conf_loss = 0
    prob_loss = 0

    route_lbox, route_mbox, route_sbox = conv_tensors
    pred_lbox = decode(route_lbox, n_classes, anchors[6:9], (416, 416), 'channels_last', True)
    pred_mbox = decode(route_mbox, n_classes, anchors[3:6], (416, 416), 'channels_last', True)
    pred_sbox = decode(route_sbox, n_classes, anchors[0:3], (416, 416), 'channels_last', True)
    
    pred_tensors = [pred_lbox, pred_mbox, pred_sbox]

    grid = 3
    for i in range(grid):
        loss_items = compute_loss(pred_tensors[i], conv_tensors[i], *labels[i])
        giou_loss += loss_items[0]
        print("giou loss: ", giou_loss)
        conf_loss += loss_items[1]
        print("conf loss: ", conf_loss)
        prob_loss += loss_items[2]
        print("prob loss:", prob_loss)
    
    total_loss = giou_loss + conf_loss + prob_loss

    print(total_loss)
