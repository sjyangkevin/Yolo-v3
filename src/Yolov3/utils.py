import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from seaborn import color_palette
import tensorflow as tf
from Yolov3.config import *

def load_anchors(file_name):
    with open(file_name, 'r') as f:
        anchors = f.readline().split(' ')
    results = []
    for anchor in anchors:
        anchor = anchor.split(',')
        results.append((int(anchor[0]), int(anchor[1])))
    return results

def load_class_names(file_name):
    """return a list of class names read from 'file_name'"""
    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

def load_images(img_names, model_size):
    """
    load images in a 4D array
    
    args:
        img_names: a list of image names
        model_size: the input size of the model
        data_format: a format for the array returned
        
    return:
        a 4D NumPy array
    """
    imgs = []
    
    for img_name in img_names:
        img = Image.open(img_name)
        img = img.resize(size=model_size)
        img = np.array(img, dtype=np.float32)
        img = np.expand_dims(img, axis=0)
        imgs.append(img)
        
    imgs = np.concatenate(imgs)
    
    return imgs

def box_iou(b1, b2):
    '''Return iou tensor
    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh
    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)
    '''

    # Expand dim to apply broadcasting.
    b1 = tf.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = tf.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = tf.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou

def box_giou(boxes1, boxes2):
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    # Calculate the iou value between the two bounding boxes
    iou = inter_area / union_area

    # Calculate the coordinates of the upper left corner and the lower right corner of the smallest closed convex surface
    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

    # Calculate the area of the smallest closed convex surface C
    enclose_area = enclose[..., 0] * enclose[..., 1]

    # Calculate the GIoU value according to the GioU formula  
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou

def build_boxes(inputs):
    """computes top left and bottom right points of the boxes."""
    center_x, center_y, width, height, confidence, classes = \
    tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)
    
    top_left_x = center_x - width / 2
    top_left_y = center_y - height / 2
    bottom_right_x = center_x + width / 2
    bottom_right_y = center_y + height / 2
    
    boxes = tf.concat([
        top_left_x,
        top_left_y,
        bottom_right_x,
        bottom_right_y,
        confidence,
        classes
    ], axis=-1)
    
    return boxes

def non_max_suppression(inputs, n_classes, max_output_size, iou_threshold, confidence_threshold):
    """
    perform non-max suppression separately for each class
    
    args:
        inputs: tensor input.
        n_classes: number of classes
        max_output_size: max number of boxes to be selected for each class
        iou_threshold: threshold for the IOU
        confidence_threshold: threshold for the confidence score
        
    returns:
        a list containing class-to-boxes dictionaries
        for each sample in the batch
    """
    # unpack along the batch_size dimension -> batch = num_examples
    # each batch is of shape (total_num_pred_box, 5 + num_classes)
    batch = tf.unstack(inputs)
    boxes_dicts = []
    for boxes in batch:
        # filter out the boxes that has confidence less than the confidence_threshold
        boxes = tf.boolean_mask(boxes, boxes[:, 4] > confidence_threshold)
        # for classes, get the argmax -> return a single column of class index
        classes = tf.argmax(boxes[:, 5:], axis=-1)
        classes = tf.expand_dims(tf.cast(classes, dtype=tf.float32), axis=-1)
        boxes = tf.concat([boxes[:, :5], classes], axis=-1)
        
        boxes_dict = dict()
        for cls in range(n_classes):
            # get the boxes of a specific class - returns a boolean tensor
            # with the one equal to cls as True
            mask = tf.equal(boxes[:, 5], cls)
            mask_shape = mask.get_shape()
            if mask_shape.ndims != 0:
                # get the boxes of the current class
                class_boxes = tf.boolean_mask(boxes, mask)
                boxes_coords, boxes_conf_scores, _ = tf.split(class_boxes, [4, 1, -1], axis=-1)
                boxes_conf_scores = tf.reshape(boxes_conf_scores, [-1])
                indices = tf.image.non_max_suppression(boxes_coords, boxes_conf_scores, max_output_size, iou_threshold)
                class_boxes = tf.gather(class_boxes, indices)
                boxes_dict[cls] = class_boxes[:, :5]
        
        boxes_dicts.append(boxes_dict)
    
    return boxes_dicts

def draw_boxes(img_names, boxes_dicts, class_names, model_size):
    """
    Draws detected boxes
    
    args:
        img_names: a list of input images names
        boxes_dict: a class-to-boxes dictionary
        class_names: a class names list
        model_size: input size of model
    """
    # coco dataset has 80 classes -> a color for each class
    colors = ((np.array(color_palette("hls", 80)) * 255)).astype(np.uint8)
    for num, img_name, boxes_dict in zip(range(len(img_names)), img_names, boxes_dicts):
        img = Image.open(img_name)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font=FONT_DIR, size=(img.size[0] + img.size[1]) // 100)
        
        resize_factor = \
        (img.size[0] / model_size[0], img.size[1] / model_size[1])
        
        for cls in range(len(class_names)):
            boxes = boxes_dict[cls]
            if np.size(boxes) != 0:
                color = colors[cls]
                for box in boxes:
                    xy, confidence = box[:4], box[4]
                    xy = [xy[i] * resize_factor[i % 2] for i in range(4)]
                    x0, y0 = xy[0], xy[1]
                    thickness = (img.size[0] + img.size[1]) // 200
                    for t in np.linspace(0, 1, thickness):
                        xy[0], xy[1] = xy[0] + t, xy[1] + t
                        xy[2], xy[3] = xy[2] - t, xy[3] - t
                        draw.rectangle(xy, outline=tuple(color))
                    text = '{} {:.1f}%'.format(class_names[cls], confidence * 100)
                    text_size = draw.textsize(text, font=font)
                    draw.rectangle(
                        [x0, y0 - text_size[1], x0 + text_size[0], y0],
                        fill=tuple(color)
                    )
                    draw.text((x0, y0 - text_size[1]), text, fill='black', font=font)
        img.save(os.path.join(OUTPUT_DIR, os.path.basename(img_name)))