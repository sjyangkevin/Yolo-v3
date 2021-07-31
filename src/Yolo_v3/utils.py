import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from seaborn import color_palette
from configparser import ConfigParser
import tensorflow as tf

parser = ConfigParser()
parser.read('config.ini')
FONT_DIR = parser.get('yolo', 'font_dir')
OUTPUT_DIR = parser.get('yolo', 'output_dir')

def load_anchors(file_name):
    with open(file_name, 'r') as f:
        anchors = f.readline().split(' ')
    results = []
    for anchor in anchors:
        anchor = anchor.split(',')
        results.append((int(anchor[0]), int(anchor[1])))
    return results

def load_single_frame(frame, model_size):
    img = Image.fromarray(frame)
    img = img.resize(size=model_size)
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)
    return img

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

def load_class_names(file_name):
    """return a list of class names read from 'file_name'"""
    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

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

def draw_boxes(img_names, boxes_dicts, class_names, model_size, save_output=False):
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
        if save_output:
            img.save(os.path.join(OUTPUT_DIR, os.path.basename(img_name)))
    return img

def draw_boxes_on_frame(frame, boxes_dicts, class_names, model_size):
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
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font=FONT_DIR, size=(img.size[0] + img.size[1]) // 100)
    boxes_dict = boxes_dicts[0]

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
    return img