import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from numpy.core.arrayprint import dtype_short_repr
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

# def load_single_frame(frame, model_size):
#     img = Image.fromarray(frame)
#     img = img.resize(size=model_size)
#     img = np.array(img, dtype=np.float32)
#     img = np.expand_dims(img, axis=0)
#     return img

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

def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    """
    Preprocess true boxes to training input format

    true_boxes: array, shape  (m, T, 5): x_min, y_min, x_max, y_max, class_id relateive to input_shape
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N,2), wh
    num_classes: integer

    return:
        y_true: list of array, like output from yolo_layer, xywh are relative value
    """
    
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    
    # we use 3 anchors for each featuer map
    num_layers = len(anchors) // 3
    # index of anchor for each feature map
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    # get the center coordinate of the true box
    # box are in the format of x_min, y_min, x_max, y_max
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    # get the true box's width and height
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    # convert the box coordinate to be relative to the input size
    # and we need to reverse it, since it is hw rather than wh
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    # get the batch size
    m = true_boxes.shape[0]
    # 32, 16, 8 are the scaling factor of the grid respectively, (13, 13), (26, 26), (52, 52)
    # for input size of 416
    grid_shapes = [input_shape // {0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    # create a list of zero array, each array has the following shape
    # (13, 13)
    # (26, 26)
    # (52, 52)
    # since we use 3 anchors for each feature map, so the fourth dimension is 3
    # (m, 13, 13, 3, 85)
    # (m, 26, 26, 3, 85)
    # (m, 52, 52, 3, 85)
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes), dtype='float32') for l in range(num_layers)]

    # expand dim to apply broadcasting
    # anchors shape from (N, 2) become to (1, N, 2) where N = n_anchors
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # discard zero rows
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        # shape = (n_boxes, 2)
        # then shape = (n_boxes, 1, 2)
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        # for each box in the image, calculate its intersection
        # with all the anchors provided
        # (n_boxes, n_anchors, 2)
        intersect_mins = np.maximum(box_mins, anchor_mins)
        # (n_boxes, n_anchors, 2)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        # (n_boxes, n_anchors)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        # (n_boxes, 1)
        box_area = wh[..., 0] * wh[..., 1]
        # (1, n_anchors)
        anchor_area = anchors[..., 0] * anchors[..., 1]
        # (n_boxes, n_anchors)
        # represent the iou of each box with all provided anchors
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # find best anchor for each true box
        # shape = (n_boxes, ), the index of max overlap anchor with the box
        # each element is the index of the anchor has largest iou with the box
        best_anchor = np.argmax(iou, axis=-1)

        # t represent the box, n is the index of the anchor has largest iou with t
        for t,n in enumerate(best_anchor):
            # (t, n) -> (box_idx, anchor_idx)
            # l means the index of the feature maps
            for l in range(num_layers):
                # if the anchor is used in current feature map, do the following
                if n in anchor_mask[l]:
                    # n is the index of anchor in anchors
                    # i is the width of the grid  - maps box to grid
                    # j is the height of the grid - maps box to grid
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    # find the index of the anchor in anchor_mask, for a feature map [l]
                    # anchor_mask[0] = [6, 7, 8], if n = 6, then k = 0
                    k = anchor_mask[l].index(n)
                    # set the c to the class_id
                    c = true_boxes[b,t, 4].astype('int32')
                    # for feature map - l, current image - b, grid_h, grid_w, anchor, box_cord, confidence = 1, class_id = 1
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1
        
        # y_true[l] -> for feature map [l]
        # y_true[l][b, j, i, k] for image b, grid location (j, i), and anchor k
        # 5 + c = (true_box_xmin, true_box_ymin, true_box_xmax, true_box_ymax, conf, class)
        return y_true

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

# def draw_boxes_on_frame(frame, boxes_dicts, class_names, model_size):
#     """
#     Draws detected boxes
    
#     args:
#         img_names: a list of input images names
#         boxes_dict: a class-to-boxes dictionary
#         class_names: a class names list
#         model_size: input size of model
#     """
#     # coco dataset has 80 classes -> a color for each class
#     colors = ((np.array(color_palette("hls", 80)) * 255)).astype(np.uint8)
#     img = Image.fromarray(frame)
#     draw = ImageDraw.Draw(img)
#     font = ImageFont.truetype(font=FONT_DIR, size=(img.size[0] + img.size[1]) // 100)
#     boxes_dict = boxes_dicts[0]

#     resize_factor = \
#     (img.size[0] / model_size[0], img.size[1] / model_size[1])
    
#     for cls in range(len(class_names)):
#         boxes = boxes_dict[cls]
#         if np.size(boxes) != 0:
#             color = colors[cls]
#             for box in boxes:
#                 xy, confidence = box[:4], box[4]
#                 xy = [xy[i] * resize_factor[i % 2] for i in range(4)]
#                 x0, y0 = xy[0], xy[1]
#                 thickness = (img.size[0] + img.size[1]) // 200
#                 for t in np.linspace(0, 1, thickness):
#                     xy[0], xy[1] = xy[0] + t, xy[1] + t
#                     xy[2], xy[3] = xy[2] - t, xy[3] - t
#                     draw.rectangle(xy, outline=tuple(color))
#                 text = '{} {:.1f}%'.format(class_names[cls], confidence * 100)
#                 text_size = draw.textsize(text, font=font)
#                 draw.rectangle(
#                     [x0, y0 - text_size[1], x0 + text_size[0], y0],
#                     fill=tuple(color)
#                 )
#                 draw.text((x0, y0 - text_size[1]), text, fill='black', font=font)
#     return img