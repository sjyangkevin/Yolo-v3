import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras import backend as K

def convert_color(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

def resize_image(image, size, is_padded):
    iw, ih = image.size
    w, h   = size
    if is_padded:
        scale     = min(w / iw, h / ih)
        nw        = int(iw * scale)
        nh        = int(ih * scale)

        image     = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

def preprocess_input(image):
    return image / 255.0

def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def get_anchors(anchors_path):
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    # anchors' shape: [9, 2]
    return anchors, len(anchors)

def get_anchors_and_decode(feats, anchors, num_classes, input_shape, calc_loss=False):
    
    num_anchors = len(anchors)
    grid_shape  = K.shape(feats)[1:3]

    grid_x  = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, num_anchors, 1])
    grid_y  = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], num_anchors, 1])
    # grid's shape: [grid_x, grid_y, num_anchors, 2]
    grid    = K.cast(K.concatenate([grid_x, grid_y]), K.dtype(feats))

    # anchors_tensor's shape: [grid_x, grid_y, num_anchors, 2]
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, num_anchors, 2])
    anchors_tensor = K.tile(anchors_tensor, [grid_shape[0], grid_shape[1], 1, 1])

    # feats' shape: [batch_size, grid_x, grid_y, 3, num_classes + (4 + 1)]
    # original is [batch_size, grid_x, grid_y, 3 * (num_classes + 5)]
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # formula for calculating box coordinate prediction from raw prediction
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))

    # formula for calculating the confidence score and class score, just do sigmoid
    box_confidence  = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    # return different set of calculation results, if we want to calculate loss
    if calc_loss:
        #######################################################
        # grid's shape:   [grid_x, grid_y, 3, 2]
        # feats' shape:   [batch_size, grid_x, grid_y, 3, num_classes + (4 + 1)]
        # box_xy's shape: [batch_size, grid_x, grid_y, 3, 2]
        # box_wh's shape: [batch_size, grid_x, grid_y, 3, 2]
        #######################################################
        return grid, feats, box_xy, box_wh
    
    return box_xy, box_wh, box_confidence, box_class_probs

def box_iou(b1, b2):
    # b1's shape: (grid_x, grid_y, num_anchors, 1, 4)
    # calculate box corner coordinates (top left, bottom right)
    b1          = K.expand_dims(b1, -2)
    b1_xy       = b1[..., :2]
    b1_wh       = b1[..., 2:4]
    b1_wh_half  = b1_wh / 2.
    b1_mins     = b1_xy - b1_wh_half
    b1_maxes    = b1_xy + b1_wh_half

    # b2's shape: (1, n, 4) where n is number of boxes
    # calculate box corner coordinates (top left, bottom right)
    b2          = K.expand_dims(b2, 0)
    b2_xy       = b2[..., :2]
    b2_wh       = b2[..., 2:4]
    b2_wh_half  = b2_wh / 2.
    b2_mins     = b2_xy - b2_wh_half
    b2_maxes    = b2_xy + b2_wh_half

    # shape of: (grid_x, grid_y, num_anchors, num_boxes, 2)
    intersect_mins  = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh    = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area         = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area         = b2_wh[..., 0] * b2_wh[..., 1]
    iou             = intersect_area / (b1_area + b2_area - intersect_area)

    return iou

def correct_boxes(box_xy, box_wh, input_shape, image_shape, is_padded):
    # since image is in (height, width) and box is in (width, height)
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    if is_padded:
        new_shape = K.round(image_shape * K.min(input_shape / image_shape))
        offset    = (input_shape - new_shape) / 2. / input_shape
        scale     = input_shape / new_shape

        box_yx    = (box_yx - offset) * scale
        box_hw   *= scale
    
    box_mins  = box_yx - (box_hw / 2.)
    box_maxes = box_yx - (box_hw / 2.)

    boxes  = K.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]])
    boxes *= K.concatenate([image_shape, image_shape])
    
    return boxes

def decode_box(outputs, anchors, num_classes, input_shape, anchors_mask, max_boxes, confidence, nms_iou, is_padded):
    
    image_shape     = K.reshape(outputs[-1], [-1])

    box_xy          = []
    box_wh          = []
    box_confidence  = []
    box_class_probs = []

    for i in range(len(anchors_mask)):
        _box_xy, _box_wh, _box_confidence, _box_class_probs = get_anchors_and_decode(outputs[i], anchors[anchors_mask[i]], num_classes, input_shape)
        box_xy.append(K.reshape(_box_xy, [-1, 2]))
        box_wh.append(K.reshape(_box_wh, [-1, 2]))
        box_confidence.append(K.reshape(_box_confidence, [-1, 1]))
        box_class_probs.append(K.reshape(_box_class_probs, [-1, num_classes]))
    
    box_xy          = K.concatenate(box_xy, axis=0)
    box_wh          = K.concatenate(box_wh, axis=0)
    box_confidence  = K.concatenate(box_confidence, axis=0)
    box_class_probs = K.concatenate(box_class_probs, axis=0)

    boxes = correct_boxes(box_xy, box_wh, input_shape, image_shape, is_padded)
    box_scores = box_confidence * box_class_probs

    mask             = box_scores >= confidence
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_out   = []
    scores_out  = []
    classes_out = []

    for c in range(num_classes):
        class_boxes      = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=nms_iou)

        class_boxes         = K.gather(class_boxes, nms_index)
        class_box_scores    = K.gather(class_box_scores, nms_index)
        classes             = K.ones_like(class_box_scores, 'int32') * c

        boxes_out.append(class_boxes)
        scores_out.append(class_box_scores)
        classes_out.append(classes)
    
    boxes_out      = K.concatenate(boxes_out, axis=0)
    scores_out     = K.concatenate(scores_out, axis=0)
    classes_out    = K.concatenate(classes_out, axis=0)

    return boxes_out, scores_out, classes_out










