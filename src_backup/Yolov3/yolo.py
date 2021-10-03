import numpy as np
import tensorflow as tf
from Yolov3.config import *
from Yolov3.utils import build_boxes, load_class_names, non_max_suppression, box_giou

def batch_norm(inputs, training, data_format):
    """Performs a batch normalization using a standard set of parameters"""
    return tf.keras.layers.BatchNormalization(
        axis = 1 if data_format == 'channels_first' else 3,
        momentum = BATCH_NORM_DECAY,
        epsilon = BATCH_NORM_EPSILON,
        scale = True
    )(inputs, training=training)

def fixed_padding(inputs, kernel_size, data_format):
    """ResNet implementation of fixed padding"""
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    
    if data_format == 'channels_first':
        # (batch_size, channels, height, width)
        padded_inputs = tf.pad(inputs, [[0, 0], [0,0], [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        # (batch_size, height, width, channels)
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        
    return padded_inputs

def conv2d_fixed_padding(inputs, filters, kernel_size, data_format, strides=1):
    """strided 2-d convolution with explicit padding"""
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)
    
    return tf.keras.layers.Conv2D(
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = ('same' if strides == 1 else 'valid'),
        use_bias = False,
        data_format = data_format
    )(inputs)

def darknet53_residual_block(inputs, filters, training, data_format, strides=1):
    """creates a residual block for darknet"""
    shortcut = inputs
    
    inputs = conv2d_fixed_padding(inputs, filters=filters, kernel_size=1, strides=strides, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(inputs)
    
    inputs = conv2d_fixed_padding(inputs, filters=2*filters, kernel_size=3, strides=strides, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(inputs)
    
    inputs += shortcut
    
    return inputs

def darknet53(inputs, training, data_format):
    """create darknet53 model for feature extraction"""
    x = conv2d_fixed_padding(inputs, filters=32, kernel_size=3, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
    x = conv2d_fixed_padding(x, filters=64, kernel_size=3, strides=2, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
    x = darknet53_residual_block(x, filters=32, training=training, data_format=data_format)
    x = conv2d_fixed_padding(x, filters=128, kernel_size=3, strides=2, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
    
    for _ in range(2):
        x = darknet53_residual_block(x, filters=64, training=training, data_format=data_format)
    
    x = conv2d_fixed_padding(x, filters=256, kernel_size=3, strides=2, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
    
    for _ in range(8):
        x = darknet53_residual_block(x, filters=128, training=training, data_format=data_format)
    
    # (52, 52, 256)
    route1 = x
    
    x = conv2d_fixed_padding(x, filters=512, kernel_size=3, strides=2, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
    
    for _ in range(8):
        x = darknet53_residual_block(x, filters=256, training=training, data_format=data_format)
    
    # (26, 26, 512)
    route2 = x
    
    x = conv2d_fixed_padding(x, filters=1024, kernel_size=3, strides=2, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
    
    for _ in range(4):
        # (13, 13, 1024)
        x = darknet53_residual_block(x, filters=512, training=training, data_format=data_format)
    
    return [route1, route2, x]

def upsample(inputs, out_shape, data_format):
    """upsamples to 'out_shape' using nearest neighbor interpolation."""
    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
        new_height = out_shape[3]
        new_width = out_shape[2]
    else:
        new_height = out_shape[2]
        new_width = out_shape[1]
        
    inputs = tf.image.resize(inputs, (new_height, new_width), method='nearest')
    
    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])
    
    return inputs

def yolo_convolution_block(inputs, filters, training, data_format):
    """creates convolution operations layer used after Darknet."""
    x = conv2d_fixed_padding(inputs, filters=filters, kernel_size=1, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
    
    x = conv2d_fixed_padding(x, filters=2*filters, kernel_size=3, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
    
    x = conv2d_fixed_padding(x, filters=filters, kernel_size=1, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
    
    x = conv2d_fixed_padding(x, filters=2*filters, kernel_size=3, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
    
    x = conv2d_fixed_padding(x, filters=filters, kernel_size=1, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
    
    # yolo-v3 make prediction at this route
    route = x
    
    x = conv2d_fixed_padding(x, filters=2*filters, kernel_size=3, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
    
    # inputs will be used by Upsampling2D such that it can be merge to other feature maps
    return [route, x]

def YOLOv3(inputs, n_classes, n_anchors, training, data_format):

    route1, route2, route3 = darknet53(inputs, training=training, data_format=data_format)
    
    route, x = yolo_convolution_block(route3, filters=512, training=training, data_format=data_format)
    
    route_lbox = tf.keras.layers.Conv2D(filters=n_anchors * (5 + n_classes), 
                                     kernel_size=1,
                                     strides=1,
                                     use_bias=True,
                                     data_format=data_format)(x)
    
    route = conv2d_fixed_padding(route, filters=256, kernel_size=1, data_format=data_format)
    route = batch_norm(route, training=training, data_format=data_format)
    route = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(route)
    
    upsample_size = route2.get_shape().as_list()
    route = upsample(route, out_shape=upsample_size, data_format=data_format)
    route = tf.concat([route, route2], axis= 1 if data_format == 'channels_first' else 3)
    
    route, x = yolo_convolution_block(route, filters=256, training=training, data_format=data_format)
    
    route_mbox = tf.keras.layers.Conv2D(filters=n_anchors * (5 + n_classes),
                                     kernel_size=1,
                                     strides=1,
                                     use_bias=True,
                                     data_format=data_format)(x)
    
    route = conv2d_fixed_padding(route, filters=128, kernel_size=1, data_format=data_format)
    route = batch_norm(route, training=training, data_format=data_format)
    route = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(route)
    
    upsample_size = route1.get_shape().as_list()
    route = upsample(route, out_shape=upsample_size, data_format=data_format)
    route = tf.concat([route, route1], axis= 1 if data_format == 'channels_first' else 3)
    
    route, x = yolo_convolution_block(route, filters=128, training=training, data_format=data_format)
    
    route_sbox = tf.keras.layers.Conv2D(filters=n_anchors * (5 + n_classes),
                                     kernel_size=1,
                                     strides=1,
                                     use_bias=True,
                                     data_format=data_format)(x)
    
    return [route_lbox, route_mbox, route_sbox]

def create_model(n_classes, n_anchors, input_size, training, data_format):
    input_layer = tf.keras.layers.Input(shape=(input_size, input_size, 3))

    route_lbox, route_mbox, route_sbox = YOLOv3(
        input_layer,
        n_classes,
        n_anchors,
        training,
        data_format    
    )

    return tf.keras.models.Model(
        inputs  = input_layer, 
        outputs = [route_lbox, route_mbox, route_sbox]
    )

def decode(inputs, n_classes, anchors, img_size, data_format, training):

    n_anchors = len(anchors)
    shape = inputs.get_shape().as_list()
    grid_shape = shape[2:4] if data_format == 'channels_first' else shape[1:3]
    if data_format == 'channels_first':
        # make it channel last
        inputs = tf.transpose(inputs, [0, 2, 3, 1])

    # the scale between the original input size and feature map size
    strides = (img_size[0] // grid_shape[0], img_size[1] // grid_shape[1])

    # this format of decoding is used during the inference mode
    # it can be directly passed into tf.image.non_max_suppression
    if not training:

        # adjust the input shapes to (batch_size, total_num_pred_box, 4 + 1 + n_classes)
        inputs = tf.reshape(inputs, [-1, n_anchors * grid_shape[0] * grid_shape[1], 5 + n_classes])
        
        # we split inputs tensor to box_xy, box_wh, box_confidence, n_classes, such that
        # box_centers = (batch_size, total_num_pred_box, x, y)
        # box_shapes = (batch_size, total_num_pred_box, w, h)
        # confidence = (batch_size, total_num_pred_box, confidence)
        # classes = (batch_size, total_num_pred_box, [class 1, class 2, ..., class n])
        box_centers, box_shapes, confidence, classes = tf.split(inputs, [2, 2, 1, n_classes], axis=-1)
        
        # construct grid coordinate for bounding box adjustment
        x = tf.range(grid_shape[0], dtype=tf.float32)
        y = tf.range(grid_shape[1], dtype=tf.float32)
        x_offset, y_offset = tf.meshgrid(x, y)
        # mesh grid return a 2D array, in this case
        # e.g. x = [1,2,3], meshgrid -> x = [[1,2,3],[1,2,3],[1,2,3]]
        x_offset = tf.reshape(x_offset, (-1, 1))
        # reshape in this way is equivalent to unroll the 2d array to 1d
        y_offset = tf.reshape(y_offset, (-1, 1))
        # after concat, it gives an array of (grid_shape[0] * grid_shape[1], 2)
        # it represents the coordinate of the grid
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        # after tile it, keep the first dimension not change, return a tensor 
        # of shape (grid_shape[0] * grid_shape[1], n_anchors * 2)
        # it means for each anchor, there is a coordinate grid
        x_y_offset = tf.tile(x_y_offset, [1, n_anchors])
        # after reshape, the tensor becomes (1, grid_shape[0] * grid_shape[1] * n_anchors, 2)
        # it just stack them along the second axis
        x_y_offset = tf.reshape(x_y_offset, [1, -1, 2])
        box_centers = tf.sigmoid(box_centers)
        # adjust box to feature map grid, and scale it back to image
        box_centers = (box_centers + x_y_offset) * strides
        
        anchors = tf.tile(anchors, [grid_shape[0] * grid_shape[1], 1])
        box_shapes = tf.exp(box_shapes) * tf.cast(anchors, dtype=tf.float32)
        
        confidence = tf.sigmoid(confidence)
        
        classes = tf.sigmoid(classes)
        
        # back to shape (batch_size, num_boxes, 5 + num_classes)
        # for passing into nms
        inputs = tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)

        return inputs

    # this format of decoding is used during training, for loss computation
    else:

        batch_size = tf.shape(inputs)[0]

        inputs = tf.reshape(inputs, [batch_size, grid_shape[0], grid_shape[1], n_anchors, 5 + n_classes])

        box_centers, box_shapes, confidence, classes = tf.split(inputs, (2, 2, 1, n_classes), axis=-1)

        x_y_offset = tf.meshgrid(tf.range(grid_shape[0]), tf.range(grid_shape[1]))
        x_y_offset = tf.expand_dims(tf.stack(x_y_offset, axis=-1), axis=2)  # [gx, gy, 1, 2]
        # shape = (batch_size, gx, gy, 3, 2) # num_anchors, and each anchor has a x, y offset
        x_y_offset = tf.tile(tf.expand_dims(x_y_offset, axis=0), [batch_size, 1, 1, 3, 1])
        x_y_offset = tf.cast(x_y_offset, tf.float32)

        box_centers = tf.sigmoid(box_centers)
        box_centers = (box_centers + x_y_offset) * strides

        box_shapes = tf.exp(box_shapes) * tf.cast(anchors, dtype=tf.float32)

        confidence = tf.sigmoid(confidence)

        classes = tf.sigmoid(classes)

        inputs = tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)

        return inputs

def predict(inputs, n_classes, anchors, img_size, max_output_size, iou_threshold, confidence_threshold, data_format):
    feat_lbox, feat_mbox, feat_sbox = inputs

    dect_lbox = decode(feat_lbox, n_classes, anchors[6:9], img_size, data_format, training=False)
    dect_mbox = decode(feat_mbox, n_classes, anchors[3:6], img_size, data_format, training=False)
    dect_sbox = decode(feat_sbox, n_classes, anchors[0:3], img_size, data_format, training=False)

    pre_boxes = tf.concat([dect_lbox, dect_mbox, dect_sbox], axis=1)

    boxes = build_boxes(pre_boxes)

    boxes_dicts = non_max_suppression(
        boxes,
        n_classes=n_classes,
        max_output_size=max_output_size,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold
    )

    return boxes_dicts

def bbox_iou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area

def compute_loss(pred, raw_pred, label, bboxes):
    n_class = len(load_class_names(CLASS_NAME_FILE))
    raw_pred_shape = tf.shape(raw_pred)
    batch_size     = raw_pred_shape[0]
    output_size    = raw_pred_shape[1]
    input_size     = MODEL_INPUT_SIZE

    raw_pred = tf.reshape(raw_pred, [batch_size, output_size, output_size, 3, 5 + n_class])

    raw_pred_conf = raw_pred[:, :, :, :, 4:5]
    raw_pred_prob = raw_pred[:, :, :, :, 5:]

    pred_xywh = pred[:, :, :, :, 0:4]
    # pred_conf = pred[:, :, :, :, 4:5]

    label_xywh  = label[:, :, :, :, 0:4]
    object_mask = label[:, :, :, :, 4:5]
    label_prob  = label[:, :, :, :, 5:]

    giou = tf.expand_dims(box_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = object_mask * bbox_loss_scale * (1 - giou)

    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    # Find the value of IoU with the real box The largest prediction box
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    non_object_mask = (1.0 - object_mask) * tf.cast( max_iou < IOU_THRESHOLD, tf.float32 )

    # Calculate the loss of confidence
    # we hope that if the grid contains objects, then the network output prediction box has a confidence of 1 and 0 when there is no object.
    conf_loss = (
            object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=raw_pred_conf)
            +
            non_object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=raw_pred_conf)
    )    

    prob_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=raw_pred_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

    return giou_loss, conf_loss, prob_loss