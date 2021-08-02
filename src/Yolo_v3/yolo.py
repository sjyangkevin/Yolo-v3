import tensorflow as tf
from configparser import ConfigParser
from darknet53 import batch_norm, conv2d_fixed_padding, darknet53
from utils import build_boxes, non_max_suppression, preprocess_true_boxes

parser = ConfigParser()
parser.read('config.ini')

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
    x = tf.keras.layers.LeakyReLU(alpha=parser.getfloat('model', 'leaky_relu_alpha'))(x)
    
    x = conv2d_fixed_padding(x, filters=2*filters, kernel_size=3, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = tf.keras.layers.LeakyReLU(alpha=parser.getfloat('model', 'leaky_relu_alpha'))(x)
    
    x = conv2d_fixed_padding(x, filters=filters, kernel_size=1, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = tf.keras.layers.LeakyReLU(alpha=parser.getfloat('model', 'leaky_relu_alpha'))(x)
    
    x = conv2d_fixed_padding(x, filters=2*filters, kernel_size=3, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = tf.keras.layers.LeakyReLU(alpha=parser.getfloat('model', 'leaky_relu_alpha'))(x)
    
    x = conv2d_fixed_padding(x, filters=filters, kernel_size=1, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = tf.keras.layers.LeakyReLU(alpha=parser.getfloat('model', 'leaky_relu_alpha'))(x)
    
    # yolo-v3 make prediction at this route
    route = x
    
    x = conv2d_fixed_padding(x, filters=2*filters, kernel_size=3, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = tf.keras.layers.LeakyReLU(alpha=parser.getfloat('model', 'leaky_relu_alpha'))(x)
    
    # inputs will be used by Upsampling2D such that it can be merge to other feature maps
    return [route, x]

def yolo_layer(inputs, n_classes, n_anchors, training, data_format):
    route1, route2, route3 = darknet53(inputs, training=training, data_format=data_format)
    
    route, x = yolo_convolution_block(route3, filters=512, training=training, data_format=data_format)
    
    feat_map_1 = tf.keras.layers.Conv2D(filters=n_anchors * (5 + n_classes), 
                                     kernel_size=1,
                                     strides=1,
                                     use_bias=True,
                                     data_format=data_format)(x)
    
    route = conv2d_fixed_padding(route, filters=256, kernel_size=1, data_format=data_format)
    route = batch_norm(route, training=training, data_format=data_format)
    route = tf.keras.layers.LeakyReLU(alpha=parser.getfloat('model', 'leaky_relu_alpha'))(route)
    
    upsample_size = route2.get_shape().as_list()
    route = upsample(route, out_shape=upsample_size, data_format=data_format)
    route = tf.concat([route, route2], axis= 1 if data_format == 'channels_first' else 3)
    
    route, x = yolo_convolution_block(route, filters=256, training=training, data_format=data_format)
    
    feat_map_2 = tf.keras.layers.Conv2D(filters=n_anchors * (5 + n_classes),
                                     kernel_size=1,
                                     strides=1,
                                     use_bias=True,
                                     data_format=data_format)(x)
    
    route = conv2d_fixed_padding(route, filters=128, kernel_size=1, data_format=data_format)
    route = batch_norm(route, training=training, data_format=data_format)
    route = tf.keras.layers.LeakyReLU(alpha=parser.getfloat('model', 'leaky_relu_alpha'))(route)
    
    upsample_size = route1.get_shape().as_list()
    route = upsample(route, out_shape=upsample_size, data_format=data_format)
    route = tf.concat([route, route1], axis= 1 if data_format == 'channels_first' else 3)
    
    route, x = yolo_convolution_block(route, filters=128, training=training, data_format=data_format)
    
    feat_map_3 = tf.keras.layers.Conv2D(filters=n_anchors * (5 + n_classes),
                                     kernel_size=1,
                                     strides=1,
                                     use_bias=True,
                                     data_format=data_format)(x)
    
    return [feat_map_1, feat_map_2, feat_map_3]

def yolo_head(inputs, n_classes, anchors, img_size, data_format):
    
    n_anchors = len(anchors)
    
    shape = inputs.get_shape().as_list()
    grid_shape = shape[2:4] if data_format == 'channels_first' else shape[1:3]
 
    if data_format == 'channels_first':
        # make it channel last
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
    # adjust the input shapes to (batch_size, total_num_pred_box, 4 + 1 + n_classes)
 
    inputs = tf.reshape(inputs, [-1, n_anchors * grid_shape[0] * grid_shape[1], 5 + n_classes])
    
    # the scale between the original input size and feature map size
    strides = (img_size[0] // grid_shape[0], img_size[1] // grid_shape[1])
    
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
    box_centers = tf.keras.layers.Activation('sigmoid')(box_centers)
    # adjust box to feature map grid, and scale it back to image
    box_centers = (box_centers + x_y_offset) * strides
    
    anchors = tf.tile(anchors, [grid_shape[0] * grid_shape[1], 1])
    box_shapes = tf.exp(box_shapes) * tf.cast(anchors, dtype=tf.float32)
    
    confidence = tf.keras.layers.Activation('sigmoid')(confidence)
    
    classes = tf.keras.layers.Activation('sigmoid')(classes)
    
    # back to shape (batch_size, total_num_pred_box, 5 + num_classes)
    inputs = tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)
    
    return inputs

def yolo_predict(inputs, n_classes, anchors, img_size, max_output_size, iou_threshold, confidence_threshold, data_format):
    
    feat1, feat2, feat3 = inputs[0], inputs[1], inputs[2]
    
    detect1 = yolo_head(feat1, n_classes, anchors[6:9], img_size, data_format)
    detect2 = yolo_head(feat2, n_classes, anchors[3:6], img_size, data_format)
    detect3 = yolo_head(feat3, n_classes, anchors[0:3], img_size, data_format)
    
    x = tf.concat([detect1, detect2, detect3], axis=1)
    
    x = build_boxes(x)
    
    boxes_dicts = non_max_suppression(
        x,
        n_classes=n_classes,
        max_output_size=max_output_size,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold
    )
    
    return boxes_dicts

def yolo_loss(args, anchors, num_classes, ignore_threshold=.5, print_loss=False):
    """
    Return yolo_loss tensor

    yolo_outputs: list of tensor, the output of yolo_layer
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape = (N, 2), wh
    num_classes: integer
    ignore_threshold: float, the iou threshold whether to ignore object confidence loss

    return:
        loss: tensor, shape=(1, )
    """
    pass
