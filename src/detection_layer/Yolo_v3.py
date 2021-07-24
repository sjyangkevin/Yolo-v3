import tensorflow as tf
from feature_extractor.Darknet53 import *
from utils.Yolo_v3_utils import *

def yolo_convolution_block(inputs, filters, training, data_format):
    """creates convolution operations layer used after Darknet."""
    inputs = conv2d_fixed_padding(inputs, filters=filters, kernel_size=1, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.compat.v1.nn.leaky_relu(inputs, alpha=LEAKY_RELU_ALPHA)
    
    inputs = conv2d_fixed_padding(inputs, filters=2*filters, kernel_size=3, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.compat.v1.nn.leaky_relu(inputs, alpha=LEAKY_RELU_ALPHA)
    
    inputs = conv2d_fixed_padding(inputs, filters=filters, kernel_size=1, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.compat.v1.nn.leaky_relu(inputs, alpha=LEAKY_RELU_ALPHA)
    
    inputs = conv2d_fixed_padding(inputs, filters=2*filters, kernel_size=3, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.compat.v1.nn.leaky_relu(inputs, alpha=LEAKY_RELU_ALPHA)
    
    inputs = conv2d_fixed_padding(inputs, filters=filters, kernel_size=1, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.compat.v1.nn.leaky_relu(inputs, alpha=LEAKY_RELU_ALPHA)
    
    # yolo-v3 make prediction at this route
    route = inputs
    
    inputs = conv2d_fixed_padding(inputs, filters=2*filters, kernel_size=3, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.compat.v1.nn.leaky_relu(inputs, alpha=LEAKY_RELU_ALPHA)
    
    # inputs will be used by Upsampling2D such that it can be merge to other feature maps
    return route, inputs

def yolo_layer(inputs, n_classes, anchors, img_size, data_format):
    """
    create yolo's final detection layer
    detect boxes with respect to anchors
    
    args:
        inputs: tensor input
        n_classes: number of labels
        anchors: a list of anchor sizes
        img_size: the input size of the model
        data_format: the input format
        
    return:
        tensor output
    """
    # we have 9 anchors in total
    n_anchors = len(anchors)
    
    inputs = tf.keras.layers.Conv2D(filters=n_anchors * (5 + n_classes), kernel_size=1, strides=1, use_bias=True, data_format=data_format)(inputs)
    
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
    box_centers = tf.compat.v1.nn.sigmoid(box_centers)
    # adjust box to feature map grid, and scale it back to image
    box_centers = (box_centers + x_y_offset) * strides
    
    anchors = tf.tile(anchors, [grid_shape[0] * grid_shape[1], 1])
    box_shapes = tf.exp(box_shapes) * tf.cast(anchors, dtype=tf.float32)
    
    confidence = tf.compat.v1.nn.sigmoid(confidence)
    
    classes = tf.compat.v1.nn.sigmoid(classes)
    
    # back to shape (batch_size, total_num_pred_box, 5 + num_classes)
    inputs = tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)
    
    return inputs

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

class Yolo_v3:
    """Yolo v3 model class"""
    
    def __init__(self, n_classes, model_size, max_output_size, iou_threshold, confidence_threshold, data_format=None):
        if not data_format:
            if tf.test.is_built_with_cuda():
                data_format = 'channels_first'
            else:
                data_format = 'channels_last'
        
        self.n_classes = n_classes
        self.model_size = model_size
        self.max_output_size = max_output_size
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.data_format = data_format
    
    def __call__(self, inputs, training):
        """
        add operations to detect boxes for a batch of input images
        
        args:
            inputs: a tensor representing a batch of input images
            training: a boolean, whether to use in training or inference mode
            
        returns:
            a list containing class-to-boxes dictionaries for each sample in the batch
        """
        
        with tf.compat.v1.variable_scope('yolo_v3_model'):
            if self.data_format == 'channels_first':
                inputs = tf.transpose(inputs, [0, 3, 1, 2])
                
            inputs = inputs / 255
            
            route1, route2, inputs = darknet53(inputs, training=training, data_format=self.data_format)
            
            route, inputs = yolo_convolution_block(
                inputs,
                filters=512,
                training=training,
                data_format=self.data_format
            )
            
            detect1 = yolo_layer(
                inputs,
                n_classes=self.n_classes,
                anchors=ANCHORS[6:9],
                img_size=self.model_size,
                data_format=self.data_format
            )
            
            inputs = conv2d_fixed_padding(route, filters=256, kernel_size=1, data_format=self.data_format)
            inputs = batch_norm(inputs, training=training, data_format=self.data_format)
            inputs = tf.compat.v1.nn.leaky_relu(inputs, alpha=LEAKY_RELU_ALPHA)
            
            # concate the first feature map to the one above it
            upsample_size = route2.get_shape().as_list()
            inputs = upsample(inputs, out_shape=upsample_size, data_format=self.data_format)
            axis = 1 if self.data_format == 'channels_first' else 3
            inputs = tf.concat([inputs, route2], axis=axis)
            
            route, inputs = yolo_convolution_block(
                inputs,
                filters=256,
                training=training,
                data_format=self.data_format
            )
            
            detect2 = yolo_layer(
                inputs,
                n_classes=self.n_classes,
                anchors=ANCHORS[3:6],
                img_size=self.model_size,
                data_format=self.data_format
            )
            
            inputs = conv2d_fixed_padding(route, filters=128, kernel_size=1, data_format=self.data_format)
            inputs = batch_norm(inputs, training=training, data_format=self.data_format)
            inputs = tf.compat.v1.nn.leaky_relu(inputs, alpha=LEAKY_RELU_ALPHA)
            
            upsample_size = route1.get_shape().as_list()
            inputs = upsample(inputs, out_shape=upsample_size, data_format=self.data_format)
            inputs = tf.concat([inputs, route1], axis=axis)
            
            route, inputs = yolo_convolution_block(
                inputs,
                filters=128,
                training=training,
                data_format=self.data_format
            )
            
            detect3 = yolo_layer(
                inputs,
                n_classes=self.n_classes,
                anchors=ANCHORS[0:3],
                img_size=self.model_size,
                data_format=self.data_format
            )
            
            # prediction at three different scales
            inputs = tf.concat([detect1, detect2, detect3], axis=1)
            
            inputs = build_boxes(inputs)
            
            boxes_dicts = non_max_suppression(
                inputs,
                n_classes=self.n_classes,
                max_output_size=self.max_output_size,
                iou_threshold=self.iou_threshold,
                confidence_threshold=self.confidence_threshold
            )
            
            return boxes_dicts