import tensorflow as tf
from configurations.Yolo_v3_Config import * 

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
    inputs = tf.compat.v1.nn.leaky_relu(inputs, alpha=LEAKY_RELU_ALPHA)
    
    inputs = conv2d_fixed_padding(inputs, filters=2*filters, kernel_size=3, strides=strides, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.compat.v1.nn.leaky_relu(inputs, alpha=LEAKY_RELU_ALPHA)
    
    inputs += shortcut
    
    return inputs

def darknet53(inputs, training, data_format):
    """create darknet53 model for feature extraction"""
    inputs = conv2d_fixed_padding(inputs, filters=32, kernel_size=3, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.compat.v1.nn.leaky_relu(inputs, alpha=LEAKY_RELU_ALPHA)
    inputs = conv2d_fixed_padding(inputs, filters=64, kernel_size=3, strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.compat.v1.nn.leaky_relu(inputs, alpha=LEAKY_RELU_ALPHA)
    inputs = darknet53_residual_block(inputs, filters=32, training=training, data_format=data_format)
    inputs = conv2d_fixed_padding(inputs, filters=128, kernel_size=3, strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.compat.v1.nn.leaky_relu(inputs, alpha=LEAKY_RELU_ALPHA)
    
    for _ in range(2):
        inputs = darknet53_residual_block(inputs, filters=64, training=training, data_format=data_format)
    
    inputs = conv2d_fixed_padding(inputs, filters=256, kernel_size=3, strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.compat.v1.nn.leaky_relu(inputs, alpha=LEAKY_RELU_ALPHA)
    
    for _ in range(8):
        inputs = darknet53_residual_block(inputs, filters=128, training=training, data_format=data_format)
    
    # (52, 52)
    route1 = inputs
    
    inputs = conv2d_fixed_padding(inputs, filters=512, kernel_size=3, strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.compat.v1.nn.leaky_relu(inputs, alpha=LEAKY_RELU_ALPHA)
    
    for _ in range(8):
        inputs = darknet53_residual_block(inputs, filters=256, training=training, data_format=data_format)
    
    # (26, 26)
    route2 = inputs
    
    inputs = conv2d_fixed_padding(inputs, filters=1024, kernel_size=3, strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.compat.v1.nn.leaky_relu(inputs, alpha=LEAKY_RELU_ALPHA)
    
    for _ in range(4):
        inputs = darknet53_residual_block(inputs, filters=512, training=training, data_format=data_format)
    
    # (13, 13)
    return route1, route2, inputs