import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU
from config import bn_momentum, bn_epsilon, leakyrelu_alpha

def batch_norm(inputs, training, data_format):
    return BatchNormalization(
        axis = 1 if data_format == 'channels_first' else 3,
        momentum = bn_momentum,
        epsilon = bn_epsilon,
        scale = True
    )(inputs, training=training)

def fixed_padding(inputs, kernel_size, data_format):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    
    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0,0], [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        
    return padded_inputs

def conv2d_fixed_padding(inputs, filters, kernel_size, data_format, strides=1):
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)
    
    return Conv2D(
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = ('same' if strides == 1 else 'valid'),
        use_bias = False,
        data_format = data_format
    )(inputs)

def residual_block(inputs, filters, training, data_format, strides=1):
    shortcut = inputs
    
    inputs = conv2d_fixed_padding(inputs, filters=filters, kernel_size=1, strides=strides, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = LeakyReLU(alpha=leakyrelu_alpha)(inputs)
    
    inputs = conv2d_fixed_padding(inputs, filters=2*filters, kernel_size=3, strides=strides, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = LeakyReLU(alpha=leakyrelu_alpha)(inputs)
    
    inputs += shortcut
    
    return inputs