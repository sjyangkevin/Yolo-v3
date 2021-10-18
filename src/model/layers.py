import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU
from tensorflow.keras.initializers import RandomNormal
import model.config as cfg

def batch_norm(inputs, training, data_format):
    return BatchNormalization(
        axis     = 1 if data_format == 'channels_first' else 3,
        momentum = cfg.BATCH_NORM_MOMENTUM,
        epsilon  = cfg.BATCH_NORM_EPSILON,
        scale    = cfg.BATCH_NORM_SCALE,
    )(inputs, training=training)

def fixed_padding(inputs, kernel_size, data_format):
    pad_total = kernel_size - 1
    pad_begin = pad_total // 2
    pad_end   = pad_total - pad_begin

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_begin, pad_end], [pad_begin, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_begin, pad_end], [pad_begin, pad_end], [0, 0]])
    
    return padded_inputs

def conv2d_fixed_padding(inputs, filters, kernel_size, data_format, strides=1):
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)
    
    return Conv2D(
        filters            = filters,
        kernel_size        = kernel_size,
        strides            = strides,
        padding            = ('same' if strides == 1 else 'valid'),
        use_bias           = cfg.CONV2D_USE_BIAS,
        data_format        = data_format,
        kernel_initializer = RandomNormal(stddev=cfg.CONV2D_INITIALIZER_STDDEV)
    )(inputs)

def residual_block(inputs, filters, training, data_format, strides=1):
    shortcut = inputs
    
    inputs = conv2d_fixed_padding(inputs, filters=filters, kernel_size=1, strides=strides, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = LeakyReLU(alpha=cfg.LEAKY_RELU_ALPHA)(inputs)
    
    inputs = conv2d_fixed_padding(inputs, filters=2*filters, kernel_size=3, strides=strides, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = LeakyReLU(alpha=cfg.LEAKY_RELU_ALPHA)(inputs)
    
    inputs += shortcut
    
    return inputs

def upsample(inputs, out_shape, data_format):
    if data_format == 'channels_first':
        inputs     = tf.transpose(inputs, [0, 2, 3, 1])
        new_height = out_shape[3]
        new_width  = out_shape[2]
    else:
        new_height = out_shape[2]
        new_width  = out_shape[1]
    
    inputs = tf.image.resize(inputs, (new_height, new_width), method='nearest')

    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])
    
    return inputs

def yolo_conv_block(inputs, filters, training, data_format):
    x = conv2d_fixed_padding(inputs, filters=filters, kernel_size=1, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = LeakyReLU(cfg.LEAKY_RELU_ALPHA)(x)
    
    x = conv2d_fixed_padding(x, filters=2*filters, kernel_size=3, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = LeakyReLU(cfg.LEAKY_RELU_ALPHA)(x)
    
    x = conv2d_fixed_padding(x, filters=filters, kernel_size=1, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = LeakyReLU(cfg.LEAKY_RELU_ALPHA)(x)
    
    x = conv2d_fixed_padding(x, filters=2*filters, kernel_size=3, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = LeakyReLU(cfg.LEAKY_RELU_ALPHA)(x)
    
    x = conv2d_fixed_padding(x, filters=filters, kernel_size=1, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = LeakyReLU(cfg.LEAKY_RELU_ALPHA)(x)
    
    route = x
    
    x = conv2d_fixed_padding(x, filters=2*filters, kernel_size=3, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = LeakyReLU(cfg.LEAKY_RELU_ALPHA)(x)
    
    return [route, x]
