import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, Conv2D
from layers import conv2d_fixed_padding, batch_norm, upsample
from backbone import darknet53
from config import leakyrelu_alpha

def yolo_conv_block(inputs, filters, training, data_format):
    x = conv2d_fixed_padding(inputs, filters=filters, kernel_size=1, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = LeakyReLU(leakyrelu_alpha)(x)
    
    x = conv2d_fixed_padding(x, filters=2*filters, kernel_size=3, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = LeakyReLU(leakyrelu_alpha)(x)
    
    x = conv2d_fixed_padding(x, filters=filters, kernel_size=1, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = LeakyReLU(leakyrelu_alpha)(x)
    
    x = conv2d_fixed_padding(x, filters=2*filters, kernel_size=3, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = LeakyReLU(leakyrelu_alpha)(x)
    
    x = conv2d_fixed_padding(x, filters=filters, kernel_size=1, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = LeakyReLU(leakyrelu_alpha)(x)
    
    route = x
    
    x = conv2d_fixed_padding(x, filters=2*filters, kernel_size=3, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = LeakyReLU(leakyrelu_alpha)(x)
    
    return [route, x]

def yolo_layer(inputs, n_classes, n_anchors, training, data_format):
    route1, route2, route3 = darknet53(inputs, training=training, data_format=data_format)
    
    route, x = yolo_conv_block(route3, filters=512, training=training, data_format=data_format)
    
    feat1 = Conv2D(filters=n_anchors * (5 + n_classes), 
                                     kernel_size=1,
                                     strides=1,
                                     use_bias=True,
                                     data_format=data_format)(x)
    
    route = conv2d_fixed_padding(route, filters=256, kernel_size=1, data_format=data_format)
    route = batch_norm(route, training=training, data_format=data_format)
    route = LeakyReLU(leakyrelu_alpha)(route)
    
    upsample_size = route2.get_shape().as_list()
    route = upsample(route, out_shape=upsample_size, data_format=data_format)
    route = tf.concat([route, route2], axis= 1 if data_format == 'channels_first' else 3)
    
    route, x = yolo_conv_block(route, filters=256, training=training, data_format=data_format)
    
    feat2 = Conv2D(filters=n_anchors * (5 + n_classes),
                                     kernel_size=1,
                                     strides=1,
                                     use_bias=True,
                                     data_format=data_format)(x)
    
    route = conv2d_fixed_padding(route, filters=128, kernel_size=1, data_format=data_format)
    route = batch_norm(route, training=training, data_format=data_format)
    route = LeakyReLU(leakyrelu_alpha)(route)
    
    upsample_size = route1.get_shape().as_list()
    route = upsample(route, out_shape=upsample_size, data_format=data_format)
    route = tf.concat([route, route1], axis= 1 if data_format == 'channels_first' else 3)
    
    route, x = yolo_conv_block(route, filters=128, training=training, data_format=data_format)
    
    feat3 = Conv2D(filters=n_anchors * (5 + n_classes),
                                     kernel_size=1,
                                     strides=1,
                                     use_bias=True,
                                     data_format=data_format)(x)
    
    return [feat1, feat2, feat3]

# if __name__ == "__main__":
#     from tensorflow.keras.utils import plot_model
#     from tensorflow.keras.layers import Input
#     from tensorflow.keras import Model
#     from config import n_classes, n_anchors, training, data_format

#     inputs = Input((416, 416, 3))
#     outputs = yolo_layer(inputs, n_classes, n_anchors, training, data_format)
#     model = Model(inputs=inputs, outputs=outputs)
    
#     for feat in outputs:
#         print(feat.shape)

    # plot_model(model, to_file='model.png', show_shapes=True)