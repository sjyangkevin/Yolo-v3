from os import name
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, Input, Lambda
from tensorflow.keras.models import Model
from model.layers import batch_norm, conv2d_fixed_padding, residual_block, upsample, yolo_conv_block
from utils.loss import loss_fn
import model.config as cfg

def darknet53(inputs, training, data_format):
    """create darknet53 model for feature extraction"""
    x = conv2d_fixed_padding(inputs, filters=32, kernel_size=3, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = LeakyReLU(cfg.LEAKY_RELU_ALPHA)(x)
    x = conv2d_fixed_padding(x, filters=64, kernel_size=3, strides=2, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = LeakyReLU(cfg.LEAKY_RELU_ALPHA)(x)
    x = residual_block(x, filters=32, training=training, data_format=data_format)
    x = conv2d_fixed_padding(x, filters=128, kernel_size=3, strides=2, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = LeakyReLU(cfg.LEAKY_RELU_ALPHA)(x)
    
    for _ in range(2):
        x = residual_block(x, filters=64, training=training, data_format=data_format)
    
    x = conv2d_fixed_padding(x, filters=256, kernel_size=3, strides=2, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = LeakyReLU(cfg.LEAKY_RELU_ALPHA)(x)
    
    for _ in range(8):
        x = residual_block(x, filters=128, training=training, data_format=data_format)
    
    # (52, 52)
    route1 = x
    
    x = conv2d_fixed_padding(x, filters=512, kernel_size=3, strides=2, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = LeakyReLU(cfg.LEAKY_RELU_ALPHA)(x)
    
    for _ in range(8):
        x = residual_block(x, filters=256, training=training, data_format=data_format)
    
    # (26, 26)
    route2 = x
    
    x = conv2d_fixed_padding(x, filters=1024, kernel_size=3, strides=2, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = LeakyReLU(cfg.LEAKY_RELU_ALPHA)(x)
    
    for _ in range(4):
        x = residual_block(x, filters=512, training=training, data_format=data_format)
    
    # (13, 13)
    return [route1, route2, x]

def yolo_body(inputs, n_classes, n_anchors, training, data_format):
    assert data_format == 'channels_first' or data_format == 'channels_last', \
        'data format has to be one of ("channels_first", channels_last")'

    inputs = Input(inputs)

    route1, route2, route3 = darknet53(inputs, training=training, data_format=data_format)
    
    route, x = yolo_conv_block(route3, filters=512, training=training, data_format=data_format)
    
    feat1 = Conv2D(filters=n_anchors * (5 + n_classes), 
                                     kernel_size=1,
                                     strides=1,
                                     use_bias=True,
                                     data_format=data_format)(x)
    
    route = conv2d_fixed_padding(route, filters=256, kernel_size=1, data_format=data_format)
    route = batch_norm(route, training=training, data_format=data_format)
    route = LeakyReLU(cfg.LEAKY_RELU_ALPHA)(route)
    
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
    route = LeakyReLU(cfg.LEAKY_RELU_ALPHA)(route)
    
    upsample_size = route1.get_shape().as_list()
    route = upsample(route, out_shape=upsample_size, data_format=data_format)
    route = tf.concat([route, route1], axis= 1 if data_format == 'channels_first' else 3)
    
    route, x = yolo_conv_block(route, filters=128, training=training, data_format=data_format)
    
    feat3 = Conv2D(filters=n_anchors * (5 + n_classes),
                                     kernel_size=1,
                                     strides=1,
                                     use_bias=True,
                                     data_format=data_format)(x)
    
    return Model(inputs, [feat1, feat2, feat3])

def get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask):
    y_true = [Input(shape = (input_shape[0] // {0:32, 1:16, 2:8}[l], input_shape[1] // {0:32, 1:16, 2:8}[l], \
                len(anchors_mask[l]), num_classes + 5)) for l in range(len(anchors_mask))]
    
    # make loss as a Layer, this layer takes model prediction and ground truth-box
    model_loss = Lambda(
        loss_fn,
        output_shape = (1, ),
        name         = 'yolo_loss',
        arguments    = {'input_shape' : input_shape, 'anchors' : anchors, 'anchors_mask' : anchors_mask, 'num_classes' : num_classes}
    )([*model_body.output, *y_true])
    # combine the model and the loss to form an entire model, it takes inputs
    # of image and true boxes, and output a single loss value
    model = Model([model_body.input, *y_true], model_loss)
    return model


# if __name__ == "__main__":
#     from tensorflow.keras.layers import Input
#     #######################################################
#     # The weights being used is trained on COCO dataset
#     # If the implementation of model is correct, it should
#     # load the weights without any problem
#     #######################################################
    
#     # Success - channel first test
#     # m = yolo_body((3, 416, 416), 80, 3, False, 'channels_first')
#     # m.load_weights("../Yolo-v3/src/data/weights/yolo_weights.h5")

#     # # Success - channel last test
#     m = yolo_body((416, 416, 3), 80, 3, False, 'channels_last')
#     m.load_weights("../Yolo-v3/src/data/weights/yolo_weights.h5")

#     # Failed - num_classes not matches the COCO dataset
#     # m = yolo_body((416, 416, 3), 20, 3, False, 'channels_last')
#     # m.load_weights("../Yolo-v3/src/data/weights/yolo_weights.h5")