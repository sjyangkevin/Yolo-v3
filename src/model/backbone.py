from tensorflow.keras.layers import LeakyReLU
from model.layers import batch_norm, conv2d_fixed_padding, residual_block
from model.config import leakyrelu_alpha

def darknet53(inputs, training, data_format):
    """create darknet53 model for feature extraction"""
    x = conv2d_fixed_padding(inputs, filters=32, kernel_size=3, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = LeakyReLU(leakyrelu_alpha)(x)
    x = conv2d_fixed_padding(x, filters=64, kernel_size=3, strides=2, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = LeakyReLU(leakyrelu_alpha)(x)
    x = residual_block(x, filters=32, training=training, data_format=data_format)
    x = conv2d_fixed_padding(x, filters=128, kernel_size=3, strides=2, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = LeakyReLU(leakyrelu_alpha)(x)
    
    for _ in range(2):
        x = residual_block(x, filters=64, training=training, data_format=data_format)
    
    x = conv2d_fixed_padding(x, filters=256, kernel_size=3, strides=2, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = LeakyReLU(leakyrelu_alpha)(x)
    
    for _ in range(8):
        x = residual_block(x, filters=128, training=training, data_format=data_format)
    
    # (52, 52)
    route1 = x
    
    x = conv2d_fixed_padding(x, filters=512, kernel_size=3, strides=2, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = LeakyReLU(leakyrelu_alpha)(x)
    
    for _ in range(8):
        x = residual_block(x, filters=256, training=training, data_format=data_format)
    
    # (26, 26)
    route2 = x
    
    x = conv2d_fixed_padding(x, filters=1024, kernel_size=3, strides=2, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = LeakyReLU(leakyrelu_alpha)(x)
    
    for _ in range(4):
        x = residual_block(x, filters=512, training=training, data_format=data_format)
    
    # (13, 13)
    return [route1, route2, x]

# if __name__ == "__main__":
#     from tensorflow.keras.utils import plot_model
#     from tensorflow.keras.layers import Input
#     from tensorflow.keras import Model

#     inputs = Input((416, 416, 3))
#     outputs = darknet53(inputs, False, "channels_last")
#     model = Model(inputs=inputs, outputs=outputs)

#     plot_model(model, to_file='model.png', show_shapes=True)
