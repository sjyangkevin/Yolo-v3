import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display
from seaborn import color_palette
import cv2
import struct

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-05
LEAKY_RELU_ALPHA = 0.1
# bounding box priors
ANCHORS = [
    (10, 13), (16, 30), (33, 23),
    (30, 61), (62, 45), (59, 119),
    (116, 90), (156, 198), (373, 326)
]
# input size of model
MODEL_SIZE = (416, 416)

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
    
    # (52, 52)
    route1 = x
    
    x = conv2d_fixed_padding(x, filters=512, kernel_size=3, strides=2, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
    
    for _ in range(8):
        x = darknet53_residual_block(x, filters=256, training=training, data_format=data_format)
    
    # (26, 26)
    route2 = x
    
    x = conv2d_fixed_padding(x, filters=1024, kernel_size=3, strides=2, data_format=data_format)
    x = batch_norm(x, training=training, data_format=data_format)
    x = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
    
    for _ in range(4):
        x = darknet53_residual_block(x, filters=512, training=training, data_format=data_format)
    
    # (13, 13)
    return [route1, route2, x]

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
    route = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(route)
    
    upsample_size = route2.get_shape().as_list()
    route = upsample(route, out_shape=upsample_size, data_format=data_format)
    route = tf.concat([route, route2], axis= 1 if data_format == 'channels_first' else 3)
    
    route, x = yolo_convolution_block(route, filters=256, training=training, data_format=data_format)
    
    feat_map_2 = tf.keras.layers.Conv2D(filters=n_anchors * (5 + n_classes),
                                     kernel_size=1,
                                     strides=1,
                                     use_bias=True,
                                     data_format=data_format)(x)
    
    route = conv2d_fixed_padding(route, filters=256, kernel_size=1, data_format=data_format)
    route = batch_norm(route, training=training, data_format=data_format)
    route = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(route)
    
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

def yolo_model(n_classes, n_anchors, model_size, training, data_format):
    inputs = tf.keras.layers.Input(shape=(model_size[0], model_size[1], 3))
    
    f1, f2, f3 = yolo_layer(inputs, n_classes, n_anchors, training, data_format)
    
    return tf.keras.models.Model(inputs=inputs, outputs=[f1, f2, f3])

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

def build_boxes(inputs):
    """computes top left and bottom right points of the boxes."""
    center_x, center_y, width, height, confidence, classes = \
    tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)
    
    top_left_x = center_x - width / 2
    top_left_y = center_y - height / 2
    bottom_right_x = center_x + width / 2
    bottom_right_y = center_y + height / 2
    
    boxes = tf.concat([
        top_left_x,
        top_left_y,
        bottom_right_x,
        bottom_right_y,
        confidence,
        classes
    ], axis=-1)
    
    return boxes

def non_max_suppression(inputs, n_classes, max_output_size, iou_threshold, confidence_threshold):
    """
    perform non-max suppression separately for each class
    
    args:
        inputs: tensor input.
        n_classes: number of classes
        max_output_size: max number of boxes to be selected for each class
        iou_threshold: threshold for the IOU
        confidence_threshold: threshold for the confidence score
        
    returns:
        a list containing class-to-boxes dictionaries
        for each sample in the batch
    """
    # unpack along the batch_size dimension -> batch = num_examples
    # each batch is of shape (total_num_pred_box, 5 + num_classes)
    batch = tf.unstack(inputs)
    boxes_dicts = []
    for boxes in batch:
        # filter out the boxes that has confidence less than the confidence_threshold
        boxes = tf.boolean_mask(boxes, boxes[:, 4] > confidence_threshold)
        # for classes, get the argmax -> return a single column of class index
        classes = tf.argmax(boxes[:, 5:], axis=-1)
        classes = tf.expand_dims(tf.cast(classes, dtype=tf.float32), axis=-1)
        boxes = tf.concat([boxes[:, :5], classes], axis=-1)
        
        boxes_dict = dict()
        for cls in range(n_classes):
            # get the boxes of a specific class - returns a boolean tensor
            # with the one equal to cls as True
            mask = tf.equal(boxes[:, 5], cls)
            mask_shape = mask.get_shape()
            if mask_shape.ndims != 0:
                # get the boxes of the current class
                class_boxes = tf.boolean_mask(boxes, mask)
                boxes_coords, boxes_conf_scores, _ = tf.split(class_boxes, [4, 1, -1], axis=-1)
                boxes_conf_scores = tf.reshape(boxes_conf_scores, [-1])
                indices = tf.image.non_max_suppression(boxes_coords, boxes_conf_scores, max_output_size, iou_threshold)
                class_boxes = tf.gather(class_boxes, indices)
                boxes_dict[cls] = class_boxes[:, :5]
        
        boxes_dicts.append(boxes_dict)
    
    return boxes_dicts

def yolo_predict(inputs, n_classes, anchors, img_size, max_output_size, iou_threshold, confidence_threshold, data_format):
    
    feat1, feat2, feat3 = inputs[0], inputs[1], inputs[2]
    
    detect1 = yolo_head(feat1, n_classes, anchors, img_size, data_format)
    detect2 = yolo_head(feat2, n_classes, anchors, img_size, data_format)
    detect3 = yolo_head(feat3, n_classes, anchors, img_size, data_format)
    
    x = tf.concat([detect1, detect2, detect3])
    
    x = build_boxes(x)
    
    boxes_dicts = non_max_suppression(
        x,
        n_classes=n_classes,
        max_output_size=max_output_size,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold
    )
    
    return boxes_dicts

def load_images(img_names, model_size):
    """
    load images in a 4D array
    
    args:
        img_names: a list of image names
        model_size: the input size of the model
        data_format: a format for the array returned
        
    return:
        a 4D NumPy array
    """
    imgs = []
    
    for img_name in img_names:
        img = Image.open(img_name)
        img = img.resize(size=model_size)
        img = np.array(img, dtype=np.float32)
        img = np.expand_dims(img, axis=0)
        imgs.append(img)
        
    imgs = np.concatenate(imgs)
    
    return imgs

def load_class_names(file_name):
    """return a list of class names read from 'file_name'"""
    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

def draw_boxes(img_names, boxes_dicts, class_names, model_size):
    """
    Draws detected boxes
    
    args:
        img_names: a list of input images names
        boxes_dict: a class-to-boxes dictionary
        class_names: a class names list
        model_size: input size of model
    """
    # coco dataset has 80 classes -> a color for each class
    colors = ((np.array(color_palette("hls", 80)) * 255)).astype(np.uint8)
    for num, img_name, boxes_dict in zip(range(len(img_names)), img_names, boxes_dicts):
        img = Image.open(img_name)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font='../input/data-for-yolo-v3-kernel/futur.ttf', size=(img.size[0] + img.size[1]) // 100)
        
        resize_factor = \
        (img.size[0] / model_size[0], img.size[1] / model_size[1])
        
        for cls in range(len(class_names)):
            boxes = boxes_dict[cls]
            if np.size(boxes) != 0:
                color = colors[cls]
                for box in boxes:
                    xy, confidence = box[:4], box[4]
                    xy = [xy[i] * resize_factor[i % 2] for i in range(4)]
                    x0, y0 = xy[0], xy[1]
                    thickness = (img.size[0] + img.size[1]) // 200
                    for t in np.linspace(0, 1, thickness):
                        xy[0], xy[1] = xy[0] + t, xy[1] + t
                        xy[2], xy[3] = xy[2] - t, xy[3] - t
                        draw.rectangle(xy, outline=tuple(color))
                    text = '{} {:.1f}%'.format(class_names[cls], confidence * 100)
                    text_size = draw.textsize(text, font=font)
                    draw.rectangle(
                        [x0, y0 - text_size[1], x0 + text_size[0], y0],
                        fill=tuple(color)
                    )
                    draw.text((x0, y0 - text_size[1]), text, fill='black', font=font)
        display(img)


def load_weights(variables, file_name):
    """Reshapes and loads official pretrained Yolo weights.

    Args:
        variables: A list of tf.Variable to be assigned.
        file_name: A name of a file containing weights.

    Returns:
        A list of assign operations.
    """
    with open(file_name, "rb") as f:
        # Skip first 5 values containing irrelevant info
        np.fromfile(f, dtype=np.int32, count=5)
        weights = np.fromfile(f, dtype=np.float32)

        assign_ops = []
        ptr = 0

        # Load weights for Darknet part.
        # Each convolution layer has batch normalization.
        for i in range(52):
            conv_var = variables[5 * i]
            gamma, beta, mean, variance = variables[5 * i + 1:5 * i + 5]
            batch_norm_vars = [beta, gamma, mean, variance]

            for var in batch_norm_vars:
                shape = var.shape.as_list()
                num_params = np.prod(shape)
                var_weights = weights[ptr:ptr + num_params].reshape(shape)
                ptr += num_params
                assign_ops.append(tf.compat.v1.assign(var, var_weights))

            shape = conv_var.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(tf.compat.v1.assign(conv_var, var_weights))

        # Loading weights for Yolo part.
        # 7th, 15th and 23rd convolution layer has biases and no batch norm.
        ranges = [range(0, 6), range(6, 13), range(13, 20)]
        unnormalized = [6, 13, 20]
        for j in range(3):
            for i in ranges[j]:
                current = 52 * 5 + 5 * i + j * 2
                conv_var = variables[current]
                gamma, beta, mean, variance =  \
                    variables[current + 1:current + 5]
                batch_norm_vars = [beta, gamma, mean, variance]

                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.compat.v1.assign(var, var_weights))

                shape = conv_var.shape.as_list()
                num_params = np.prod(shape)
                var_weights = weights[ptr:ptr + num_params].reshape(
                    (shape[3], shape[2], shape[0], shape[1]))
                var_weights = np.transpose(var_weights, (2, 3, 1, 0))
                ptr += num_params
                assign_ops.append(tf.compat.v1.assign(conv_var, var_weights))

            bias = variables[52 * 5 + unnormalized[j] * 5 + j * 2 + 1]
            shape = bias.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr:ptr + num_params].reshape(shape)
            ptr += num_params
            assign_ops.append(tf.compat.v1.assign(bias, var_weights))

            conv_var = variables[52 * 5 + unnormalized[j] * 5 + j * 2]
            shape = conv_var.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(tf.compat.v1.assign(conv_var, var_weights))

    return assign_ops