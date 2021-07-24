import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display
from seaborn import color_palette
import cv2

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
        font = ImageFont.truetype(font='data/futur.ttf', size=(img.size[0] + img.size[1]) // 100)
        
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