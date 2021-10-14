import tensorflow as tf
from tensorflow.keras import backend as K
from loss.utils import box_iou, decode

def loss_fn(args, input_shape, anchors, anchors_mask, num_classes, ignore_thresh=0.5, print_loss=False):

    num_layers = len(anchors_mask)

    # args is [*model.output, *y_true]
    #   y_true:
    #   (m,13,13,3,85)
    #   (m,26,26,3,85)
    #   (m,52,52,3,85)

    #   yolo_outputs:
    #   (m,13,13,3,85)
    #   (m,26,26,3,85)
    #   (m,52,52,3,85)
    y_true      = args[num_layers:]
    yolo_outputs = args[:num_layers]

    # input shape and cast it to integer
    input_shape = K.cast(input_shape, K.dtype(y_true[0]))
    # [13, 13], [26, 26], [52, 52]
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]

    # batch size
    m = K.shape(yolo_outputs[0])[0]

    loss    = 0
    num_pos = 0

    # for every scale
    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5] # confidence score, 1 if the grid cell has an object
        true_class_probs = y_true[l][..., 5:] # ground-truth class prob, 1 if it is an instance of a class

        # if running on scale (13, 13)
        # grid: (13, 13, 1, 2) the coordinate of the grid
        # raw_pred: (m, 13, 13, 3, 85) the before-processed prediction
        # pred_xy:  (m, 13, 13, 3, 2) the center coordinate after decode the raw_pred
        # pred_wh:  (m, 13, 13, 3, 2) the width and height after decode the raw_pred
        grid, raw_pred, pred_xy, pred_wh = decode(yolo_outputs[l], anchors[anchors_mask[l]], num_classes, input_shape, calc_loss=True)

        # of shape (m, 13, 13, 3, 4)
        pred_box = K.concatenate([pred_xy, pred_wh])

        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):
            # return the box that object_mask_bool is true, the box with object, of shape (n, 4)
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])

            # calculate the IoU among all predicted boxes and true boxes 
            # pred_box: (13, 13, 3, 4)
            # true_box: (n, 4)
            # iou     : (13, 13, 3, n) - the IoU of every predicted box with all the true boxes
            iou = box_iou(pred_box[b], true_box)

            # of shape (13, 13, 3) - the IoU score of each predicted box with all the true boxes
            best_iou = K.max(iou, axis=-1)

            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b + 1, ignore_mask

        _, ignore_mask = tf.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])

        # of shape (m, 13, 13 ,3)
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1) # of shape (m, 13, 13, 3, 1)

        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][:] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchors_mask[l]] * input_shape[::-1])

        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))

        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])

        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) + \
            (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask

        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

        xy_loss         = K.sum(xy_loss)
        wh_loss         = K.sum(wh_loss)
        confidence_loss = K.sum(confidence_loss)
        class_loss      = K.sum(class_loss)

        num_pos += tf.maximum(K.sum(K.cast(object_mask, tf.float32)), 1)
        loss    += xy_loss + wh_loss + confidence_loss + class_loss
    
    loss = loss / num_pos
    return loss

