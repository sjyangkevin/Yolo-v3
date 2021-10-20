import tensorflow as tf
from tensorflow.keras import backend as K
from utils.utils import get_anchors_and_decode, box_iou

def loss_fn(args, input_shape, anchors, anchors_mask, num_classes, ignore_thresh=.5):

    num_layers = len(anchors_mask)

    # both y_true and yolo_outputs are a list of array, each one with shape of
    # [m, 13, 13, 3, 85]
    # [m, 26, 26, 3, 85]
    # [m, 52, 52, 3, 85]
    y_true       = args[num_layers:]
    yolo_outputs = args[:num_layers]

    # [416, 416], the model input shape
    input_shape = K.cast(input_shape, K.dtype(y_true[0]))
    # a list of array with shape of [13, 13], [26, 26], [52, 52]
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]

    # batch size
    m = K.shape(yolo_outputs[0])[0]

    loss    = 0
    num_pos = 0

    # iterate over 3 scales (13, 13), (26, 26), (52, 52)
    for l in range(num_layers):
        # for scale of (13, 13)
        # object_mask' shape [m, 13, 13, 3, 1]
        object_mask      = y_true[l][..., 4:5] # where object is presence
        # object_mask' shape [m, 13, 13, 3, num_classes]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = get_anchors_and_decode(yolo_outputs[l], anchors[anchors_mask[l]], num_classes, input_shape, calc_loss=True)

        # pred_box's shape: [m, 13, 13, 3, 4]
        pred_box = K.concatenate([pred_xy, pred_wh])

        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):
            # true_box's shape: [n, 4]
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            # pred_box[b]'s shape: [13, 13, 3, 4]
            iou      = box_iou(pred_box[b], true_box)
            # best_iou's shape: [13, 13, 3]
            best_iou = K.max(iou, axis=-1)

            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b + 1, ignore_mask
        
        _, ignore_mask = tf.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])

        ignore_mask = ignore_mask.stack()
        # from (m, 13, 13, 3) to (m,13,13,3,1)
        # each element is a bool indicates that whether the anchor's iou less than threshold
        ignore_mask = K.expand_dims(ignore_mask, -1)

        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][:] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchors_mask[l]] * input_shape[::-1])

        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))

        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])

        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) + \
            (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

        xy_loss         = K.sum(xy_loss)
        wh_loss         = K.sum(wh_loss)
        confidence_loss = K.sum(confidence_loss)
        class_loss      = K.sum(class_loss)

        num_pos += tf.maximum(K.sum(K.cast(object_mask, tf.float64)), 1)
        loss    += xy_loss + wh_loss + confidence_loss + class_loss

    loss = loss / num_pos
    return loss        




