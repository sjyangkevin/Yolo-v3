import tensorflow as tf
from tensorflow.keras import backend as K

def box_iou(b1, b2):
    # shape of (num_anchors, 1, 4)
    b1    = tf.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins  = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # shape of (1, n, 4)
    b2    = tf.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins  = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins  = tf.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou     = intersect_area / (b1_area + b2_area - intersect_area)

    return iou

def decode(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)
    grid_shape  = K.shape(feats)[1:3]

    # grid of shape (grid_size, grid_size, num_anchors, 2)
    grid_x  = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, num_anchors, 1])
    grid_y  = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], num_anchors, 1])
    grid    = K.cast(K.concatenate([grid_x, grid_y]), K.dtype(feats))

    # anchors of shape (grid_size, grid_size, num_anchors, 2)
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, num_anchors, 2])
    anchors_tensor = K.tile(anchors_tensor, [grid_shape[0], grid_shape[1], 1, 1])

    # feats (predicted results) to shape (m, grid_size, grid_size, num_anchors, num_classes + 5)
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))

    box_confidence  = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss:
        return grid, feats, box_xy, box_wh
    
    return box_xy, box_wh, box_confidence, box_class_probs

# if __name__ == "__main__":
#     # from: https://github.com/bubbliiiing/yolo3-tf2/blob/main/utils/utils_bbox.py
#     import matplotlib.pyplot as plt
#     import numpy as np

#     def sigmoid(x):
#         s = 1 / (1 + np.exp(-x))
#         return s
#     #---------------------------------------------------#
#     #   将预测值的每个特征层调成真实值
#     #---------------------------------------------------#
#     def decode(feats, anchors, num_classes):
#         # feats     [batch_size, 13, 13, 3 * (5 + num_classes)]
#         # anchors   [3, 2]
#         # num_classes 
#         # 3
#         num_anchors = len(anchors)
#         #------------------------------------------#
#         #   grid_shape指的是特征层的高和宽
#         #   grid_shape [13, 13] 
#         #------------------------------------------#
#         grid_shape = np.shape(feats)[1:3]
#         #--------------------------------------------------------------------#
#         #   获得各个特征点的坐标信息。生成的shape为(13, 13, num_anchors, 2)
#         #   grid_x [13, 13, 3, 1]
#         #   grid_y [13, 13, 3, 1]
#         #   grid   [13, 13, 3, 2]
#         #--------------------------------------------------------------------#
#         grid_x  = np.tile(np.reshape(np.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, num_anchors, 1])
#         grid_y  = np.tile(np.reshape(np.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], num_anchors, 1])
#         grid    = np.concatenate([grid_x, grid_y], -1)
#         #---------------------------------------------------------------#
#         #   将先验框进行拓展，生成的shape为(13, 13, num_anchors, 2)
#         #   [1, 1, 3, 2]
#         #   [13, 13, 3, 2]
#         #---------------------------------------------------------------#
#         anchors_tensor = np.reshape(anchors, [1, 1, num_anchors, 2])
#         anchors_tensor = np.tile(anchors_tensor, [grid_shape[0], grid_shape[1], 1, 1]) 

#         #---------------------------------------------------#
#         #   将预测结果调整成(batch_size,13,13,3,85)
#         #   85可拆分成4 + 1 + 80
#         #   4代表的是中心宽高的调整参数
#         #   1代表的是框的置信度
#         #   80代表的是种类的置信度
#         #   [batch_size, 13, 13, 3 * (5 + num_classes)]
#         #   [batch_size, 13, 13, 3, 5 + num_classes]
#         #---------------------------------------------------#
#         feats           = np.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
#         #------------------------------------------#
#         #   对先验框进行解码，并进行归一化
#         #------------------------------------------#
#         box_xy          = sigmoid(feats[..., :2]) + grid
#         box_wh          = np.exp(feats[..., 2:4]) * anchors_tensor
#         #------------------------------------------#
#         #   获得预测框的置信度
#         #------------------------------------------#
#         box_confidence  = sigmoid(feats[..., 4:5])
#         box_class_probs = sigmoid(feats[..., 5:])

#         box_wh = box_wh / 32
#         anchors_tensor = anchors_tensor / 32
#         fig = plt.figure()
#         ax = fig.add_subplot(121)
#         plt.ylim(-2,15)
#         plt.xlim(-2,15)
#         plt.scatter(grid_x,grid_y)
#         plt.scatter(5,5,c='black')
#         plt.gca().invert_yaxis()


#         anchor_left = grid_x - anchors_tensor/2 
#         anchor_top = grid_y - anchors_tensor/2 
#         print(np.shape(anchors_tensor))
#         print(np.shape(box_xy))
#         rect1 = plt.Rectangle([anchor_left[5,5,0,0],anchor_top[5,5,0,1]],anchors_tensor[0,0,0,0],anchors_tensor[0,0,0,1],color="r",fill=False)
#         rect2 = plt.Rectangle([anchor_left[5,5,1,0],anchor_top[5,5,1,1]],anchors_tensor[0,0,1,0],anchors_tensor[0,0,1,1],color="r",fill=False)
#         rect3 = plt.Rectangle([anchor_left[5,5,2,0],anchor_top[5,5,2,1]],anchors_tensor[0,0,2,0],anchors_tensor[0,0,2,1],color="r",fill=False)

#         ax.add_patch(rect1)
#         ax.add_patch(rect2)
#         ax.add_patch(rect3)

#         ax = fig.add_subplot(122)
#         plt.ylim(-2,15)
#         plt.xlim(-2,15)
#         plt.scatter(grid_x,grid_y)
#         plt.scatter(5,5,c='black')
#         plt.scatter(box_xy[0,5,5,:,0],box_xy[0,5,5,:,1],c='r')
#         plt.gca().invert_yaxis()

#         pre_left = box_xy[...,0] - box_wh[...,0]/2 
#         pre_top = box_xy[...,1] - box_wh[...,1]/2 

#         rect1 = plt.Rectangle([pre_left[0,5,5,0],pre_top[0,5,5,0]],box_wh[0,5,5,0,0],box_wh[0,5,5,0,1],color="r",fill=False)
#         rect2 = plt.Rectangle([pre_left[0,5,5,1],pre_top[0,5,5,1]],box_wh[0,5,5,1,0],box_wh[0,5,5,1,1],color="r",fill=False)
#         rect3 = plt.Rectangle([pre_left[0,5,5,2],pre_top[0,5,5,2]],box_wh[0,5,5,2,0],box_wh[0,5,5,2,1],color="r",fill=False)

#         ax.add_patch(rect1)
#         ax.add_patch(rect2)
#         ax.add_patch(rect3)

#         plt.show()
#         #
#     feat = np.random.normal(0,0.5,[4,13,13,75])
#     anchors = [[142, 110],[192, 243],[459, 401]]
#     decode(feat,anchors,20)