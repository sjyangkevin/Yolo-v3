import math
from random import shuffle

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from utils.utils import convert_color, normalize_image

class Dataset(Sequence):
    def __init__(self, annot, input_shape, anchors, batch_size, num_classes, anchors_mask, train):
        self.annotations = annot
        self.length =len(self.annotations)
        self.input_shape = input_shape
        self.anchors = anchors
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.anchors_mask = anchors_mask
        self.train = train
        self.max_boxes = 100

    def __len__(self):
        # number of batches in the dataset
        return math.ceil(len(self.annotations) / float(self.batch_size))
    
    def __getitem__(self, index):
        image_data = []
        box_data = []

        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            i = i % self.length
            image, box = self.get_data(self.annotations[i], self.input_shape, self.max_boxes)
            image_data.append(normalize_image(np.array(image)))
            box_data.append(box)
        
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = self.preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.num_classes)
        return [image_data, *y_true], np.zeros(self.batch_size)

    def get_data(self, annotations, input_shape, max_boxes):
        data = annotations.split()
        image = Image.open(data[0])
        image = convert_color(image)

        iw, ih = image.size
        h,  w  = input_shape

        box = np.array([np.array(list(map(int, box.split(',')))) for box in data[1:]])

        scale = min(w/iw, h/ih)
        nw    = int(iw * scale)
        nh    = int(ih * scale)
        dx    = (w - nw) // 2
        dy    = (h - nh) // 2

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w,h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image, np.float32)

        # the second dimension is: 4 box coords + 1 confidence score
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            # if the box is out of the boundary
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w]     = w
            box[:, 3][box[:, 3] > h]     = h
            
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]

            box   = box[np.logical_and(box_w > 1, box_h > 1)]

            if len(box) > max_boxes:
                box = box[:max_boxes]
            box_data[:len(box)] = box
        
        return image_data, box_data

    def generate(self):
        pass

    def on_epoch_begin(self):
        # shuffle data when a new epoch starts
        shuffle(self.annotations)

    def preprocess_true_boxes(self, true_boxes, input_shape, anchors, num_classes):
        assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'

        # coordinates of bounding box
        true_boxes  = np.array(true_boxes, dtype='float32')
        # size of images
        input_shape = np.array(input_shape, dtype='int32')

        # there are three layers because there are three scales
        num_layers = len(self.anchors_mask)

        # number of images
        m = true_boxes.shape[0]
        # [(13, 13), (26, 26), (52, 52)], down to these grid size from input size 416
        grid_shapes = [input_shape // {0: 32, 1:16, 2:8}[l] for l in range(num_layers)]
        # y_true is a list of the following array
        # (m, 13, 13, 3, 4 + 1 + num_classes)
        # (m, 26, 26, 3, 4 + 1 + num_classes)
        # (m, 52, 52, 3, 4 + 1 + num_classes)
        y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(self.anchors_mask[l]), 5 + num_classes), dtype='float32') for l in range(num_layers)]

        # box center coordinate, of shape (m, n, 2) where n is number of boxes
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        # box width/height, of shape (m, n, 2) where n is number of boxes
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

        # normalize boxes coordinate and width height
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

        # shape from (9, 2) -> (1, 9, 2)
        anchors = np.expand_dims(anchors, 0)
        anchors_maxes = anchors / 2.
        anchors_mins = -anchors_maxes

        # for a box to be valid, its width and height should be greater than 0
        valid_mask = boxes_wh[..., 0] > 0

        # for each image
        for b in range(m):
            # get rid of the image with no valid boxes
            wh = boxes_wh[b, valid_mask[b]]
            if len(wh) == 0:
                continue
            
            wh = np.expand_dims(wh, -2) # shape from (n, 2) to (n, 1, 2)
            box_maxes = wh / 2.
            box_mins = -box_maxes

            #############################
            # calculate the IoU among all the anchors and boxes
            # intersect_area -> (n, 9)
            # box_area       -> (n, 1)
            # anchor_area    -> (1, 9)
            # iou            -> (n, 9) means for a box, the IoU with 9 anchors
            #############################
            intersect_mins  = np.maximum(box_mins, anchors_mins)
            intersect_maxes = np.minimum(box_maxes, anchors_maxes)
            intersect_wh    = np.maximum(intersect_maxes, intersect_mins, 0.)
            intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]

            box_area    = wh[..., 0] * wh[..., 1]
            anchor_area = anchors[..., 0] * anchors[..., 1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)

            # it will be an array of index (0, 1, 2, ..., 8), each is an anchor
            best_anchor = np.argmax(iou, axis=-1) # shape of (n, )

            for t, n in enumerate(best_anchor):
                # find the corresponding grid of every anchor
                for l in range(num_layers):
                    if n in self.anchors_mask[l]:
                        # the coordinate of the grid cell that responsible for the prediction
                        i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                        j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                        # the index to the scale, we have 3 scale (13, 13), (26, 26), (52, 52)
                        k = self.anchors_mask[l].index(n)
                        c = true_boxes[b, t, 4].astype('int32') # the class label
                        # coordinates
                        y_true[l][b, j, i, k, 0:4]   = true_boxes[b, t, 0:4]
                        # confidence score
                        y_true[l][b, j, i, k, 4]     = 1
                        # class one-hot encode vector
                        y_true[l][b, j, i, k, 5 + c] = 1
        
        return y_true

