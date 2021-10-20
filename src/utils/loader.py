import math
from random import shuffle

from PIL import Image
import numpy as np
from tensorflow.keras.utils import Sequence
from utils.utils import convert_color, preprocess_input
# config from root directory
import config as cfg

class YoloDatasets(Sequence):
    def __init__(self, annotation_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, train):
        self.annotation_lines   = annotation_lines
        self.length             = len(self.annotation_lines)
        
        self.input_shape        = input_shape
        self.anchors            = anchors
        self.batch_size         = batch_size
        self.num_classes        = num_classes
        self.anchors_mask       = anchors_mask
        self.train              = train
    
    def __len__(self):
        return math.ceil(len(self.annotation_lines) / float(self.batch_size))

    def __getitem__(self, index):
        image_data = []
        box_data   = []
        # generate a batch of (image, box) with given index
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            i = i % self.length
            image, box = self.get_data(self.annotation_lines[i], self.input_shape)
            image_data.append(preprocess_input(np.array(image)))
            box_data.append(box)
        
        # (m, model_input_size, model_input_size, 3)
        image_data = np.array(image_data)
        # (m, num_boxes, 5)
        box_data   = np.array(box_data)
        #[(m, 13, 13, 3, 85), (m, 26, 26, 3, 85), (m, 52, 52, 3, 85)]
        # if on COCO dataset with 3 anchors
        y_true     = self.preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.num_classes)
        return [image_data, *y_true], np.zeros(self.batch_size)

    def generate(self):
        # for eager mode, same as __getitem__(self, index)
        i = 0
        while True:
            image_data = []
            box_data   = []
            for b in range(self.batch_size):
                if i == 0:
                    np.random.shuffle(self.annotation_lines)
                image, box = self.get_data(self.annotation_lines[i], self.input_shape)
                i          = (i + 1) % self.length
                image_data.append(preprocess_input(np.array(image)))
                box_data.append(box)
            image_data = np.array(image_data)
            box_data   = np.array(box_data)
            y_true     = self.preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.num_classes)
            yield image_data, y_true[0], y_true[1], y_true[2]

    def on_epoch_begin(self):
        shuffle(self.annotation_lines)

    def get_data(self, annotation_line, input_shape, max_boxes=cfg.max_boxes):
        # extract the image path, and create an PIL image from annotation
        line  = annotation_line.split()
        image = Image.open(line[0])
        image = convert_color(image)
        # get the image's size (any, any) and model's input size (416, 416)
        iw, ih = image.size
        h, w   = input_shape
        # extract bounding box coordinates from annotation, and make it an array
        box    = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        # these variables are used to resize the image to model input size
        scale = min(w / iw, h / ih)
        nw    = int(iw * scale)
        nh    = int(ih * scale)
        dx    = (w - nw) // 2
        dy    = (h - nh) // 2
        # create an new image with model's input size, and preserve the image
        # shape (no distortion) after resizing, and pad it with value 128
        image      = image.resize((nw, nh), Image.BICUBIC)
        new_image  = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image, np.float64)
        # convert box coordinate to fit model's input size, and project it to
        # the correct position after resizing the image
        box_data   = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]] * nw / iw + dx
            box[:, [1,3]] = box[:, [1,3]] * nh / ih + dy
            # set the bounding box to image's boundary if the box is out
            box[:, 0:2][box[:, 0:2]<0]  = 0
            box[:, 2][box[:, 2]>w]      = w
            box[:, 3][box[:, 3]>h]      = h
            box_w   = box[:, 2] - box[:, 0]
            box_h   = box[:, 3] - box[:, 1]
            # if width and height is less than 1, remove it, and treat it invalid
            box     = box[np.logical_and(box_w>1, box_h>1)]
            if len(box)>max_boxes: 
                box = box[:max_boxes]
            # take the first max boxes
            box_data[:len(box)] = box
        
        return image_data, box_data

    def preprocess_true_boxes(self, true_boxes, input_shape, anchors, num_classes):
        assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'

        true_boxes  = np.array(true_boxes, dtype='float64')
        input_shape = np.array(input_shape, dtype='int32')

        num_layers  = len(self.anchors_mask)

        # batch size
        m           = true_boxes.shape[0]
        grid_shapes = [input_shape // {0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    
        # zero-initialized array to store true boxes for different scales
        y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(self.anchors_mask[l]), 5 + num_classes),
                    dtype='float64') for l in range(num_layers)]
        
        # shape is (m, n, 2) - batch_size, num_boxes, 2
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh =  true_boxes[..., 2:4] - true_boxes[..., 0:2]

        # normalize box coordinate
        # input_shape[::-1] reverse width and height, since input shape if height, width
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

        # anchors from shape [9, 2] to [1, 9, 2]
        anchors         = np.expand_dims(anchors, 0)
        anchor_maxes    = anchors / 2.
        anchor_mins     = -anchor_maxes

        valid_mask = boxes_wh[..., 0] > 0

        for b in range(m): # for every box in batch
            wh = boxes_wh[b, valid_mask[b]]
            if len(wh) == 0:
                continue
            
            wh = np.expand_dims(wh, -2) # from shape [n, 2] to [n, 1, 2]
            box_maxes = wh / 2.
            box_mins  = - box_maxes

            intersect_mins  = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh    = np.maximum(intersect_maxes - intersect_mins, 0.)
            # shape of [n, 9]
            intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]
            # shape of [n, 1]
            box_area    = wh[..., 0] * wh[..., 1]
            # shape of [1, 9]
            anchor_area = anchors[..., 0] * anchors[..., 1]
            # shape of [n, 9]
            iou = intersect_area / (box_area + anchor_area - intersect_area)
            # shape of [n,] each element is the index to the anchor has max iou
            best_anchor = np.argmax(iou, axis=-1)

            for t, n in enumerate(best_anchor):
                for l in range(num_layers):
                    if n in self.anchors_mask[l]:
                        # the cell position in the grid, which cell is used to predict
                        i = np.floor(true_boxes[b,t,0] * grid_shapes[l][1]).astype('int32')
                        j = np.floor(true_boxes[b,t,1] * grid_shapes[l][0]).astype('int32')

                        # which anchor
                        k = self.anchors_mask[l].index(n)
                        c = true_boxes[b, t, 4].astype('int32')

                        y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                        y_true[l][b, j, i, k, 4] = 1
                        y_true[l][b, j, i, k, 5+c] = 1
            
            return y_true








