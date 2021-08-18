import os
import numpy as np
import tensorflow as tf
from PIL import Image
from Yolov3.utils import load_anchors, load_class_names, box_iou
from Yolov3.config import *

class Dataset(object):
    def __init__(self, dataset_type):
        self.annot_path       = TRAIN_ANNOTATION_FILE if dataset_type == 'train' else TEST_ANNOTATION_FILE
        self.input_size       = MODEL_INPUT_SIZE
        self.batch_size       = BATCH_SIZE
        self.data_aug         = True if dataset_type == 'train' else False
        
        self.classes          = load_class_names(CLASS_NAME_FILE)
        self.num_class        = len(self.classes)
        self.anchors          = load_anchors(ANCHOR_FILE)
        self.anchor_per_scale = len(self.anchors) // 3
        self.max_output_size  = MAX_OUTPUT_SIZE
        self.strides          = np.array(STRIDES)
        self.anchors          = np.reshape(self.anchors, (self.anchor_per_scale, len(self.strides), 2))
        self.anchors_mask     = [2, 1, 0]

        self.annotations      = self.load_annotations(dataset_type)
        self.num_samples      = len(self.annotations)
        self.num_batches      = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count      = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        with tf.device('/cpu:0'):
            self.train_input_size = self.input_size
            self.train_output_sizes = self.train_input_size // self.strides

            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3), dtype=np.float32)

            batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2], self.anchor_per_scale, 5 + self.num_class), dtype=np.float32)
            batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1], self.anchor_per_scale, 5 + self.num_class), dtype=np.float32)
            batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0], self.anchor_per_scale, 5 + self.num_class), dtype=np.float32)

            batch_sbboxes = np.zeros((self.batch_size, self.max_output_size, 4), dtype=np.float32)
            batch_mbboxes = np.zeros((self.batch_size, self.max_output_size, 4), dtype=np.float32)
            batch_lbboxes = np.zeros((self.batch_size, self.max_output_size, 4), dtype=np.float32)

            num = 0
            if self.batch_count < self.num_batches:
                print(self.batch_count * self.batch_size)
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples:
                        index -= self.num_samples
                    annotation = self.annotations[index]
                    image, bboxes = self.parse_annotation(annotation)

                    label_lbbox, label_mbbox, label_sbbox, lbboxes, mbboxes, sbboxes = self.preprocess_true_boxes(bboxes)

                    batch_image[num, :, :, :] = image
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_lbboxes[num, :, :] = lbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_sbboxes[num, :, :] = sbboxes

                    num += 1

                self.batch_count += 1
                batch_larger_target  = batch_label_lbbox, batch_lbboxes
                batch_medium_target  = batch_label_mbbox, batch_mbboxes
                batch_smaller_target = batch_label_sbbox, batch_sbboxes

                return batch_image, (batch_larger_target, batch_medium_target, batch_smaller_target)
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def load_annotations(self, dataset_type):
        parsed_annotations = []
        with open(self.annot_path, 'r') as f:
            txt = f.read().splitlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)

        for annotation in annotations:
            # fully parse annotations
            line = annotation.split()
            image_path, index = "", 1
            for i, one_line in enumerate(line):
                if not one_line.replace(",","").isnumeric():
                    if image_path != "": image_path += " "
                    image_path += one_line
                else:
                    index = i
                    break
            if not os.path.exists(image_path):
                raise KeyError("%s does not exist ... " %image_path)
            parsed_annotations.append([image_path, line[index:]])
        return parsed_annotations

    def parse_annotation(self, annotation):
        image_path = annotation[0]
        image = Image.open(image_path)

        bboxes = np.array([list(map(int, box.split(','))) for box in annotation[1]])

        # data augmentation
        if self.data_aug:
            pass

        image, bboxes = self.image_preprocess(image, [self.input_size, self.input_size], np.copy(bboxes))
        return image, bboxes

    def image_preprocess(self, image, input_size, gt_boxes=None):
        iw, ih = image.size
        h,  w  = input_size

        scale  = min(w / iw, h / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        dx, dy = (w - nw) // 2, (h - nh) // 2

        image = image.resize((nw, nh), Image.BICUBIC)
        image_paded = Image.new('RGB', (w, h), (128, 128, 128))
        image_paded.paste(image, (dx, dy))
        image_paded = np.array(image_paded, np.float32)
 
        image_paded /= 255.

        if gt_boxes is None:
            return image_paded

        else:
            gt_boxes[:, [0,2]] = gt_boxes[:, [0,2]]*nw/iw + dx
            gt_boxes[:, [1,3]] = gt_boxes[:, [1,3]]*nh/ih + dy
            return image_paded, gt_boxes

    def preprocess_true_boxes(self, bboxes):
        # calculate the IOU between true boxes and anchors, use it as predictor
        # reshape y_true to a list 
        # (m, 13, 13, 3, 85)
        # (m, 26, 26, 3, 85)
        # (m, 52, 52, 3, 85)
        NUM_ROUTES = len(self.strides)

        label = [np.zeros(
            (
                self.train_output_sizes[i], 
                self.train_output_sizes[i], 
                self.anchor_per_scale,
                5 + self.num_class
                )
            ) for i in range(NUM_ROUTES)
        ]

        bboxes_xywh = [np.zeros((self.max_output_size, 4)) for _ in range(NUM_ROUTES)]
        bbox_count = np.zeros((NUM_ROUTES,))

        for bbox in bboxes:
            bbox_coor                = bbox[:4]
            bbox_class_ind           = bbox[4]
            bbox_cls                 = np.zeros(self.num_class, dtype=np.float)
            bbox_cls[bbox_class_ind] = 1.0 
            
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            j = NUM_ROUTES - 1
            for i in range(NUM_ROUTES):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                # TODO: Fix this reverse order (corresponding to boxes)
                # anchors is (52,52) (26,26) (13,13)
                # boxes   is (13,13) (26,26) (52,52)
                # DONE
                anchors_xywh[:, 2:4] = self.anchors[j - i]

                iou_scale = box_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
            
            best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
            best_detect = int(best_anchor_ind / self.anchor_per_scale)
            best_anchor = int(best_anchor_ind % self.anchor_per_scale)
            xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

            label[best_detect][yind, xind, best_anchor, :] = 0
            label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
            label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
            label[best_detect][yind, xind, best_anchor, 5:] = bbox_cls

            bbox_ind = int(bbox_count[best_detect] % self.max_output_size)
            bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
            bbox_count[best_detect] += 1

        label_lbbox, label_mbbox, label_sbbox = label
        lbboxes, mbboxes, sbboxes = bboxes_xywh
        return label_lbbox, label_mbbox, label_sbbox, lbboxes, mbboxes, sbboxes





