import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.python import eager
from tensorflow.python.keras.backend import zeros
from tensorflow.python.keras.engine.training import Model

from model.yolov3 import yolo_body, get_train_model
from utils.loss import loss_fn
from utils.loader import YoloDatasets
from utils.utils import get_anchors, get_classes
from utils.callbacks import ExponentDecayScheduler, LossHistory, ModelCheckpoint

import os

if __name__ == "__main__":
    eager_exec          = False
    
    train_annotation_path = ""
    valid_annotation_path = ""

    classes_path        = "data/voc/voc_classes.txt"
    anchors_path        = "data/yolov3/anchors.txt"
    anchors_mask        = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    weights_path        = "data/yolov3/yolo_weights.h5"
    logging_dir         = "data/logs"
    input_shape         = [416, 416]
    num_channels        = 3
    data_format         = "channels_last"

    init_epoch          = 0
    freeze_epoch        = 50
    freeze_batch_size   = 8
    freeze_lr           = 1e-3

    unfrezze_epoch      = 100
    unfreeze_batch_size = 4
    unfreeze_lr         = 1e-4
    freeze_train        = True
    freeze_layers       = 184

    num_workers         = 0

    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors     = get_anchors(anchors_path)

    model_body = yolo_body(
        (input_shape[0], input_shape[1], num_channels), 
        num_classes, 
        len(anchors_mask), 
        training=True, 
        data_format=data_format
    )

    if weights_path != "":
        print("Load weights from: {}",format(weights_path))
        # normally, it will ignore [58, 66, 74], these three are prediction layer
        # shape depends on the task being run
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
    
    if not eager_exec:
        model = get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask)

    logging = TensorBoard(log_dir = logging_dir)
    checkpoint = ModelCheckpoint(
        os.path.join(logging_dir, "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"),
        monitor           = "val_loss",
        save_weights_only = True,
        save_best_only    = False, 
        period            = 1
    )
    reduce_lr       = ExponentDecayScheduler(decay_rate = 0.94, verbose = 1)
    early_stopping  = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 10, verbose = 1)
    loss_history    = LossHistory(logging_dir)

    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(valid_annotation_path) as f:
        valid_lines = f.readlines()

    num_train   = len(train_lines)
    num_valid   = len(valid_lines)

    if freeze_train:
        for i in range(freeze_layers):
            model_body.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))

    batch_size  = freeze_batch_size
    lr          = freeze_lr
    start_epoch = init_epoch
    end_epoch   = freeze_epoch

    epoch_step     = num_train // batch_size
    epoch_step_val = num_valid // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError('Cannot train due to dataset is too small')
    
    train_dataloader = YoloDatasets(train_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, train = True)
    valid_dataloader = YoloDatasets(valid_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, train = False)

    print('(Freeze Phase): Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_valid, batch_size))

    model.compile(optimizer=Adam(lr = lr), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

    model.fit_generator(
        generator           = train_dataloader,
        steps_per_epoch     = epoch_step,
        validation_data     = valid_dataloader,
        validation_steps    = epoch_step_val,
        epochs              = end_epoch,
        initial_epoch       = start_epoch,
        use_multiprocessing = True if num_workers > 0 else False,
        workers             = num_workers,
        callbacks           = [logging, checkpoint, reduce_lr, early_stopping, loss_history]
    )

    if freeze_train:
        for i in range(freeze_layers): model_body.layers[i].trainable = True

    batch_size     = unfreeze_batch_size
    lr             = unfreeze_lr
    start_epoch    = freeze_epoch
    end_epoch      = unfrezze_epoch

    epoch_step     = num_train // batch_size
    epoch_step_val = num_valid // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError('Cannot train due to dataset is too small')
    
    train_dataloader    = YoloDatasets(train_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, train = True)
    valid_dataloader    = YoloDatasets(valid_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, train = False)

    print('(Unfreeze Phase): Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_valid, batch_size))

    model.compile(optimizer=Adam(lr = lr), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

    model.fit_generator(
        generator           = train_dataloader,
        steps_per_epoch     = epoch_step,
        validation_data     = valid_dataloader,
        validation_steps    = epoch_step_val,
        epochs              = end_epoch,
        initial_epoch       = start_epoch,
        use_multiprocessing = True if num_workers > 0 else False,
        workers             = num_workers,
        callbacks           = [logging, checkpoint, reduce_lr, early_stopping, loss_history]
    )