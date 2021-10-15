from functools import partial
import os
from re import L
import numpy as np
import scipy.signal
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops.gen_batch_ops import batch

from model.yolo_layers import yolo_layer
from loss.loss_fn import loss_fn
from loader.dataloader import Dataset as YoloDataset
from utils.utils import get_anchors, get_classes

import matplotlib.pyplot as plt
import warnings

import config as cfg

class LossHistory(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        import datetime

        curr_time = datetime.datetime.now()
        time_str  = datetime.datetime.strftime(curr_time,'%Y_%m_%d_%H_%M_%S')
        self.log_dir  = log_dir
        self.time_str = time_str
        self.save_dir = os.path.join(self.log_dir, "loss_" + str(self.time_str))  
        self.losses   = []
        self.val_loss = []

        os.makedirs(self.save_dir)

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

        with open(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('loss')))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('val_loss')))
            f.write("\n")

        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')

        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('A Loss Curve')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_dir, "epoch_loss_" + str(self.time_str) + ".png"))

        plt.cla()
        plt.close("all")

class ExponentDecayScheduler(tf.keras.callbacks.Callback):
    def __init__(self, decay_rate, verbose=0):
        super(ExponentDecayScheduler, self).__init__()
        self.decay_rate     = decay_rate
        self.verbose        = verbose
        self.learning_rates = []
    
    def on_epoch_end(self, batch, logs=None):
        learning_rate = K.get_value(self.model.optimizer.lr) * self.decay_rate
        K.set_value(self.model.optimizer.lr, learning_rate)
        if self.verbose > 0:
            print(f'Setting learning rate to {learning_rate}.')

class ModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor                 = monitor
        self.verbose                 = verbose
        self.filepath                = filepath
        self.save_best_only          = save_best_only
        self.save_weights_only       = save_weights_only
        self.period                  = period
        self.epochs_since_last_save  = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn(f'ModelCheckpoint mode {mode} is unknown, fallback to [auto] mode.', RuntimeWarning)
            mode = 'auto'
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best       = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best       = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best       = -np.Inf
            else:
                self.monitor_op = np.less
                self.best       = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch = epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn(f'Can save best model only with {self.monitor} available, skipping.', RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f, saving model to %s' % (epoch + 1, self.monitor, self.best, current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve' % (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)

def create_train_model(model, input_shape, num_classes, anchors, anchors_mask):
    y_true = [Input(shape = (input_shape[0] // {0:32, 1:16, 2:8}[l], input_shape[1] // {0:32, 1:16, 2:8}[l], len(anchors_mask[l]), num_classes + 5)) for l in range(len(anchors_mask))]

    model_loss = Lambda(
        loss_fn, 
        output_shape=(1, ), 
        name='yolo_loss', 
        arguments={'input_shape' : input_shape, 'anchors' : anchors, 'anchors_mask' : anchors_mask, 'num_classes' : num_classes}
    )([*model.output, *y_true])
    model = Model([model.input, *y_true], model_loss)
    return model

def fit_one_epoch():
    pass

if __name__ == "__main__":

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    eager        = cfg.eager_mode
    classes_path = cfg.classes_path
    anchors_path = cfg.anchors_path
    anchors_mask = cfg.anchors_mask
    model_path   = cfg.model_weight_path
    input_shape  = cfg.input_shape

    init_epoch          = cfg.init_epoch
    freeze_epoch        = cfg.freeze_epoch
    freeze_batch_size   = cfg.freeze_batch_size
    freeze_lr           = cfg.freeze_lr

    unfreeze_epoch      = cfg.unfreeze_epoch
    unfreeze_batch_size = cfg.unfreeze_batch_size
    unfreeze_lr         = cfg.unfreeze_lr

    freeze_train        = cfg.freeze

    num_workers         = cfg.num_workers

    train_annotation_path = cfg.train_annotation_path
    valid_annotation_path = cfg.valid_annotation_path

    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors     = get_anchors(anchors_path)

    YoloModel = yolo_layer(Input((416, 416, 3)), num_classes, len(anchors_mask), cfg.train, cfg.data_format)
    if os.path.exists(model_path):
        print('Load weights {}.'.format(model_path))
        YoloModel.load_weights(model_path)

    if not eager:
        model = create_train_model(YoloModel, input_shape, num_classes, anchors, anchors_mask)
    
    logging        = TensorBoard(log_dir=cfg.log_dir)
    checkpoint     = ModelCheckpoint(
                        'logs/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                        monitor = 'val_loss',
                        save_weights_only = True, 
                        save_best_only = False, 
                        period = 1
                    )
    reduce_lr      = ExponentDecayScheduler(decay_rate=cfg.decay_rate, verbose=cfg.verbose)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=cfg.min_delta, patience=cfg.patience, verbose=cfg.verbose)
    loss_history   = LossHistory(cfg.log_dir)

    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    
    with open(valid_annotation_path) as f:
        valid_lines = f.readlines()
    
    num_train = len(train_lines)
    num_valid = len(valid_lines)

    if freeze_train:
        freeze_layers = cfg.freeze_layers
        for i in range(freeze_layers):
            YoloModel.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(YoloModel.layers)))

    batch_size  = freeze_batch_size
    lr          = freeze_lr
    start_epoch = init_epoch
    end_epoch   = freeze_epoch

    epoch_step     = num_train // batch_size
    epoch_step_val = num_valid // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError('dataset too small')
    
    train_dataloader = YoloDataset(train_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, train = True)
    valid_dataloader = YoloDataset(valid_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, train = False)
    
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_valid, batch_size))

    if eager:
        gen     = tf.data.Dataset.from_generator(partial(train_dataloader.generate), (tf.float32, tf.float32, tf.float32, tf.float32))
        gen_val = tf.data.Dataset.from_generator(partial(valid_dataloader.generate), (tf.float32, tf.float32, tf.float32, tf.float32))

        gen     = gen.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
        gen_val = gen_val.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = lr, decay_steps = epoch_step, decay_rate=cfg.decay_rate, staircase=True)

        optimizer = Adam(learning_rate = lr_schedule)

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(YoloModel, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, end_epoch, input_shape, anchors, anchors_mask, num_classes)
    else:
        model.compile(optimizer=Adam(lr = lr), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        model.fit_generator(
            generator           = train_dataloader,
            steps_per_epoch     = epoch_step,
            validation_data     = valid_dataloader,
            validation_steps    = epoch_step_val,
            epochs              = end_epoch,
            initial_epoch       = start_epoch,
            use_multiprocessing = True if num_workers != 0 else False,
            workers             = num_workers,
            callbacks           = [logging, checkpoint, reduce_lr, early_stopping, loss_history]
        )

    if freeze_train:
        for i in range(freeze_layers): 
            YoloModel.layers[i].trainable = True
    
    batch_size  = unfreeze_batch_size
    lr          = unfreeze_lr
    start_epoch = freeze_epoch
    end_epoch   = unfreeze_epoch

    epoch_step     = num_train // batch_size
    epoch_step_val = num_valid // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError('dataset too small')
    
    train_dataloader    = YoloDataset(train_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, train = True)
    valid_dataloader    = YoloDataset(valid_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, train = False)

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_valid, batch_size))

    if eager:
        gen     = tf.data.Dataset.from_generator(partial(train_dataloader.generate), (tf.float32, tf.float32, tf.float32, tf.float32))
        gen_val = tf.data.Dataset.from_generator(partial(valid_dataloader.generate), (tf.float32, tf.float32, tf.float32, tf.float32))

        gen     = gen.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
        gen_val = gen_val.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = lr, decay_steps = epoch_step, decay_rate=0.94, staircase=True)
        
        optimizer = Adam(learning_rate = lr_schedule)

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(YoloModel, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, end_epoch, input_shape, anchors, anchors_mask, num_classes)

    else:
        model.compile(optimizer=Adam(lr = lr), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        model.fit_generator(
            generator           = train_dataloader,
            steps_per_epoch     = epoch_step,
            validation_data     = valid_dataloader,
            validation_steps    = epoch_step_val,
            epochs              = end_epoch,
            initial_epoch       = start_epoch,
            use_multiprocessing = True if num_workers != 0 else False,
            workers             = num_workers,
            callbacks           = [logging, checkpoint, reduce_lr, early_stopping, loss_history]
        )