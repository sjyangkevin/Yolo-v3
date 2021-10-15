###########################################################
# Model Settings
###########################################################
eager_mode            = False

classes_path          = "src/data/coco/coco_classes.txt"
anchors_path          = "src/data/common/model/anchors.txt"

train_annotation_path = ""
valid_annotation_path = ""

model_weight_path     = "src/data/weights/yolo_weights.h5"

anchors_mask          = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
input_shape           = [416, 416]
data_format           = "channels_last"


###########################################################
# Train Settings
###########################################################
init_epoch            = 0
freeze_epoch          = 50
freeze_batch_size     = 8
freeze_lr             = 1e-3
freeze_layers         = 184

unfreeze_epoch        = 100
unfreeze_batch_size   = 4
unfreeze_lr           = 1e-4

decay_rate            = 0.94
verbose               = 1
min_delta             = 0
patience              = 10

freeze                = True
num_workers           = 0

train                 = True

log_dir               = "src/data/logs"
