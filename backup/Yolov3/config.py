INPUT_IMAGE_DIR       = "../ODLib/data/Yolo_v3/inputs" 
OUTPUT_DIR            = "outputs"
FONT_DIR              = "../ODLib/data/futur.ttf"
CLASS_NAME_FILE       = "../ODLib/data/coco.names"
ANCHOR_FILE           = "../ODLib/data/Yolo_v3/anchors.txt"
TRAIN_ANNOTATION_FILE = "2012_train.txt"
TEST_ANNOTATION_FILE  = "2012_val.txt"
MODEL_WEIGHTS_DIR     = "../yolov3_weights.h5"


MODEL_INPUT_SIZE      = 416
BATCH_SIZE            = 4
STRIDES               = [32, 16, 8]
DATA_FORMAT           = "channels_last"
BATCH_NORM_DECAY      = 0.9
BATCH_NORM_EPSILON    = 1e-05
LEAKY_RELU_ALPHA      = 0.1
MAX_OUTPUT_SIZE       = 100
IOU_THRESHOLD         = 0.5
CONFIDENCE_THRESHOLD  = 0.5


TRAIN_LOG_DIR         = "checkpoints"