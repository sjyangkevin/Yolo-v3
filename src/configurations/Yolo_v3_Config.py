BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-05
LEAKY_RELU_ALPHA = 0.1
# bounding box priors
ANCHORS = [
    (10, 13), (16, 30), (33, 23),
    (30, 61), (62, 45), (59, 119),
    (116, 90), (156, 198), (373, 326)
]
# input size of model
MODEL_SIZE = (416, 416)