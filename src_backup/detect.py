import tensorflow as tf
from Yolov3.config import *
from Yolov3.utils import *
from Yolov3.yolo import create_model, predict

if __name__ == "__main__":

    class_names = load_class_names(CLASS_NAME_FILE)
    n_classes   = len(class_names)
    anchors     = load_anchors(ANCHOR_FILE)
    # number of anchors per grid cell
    n_anchors   = len(anchors) // 3
    image_names = [os.path.join(INPUT_IMAGE_DIR, img) for img in os.listdir(INPUT_IMAGE_DIR)]
    image_names.append('test.png')

    model = create_model(
        n_classes, 
        n_anchors, 
        MODEL_INPUT_SIZE,
        training=False,
        data_format=DATA_FORMAT
    )

    model.load_weights(MODEL_WEIGHTS_DIR)

    images = load_images(image_names, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))

    images /= 255.

    preds = model(images)

    boxes = predict(
        preds,
        n_classes,
        anchors,
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        MAX_OUTPUT_SIZE,
        IOU_THRESHOLD,
        CONFIDENCE_THRESHOLD,
        DATA_FORMAT
    )

    draw_boxes(
        image_names,
        boxes,
        class_names,
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    )