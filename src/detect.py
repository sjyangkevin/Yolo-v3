from configurations.Yolo_v3_Config import *
from detection_layer.Yolo_v3 import Yolo_v3
from feature_extractor.Darknet53 import *
from utils.Yolo_v3_utils import *

if __name__ == '__main__':
    img_names = ['data/dog.jpg', 'data/office.jpg']
    
    batch_size = len(img_names)
    batch = load_images(img_names, model_size=MODEL_SIZE)
    class_names = load_class_names('data/coco.names')
    n_classes = len(class_names)
    max_output_size = 10
    iou_threshold = 0.5
    confidence_threshold = 0.5

    model = Yolo_v3(
        n_classes=n_classes,
        model_size=MODEL_SIZE,
        max_output_size=max_output_size,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold
    )

    inputs = tf.compat.v1.placeholder(tf.float32, [batch_size, 416, 416, 3])

    detections = model(inputs, training=False)

    model_vars = tf.compat.v1.global_variables(scope='yolo_v3_model')
    assign_ops = load_weights(model_vars, 'data/yolov3.weights')

    with tf.compat.v1.Session() as sess:
        sess.run(assign_ops)
        detection_result = sess.run(detections, feed_dict={inputs: batch})

    draw_boxes(img_names, detection_result, class_names, MODEL_SIZE)