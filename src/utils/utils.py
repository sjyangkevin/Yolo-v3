import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow.python.keras.backend import dtype

def get_classes(filepath):
    with open(filepath, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def get_anchors(filepath):
    with open(filepath, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(anchor) for anchor in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)

def normalize_image(image):
    return image / 255.0

def convert_color(image):
    """
    convert images of all other formats to RGB
    """
    if len(np.shape(image)) == 3 and np.shape(image)[-2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

def resize_image(image, size, pad):
    # for PIL image
    iw, ih = image.size
    w, h   = size # new size
    if pad:
        # prevent image distorsion by adding some padding grey bar
        scale = min(w / iw, h / ih)
        nw    = int(iw * scale)
        nh    = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


# if __name__ == "__main__":
    # class_name_file = "src/data/coco/coco_classes.txt"
    # anchor_file = "src/data/common/model/anchors.txt"

    # CLASS_NAMES, NUM_CLASSES = get_classes(class_name_file)
    # ANCHORS, NUM_ANCHORS = get_anchors(anchor_file)

    # print(NUM_CLASSES, "\n", CLASS_NAMES)
    # print(NUM_ANCHORS, "\n", ANCHORS)