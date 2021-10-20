import time
import numpy as np
import tensorflow as tf
from PIL import Image
from YOLOv3 import YOLO

if __name__ == "__main__":
    
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    model = YOLO()

    mode  = "predict"

    if mode == "predict":
        while True:
            img = input('Input image file name, or input "exit" to exit: ')
            if img == "exit":
                break

            try:
                image = Image.open(img)
            except:
                print('Open Error! Try Again!')
                continue
            else:
                result = model.detect_image(image)
                result.show()
    
