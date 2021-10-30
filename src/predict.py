import time
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from YOLOv3 import YOLO

if __name__ == "__main__":
    
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    video_path      = "data/videos/toronto_street_clip.mp4"
    video_save_path = "data/videos/toronto_street_clip_out.mp4"
    video_fps       = 25.0

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
    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            size   = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out    = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
        
        fps = 0.0
        while(True):
            t1 = time.time()
            ref, frame = capture.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(model.detect_image(frame))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            frame = cv2.putText(frame, "fps = %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)
            
            if c == 27:
                capture.release()
                break
        
        capture.release()
        out.release()
        cv2.destroyAllWindows()