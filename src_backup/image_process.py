from Yolov3.config import CLASS_NAME_FILE
import numpy as np
from PIL import Image, ImageDraw
from Yolov3.utils import load_class_names
from Yolov3.config import *

if __name__ == "__main__":
    image_path = '/Users/shijinyang/Downloads/VOCdevkit/VOC2012/JPEGImages/2008_006482.jpg'
    class_name = load_class_names(CLASS_NAME_FILE)
    box = np.array([1,341,201,411])
    
    img = Image.open(image_path)

    iw, ih = img.size
    h,  w  = 416, 416

    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2

    img = img.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(img, (dx, dy))

    box[0] = box[0]*nw/iw + dx
    box[2] = box[2]*nw/iw + dx
    box[1] = box[1]*nh/ih + dy
    box[3] = box[3]*nh/ih + dy

    print(box)
 
    d = ImageDraw.Draw(new_image)
    d.rectangle([box[0], box[1], box[2], box[3]])
    d.text((box[0], box[1]), class_name[60], fill='black')

    new_image.save('test.png')




    
