
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from Yolov3.dataset import Dataset

if __name__ == "__main__":
    ds = Dataset('train')
    annot = ds.load_annotations('train')

    print(annot[3])

    image, bboxes = ds.parse_annotation(annot[3])
    print(image.shape)
    print(bboxes.shape)
    print(bboxes)

    img = Image.fromarray(np.uint8(image*255))
    d = ImageDraw.Draw(img)
    for i in range(bboxes.shape[0]):

        d.rectangle([bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]], outline='black')


    img.save('test_parse_annot.png')
    