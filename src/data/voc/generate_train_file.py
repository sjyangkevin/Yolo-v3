import os
import random
import xml.etree.ElementTree as ET
from utils.utils import get_classes

if __name__ == "__main__":
    
    VOCdevkit_path = ""
    VOCdevkit_sets = [('2007', 'train'), ('2007', 'val')]
    classes_path   = "data/voc/voc_classes.txt"
    classes, _     = get_classes(classes_path)

    # ratio between (train + valid) and test sets
    test_ratio   = 0.9
    # ratio between train and valid sets
    valid_ratio  = 0.9

    def convert_annotation(year, image_id, list_file):
        in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml'%(year, image_id)), encoding='utf-8')
        tree    = ET.parse(in_file)
        root    = tree.getroot()

        for obj in root.iter('object'):
            difficult = 0
            if obj.find('difficult') != None:
                difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue

            cls_id  = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
            list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


if __name__ == "__main__":
    random.seed(0)
    
    
