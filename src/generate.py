import os
import random
import xml.etree.ElementTree as ET
from utils.utils import get_classes

if __name__ == "__main__":
    
    VOCdevkit_path = "data/voc/2007/VOCdevkit"
    VOCdevkit_sets = [('2007', 'train'), ('2007', 'val')]
    VOCdevkit_year = 2007
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
    
    print("Generate txt in ImageSets.")
    xmlfilepath  = os.path.join(VOCdevkit_path, 'VOC{}/Annotations'.format(VOCdevkit_year))
    saveBasePath = os.path.join(VOCdevkit_path, 'VOC{}/ImageSets/Main'.format(VOCdevkit_year))
    temp_xml     = os.listdir(xmlfilepath)
    total_xml    = []
    for xml in temp_xml:
        if xml.endswith('.xml'):
            total_xml.append(xml)

    num      = len(total_xml)
    list     = range(num)
    tv       = int(num * test_ratio)
    tr       = int(tv * valid_ratio)
    trainval = random.sample(list, tv)
    train    = random.sample(trainval, tr) 

    print("train and val size",tv)
    print("train size",tr)

    ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
    ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
    ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
    fval        = open(os.path.join(saveBasePath,'val.txt'), 'w') 
    
    for i in list:
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)
    
    ftrainval.close()  
    ftrain.close()  
    fval.close()  
    ftest.close()
    print("Generate txt in ImageSets done.")

    print("Generate {}_train.txt and {}_val.txt for train.".format(VOCdevkit_year, VOCdevkit_year))
    for year, image_set in VOCdevkit_sets:
        image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt'%(year, image_set)), encoding='utf-8').read().strip().split()
        list_file = open('data/voc/%s_%s.txt'%(year, image_set), 'w', encoding='utf-8')
        for image_id in image_ids:
            list_file.write('%s/VOC%s/JPEGImages/%s.jpg'%(os.path.abspath(VOCdevkit_path), year, image_id))
        
            convert_annotation(year, image_id, list_file)
            list_file.write('\n')
        list_file.close()
    print("Generate {}_train.txt and {}_val.txt for train done.".format(VOCdevkit_year, VOCdevkit_year))