"""
Created on Tue Dec  3 19:59:52 2019

@author: RoKu
"""
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.utils import Dataset
import numpy as np


class PcbDataset(Dataset):
    def load_dataset(self, dataset_dir, is_train=True):
        self.add_class("missing", 1, "missinghole")
        images_dir = dataset_dir + 'images/'
        annotations_dir = dataset_dir + 'annots/'
        for filename in listdir(images_dir):
            image_id = filename[:-4]
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
            self.add_image('missing', image_id=image_id, path=img_path, annotation=ann_path)

    def extract_boxes(self, filename):
        root = ElementTree.parse(filename)
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']
        boxes, w, h = self.extract_boxes(path)
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('missinghole'))
        return masks, asarray(class_ids, dtype='int32')

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


class PredictionConfig(Config):
    NAME = "predict_cfg"
    NUM_CLASSES = 1 + 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

#
# train_set = PcbDataset()
# train_set.load_dataset("./missing_train/", is_train=True)
# train_set.prepare()
# print('Train: %d' % len(train_set.image_ids))
# test_set = PcbDataset()
# test_set.load_dataset("./missing_test/", is_train=False)
# test_set.prepare()
# print('Test: %d' % len(test_set.image_ids))

cfg = PredictionConfig()
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# path to model is given below --> specify this according to your system
model_path = 'Weights/new.h5'
model.load_weights(model_path, by_name=True)

from PIL import Image

# image = Image.open('modifies_image.jpeg').convert('RGB')
# img_resized = image.resize((600, 600))
# image = np.array(img_resized)

image = pyplot.imread('l_light_08_missing_hole_05_1_600.jpg')
scaled_image = mold_image(image, cfg)
sample = expand_dims(scaled_image, 0)
yhat = model.detect(sample, verbose=0)

pyplot.imshow(image)
pyplot.title('Predicted')
ax = pyplot.gca()
for box in yhat[0]['rois']:
    y1, x1, y2, x2 = box
    width, height = x2 - x1, y2 - y1
    rect = Rectangle((x1, y1), width, height, fill=False, color='red')
    ax.add_patch(rect)
pyplot.show()

from mrcnn import visualize

visualize.display_weight_stats(model)
