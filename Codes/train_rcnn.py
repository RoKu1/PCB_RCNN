# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 19:59:31 2019

@author: RoKu
"""
from os import listdir
import numpy as np
from xml.etree import ElementTree
from mrcnn.utils import Dataset
import matplotlib.pyplot as plt
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.model import MaskRCNN
from mrcnn.config import Config
import mrcnn.utils
import os

class PcbDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train = True):
    	# define one class
    	self.add_class("missing", 1, "missinghole")
    	# define data locations
    	images_dir = dataset_dir + "//images//"
    	annotations_dir = dataset_dir + "//annots//"
    	# find all images
    	for filename in listdir(images_dir):
    		# extract image id
    		image_id = filename[:-4]
    		img_path = images_dir + filename
    		ann_path = annotations_dir + image_id + '.xml'
    		# add to dataset
    		self.add_image('missing', image_id=image_id, path=img_path, annotation=ann_path)
    
  
    # load the masks for an image
    def load_mask(self, image_id):
    	# get details of image
    	info = self.image_info[image_id]
    	# define box file location
    	path = info['annotation']
    	# load XML
    	boxes, w, h = self.extract_boxes(path)
    	# create one array for all masks, each on a different channel
    	masks = np.zeros([h, w, len(boxes)], dtype='uint8')
    	# create masks
    	class_ids = list()
    	for i in range(len(boxes)):
    		box = boxes[i]
    		row_s, row_e = box[1], box[3]
    		col_s, col_e = box[0], box[2]
    		masks[row_s:row_e, col_s:col_e, i] = 1
    		class_ids.append(self.class_names.index("missinghole"))
    	return masks, np.asarray(class_ids, dtype='int32')
    
    
    
    # load an image reference
    def image_reference(self, image_id):
    	info = self.image_info[image_id]
    	return info['path']
    
    # function to extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
    	# load and parse the file
    	tree = ElementTree.parse(filename)
    	# get the root of the document
    	root = tree.getroot()
    	# extract each bounding box
    	boxes = list()
    	for box in root.findall('.//bndbox'):
    		xmin = int(box.find('xmin').text)
    		ymin = int(box.find('ymin').text)
    		xmax = int(box.find('xmax').text)
    		ymax = int(box.find('ymax').text)
    		coors = [xmin, ymin, xmax, ymax]
    		boxes.append(coors)
    	# extract image dimensions
    	width = int(root.find('.//size/width').text)
    	height = int(root.find('.//size/height').text)
    	return boxes, width, height

class MissingConfig(Config):
    # Give the configuration a recognizable name
	NAME = "missing_cfg"
	# Number of classes (background + kangaroo)
	NUM_CLASSES = 1 + 1 ; GPU_COUNT = 1;  IMAGES_PER_GPU = 2;
	# Number of training steps per epoch
	STEPS_PER_EPOCH = 1282


train_set = PcbDataset()
train_set.load_dataset("missing_train//")        
train_set.prepare()

test_set = PcbDataset()
test_set.load_dataset("missing_test//")        
test_set.prepare()
 
# prepare config
config = MissingConfig()

# define the model
model = MaskRCNN(mode='training', model_dir='./mask_rcnn_coco.h5', config=config)

# load weights (mscoco)
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

# train weights (output layers or 'heads')
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')

model_path = os.path.join('./', 'new.h5')
model.keras_model.save_weights(model_path)