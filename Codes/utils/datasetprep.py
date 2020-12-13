# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 07:32:08 2019

@author: RoKu
"""

import os
from os import listdir
import numpy as np
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset

class PcbDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train = True):
    	# define one class
    	self.add_class("missing", 1, "missinghole")
    	# define data locations
    	images_dir = dataset_dir + "/images/"
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

#training
train_set = PcbDataset()
train_set.load_dataset("missing_train")        
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))   

#testing
test_set = PcbDataset()
test_set.load_dataset("missing_test")        
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))   

