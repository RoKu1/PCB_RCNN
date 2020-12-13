# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 19:49:59 2019

@author: RoKu
"""
import numpy as np
from os import listdir
from xml.etree import ElementTree
from numpy import expand_dims
from numpy import mean
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image

class PcbDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train = True):
    	# define one class
    	self.add_class("missing", 1, "missinghole")
    	# define data locations
    	images_dir = dataset_dir + "images/"
    	annotations_dir = dataset_dir + "annots/"
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


# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "predict_cfg"
	# number of classes (background + missingho;e)
	NUM_CLASSES = 1 + 1
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
 
# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
	APs = list()
	for image_id in dataset.image_ids:
		# load image, bounding boxes and masks for the image id
		image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)
		# extract results for first sample
		r = yhat[0]
		# calculate statistics, including AP
		AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
		# store
		APs.append(AP)
	# calculate the mean AP across all images
	mAP = mean(APs)
	return mAP

train_set = PcbDataset()
train_set.load_dataset("./missing_train/missing_train/")        
train_set.prepare()

test_set = PcbDataset()
test_set.load_dataset("./missing_test/")        
test_set.prepare()

# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='./new.h5', config=cfg)
# load model weights
model.load_weights('new.h5', by_name=True)

# evaluate model on training dataset
train_mAP = evaluate_model(train_set, model, cfg)
print("Train mAP: %.3f" % train_mAP)
# evaluate model on test dataset
test_mAP = evaluate_model(test_set, model, cfg)
print("Test mAP: %.3f" % test_mAP)