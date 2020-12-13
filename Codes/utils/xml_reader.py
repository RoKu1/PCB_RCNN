# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 07:23:30 2019

@author: RoKu
"""
from xml.etree import ElementTree

# function to extract bounding boxes from an annotation file
def extract_boxes(filename):
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

# extract details form annotation file
boxes, w, h = extract_boxes('l_light_01_missing_hole_01_1_600.xml')
# summarize extracted details
print(boxes, w, h)
    