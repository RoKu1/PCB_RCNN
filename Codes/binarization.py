# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:05:11 2019

@author: RoKu
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('l_light_01_missing_hole_01_1_600.jpg',0)
img1 = cv2.imread('l_light_01_missing_hole_01_1_600_f.jpg',0)


ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret2, thresh2 = cv2.threshold(img1,127,255,cv2.THRESH_BINARY)

thresh3 = thresh1 ^ thresh2;

        
plt.imshow(thresh1,'gray')
plt.title('Correct')
plt.show()


plt.imshow(thresh2,'gray')
plt.title('Faulty')
plt.show()

plt.imshow(thresh3,'gray')
plt.title('Binarization')
plt.show()