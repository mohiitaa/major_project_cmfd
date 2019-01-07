# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:25:03 2018

@author: Subbalakshmi
"""

import cv2

import numpy as np

img=cv2.imread('tampered1.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#sift=cv2.xfeatures2d.SIFT_create()
sift=cv2.SIFT()
kp=sift.detect(gray,None)

