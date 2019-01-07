# SHOW_INLIERS: Given an image a two sets of points, the function shows 
#               only the inliers.

import cv2
import matplotlib.pyplot as plt

def show_inliers(imagefile, zz1, zz2, inliers):
    #zz1=[[11,12,13,14,15],[21,22,23,24,25]]
    #zz2=[[55,56,57,58,59],[65,66,67,68,68]]
    #inliers=[[1,2,3],[4,5,6]]
    
    image1=cv2.imread(imagefile)
    cv2.imshow('meh',image1)
    
    for i in range(len(inliers[0])):
        plt.plot([ zz1[0][inliers[0][i]], zz2[0][inliers[0][i]] ] , [zz1[1][inliers[0][i]], zz2[1][inliers[0][i]]], color='C2')
        plt.show()
    
