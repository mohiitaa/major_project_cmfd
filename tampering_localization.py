import numpy as np
import cv2
import matplotlib.pyplot as plt


def tampering_localization(*argv):	
	bound = bwboundaries(bw); ########
	inModel_x =np.concatenate((z1[1,inliers], z2[1,inliers]),axis=1)
	inModel_x=inModel_x.transpose()
	inModel_y =np.concatenate((z1[2,inliers], z2[2,inliers]),axis=1)
	inModel_y=inModel_y.transpose()
	
	img_out=np.zeros([im.shape[0], im.shape[1]])
	for k in range(0,bound.shape[0]):
		b=bound[k]######
		inp=inpolygon(inModel_x,inModel_y, b[:,1],b[:,0])#####
		find_nonzero=np.nonzero(inp)
		if (find_nonzero!=[]):
			bw_b=np.zeros(im.shape[0],im.shape[1])
			bw_b = roipoly(bw_b,b(:,2),b(:,1));  ####
			img_out=np.logical_or(img_out,bw_b)
			img_out=1*img_out
			
	if (show_mask):
		fig = plt.figure()
		plt.imshow(img_out)
		
	return img_out
		
		