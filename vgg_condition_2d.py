
import numpy as np
import vgg_get_nonhomg as vgn
#import vgg_get_homg as vgh 
import homography2d as h

def vgg_condition_2d(p,C):

	[r,c]=p.shape
 
	if (r==2):
		temp=vgh.vgg_get_homg(p)
		# temp=h.homography2d(p)
		
		pc=vgn.vgg_get_nonhomg(C * temp);			
	elif (r==3):
		pc=C*p
	else:
		raise ValueError("rows != 2 or 3")
	return pc
	



