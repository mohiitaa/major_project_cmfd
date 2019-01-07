import numpy as np
import homography2d as h
import math
import random as rdm
import warnings
import ransacfithomography2 as rgh
import sys
# from ransacfithomography2 import isdegenerate

def ransac(*argv):
	#Octave=exist('OCTAVE_VERSION') ~= 0;   ################
	if ((len(argv)<3) or (len(argv)>6)):
		raise ValueError('Invalid no of arguments')
	print "Entered RANSAC"
	if (len(argv)<6):
		maxTrials=1000
	if (len(argv)<5):
		 maxDataTrials = 100
	if (len(argv)<4):
		feedback = 0
	x=np.asmatrix(argv[0])
	s=argv[1]
	t=argv[2]
	
	[rows,npts]=x.shape
	p=0.95
	bestM=float('nan')
	trialcount=0
	bestscore=0
	N=1
	print "Entering while loop in RANSAC"
	
	while (N>trialcount):  
	# kla = 7
	# while(kla > 4):      
		degenerate = 1
		count=1
		print "trial"
		M=h.homography2d(x)
		while (degenerate):
		# 
		# while(kla > 4):
			ind=[]
		
			for i in range(s):
				temp=rdm.randint(0,npts-1)
				ind.append(temp)
			
			# ind=np.array(ind)
			x = np.asmatrix(x)
			# print "[ransac] x is : \n"
			# print x
			# print "ind \n", ind
			# print "x[:,ind]", x[:,ind]
			degenerate=rgh.isdegenerate(x[:,ind])

			if (degenerate==0):
				M=h.homography2d(x[:,ind])
			# h = cv2.findHomography(, points2, cv2.RANSAC)
				if (M==[]):
					degenerate=1
			count=count+1
			if (count>maxDataTrials):
				warnings.warn('Unable to select a nondegenerate data set')
				break
				
				
			
		# kla = kla - 1
		[inliers,M]=rgh.homogdist2d(M,x,t)
		ninliers=len(inliers)
		if (ninliers > bestscore):  
			bestscore = ninliers
			bestinliers = inliers
			bestM = M
			fracinliers =  ninliers/npts
			pNoOutliers = 1 -  fracinliers**s
			eps=sys.float_info.epsilon
			pNoOutliers = max(eps, pNoOutliers)
			pNoOutliers = min(1-eps, pNoOutliers)
			N = math.log(1-p)/math.log(pNoOutliers)
	
		
		trialcount = trialcount + 1
		print "\n \ntrial count: ", trialcount
		print "\n \nN: ", N
		
		if feedback:
			print "trial",trialcount,"out of",math.ceil(N)
		if trialcount > maxTrials:
			warnings.warn("ransac reached the maximum number of trials")
			# break	

	print "While loop ended in RANSAC"		
	if not np.isnan(bestM):
		M=bestM
		inliers=bestinliers
	else:
		M = []
		inliers = []
		warnings.warn('ransac was unable to find a useful solution')
			
	return M, inliers
					
					
		
	
	
	
		
		
		
		
	
	
	