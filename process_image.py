import numpy as np
import cv2
from scipy import spatial
from scipy import cluster
import matplotlib.pyplot as plt
import match_features as mf
import ransacfithomography2 as rfh
from itertools import combinations
import seaborn as sns # Need to install seaborn
import os
import match_features
# import vgg_get_homg as gh 

## Somelines have to be inserted here.

def process_image(*argv):
	imagefile=argv[0]
	if (len(argv)==1):
		metric='ward'
		thc=2.2
		min_cluster_pts=3
		plotimg=1
		siftfile='nofile'
	else:
		metric=argv[1]
		if (len(argv)==2):
			thc=2.2
			min_cluster_pts=3
			plotimg=1
			siftfile='nofile'
		else:	
			thc=argv[2]
			if (len(argv)==3):
				min_cluster_pts=3
				plotimg=1
				siftfile='nofile'
			else:
				min_cluster_pts=argv[3]
				if (len(argv)==4):
					plotimg=1
					siftfile='nofile'
				else:
					plotimg=argv[4]
					if (len(argv)==5):
						siftfile='nofile'
					else:
						siftfile=argv[5]
		
	# f_name = os.path.basename(imagefile)
	# pf,nf_temp = os.path.split(f_name)
    # nf,ef = os.path.splitext(nf_temp)	
	# desc_file = os.path.join(pf,nf) + '.txt'


	# file = open(desc_file, 'w')


	metric = 'ward'	
	thc = 2
	min_cluster_pts = 3

	image1=cv2.imread(imagefile)
	inliers1=[]
	inliers2=[]

	[num,p1,p2,tp] = mf.match_features(imagefile, siftfile)

	if (p1.shape[0]==0):
		num_gt=0
	else:
		p=np.concatenate((p1[0:2,:],p2[0:2,:]),axis=1)
		p=p.transpose()
		# print "p:"
		# print "\n"
		# print p
		# print "\n"
		# print "in process image, p shape is:",p.shape
		distance_p=spatial.distance.pdist(p,'euclidean')
		# distance_p = np.asmatrix(distance_p)
		# distance_p = distance_p.transpose()
		# print "distance_p.shape: ", distance_p.shape
		Z=cluster.hierarchy.linkage(distance_p, metric = 'ward')
		# print "Z shape: ", Z.shape
		# print "Z:"
		# print "\n"
		# print Z
		# print "\n"
		#c = cluster(Z,'cutoff',thc,'depth',4); ###########
		c=cluster.hierarchy.fcluster(Z,thc,depth=4)
		# print " THE CLUSTER BEGINS IN 3...2...1...\n"
		# print c
		# print "The actual and matched points are:\n"
		# print p
		# print "\n"
		# print "OH MAAAAY GAAAAD Its over "
		if (plotimg==1):
		#fig = plt.figure()
			cv2.namedWindow('image', cv2.WINDOW_NORMAL)
			
			for i in range(0,p1.shape[1]):
				x=np.asmatrix([p1[1,i],p2[1,i]])
				y=np.asmatrix([p1[0,i],p2[0,i]])
				linethickness = 2
				cv2.line(image1, (x[:,0], y[:,0]), (x[:,1], y[:,1]), (0,255,255), linethickness)
				cv2.imshow('image',image1)
				cv2.waitKey(1)
				# print "We are in the FORRRRR LOOOPP"
				# print x
				# print y
				# print "______________________________________________________________"
				# # plt.plot(x, y) ## Don't know what is c in colour
				
				# plt.close()
				#gscatter(p(:,1),p(:,2),c)  #####
				
				#sns.lmplot( x="sepal_length", y="sepal_width", data=df, fit_reg=False, hue='species', legend=False)
				#sns.lmplot(x=" ", y=" ",data=    )###
			
			
			cv2.destroyAllWindows()
			# plt.scatter(p[:,0], p[:,1])
		H = []	 
		num_gt=0
		c_max=np.max(c)
		if (c_max>1):
				#n_combination_cluster = combntns(1:c_max,2); #####
			temp=range(1,c_max+1)
			temp_combination_cluster=combinations(temp,2)
			n_combination_cluster = list(temp_combination_cluster)
			n_combination_cluster = np.asmatrix(n_combination_cluster)

				
			for i in range(0,n_combination_cluster.shape[0]):
					k=n_combination_cluster[i,0]
					j=n_combination_cluster[i,1]
				
					z1=[]
					z2=[]
					for r in range(0,p1.shape[1]):
						# if ((c[r]==k) and c[r+p1.shape[1]]==j):
						if (z1==[]):
							z1=np.concatenate((p[r,:],np.asmatrix(1)),axis=1)								
						else:
							temp=np.concatenate((p[r,:],np.asmatrix(1)),axis=1)
							z1=np.concatenate((z1,temp))
						if (z2==[]):
							z2=np.concatenate((p[r+p1.shape[1],:],np.asmatrix(1)),axis=1)
						else:
							temp=np.concatenate((p[r+p1.shape[1],:],np.asmatrix(1)),axis=1)
							z2=np.concatenate((z2,temp))
							
						# if ((c[r]==j) and c[r+p1.shape[1]]==k):
							# if (z1==[]):
							# 	z1=np.concatenate((p[r+p1.shape[1],:],np.asmatrix(1)),axis=1)
							# else:
							# 	temp=np.concatenate((p[r+p1.shape[1],:],np.asmatrix(1)),axis=1)
							# 	z1=np.concatenate((z1,temp))
							# if (z2==[]):
							# 	z2=np.concatenate((p[r,:],np.asmatrix(1)),axis=1)
							# else:
							# 	temp=np.concatenate((p[r,:],np.asmatrix(1)),axis=1)
							# 	z2=np.concatenate((z2,temp))
				
					
							
					if ((z1.shape[0]> min_cluster_pts) and (z2.shape[0]>min_cluster_pts)):
						# print "z1, z2"
						# print "\n"
						# print "z1: \n",z1
						# print "\n"
						# print "z2:\n", z2
						# print "\n"
						# print "\n"
		
		##################### New code ###############################
		
		# print "p1 \n"
		# print p1
		# print "p2 \n"
		# print p2
		
		################################################################
						[H, mask, dx, dy, xc, yc] = rfh.ransacfithomography2(p1, p2, 0.05)
						# H = np.asmatrix(H)
						H = np.asmatrix(H)
						# print "H", H
						if (np.size(H) == 1):
							num_gt=num_gt
						else:							
							# print "H",H
							H=H/H[2,2]
							num_gt=num_gt+1
			# #if (inliers1==[]):                            
			# #inliers1=np.asmatrix([z1[inliers,0], z1[inliers,1]])
			# #else:
			# 	#temp=np.asmatrix([z1[inliers,0] ,z1[inliers,1]])
			# 	#inliers1=np.concatenate((z1,temp))
			# if (inliers2==[]):
			# 	inliers2=np.asmatrix([z2[inliers,0], z2[inliers,1]])
			# else:
			# 	temp=np.asmatrix([z2[inliers,0] ,z2[inliers,1]])
			# 	inliers2=np.concatenate((z2,temp))
            #             # print "Hi"
            #             #if (inliers1==[]):
			# 				#	inliers1=np.asmatrix([z1[inliers,0], z1[inliers,1]])
			# 			   #else:
			# 				#	temp=np.asmatrix([z1[inliers,0] ,z1[inliers,1]])
							#	inliers1=np.concatenate((z1,temp))
	
	
							x = np.size(mask)	
							count = 0	
							# corrected_actual = []
							# corrected_matches = 
							# print "mask shape is:\n"
							# print np.size(mask)
							# print "mask"
							# print mask
							if(np.size(mask) != 1):
								r,c = np.where(mask == 1)
								corrected_actual = np.zeros(shape = (np.size(r),2))
								corrected_matches = np.zeros(shape = (np.size(r),2))
								# print "Shapes (p,mask)\n" 
								# print p.shape, mask.shape                 
								# print "Homography Matrix is :\n"
								# print H
								# print "inliers & outliers are:\n"
								# print mask
							

								for i in range (len(mask)): 
									if(mask[i]==1):
										# print count
										corrected_actual[count,:] = p[i,0:2]
										corrected_matches[count,:] = p[i+len(mask),0:2]
										count = count + 1
										
								# print corrected_actual, "\n"
								# print corrected_matches, "\n"

								cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
								for i in range(len(corrected_actual)):
									cv2.circle(image1,(int(corrected_actual[i,1]),int(corrected_actual[i,0])),5,(255,0,0),-1)
									cv2.circle(image1,(int(corrected_matches[i,1]),int(corrected_matches[i,0])),5,(0,0,255),-1)

								

	# cv2.destroyAllWindows()
	# cv2.imshow('img2', image1)	
	# cv2.waitKey(0)
	nrows, ncols, cchal = image1.shape 
	f_name = os.path.basename(imagefile)
	pf,nf_temp = os.path.split(f_name)
	nf,ef = os.path.splitext(nf_temp)	
	desc_file = os.path.join(pf+'results/',nf) + '_result.bmp'
	desc_file2 = os.path.join(pf+'results/',nf) + '_result.txt'
	file = open(desc_file2,"w") 
	
		

	cv2.imwrite(desc_file, image1)
	cv2.destroyAllWindows()
	if H!=[]:
		print "Homography matrix is: \n"
		print H
# tampering detection
	if(num_gt!=0):
		tamperStatus = 'Tampering Detected.\n\n'  
	else:
		tamperStatus = 'Image not tampered.\n\n'
	
	print tamperStatus
	file.write(tamperStatus)
	file.write("Size of the image (rows, cols): "+ repr(nrows) + "," + repr(ncols))
	file.write("\n\nTime elapsed: " + repr(tp)) 
	if H != []:
		file.write("\n\nHomography matrix is:\n\n"+ repr(H))
		file.write("\n\nAffine Transformation:\n\n"+ repr(H[0:2,0:2]))
		file.write("\n\nTranslation:\n\n"+ repr(H[0:2,2]))
	file.close()
	return num_gt,inliers1, inliers2
	


							
				
						

							
					
						
	
	
							
							
							
							
			
		
		
			 
		
			 
			
			
		
	 
	 
			
				
				
				
			
			
		
		
		



