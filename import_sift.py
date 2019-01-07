import cv2
import numpy as np
import warnings

def import_sift(filename):

	g=open(filename,'r')
	if (g==-1):
#	raise ValueError("Could not open sift file")
		warnings.warn('Could not open sift file')
		num=0
		locs=[]
		descs=[]
	
	else:
		#[header, count] = fscanf(g, '%d %d', [1 2]);   ####################
		
		lines=g.readlines()
		l=lines[0]
		l=l.split(' ')
		if (len(l)>=2):
			header=[int(l[0]), int(l[1])]
		count=len(header)
		if count!=2:	
			raise ValueError('Invalid keypoint file beginning')
		num=header[0]
		length=header[1]
		if (length!=128):	
			raise ValueError('Keypoint descriptor length invalid (should be 128).')
		z_temp=np.zeros([num,4])
		locs=z_temp.astype(float)
		z_temp=np.zeros([num,128])
		descs=z_temp.astype(float)
	    
		k=0
		
		lines=g.readlines()
		mat=[]
		for l in lines:
            
			l=l.split(' ')
			for j in range(len(l)):
				mat.append(l[j])
		for i in range(1,num):
			#[vector, count] = fscanf(g, '%f %f %f %f', [1 4])
			
			vector=[0.0]*4
			for i in range(4):
				vector[i]=mat[k+i]
			k=k+4
			count=len(vector)		
			if (count!=4):
				raise ValueError('Invalid keypoint file format')
			locs[i,:]=vector[0,:]
			descrip=[0]*length
			for i in range(length):
				descrip[i]=mat[k+i]
			k=k+length
			count=len(descrip)
			#[descrip, count] = fscanf(g, '%d', [1 len]);
			
			if (count!=128):	
				raise ValueError('Invalid keypoint file value')
		g.close()
		return num,locs,descs
	
	
		
	
		
		
		
	
	


