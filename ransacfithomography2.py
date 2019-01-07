import numpy as np
import cv2
from iscollinear import iscolinear
from normalise2dpts2 import normalise2dpts2
from ransac import ransac
from hnormalise import hnormalise

def ransacfithomography2(x1,x2,t):

    if(np.size(x1) != np.size(x2)):
        print('The two set of points must have the same dimensions!')
    
    [rows,npts]= x1.shape
    
    if(rows != 2 and rows !=3):
        print('x1 and x2 must have two or three rows')
    
    if(npts < 3):
        print('Must have at least 3 points to fit homography')
        exit()
    
    if (rows == 2):
        x1_n= np.ones(shape=((rows+1),npts))
        x2_n= np.ones(shape=((rows+1),npts))
        
        x1_n[:rows,:npts] = x1
        x2_n[:rows,:npts] = x2
    
        [x1, T1, c1] = normalise2dpts2(x1_n)
        [x2, T2, c2] = normalise2dpts2(x2_n)
    else:
        [x1, T1, c1] = normalise2dpts2(x1)
        [x2, T2, c2] = normalise2dpts2(x2)
        
    
    # c1 and c2 are centers of the clusters
    # print "[Inside ransacfithomography2.py] x1,x2 \n"
    # print x1, "\n"
    # print x2, "\n"

    # print " Yay after normalise2dpnts"
    cx = c1[0]
    cy = c1[1]
    dx = c2[0]-c1[0]
    dy = c2[1]-c1[1]
    # print "Hansal's matlab error"
    s = 4 # Minimum No of points needed to fit a homography.
    x1_x2=np.concatenate((x1,x2))
    # print "Going to ransac, let us see if we return...."
    x1 = np.asmatrix(x1)
    x2 = np.asmatrix(x2)
    # [H, inliers] = ransac(x1_x2, s, t)
 
    H, mask = cv2.findHomography(x1.transpose(), x2.transpose(), cv2.RANSAC,t)
    # print "RANSAC done!"
    # print "MASK ISSSSSS: \n", mask
    # inliers = mask[i for mask[i] == 0]
#     inliers = []
# [[i] for i in range(len(mask)) if mask[i]]
#     inliers = 
    
    return [H, mask, dx, dy, cx, cy]
    
    # return [H, inliers, dx, dy, cx, cy]
    
def homogdist2d(H,x,t):

    [rows,npts]= x.shape

    x1= np.zeros(shape=((rows/2),npts))
    x1= x[0:3,:]
    x2= np.zeros(shape=((rows/2),npts))
    x2= x[3:,:] 

    Hx1=np.matmul(H,x1)
    temp = np.linalg.pinv(H)
    invHx2 = temp.dot(x2)

    x1= hnormalise(x1)
    x2= hnormalise(x2)     
    Hx1= hnormalise(Hx1)
    invHx2= hnormalise(invHx2) 

    sum_1= np.zeros(shape=(npts,1))
    sum_2= np.zeros(shape=(npts,1))
    d2= np.zeros(shape=(npts,1))
    sum_1= np.sum(np.power(np.subtract(x1,invHx2),2),0)
    sum_2= np.sum(np.power(np.subtract(x2,Hx1),2),0)
    # d2=np.sum(sum_1,sum_2)
    d2 = sum_1 + sum_2
    
    [w,inliers] = np.where(np.absolute(d2)<t)  # Here inliers has the y coordinates and w has co-ordinates 
    return inliers, H



def isdegenerate(x):
    [rows, npts] = x.shape
    # print "[inside rfh; isdegenerate] rows, npts \n"
    # print rows, npts
    x1= np.zeros(shape=((rows/2),npts))
    x1= x[0:3,:]
    x2= np.zeros(shape=((rows/2),npts))
    x2= x[3:6,:]
    r= iscolinear(x1[:,0],x1[:,1],x1[:,2],0) or iscolinear(x1[:,0],x1[:,1],x1[:,3],0) or iscolinear(x1[:,0],x1[:,2],x1[:,3],0) or iscolinear(x1[:,1],x1[:,2],x1[:,3],0) or iscolinear(x2[:,0],x2[:,1],x2[:,2],0) or iscolinear(x2[:,0],x2[:,1],x2[:,3],0) or iscolinear(x2[:,0],x2[:,2],x2[:,3],0) or iscolinear(x2[:,1],x2[:,2],x2[:,3],0)
    return r
        
        
    
