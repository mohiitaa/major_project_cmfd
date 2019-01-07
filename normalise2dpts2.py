import cv2
import numpy as np



def normalise2dpts2(pts):
    if ((np.size(pts,0) != 3)):
        print('pts must be 3xN')
    r,col = pts.shape
    
    [x,y]= np.where(np.absolute(pts[2,:])>0.00000001)
    
    length = np.size(y)
    
    for i in np.arange(length):
        pts[0,y[i]] = pts[0,y[i]]/pts[2,y[i]]
        pts[1,y[i]]= pts[1,y[i]]/pts[2,y[i]]
        pts[2,y[i]] = 1
    c = pts.transpose()
    # print y
    d = c[y]
    # print pts 
    e = d.transpose()
    m = np.mean(e[0:2],1)
    # print pts
    # print "m----------", m
    h=r-1
    # print "h----------",h
    # print "c----------",col
    newp=np.zeros([h,col])
    
    
    for i in np.arange(length):
        newp[0,y[i]]= pts[0,y[i]] - m[0]
        newp[1,y[i]]= pts[1,y[i]] - m[1]
    sum=np.zeros(shape=(length,1))
    for i in np.arange(length):
        sum[i] +=np.power(newp[0,y[i]],2)
        sum[i] +=np.power(newp[1,y[i]],2)
    
    dist = np.sqrt(sum)
    
    meandist=np.mean(dist)
    
    if (meandist != 0):
        scale=np.sqrt(2)/meandist
    else:
        scale = 1
    
    T = np.zeros(shape=(3,3))
    
    T[0,0]=scale
    T[1,1]=scale
    T[2,2]= 1
    T[0,2]= -scale*m[0]
    T[1,2]= -scale*m[1]
    
    newpts= np.zeros(shape=(r,col))
    newpts= np.matmul(T,pts)
    # print "RETURNED FROM NORMALISE2Dpoibts"
    return [newpts,T,m]