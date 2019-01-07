import numpy as np
from numpy import linalg as LA

# p1 = [1,2]
# p2 = [2,4]
# p3 = [3,5]

def iscolinear(p1,p2,p3,flag):             # flag == 0 means homogenoeous coordinates
    
    #x = np.shape(p1)
    #y = np.shape(p2)
    #z = (x == y)
    #print z
    
    
    if ((np.shape(p1)!= np.shape(p2)) or (np.shape(p2) != np.shape(p3)) or (np.size(p1)!=2 and np.size(p1)!=3)):
        # print ('All good')
        print ('ERROR: All three must have the same dimensions of either 2 or 3')
    
    
    # print " [iscolinear] shapes : p1, p2, p3 \n"
    # print p1.shape, p2.shape, p3.shape
    if(np.size(p1)==2):
        ph1=np.zeros([3])
        ph2=np.zeros([3])
        ph3=np.zeros([3])
        ph1[0:2]=p1
        ph1[2]=1
        ph2[0:2]=p2
        ph2[2]=1
        ph3[0:2]=p3
        ph3[2]=1
        print "hansal:::::::::::::::::::::::::::::::::::::::::::ann"
        
        
    
    
    if (flag == 0):
        r = (np.absolute(np.dot(np.asmatrix(np.cross(p1.transpose(),p2.transpose())),p3)) < 0.000001)
        
       
    else:
        r = (LA.norm(np.cross(ph2-ph1, ph3-ph1)) < 0.000001)
        
    return r

# x = iscolinear(p1,p2,p3,0)
# print x