# % HOMOGRAPHY2D - computes 2D homography
# %
# % Usage:   H = homography2d(x1, x2)
# %          H = homography2d(x)
# %
# % Arguments:
# %          x1  - 3xN set of homogeneous points
# %          x2  - 3xN set of homogeneous points such that x1<->x2
# %         
# %           x  - If a single argument is supplied it is assumed that it
# %                is in the form x = [x1; x2]
# % Returns:
# %          H - the 3x3 homography such that x2 = H*x1
# %
# % This code follows the normalised direct linear transformation 
# % algorithm given by Hartley and Zisserman "Multiple View Geometry in
# % Computer Vision" p92.
# %

# % Peter Kovesi
# % School of Computer Science & Software Engineering
# % The University of Western Australia
# % pk at csse uwa edu au
# % http://www.csse.uwa.edu.au/~pk
# %
# % May 2003  - Original version.
# % Feb 2004  - Single argument allowed for to enable use with RANSAC.
# % Feb 2005  - SVD changed to 'Economy' decomposition (thanks to Paul O'Leary)
import cv2
import numpy as np 
import copy
# import scipy as sp
from normalise2dpts2 import normalise2dpts2
# function H = homography2d(varargin)
def homography2d(*argv):   
    [x1, x2] = checkargs(argv[:])  ###### What is checkargs ?? varargin(:)
    # print "HEY I am in homography 2d"
    # % Attempt to normalise each set of points so that the origin 
    # % is at centroid and mean distance from origin is sqrt(2).
    # print "x1.shape",x1.shape
    [x1, T1, c1] = normalise2dpts2(x1)
    [x2, T2, c2] = normalise2dpts2(x2)
    
    # % Note that it may have not been possible to normalise
    # % the points if one was at infinity so the following does not
    # % assume that scale parameter w = 1.
    
    Npts = max(x1.shape)
    # A = zeros(3*Npts,9);
    A = np.zeros((3*Npts,9), dtype=int)
    
    # O = [0 0 0];
    O = np.asmatrix([0,0,0])


    for n in range(0,Npts):
        X = x1[:,n].transpose()
        X = X.reshape(1,x1.shape[0])
        x = x2[0,n]
        y = x2[1,n]
        w = x2[2,n]
        # # temp1 = w*X
        # w = np.asmatrix(w)
        # X = np.asmatrix(X)
        # y = np.asmatrix(y)
        # x = np.asmatrix(x)
        
        # print "Shapes (w,X,y,x) : \n"
        # print w.shape, X.shape, y.shape, x.shape

        temp1 = w*X

        # temp1 = X.temp1(1,x1.shape[0])
        temp2 = y*X
        # temp2 = X.temp2(1,x1.shape[0])
        temp3 = x*X
        # temp3 = X.temp3(1,x1.shape[0])
        # print "temp1, temp2, temp3 \n"
        # print temp1, temp2, temp3
        # cv2.waitKey(0)
        A[3*n-2][:] = np.concatenate((O,-temp1,temp2), axis = 1)
        # A[3*n-2][:] = np.asmatrix([0, -temp1, temp2])
        
        A[3*n-1][:] = np.concatenate((temp1,O,-temp3), axis = 1)
        # A[3*n-2][:] = np.asmatrix([temp1, O, -temp3])
        A[3*n][:] = np.concatenate((-temp2,temp3,O), axis = 1)
        # A[3*n-2][:] = np.asmatrix([-temp2, temp3, O])
    
    
    # [U,D,V] = svd(A,0); #% 'Economy' decomposition for speed
    # https://in.mathworks.com/help/matlab/ref/svd.html 
    if A.shape[0] > A.shape[1]:
        U, temp_D, V = np.linalg.svd(A)  #Economy decomposition
        U = U[:,0:A.shape[1]] # since m > n, only first n columns
        D = np.diag(temp_D) # diagonalizing the vector into a matrix

    else:
        U, D, V = np.linalg.svd(A)

    # % Extract homography
    # H = reshape(V(:,9),3,3)';
    H_transpose = V[:,8].reshape(3,3)
    H = H_transpose.transpose()

    # % Denormalise
    # H = T2\H*T1;
    # Matrix Multiplication using dot as given as https://stackoverflow.com/questions/21562986/numpy-matrix-vector-multiplication

    temp = np.linalg.pinv(T2)
    H = temp.dot((H.dot(T1)))
    return H

# %--------------------------------------------------------------------------
# % Function to check argument values and set defaults

# function [x1, x2] = checkargs(arg);
############################################### Cell array in MATLAB equivalent?? #######################
def checkargs(arg):
    
    # if length(arg) == 2
    print "Entering checkargs"
    if len(arg) == 2:
        # x1 = arg{1}:
        # x2 = arg{2};
        print "We have 2 args"
        x1 = np.asmatrix(arg[0])
        x2 = np.asmatrix(arg[1])

        # x1 = np.asmatrix(x1)
        # x2 = np.asmatrix(x2)
	    
        # print x1
        # print x2

        if x1.shape != x2.shape:
            raise ValueError('x1 and x2 must have the same size')
        elif (x1.shape[0] != 3):
	        raise ValueError('x1 and x2 must be 3xN')
	
	
    elif len(arg) == 1:
        print "We have 1 args"
        # if size(arg{1},1) ~= 6
        
        if arg[0].shape[0] != 6:
	        raise ValueError('Single argument x must be 6xN') 
        else:
            x1 = arg[0]
            # print "arg[0] before change", arg
            x1 = x1[0:3,:]
            # print "arg[0] after change", arg
            x2 = arg[0]
            x2 = x2[3:6,:]
	        # x1 = arg{1}(1:3,:);
	        # x2 = arg{1}(4:6,:);  
	
    else:
	    raise ValueError('Wrong number of arguments supplied')
    # print x1, "\n"
    # print x2, "\n"
    
    return x1, x2

# a1 = np.array([1,2,3])
# a2 = np.array([[1,2,3],[4,5,6]])

# checkargs(a1,a2)