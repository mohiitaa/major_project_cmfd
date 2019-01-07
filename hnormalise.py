# % HNORMALISE - Normalises array of homogeneous coordinates to a scale of 1
# %
# % Usage:  nx = hnormalise(x)
# %
# % Argument:
# %         x  - an Nxnpts array of homogeneous coordinates.
# %
# % Returns:
# %         nx - an Nxnpts array of homogeneous coordinates rescaled so
# %              that the scale values nx(N,:) are all 1.
# %
# % Note that any homogeneous coordinates at infinity (having a scale value of
# % 0) are left unchanged.

# % Peter Kovesi  
# % School of Computer Science & Software Engineering
# % The University of Western Australia
# % pk at csse uwa edu au
# % http://www.csse.uwa.edu.au/~pk
# %
# % February 2004
import numpy as np
import copy
import sys

epsilon = sys.float_info.epsilon
# function nx = hnormalise(x)
def hnormalise(x):
    ## Accepts Numpy array
    rows,npts = x.shape
    nx = copy.deepcopy(x)
    nx = np.asmatrix(nx)

    # % Find the indices of the points that are not at infinity
    # finiteind = find(abs(x[rows,:]) > eps);

    finiteind = np.where(np.absolute(x[rows-1,:]) > epsilon)

    finiteind = np.asmatrix(finiteind)
    if max(finiteind.shape) != npts:
        raise Warning('Some points are at infinity');
    
    print "nx.shape", nx.shape
    print "finiteind", finiteind
    print "x", x

    
    # % Normalise points not at infinity
    for r in range(0,rows):
        for j in range(0,max(finiteind.shape)):
            nx[r,finiteind[0,j].astype(int)] = x[r,finiteind[0,j].astype(int)]/x[rows-1,finiteind[0,j].astype(int)]

    nx[rows-1,finiteind.astype(int)] = 1
    # for j in range(0,max(finiteind.shape)):
    #         nx[rows-1,int(finiteind[0,j])] = x[r,int(finiteind[0,j])]/x[rows-1,int(finiteind[0,j])]
   
    return nx

