# p = vgg_get_nonhomg(h)
#
# Convert a set of non-homogeneous points to homogeneous form
# Points are stored as column vectors, stacked horizontally, e.g.
#  [x0 x1 x2 ... xn ;
#   y0 y1 y2 ... yn ;
#   w0 w1 w2 ... wn ]

#x=[[1,2],[3,4],[4,5]]
import numpy as np

def vgg_get_homg(x):
    
    if len(x)==0:
        y=[]
        return y

    y = np.concatenate((x, np.ones([1,x.shape[1]]))

    return y


