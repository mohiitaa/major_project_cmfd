# p = vgg_get_nonhomg(h)
#
# Convert a set of homogeneous points to non-homogeneous form
# Points are stored as column vectors, stacked horizontally, e.g.
#  [x0 x1 x2 ... xn ;
#   y0 y1 y2 ... yn ;
#   w0 w1 w2 ... wn ]

#x=[[1,2],[3,4],[4,5]]
import numpy as np

def vgg_get_nonhomg(x):
    
    if len(x)==0:
        x=[]
        return x

    d=len(x)-1
    y=np.ones((d,1))*x[d]  #---duplicates the last row d times
    x = x[0:d]             #---modifies x to remove the last row

    for i in range(len(y)):
        for j in range(len(y[0])):
            x[i][j]=x[i][j]/y[i][j]   #---elementwise division of x by y

    return x


