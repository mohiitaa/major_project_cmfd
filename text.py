import numpy as np

def checkargs(*argv):
    

    # if length(arg) == 2
    print argv
    print argv[0]
    print argv[1]
    if (len(argv)==2):
	    
		x1=np.asmatrix(argv[0])
		x2=np.asmatrix(argv[1])
		print x1
		print x2
		if (x1.shape!=x2.shape):
			raise ValueError('x1 and x2 must have the same size')
        else(x1.shape[0] != 3):
			raise ValueError('Hiiii x1 and x2')
	elif (len(argv)==1):
		t=np.asmatrix(argv[0])
		if (t.shape[0]!=6):
			raise ValueError("helloooo")
		else:
			x=np.asmatrix(argv[0])
			x1=x[0:3,:]
			x2=x[3:6,:]
			print x1
			print x2  
			
	else:
		raise ValueError('wrong no of args')
 
    return x1,x2

a1 = np.array([1,2,3])
a2 = np.array([[1,2],[3,4]])
x1, x2 = checkargs(a1,a2)    