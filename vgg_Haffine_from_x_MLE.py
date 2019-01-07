
import numpy as np 
import sys
import vgg_get_nonhomg as vgn
import vgg_condition_2d as vc2

def vgg_Haffine_from_x_MLE(xs1,xs2):

	if (xs1.shape!=xs2.shape):
		raise ValueError("Input point sets are different sizes!")

	nonhomg = np.asmatrix(vgn.vgg_get_nonhomg(xs1))
	nonhomg_t=nonhomg.transpose()
	means=nonhomg_t.mean(0)
	stddev=nonhomg_t.std(0) # Or 1?
	#stddev_t=stddev.transpose()
	#maxstds=stddev_t.max(0)
	maxstds=stddev.max(1)
	#maxstds=maxstds.item(0)
	C1=np.diag([1/maxstds, 1/maxstds, 1])
	#temp=np.matrix([-(means.item(0))/(maxstds), -(means.item(0))/(maxstds), 1])
	#C1[:,2]=temp
	C1[:,2]=np.concatenate((-means/maxstds,np.asmatrix(1)),axis=1)
	

	nonhomg = vgn.vgg_get_nonhomg(xs2);
	nonhomg_t=nonhomg.transpose()
	means=nonhomg_t.mean(0)
	C2=C1
	#temp=np.matrix([-(means.item(0))/(maxstds), -(means.item(0))/(maxstds), 1])
	#C2[:,2]=temp
	C2[:,2]=np.concatenate((-means/maxstds,np.asmatrix(1)),axis=1)
	

	xs1 = vc2.vgg_condition_2d(xs1,C1);
	xs2 = vc2.vgg_condition_2d(xs2,C2);

	xs1nh = vgn.vgg_get_nonhomg(xs1);
	xs2nh = vgn.vgg_get_nonhomg(xs2);

	xsnh1=np.asmatrix(xsnh1)
	xsnh2=np.asmatrix(xsnh2)
	A=np.concatenate((xsnh1,xsnh2))
	A=A.transpose()


	[u,s,v]=np.linalg.svd(A)
	s=np.diag(s)

	nullspace_dimension=np.sum(s<s[:,2]*sys.float_info.epsilon*1000)
	if (nullspace_dimension>2):
		print "Nullspace is a bit roomy...";
	
	B=v[0:2,0:2]
	C=v[2:4,0:2]

	temp=np.asmatrix(C*np.linalg.pinv(B))
	temp=np.concatenate((temp,np.zeros([2,1])),axis=1)
	H=np.concatenate((temp,np.asmatrix([0,0,1])))

	H=(np.linalg.inv(C2))*H*C1
	H=H/H[2,2]
	return H

	





	




