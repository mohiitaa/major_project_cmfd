import cv2
import numpy as np

img=cv2.imread('tampered1.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow('yay', gray)
cv2.waitKey()

cv2.destroyAllWindows()

print "Hello"

sift=cv2.xfeatures2d.SIFT_create()
kp, des=sift.detectAndCompute(gray,None)

print len(des[0])
index = []
for i in kp:
	temp = (i.pt, i.size, i.angle)
	index.append(temp)

# print index[0][0][0]

locs = np.zeros((len(kp),4), dtype = float)
descs = np.zeros((len(kp), 128), dtype = float)

i = 0
for i in range(len(index)):
    locs[i][0] = index[i][0][0]
    locs[i][1] = index[i][0][1]
    locs[i][2] = index[i][1]
    locs[i][3] = index[i][2]
    # locs[i][0] = temp_2[0]

descs = np.array(des)
print len(des)
print descs[0:2]