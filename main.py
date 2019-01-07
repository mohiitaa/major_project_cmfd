import process_image as pi
import sys

filename=str(sys.argv[1])
[num_gt,inliers1, inliers2]=pi.process_image(filename)