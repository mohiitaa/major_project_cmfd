# % MATCH_FEATURES: Match SIFT features in a single image using our multiple
# %                 match strategy.
# % 
# % INPUTS:
# %   filename        - image filename (if features have to be computed) or
# %                     descriptors filename in the other case
# %  
# % OUTPUTS:
# %   num             - number of matches
# %   p1,p2           - coordinates of pair of matches
# %   tp              - computational time    
# %
# % EXAMPLES:
# %   e.g. extract features and matches from a '.jpg' image:
# %   [num p1 p2 tp] = match_features('examples/tampered1.jpg')
# %
# %   e.g. import features and matches from a '.sift' descriptors file:
# %   [num p1 p2 tp] = match_features('examples/tampered2.sift',0)                      
# % 
# % ---
# % Authors: I. Amerini, L. Ballan, L. Del Tongo, G. Serra
# % Media Integration and Communication Center
# % University of Florence
# % May 7, 2012
import cv2
import scipy as sp
from scipy import spatial
import numpy as np
import numpy.matlib
from numpy import linalg as LA
import timeit
import os
from import_sift import import_sift
import copy
# code you want to evaluate
# 
# function [num p1 p2 tp] = match_features(filename, filesift)
def match_features(*argv):
    # thresholds used for g2NN test
    
    dr2 = 0.85
    filename = argv[0]
    filesift = argv[1]
    extract_feat = 1 #by default extract SIFT features using Hess' code
    # FIND PYTHON EQUIVALENT exist('filesift','var') means it checks filesift exists in workspace. No workspace in python
    #### if(exist('filesift','var') and (filesift != 'nofile'))

    #####################################################################################
    # if (len(argv) == 2 and filesift != 'nofile'):
    #     extract_feat = 0; 


    start_time = timeit.default_timer()

    # if (extract_feat == 1):
    #     # sift_bin = fullfile('lib','sift','bin','siftfeat');   #e.g. 'lib/sift/bin/siftfeat' in linux
    #     sift_bin =  os.path.join('lib','sift','bin','siftfeat')
    #     #[pf,nf,ef] = fileparts(filename);
    #     f_name = os.path.basename(filename)
    #     pf,nf_temp = os.path.split(f_name)
    #     nf,ef = os.path.splitext(nf_temp)

    #     #desc_file = [fullfile(pf,nf) '.txt'];
    #     desc_file = os.path.join(pf,nf) + '.txt'
    
    #     im1=cv2.imread(filename)
    #     im1_row, im1_col = im1.shape[:2]

    #     if (im1_row<=1000 and im1_col<=1000):
    #         command=sift_bin + ' -x -o '+ desc_file + ' '+ filename
    #         status1 = os.system(command)
    #         # Find python and windows equivalent 
    #         #status1=0
            
    #     else:
    #         #status1 = system([sift_bin ' -d -x -o ' desc_file ' ' filename])
    #         # Find python and windows equivalent 
    #         #status1=0
    #         command=sift_bin+' -d -x -o ' +desc_file+ ' '+ filename
    #         status1=os.system(command)

    #     if status1 != 0 : 
    #         raise ValueError('error calling executables')
        

    #     # import sift descriptors
    #     num, locs, descs = import_sift(desc_file)
    #     #### system(['rm ' desc_file])
    #     # Find python and windows equivalent 

    # else:
    #     # import sift descriptors
    #     num, locs, descs = import_sift(filesift)

    ############################################################################
    im1=cv2.imread(filename)

    im1_row, im1_col = im1.shape[:2]
    im_gr = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    d = cv2.xfeatures2d.SIFT_create()
    kp, des = d.detectAndCompute(im_gr, None)

    print "Number of keypoints:", len(kp)
    # cv2.drawKeypoints(im_gr,kp,im_gr)
    # print "Showing the image now"
    # cv2.namedWindow('image_kp', cv2.WINDOW_NORMAL)
    # cv2.imshow('image_kp',im_gr)
    # cv2.waitKey()


    num = len(des)
    # print num
    locs = np.zeros((num,4), dtype = float)
    descs = np.zeros((num, 128), dtype = float)

    index = []
    for i in kp:
	    temp_1 = (i.pt, i.size, i.angle) 
	    index.append(temp_1)

    i = 0
    for i in range(len(index)):
        locs[i][0] = index[i][0][0]
        locs[i][1] = index[i][0][1]
        locs[i][2] = index[i][1]
        locs[i][3] = index[i][2]
         
        
    descs = np.asmatrix(des)


    if (num==0):
        p1=[]
        # p1 = np.array(p1)
        p2=[]
        # p2 = np.array(p2)
        tp=[]
        tp = np.array(tp)
    else:
        p1=[]
        # p1 = np.array(p1)
        p2=[]
        # p2 = np.array(p2)
        num=0
    
    # load data

    # loc1 = locs(:,1:2)
    loc1 = locs[:,0:2]
    # %scale1 = locs(:,3);
    # %ori1 = locs(:,4);
    des1 = copy.deepcopy(descs)
    # print len(descs)
    # descriptor are normalized with norm-2
    des1_row, des1_col = des1.shape[:2]

    print "des1_row = ", des1_row
    if (des1_row <15000):
        des1_mult = np.matmul(des1,des1.transpose())
        temp_2 = np.asmatrix(np.sqrt(des1_mult.diagonal()))

        des1 = np.divide(des1, np.matlib.repmat(temp_2.transpose(),1,des1_col))
        # print len(des1)
    # des1_mult = np.matmul(des1,des1.transpose())
    # temp_2 = np.asmatrix(np.sqrt(des1_mult.diagonal()))

    # des1 = np.divide(des1, np.matlib.repmat(temp_2.transpose(),1,des1_col))
    
    else:
        des1_norm = des1 
        print "Hey I am in else"
        # for j in range(1,des1_col):
        for j in range(0,des1_col):
            # des1_j = des1_norm(j,:)  ### WHAT???
            des1_j = des1_norm[j,:]
            des1_norm[j,:] = np.divide(des1_j,LA.norm(des1_j))
            
            if j == 2832:
                print "Normalised value is:", des1_norm[j,:]
        
        # des1 = des1_norm
        
    des1= np.asmatrix(des1)
    # print des1.shape
    #sift matching
    des2t = des1.transpose()
    match=np.zeros((1,des1_row))
    count = 0
    # precompute matrix transpose
    if (des1_row > 1): #start the matching procedure iff there are at least 2 points
        for i in range(0,des1_row):#MATLAB Array indexing starts with 1, Compensating
            dotprods = np.matmul(des1[i,:],des2t)     #Computes vector of dot products #### CHECK THIS ONCE
 ### This abovr takes the cross correlation between descriptors of keypoint
 
 
 ###########  24 Oct 2018 ##############
            #vals,indx = sort(np.arccos(dotprods))  #Take inverse cosine and sort results  ### Don't know val index returning sort function
            inv_cos=np.arccos(dotprods)
            # print inv_cos
            vals=np.sort(inv_cos)
            indx=np.argsort(inv_cos)
            

            # print vals
            # j=2
            j = 1
            
            # while vals[j]<dr2* vals[j+1]: 
            # print vals.shape
            if (i == 2832):
                print "vals for i", i, 'is', vals
            # vals = np.array(vals)
            while (vals[0,j]<dr2*vals[0,j+1]): 
                j = j+1
                count = count + 1

            # print "count is : ", count    

            # print "HEYYaaa"
            # print "j is ", j
            # print indx.shape
            for k in range(1,j-1):
                # print "HEYYYYYY"
                # print i,  j,  k
                # print match.shape
                # print "_______________________________k: ", k
                match[0,i] = indx[0,k]
                # print match[0,i]
                temp_3 = np.array([[loc1[i,0] , loc1[i,1]],[loc1[int(match[0,i]),0] , loc1[int(match[0,i]),1]]])
                temp_3 = np.asmatrix(temp_3)
                # print "temp_3", temp_3
                
                # if pdist([loc1(i,1) loc1(i,2); loc1(match(i),1) loc1(match(i),2)]) >10  
                # if (LA.norm(np.array([ loc1[i,1] , loc1[i,2] ]) - np.array([ loc1[match[i],1] , loc1[match[i],2] ]) )):
                # print "SPATIAL DIST: for i = ",i,"is___" , spatial.distance.pdist(temp_3,'euclidean')
                if(spatial.distance.pdist(temp_3,'euclidean') > 10):
                    
                    temp1 = np.asmatrix([loc1[i,1], loc1[i,0], 1])
                    # p1 = np.array([p1, temp1.transpose()])
                    # p1.append(temp1.transpose())
                    if(p1 == []):
                        p1 = temp1.transpose()
                        # print "IN THE IFFFFFFFFFFF"
                        p1 = np.asmatrix(p1)
                    else:
                        # print "##############p1:", p1
                        # print "IN THE ELSEEEEEEEEEE"
                        p1 = np.concatenate((p1,temp1.transpose()),axis= 1)

                    temp2 = np.asmatrix([loc1[int(match[0,i]),1], loc1[int(match[0,i]),0], 1])
                    # p2 = np.array([p2, temp2.transpose()])
                    if(p2 == []):
                        p2 = temp2.transpose()
                        p2 = np.asmatrix(p2)
                    else:    
                        p2 = np.concatenate((p2,temp2.transpose()),axis = 1)
                    # p2.append(temp2.transpose())

                    num=num+1
                
            
        
       
    # tp = toc; % processing time (features + matching)
    
    # print elapsed
    # print p1
    # print p2
    p1 = np.asmatrix(p1)
    p2 = np.asmatrix(p2)
    p1_row, p1_col = p1.shape
    
    # f1=open('./match_features_testfile_if.txt', 'w+')
    # f2=open('./match_features_testfile_else.txt', 'w+')
    tp= timeit.default_timer() - start_time
    # print tp
    if (p1_row==0):
        # fprintf('Found %d matches.\n', num); ########################################
        # print >>f1, 'Found %d matches.\n', num
        print 'Found', num, 'matches'
        # print 'we are in the IF'
    else:
        # p=np.array([p1[0:2,:].transpose(), p2[0:2,:].transpose()]);
        # print "p1___", p1.shape
        # print "p2___", p2.shape
        
        temp_1 = p1[0:2,:]
        temp_2 = p2[0:2,:]
        # print "temp1___", temp_1
        # print "temp2___", temp_2
        # print temp_1
        # print "((((_(__________________________________)))))"
        # print temp_2
        p=np.concatenate((temp_1.transpose(), temp_2.transpose()),axis = 1)
        # print p
        p= np.unique(p, axis = 0)
        # p= np.unique(p)
        # print p
        # p1=np.array([p[:,0:2].transpose(), np.ones((p.shape[0],), dtype = int)]).transpose();
        temp3 = np.ones((1,p.shape[0]), dtype = int)
        temp3 = np.asmatrix(temp3)
        # print "temp3 shape:  ",temp3.shape
        # temp3 = temp3.reshape(1,2)
        temp3 = np.asmatrix(temp3)
        p1 = np.concatenate((p[:,0:2].transpose(), temp3))
        # temp3 = np.asmatrix(temp3)
        # p2=[p(:,3:4)'; ones(1,size(p,1))];
        temp4 = np.ones((p.shape[0],), dtype = int)
        temp4 = np.asmatrix(temp4)
        # temp4 = temp4.reshape(1,2)
        p2 = np.concatenate((p[:,2:4].transpose(), temp4))
        # num=size(p1,2);
        num = p1.shape[1]
        # fprintf('Found %d matches.\n', num);
        print 'Found', num, 'Matches'
        # print 'we are in the ELSE'

    return num, p1, p2, tp