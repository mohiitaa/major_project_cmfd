# %figure; imshow(dispMap);
import numpy as np
import cv2 
# import gaussian2d as gs 
# import cv2
# import numpy as np

def tampering_localization(*argv):
    if (len(argv)==5):
        th_bin_mask=0.3
        show_mask=0
    if (len(argv)==6):
        show_mask=0

    im= cv2.imread('filename')
    dispMap[np.isnan(dispMap)]=0;
    dispMap[dispMap<0]=0;
    dispMapG = gray2ind(dispMap, 256);  ############## What is gray2ind in python
    # G = gs.gaussian2D((3,3),0.5);

    # % filtering image
    dispMapG = cv2.GaussianBlur(dispMap,(7,7),0.5)
    # dispMapG = imfilter(dispMapG,G,'same');

    # % fill holes
    # bw = im2bw(dispMapG,th_bin_mask);
    im_bw = cv2.threshold(dispMapG, th_bin_mask, 255, cv2.THRESH_BINARY)[1]

    # bw = imfill(bw,'holes');
    im_floodfill = im_bw.copy()


    h, w = im_bw.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);

    # Invert floodfilled image
    bw = cv2.bitwise_not(im_floodfill)
    # %hold on; imshow(bw);

    # % discard regions not containing any sift match
    # bound = bwboundaries(bw);
    im2, bound, hierarchy = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ### Use bound from here


    #####################################################################################
    # 	inModel_x = [ z1(1,inliers) z2(1,inliers)]';
    #  	inModel_y = [ z1(2,inliers) z2(2,inliers)]';

    #     img_out = false(size(im,1),size(im,2));    
    #     for k=1:size(bound,1)
    # 		b= bound{k};
    #         in = inpolygon(inModel_x,inModel_y,b(:,2),b(:,1));
    #         if ~isempty(find(in,1))
    #             bw_b =false(size(im,1),size(im,2)); 
    #             bw_b = roipoly(bw_b,b(:,2),b(:,1));
    #             img_out = img_out | bw_b;
    #         end
    #     end

    #     % show localization binary mask
    #     if show_mask
    #         figure; imshow(img_out);
    #     end

    # end
    # Subbulaaaxmi Narayanan
    bound = bwboundaries(bw); ########
    inModel_x =np.concatenate((z1[1,inliers], z2[1,inliers]),axis=1)
    inModel_x=inModel_x.transpose()
    inModel_y =np.concatenate((z1[2,inliers], z2[2,inliers]),axis=1)
    inModel_y=inModel_y.transpose()

    img_out=np.zeros([im.shape[0], im.shape[1]])
    for k in range(0,bound.shape[0]):
        b=bound[k]######
        inp=inpolygon(inModel_x,inModel_y, b[:,1],b[:,0])#####
        find_nonzero=np.nonzero(inp)
        if (find_nonzero!=[]):
            bw_b=np.zeros(im.shape[0],im.shape[1])
            bw_b = roipoly(bw_b,b(:,2),b(:,1));  ####
            img_out=np.logical_or(img_out,bw_b)
            img_out=1*img_out
            
    if (show_mask):
        fig = plt.figure()
        plt.imshow(img_out)
        
    return img_out