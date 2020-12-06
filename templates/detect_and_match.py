# -*- coding: utf-8 -*-
import numpy as np
import cv2 
import matplotlib.pyplot as plt


def detect_and_match(img1, img2, Draw = 0):
    """
    This function detects keypoints between 2 images using a corner detection method
    and then applies SIFT to determine scale-invariant descriptors that can be used to
    find matches between the images. After doing this, the kNN matching algorithm 
    is applied with 2 nearest neighbours. The 2 matches per keypoint of the left image
    are then passed through Lowe's ratio test to determine one true match (removes False positives).
    
    Inputs:
    
    imgL: A uint8 RGB image
    imgR: A uint8 RGB image
    Draw: Plot matches and two images to visualize function working (default = no visualization)
    
    Returns:
    
    Matches: List of matches that pass the ratio test
    
    """
    #Make sure image types are uint8
    imgL = img1.astype(np.uint8)  
    imgR = img2.astype(np.uint8) 
    
    #Convert images to grayscale
    gray_left = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    
    #Create instance of SIFT and find keypoints in left and right grayscale images
    surf = cv2.(400)
    kpL, desL = surf.detectAndCompute(gray_left,None)
    kpR, desR = surf.detectAndCompute(gray_right,None)
    return len(kpL)
#     #Match keypoints using KNN, with a neighbourhood of 2
#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(desL, desR, k=2)
    
#     # Apply ratio test
#     Matches = []
#     for m,n in matches:
#         if m.distance < 0.75*n.distance:
#             good.append([m])
    
#     if Draw != 0:
#         img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2)
#         plt.imshow(img3),plt.show()
#     else:
#         pass

#     return Matches
    

img1 = plt.imread('Frame_0.png')
img2 = plt.imread('Frame_2.png')
img1 = img1*255
img2 = img2*255     
        
kpL = detect_and_match(img1, img2)
print(kpL)



