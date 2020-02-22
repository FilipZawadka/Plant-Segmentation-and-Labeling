# Filip Zawadka 290569
# Project 1
# Plant Segmentation and Labeling

import cv2
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline #only for my compiler
import math
import sklearn.metrics as sm

def display_img(img,cmap=None):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap)
    
def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])*(p0[0] - p1[0]) + (p0[1] - p1[1])*(p0[1] - p1[1]))

for a in range(3):
    for b in range(5):
        for c in range(10):
            for d in range(6):
                img = cv2.imread('multi_plant/rgb_0'+str(a)+'_0'+str(b)+'_00'+str(c)+'_0'+str(d)+'.png')

                hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

                GREEN_MIN = np.array([55, 45, 25],np.uint8)
                GREEN_MAX = np.array([75, 255, 255],np.uint8)

                frame_threshed = cv2.inRange(hsv_img, GREEN_MIN, GREEN_MAX)
                kernel=np.ones((5,5),np.uint8)

                median = cv2.medianBlur(frame_threshed,5)

                closing = cv2.morphologyEx(median, cv2.MORPH_CLOSE, kernel)

                opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

                image, contours , hierarchy = cv2.findContours(opening,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
                # Set up empty array
                l_contours = np.zeros(image.shape)
               

                # For every entry in contours
                for i in range(len(contours)):
                    for k in range(len(contours[i])):

                        if(distance((240,240),(contours[i][k][0]))<100):
                            cv2.drawContours(l_contours, contours, i, 255, -1)
                            break
               
         #Now draw all of the inside contours in case of overlaping leaves

                for i in range(len(contours)):
                    if hierarchy[0][i][3] != -1:
                        cv2.drawContours(l_contours, contours, i, 0, -1)

                mask = np.uint8(l_contours)
                cv2.imwrite ('mask/rgb_0'+str(a)+'_0'+str(b)+'_00'+str(c)+'_0'+str(d)+'.png',mask)
                
                sure_bg = cv2.dilate(l_contours,kernel,iterations=3)
                dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2,5)
                ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)

                sure_fg = np.uint8(sure_fg)
                sure_bg = np.uint8(sure_bg)
                unknown = cv2.subtract(sure_bg,sure_fg)

                # Marker labelling
                ret, markers = cv2.connectedComponents(sure_fg)
                # Add one to all labels so that sure background is not 0, but 1
                markers = markers+1
                # Now, mark the region of unknown with zero
                markers[unknown==255] = 0

                markers = cv2.watershed(img,markers)
                
                #Change the markers to bgr image like in the multi_label folder
                mask_rgb = np.zeros((480,480,3),np.uint8)
                
                for i in range(480):
                    for k in range(480):
                        if markers[i][k] == 2:#g-green
                            mask_rgb[i][k][1]=255 
                        elif markers[i][k] == 3:#b-blue
                            mask_rgb[i][k][0]=255 
                        elif markers[i][k] == 4:#gb-light blue/cyan
                            mask_rgb[i][k][1]=255 
                            mask_rgb[i][k][0]=255 
                        elif markers[i][k] == 5:#r-red
                            mask_rgb[i][k][2]=255 
                        elif markers[i][k] == 6:#rb-violet
                            mask_rgb[i][k][0]=255
                            mask_rgb[i][k][2]=255
                        elif markers[i][k] == 7:#rg-yellow
                            mask_rgb[i][k][2]=255
                            mask_rgb[i][k][1]=255
                        elif markers[i][k] == 8:#rgb-white
                            mask_rgb[i][k][0]=255
                            mask_rgb[i][k][1]=255 
                            mask_rgb[i][k][2]=255 
                cv2.imwrite ('rgb_mask/rgb_0'+str(a)+'_0'+str(b)+'_00'+str(c)+'_0'+str(d)+'.png',mask_rgb)


#Computing the Similarity Scores

jss=0 #Mean Jaccard Similarity Score
dice=0 #Mean Sice Similarity Score

dice = float(dice)
for a in range(3):
    for b in range(5):
        for c in range(10):
            for d in range(6):
                img = cv2.imread('multi_label/label_0'+str(a)+'_0'+str(b)+'_00'+str(c)+'_0'+str(d)+'.png')
                myimg = cv2.imread('rgb_mask/rgb_0'+str(a)+'_0'+str(b)+'_00'+str(c)+'_0'+str(d)+'.png')
                #resize to make sure they are of the correct size
                img.resize((480,480,3))
                myimg.resize((480,480,3))
                jss = jss + (sm.jaccard_similarity_score(img.ravel(), myimg.ravel()))
                band = cv2.bitwise_and(myimg,img)
                dice = dice + float(2*(sum(band.ravel()/255)))/((sum(img.ravel()/255)+sum(myimg.ravel()/255)))
jss = jss/900
dice = float(dice)/900
print(jss)
print(dice) 
