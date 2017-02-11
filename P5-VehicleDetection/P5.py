#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 21:29:47 2017

@author: yifei
"""


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from lesson_functions import *
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import time
from collections import deque



# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='YCrCb', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, color_space='YCrCb', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        # resize test_img to (64, 64) because the classifier is trained with image of size (64, 64)
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        

        #5) Scale extracted features to be fed to classifier
        test_features = np.array(features).reshape(1, -1)
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

# remove false_positives that has an area less than 5000
def remove_false_positives(labels):
    bboxes = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        if (bbox[1][0]-bbox[0][0]+1)*(bbox[1][1]-bbox[0][1]+1) > 4000 and (bbox[1][0]-bbox[0][0]+1)*(bbox[1][1]-bbox[0][1]+1) < 40000:
            bboxes.append(bbox)
    # Return the bounding bounding boses
    return bboxes

class Vehicle():
    def __init__(self):
        self.current_position = []
        self.previous_position=deque()
        self.best_position = []
        self.number = 0
        
    
    def current(self, bboxes):
        self.current_position = bboxes

    
    def best(self):
        # Method 1:
        # Calculate the average of last 5 frames
#        best_bboxes = []
#        if self.current_position == []:
#            best_bboxes = []
#        elif len(self.previous_position) < 5:
#            self.previous_position.appendleft(self.current_position)
#            l = list(self.previous_position)
#            self.best_position = np.mean(l, axis=0).astype(int)
#        else:
#            self.previous_position.appendleft(self.current_position)
#            l = list(self.previous_position)
#            self.best_position = np.mean(l, axis=0).astype(int)
#            self.previous_position.pop()
#            
#        for i in range(len(self.best_position)):
#            best_bboxes.append(tuple(map(tuple, self.best_position[i])))
            
        
        # Method 2:
        # Heatmaps thresholding
         if self.current_position == []:
             best_bboxes = []
             self.best_position = best_bboxes
             
         elif len(self.previous_position) <5:
             heatmap = np.zeros((720, 1280), np.uint8)  
             for each in self.current_position:
                 self.previous_position.appendleft(each)
            
             l = list(self.previous_position)
             add_heat(heatmap, l)
             apply_threshold(heatmap, 3)
             labels = label(heatmap)
             best_bboxes = remove_false_positives(labels)
             self.best_position = best_bboxes
             
         else:
             heatmap = np.zeros((720, 1280), np.uint8)  
             for each in self.current_position:
                 self.previous_position.appendleft(each)
             l = list(self.previous_position)
             self.previous_position.pop()
             add_heat(heatmap, l)
             apply_threshold(heatmap, 5)  
             labels = label(heatmap)
             best_bboxes = remove_false_positives(labels) 
             self.best_position = best_bboxes
             
             if len(best_bboxes)==1:
                 if (best_bboxes[0][1][0]-best_bboxes[0][0][0]+1)*(best_bboxes[0][1][1]-best_bboxes[0][0][1]+1) > 40000:
                     self.previous_position = deque()
                     self.best_position = []
             elif len(best_bboxes)==2:
                 if (best_bboxes[0][1][0]-best_bboxes[0][0][0]+1)*(best_bboxes[0][1][1]-best_bboxes[0][0][1]+1) > 30000 or (best_bboxes[1][1][0]-best_bboxes[1][0][0]+1)*(best_bboxes[1][1][1]-best_bboxes[1][0][1]+1) > 30000:
                     self.previous_position = deque()
                     self.best_position = []
                 
             else:
                 return self.best_position
         return self.best_position

def process(image):

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
#            image = image.astype(np.float32)/255

    draw_image = np.copy(image)
    
    heatmap = np.zeros((image.shape[0], image.shape[1]), np.uint8)            
    
    for each in pyramid:
        windows = slide_window(image, x_start_stop=each[1], y_start_stop=each[2], 
                        xy_window=each[0], xy_overlap=each[3])
        hot_windows = search_windows(image, windows, svc, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
#        draw_image = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)  
        heatmap = add_heat(heatmap, hot_windows)
   
    heatmap = apply_threshold(heatmap,2)   
    labels = label(heatmap)  
    # remove false positives
    new_bboxes = remove_false_positives(labels)
    
    
    
    # Method 2
    # Heatmap thresholding
    number_of_cars = len(new_bboxes)
    if number_of_cars < 3:
        car1.current(new_bboxes)
        car1.best()
        final_result = draw_boxes(draw_image, car1.best_position)
        
    else:
        final_result = draw_boxes(draw_image, car1.best_position)
    
#    # Method 1
#    # Calculate the average of last 5 frames 
#    number_of_cars = len(new_bboxes)
#    
#    if number_of_cars < 2:
#        # average car bounding boxes of current and last 5 frames
#        car1.current(new_bboxes)
#        best_bboxes = car1.best()
#        final_result = draw_boxes(draw_image, best_bboxes)
#        
#    elif number_of_cars ==2:
#        # average car bounding boxes of current and last 5 frames
#        car2.current(new_bboxes)
#        best_bboxes = car2.best()
#        final_result = draw_boxes(draw_image, best_bboxes)
##
##    elif number_of_cars <4:
##        
##        car1.current(np.array(new_bboxes[1]))
##        car2.current(np.array(new_bboxes[0]))
##        final_result = draw_boxes(draw_image, car1.best())
##        final_result = draw_boxes(final_result, car2.best())
#    else:
#        final_result = draw_image
#  
    
    
    
    return final_result
    

if __name__ == "__main__":  
    
    # load model
    filename = "./models/model_yuv_982all.pkl"
    svc = joblib.load(filename)

    color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 16    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off

    global pyramid
    pyramid = [((64,64), [650, None], [400, 500], (0.8, 0.8)), 
                       ((96,96), [650, None], [500, 600], (0.7, 0.7)), 
                       ((128,128), [650, None], [500, 650], (0.8, 0.8)), 
                       ((160, 160), [650, None], [550, None], (0.6, 0.6))]
    
    test_mode = 'video'
    
    if test_mode == 'image':
        images = glob.glob("./test_images/test*.jpg")
        print('Using:',orient,'orientations',pix_per_cell, 'pixels per cell and', cell_per_block,'cells per block')
        
        
        for each in images:
            time1 = time.time()
            image = mpimg.imread(each)
            # Uncomment the following line if you extracted training
            # data from .png images (scaled 0 to 1 by mpimg) and the
            # image you are searching is a .jpg (scaled 0 to 255)
#            image = image.astype(np.float32)/255

            print("test image shape is:", image.shape)
            draw_image = np.copy(image)
            
            heatmap = np.zeros((image.shape[0], image.shape[1]), np.uint8)            
            
            for each in pyramid:
                windows = slide_window(image, x_start_stop=each[1], y_start_stop=each[2], 
                                xy_window=each[0], xy_overlap=each[3])
                hot_windows = search_windows(image, windows, svc, color_space=color_space, 
                                    spatial_size=spatial_size, hist_bins=hist_bins, 
                                    orient=orient, pix_per_cell=pix_per_cell, 
                                    cell_per_block=cell_per_block, 
                                    hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                    hist_feat=hist_feat, hog_feat=hog_feat)
                draw_image = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)  
                heatmap = add_heat(heatmap, hot_windows)
       
            heatmap = apply_threshold(heatmap,2)   
            labels = label(heatmap)
            time2 = time.time()
            print(labels[1], "car found in", round(time2-time1,2), "seconds")
            
            final_img = draw_labeled_bboxes(np.copy(image), labels)
            
            fig = plt.figure(figsize=(16,9))
            fig.add_subplot(221), plt.imshow(draw_image)                    
            fig.add_subplot(222), plt.imshow(heatmap, cmap='hot')
            fig.add_subplot(223), plt.imshow(labels[0], cmap='gray')
            fig.add_subplot(224), plt.imshow(final_img)
    else:
        # initialize an instance
        car1 = Vehicle()
        car2 = Vehicle()
        # Test Video        
        project_output = 'project_output.mp4'
        clip = VideoFileClip("project_video.mp4")
        project_clip = clip.fl_image(process) #NOTE: this function expects color images!!
        project_clip.write_videofile(project_output, audio=False)
        print("Complete video output")
