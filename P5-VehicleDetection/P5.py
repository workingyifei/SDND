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


def process(image):
    draw_image = np.copy(image)
            
    # top
    windows_top = slide_window(image, x_start_stop=[650, None], y_start_stop=[350, 500], 
                        xy_window=(64, 64), xy_overlap=(0.7, 0.7))
    hot_windows_top = search_windows(image, windows_top, svc, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)                       
    # mid
    windows_mid = slide_window(image, x_start_stop=[650,None], y_start_stop=[400, 600], 
                        xy_window=(96, 96), xy_overlap=(0.8, 0.7))
    hot_windows_mid = search_windows(image, windows_mid, svc, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat) 
    
    # mid1
    windows_mid1 = slide_window(image, x_start_stop=[650,None], y_start_stop=[410, 600], 
                        xy_window=(128, 128), xy_overlap=(0.8, 0.7))
    hot_windows_mid1 = search_windows(image, windows_mid1, svc, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat) 
    
    # bottom
    windows_bot = slide_window(image, x_start_stop=[650,None], y_start_stop=[500, None], 
                        xy_window=(160, 160), xy_overlap=(0.5, 0.5))
    hot_windows_bot = search_windows(image, windows_bot, svc, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat) 
    
    

    window_img = draw_boxes(draw_image, hot_windows_top, color=(0, 0, 255), thick=6)   
    window_img = draw_boxes(window_img, hot_windows_mid, color=(0, 0, 255), thick=6) 
    window_img = draw_boxes(window_img, hot_windows_mid1, color=(0, 0, 255), thick=6) 
    window_img = draw_boxes(window_img, hot_windows_bot, color=(0, 0, 255), thick=6) 
    
    heatmap = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    heatmap = add_heat(heatmap, hot_windows_mid)
    heatmap = add_heat(heatmap, hot_windows_top)
    heatmap = add_heat(heatmap, hot_windows_bot)
    heatmap = apply_threshold(heatmap,2)   
    labels = label(heatmap)
    
    final_img = draw_labeled_bboxes(np.copy(image), labels)                    
    return final_img


if __name__ == "__main__":  
    
    # load model
    filename = "./model_yuv_982all.pkl"
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

    
    test_mode = 'movie'
    
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
            
            # top
            windows_top = slide_window(image, x_start_stop=[650, None], y_start_stop=[350, 500], 
                                xy_window=(64, 64), xy_overlap=(0.7, 0.7))
            hot_windows_top = search_windows(image, windows_top, svc, color_space=color_space, 
                                    spatial_size=spatial_size, hist_bins=hist_bins, 
                                    orient=orient, pix_per_cell=pix_per_cell, 
                                    cell_per_block=cell_per_block, 
                                    hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                    hist_feat=hist_feat, hog_feat=hog_feat)                       
            # mid
            windows_mid = slide_window(image, x_start_stop=[650,None], y_start_stop=[400, 600], 
                                xy_window=(96, 96), xy_overlap=(0.8, 0.7))
            hot_windows_mid = search_windows(image, windows_mid, svc, color_space=color_space, 
                                    spatial_size=spatial_size, hist_bins=hist_bins, 
                                    orient=orient, pix_per_cell=pix_per_cell, 
                                    cell_per_block=cell_per_block, 
                                    hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                    hist_feat=hist_feat, hog_feat=hog_feat) 
            
            # mid1
            windows_mid1 = slide_window(image, x_start_stop=[650,None], y_start_stop=[410, 600], 
                                xy_window=(128, 128), xy_overlap=(0.8, 0.7))
            hot_windows_mid1 = search_windows(image, windows_mid1, svc, color_space=color_space, 
                                    spatial_size=spatial_size, hist_bins=hist_bins, 
                                    orient=orient, pix_per_cell=pix_per_cell, 
                                    cell_per_block=cell_per_block, 
                                    hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                    hist_feat=hist_feat, hog_feat=hog_feat) 
            
            # bottom
            windows_bot = slide_window(image, x_start_stop=[650,None], y_start_stop=[500, None], 
                                xy_window=(160, 160), xy_overlap=(0.5, 0.5))
            hot_windows_bot = search_windows(image, windows_bot, svc, color_space=color_space, 
                                    spatial_size=spatial_size, hist_bins=hist_bins, 
                                    orient=orient, pix_per_cell=pix_per_cell, 
                                    cell_per_block=cell_per_block, 
                                    hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                    hist_feat=hist_feat, hog_feat=hog_feat) 
            

            window_img = draw_boxes(draw_image, hot_windows_top, color=(0, 0, 255), thick=6)   
            window_img = draw_boxes(window_img, hot_windows_mid, color=(0, 0, 255), thick=6) 
            window_img = draw_boxes(window_img, hot_windows_mid1, color=(0, 0, 255), thick=6) 
            window_img = draw_boxes(window_img, hot_windows_bot, color=(0, 0, 255), thick=6) 
            
            heatmap = np.zeros((image.shape[0], image.shape[1]), np.uint8)
            heatmap = add_heat(heatmap, hot_windows_mid)
            heatmap = add_heat(heatmap, hot_windows_top)
            heatmap = add_heat(heatmap, hot_windows_bot)
            heatmap = apply_threshold(heatmap,2)   
            labels = label(heatmap)
            time2 = time.time()
            print(labels[1], "car found in", round(time2-time1,4), "seconds")
            
            final_img = draw_labeled_bboxes(np.copy(image), labels)
            
            fig = plt.figure(figsize=(16,9))
            fig.add_subplot(221), plt.imshow(window_img)                    
            fig.add_subplot(222), plt.imshow(heatmap, cmap='hot')
            fig.add_subplot(223), plt.imshow(labels[0], cmap='gray')
            fig.add_subplot(224), plt.imshow(final_img)
    else:
        # Test Video        
        project_output = 'test_video_yuv982all.mp4'
        clip = VideoFileClip("project_video.mp4").subclip(25, 42)
        project_clip = clip.fl_image(process) #NOTE: this function expects color images!!
        project_clip.write_videofile(project_output, audio=False)
        print("Complete video output")
