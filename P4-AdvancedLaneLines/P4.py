import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip




# return calibratino matrix and distortion matrix
def calibrate_camera(dir, nx=9, ny=6):
    global mtx, dist
    assert os.path.exists(dir)
    
    # Read in and make a list of calibration images
    # It's recommend to read at least 20 images to get a good calibration
    images = glob.glob('./camera_cal/calibration*.jpg')
    
    # arrays to store object points and image points from all the images
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane

    # prepare object points
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2) # x, y coordinates

    for idx, fname in enumerate(images):
        # read in an image
        img = cv2.imread(fname)

        # convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        print("Image {} corner found?: {}".format(idx, ret))

        # If found, add object points, image points
        if ret==True:
            imgpoints.append(corners)
            objpoints.append(objp)

            # draw and display the corners
            img = cv2.drawChessboardCorners(gray, (8,6), corners, ret)

    # calibreate Camera
    ret, mtx, dist, rvec, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Camera calibrated.")
    return mtx, dist

# Undistort an image
def undistort(img, mtx=None, dist=None):
    img_undistort = cv2.undistort(img, mtx, dist, None, mtx)
    return img_undistort

def perspective_transform(image, src_in = None, dst_in = None, display=False):
    img_size = image.shape
    if src_in is None:
        src = np.array([[585. /1280.*img_size[1], 455./720.*img_size[0]],
                        [710. /1280.*img_size[1], 455./720.*img_size[0]],
                        [1130./1280.*img_size[1], 710./720.*img_size[0]],
                        [190. /1280.*img_size[1], 710./720.*img_size[0]]], np.float32)
    else:
        src = src_in

    if dst_in is None:
        dst = np.array([[300. /1280.*img_size[1], 100./720.*img_size[0]],
                        [1000./1280.*img_size[1], 100./720.*img_size[0]],
                        [1000./1280.*img_size[1], 720./720.*img_size[0]],
                        [300. /1280.*img_size[1], 720./720.*img_size[0]]], np.float32)
    else:
        dst = dst_in

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    img_warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    if display:
        plt.figure(figsize=(16,9))
        plt.subplot(1,2,1)
        plt.hold(True)
        plt.imshow(image, cmap='gray')
        colors = ['r+','g+','b+','w+']
        for i in range(4):
            plt.plot(src[i,0],src[i,1],colors[i])

        plt.subplot(1,2,2)
        plt.hold(True)
        plt.imshow(img_warped, cmap='gray')
        for i in range(4):
            plt.plot(dst[i,0],dst[i,1],colors[i])
        plt.show()
        
    return img_warped, M, Minv

def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    if orient == 'x':
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    else:
        abs_sobely = np.absolute(sobely)
        scaled_sobel = np.uint8(255*abs_sobely/np.max(abs_sobely))
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1    
    grad_binary = sxbinary
    return grad_binary

def mag_thresh(image, sobel_kernel=9, mag_thresh=(30, 100)):
    # Calculate gradient magnitude
    # Apply threshold
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # Calculate the magnitude 
    abs_sobelxy = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    mag_binary = binary_output
    return mag_binary

def dir_threshold(image, sobel_kernel=15, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # calculate the direction of the gradient 
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    dir_binary = binary_output
    return dir_binary

def color_threshold(image, channel='s'):
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]    
    s_channel = hls[:,:,2]
    h_thresh_min = 170
    h_thresh_max = 255
    l_thresh_min = 170
    l_thresh_max = 255
    s_thresh_min = 200
    s_thresh_max = 255
    
    if channel == 'h':
        # Threshold color channel
        h_binary = np.zeros_like(h_channel)
        h_binary[(h_channel >= h_thresh_min) & (h_channel <= h_thresh_max)] = 1
        color_binary = h_binary
    elif channel == 'l':
        # Threshold color channel
        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1
        color_binary = l_binary
    else:
         # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
        color_binary = s_binary
    
    return color_binary

def threshold(image):
   # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(50, 255))
    # grady = abs_sobel_thresh(image, orient='y', sobel_kernel=3, thresh=(200, 255))
    mag_binary = mag_thresh(image, sobel_kernel=3, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1,2))
    s_binary = color_threshold(image, channel='s')
    
    combined = np.zeros_like(dir_binary)
    combined[(gradx == 1) | ((mag_binary==1) & (gradx==1))] = 1
    
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(gradx)
    combined_binary[(s_binary == 1) | (combined == 1)] = 1

    return combined_binary

def sliding_window(image, display=False):
    histogram = np.sum(image[image.shape[0]/2:,:], axis=0)
    out_img = np.dstack((image, image, image))*255
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 10
    window_height = image.shape[0]/nwindows
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = int(image.shape[0] - window_height*(window+1))
        win_y_high = int(image.shape[0] - window_height*window)
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255,0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        good_left_inds = ((nonzeroy>=win_y_low) & (nonzeroy< win_y_high) & (nonzerox>= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Visualize the result
    if display == True:
        fity = np.linspace(0, image.shape[0]-1, image.shape[0] )
        fit_leftx = left_fit[0]*fity**2 + left_fit[1]*fity + left_fit[2]
        fit_rightx = right_fit[0]*fity**2 + right_fit[1]*fity + right_fit[2]
    
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(fit_leftx, fity, color='yellow')
        plt.plot(fit_rightx, fity, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

    return left_fit, right_fit, leftx, lefty, rightx, righty

def skip_sliding_windows(image, left_fit, right_fit, display=False):
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    fity = np.linspace(0, image.shape[0]-1, image.shape[0] )
    fit_leftx = left_fit[0]*fity**2 + left_fit[1]*fity + left_fit[2]
    fit_rightx = right_fit[0]*fity**2 + right_fit[1]*fity + right_fit[2]
    
    if display == True:
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((image, image, image))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([fit_leftx-margin, fity]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([fit_leftx+margin, fity])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([fit_rightx-margin, fity]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([fit_rightx+margin, fity])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(fit_leftx, fity, color='yellow')
        plt.plot(fit_rightx, fity, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
    return left_fit, right_fit, leftx, lefty, rightx, righty


def draw_lane(image_undistort, image_warped, left_fit, right_fit, Minv):
    image = image_warped
    # Generate x and y values for plotting
    fity = np.linspace(0, image.shape[0]-1, image.shape[0] )
    fit_leftx = left_fit[0]*fity**2 + left_fit[1]*fity + left_fit[2]
    fit_rightx = right_fit[0]*fity**2 + right_fit[1]*fity + right_fit[2]
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([fit_leftx, fity]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([fit_rightx, fity])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
#     fig = plt.figure(figsize=(16,9))
#     fig.add_subplot(121), plt.imshow(color_warp), plt.title("original")
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(image_undistort, 1, newwarp, 0.3, 0)
    
    return result



# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, detected):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
     
    def check(self):
        
        return self.detected
            
    
    def calculate_radius_of_curvature(self):
        # Generate some fake data to represent lane-line pixels
        fity = np.linspace(0, 719, num=720) # to cover same y-range as image

        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image     
        y_eval = np.max(fity)
#        fit = self.current_fit
#        curverad = ((1 + (2*fit[0]*y_eval + fit[1])**2)**1.5) / np.absolute(2*fit[0])
#        right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
#        print(left_curverad, right_curverad)
        
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(self.ally*ym_per_pix, self.allx*xm_per_pix, 2)
#        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        self.radius_of_curvature = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
#        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
#        print(left_curverad, 'm', right_curverad, 'm')

        return self.radius_of_curvature



def process_image(image, left_lane=left_lane, right_lane=right_lane, display=False): 
    # undistort image
    image_undistort = undistort(image, mtx=mtx, dist=dist)
    
    # change image to threshoded binary image
    image_threshold = threshold(image_undistort)
    
    # transfer image to a bird eye view
    image_warped, M, Minv = perspective_transform(image_threshold)
    
    if left_lane.detected==True and right_lane.detected==True:
        left_fit, right_fit, leftx, lefty, rightx, righty = skip_sliding_windows(image_warped, left_lane.current_fit, right_lane.current_fit, display=False)
        left_lane.allx = leftx
        left_lane.ally = lefty
        left_lane.calculate_radius_of_curvature()
        left_lane.current_fit = left_fit
    
        
        right_lane.allx = rightx
        right_lane.ally = righty
        right_lane.calculate_radius_of_curvature()
        right_lane.current_fit = right_fit
        
    else:
        # current lane lines
        left_fit, right_fit, leftx, lefty, rightx, righty = sliding_window(image_warped)
        left_lane.allx = leftx
        left_lane.ally = lefty
        left_lane.calculate_radius_of_curvature()
        left_lane.current_fit = left_fit
    
        
        right_lane.allx = rightx
        right_lane.ally = righty
        right_lane.calculate_radius_of_curvature()
        right_lane.current_fit = right_fit

    # check if detected lines are real-thing
    curv_diff = abs(left_lane.radius_of_curvature - right_lane.radius_of_curvature)
    
    if (curv_diff >40) and (curv_diff<60): # normal curv_diff is around 50m
        if left_lane.allx.shape[0]>100:    
            left_lane.detected=True
        else:
            left_lane.detected=False
            
        if right_lane.allx.shape[0]>100:
            right_lane.detected=True
        else:
            right_lane.detected=False
    else:
            left_lane.detected=False
            right_lane.detected=False
    

#    print(abs(left_lane.radius_of_curvature - right_lane.radius_of_curvature))
#    print(abs(left_lane.current_fit[0] - right_lane.current_fit[0]))
#    print(left_lane.current_fit)
#    print(right_lane.current_fit)
#    print(left_fit.shape, leftx.shape, lefty.shape, rightx.shape, righty.shape)
    
    # draw the lines back down onto the road
    result = draw_lane(image_undistort,image_warped, left_fit, right_fit, Minv)

    
    left_curvature = left_lane.radius_of_curvature
    right_curvature = right_lane.radius_of_curvature
    # put text on video
    cv2.putText(result, 'Left radius of curvature = %.2f m'%(left_curvature),(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(result, 'Right radius of curvature = %.2f m'%(right_curvature), (50,80), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
#    cv2.putText(result, 'Vehicle position: %.2f m)
    
    if display==True:
        fig = plt.figure(figsize=(16,9))
        fig.add_subplot(231), plt.imshow(image), plt.title("original")
        fig.add_subplot(232), plt.imshow(image_undistort), plt.title("undistorted")
        fig.add_subplot(233), plt.imshow(image_threshold, cmap='gray'), plt.title("thresholded binary")
        fig.add_subplot(234), plt.imshow(image_warped, cmap='gray'), plt.title("warped binary")
        fig.add_subplot(235), plt.imshow(result), plt.title("final")
    
    return result
    

if __name__ == "__main__":  
    # Calibrate the camera
    print("Calibrating camera")
    mtx, dist = calibrate_camera('camera_cal/')
    
    # define left and right lane clas substances
    left_lane = Line(False)
    right_lane = Line(False)
            

    test_mode = 'video'
    
    if test_mode == 'image':
        # Test Images
        image = mpimg.imread('test_images/test3.jpg')
        print("image shape is:", image.shape)
        result = process_image(image, left_lane=left_lane, right_lane=right_lane, display=True)

        
    else:
        # Test Video        
        project_output = 'project_output.mp4'
        clip1 = VideoFileClip("project_video.mp4")
        project_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
        project_clip.write_videofile(project_output, audio=False)
        print("Complete video output")