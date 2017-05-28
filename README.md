# P4 - Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Overview

In this project, we will explore and develop a pipeline to detect lines on the road and track it.  The goals / steps of this project are the following:

* Correct camera distortion and retrieve distortion coefficients using the calibration chessboard images.
* Explore color transformations and gradients to make robust thresholded binary image.
* Perform perspective transform to get "birds-eye view" of the road.
* Identify line pixels in processed image and fit polynomial line to predict line.
* Calculate lane curvature and distance from center.
* Implement entire pipeline on image.
* Implement entire pipeline on video.

[//]: # (Image References)
[image1]: ./output_images/undistorted_chessboard.PNG
[image2]: ./output_images/undistorted_road.PNG
[image3]: ./output_images/experiment_masking.PNG
[image4]: ./output_images/final_masked.PNG
[image5]: ./output_images/perspective_transform.PNG
[image6]: ./output_images/find_lane.PNG
[image7]: ./output_images/curvature_equation.PNG
[image8]: ./output_images/pipeline_image.PNG
[image9]: ./output_images/projectvideo.jpg
[image10]: ./output_images/challengevideo.jpg

---

## Camera Calibration

In order to correctly identify lanes in video, the camera distortion must be corrected.  20 pictures of chessboard was used to perform camera calibration and extract distortion coefficients built in OpenCV function.  The chessboard used in the project had 9 X 6 corner points.  Using cv2.findChessboardCorners() function, grayscale image of the chessboard and the corner points dimension was inputed to extract corner points in 2D image world and object points in 3D space.  After extracting all image points and object points, distortion coefficient could be calculated using cv2.CalibrateCamera() function.  The code for this step can be found in calibration.ipynb.  Below are few examples of using the distortion coefficient to fix distortion on chessboard images.

![alt text][image1]

Using the distortion coefficients, the test images for our lane videos were fixed of distortion.

![alt text][image2]

```
img = mpimg.imread('test_images/'+image.strip())
    img_size = (img.shape[1], img.shape[0])
    
    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
```

## Pipeline

### Thresholded binary image

To successfully detect lanes in varying conditions on the road, combination of color and gradient thresholded binary image needed to be made.  The changing lighting condition, lane colors, and road color made this extremely challenging.  Various color spaces including 'RGB', 'HSV', 'HLS', 'YUV', 'LUV', and 'YCrCb' was used to determine which color space and threshold extracts white and yellow line accurately.  Below is a code snippet of masking function.  The code can be found in pipeline.ipynb.

![alt text][image3]

```
def masking(img):

    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(20, 100))
    dir_binary = dir_threshold(img, sobel_kernel=9, thresh=(0.2, 1.5))
    hls_s = color_space(img, color='HLS', thresh=(100,255), selection=2)
    hls_l = color_space(img, color='HLS', thresh=(120,255), selection=1)
    rgb_r = color_space(img, 'RGB', thresh=(150,255), selection = 0)
    rgb_g = color_space(img, 'RGB', thresh=(150,255), selection = 1)
    luv_l = color_space(img, 'LUV', thresh=(225,255), selection = 0)
    lab_b = color_space(img, 'LAB', thresh=(155,200), selection = 2)
    combined_1 = np.zeros_like(dir_binary)
    combined_1[(((rgb_r==1) & (rgb_g==1)) & (hls_l==1)) & (((hls_s==1) | ((gradx==1) & (dir_binary==1))))]=1
    combined_2 = np.zeros_like(rgb_r)
    combined_2[((rgb == 1) | (hls == 1)) | (gradx == 1) & ((dir_binary == 1))] = 1
    combined_2[(rgb_r == 1) | (hls_s == 1)] = 1
    combined_3 = np.zeros_like(dir_binary)
    combined_3[(luv_l == 1) | (lab_b == 1)] = 1
  
    return combined_1
```

After experimenting with all the parameters, following thresholding parameters were used.

| Parameter  | Channel   | Threshold  |
|:--------------:|:-----------------:|:----------:|
| RGB  | R | (150,255) |
| RGB   | G     |   (150,255) |
| HLS | L     |  (120,255) |
| HLS | S | (100,255) |
| Gradient | X | (20,100) |
| Dir. of Grad. | - | (0.2,1.5) |

Here is an Example image with finalized binary threshold.

![alt text][image4]

### Perspective Transform

For detecting lane lines and calculating curvature, it is useful to transform the perspective to bird's-eye view, which is looking down on to the road.  To perform the perspective transformation, handy OpenCV built in function was used.  Here is a code snippet from pipeline.ipynb.

```
def warp(img):
    corners = np.float32([[190,720],[589,457],[698,457],[1145,720]])
    top_left=np.array([corners[0,0],0])
    top_right=np.array([corners[3,0],0])
    offset=[150,0]
    
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([corners[0],corners[1],corners[2],corners[3]])
    dst = np.float32([corners[0]+offset,top_left+offset,top_right-offset ,corners[3]-offset])    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(img, M, img_size , flags=cv2.INTER_LINEAR)    
    return warped, M, src, dst, Minv
```

The corners were found by experimentation.  The cv2.getPerspectiveTransform function uses 'src' and 'dst' points as arguments which was derived from predetermined corner points.  Below is image of perspective transformation with corners ploted on the image.

![alt text][image5]

### Finding Lane Lines

Now that we have appropriate perspective and thresholded binary image, we can move on to find lane lines.  To find lane lines, first we take histogram of the processed image to determine which area of the image has the highest value to find the lane.  We split the histogram in half for left and right lanes.  After the predicted lane points were located, sliding window search technique was used to find pixel positions of the detected lane lines.  

![alt text][image6]

As seen in the figure above, sliding search window starts the search at the peak points of histogram for left and right side of the image.  Green boxes in the 'Found Lanes' image shows the sliding windows that readjusted to mean center as it moved upward.  Below code snippet shows how the best fit line was calculated using np.polyfit() function.

```
 # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
```

### Calculate Lane Curvature

For calculating lane curvature, following equation was used.

![alt text][image7]

Since the x and y values used in the equation is pixels, we have to convert the pixel values to meters.  

```
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Fit converted x,y values to polynomial
    fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
    
    # Calculate the curvature
    curverad = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
```

The approximate location of the vehicle from the center of the lane can be predicted by fitting y value of the lowest part of the image in the polynomial found.  

```
    left = left[0]*720**2 + left[1]*720 + left[2]
    right = right[0]*720**2 + right[1]*720 + right[2]
   
    center = 640 - ((right + left) / 2)
    
    xm_per_pix = 3.7/700 
    
    ret = center * xm_per_pix
```

Now that we found the lane lines, curvature, and the position of the vehicle, let's try it out on test images.

![alt text][image8]

## Pipeline(video)

To keep track of detected lane lines unlike applying the pipeline to image, Line class was computed.  The code for Line() class can be found in pipeline.ipynb at "Decine class" section.  Each side of the line was instantiated into Line() class.  Here is a part from the Line() class.

```
class Line():
    
    def __init__(self, side):
        
        # was the line detected in the last iteration?
        self.detected = False  
        
        # was their enough points found through line search
        self.trigger = False
        
        # Set which side the Line object is
        self.side = side
        
        # Count how many values in history list
        self.history = 0
        
        # fit from previous iteration
        self.fit = None
        
        # fitted x values from previous iteration
        self.fitx = None
        
        # recent list of fit values
        self.fit_list = deque(maxlen = 10)
        
        # recent list of x values
        self.fitx_list = deque(maxlen = 10)
        
        # average of last 10 fits
        self.avg_fit = None#np.mean(self.fit_list, axis=0)
        
        # average of last 10 x values
        self.avg_fitx = None#np.mean(self.fitx_list, axis=0)
```

By keeping track of previous state of the frame and average of last 10 frames, the lane drawings were stabilized even if the lane was not detected for certain duration.  Sanity check was applied to differentiate between good detected lane and bad lane which was disregarded.

![Project Video Output][image9]
(https://www.youtube.com/watch?v=47-XMuuABts)

![Challenge Video Output][image10]
(https://www.youtube.com/watch?v=5f5We12tdY4)

## Discussion

Overall, the pipeline performed at satisfactory level on both project video and challenge video.  More robust pipeline would be needed to successfully complete harder challenge video.  One of the many things that can be improved would be an adaptive thresholded binary image depending on the lighting situation as there were extensive glares that made the pipeline fail in the harder challenge video.  Also for the extreme curves where only one lane can be visible in camera for a considerable amount of time, integrating interpolation to the other side of the line would help improve the pipeline to be more robust.