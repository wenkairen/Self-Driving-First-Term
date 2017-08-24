
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[image1]: ./camera_cal/undst_calibration1.png "calibraiton image test "
[image2]: ./output_images/undist_straight_lines1.jpg "Straight line1 unidistort"

[image3]: ./output_images/unwarp_straight_lines1.jpg "Wapper Example"
[image4]: ./output_images/unwarp_straight_lines2.jpg "Warpper Example"
[image5]: ./output_images/unwarp_test1.jpg "Wapper Example"
[image6]: ./output_images/unwarp_test2.jpg "Warpper Example"
[image7]: ./output_images/unwarp_test3.jpg "Wapper Example"
[image8]: ./output_images/unwarp_test4.jpg "Warpper Example"
[image9]: ./output_images/unwarp_test5.jpg "Wapper Example"
[image10]: ./output_images/unwarp_test6.jpg "Warpper Example"

[image11]: ./output_images/combined_binary_straight_lines1.jpg "Binary Example"
[image12]: ./output_images/combined_binary_straight_lines2.jpg "Binary Example"
[image13]: ./output_images/combined_binary_test1.jpg "Binary Example"
[image14]: ./output_images/combined_binary_test2.jpg "Binary Example"
[image15]: ./output_images/combined_binary_test3.jpg "Binary Example"
[image16]: ./output_images/combined_binary_test4.jpg "Binary Example""
[image17]: ./output_images/combined_binary_test5.jpg "Binary Example"
[image18]: ./output_images/combined_binary_test6.jpg "Binary Example"
[image19]: ./output_images/find_line_test1.png "find line Example"

[image20]: ./output_images/draw_back_oringial.png "draw line back Example"

[video1]: ./project_video_output.mp4 "Video"
[video2]: ./challenge_video_output.mp4 "Video"
[video3]: ./harder_challenge_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the example folder, the example.ipynb is the file for the calibration of all the images in the camera_cal folder.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  
I applied this distortion correction to the first calibriation1 image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The final code is shown as Advanced_Lane_Detecting.ipynb, After I reload the undistort coeffcient, I demonstrate this step, to apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `unwarped`, which is the Pespective Transform cell,  The `unwarped` function takes as inputs an image (`img`), as well as the undistort orignial image, the source (`src`) and destination (`dst`) points are written indside of the function. I chose the dynamic code  he source and destination points in the following manner:

```python
    w,h = 1280,720
    x,y = 0.5*w, 0.8*h
    src = np.float32([[200./1280*w,720./720*h],
                  [453./1280*w,547./720*h],
                  [835./1280*w,547./720*h],
                  [1100./1280*w,720./720*h]])
    dst = np.float32([[(w-x)/2.,h],
                  [(w-x)/2.,0.82*h],
                  [(w+x)/2.,0.82*h],
                  [(w+x)/2.,h]])
```
I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. Here lists all the images of all the test image:

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of HLS color and Lab clolor channel thresholds to generate a binary image, becasue during the testing process, the L Channel is more robost in different situation, also for the Lab Channel, the B channel can pick up the yellow line very well.  Here lists all the resutls of the thresh and unwarpped image:
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]
![alt text][image17]
![alt text][image18]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In order to find the best fiting lanes, I used the `Window_line()` function based on the course  instruction to first
plots the histogram to identify the left and right start points with highest signal, then I use the window method for each lane detect in warped image, in each window, the `nonezero()` method is used to pick the good points of the left and right lanes, after I have appened good ints for both left and right lanes, I use the `polyfit()` method get the yellow lane for each, In order to make the lanes more robust, the `New_fit` function is used to clean even more based on the previouse fits, to concentrate all the points close to the fit lanes , then I recollect good ints for left and right and polygit again, Also I use the `cv2.fillpoly` function to fill the lane margin on the image.
The result is shown below:
![alt text][image19]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines Curvature and Distance section, based on the previous fit for each lane and ints of the points on each lane, I use the curvature funtion described in the class, with the conversion from camer to actual points 
by using :

    left_c = l_fit[0] * binary_img.shape[0] ** 2 + l_fit[1] * binary_img.shape[0] + l_fit[2]
    right_c = r_fit[0] * binary_img.shape[0] ** 2 + r_fit[1] * binary_img.shape[0] + r_fit[2]
    width = right_c - left_c
    xm_per_pix = 3.7 / width  # meters per pixel in y dimension
    ym_per_pix = 3*7/720 # meters per pixel in y dimension.
    
Also I calaulte the distance of the car from the center of the lane.
    
the result is shown below:

Radius of curvature =  238.427605991 (m), 331.572349644 (m)

Distance from lane center =  -0.214762597379 (m)


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in Draw back to Orinal image.  Here is an example of my result on a test image:

![alt text][image20]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's the project_video_output [link to my video result](./project_video_output.mp4)

Here's the challenge_video_output [link to my video result](./challenge_video_output.mp4)

Here's a the harder_challenge_cideo_output[link to my video result](./harder_challenge_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

In this project, I have spents lots of time on making the pipeline more robust so that it can be applied in different situation, I found it very helpful to check the questions post on the form, So I can have a dynamic pespective tranform, becasue this would largely affect how the threshhold combination identify the lane, also the combined bianary image is also very triky, I have experiment different combines of color, and I found Lab is one of very robust method to pick out the yellow color. For the window and draw line method, I follow the udacity course instrcution fit the poly line as expected, and the curvative calculation method provided, what's more the line Class is also very helpful to keep track of the fits all the time, like when to choose the best fit, how to abort the bad fits , this are also worth to explore, 

When I experience this project, I find a hard time of identify the lanes in the shawdow, sometimes it just lost the track or other tiny stuff affects the performance, that's what I have mentionde before about find better theresh tech method, this would make the process much easier. Also my test is able to passs the first two videos, but not the harder challenge one there are many reasons that I can think of to make it better improvement in the future, the fits update should be more accuracte to adapt to the constatly change of large curves, in the two lane road, when it comes to fast changes in the curve, the car lost the track, this is also an very important problem avoid, also the constantly switch from the sunlight and show in a curved road would cause much more chanllanging to the identify the lane, especially the tress are around the road, the sun spot is a huge influce.

Above are the isssues and reasons that I can think of for my project, I will continue improve my project in the future to more robust, and try to use different method to achieve it.




```python

```
