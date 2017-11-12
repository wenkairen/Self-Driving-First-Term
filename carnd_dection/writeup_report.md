


**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_nonecar.png
[image2]: ./output_images/hog_feature.png
[image3]: ./output_images/test1_output.png
[image4]: ./output_images/test2_output.png
[image5]: ./output_images/test3_output.png
[image6]: ./output_images/test4_output.png
[image7]: ./output_images/test5_output.png
[image8]: ./output_images/test6_output.png
[video1]: ./project_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is in the Vehicle_detecting_tracking.ipython notebook, I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:
![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I use the first two images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the the R color space and HOG parameters of `orientations = 9 `, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters , for the color channels, I tried `RGB`, `YCrCb`, `YUV`. Based on the test result, and the `YCrCb` gives me good results of the testing image, this will shown in the later section, Also I have expenrienced different parameters of orentation, like 9, 10, 11, but it turns out with orentation 10 gives the best performance. So I use these parameters for feature extraction.

Here lists all the paramters that I choose:

`color_space = 'YCrCb'
orient = 10
pix_per_cell = 8 
cell_per_block = 2 
hog_channel = 'ALL' 
spatial_size = (32, 32) 
hist_bins = 64 
spatial_feat = True 
hist_feat = True 
hog_feat = True `

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Based On the Lecture notes, I use three different features to combine together, I extract the spatial size of the image feature by resize, use the histbins to get the color space feature, and use the hog feature, the function is shown in `extract_features`, with these combined features, I extract both from cars and nonecars images, and I trained with linear SVM classifier. 
The results shows :
`Using: 10 orientations, 8 pixels per cell and 2 cells per block
Feature vector length: 9144
45.01 Seconds to train SVC...
Test Accuracy of SVC = 98.82%`


### Hog Sub-sampling Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I use the `find_car`functions to define my secrching windows, this method is based on the course lecture notes of the sub-sampling window search, The `find_cars` only has to extract hog features once and then can be sub-sampled to get all of its overlaying windows. Each window is defined by a scaling factor where a scale of 1 would result in a window that's 8 x 8 cells then the overlap of each window is in terms of the cell distance. I run this same function 3 times for different scale values(1.0, 1.5, 2.0) to generate multiple-scaled search windows.

Below is the test1 image, the `bboxes` shows all the search windows in the image
![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. In order to make a better performance, I use the `
X_scaler` to normalized the feature vecture to improve and I also randomlized to shuffle my data into training and testing to get good results.

![alt text][image3]
![alt text][image6]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. These process is then combined together, which can limit muli- detect windows to be single labeled box for each identified car, Also, I write the `Object Class` to track and store the data,in this way, I can keep track all the rectangles found in the video and update according by removing the unnecessary windows. The results is shown in the next section with all the process for the 6 test images

## Combined all the images together for Better Visulization

### Here are six frames and their corresponding heatmaps , Labels , and resulting bounding boxes

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

In this project, firstly I have a better understanding of the feature vectures in the computer vision filed, becasue this time we use SVM model to based on the extract features which is different than the CNN that we have used before.
The SVM is based on lots of parameters to tune to get a better results, but comparing to CNN, the trainning time is faster, Also I have learned a lot of test different color channels based on different situations, and includin the hog feature tune, I have failed so many times with dirrerent combinations of these prameters, and in the end I find different ways to change the parameter more efficient, What's more, I have learned great argothems of slide window search based on setting different scales, which is great tech to use in other situations like pedestrain detection. Also smooth the detected windows is very important to the project, like how to tune and update the heatmap and applying different threshhold values to reduce the false detection. All these great methods are very helpful. The future work of the pipeline can be more rubost.


```python

```
