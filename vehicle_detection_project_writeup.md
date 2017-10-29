### Udacity Self Driving Car Nanodegree
### Vehicle Detection Project Writeup
### Brian Fischer


[//]: # (Image References)
[bounding_box]: ./output_images/bounding_box.jpg
[find_cars_start]: ./output_images/find_cars_start.jpg
[heat_map]: ./output_images/heat_map.jpg
[sliding_window_full_image]: ./output_images/sliding_window_full_image.jpg
[sliding_window_roi]: ./output_images/sliding_window_roi.jpg
[testing_scale_and_region]: ./output_images/testing_scale_and_region.jpg

[project_video_final]: ./project_video_final.mp4


**Vehicle Detection Project Overview**

The goals / steps of this project are the following:

1.  **Data Exploration**  
    Review the available data set contents and quality

2.  **Feature Extraction**  
    Perform a Histogram of Oriented Gradients (HOG) feature extraction on a
    labeled training set of images

3.  **Train a Classifier**  
    From SVM, use the Linear Support Vector Classification (SVC)

4.  **Sliding Window Algorithm**  
    Implement a sliding-window technique and use your trained classifier to
    search for vehicles in images.

5.  **Test Sliding Window Algorithm**  
    Test on static images and tweak the algorithm parameters 

6.  **Heat Map Algorithm**  
    Create a heat map of recurring detections in the pipeline, reject outliers
    and follow detected vehicles. Threshold and label the heatmap. 

7.  **Bounding Box Algorithm**  
    Estimate a bounding box for vehicles detected. Needs to account for overlap
    and multiple vehicles 

8.  **Implement a Video Pipeline**  
    Allow the algorithm to run frame by frame 

9.  **Run Pipeline on Project Video**  
    Run your pipeline on the project videos and Optional: Output a visual
    display of the lane boundaries and numerical estimation of lane curvature
    and vehicle position!

------
# Rubric Points
------

Here I will consider the rubric points individually and describe how I addressed
each point in my implementation.

-------------------
## 1) Writeup / README

**Provide a Writeup / README that includes all the rubric points and how you
addressed each one.**

You're reading it! Details of the project can be found in the Jupyter notebook
"vehicle_detection_project.ipynb"

----------------------------------------
## 2) Histogram of Oriented Gradients (HOG)


### 2a) HOG Features and Parameters

**Explain how (and identify where in your code) you extracted HOG features from
the training images. Explain how you settled on your final choice of HOG
parameters.**

The following features are extracted from the labeled training set images: \*
Histogram of Oriented Gradients (HOG) \* spatial binning (binned color) \* color
histograms

With a lot of testing, it was clear that all 3 methods for feature extraction
were necessary to get a test accuracy of over 95%.

### 2b) Classifier Training

**Describe how (and identify where in your code) you trained a classifier using
your selected HOG features (and color features if you used them).**

Prior to classifier selection and training, some data exploration was necessary
(section 1). Review of the provided data sets indicated a good balance between
vehicles and non-vehicles. This data set would work well for binary
classification and the Linear SVC was a good match.

A few of the classifier parameters are listed below:

```python
    color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = 0 # Can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32) # Spatial binning dimensions
    hist_bins = 32    # Number of histogram bins
    hist_range = (0, 256)
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
    y_start_stop = [None, None] # Min and max in y to search in slide_window()
```

Initial data sets provided above 95% accuracy which seemed OK, but further
tweaking of these parameters in the *4. Sliding Window Algorithm* section
revealed the YCrCb color space with a bit more HOG orientations and larger color
histogram bins could achieve 99%+ accuracy.

------------------------
## 3) Sliding Window Search

### 3a) Describe Implementation and Parameters

**Describe how (and identify where in your code) you implemented a sliding
window search. How did you decide what scales to search and how much to overlap
windows?**

Working through the Udacity exercises provided a good lesson on how to
efficiently implement a sliding window technique. The original sliding window
method executed the HOG many times for each window, this was later improved via
the use of a single HOG with sub-sampling. This improved algorithm speeds by
10x.

In addition, it was very important to define a region of interest along with
overlapping windows to make sure all vehicles in the field of view would be
detected. These scaling and overlapping parameters introduced some false
positives that would later be filtered out using heat maps and information from
successive frames.


### 3b) Example Images and Pipeline Description

**Show some examples of test images to demonstrate how your pipeline is working.
How did you optimize the performance of your classifier?**

Below are some example images from the pipeline.  The classifier performance was optimized through a lot of testing on the test_images to understand the influence of each parameter.  Additional test images were added for some scenarios that had difficulty when the video was processed.

Raw classifier:
![alt text][sliding_window_full_image]

Region of Interest for Vehicles:
![alt text][sliding_window_roi]

**find_cars** algorithm:
![alt text][find_cars_start]

Testing scale and region parameters:
![alt text][testing_scale_and_region]

Heat maps implemented:
![alt text][heat_map]

Bounding box on above heat map:
![alt text][bounding_box]

-----------------------
## 4) Video Implementation


### 4a) Final Video Output

**Provide a link to your final video output. Your pipeline should perform
reasonably well on the entire project video (somewhat wobbly or unstable
bounding boxes are ok as long as you are identifying the vehicles most of the
time with minimal false positives.)**

The final project video is available here:
![alt text][project_video_final]

### 4b) Filtering and Bounding Boxes

**Describe how (and identify where in your code) you implemented some kind of
filter for false positives and some method for combining overlapping bounding
boxes.**

I added a helper class *BoxTracker* that stores the last "n" boxes found.  This history information is used to filter out false positives and combine nearby boxes (a buffer of 10 pixels to the left was added to the bounding box function *draw_labeled_bboxes* to account for any lag of this averaging).

-------------
## 5) Discussion

**Briefly discuss any problems / issues you faced in your implementation of this
project. Where will your pipeline likely fail? What could you do to make it more
robust?**

The pipeline was displaying boxes for oncoming traffic so I cropped out the region of the image where oncoming traffic would appear.
The pipeline was also not perfect in detecting vehicles in 100% of the frames.  I would most likely move from a SVM with HOG feature extraction to a deep learning neural network since this is a binary classification problem.
