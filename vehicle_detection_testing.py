import numpy as np
import cv2
import glob
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog

from scipy.ndimage.measurements import label

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB',
                     orient=9, pix_per_cell=8, cell_per_block=2,
                     spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:

        # Read in each one by one
        image = mpimg.imread(file)

        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        else:
            feature_image = np.copy(image)

        # Get HOG features (can only do one color channel at a time)
        hog_features = []
        color_channels = feature_image.shape[2]
        for channel in color_channels:
            hog_features.append(get_hog_features(feature_image[:, :, channel], orient,
                                                 pix_per_cell, cell_per_block,
                                                 vis=False, feature_vec=True))
        # Flatten all channels into a single array
        hog_features = np.ravel(hog_features)

        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)

        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)

        # Append the new feature vector to the features list
        features.append(np.concatenate((hog_features, spatial_features, hist_features)))

    # Return list of feature vectors
    return features


# From Udacity 17.22 norm_shuffle.py

# Define HOG parameters
cspace = 'YUV'  # Options are RGB, HSV, LUV, HLS, YUV
orient = 9
pix_per_cell = 8
cell_per_block = 2
spatial_size = (32, 32)
hist_bins = 32
hist_range = (0, 256)

# Load Images
vehicles     = glob.glob('data/vehicles/**/*.png', recursive=True)
non_vehicles = glob.glob('data/non-vehicles/**/*.png', recursive=True)
print("Vehicle Images:\t\t", len(vehicles))
print('Non-vehicle Images:\t', len(non_vehicles))

t = time.time()
car_features = extract_features(vehicles, cspace, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
                                hist_range)

elapsed = time.time() - t
print('Elapsed: %s' % elapsed)
t = time.time()
notcar_features = extract_features(non_vehicles, cspace, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
                                hist_range)
elapsed = time.time() - t
print('Elapsed: %s' % elapsed)

if len(car_features) > 0:
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Since we are combining different features (HOG, binned color, color histogram) we need to normalize
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Plot an example of raw and scaled features from a random image
    car_ind = np.random.randint(0, len(vehicles))
    fig = plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(mpimg.imread(vehicles[car_ind]))
    plt.title('Original Image')
    plt.subplot(132)
    plt.plot(X[car_ind])
    plt.title('Raw Features')
    plt.subplot(133)
    plt.plot(scaled_X[car_ind])
    plt.title('Normalized Features')
    fig.tight_layout()
else:
    print('Your function only returns empty feature vectors...')

print('cspace', 'orient', 'pix_per_cell', 'cell_per_block', 'spatial_size', 'hist_bins', 'hist_range')
print(cspace, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hist_range)