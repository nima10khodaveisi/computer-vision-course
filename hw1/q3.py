#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
from matplotlib import pyplot as plt
import numpy as np


# In[2]:


def fix_image_to_show(img):
    image = img.copy()
    image = image.astype('float32')
    max_pixel = np.max(image)
    min_pixel = np.min(image)
    if max_pixel == min_pixel:
        image[:, :, :] = 127
        image = image.astype('uint8')
        return image
    m = 255 / (max_pixel - min_pixel)
    image = image * m - min_pixel * m
    image = image.astype('uint8')
    return image

def show_image(img, name=None, cmap=None):
    image = img.copy()
    image = fix_image_to_show(image)
    plt.imshow(image, vmin=0, vmax=255, cmap=cmap)
    plt.show()
    if name != None:
        plt.imsave(name + '.jpg', image, cmap=cmap)


# In[3]:


smallImage = cv2.imread('resources/im03.jpg')
smallImage = cv2.cvtColor(smallImage, cv2.COLOR_BGR2RGB)

bigImage = cv2.imread('resources/im04.jpg')
bigImage = cv2.cvtColor(bigImage, cv2.COLOR_BGR2RGB)


# In[4]:


sift = cv2.SIFT_create()

kp1, desc1 = sift.detectAndCompute(smallImage, None)
kp2, desc2 = sift.detectAndCompute(bigImage, None)


# In[5]:



sift_small_image = cv2.drawKeypoints(smallImage, kp1, smallImage.copy(), color=(0, 255, 0))
sift_big_image = cv2.drawKeypoints(bigImage, kp2, bigImage.copy(), color=(0, 255, 0))

diff_image = np.zeros((smallImage.shape[0] - bigImage.shape[0], bigImage.shape[1], 3), dtype='uint8')

sift_big_image = np.concatenate((sift_big_image, diff_image), axis=0)
sift_image = np.concatenate((sift_small_image, sift_big_image), axis=1)
show_image(sift_image, 'res03_corners')


# In[6]:


bf_matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf_matcher.match(desc1, desc2)
matches = sorted(matches, key = lambda x: x.distance)


# In[7]:


small_image_matches = [kp1[o.queryIdx] for o in matches]
big_image_matches = [kp2[o.trainIdx] for o in matches]

sift_small_image1 = cv2.drawKeypoints(smallImage, kp1, smallImage.copy(), color=(0, 255, 0))
sift_small_image1 = cv2.drawKeypoints(sift_small_image1, small_image_matches, smallImage.copy(), color=(0, 0, 255))

sift_big_image1 = cv2.drawKeypoints(bigImage, kp2, bigImage.copy(), color=(0, 255, 0))
sift_big_image1 = cv2.drawKeypoints(sift_big_image1, big_image_matches, bigImage.copy(), color=(0, 0, 255))

sift_big_image1 = np.concatenate((sift_big_image1, diff_image), axis=0)

sift_image1 = np.concatenate((sift_small_image1, sift_big_image1), axis=1)
show_image(sift_image1, 'res14_correspondences')


# In[8]:


matched_image = cv2.drawMatches(smallImage, kp1, bigImage, kp2, matches, sift_image1, flags=2, matchColor=(0, 0, 255))
show_image(matched_image, 'res15_matches')

matched_image = cv2.drawMatches(smallImage, kp1, bigImage, kp2, matches[:20], sift_image1, flags=2, matchColor=(0, 0, 255))
show_image(matched_image, 'res16')


# In[9]:


small_image_match_list = np.zeros((len(matches), 1, 2), dtype='float32')
big_image_match_list = np.zeros((len(matches), 1, 2), dtype='float32')

for i in range(len(matches)):
    match = matches[i]
    small_image_match_list[i] = kp1[match.queryIdx].pt
    big_image_match_list[i] = kp2[match.trainIdx].pt

homography, mask = cv2.findHomography(big_image_match_list, small_image_match_list, cv2.RANSAC, maxIters=200000, ransacReprojThreshold=5)


# In[10]:


small_image_inliers = []
big_image_inliers = []
inliers_mathces = []
cnt = 0

for i in range(len(matches)):
    if mask[i, 0] == 1:
        small_image_inliers.append(kp1[matches[i].queryIdx])
        big_image_inliers.append(kp2[matches[i].trainIdx])
        inliers_mathces.append(cv2.DMatch(_imgIdx=0, _distance=matches[i].distance, _queryIdx=cnt, _trainIdx=cnt))
        cnt = cnt + 1
small_image_show_inliers = cv2.drawKeypoints(smallImage, small_image_matches, smallImage.copy(), color=(0, 0, 255))
small_image_show_inliers = cv2.drawKeypoints(small_image_show_inliers, small_image_inliers, smallImage.copy(), color=(255, 0, 0))

big_image_show_inliers = cv2.drawKeypoints(bigImage, big_image_matches, bigImage.copy(), color=(0, 0, 255))
big_image_show_inliers = cv2.drawKeypoints(big_image_show_inliers, big_image_inliers, bigImage.copy(), color=(255, 0, 0))

big_image_show_inliers1 = np.concatenate((big_image_show_inliers, diff_image), axis=0)

inliers_show_image = np.concatenate((small_image_show_inliers, big_image_show_inliers1), axis=1)

inlier_matched_image = cv2.drawMatches(small_image_show_inliers, small_image_inliers, big_image_show_inliers, big_image_inliers, inliers_mathces, inliers_show_image, matchColor=(255, 0, 0), flags=2)

show_image(inlier_matched_image, name='res17')


# In[11]:


warp_image = cv2.warpPerspective(bigImage, homography, (5000, 4000))
show_image(warp_image, 'res19')

