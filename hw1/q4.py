#!/usr/bin/env python
# coding: utf-8

# In[39]:


import cv2
from matplotlib import pyplot as plt
import numpy as np
import random
import math


# In[40]:


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


# In[41]:


smallImage = cv2.imread('resources/im03.jpg')
smallImage = cv2.cvtColor(smallImage, cv2.COLOR_BGR2RGB)

bigImage = cv2.imread('resources/im04.jpg')
bigImage = cv2.cvtColor(bigImage, cv2.COLOR_BGR2RGB)


# In[42]:


sift = cv2.SIFT_create()

kp1, desc1 = sift.detectAndCompute(smallImage, None)
kp2, desc2 = sift.detectAndCompute(bigImage, None)


# In[43]:


bf_matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf_matcher.match(desc1, desc2)
matches = sorted(matches, key = lambda x: x.distance)


# In[44]:


small_image_match_list = np.zeros((len(matches), 3), dtype='float32')
big_image_match_list = np.zeros((len(matches), 3), dtype='float32')

for i in range(len(matches)):
    match = matches[i]
    small_image_match_list[i] = [kp1[match.queryIdx].pt[0], kp1[match.queryIdx].pt[1], 1]
    big_image_match_list[i] = [kp2[match.trainIdx].pt[0], kp2[match.trainIdx].pt[1], 1]

print("len of matches ", len(matches))


# In[45]:


def dist(x, y, xx, yy):
    return math.sqrt((x - xx) ** 2 + (y - yy) ** 2)


# In[47]:


def solve(A):
    import numpy.linalg.linalg as linalg
    u, s, v = linalg.svd(A)
    h = np.reshape(v[8], (3, 3))
    h = (1 / h[2, 2]) * h
    return h

def calc_homography(src, dst, myList):
    n = src.shape[0]
    A = np.zeros((2 * len(myList), 9), dtype='float32')
    for ind in range(len(myList)):
            p_ind = myList[ind]
            x = src[p_ind, 0]
            y = src[p_ind, 1]
            z = 1
            xx = dst[p_ind, 0]
            yy = dst[p_ind, 1]
            zz = 1
            A[2*ind, 3] = -x
            A[2*ind, 4] = -y
            A[2*ind, 5] = -z
            A[2*ind, 6] = x * yy
            A[2*ind, 7] = y * yy
            A[2*ind, 8] = yy
            A[2*ind + 1, 0] = x
            A[2*ind + 1, 1] = y
            A[2*ind + 1, 2] = z
            A[2*ind + 1, 6] = -x * xx
            A[2*ind + 1, 7] = -y * xx
            A[2*ind + 1, 8] = -xx
    return solve(A)

def calc_consistents(src, dst, homography, threshold):
    n = src.shape[0]
    result = np.transpose(np.matmul(homography, np.transpose(src)))
    try:
        result[:, 0] /= result[:, 2]
        result[:, 1] /= result[:, 2]
        result[:, 2] /= result[:, 2]
    except:
        print("divide by zero")
        return []

    result[:, 0] = np.subtract(result[:, 0], dst[:, 0])
    result[:, 1] = np.subtract(result[:, 1], dst[:, 1])
    result[:, 2] = np.subtract(result[:, 2], dst[:, 2])

    result = np.power(result, 2)
    distances = np.zeros((result.shape[0]), dtype='float32')

    distances = result[:, 0] + result[:, 1] + result[:, 2]

    points = np.argwhere(distances <= threshold)
    points = np.reshape(points, (points.shape[0]))

    return points

def find_homography(src, dst, iterations=200000, threshold=100):
    n = src.shape[0]
    consistents_points = []
    best_homography = np.zeros((3, 3), dtype='float32')
    for it in range(iterations):
        randomList = random.sample(range(n), 4)
        homography = calc_homography(src, dst, randomList)
        my_consistents = calc_consistents(src, dst, homography, threshold)
        if len(my_consistents) > len(consistents_points):
            consistents_points = my_consistents.copy()
            best_homography = homography.copy()
    best_homography = calc_homography(src, dst, consistents_points)

    return best_homography, consistents_points

homography, consistents = find_homography(big_image_match_list, small_image_match_list, 500000)


# In[53]:


warp_image = cv2.warpPerspective(bigImage, homography, (5000, 2300))
show_image(warp_image, name='res20')

