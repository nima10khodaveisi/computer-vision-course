#!/usr/bin/env python
# coding: utf-8

# In[2]:


# get_ipython().system('pip3 install opencv-python ')
# get_ipython().system('pip3 install matplotlib ')


# In[20]:


import cv2
from matplotlib import pyplot as plt
import numpy as np


# In[22]:


size = (6, 9)

block_size = 22 * 1e-3 

myList = []

for i in range(20): 
    for j in range(size[0]):
        for k in range(size[1]): 
           myList.append([j * block_size, k * block_size, 0])
        
points = np.asarray(myList, dtype='float32')
points = points.reshape((20, size[0] * size[1], 3))

        


# In[18]:


images = []

for i in range(1, 21, 1):
    name = 'im'
    if i < 10: 
        name = name + '0' + str(i)
    else: 
        name = name + str(i)
    name += '.jpg'
    image = cv2.imread(name)
    images.append(image)
    

def find_camera_calib(arr):
    img_points = []
    for img in arr:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        r, cs1 = cv2.findChessboardCorners(gray, (size[1], size[0]), None)
        cs2 = cv2.cornerSubPix(gray, cs1, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
        img_points.append(cs2)
    img_points = np.array(img_points)
    print(points.shape, img_points.shape)
    _, m, _, _, _ = cv2.calibrateCamera(points[:len(arr)], img_points, (arr[0].shape[1], arr[0].shape[0]), None, None)
    _, m2, _, _, _ = cv2.calibrateCamera(points[:len(arr)], img_points, (arr[0].shape[1], arr[0].shape[0]), None, None, flags=cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_FIX_PRINCIPAL_POINT)
    return m, m2

x1, y1 = find_camera_calib(images[0:10])
x2, y2 = find_camera_calib(images[5:15])
x3, y3 = find_camera_calib(images[10:20])
x4, y4 = find_camera_calib(images[0:20])
print(x1)
print(x2)
print(x3)
print(x4)
print(y1)
print(y2)
print(y3)
print(y4)