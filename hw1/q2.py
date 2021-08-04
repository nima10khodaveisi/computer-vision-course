#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
from matplotlib import pyplot as plt
import numpy as np
import math


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


logo = cv2.imread('resources/logo.png')
logo = cv2.cvtColor(logo, cv2.COLOR_BGR2RGB)


# In[53]:


K = np.asarray([
    [500, 0, 128], 
    [0, 500, 128],
    [0, 0, 1]
])

Kp = np.asarray([
    [500, 0, 800],
    [0, 500, 800],
    [0, 0, 1]
])

teta = math.atan(40 / 25)

n = np.asarray([
    [0],
    [-math.sin(teta)],
    [-math.cos(teta)]
])

d = 25 

C = np.asarray([
    [0],
    [-math.cos(teta) * 40],
    [math.sin(teta) * 40]
])

R = np.asarray([
    [1, 0, 0],
    [0, math.cos(teta), -math.sin(teta)],
    [0, math.sin(teta), math.cos(teta)]
])

t = -np.matmul(R, C)

K_inv = np.linalg.inv(K)

P = R - (1 / d) * (np.matmul(t, np.transpose(n)))

H = np.matmul(np.matmul(Kp, P), K_inv)


result_image = cv2.warpPerspective(logo, H, dsize=(1600, 1600))
show_image(result_image, name='res12')

