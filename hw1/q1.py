#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
from matplotlib import pyplot as plt
import numpy as np
import math


# In[ ]:


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


# In[ ]:


def dist(x0, y0, x1, y1):
    return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)


# In[ ]:


def non_maximum_suppression(img):
    image = img.copy()
    search = np.where(image > 0)
    interest_points = list(zip(search[0], search[1]))

    print(interest_points.__len__())

    final_list = []

    for c1 in interest_points:
        min_distance = 100 * 1000 * 1000
        for c2 in interest_points:
            if c1 != c2 and image[c2[0], c2[1]] > image[c1[0], c1[1]]:
                min_distance = min(dist(c1[0], c1[1], c2[0], c2[1]), min_distance)

        final_list.append([min_distance, c1[0],  c1[1]])

    final_list.sort(key=lambda x: x[0])
    number = 8000
    final_list = final_list[0: number]

    res = np.zeros(image.shape, dtype='uint8')
    for c in final_list:
        res[c[1], c[2]] = 255
    return res


def find_interest_points(img, number):
    image = img.copy()
    sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)
    sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)

    sobelX = sobelX.astype('float32')
    sobelY = sobelY.astype('float32')

    sobelX2 = sobelX * sobelX
    sobelY2 = sobelY * sobelY
    sobelXY = sobelX * sobelY

    gradient = np.sqrt(sobelX2 + sobelY2)
    if number == 1:
        show_image(gradient, name='res01_grad', cmap='gray')
    else:
        show_image(gradient, name='res02_grad', cmap='gray')

    sigma = 2
    Euu = cv2.GaussianBlur(sobelX2, (3, 3), sigmaX=sigma)
    Evv = cv2.GaussianBlur(sobelY2, (3, 3), sigmaX=sigma)
    Euv = cv2.GaussianBlur(sobelXY, (3, 3), sigmaX=sigma)

    Euu = Euu.astype('float32')
    Evv = Evv.astype('float32')
    Euv = Euv.astype('float32')

    det = Euu * Evv - Euv * Euv
    trace = Euu + Evv

    k = 0.04
    R_3d = det - k * trace * trace

    R = R_3d[:, :, 0] + R_3d[:, :, 1] + R_3d[:, :, 2]

    if number == 1:
        show_image(R, name='res03_score', cmap='gray')
    else:
        show_image(R, name='res04_score', cmap='gray')

    R_copy = R.copy()
    threshhold = 1000 * 1000
    R[R < threshhold] = 0
    R[R > threshhold] = 255
    if number == 1:
        show_image(R, name='res05_thresh', cmap='gray')
    else:
        show_image(R, name='res06_thresh', cmap='gray')
    R = R_copy.copy()
    R[R < threshhold] = 0
    return R, non_maximum_suppression(R)


image1 = cv2.imread('resources/im01.jpg')
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image1 = image1.astype('float32')

show_image(image1)

image2 = cv2.imread('resources/im02.jpg')
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
image2 = image2.astype('float32')

show_image(image2)

image1_copy = image1.copy()

R1, R_n1 = find_interest_points(image1, 1)
image1_copy[R_n1 > 0, 0] = 255
image1_copy[R_n1 > 0, 1] = image1_copy[R_n1 > 0, 2] = 0
show_image(image1_copy, 'res07_harris')

R2, R_n2 = find_interest_points(image2, 2)
image2_copy = image2.copy()
image2_copy[R_n2 > 0, 0] = 255
image2_copy[R_n2 > 0, 1] = image2_copy[R_n2 > 0, 2] = 0
show_image(image2_copy, 'res08_harris')


# In[ ]:


def get_features(img, R):
    image = img.copy()
    search = np.where(R > 0)
    interest_points = list(zip(search[0], search[1]))

    n = 2 # (2n + 1) * (2n + 1)

    features = []

    for c in interest_points:
        x = c[0]
        y = c[1]
        feature = []
        for dx in range(x - n, x + n):
            for dy in range(y - n, y + n):
                for channel in range(3):
                    feature.append(image[dx, dy, channel])

        features.append([x, y, feature])

    return features

def vector_dist(f1, f2):
    a = np.array(f1)
    b = np.array(f2)
    return np.linalg.norm(a - b)

features1 = get_features(image1, R_n1)
features2 = get_features(image2, R_n2)


# In[ ]:


def get_close_list(features1, features2):
    close_list = []
    for p1 in features1:
        closest = []
        second_closest = []
        for p2 in features2:
            f1 = p1[2]
            f2 = p2[2]
            distance = vector_dist(f1, f2)
            if closest.__len__() > 0:
                if distance < closest[2]:
                    closest = [p2[0], p2[1], distance]
                else:
                    if second_closest.__len__() > 0:
                        if distance < second_closest[2]:
                            second_closest = [p2[0], p2[1], distance]
                    else:
                        second_closest = [p2[0], p2[1], distance]
            else:
                closest = [p2[0], p2[1], distance]

        d = (second_closest[2] / closest[1])

        close_list.append([p1[0], p1[1], closest])


    return close_list


close_list1 = get_close_list(features1, features2)
close_list2 = get_close_list(features2, features1)


# In[ ]:


close_list_dict1 = {}
for c in close_list1:
    close_list_dict1[(c[0], c[1])] = (c[2][0], c[2][1])

close_list_dict2 = {}
for c in close_list2:
    close_list_dict2[(c[0], c[1])] = (c[2][0], c[2][1])

match1 = []
match2 = []
for c in close_list1:
    if close_list_dict2.get((c[2][0], c[2][1])) == (c[0], c[1]):
        match1.append([c[0], c[1]])
        match2.append([c[2][0], c[2][1]])


# In[ ]:


image1 = image1.astype('uint8')
image2 = image2.astype('uint8')

final_image = np.concatenate((image1, image2), axis = 1)
for i in range(len(match1)):
    if i % 30 != 5:
        continue
    c0 = match1[i]
    c1 = match2[i]
    final_image = cv2.line(final_image, (c0[1], c0[0]), (c1[1] + image1.shape[1], c1[0]), color=(255, 0, 0), thickness=1)

plt.imshow(final_image)
plt.show()
plt.imsave('res11.jpg', final_image)

