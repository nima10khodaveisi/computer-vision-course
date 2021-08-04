#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('pip3 install opencv-python')
# get_ipython().system('pip3 install matplotlib ')


# In[1]:


import cv2
import numpy as np
import gc


video = cv2.VideoCapture('video.mp4')
frames = []
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def imply_homography_to_points(points, homography):
    res_points = np.transpose(np.matmul(homography, np.transpose(points)))
    res_points[:, 0] /= res_points[:, 2]
    res_points[:, 1] /= res_points[:, 2]
    return res_points[:, :-1].astype('int32')


def imply_homography_to_image(image, homography, size):
    translation_mat = np.asarray([
        [1, 0, 2000],
        [0, 1, 600],
        [0, 0, 1],
    ], dtype='float32')

    if homography is not None:
        homography = np.matmul(translation_mat, homography)
    else:
        homography = translation_mat

    warp_image = cv2.warpPerspective(image.copy(), homography, size)
    return warp_image


def get_translated_homography(image, homography):
    points_homography = imply_homography_to_points(
        np.asarray([
            [0, 0, 1],
            [image.shape[0], 0, 1],
            [0, image.shape[1], 1],
            [image.shape[0], image.shape[1], 1]
        ]),
        homography
    )

    min_values = points_homography.min(axis=0)
    tx = 0
    if min_values[0] < 0:
        tx = -min_values[0]
    ty = 0
    if min_values[1] < 0:
        ty = -min_values[1]

    translation_mat = np.asarray([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1],
    ], dtype='float32')

    homography = np.matmul(translation_mat, homography)

    return homography, translation_mat

homographies = np.load('homographies.npy')

scale_mat = np.asarray([
    [480/1080, 0, 0],
    [0, 480/1080, 0],
    [0, 0, 1],
])

inv_scale = np.linalg.inv(scale_mat)

for i in range(900):
    homographies[i] = np.matmul(inv_scale, np.matmul(homographies[i], scale_mat))

gc.collect()


block = 50

size = (5900, 2500)

import math

arr = np.zeros((math.floor(size[0] / block), 230, size[1], block, 3), dtype='uint8')
image_count = np.zeros((math.floor(size[0] / block)), dtype='uint8')


for idx in range(900): 
    cnt_block = 0
    flag = False 
    image = imply_homography_to_image(frames[idx].copy(), homographies[idx], size).astype('uint8')
    for t in range(0, size[0] - block, block): 
        min_x = t 
        max_x = t + block
        if image_count[cnt_block] >= 230:
            cnt_block = cnt_block + 1
            continue
        if np.max(image[:, min_x: max_x, :]) > 0: 
            flag = True
            arr[cnt_block, image_count[cnt_block], :, :, :] = image[:, min_x: max_x, :]
            image_count[cnt_block] = image_count[cnt_block] + 1
        elif flag == True:
            break 
        
        cnt_block = cnt_block + 1 


# In[ ]:


res06 = np.zeros((2500, 5900, 3), dtype='uint8')

for i in range(math.floor(size[0] / block) ):
    min_x = i * block
    max_x = (i + 1) * block
    if np.max(arr[i]) == 0:
        continue
    res06[:, min_x: max_x, :] = np.ma.median(np.ma.masked_where(arr[i, :image_count[i]] == 0, arr[i, :image_count[i]]), axis=0).filled(0).astype(dtype=np.uint8)
    
del arr
gc.collect()

# In[ ]:


cv2.imwrite('res06-background-panaroma.jpg', cv2.cvtColor(res06, cv2.COLOR_RGB2BGR))


# In[ ]:


writer = cv2.VideoWriter('res07-background-video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frames[0].shape[1], frames[0].shape[0]))


for i in range(900): 
    homo = homographies[i].copy()
    translation_mat = np.asarray([
        [1, 0, 2000],
        [0, 1, 600],
        [0, 0, 1],
    ], dtype='float32')
    homo = np.linalg.inv(np.matmul(translation_mat, homo))
    image = cv2.warpPerspective(res06.copy(), homo, (frames[0].shape[1], frames[0].shape[0]))
    writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
writer.release()


# In[ ]:


import math
writer = cv2.VideoWriter('res09-background-video-wider.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (math.floor(frames[0].shape[1] * (3 / 2)), frames[0].shape[0]))


for i in range(600): 
    homo = homographies[i].copy()
    translation_mat = np.asarray([
        [1, 0, 2000],
        [0, 1, 600],
        [0, 0, 1],
    ], dtype='float32')

    homo = np.linalg.inv(np.matmul(translation_mat, homo))
    image = cv2.warpPerspective(res06.copy(), homo, (math.floor(frames[0].shape[1] * (3 / 2)), frames[0].shape[0]))
    writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
writer.release()


# In[ ]:


writer = cv2.VideoWriter('res08-foreground-video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frames[0].shape[1], frames[0].shape[0]))



for i in range(900): 
    homo = homographies[i].copy()
    translation_mat = np.asarray([
        [1, 0, 2000],
        [0, 1, 600],
        [0, 0, 1],
    ], dtype='float32')
    homo = np.linalg.inv(np.matmul(translation_mat, homo))
    image1 = cv2.warpPerspective(res06.copy(), homo, (frames[0].shape[1], frames[0].shape[0]))
    
    image2 = frames[i].copy()

    image1 = image1.astype('float32')
    image2 = image2.astype('float32')
    
    image = image1 - image2 
    image = abs(image)
    
    image[image > 255] = 255 
    
    image = image.astype('uint8')
    
    sum_image = image[:, :, 0] + image[:, :, 1] + image[:, :, 2]
    
    thr = 225

    
    image2[sum_image > thr, 0] = 255
    image2[sum_image > thr, 1] = image2[sum_image > thr, 2] = 0
    
    image2 = image2.astype('uint8')
    
    writer.write(cv2.cvtColor(image2, cv2.COLOR_RGB2BGR))
    
    
writer.release()
    

