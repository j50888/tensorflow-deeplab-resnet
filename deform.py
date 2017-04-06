import cv
import cv2
import os
import numpy as np
import random

data_list='./mix_dataset/train.txt'
data_dir='/tmp3/haowei/Dataset'
write_dir='/tmp3/haowei/Dataset/Deform'
KERNEL_SIZE=11
ITER_NUM=1
SCALE_RATE=0.05
TRANSLATE_RATE=0.1
NOT_EXIST=1
if NOT_EXIST:
    os.makedirs(write_dir+'/ECSSD/ground_truth_mask')
    os.makedirs(write_dir+'/PASCAL-S/datasets/masks/ft')
    os.makedirs(write_dir+'/PASCAL-S/datasets/masks/imgsal')
    os.makedirs(write_dir+'/MSRA10K/MSRA10K_Imgs_GT/Imgs')



f = open(data_list, 'r')
images = []
masks = []
for line in f:
    try:
	    image, mask = line.strip("\n").split(' ')
    except ValueError: # Adhoc for test.
	    image = mask = line.strip("\n")
    im = cv2.imread(data_dir+mask)
    ret,im_bw = cv2.threshold(im,0,255,cv2.THRESH_BINARY)
    h,w = im.shape[:2]
    scale=1-SCALE_RATE+2*SCALE_RATE*random.random()
    translate_x = (-TRANSLATE_RATE)+2*TRANSLATE_RATE*random.random()
    translate_y = (-TRANSLATE_RATE)+2*TRANSLATE_RATE*random.random()
    M = np.float32([[scale,0,translate_x*w],[0,scale,translate_y*h]])
    dst = cv2.warpAffine(im_bw,M,(w,h))
    kernel = np.ones((KERNEL_SIZE,KERNEL_SIZE),np.uint8)
    dilation = cv2.dilate(dst,kernel,iterations=ITER_NUM)
#    filename = os.path.basename(mask)
    cv2.imwrite(write_dir+mask,dilation)
