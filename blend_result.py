import cv
import cv2
import numpy as np
import os

JPG_DIR = '/tmp3/haowei/Dataset/DAVIS-data/DAVIS/JPEGImages/480p/bus/'
PNG_DIR = 'multi_output/DAVIS/480p/bus_mix_20k/'
SAVE_DIR = 'multi_output/DAVIS/480p/bus_mix_20k/blend'
if not os.path.isdir(SAVE_DIR):
    print 'hi'
    os.makedirs(SAVE_DIR)
START=0
IMG_NUM=50

for i in range(START,IMG_NUM+1):
    im_jpg = cv2.imread(JPG_DIR+'%05d'%i+'.jpg')
    im_png = cv2.imread(PNG_DIR+'%05d'%i+'.png')
    gray = cv2.cvtColor(im_png, cv.CV_BGR2GRAY)
    im_mask = np.zeros_like(im_jpg)
    im_mask[:,:,1] = gray * 0.3
    dst = cv2.add(im_jpg, im_mask)
    cv2.imwrite(SAVE_DIR+'/%05d'%i+'.jpg',dst)
