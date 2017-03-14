import cv
import cv2
import numpy as np

data_list='./dataset/train.txt'
data_dir='/tmp3/haowei/VOCdevkit/VOC2012'
write_dir='/tmp3/haowei/VOCdevkit/VOC2012/Erosion'
KERNEL_SIZE=5
ITER_NUM=1

f = open(data_list, 'r')
images = []
masks = []
for line in f:
	try:
		image, mask = line.strip("\n").split(' ')
	except ValueError: # Adhoc for test.
		image = mask = line.strip("\n")
	im=cv2.imread(data_dir+mask,0)
	ret,im_bw=cv2.threshold(im,0,255,cv2.THRESH_BINARY)
	kernel=np.ones((KERNEL_SIZE,KERNEL_SIZE),np.uint8)
	dilation=cv2.erode(im_bw,kernel,iterations=ITER_NUM)
	cv2.imwrite(write_dir+mask,dilation)
