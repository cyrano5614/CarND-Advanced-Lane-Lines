import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


image = mpimg.imread('camera_cal/signs_vehicles_xygrad.png')


def grayscale(img):
	return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def normalize(sobel_in):
	return np.uint8(255 * sobel_in / np.max(sobel_in))

def sobel_helper(img, orient = 'x', k = 3):
	if orient == 'x':
		return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=k)
	if orient =='y':
		return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=k)

def binary_output(output, thresh=(0, 255)):
	temp = np.zeros_like(output)
	temp[(output >= thresh[0]) & (output <= thresh[1])] = 1
	return temp


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):

	gray = grayscale(img)
	
	if orient =='x':
		sobel = sobel_helper(gray, 'x', sobel_kernel)
		abs_sobel = np.absolute(sobel)
	if orient =='y':
		sobel = sobel_helper(gray, 'y', sobel_kernel)
		abs_sobel = np.absolute(sobel)

	norm_sobel = normalize(abs_sobel)

	output = binary_output(norm_sobel, thresh)

	return output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):

	gray = grayscale(img)

	sobelx = sobel_helper(gray, 'x', sobel_kernel)
	sobely = sobel_helper(gray, 'y', sobel_kernel)

	mag_sobel = np.sqrt((sobelx ** 2) + (sobely ** 2))

	norm_sobel = normalize(mag_sobel)

	output = binary_output(norm_sobel, mag_thresh)

	return output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):

	gray = grayscale(img)

	sobelx = sobel_helper(gray, 'x', sobel_kernel)
	sobely = sobel_helper(gray, 'y', sobel_kernel)

	grad_dir = np.arctan2(np.abs(sobely), np.abs(sobelx))

	output = binary_output(grad_dir, thresh)

	return output

ksize = 3

gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(0, 255))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(100, 255))
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0, np.pi/2))

combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

plt.imshow(combined)
plt.show()