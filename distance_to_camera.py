# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# I developed this code by using http://www.pyimagesearch.com/
# posts. Adrian posted fantastic image processing tutorials there.
# This code is ONLY used for personal hobby.

os.chdir('~\\Desktop\\Drone')

KNOWN_DISTANCE = 24.0
KNOWN_WIDTH = 11.0
 
def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)
 
	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	c = max(cnts, key = cv2.contourArea)
 
	# compute the bounding box of the of the paper region and return it
	return cv2.minAreaRect(c)
 
def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

# initialize the list of images that we'll be using
IMAGE_PATHS = ["Dis00.png", "Dis01.png", "Dis02.png", "Dis03.png", "Dis04.png", "Dis05.png"]
 
image = cv2.imread(IMAGE_PATHS[0])
marker = find_marker(image)
#focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH 
focalLength = 8400

# loop over the images
for imagePath in IMAGE_PATHS:
	# load the image, find the marker in the image, then compute the
	# distance to the marker from the camera
	image = cv2.imread(imagePath)
	# resize the image
	image = cv2.resize(image, (0,0), fx=0.35, fy=0.25)
	marker = find_marker(image)
	inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
 
	# draw a bounding box around the image and display it
	box = np.int0(cv2.cv.BoxPoints(marker))
	cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
	cv2.putText(image, "%.2fin" % (inches / 12),
		(image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
		2.0, (0, 255, 0), 3)
	cv2.imshow("image", image)
	cv2.waitKey(0)