# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# I developed this code by using http://www.pyimagesearch.com/
# posts. Adrian posted fantastic image processing tutorials there.
# This code is ONLY used for personal hobby.

os.chdir('~\\Desktop\\Drone')

IMAGE_PATHS = ["circle6.png"]
image = cv2.imread(IMAGE_PATHS[0])

# load the image, clone it for output, and then convert it to grayscale
output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect circles in the image
circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1.8, 70)
 
# ensure at least some circles were found
if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circles = np.round(circles[0, :]).astype("int")
 
	# loop over the (x, y) coordinates and radius of the circles
	for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
 
	# show the output image
	image = cv2.resize(output, (0,0), fx=0.50, fy=0.50)
	#cv2.imshow("output", np.hstack([image, output]))
	cv2.imshow("output", image)
	cv2.waitKey(0)
