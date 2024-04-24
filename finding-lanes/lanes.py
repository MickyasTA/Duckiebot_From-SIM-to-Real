import cv2 # by importing this we got access to imread() and imshow() functions
import numpy as np

# Edge detection algorithm using Canny
# Finding Lane lines (Gryascale convertion)
# Edge Detection:-- is identifying sharp changes in the intensity in adjacent pixels
# Gradient:-- is a measure of change in brightness over adjacent pixels


image = cv2.imread('test_image.jpg') # read the image
lane_image=np.copy(image) # copying our array in to a new value not to make change to the first variable latter 

gray=cv2.cvtColor(lane_image,cv2.COLOR_BGR2GRAY) # converting to Gray
cv2.imshow('Gray Image',gray)# the first argument is the name of the window that we open and the second is the name of the image that we want to show 

#cv2.imshow('result',image)# the first argument is the name of the window that we open and the second is the name of the image that we want to show 
cv2.waitKey()
