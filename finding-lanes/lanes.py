import cv2 # by importing this we got access to imread() and imshow() functions

image = cv2.imread('test_image.jpg') # read the image

cv2.imshow('result',image)# the first argument is the name of the window that we open and the second is the name of the image that we want to show 

# cv2.waitKey()