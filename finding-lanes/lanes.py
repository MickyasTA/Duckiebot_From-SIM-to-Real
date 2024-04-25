import cv2 ## by importing this we got access to imread() and imshow() functions 
import numpy as np
import matplotlib.pyplot as plt

# detecting Edges using canny 
def CannyImg(lane_image):
    gray =cv2.cvtColor(lane_image,cv2.COLOR_BGR2GRAY)
    blur =cv2.GaussianBlur(gray,(5,5),0)
    canny_img=cv2.Canny(blur,10,150)
    return canny_img

# Creating a mask
def Region_of_interest(image):
    hight=image.shape[0]
    polygone=np.array([[(200,hight),(1100,hight),(550,250)]])
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,polygone,255)# this will fill the mask with the color white
    # Finding Lane Lines Bitwise_and
    masked_image = cv2.bitwise_and(image,mask) # this will show the lane lines only
    return masked_image 



    
image=cv2.imread("test_image.jpg")
lane_image=np.copy(image)
Modified_Canny_Image=CannyImg(lane_image)

cv2.imshow("mage",Region_of_interest(image))
cv2.imshow("Modified_Canny_Image",Modified_Canny_Image)
cv2.waitKey(0)
 
# we can also use matplotlib to show the images
#plt.imshow(Modified_Canny_Image)
#plt.show()
 

 












#_____________________theory______________________

"""# Edge detection algorithm using Canny
# Finding Lane lines (Gryascale convertion)
# Edge Detection:-- is identifying sharp changes in the intensity in adjacent pixels
# Gradient:-- is a measure of change in brightness over adjacent pixels


image = cv2.imread('test_image.jpg') # read the image
lane_image=np.copy(image) # copying our array in to a new value not to make change to the first variable latter 

gray=cv2.cvtColor(lane_image,cv2.COLOR_BGR2GRAY) # converting to Gray
# cv2.imshow('Gray Image',gray)# the first argument is the name of the window that we open and the second is the name of the image that we want to show 

#cv2.imshow('result',image)# the first argument is the name of the window that we open and the second is the name of the image that we want to show 
# cv2.waitKey()


# Finding Lane Line (Gaussian Blur)
# Gaussian Blur:-- is used to reduce noise in the image
# Noise:-- is the random variation of brightness or color in the image
# Kernel:-- is a matrix that slides over the image and does a mathematical operation on the pixels
# Gaussian Blur:-- is a way to reduce noise in the image by averaging the pixels

blur=cv2.GaussianBlur(gray,(5,5),0) # the second argument is the kernel size and the third is the deviation 

# Edge Detection (Canny)
# Canny:-- is an edge detection algorithm that detects a wide range of edges in images. 
#          it is a multi-step algorithm that involves smoothing the image, finding the gradient, and applying non-maximum suppression

canny=cv2.Canny(blur,50,150) # the second and third arguments are the threshold values that we use to detect the edges in the image 

cv2.imshow("cann y",canny)
cv2.waitKey(0)"""


