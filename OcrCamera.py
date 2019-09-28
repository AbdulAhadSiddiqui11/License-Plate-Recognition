import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from PIL import Image
from io import BytesIO
import pytesseract
import cv2
import imutils
#import DetectPlateVideo # For video processing

path = './sample_clips/video12.mp4'

try:
	capture = cv2.VideoCapture(0)
	count = 0
	while capture.isOpened():
	    ret,frame = capture.read()
	    if ret == True:
	        cv2.imshow('Processing Video',frame)
	        cv2.resizeWindow('Processing Video', 923, 500)
	        cv2.imwrite("./output/frame%d.jpg" % count, frame)
	        count += 1
	        if cv2.waitKey(10) and 0xFF == ord('q') or count == 30:
	            break
	    else:
	        break
	capture.release()
	cv2.destroyAllWindows()	
except:
	print("File doesn't exists - Terminating process")
	exit(0)

# Load the car image with license plate 
car_img = imread("./output/frame%d.jpg"%(count-1), as_gray=True)
# Normalizing the pixles
gray_img = car_img * 255
# Defining subplots
fig, (ax1,ax2) = plt.subplots(1,2)
fig.suptitle('Processed Images')
ax1.imshow(gray_img, cmap = 'gray')
ax1.set_title('Gray Image')
# Performing Otsu Thresholding (To separate foreground / background)
# Read more : https://en.wikipedia.org/wiki/Otsu%27s_method
threshold_val = threshold_otsu(gray_img)
# Each pixel of binary image is True (1) if the corresponding pixel in
# gray image has intensity value greater than thrrshold_val
binary_img = gray_img > threshold_val
ax2.imshow(binary_img, cmap = 'gray')
ax2.set_title('Binary Image')
plt.show()


def predict():
	license_plate = binary_img
	# license_plate = SegmentChars.DetectPlateImage.plate_objects[0] # For image processing
	plate = pytesseract.image_to_string(license_plate).strip(' !@#$%^&*()_+{\\}\\[];/><,.-')
	print("License plate is :{}".format(plate))
	print("\n\n\n")
	return plate
