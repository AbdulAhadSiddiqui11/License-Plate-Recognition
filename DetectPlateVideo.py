import cv2
import imutils
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage import measure
from skimage.measure import regionprops

# Change the path if required
path = './sample_clips/video12.mp4'

try:
	capture = cv2.VideoCapture(path)
	count = 0
	while capture.isOpened():
	    ret,frame = capture.read()
	    if ret == True:
	        cv2.imshow('Processing Video',frame)
	        cv2.resizeWindow('Processing Video', 923, 500)
	        cv2.imwrite("./output/frame%d.jpg" % count, frame)
	        count += 1
	        if cv2.waitKey(10) and 0xFF == ord('q') or count == 20:
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
car_img = imutils.rotate(car_img, 270)
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



'''Label connected regions of an integer array.
Two pixels are connected when they are neighbors and have the same value. 
In 2D, they can be neighbors either in a 1- or 2-connected sense. 
The value refers to the maximum number of orthogonal hops to consider a pixel/voxel a neighbor'''

'''1-connectivity     2-connectivity     diagonal connection close-up

     [ ]           [ ]  [ ]  [ ]             [ ]
      |               \  |  /                 |  <- hop 2
[ ]--[x]--[ ]      [ ]--[x]--[ ]        [x]--[ ]
      |               /  |  \             hop 1
     [ ]           [ ]  [ ]  [ ]
'''
label_img = measure.label(binary_img)

'''label_img is ndarray of dtype int
   Labeled array, where all connected regions are assigned the same integer value.
'''


# Assuming the various possible dimensions of a number_plate in an image
plate_dimensions = []
plate_dimensions.append((0.03*label_img.shape[0], 0.08*label_img.shape[0], 0.15*label_img.shape[1], 0.3*label_img.shape[1]))
plate_dimensions.append((0.08*label_img.shape[0], 0.2*label_img.shape[0], 0.15*label_img.shape[1], 0.4*label_img.shape[1]))
# label_img.shape[0] is width of car_img similarly, 
# label_img.shape[1] is height of car_img

# We'll store all the boundry boxes which 'may' be the require license plate and their cordinates here. 
plate_objects = []
plate_objects_cordinates = []

# Defining plots for detected plate
fig,(ax1) = plt.subplots(1)
ax1.imshow(gray_img, cmap='gray')
fig.suptitle('License Plate')
fig2,(ax2) = plt.subplots(1)
ax2.imshow(gray_img, cmap='flag')
fig2.suptitle('Abstract view')

# Flag to check if the number plate is found.
plate_found = False
# Flag to count the number of dimensions. 
dim_counter = 0


while not(plate_found):
	min_height, max_height, min_width, max_width = plate_dimensions[dim_counter]
	for region in regionprops(label_img):
		if region.area < 50:
			pass
		else:
			# Cordinates of current boundry box
			min_row, min_col, max_row, max_col = region.bbox
			region_height = max_row - min_row
			region_width = max_col - min_col

			# Checking where the current region satisfies our assumed dimensions
			if region_height >= min_height and region_height <= max_height\
			and region_width >= min_width and region_width <= max_width\
			and region_width > region_height:
				plate_found = True
				# Select the region from min_row to max_row and min_col to max_col from the original binary image
				plate_objects.append(binary_img[min_row:max_row, min_col:max_col])
				plate_objects_cordinates.append((min_row,min_col,max_row,max_col))
				# Drawing the boundry box around the found plate
				rectBorder = patches.Rectangle((min_col,min_row), max_col - min_col, max_row - min_row,
					                           edgecolor = 'red', linewidth = 2, fill = False )
				ax1.add_patch(rectBorder)
				# This plot is optional (You may remove it)
				rectBorder2 = patches.Rectangle((min_col,min_row), max_col - min_col, max_row - min_row,
					                           edgecolor = 'black',facecolor = 'black', linewidth = 2, fill = True )
				ax2.add_patch(rectBorder2)
				plt.show()
	dim_counter += 1
