import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt
#import DetectPlateImage
import DetectPlateVideo # For video processing

# Inverting the pixels 
license_plate = np.invert(DetectPlateVideo.plate_objects[0]) # For video processing
# license_plate = np.invert(DetectPlateImage.plate_objects[0]) # For image processing


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

label_plate = measure.label(license_plate)

'''label_img is ndarray of dtype int
   Labeled array, where all connected regions are assigned the same integer value.
'''

fig,(ax1) = plt.subplots(1)
ax1.imshow(license_plate, cmap = 'gray')


# Assuming the dimensions of characters on the number plate.
# Assumption : width of each char. must be greater than 5% and less than 15% of total plate width
#              height of each char. must be greater than 35% and less than 60% of total plate height

character_dim = (0.35 * license_plate.shape[0], 0.60 * license_plate.shape[0],\
				 0.03 * license_plate.shape[1], 0.15 * license_plate.shape[1])
min_height, max_height, min_width, max_width = character_dim 
# license_plate.shape[0] is width of license_plate similarly, 
# license_plate.shape[1] is height of license_plate

# We'll store the extracted characters and their order in chars[] and columns[] respectively
chars = []
columns = []

for region in regionprops(label_plate):
	# Cordinates of current boundry box
	min_row, min_col, max_row, max_col = region.bbox
	region_height = max_row - min_row
	region_width = max_col - min_col

	# Checking where the current region satisfies our assumed dimensions
	if region_height > min_height and region_height < max_height and\
		   region_width > min_width and region_width < max_width and\
		   region_width < region_height:
		   
		   # Select the region from min_row to max_row and min_col to max_col from the original binary image
		   region_of_interest = license_plate[min_row : max_row, min_col : max_col]

		   # Drawing the boundry box around the found character
		   rectBorder = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row,
		   								  linewidth = '2', fill = False, edgecolor = 'red')
		   ax1.add_patch(rectBorder)
		   # Resizing characters (boundry boxes to 20x20)
		   resized_char = resize(region_of_interest, (28,28))
		   chars.append(resized_char)
		   columns.append(min_col)
plt.show()

