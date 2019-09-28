import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import SegmentChars
from PIL import Image
from io import BytesIO
import pytesseract
#import DetectPlateVideo # For video processing

def predict():
	license_plate = SegmentChars.DetectPlateVideo.plate_objects[0]
	# license_plate = SegmentChars.DetectPlateImage.plate_objects[0] # For image processing
	plate = pytesseract.image_to_string(license_plate).strip(' !@#$%^&*()_+{\\}\\[];/><,.-—')
	print("License plate is :{}".format(plate))
	print("\n\n\n")
	return plate

predict()