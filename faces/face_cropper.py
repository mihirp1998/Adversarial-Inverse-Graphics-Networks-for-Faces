#!/usr/bin/python

import sys
import os
import dlib
import pickle
import cv2
import numpy as np
import random
import time
from PIL import Image
import pandas as pd
import errno

def face_detection(detector, image, upsample_factor):
	return detector(image, upsample_factor)

def _main(child_dir,save_dir_path):
	
	# Initialize hog face detector
	detector = dlib.get_frontal_face_detector()

	# Initialize cnn face detector
	# cnn_face_detector_path = cnn_path
	# cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detector_path)

	# DETECTION_MODEL = 'cnn' # or 'cnn'
	UPSAMPLE_FACTOR = 1
	margin = 10

	label_count = 0
	try:
		os.makedirs(save_dir_path)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise
		else:
			pass
				
		print(child_dir)
		for image_name in os.listdir(child_dir):
			image_path = child_dir+'/'+image_name
			print(image_path)
			
			try:
				image = cv2.imread(image_path)	
				imgheight,imgwidth = image.shape[:2]
				dets = face_detection(detector, image, UPSAMPLE_FACTOR)
				# Now process each face we found.
				face_encodings = []
				for k, d in enumerate(dets):
					
					# if DETECTION_MODEL == 'cnn':
					# 	left, top, right, bottom = d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()
					# else:
					left, top, right, bottom = 	d.left(), d.top(), d.right(), d.bottom()
					
					marginV =  int(abs(top - bottom) * 0.5)
					marginV1 = marginV
					marginH = int(abs(left - right) *0.2)
					marginH1 = marginH

					print(marginH,marginV,image.shape)
					print(top,marginV)
					#save_image_name = int(round(time.time() * 1000))
					if top < marginV1:
						marginV1 = top

					if left < marginH1:
						marginH1 = left 
					# marginV1,marginV,marginH1,marginH = (0,0,0,0)
					cropped_img = image[top-marginV1:bottom+marginV, left-marginH1:right+marginH]
					
					height, width = cropped_img.shape[:2]
					
					top = max(0, top+2)
					left = max(0, left+2)
					right = min(width, right+2)
					bottom = min(bottom, right+2)
					
					save_dir = save_dir_path
					if not os.path.exists(save_dir):
						os.mkdir(save_dir)

					cv2.imwrite(save_dir+'/'+image_name, cropped_img)
			except Exception as e:
				print image_path, e
				continue

if __name__ == '__main__':
	_main(sys.argv[1],sys.argv[2])

