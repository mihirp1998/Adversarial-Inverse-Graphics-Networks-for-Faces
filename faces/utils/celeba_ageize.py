

import shutil
import csv
import os
from glob import glob


data_dir = '/home/wseto/datasets/img_align_celeba128'
old_path = '/home/wseto/datasets/celeba_old'
young_path = '/home/wseto/datasets/celeba_young'


with open('/home/wseto/datasets/list_attr_celeba.txt', 'rb') as csvfile:
	csvreader = csv.reader(csvfile)

	filecount = csvreader.next()[0]
	print filecount
	attributes = csvreader.next()[0].split()
	print attributes

	for n in range(len(attributes)):
		if attributes[n] == 'Young':
			# add one since our row info will have the image file name as the first item
			young_loc = n+1
			break
	print young_loc

	print len(attributes)
	for row in csvreader:
		rowinfo = row[0].split()

		# original files were .jpg, alignment converted them to .png
		imagename = rowinfo[0].split('.')[0] + '.png'

		try:
			if rowinfo[young_loc] == '1':
				shutil.copyfile(os.path.join(data_dir, imagename), "%s/%s" % (young_path, imagename) ) 
			elif rowinfo[young_loc] == '-1':
				shutil.copyfile(os.path.join(data_dir, imagename), "%s/%s" % (old_path, imagename) ) 
		except:
			print "couldnt find", imagename
