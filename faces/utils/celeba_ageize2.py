
import shutil
import csv
import os
from glob import glob

# Smiling
# Blond_Hair
data_dir = '/home_01/f20150198/datasets/celebA/img_align_celeba_png'
old_path = '/home_01/f20150198/datasets/celebA/celeba_nonblond'
young_path = '/home_01/f20150198/datasets/celebA/celeba_blackhair'


# 5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose
#  Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones
# Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose
#  Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings
#   Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young

with open('/home_01/f20150198/datasets/celebA/list_attr_celeba.txt', 'rb') as csvfile:
	csvreader = csv.reader(csvfile)

	filecount = csvreader.next()[0]
	print filecount
	attributes = csvreader.next()[0].split()
	print attributes

	for n in range(len(attributes)):
		if attributes[n] == 'Smiling':
			# add one since our row info will have the image file name as the first item
			young_loc = n+1
			break
		# if attributes[n] == 'Blond_Hair':
		# 	# add one since our row info will have the image file name as the first item
		# 	old_loc = n+1
		# 	break
				
	print young_loc

	print len(attributes)
	for row in csvreader:
		rowinfo = row[0].split()

		# original files were .jpg, alignment converted them to .png
		imagename = rowinfo[0].split('.')[0] + '.png'

		try:
			if rowinfo[young_loc] == '1':
				shutil.copyfile(os.path.join(data_dir, imagename), "%s/%s" % (young_path, imagename) ) 
			# if rowinfo[old_loc] == '1':
			# 	shutil.copyfile(os.path.join(data_dir, imagename), "%s/%s" % (old_path, imagename) ) 
		except:
			print "couldnt find", imagename
