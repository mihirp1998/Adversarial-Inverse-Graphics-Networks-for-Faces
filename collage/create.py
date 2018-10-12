def merge(images, size):
	print('image shape',images.shape)
	h, w = images.shape[1], images.shape[2]
	img = np.zeros((h * size[0], w * size[1], 3))
	print('img shape ',img.shape)
	for idx, image in enumerate(images):
		i = idx % size[1]
		j = idx // size[1]
		img[j*h:j*h+h, i*w:i*w+w, :] = image

	return img

import cv2
import pickle
import numpy as np
images = []
a = pickle.load(open('human2avatar.p','rb'))    
for k,v in a.items():
	h = cv2.imread("alldata/"+v[0])
	a = cv2.imread("alldata/"+v[1])
	v = np.concatenate([h,a],axis=1)
	images.append(v)
	
print(np.array(images).shape[0])
img = merge(np.array(images[:295]),[25,12])	
# print(img)
cv2.imwrite("final.jpg",img) 
img = img/255.0
print(img.shape)
img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
cv2.imshow('images',img)
cv2.waitKey(0)
cv2.destroyAllWindows()