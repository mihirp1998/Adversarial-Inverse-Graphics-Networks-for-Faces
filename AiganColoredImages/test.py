from glob import glob
import numpy as np
from utils import *

dataA = glob('./datasets/{}/*.*'.format("ppl2avatar" + '/trainA'))
dataB = glob('./datasets/{}/*.*'.format("ppl2avatar" + '/trainB'))
np.random.shuffle(dataA)
np.random.shuffle(dataB)
# print(len(dataA))

batch_files_A = list(dataA[:2])
batch_files_B = list(dataB[:2])
# print(batch_files_A)
batch_images_A = [load_train_data(batch_file, 32) for batch_file in batch_files_A]

batch_images_B = [load_train_data(batch_file,128) for batch_file in batch_files_B]

sample_images_A = np.array(batch_images_A).astype(np.float32)

sample_images_B = np.array(batch_images_B).astype(np.float32)

print(batch_files_B)
real_B = sample_images_B
import cv2
cv2.imshow('image',(real_B[0]+1)/2)
cv2.waitKey(0)

save_images(real_B, [2, 1],
            '{}/B_Real_Check_{:04d}.jpg'.format("sample", 9))

