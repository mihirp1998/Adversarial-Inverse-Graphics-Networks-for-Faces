from glob import glob
import numpy as np
from utils import *

dataA = glob('./datasets/{}/*.*'.format("ppl2avatar" + '/trainA'))
dataB = glob('./datasets/{}/*.*'.format("ppl2avatar" + '/trainB'))
np.random.shuffle(dataA)
np.random.shuffle(dataB)
# print(len(dataA))
# idx = idx%min(len(dataA),len(dataB))
# print(idx)
batch_files_A = list(dataA[:1])
batch_files_B = list(dataB[:1])
# print(batch_files_A)
batch_images_A = [load_train_data(batch_file, 32) for batch_file in batch_files_A]

batch_images_B = [load_train_data(batch_file,128) for batch_file in batch_files_B]

sample_images_A = np.array(batch_images_A).astype(np.float32)

sample_images_B = np.array(batch_images_B).astype(np.float32)

print(batch_files_B)
real_B = sample_images_B
import cv2
cv2.imshow('image',real_B[0])
cv2.waitKey(0)
# print("sampleing ",sample_images_A.shape)
# [fake_B,real_B] = self.sess.run([self.fake_B,self.real_B], feed_dict = {self.real_A: sample_images_A,self.real_B: sample_images_B})


# save_images(fake_A, [self.batch_size, 1],
#             './{}/A_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
# save_images(fake_B, [self.batch_size, 1],
#             '{}/B_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
save_images(real_B, [1, 1],
            '{}/B_Real_Check_{:04d}.jpg'.format("sample", 9))

