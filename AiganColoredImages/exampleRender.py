import tensorflow as tf
import cv2
import numpy as np
img = cv2.imread("horse.jpg")
img = np.expand_dims(img,0)
# n_img = np.reshape(img,(1) +img.shape)
a = tf.placeholder(tf.float32,shape  = (None,256,256,3))
avg = tf.layers.max_pooling2d(a,4,4)
sess = tf.Session()
avg = sess.run(avg,feed_dict={a:img})
print(img,avg.shape)

cv2.imshow('image',avg[0].astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()