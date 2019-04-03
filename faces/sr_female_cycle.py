# experiment: trying cyclegan for male -> female

from scipy.misc import imread, imshow, imresize, imsave
import os
from glob import glob
import numpy as np
# this for sklearn 0.17, for 0.18: use sklearn.model_selection
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
from PIL import Image
import pdb

def print_shape(t):
    print(t.name, t.get_shape().as_list())

def optimistic_restore(session, save_file, graph=tf.get_default_graph()):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])    
    restore_vars = []    
    for var_name, saved_var_name in var_names:            
        curr_var = graph.get_tensor_by_name(var_name)
        var_shape = curr_var.get_shape().as_list()
        if var_shape == saved_shapes[saved_var_name]:
            restore_vars.append(curr_var)
    opt_saver = tf.train.Saver(restore_vars)
    opt_saver.restore(session, save_file)

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
    
    filename = filename_queue.dequeue()
    image_bytes = tf.read_file(filename)
    decoded_image = tf.image.decode_png(image_bytes)
    #decoded_image.set_shape([128,128,3])
    normed_image = tf.to_float(decoded_image)/127.5 - 1

    image_queue = tf.FIFOQueue(batch_size, tf.float32, shapes=(128,128,3))
    enqueue_op = image_queue.enqueue(normed_image)

    # Create a queue runner that will enqueue decoded images into `image_queue`.                                                                                   
    NUM_THREADS = 8
    queue_runner = tf.train.QueueRunner(
        image_queue,
        [enqueue_op] * NUM_THREADS,  # Each element will be run from a separate thread.                                                                                       
        image_queue.close(),
        image_queue.close(cancel_pending_enqueues=True))

    # Ensure that the queue runner threads are started when we call                                                                                               
    # `tf.train.start_queue_runners()` below.                                                                                                                      
    tf.train.add_queue_runner(queue_runner)

    # this is the op that we pass to our model
    inputs = image_queue.dequeue_many(batch_size)
    print inputs
    
    return inputs


########
# CONFIG 
#########
adam_learning_rate = 0.0001
adam_beta1 = 0.9

batch_size = 32
image_h = 128
image_w = 128

num_epochs = 20

###############################
# BUILDING THE MODEL
###############################

female_data_dir = '/home/wseto/datasets/celeba_female'
male_data_dir = '/home/wseto/datasets/celeba_male'

female_data = glob(os.path.join(female_data_dir, "*.png"))
male_data = glob(os.path.join(male_data_dir, "*.png"))

data_disc, female_data_nondisc = train_test_split(female_data, test_size=0.5, random_state=42)

female_data_train, female_data_sample = train_test_split(female_data_nondisc, test_size=0.1, random_state=42)
male_data_train, male_data_sample = train_test_split(male_data, test_size=0.1, random_state=42)

data_train = female_data_train + male_data_train
data_sample = female_data_sample + male_data_sample


print "data train:", len(data_train)
print "data disc:", len(data_disc)
print "data sample:", len(data_sample)


real_ims = input_pipeline(data_disc, batch_size, num_epochs)
inputs = input_pipeline(data_train, batch_size, num_epochs)
sample_inputs = input_pipeline(data_sample, batch_size, num_epochs)



# generator section
print "GENERATOR"
print "-----------"

nb_residual = 5
def create_generator(inputs, b_training=True):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        padding='SAME',
                        activation_fn=None,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        with slim.arg_scope([slim.batch_norm],
                    is_training=b_training, scale=True, decay=0.9, epsilon=1e-5):
            net = inputs
            print_shape(net)

            # need to add some conv layers here since input is not downsampled anymore
            net = tf.nn.relu(slim.conv2d(net, 16, [3, 3], scope='pre_gconv1'))
            print_shape(net)

            net = tf.nn.relu(slim.conv2d(net, 32, [3, 3], stride=2, scope='pre_gconv2'))
            print_shape(net)

            net = tf.nn.relu(slim.conv2d(net, 64, [3, 3], stride=2, scope='gconv1'))
            print_shape(net)

            net1 = net

            res_inputs = net1
            for n in range(nb_residual):
                net = tf.nn.relu(slim.batch_norm(slim.conv2d(res_inputs, 64, [3, 3], scope='conv1_res'+str(n)), scope='bn_'+str(n*2)))
                net = slim.batch_norm(slim.conv2d(net, 64, [3, 3], scope='conv2_res'+str(n)), scope='bn_'+str(n*2+1))
                net = net + res_inputs
                res_inputs = net


            print_shape(net)

            net = slim.batch_norm(slim.conv2d(net, 64, [3, 3], scope='gconv2'), scope='bn_'+str(nb_residual*2)) + net1
            print_shape(net)

            # deconv
            net = tf.nn.relu(slim.conv2d_transpose(net, 256, [5, 5], stride=2, scope='deconv1'))
            print_shape(net)

            net = tf.nn.relu(slim.conv2d_transpose(net, 256, [5, 5], stride=2, scope='deconv2'))
            print_shape(net)

            # tanh since images have range [-1,1]
            net = slim.conv2d(net, 3, [3, 3], scope='gconv3', activation_fn=tf.nn.tanh)
            print_shape(net)

    return net

with tf.variable_scope("generator") as scope:
    gen = create_generator(inputs)
    scope.reuse_variables()
    gen_test = create_generator(sample_inputs, False)


# create cycle generator
with tf.variable_scope("cycle_generator") as scope:
    gen_cycle = create_generator(gen)
    scope.reuse_variables()
    gen_cycle_test = create_generator(gen_test, False)



print "DISCRIMINATOR"
print "--------------"

# anneal our noise stddev
decay_counter = tf.Variable(0, name="counter", dtype=tf.float32)

def create_discriminator(inputs, counter, b_reuse=False):
    with slim.arg_scope([slim.conv2d],
                    padding='SAME',
                    activation_fn=None,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                    weights_regularizer=slim.l2_regularizer(0.0005),
                    stride=2):

        # 0.05 starting stddev
        #noise_stddev = tf.constant(0.05, dtype=tf.float32) - tf.constant(5.5e-7, dtype=tf.float32)*counter
        #noisy_inputs = inputs + tf.random_normal(shape=tf.shape(inputs), mean=0.0, stddev=noise_stddev, dtype=tf.float32)
        noisy_inputs = inputs

        disc = lrelu(slim.conv2d(noisy_inputs, 64, [5, 5], scope='conv1'))
        print_shape(disc)

        with slim.arg_scope([slim.batch_norm], scale=True, decay=0.9, epsilon=1e-5):
            disc = lrelu(slim.batch_norm(slim.conv2d(disc, 128, [5, 5], scope='conv2'), scope='d_bn1'))
            print_shape(disc)

            disc = lrelu(slim.batch_norm(slim.conv2d(disc, 256, [5, 5], scope='conv3'), scope='d_bn2'))
            print_shape(disc)

            disc = lrelu(slim.batch_norm(slim.conv2d(disc, 512, [5, 5], scope='conv4'), scope='d_bn3'))
            print_shape(disc)

            disc = lrelu(slim.batch_norm(slim.conv2d(disc, 512, [5, 5], scope='conv5'), scope='d_bn4'))
            print_shape(disc)

    disc = lrelu(slim.fully_connected(tf.contrib.layers.flatten(disc), 1024, activation_fn=None, scope='fc6'))
    disc_logits = slim.fully_connected(tf.contrib.layers.flatten(disc), 1, activation_fn=None, scope='fc7')

    return disc_logits
# create 2 discriminators: for fake and real images
with tf.variable_scope("discriminators") as scope:
    disc_real = create_discriminator(real_ims, decay_counter)
    scope.reuse_variables()
    disc_fake = create_discriminator(gen, decay_counter, True)


# Losses
###########

# loss on real input images; all outputs should be 1

# make labels noisy for discriminator
rand_val = tf.random_uniform([], seed=42)
labels_real = tf.cond(rand_val < 0.95, lambda: tf.ones_like(disc_real), lambda: tf.zeros_like(disc_real))
labels_fake = tf.cond(rand_val < 0.95, lambda: tf.zeros_like(disc_fake), lambda: tf.ones_like(disc_fake))

d_loss_real = tf.reduce_mean( \
    tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=labels_real))

# loss on fake input images, fakes should be 0
d_loss_fake = tf.reduce_mean( \
    tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=labels_fake))

# similar to above, but we want fake (generator) images to output 1
g_loss_adv = tf.reduce_mean( \
    tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones_like(disc_fake)))


# reconstruction loss is on input and cycle output
gen_mse_HR = tf.reduce_mean(tf.square(tf.contrib.layers.flatten(gen_cycle - inputs)), 1)
# average for the batch
gen_L2_HR = tf.reduce_mean(gen_mse_HR)


train_vars = tf.trainable_variables()
d_vars = [var for var in train_vars if 'discriminators' in var.name]
g_vars = [var for var in train_vars if 'generator' in var.name]


# optimize the generator and discriminator separately
g_loss = gen_L2_HR + 0.01*g_loss_adv
d_loss = d_loss_real + d_loss_fake

optim = tf.train.AdamOptimizer(adam_learning_rate, beta1=adam_beta1)
d_train_op = slim.learning.create_train_op(d_loss, optim, variables_to_train=d_vars)
g_train_op = slim.learning.create_train_op(g_loss, optim, variables_to_train=g_vars)

weight_saver = tf.train.Saver(max_to_keep=1)


print "initialization done"


# create directories to save checkpoint and samples
samples_dir = 'samples_female_cycle'
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

checkpoint_dir = 'checkpoint_female_cycle'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


print "TRAINING"
print "-----------"

start_time = time.time()
counter = 0

# Create the graph, etc.
init_op = [tf.local_variables_initializer(),
           tf.global_variables_initializer()]

# Create a session for running operations in the Graph.
config = tf.ConfigProto()
#config.operation_timeout_in_ms=20000  # for debugging queue hangs
sess = tf.Session(config=config)

# Initialize the variables (like the epoch counter).
sess.run(init_op)

# Start input enqueue threads.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

print "queue runners done"
num_batches = len(data_train) // batch_size

b_load = True
ckpt_dir = '/home/wseto/dcgan/checkpoint_female_cycle'
if b_load:
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    weight_saver.restore(sess, ckpt.model_checkpoint_path)
    counter = int(ckpt.model_checkpoint_path.split('-', 1)[1]) 
    print "successfully restored!" + " counter:", counter

try:
    while not coord.should_stop():

        fetches = [d_loss_fake, d_loss_real, g_loss_adv, d_train_op, g_train_op]
        errD_fake, errD_real, errG, _, _ = sess.run(fetches)

        total_errD = errD_fake + errD_real
        counter += 1

        idx = np.mod(counter, num_batches)
        epoch = counter / num_batches
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss_fake: %.3f, d_loss_real: %.3f, g_loss: %.8f" \
            % (epoch, idx, num_batches,
                time.time() - start_time, errD_fake, errD_real, errG))

        if np.mod(counter, 30) == 1:

            sample, sample_origs, sample_cycle = sess.run([gen_test, sample_inputs, gen_cycle_test])
            
            # save an image, with the original next to the generated one
            merge_im = np.zeros( (image_h, image_h*3, 3) )
            merge_im[:, :image_h, :] = (sample_origs[0]+1)*127.5
            merge_im[:, image_h:image_h*2, :] = (sample[0]+1)*127.5
            merge_im[:, image_h*2:, :] = (sample_cycle[0]+1)*127.5

            imsave(samples_dir + '/test_{:02d}_{:04d}.png'.format(epoch, idx), merge_im)
            print "saving a sample"

        if np.mod(counter, 1000) == 2:
            weight_saver.save(sess, checkpoint_dir + '/model', counter)
            print "saving a checkpoint"


except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    # When done, ask the threads to stop.
    coord.request_stop()

# Wait for threads to finish.
coord.join(threads)
sess.close()
