from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple

from module import *
from utils import *


class Aign(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size_A = args.load_size_A
        self.image_size_B = args.load_size_B
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc

        # ratio between rendered loss and gan loss
        # change the value dependent on the data
        self.L1_lambda = args.L1_lambda
        self.dataset_dir = args.dataset_dir

        self.discriminator = discriminator
        self.generator = generator_resnet

        # softmax cross entropy
        self.criterionGAN = sce_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size_A image_size_B \
                                output_c_dim is_training')
        self.options = OPTIONS._make((args.batch_size, args.load_size_A,args.load_size_B,
                                       args.output_nc,
                                      args.phase == 'train'))

        self._build_model()
        self.saver = tf.train.Saver()
        self.pool = ImagePool(args.max_size)



    def render(self,image):
        # renderer for image translation
        # change the function dependent upon the use
        rendered = tf.layers.average_pooling2d(image,6,4,padding='same')
        rendered = tf.image.rgb_to_grayscale(rendered)
        return rendered

    def _build_model(self):

        self.real_A = tf.placeholder(tf.float32,
                                        [None, self.image_size_A, self.image_size_A,
                                         self.input_c_dim ],
                                        name='real_A')

        self.real_B = tf.placeholder(tf.float32,
                                        [None, self.image_size_B, self.image_size_B, self.output_c_dim],
                                        name='real_A')

        # fake image
        self.fake_B = self.generator(self.real_A, self.options,True, False, name="generatorA2B")

        # rendered image
        self.rendered = self.render(self.fake_B) 
        # discriminator for b
        self.DB_fake = self.discriminator(self.fake_B, self.options,True, reuse=False, name="discriminatorB")
        self.realA_Gray = tf.image.rgb_to_grayscale(self.real_A)
        # loss with cross entropy and mean squared error
        self.g_loss_a2b = self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) + self.L1_lambda * mae_criterion(self.realA_Gray , self.rendered)

        # testing
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [None, 128, 128,
                                             self.output_c_dim], name='fake_B_sample')


        self.DB_real = self.discriminator(self.real_B, self.options,True, reuse=True, name="discriminatorB")

        self.DB_fake_sample = self.discriminator(self.fake_B_sample, self.options,True, reuse=True, name="discriminatorB")
        # real discrimiator loss
        self.db_loss_real = self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real))
        # fake discrim loss
        self.db_loss_fake = self.criterionGAN(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
        # total loss
        self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2

        self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)

        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)

        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)

        self.d_sum = tf.summary.merge(
            [
             self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum]
        )


        self.test_A = tf.placeholder(tf.float32,
                                     [None, self.image_size_A, self.image_size_A,
                                      self.input_c_dim], name='test_A')

        self.testB = self.generator(self.test_A, self.options,False, True, name="generatorA2B")
        self.renderTest = self.render(self.testB) 
        self.testRealA = tf.image.rgb_to_grayscale(self.test_A)
        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        print("discriminator variables ",len(self.d_vars), " generator variables ",len(self.g_vars))
        for var in t_vars: print(var.name)



    def train(self, args):

        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.db_loss, var_list=self.d_vars)

        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.g_loss_a2b, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if args.continue_train:
            if self.load(args.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load fail+ed...")
        print("training start")
        for epoch in range(args.epoch):
 
            dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
            dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB'))

            np.random.shuffle(dataA)
            np.random.shuffle(dataB)
            batch_idxs = min(min(len(dataA), len(dataB)), args.train_size) // self.batch_size
            lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)

            for idx in range(0, batch_idxs):
                batch_files_A = list(dataA[idx * self.batch_size:(idx + 1) * self.batch_size])
                batch_files_B = list(dataB[idx * self.batch_size:(idx + 1) * self.batch_size])

                batch_images_A = [load_train_data(batch_file, args.load_size_A) for batch_file in batch_files_A]

                batch_images_B = [load_train_data(batch_file, args.load_size_B) for batch_file in batch_files_B]

                batch_images_A = np.array(batch_images_A).astype(np.float32)

                batch_images_B = np.array(batch_images_B).astype(np.float32)

                fake_B, _, summary_str_G,gen_loss = self.sess.run(
                    [self.fake_B, self.g_optim, self.g_loss_a2b_sum, self.g_loss_a2b ],
                    feed_dict={self.real_A: batch_images_A,self.real_B: batch_images_B, self.lr: lr})

                self.writer.add_summary(summary_str_G, counter)

                [fake_B] = self.pool([fake_B])

                _, summary_str_D,dis_loss = self.sess.run(
                    [self.d_optim, self.d_sum,self.db_loss],
                    feed_dict={self.real_A: batch_images_A,self.real_B: batch_images_B,
                               self.fake_B_sample: fake_B,
                               self.lr: lr})

                self.writer.add_summary(summary_str_D, counter)

                counter += 1

                

                if np.mod(counter, args.print_freq) == 1:
                    self.sample_model(args.sample_dir, epoch, idx)
                    print(("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (epoch, idx, batch_idxs, time.time() - start_time)))
                    print("Gen Loss ",gen_loss," Dis Loss ",dis_loss)

                if np.mod(counter, args.save_freq) == 2:
                    self.save(args.checkpoint_dir, counter)



    def save(self, checkpoint_dir, step):
        model_name = "aign.model"
        model_dir = "%s_%s" % (self.dataset_dir, self.image_size_A)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)



    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_dir, self.image_size_A)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False



    def sample_model(self, sample_dir, epoch, idx):
        dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))

        np.random.shuffle(dataA)


        batch_files_A = list(dataA[:10])


        batch_images_A = [load_train_data(batch_file, self.image_size_A ) for batch_file in batch_files_A]
        batch_images_A_128 = [load_train_data(batch_file, 128) for batch_file in batch_files_A]

        sample_images_A = np.array(batch_images_A).astype(np.float32)

        sample_images_A_128 = np.array(batch_images_A_128).astype(np.float32)

        # [fake_B,realA_Gray,render] = self.sess.run([self.testB,self.renderTest ,self.testRealA], feed_dict = {self.test_A: sample_images_A})
        [fake_B] = self.sess.run([self.testB], feed_dict = {self.test_A: sample_images_A})

        merged = np.concatenate([fake_B,sample_images_A_128],axis =2)



        save_images(merged, [10, 1],
                    '{}/B_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))

        # merged2 = np.concatenate([realA_Gray,render],axis =2)

        # save_images(merged2, [10, 1],
        #             '{}/B_{:02d}_{:04d}new.jpg'.format(sample_dir, epoch, idx))



    def test(self, args):

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))


        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(args.test_dir, 'A2B_index.html')
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        out_var, in_var = (self.testB, self.test_A)

        for sample_file in sample_files:
            print('Processing image: ' + sample_file)
            sample_image = [load_test_data(sample_file, self.image_size_A )]
            sample_image = np.array(sample_image).astype(np.float32)

            image_path = os.path.join(args.test_dir,
                                      'A2B_{0}'.format(os.path.basename(sample_file)))
            fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image})
            save_images(fake_img, [1, 1], image_path)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s'></td>" % (sample_file if os.path.isabs(sample_file) else (
                '..' + os.path.sep + sample_file)))
            index.write("<td><img src='%s'></td>" % (image_path if os.path.isabs(image_path) else (
                '..' + os.path.sep + image_path)))
            index.write("</tr>")
        index.close()
