from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from utils import *
from LIP_model import *
import matplotlib.pyplot as plt
import scipy.misc
import scipy.io as sio


NUM_STEPS = 6 # Number of images in the validation set.
INPUT_SIZE = (384, 384)
N_CLASSES = 20
DATA_DIRECTORY = './datasets/examples'
DATA_LIST_PATH = './datasets/examples/list/val.txt'
RESTORE_FROM = './checkpoint/JPPNet-s2'
OUTPUT_DIR = './output/pose/val'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def main():
    """Create the model and start the evaluation process."""
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    h, w = INPUT_SIZE
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(DATA_DIRECTORY, DATA_LIST_PATH, None, False, False, coord)
        image = reader.image
        image_rev = tf.reverse(image, tf.stack([1]))
        image_list = reader.image_list
    
    image_batch_origin = tf.stack([image, image_rev])
    image_batch = tf.image.resize_images(image_batch_origin, [int(h), int(w)])
    image_batch125 = tf.image.resize_images(image_batch_origin, [int(h * 1.25), int(w * 1.25)])
    image_batch075 = tf.image.resize_images(image_batch_origin, [int(h * 0.75), int(w * 0.75)])

    # Create network.
    with tf.variable_scope('', reuse=False):
        net_100 = JPPNetModel({'data': image_batch}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_125 = JPPNetModel({'data': image_batch125}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_075 = JPPNetModel({'data': image_batch075}, is_training=False, n_classes=N_CLASSES)


    # parsing net
    parsing_fea1_100 = net_100.layers['res5d_branch2b_parsing']
    parsing_fea1_125 = net_125.layers['res5d_branch2b_parsing']
    parsing_fea1_075 = net_075.layers['res5d_branch2b_parsing']

    parsing_out1_100 = net_100.layers['fc1_human']
    parsing_out1_125 = net_125.layers['fc1_human']
    parsing_out1_075 = net_075.layers['fc1_human']

    # pose net
    resnet_fea_100 = net_100.layers['res4b22_relu']
    resnet_fea_125 = net_125.layers['res4b22_relu']
    resnet_fea_075 = net_075.layers['res4b22_relu']
    
    with tf.variable_scope('', reuse=False):
        pose_out1_100, pose_fea1_100 = pose_net(resnet_fea_100, 'fc1_pose')
        pose_out2_100, pose_fea2_100 = pose_refine(pose_out1_100, parsing_out1_100, pose_fea1_100, name='fc2_pose')
        parsing_out2_100, parsing_fea2_100 = parsing_refine(parsing_out1_100, pose_out1_100, parsing_fea1_100, name='fc2_parsing')
        pose_out3_100, pose_fea3_100 = pose_refine(pose_out2_100, parsing_out2_100, pose_fea2_100, name='fc3_pose')

    with tf.variable_scope('', reuse=True):
        pose_out1_125, pose_fea1_125 = pose_net(resnet_fea_125, 'fc1_pose')
        pose_out2_125, pose_fea2_125 = pose_refine(pose_out1_125, parsing_out1_125, pose_fea1_125, name='fc2_pose')
        parsing_out2_125, parsing_fea2_125 = parsing_refine(parsing_out1_125, pose_out1_125, parsing_fea1_125, name='fc2_parsing')
        pose_out3_125, pose_fea3_125 = pose_refine(pose_out2_125, parsing_out2_125, pose_fea2_125, name='fc3_pose')

    with tf.variable_scope('', reuse=True):
        pose_out1_075, pose_fea1_075 = pose_net(resnet_fea_075, 'fc1_pose')
        pose_out2_075, pose_fea2_075 = pose_refine(pose_out1_075, parsing_out1_075, pose_fea1_075, name='fc2_pose')
        parsing_out2_075, parsing_fea2_075 = parsing_refine(parsing_out1_075, pose_out1_075, parsing_fea1_075, name='fc2_parsing')
        pose_out3_075, pose_fea3_075 = pose_refine(pose_out2_075, parsing_out2_075, pose_fea2_075, name='fc3_pose')


    pose_out3 = tf.reduce_mean(tf.stack([tf.image.resize_nearest_neighbor(pose_out3_100, tf.shape(image_batch_origin)[1:3,]),
                                         tf.image.resize_nearest_neighbor(pose_out3_125, tf.shape(image_batch_origin)[1:3,]),
                                         tf.image.resize_nearest_neighbor(pose_out3_075, tf.shape(image_batch_origin)[1:3,])]), axis=0)

    head_output, tail_output = tf.unstack(pose_out3, num=2, axis=0)
    tail_list = tf.unstack(tail_output, num=16, axis=2)
    tail_list_rev = [None] * 16
    tail_list_rev[0] = tail_list[5]
    tail_list_rev[1] = tail_list[4]
    tail_list_rev[2] = tail_list[3]
    tail_list_rev[3] = tail_list[2]
    tail_list_rev[4] = tail_list[1]
    tail_list_rev[5] = tail_list[0]
    tail_list_rev[10] = tail_list[15]
    tail_list_rev[11] = tail_list[14]
    tail_list_rev[12] = tail_list[13]
    tail_list_rev[13] = tail_list[12]
    tail_list_rev[14] = tail_list[11]
    tail_list_rev[15] = tail_list[10]
    tail_list_rev[6] = tail_list[6]
    tail_list_rev[7] = tail_list[7]
    tail_list_rev[8] = tail_list[8]
    tail_list_rev[9] = tail_list[9]
    tail_output_rev = tf.stack(tail_list_rev, axis=2)
    tail_output_rev = tf.reverse(tail_output_rev, tf.stack([1]))

    output_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    output_all = tf.expand_dims(output_all, dim=0)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    if RESTORE_FROM is not None:
        if load(loader, sess, RESTORE_FROM):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    

    # Iterate over training steps.
    for step in range(NUM_STEPS):
        predict_ = sess.run(output_all)
        save_lip_images(image_list[step], predict_, OUTPUT_DIR)
        if step % 100 == 0:
            print('step {:d}'.format(step))
            print (image_list[step])

    coord.request_stop()
    coord.join(threads)
   

def save_lip_images(image_path, samples, out_dir):
    img_A = scipy.misc.imread(image_path).astype(np.float)
    rows = img_A.shape[0]
    cols = img_A.shape[1]
    image = samples[0]
    img_split = image_path.split('/')
    img_id = img_split[-1][:-4]
    with open('{}/{}.txt'.format(out_dir, img_id), 'w') as f:
        for p in xrange(image.shape[2]):
            channel_ = image[:,:,p]
            if channel_.shape[0] != rows or channel_.shape[1] != cols:
                print ('sizes do not match...')
                channel_ = scipy.misc.imresize(channel_, [rows, cols], interp='nearest')
            r_, c_ = np.unravel_index(channel_.argmax(), channel_.shape)
            f.write('%d %d ' % (int(c_), int(r_)))


if __name__ == '__main__':
    main()


