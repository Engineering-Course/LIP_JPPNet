import tensorflow as tf
from utils.ops import *


#------------------------network setting---------------------
#################################################

##  refine net version 4.   07.17

def pose_net(image, name):
  with tf.variable_scope(name) as scope:
      is_BN = False
      pose_conv1 = conv2d(image, 512, 3, 1, relu=True, bn=is_BN, name='pose_conv1')
      pose_conv2 = conv2d(pose_conv1, 512, 3, 1, relu=True, bn=is_BN, name='pose_conv2')
      pose_conv3 = conv2d(pose_conv2, 256, 3, 1, relu=True, bn=is_BN, name='pose_conv3')
      pose_conv4 = conv2d(pose_conv3, 256, 3, 1, relu=True, bn=is_BN, name='pose_conv4')
      pose_conv5 = conv2d(pose_conv4, 256, 3, 1, relu=True, bn=is_BN, name='pose_conv5')
      pose_conv6 = conv2d(pose_conv5, 256, 3, 1, relu=True, bn=is_BN, name='pose_conv6')

      pose_conv7 = conv2d(pose_conv6, 512, 1, 1, relu=True, bn=is_BN, name='pose_conv7')
      pose_conv8 = conv2d(pose_conv7, 16, 1, 1, relu=False, bn=is_BN, name='pose_conv8')

      return pose_conv8, pose_conv6


def pose_refine(pose, parsing, pose_fea, name):
  with tf.variable_scope(name) as scope:
      is_BN = False
      # 1*1 convolution remaps the heatmaps to match the number of channels of the intermediate features.
      pose = conv2d(pose, 128, 1, 1, relu=True, bn=is_BN, name='pose_remap')
      parsing = conv2d(parsing, 128, 1, 1, relu=True, bn=is_BN, name='parsing_remap')
      # concat 
      pos_par = tf.concat([pose, parsing, pose_fea], 3)
      conv1 = conv2d(pos_par, 512, 3, 1, relu=True, bn=is_BN, name='conv1')
      conv2 = conv2d(conv1, 256, 5, 1, relu=True, bn=is_BN, name='conv2')
      conv3 = conv2d(conv2, 256, 7, 1, relu=True, bn=is_BN, name='conv3')
      conv4 = conv2d(conv3, 256, 9, 1, relu=True, bn=is_BN, name='conv4')

      conv5 = conv2d(conv4, 256, 1, 1, relu=True, bn=is_BN, name='conv5')
      conv6 = conv2d(conv5, 16, 1, 1, relu=False, bn=is_BN, name='conv6')
      
      return conv6, conv4


def parsing_refine(parsing, pose, parsing_fea, name):
  with tf.variable_scope(name) as scope:
      is_BN = False
      pose = conv2d(pose, 128, 1, 1, relu=True, bn=is_BN, name='pose_remap')
      parsing = conv2d(parsing, 128, 1, 1, relu=True, bn=is_BN, name='parsing_remap')

      par_pos = tf.concat([parsing, pose, parsing_fea], 3)
      parsing_conv1 = conv2d(par_pos, 512, 3, 1, relu=True, bn=is_BN, name='parsing_conv1')
      parsing_conv2 = conv2d(parsing_conv1, 256, 5, 1, relu=True, bn=is_BN, name='parsing_conv2')
      parsing_conv3 = conv2d(parsing_conv2, 256, 7, 1, relu=True, bn=is_BN, name='parsing_conv3')
      parsing_conv4 = conv2d(parsing_conv3, 256, 9, 1, relu=True, bn=is_BN, name='parsing_conv4')

      parsing_conv5 = conv2d(parsing_conv4, 256, 1, 1, relu=True, bn=is_BN, name='parsing_conv5')
      parsing_human1 = atrous_conv2d(parsing_conv5, 20, 3, rate=6, relu=False, name='parsing_human1')
      parsing_human2 = atrous_conv2d(parsing_conv5, 20, 3, rate=12, relu=False, name='parsing_human2')
      parsing_human3 = atrous_conv2d(parsing_conv5, 20, 3, rate=18, relu=False, name='parsing_human3')
      parsing_human4 = atrous_conv2d(parsing_conv5, 20, 3, rate=24, relu=False, name='parsing_human4')
      parsing_human = tf.add_n([parsing_human1, parsing_human2, parsing_human3, parsing_human4], name='parsing_human')
      
      return parsing_human, parsing_conv4
#################################################

