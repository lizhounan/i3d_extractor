import numpy as np
import os
import json
import argparse
import time
from PIL import Image

#
import tensorflow as tf
import i3d
    
net = i3d.InceptionI3d(400, spatial_squeeze=True, final_endpoint='Logits')
# rgb_variable_map = {}
# for variable in tf.compat.v1.global_variables():
#     rgb_variable_map[variable.name.replace(':0', '')] = variable
# rgb_saver = tf.compat.v1.train.Saver(var_list=rgb_variable_map, reshape=True) #done
# with tf.compat.v1.Session() as sess:
#     rgb_saver.restore(sess, './rgb_i3d_checkpoint')

net.load_weights('./rgb_i3d_checkpoint')




















