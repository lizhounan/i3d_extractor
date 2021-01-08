from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

import json
import argparse
import time
from PIL import Image

#
import tensorflow as tf

import i3d
    
FLAGS = tf.flags.FLAGS
    
tf.flags.DEFINE_string('eval_type', 'joint', 'rgb, rgb600, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')

_IMAGE_SIZE = 224
_SAMPLE_VIDEO_FRAMES = 64

_CHECKPOINT_PATHS = {
'rgb': '/content/drive/.shortcut-targets-by-id/12kkMK3SqZj2FHaMnJsb5NSJtPLi_KcX_/kinetics-i3d/data/checkpoints/rgb_scratch/model.ckpt',
'rgb600': '/content/drive/.shortcut-targets-by-id/12kkMK3SqZj2FHaMnJsb5NSJtPLi_KcX_/kinetics-i3d/data/checkpoints/rgb_scratch_kin600/model.ckpt',
'flow': '/content/drive/.shortcut-targets-by-id/12kkMK3SqZj2FHaMnJsb5NSJtPLi_KcX_/kinetics-i3d/data/checkpoints/flow_scratch/model.ckpt',
'rgb_imagenet': '/content/drive/.shortcut-targets-by-id/12kkMK3SqZj2FHaMnJsb5NSJtPLi_KcX_/kinetics-i3d/data/checkpoints/rgb_imagenet/model.ckpt',
'flow_imagenet': '/content/drive/.shortcut-targets-by-id/12kkMK3SqZj2FHaMnJsb5NSJtPLi_KcX_/kinetics-i3d/data/checkpoints/flow_imagenet/model.ckpt',
}

tf.config.experimental.list_physical_devices('GPU') 

    
def feature_extractor():
    #tf.config.experimental.list_physical_devices('GPU') 
    tf.logging.set_verbosity(tf.logging.INFO)
    eval_type = FLAGS.eval_type

    imagenet_pretrained = FLAGS.imagenet_pretrained

    NUM_CLASSES = 400
    VIDEO_DIR = '/content/drive/.shortcut-targets-by-id/17pfO6qZSIWAqFBiMBFUU-mNwEYzJew31/GIPHY/GIPHY_final_code/'
    
    if eval_type not in ['rgb', 'rgb600', 'flow', 'joint']:
        raise ValueError('Bad `eval_type`, must be one of rgb, rgb600, flow, joint')


    if eval_type in ['rgb', 'rgb600', 'joint']:
        # RGB input has 3 channels.
        rgb_input = tf.placeholder(tf.float32,shape=(batch_size, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))


        with tf.variable_scope('RGB', reuse = tf.AUTO_REUSE):
          net = i3d.InceptionI3d(400, spatial_squeeze=True, final_endpoint='Logits')
          _,end_points = net(rgb_input, is_training=False, dropout_keep_prob=1.0)
          end_feature = end_points['avg_pool3d']
      
        rgb_variable_map = {}
        for variable in tf.global_variables():
    
          if variable.name.split('/')[0] == 'RGB':
            if eval_type == 'rgb600':
              rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
            else:
              rgb_variable_map[variable.name.replace(':0', '')] = variable
    
        rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True) #done
        
        features_rgb = []
 
  
  ## input to i3d
    with tf.compat.v1.Session() as sess:
        feed_dict = {}
    
    # RGB
        if eval_type in ['rgb', 'rgb600', 'joint']:
          if imagenet_pretrained:
            rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
            # jaywong's code
            video_list = open(VIDEO_PATH_FILE).readlines()
            video_list = [name.strip() for name in video_list]
            
            # print('video_list', video_list)
            
            if not os.path.isdir(OUTPUT_FEAT_DIR):
                os.mkdir(OUTPUT_FEAT_DIR)
        
            print('Total number of videos: %d'%len(video_list))
            
            for cnt, video_name in enumerate(video_list):
                video_path = os.path.join(VIDEO_DIR, video_name)
                feat_path = os.path.join(OUTPUT_FEAT_DIR, video_name + '.npy')
        
                if os.path.exists(feat_path):
                    print('Feature file for video %s already exists.'%video_name)
                    continue
        
                #print('video_path', video_path)
                
                n_frame = len([ff for ff in os.listdir(video_path) if ff.endswith('.jpg')])
                
                #print('Total frames: %d'%n_frame)
                
                features = []
        
                n_feat = int(n_frame // 8)
                n_batch = n_feat // batch_size + 1
                #print('n_frame: %d; n_feat: %d'%(n_frame, n_feat))
                #print('n_batch: %d'%n_batch)
        
                for i in range(n_batch):
                    input_blobs = []
                    for j in range(batch_size):
                        input_blob = []
                        for k in range(L):
                            idx = i*batch_size*L + j*L + k
                            idx = int(idx)
                            idx = idx%n_frame + 1
                            image = Image.open(os.path.join(video_path, '%d.jpg'%idx))
                            image = image.resize((resize_w, resize_h))
                            image = np.array(image, dtype='float32')
                            '''
                            image[:, :, 0] -= 104.
                            image[:, :, 1] -= 117.
                            image[:, :, 2] -= 123.
                            '''
                            image[:, :, :] -= 127.5
                            image[:, :, :] /= 127.5
                            input_blob.append(image)
                        
                        input_blob = np.array(input_blob, dtype='float32')
                        
                        input_blobs.append(input_blob)
        
                    input_blobs = np.array(input_blobs, dtype='float32')
                    
                    #print("........shape of input_blobs.....", input_blobs.shape) ## success
                    clip_feature = sess.run(end_feature,feed_dict = {rgb_input: input_blobs}) #done
                    clip_feature_resized = np.reshape(clip_feature, (-1,clip_feature.shape[-1])) # error
                    
                    features.append(clip_feature_resized)
                    
                features = np.concatenate(features, axis=0)
                print("final features shape",features.shape)
                features = features[:n_feat:2]   # 16 frames per feature  (since 64-frame snippet corresponds to 8 features in I3D)
        
                feat_path = os.path.join(OUTPUT_FEAT_DIR, video_name + '.npy')
        
                #print('Saving features for video: %s ...'%video_name)
                np.save(feat_path, features)
                
                print('--------------%d: %s has been processed...--------------'%(cnt, video_name))

  
#     #Flow  
#     if eval_type in ['flow', 'joint']:
#       if imagenet_pretrained:
#         flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
#     #   else:
#     #     flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])
      
#       tf.logging.info('Flow checkpoint restored')
#       flow_sample = np.load(_SAMPLE_PATHS['flow'])
#       tf.logging.info('Flow data loaded, shape=%s', str(flow_sample.shape))#our sample has been loaded, done
#       feed_dict[flow_input] = flow_sample
#       #print("!!!!!!!!!!!!flow feed_dict!!!!!!!!!!!")
#       #print(feed_dict) # this is our npy file . (1,40,224,224,3)
#       clip_feature = sess.run(flow_features['avg_pool3d'],feed_dict) #done
#       #print(clip_feature.shape) # (1,4,1,1,1024) 
#       #print("clip_feature.shape[-1]",clip_feature.shape[-1])
#       #print("!!!!!got the feature 1024 !!!!!!!")
#       clip_feature_resized = np.reshape(clip_feature, (-1,clip_feature.shape[-1])) # error
#       #print("clip_feature_resized",clip_feature_resized)
#       features_flow.append(clip_feature_resized)
#       #print(np.shape(features_flow))
    
      
#       ## save features
#       feat_path = os.path.join('/content/drive/.shortcut-targets-by-id/12kkMK3SqZj2FHaMnJsb5NSJtPLi_KcX_/kinetics-i3d/out/flow_features_jk/', x + '.npy')
#       print("save flow features")
      
#       np.save(feat_path,features_flow)
#       print("done flow saving")


#   ## flow done ## 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    print('******--------- Extract I3D features ------*******')
    parser.add_argument('-g', '--GPU', type=int, default=1, help='GPU id')
    parser.add_argument('-of', '--OUTPUT_FEAT_DIR', dest='OUTPUT_FEAT_DIR', type=str,
                        default='/content/drive/.shortcut-targets-by-id/17pfO6qZSIWAqFBiMBFUU-mNwEYzJew31/GIPHY/GIPHY_final_code/',
                        help='Output feature path')
    parser.add_argument('-vpf', '--VIDEO_PATH_FILE', type=str,
                        default='/content/drive/.shortcut-targets-by-id/17pfO6qZSIWAqFBiMBFUU-mNwEYzJew31/GIPHY/GIPHY_final_code/input.txt',
                        help='input video list')
    # parser.add_argument('-vd', '--VIDEO_DIR', type=str,
    #                     default='/content/drive/.shortcut-targets-by-id/17pfO6qZSIWAqFBiMBFUU-mNwEYzJew31/GIPHY/GIPHY_final_code'+ '/%s/' %video_name,
    #                     help='frame directory')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict

    OUTPUT_FEAT_DIR = params['OUTPUT_FEAT_DIR']
    VIDEO_PATH_FILE = params['VIDEO_PATH_FILE']
    # VIDEO_DIR = params['VIDEO_DIR']
    RUN_GPU = params['GPU']

    resize_w = 224
    resize_h = 224
    L = 64
    batch_size = 1

    # set gpu id
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(RUN_GPU)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    feature_extractor()





















