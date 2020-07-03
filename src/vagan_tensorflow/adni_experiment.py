# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Modified by Ricardo Bigolin Lanfredi 
# Last modified: 2020-07-02

import os
import argparse
import time
from random import randint

parser = argparse.ArgumentParser(description='Configuration for running VAGAN code')

parser.add_argument('--experiment_name', type=str, nargs='?', default='',
    help='Set the name of the folder where to save the run.')
parser.add_argument('--gpus', type=str, nargs='?', default=None,
    help='Set the gpus to use, using CUDA_VISIBLE_DEVICES syntax.')
parser.add_argument('--data_root', type=str, nargs='?', default='~/ADNI/ADNI_all_no_PP_3/',
    help="Defines the location of the preprocessed ADNI volumes")
parser.add_argument('--preproc_folder', type=str, nargs='?', default='./',
    help='If you want to load/save toy dataset files in a folder other than the local folder, change this variable.')
parser.add_argument('--project_root', type=str, nargs='?', default='./logdir_tensorflow',
    help='If you want to save files and outputs in a folder other than \'./logdir_tensorflow\', change this variable.')

args, rest = parser.parse_known_args()
print(rest)
timestamp = time.strftime("%Y%m%d-%H%M%S") + '-' + str(randint(1000,9999))
data_root = args.data_root
preproc_folder = args.preproc_folder
project_root = args.project_root
experiment_name = args.experiment_name +'_'+ timestamp
if args.gpus is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
else:
    args.gpus = os.environ['CUDA_VISIBLE_DEVICES']

import tensorflow as tf
from . import mask_generators, critics
# Model settings
critic_net = critics.C3D_fcn_16
generator_net = mask_generators.unet_16_bn

# Data settings
data_identifier = 'adni'
image_size = (128, 160, 112)
target_resolution =  (1.3, 1.3, 1.3)
offset = None
label_list = (1,2)  # 0 - normal, 1 - mci, 2 - alzheimer's
label_name = 'diagnosis'

rescale_to_one = True
image_z_slice = 56  # for displaying images during training

# Optimizer Settings
optimizer_handle = tf.train.AdamOptimizer
beta1 = 0.0
beta2 = 0.9

# Training settings
batch_size = 2
n_accum_grads = 6
learning_rate = 1e-3
divide_lr_frequency = None
critic_iter = 5
critic_iter_long = 100
critic_retune_frequency = 100
critic_initial_train_duration = 25

# Cost function settings
l1_map_weight = 100.0
use_tanh = True

# Improved training settings
improved_training = True
scale=10.0

# Normal WGAN training settings (only used if improved_training=False)
clip_min = -0.01
clip_max = 0.01

# Rarely changed settings
max_iterations = 1500
save_frequency = 10
validation_frequency = 10
num_val_batches = 20
update_tensorboard_frequency = 2
