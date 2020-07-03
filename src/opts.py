"""User configuration file

File organizing all configurations that may be set by user when running the 
train.py script. 
Call python -m src.train --help for a complete and formatted list of available user options.

by Ricardo Bigolin Lanfredi
Last modified: 2020-07-01
Project: DeFI-GAN
GNU General Public License v3.0
"""

import os
import argparse
import time
from random import randint
import socket

#convert a few possibilities of ways of inputing boolean values to a python boolean
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_opt(args_from_code = None):
    parser = argparse.ArgumentParser(description='Configuration for running VRGAN code')
    
    parser.add_argument('--skip_train', type=str2bool, nargs='?', default='false',
        help='If you just want to run validation, set this value to true.')
    parser.add_argument('--skip_val', type=str2bool, nargs='?', default='false',
        help='If you just want to run training, set this value to true.')
    parser.add_argument('--lambda_regg', type=float, nargs='?', default=50.0,
        help='Multiplier for the generator regularization loss L_{RegG}. Appears on Eq. 6 of the paper.')
    parser.add_argument('--lambda_regd', type=float, nargs='?', default=10,
        help='Multiplier for the gradient penalty on the critic, L_{RegD}.')  
    parser.add_argument('--batch_size', type=int, nargs='?', default=8,
        help='Batch size for training and validation')
    parser.add_argument('--folder_dataset', type=str, nargs='?', default='./',
        help='If you want to load/save toy dataset files in a folder other than the local folder, change this variable.')
    parser.add_argument('--save_folder', type=str, nargs='?', default='./runs',
        help='If you want to save files and outputs in a folder other than \'./runs\', change this variable.')
    parser.add_argument('--learning_rate_g', type=float, nargs='?', default=1e-4,
        help='Learning rate for the optimizer used for updating the weigths of the generator')
    parser.add_argument('--learning_rate_d', type=float, nargs='?', default=1e-4,
        help='Learning rate for the optimizer used for updating the weigths of the critic')
    parser.add_argument('--dataset_to_use', type=str, nargs='?', default='toy',
        help='Select the dataset to load. Options are "toy", "xray", "adni".' )
    parser.add_argument('--gpus', type=str, nargs='?', default=None,
        help='Set the gpus to use, using CUDA_VISIBLE_DEVICES syntax.')
    parser.add_argument('--experiment', type=str, nargs='?', default='',
        help='Set the name of the folder where to save the run.')
    parser.add_argument('--nepochs', type=int, nargs='?', default=30,
        help='Number of epochs to run training and validation')
    parser.add_argument('--split_validation', type=str, nargs='?', default='val',
        help='Use \'val\' to use the validation set for calculating scores every epoch. Use \'test\' for using the test set')
    parser.add_argument('--load_checkpoint_g', type=str, nargs='?', default=None,
        help='Set a filepath locating a model checkpoint for the generator that you want to load')
    parser.add_argument('--load_checkpoint_d', type=str, nargs='?', default=None,
        help='Set a filepath locating a model checkpoint for the critic that you want to load')
    parser.add_argument('--total_iters_per_epoch', type=int, nargs='?', default=200, 
        help='Set the number of batches that are loaded in each epoch.')
    parser.add_argument('--save_model', type=str2bool, nargs='?', default='false', 
        help='If true, will save model every epoch')
    parser.add_argument('--save_best_model', type=str2bool, nargs='?', default='true',
        help="If true, will save the model with the best ncc score model with the checkpoint name \'<generator/critic>_state_dict_best_epoch\'.")
    parser.add_argument('--generator_output', type=str, nargs='?', default='flow',
        help='Accepts two values: \'flow\' for training DeFI-GAN, and \'residual\' for using the baseline.')
    parser.add_argument('--load_to_memory', type=str2bool, nargs='?', default='true',
        help='If this is true, the dataset will be fully loaded to memory. Otherwise, the images are loaded from disk every iteration. If you have the RAM capacity needed, leave this as true to have a faster training.')   
    parser.add_argument('--use_old_schedule', type=str2bool, nargs='?', default='false',
        help='If true, uses the schedule where critic is updated 99 times for every generator update for a few epochs. Then, for the following epochs, the critic is updated 5 times for every generator update for a few epochs. The value true is used for training the baseline. If false, uses a training schedule where both critic and generator are updated at the same time, in every iteration.')
    parser.add_argument('--length_initialization_old_schedule', type=int, nargs='?', default=12,
        help="Defines for how many epochs the critic is updated 99 times for every generator update, when using use_old_schedule.")
    parser.add_argument('--n_dimensions_data', type=int, nargs='?', default=2,
        help="Use 3 (volume) for the ADNI dataset and 2 (image) for the other datasets.")
    parser.add_argument('--do_visual_validation', type=str2bool, nargs='?', default='false',
        help="If true, only generates the output images for one specific case.")
    parser.add_argument('--index_vis', type=int, nargs='?', default=74,
        help="Defines the index on the dataset of the cases used for generating images for the paper. Only used if do_visual_validation is true.")
    parser.add_argument('--ADNI_images_location', type=str, nargs='?', default='~/ADNI/ADNI_all_no_PP_3/',
        help="Defines the location of the preprocessed ADNI volumes")
    parser.add_argument('--COPD_labels_location', type=str, nargs='?', default='./Chest_Xray_Main_TJ_clean_ResearchID_PFTValuesAndInfo_WithDate_NoPHI.csv',
        help="Defines the location of file containing the metadata for the COPD dataset.")
    parser.add_argument('--COPD_lists_location', type=str, nargs='?', default='./',
        help="Defines folder where the files images2012-2016.txt, images2017.txt,  all_subjects_more_than_one_image.pkl and valids_all_subjects_more_than_one_image.pkl, containing a list of the location of every image in the dataset and a list of the subjects used for validation and for test.")
    parser.add_argument('--scale_flow_arrow', type=float, nargs='?', default=1,
        help="If 1, the arrows in the quiver plot are in units of pixels. Values greater than 1 can be used to multiple the arrow lengths and facilitate viewing their orientation.")
    parser.add_argument('--gap_between_arrows', type=int, nargs='?', default=4,
            help="Modifies the space between arrows in the quiver plot, in units of pixels. 4 seems to be a good value to not get the quiver plot too populated. Use only integer values.")
    parser.add_argument('--constant_to_multiply_flow', type=float, nargs='?', default=1,
        help='Important to have this in the order of expected deformations, in pixels. This helps in faster convergence for datasets where deformations are large. For both ADNI and COPD datasets, there was no need to tweak this constant.')
    if args_from_code is not None:
        args = parser.parse_args(args_from_code)
    else:
        args = parser.parse_args()
    
    #gets the current time of the run, and adds a four digit number for getting
    #different folder name for experiments run at the exact same time.
    timestamp = time.strftime("%Y%m%d-%H%M%S") + '-' + str(randint(1000,9999))
    args.timestamp = timestamp
    args.n_classes = 1
    args.n_channels = 1
    args.n_classes = args.n_classes + (args.n_dimensions_data if (args.generator_output=='flow') else 0)
    
    #set gpu where to run calculations. If args.gpus is not set, expects the 
    # environment variable CUDA_VISIBLE_DEVICES to be set
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    else:
        args.gpus = os.environ['CUDA_VISIBLE_DEVICES']
        
    #register a few values that might be important for reproducibility
    args.screen_name = os.getenv('STY')
    args.hostname = socket.gethostname()
    import platform
    args.python_version = platform.python_version()
    try:
        import SimpleITK as sitk
        args.sitk_version = sitk.Version.VersionString()
    except:
        pass
    import torch
    args.pytorch_version = torch.__version__ 
    import torchvision
    args.torchvision_version = torchvision.__version__
    import numpy as np
    args.numpy_version = np.__version__
    import pandas as pd
    args.pandas_version = pd.__version__
    return args
