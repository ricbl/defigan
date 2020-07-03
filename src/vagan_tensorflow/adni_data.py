# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Modified by Ricardo Bigolin Lanfredi 
# Last modified: 2020-07-01

import numpy as np
from .. import adni_data_loader
from .batch_provider import BatchProvider

class adni_data():
    def __init__(self, exp_config):
        data = adni_data_loader.load_and_maybe_process_data(
            input_folder=exp_config.data_root,
            preprocessing_folder=exp_config.preproc_folder,
            size=exp_config.image_size,
            target_resolution=exp_config.target_resolution,
            label_list=exp_config.label_list,
            offset=exp_config.offset,
            force_overwrite=False,
            rescale_to_one=exp_config.rescale_to_one
        )
        
        self.data = data
        
        if hasattr(exp_config, 'label_name'):
            label_name = exp_config.label_name
        else:
            label_name = 'diagnosis'
        
        # the following are HDF5 datasets, not numpy arrays
        images_train = data['images_train']
        labels_train = data['%s_train' % label_name]
        
        images_val = data['images_val']
        labels_val = data['%s_val' % label_name]
        
        # Map labels in to a consecutive range, e.g. [0,2] to [0,1]
        
        if label_name == 'diagnosis':
            labels_train = np.asarray([np.argwhere(i==np.asarray(exp_config.label_list)) for i in labels_train]).flatten()
            labels_val = np.asarray([np.argwhere(i==np.asarray(exp_config.label_list)) for i in labels_val]).flatten()
        
        # Extract the number of training and testing points
        N_train = images_train.shape[0]
        N_val = images_val.shape[0]
        
        # Create a shuffled range of indices for both training and testing data
        train_indices = np.arange(N_train)
        
        train_AD_indices = train_indices[np.where(labels_train[:] == 1)]
        train_CN_indices = train_indices[np.where(labels_train[:] == 0)]
        
        val_indices = np.arange(N_val)
        val_AD_indices = val_indices[np.where(labels_val[:] == 1)]
        val_CN_indices = val_indices[np.where(labels_val[:] == 0)]
        
        # Create the batch providers
        self.trainAD = BatchProvider(images_train, labels_train, train_AD_indices)
        self.trainCN = BatchProvider(images_train, labels_train, train_CN_indices)
        
        self.validationAD = BatchProvider(images_val, labels_val, val_AD_indices)
        self.validationCN = BatchProvider(images_val, labels_val, val_CN_indices)
        
        self.train = BatchProvider(images_train, labels_train, train_indices)