"""Synthetic dataset generation and dataloader

To use this script, call the init_synth_dataloader_original to get a dataloader 
for the synthetic dataset. The dataset will be generated in a h5 file the first 
time you call this function. This dataset was not used for the paper, 
but provides an example of results for the method, without having to setup data.

by Ricardo Bigolin Lanfredi
Last modified: 2020-07-01
Project: DeFI-GAN
GNU General Public License v3.0
"""

import numpy as np
from torch.utils.data import Dataset
from skimage import filters
import h5py
import os
import torch
import torchvision.transforms as transforms
import sys
from skimage.transform import resize
from . import image_preprocessing

#only allow one loading of the module
sys.modules['synth_dataset']=None

from .utils_dataset import LoadToMemory
from .utils_dataset import DatasetsDifferentSizes, return_dataloaders, RegressionToClassification, TransformsDataset
NUM_WORKERS = 0

class SynthDataset(Dataset):

    def __init__(self, output_folder, mode='train', transform=None, anomaly = False, filter_disease = False):
        super().__init__()
        self.filter_disease = filter_disease
        self.output_folder = output_folder
        self.anomaly = anomaly
        self.mode = mode
        self.transform = transform
        self.load_cache()
        self.indices = np.arange(len(self.images))
        
    def load_cache(self):
        data = load_and_generate_data(output_folder = self.output_folder, mode = self.mode)
        imsize = 224
        images = np.reshape(data['features'][:], [-1, imsize, imsize])
        images = np.expand_dims(images, 1)
        labels = data['regression_target'][:]
        if self.filter_disease:
            if self.anomaly:
                indexes_to_use = np.where(labels<0.7)[0]
            else:
                indexes_to_use = np.where(labels>=0.7)[0]
            labels = labels[indexes_to_use]
            images = images[indexes_to_use]
        self.images = images
        self.n_images = len(self.images)
        self.targets = labels
        
    def __len__(self):
        return self.n_images

    def __getitem__(self, index):
        
        index = self.indices[index]
        x = self.images[index, ...]
        y = np.expand_dims(self.targets[index, ...],axis = 1)
        if self.transform is not None:
            x = self.transform(x)
        return torch.tensor(x), torch.tensor(y)

def load_and_generate_data(output_folder, mode = 'train'):
    np.random.seed(7)
    h5_filename = 'synthetic_mode_'+mode+'.hdf5'
    h5_filepath = os.path.join(output_folder, h5_filename)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(h5_filepath):
        regression_target, features = prepare_data_squares_by_size()
        with h5py.File(h5_filepath, 'w') as hdf5_file:
            hdf5_file.create_dataset('features', 
                data=features, dtype=np.float32)
            hdf5_file.create_dataset('regression_target',
                data=regression_target, dtype=np.float32)
    return h5py.File(h5_filepath, 'r')

#resizes a square to another side length
def transform_to_target(image_size, regression_target, img):
    new_side = round(image_size*regression_target/2)*2
    resized_square = resize(img, (new_side,new_side))
    border = int(image_size/2-new_side/2)
    if new_side>image_size:
        return image_preprocessing.crop_center(resized_square[None,...], -border,-border, image_size,image_size)[0,...]
    elif new_side<image_size:
        return image_preprocessing.pad(resized_square[None,...]+0.5, [border])[0,...]-0.5
    else:
        return resized_square

# prepare a dataset of squares of differents sizes, 
#small squares are from class 1 and big squares from class 0.
#  The inside of each square is filled with smoothed gaussian noise
def prepare_data_squares_by_size(image_size = 224,
                    num_samples=10000):
    regression_target = np.around(0.75*np.random.weibull(7, num_samples), decimals = 2)
    features = np.zeros([num_samples, image_size, image_size])
    for i in range(num_samples):
        features[i,:,:] = get_clean_square(1.0, image_size)
        noise = np.random.normal(scale=1, 
            size=np.asarray([image_size, image_size]))
        smoothed_noise = filters.gaussian(noise, 2.5)
        smoothed_noise = smoothed_noise / np.std(smoothed_noise) * 0.5
        features[i,:,:] += smoothed_noise*get_clean_square(1.0, image_size, background_=0, foreground=1)
        features[i,:,:] = transform_to_target(image_size, regression_target[i], features[i,:,:])
    return regression_target, features.reshape([-1, num_samples])   

def get_clean_square(regression_target, image_size, background_=-0.5, foreground=0.5):
    half_image_size = int(image_size / 2)
    block_size = int((half_image_size*0.8)*regression_target)
    to_return = np.zeros([image_size, image_size])
    to_return += background_
    to_return[half_image_size - block_size: half_image_size + block_size, 
        half_image_size - block_size: half_image_size + block_size] = foreground
    return to_return

#dataset to get the groundtruth for the residual between an example of class 1 and its equivalent
# on the class 0 support
class ValToy(Dataset):
    def __init__(self, toy_val_dataset):
        super().__init__()
        self.original_dataset = toy_val_dataset
        
    def __len__(self):
        return len(self.original_dataset)
        
    def __getitem__(self, index):
        case = self.original_dataset[index]
        #im0 = case[0][0]
        pft0 = case[0][1]
        im1 = case[1][0]
        pft1 = case[1][1]
        weight = 1./len(self)
        x_id = index
        im_diff = self.get_groundtruth_toy(im1,pft0, pft1)
        mask_im_diff = torch.ones_like(im_diff)
        return im1, torch.ones_like(pft1), im_diff, torch.zeros_like(pft0), weight, x_id, mask_im_diff
    
    #assumes that the equivalent of one example of class 1 is a simple upsizing of the same square
    @staticmethod
    def get_groundtruth_toy(img,pft_desired, pft_true):
        return torch.tensor(transform_to_target(224, pft_desired.item()/pft_true.item(), img[0,...].cpu().numpy())[None,...]).float()-img

def get_dataloaders(opt, mode='train'):
    assert(opt.n_dimensions_data==2)
    output_folder = opt.folder_dataset
    separation_classes = True
    transform_range = transforms.Compose([])
    transform_classification = lambda x: RegressionToClassification(x)
    apply_all_dataset_modifiers = lambda x: TransformsDataset(LoadToMemory(x), transform_range, 0)
    
    instantiate_normal_dataset_ = lambda: apply_all_dataset_modifiers(SynthDataset(output_folder, mode=mode, filter_disease = (separation_classes)))
    instantiate_abnormal_dataset_ = lambda: apply_all_dataset_modifiers(SynthDataset(output_folder,anomaly = (separation_classes), mode=mode, filter_disease = (separation_classes)))
    instantiate_diffs_val_dataset = lambda: (LoadToMemory(ValToy(DatasetsDifferentSizes(instantiate_normal_dataset_(), instantiate_abnormal_dataset_()))))
    instantiate_normal_dataset = lambda: transform_classification(instantiate_normal_dataset_())
    instantiate_abnormal_dataset = lambda: transform_classification(instantiate_abnormal_dataset_())
    return return_dataloaders(instantiate_normal_dataset, instantiate_abnormal_dataset, instantiate_diffs_val_dataset, opt, split = mode)