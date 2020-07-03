"""ADNI dataset
provides Datasets and Dataloaders for the ADNI dataset used in the paper

by Ricardo Bigolin Lanfredi
Last modified: 2020-07-01
Project: DeFI-GAN
GNU General Public License v3.0
"""

from . import adni_data_loader
import numpy as np
import pandas as pd
import copy
import itertools
import torch

from .utils_dataset import LoadToMemory, SeedRandom, H5Dataset, ValImDiffDataset
from .utils_dataset import return_dataloaders

class BrainDataset(torch.utils.data.Dataset):
    def __init__(self, images, metadata, opt):
        super().__init__()
        self.images = images
        self.metadata = metadata
        assert(opt.n_dimensions_data == 3)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        label = self.metadata['label'].iloc[index]
        this_index = self.metadata['original_index'].iloc[index]
        this_image = torch.tensor(self.images[this_index, ...])
        this_image = this_image[None, ...]
        return this_image, torch.tensor(label)[None]
    
    def get_index_from_original_index(self, original_index):
        for iloc_index in range(len(self)):
            if original_index==self.metadata['original_index'].iloc[iloc_index]:
                return iloc_index

def get_dataloaders(opt, mode='train'):
    data = adni_data_loader.load_and_maybe_process_data(
            input_folder=opt.ADNI_images_location,
            preprocessing_folder=opt.folder_dataset,
            size=(128, 160, 112),
            target_resolution=(1.3, 1.3, 1.3),
            label_list=(1,2),
            offset=None,
            force_overwrite=False,
            rescale_to_one=True
        )
    metadata = pd.DataFrame()
    for column in ['diagnosis','field_strength','rid','viscode']:
        metadata[column] = np.array(data[column + '_' + mode][...])
    metadata['original_index'] = metadata.index
    images_this_split = data['images_' + mode]
    corrupted_images = []
    images_where_preprocessing_failed = ['855_0','424_4','4741_2','4741_1', '4741_5', '4741_7', '4892_0' ]
    for index in range(images_this_split.shape[0]):
        # any image with > 7000KB should be reviewed
        # any image with < 4200KB should be reviewed
        # removing images that are corrupted: wrong orientation, wrong skull stripping, ...
        if np.min(images_this_split[index, 44 , :, :])==np.max(images_this_split[index, 56 , :, :]) or \
            (str(metadata['rid'].iloc[index]) + '_' + str(metadata['viscode'].iloc[index]) in 
            images_where_preprocessing_failed):
            corrupted_images.append(index)
    metadata = metadata[~metadata['original_index'].isin(corrupted_images)]
    metadata['diagnosis'] = np.asarray([np.argwhere(i==np.asarray((1,2))) for i in metadata['diagnosis']]).flatten()
    
    if opt.load_to_memory:
        images_this_split = images_this_split[...]
    
    metadata['label'] = metadata['diagnosis']
    
    apply_loading = (lambda x: LoadToMemory(x)) if opt.load_to_memory else (lambda x: x)
    
    if mode=='train':
        instantiate_healthy_dataset = lambda: apply_loading(BrainDataset(images_this_split, metadata[metadata['diagnosis']==0], opt))
        instantiate_anomaly_dataset = lambda: apply_loading(BrainDataset(images_this_split, metadata[metadata['diagnosis']==1], opt))
    else:
        instantiate_healthy_dataset = lambda: apply_loading(BrainDataset(images_this_split, metadata, opt))
        instantiate_anomaly_dataset = lambda: apply_loading(BrainDataset(images_this_split, metadata, opt))
    instantiate_diffs_val_dataset = lambda: apply_loading(SeedRandom(
                                            ValBrain(BrainDataset(images_this_split, metadata, opt), metadata, opt) 
                                             , seed = 0))
    return return_dataloaders(instantiate_healthy_dataset, instantiate_anomaly_dataset, instantiate_diffs_val_dataset, opt, split = mode)

# We aligned longitudinal images rigidly and subtracted them 
# from each other to obtain an observed disease effect map.
class ValBrain(torch.utils.data.Dataset):
    def __init__(self, original_dataset, metadata, opt):
        super().__init__()
        # ['adas13','age','diagnosis','field_strength','gender','images',
        # 'mmse','rid','viscode','weight']
        #apply filter_min_max function to metadata_val
        metadata = metadata.copy()
        metadata['data_to_filter'] = list(zip(metadata['diagnosis'], metadata['field_strength'], metadata['rid'], metadata['viscode']))
        metadata_val_temp = (copy.deepcopy(metadata)[['rid', 'data_to_filter']].groupby('rid').aggregate(self.filter_min_max)).reset_index()
        
        metadata_val_temp.columns = ['rid', 'repeat']
        metadata_val_temp = metadata_val_temp[metadata_val_temp['repeat']]
        metadata = pd.merge(metadata, metadata_val_temp)
        index_pairs = []
        self.total_pairs_this_subject = []
        self.x_ids = []
        self.total_subjects = 0
        #create list with all pair of indexes of same subjects
        for sid in metadata['rid'].unique():
            this_metadata = metadata[metadata['rid']==sid]
            this_field_strength = this_metadata[this_metadata['viscode']==0]['field_strength'].values[0]
            max_metadata = [original_dataset.get_index_from_original_index(original_index) for original_index in this_metadata[(this_metadata['diagnosis']==1) & (this_metadata['field_strength']==this_field_strength)]['original_index'].values]
            
            min_metadata = [original_dataset.get_index_from_original_index(original_index) for original_index in this_metadata[this_metadata['viscode']==0]['original_index'].values]
            this_index_pairs = list(itertools.product(max_metadata, min_metadata))
            index_pairs = index_pairs + this_index_pairs
            self.total_subjects += 1
            self.total_pairs_this_subject += [len(this_index_pairs)]*len(this_index_pairs)
            self.x_ids += [int(sid)]*len(this_index_pairs)
        self.index_pairs = index_pairs
        self.original_dataset = original_dataset
        self.metadata = metadata
        
        self.im_diff_dataset = H5Dataset(ValImDiffDataset(self.original_dataset, self.index_pairs, opt, registration_mode = "rigid"), opt.folder_dataset, "valbrain" + str(opt.n_dimensions_data)+ "Ddataset_sitk_invertclasses_" + opt.split_validation)
    
    #filter out subjects that do not have cases both with and without AD with the same field strength
    @staticmethod
    def filter_min_max(values):
        #indexes for values: 0: 'diagnosis' 1:'field_strength' 2:'rid' 3:'viscode'
        values = np.array([*values])
        index_baseline = list(np.where(values[:,3]==0)[0])
        if len(index_baseline)!=1:
            return False
        if values[index_baseline[0],0] == 1:
            return False
        field_strength_1 = values[np.where(values[:,0]==1),1]
        field_strength_0 = values[np.where(values[:,0]==0),1]
        intersection = np.intersect1d(field_strength_1, field_strength_0)
        return len(intersection)>0
    
    def __len__(self):
        return len(self.index_pairs)
        
    def __getitem__(self,index):
        out0 = self.original_dataset[self.index_pairs[index][0]]
        im0 = out0[0].cuda()
        pft0 = out0[1].cuda()
        out1 = self.original_dataset[self.index_pairs[index][1]]
        pft1 = out1[1].cuda()
        im_diff, moving_mask = self.im_diff_dataset[index]
        
        #-1 is used for filling the backroung of the volume. Mask selects voxels where value is not -1
        mask_im_diff = moving_mask*(im0.cpu().numpy()!=-1)
        
        weight = 1./self.total_subjects/self.total_pairs_this_subject[index]
        x_id = self.x_ids[index]
        return im0, pft0, torch.tensor(im_diff), pft1, weight, x_id, torch.tensor(mask_im_diff)