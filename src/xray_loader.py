"""COPD dataset
provides Datasets and Dataloaders for the COPD dataset used in the paper.
This file is provided as reference for what was done for this dataset. It is not
usable because the dataset will not be made public.

by Ricardo Bigolin Lanfredi
Last modified: 2020-07-01
Project: DeFI-GAN
GNU General Public License v3.0
"""

import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from . import image_preprocessing 
import copy
import numpy as np
try:
    import cPickle as pickle
except:
    import _pickle as pickle
import torch
from .utils_dataset import LoadToMemory, TransformsDataset, SeedRandom, H5Dataset, RegressionToClassification, ValImDiffDataset
from .utils_dataset import return_dataloaders
from PIL import Image
import itertools
import hashlib
from pandas.util import hash_pandas_object
import sys

#preventing the module from being loaded twice
sys.modules['xray_loader']=None

#function to get a hash string that representes the content of the dataset, so that we can identify
# the whole content of the dataset in each of the hdf5 files
def get_hash(preTransformSequence, dataset_table):
    str_to_hash = str(preTransformSequence) + str(hash_pandas_object(dataset_table).apply(str).str.cat(sep=''))
    this_hash = str(hashlib.sha256(str(str_to_hash).encode()).hexdigest()[:12])
    return this_hash

def get_dataset_from_table(dataset_table, preTransformSequence, transform_to_use, opt, mode):
    t_dataset = TransformsDataset(DatasetCOPD(dataset_table), preTransformSequence, 0)
    this_hash = get_hash(preTransformSequence, dataset_table)
    print(this_hash)
    t_dataset = H5Dataset(t_dataset, opt.folder_dataset, "xraydataset_" + mode + "_" + this_hash)
    if opt.load_to_memory:
        t_dataset = LoadToMemory(t_dataset)
    t_dataset  = TransformsDataset(t_dataset, transform_to_use, 0)
    t_dataset  = TransformsDataset(t_dataset, transforms.Compose([image_preprocessing.castTensor()]), 1)
    t_dataset = RegressionToClassification(t_dataset)
    return t_dataset

def get_dataloaders(opt, mode='train'):
    assert(opt.n_dimensions_data==2)
    #get images, labels and transforms from other project functions
    all_images, preTransformSequence, trainTransformSequence, testTransformSequence, num_ftrs = get_images(opt)
    all_labels = get_labels(opt)
    
    #use only frontal images
    all_images = all_images[all_images['position']=='PA']
    
    #filtering subjects with lung transplant
    lung_transplant_count_per_subject = all_labels[['subjectid', 'lung_transplant']].groupby(['subjectid']).sum().reset_index()
    lung_transplant_count_per_subject = lung_transplant_count_per_subject[lung_transplant_count_per_subject['lung_transplant']==0]
    all_labels = pd.merge(all_labels, pd.DataFrame(lung_transplant_count_per_subject['subjectid']))
    
    #only keeping cases where there is less than 30 days between x-ray study and pft
    maximum_date_diff = 30
    all_labels = all_labels[(all_labels['Date_Diff'] < maximum_date_diff)]

    #removing one unlabeled case of lung transplant
    all_labels = all_labels.loc[~(all_labels['subjectid']==3630)]
    
    #selecting only cases with the minimum date difference between x-ray study and pft
    cases_to_use = merge_images_and_labels(all_images, all_labels)
    cases_to_use = pd.merge(cases_to_use, all_labels).sort_values('Date_Diff', ascending=True).drop_duplicates('PFTExam_Global_ID', keep = 'first')[['subjectid', 'crstudy', 'pftid']]
    cases_to_use = pd.merge(cases_to_use, all_labels).sort_values('Date_Diff', ascending=True).drop_duplicates('CRStudy_Global_ID', keep = 'first')[['subjectid', 'crstudy', 'pftid']]
    
    #loading the subject ids that have more than one image and separting them to be used for validation and testing
    pickle_filenames = {'test':'/all_subjects_more_than_one_image.pkl', 'val':'/valids_all_subjects_more_than_one_image.pkl'}
    try:
        with open(opt.COPD_lists_location + pickle_filenames['test'], 'rb') as f:
            evolving_ids = pickle.load(f)
    except (TypeError, UnicodeDecodeError):
        with open(opt.COPD_lists_location + pickle_filenames['test'], 'rb') as f:
            evolving_ids = pickle.load(f, encoding='latin1')
    try:
        with open(opt.COPD_lists_location + pickle_filenames['val'], 'rb') as f:
            evolving_valids = pickle.load(f)
    except (TypeError, UnicodeDecodeError):
        with open(opt.COPD_lists_location + pickle_filenames['val'], 'rb') as f:
            evolving_valids = pickle.load(f, encoding='latin1')
    
    val_images = cases_to_use.loc[cases_to_use['subjectid'].isin(evolving_valids)]
    test_images = cases_to_use.loc[cases_to_use['subjectid'].isin(evolving_ids) & ~cases_to_use['subjectid'].isin(evolving_valids)]
    
    #using subjects with only one image for training
    train_images = cases_to_use.loc[~cases_to_use['subjectid'].isin(evolving_ids) & ~cases_to_use['subjectid'].isin(evolving_valids)]
    split = mode
    if mode =='train':
        images_to_use = train_images
        transform_to_use = trainTransformSequence
    elif mode == 'val':
        images_to_use = val_images
        transform_to_use = testTransformSequence
    elif mode == 'test':
        images_to_use = test_images
        transform_to_use = testTransformSequence
    all_joined_table =  pd.merge(images_to_use, all_images, on=['subjectid', 'crstudy'])
    dataset_table = pd.merge(all_joined_table, all_labels, on=['subjectid', 'crstudy', 'pftid'])
    
    if mode=='train':
        instantiate_normal_dataset = lambda: get_dataset_from_table(dataset_table[dataset_table['gold']==0], preTransformSequence, transform_to_use, opt, mode)
        instantiate_abnormal_dataset = lambda: get_dataset_from_table(dataset_table[dataset_table['gold']>0], preTransformSequence, transform_to_use, opt, mode)
    else:
        instantiate_normal_dataset = lambda: get_dataset_from_table(dataset_table.copy(), preTransformSequence, transform_to_use, opt, mode)
        instantiate_abnormal_dataset = instantiate_normal_dataset
    apply_loading = (lambda x: LoadToMemory(x)) if opt.load_to_memory else (lambda x: x)
    instantiate_all_dataset = lambda: apply_loading(get_dataset_from_table(dataset_table.copy(), preTransformSequence, transform_to_use, opt, mode))
    
    instantiate_diffs_val_dataset = lambda: apply_loading(SeedRandom(
                                            ValXRay(instantiate_all_dataset(), dataset_table, opt, get_hash(preTransformSequence, dataset_table.copy()))
                                             , seed = 0))
    
    return return_dataloaders(instantiate_normal_dataset, instantiate_abnormal_dataset, instantiate_diffs_val_dataset, opt, split = split)

def merge_images_and_labels(all_images, all_labels):
    a = all_images[['subjectid', 'crstudy']].groupby(['subjectid', 'crstudy']).size().reset_index(name="count")
    
    #remove_cases_more_one_image_per_position, to prevent studies where a single image does not cover whole lung
    a = a[a['count']<2]
    
    all_images_merged = a[['subjectid', 'crstudy']]
    joined_tables = pd.merge(all_images_merged, all_labels, on=['subjectid', 'crstudy'])
    joined_tables = joined_tables[['subjectid', 'crstudy', 'pftid', 'CRStudy_Global_ID', 'PFTExam_Global_ID', 'Date_Diff']].groupby(['subjectid', 'crstudy', 'pftid', 'CRStudy_Global_ID', 'PFTExam_Global_ID', 'Date_Diff']).count().reset_index()
    cases_to_use = joined_tables[['subjectid', 'crstudy', 'pftid']].groupby(['subjectid', 'crstudy', 'pftid']).count().reset_index()
    return cases_to_use

#dataset to load image files and labels from the table defining the dataset
class DatasetCOPD(Dataset):
    def __init__ (self, listImage):
        super().__init__()
        self.listImage = listImage
        self.file = None
        self.n_images = len(self.listImage)
    
    def __getitem__(self, index):
        if not isinstance(index, int):
            index = int(index)
        imagePath = self.listImage['filepath'].iloc[index]
        imageData = Image.open(imagePath)
        imageLabel= np.array(self.listImage[['fev1fvc_predrug']].iloc[index])
        return imageData, imageLabel
    
    def __len__(self):
        return self.n_images

#get disease severity. 0 means healthy. 1 to 4 represents from mild to severe COPD
def get_gold(fev1_ratio, fev1fvc_predrug):
    return (fev1fvc_predrug<0.7)*(1+(fev1_ratio<0.8)+(fev1_ratio<0.5)+(fev1_ratio<0.3))

columns_translations = \
                     {"Subject_Global_ID": "subjectid",
                        "CRStudy_Local_ID": "crstudy",
                        "PFTExam_Local_ID": "pftid",
                        'Predicted FVC':'fvc_pred',
                        'Predicted FEV1':'fev1_pred',
                        'Predicted FEV1/FVC':'fev1fvc_pred',
                        'Pre-Drug FVC':'fvc_predrug',
                        'Pre-Drug FEV1':'fev1_predrug',
                        'Pre-Drug FEV1/FVC':'fev1fvc_predrug',
                        'Pre-%Pred FVC':'fvc_ratio',
                        'Pre-%Pred FEV1':'fev1_ratio',
                        'Pre-%Pred FEV1/FVC':'fev1fvc_ratio',
                        'TOBACCO_PAK_PER_DY':'packs_per_day',
                        'TOBACCO_USED_YEARS':'years_of_tobacco',
                        'COPD':'copd',
                        'fev1_diff':'fev1_diff',
                        'fvc_diff':'fvc_diff',
                        'AGE_AT_PFT':'age',
                        'GENDER':'gender',
                        'TOBACCO_STATUS':'tobacco_status',
                        'SMOKING_TOBACCO_STATUS':'smoking_tobacco_status',
                        'LUNG_TRANSPLANT':'lung_transplant'}

percentage_labels = ['fev1fvc_pred','fev1fvc_predrug','fvc_ratio','fev1_ratio','fev1fvc_ratio']
normalization_mean = [0.485, 0.456, 0.406]
normalization_std = [0.229, 0.224, 0.225]

def get_labels(opt):
    all_labels = pd.read_csv(opt.COPD_labels_location) 
    all_labels.rename(index=str, columns=columns_translations, inplace = True)
    all_labels.dropna(subset=['fvc_predrug'], inplace=True)
    all_labels[percentage_labels] = all_labels[percentage_labels] /100.
    all_labels['copd'] = (all_labels['fev1fvc_predrug']< 0.7)*1
    all_labels['gold'] = get_gold(all_labels['fev1_ratio'], all_labels['fev1fvc_predrug'])
    return all_labels
    
def get_images(opt, use_histogram_eq = True):
    all_images, num_ftrs, list_pretransforms = get_all_images(opt)
    
    #only using frontal, posterioranterior chest x-rays
    sets_of_images = all_images[all_images['position']=='PA']

    list_transforms = []
    list_transforms = list_transforms + list_pretransforms + [image_preprocessing.ToNumpy()]
    list_pre_transforms = copy.deepcopy(list_transforms)
    train_list_transforms = []
    test_list_transforms = []
    
    crop_size = 224
    list_pre_transforms += [image_preprocessing.NormalizeTensorMinMax01()]
    if use_histogram_eq:
        list_pre_transforms += [image_preprocessing.HistogramEqualization()]
    list_pre_transforms += [image_preprocessing.Range01To11()]
    train_list_transforms += [image_preprocessing.RandomCropNumpy(crop_size)]
    test_list_transforms += [image_preprocessing.CenterCropNumpy(crop_size)]
    train_list_transforms += [image_preprocessing.castTensor()]
    test_list_transforms += [image_preprocessing.castTensor()]
    
    train_list_transforms += [image_preprocessing.ExtractFirstChannel()]
    test_list_transforms += [image_preprocessing.ExtractFirstChannel()]
    
    preTransformSequence = transforms.Compose(list_pre_transforms)
    trainTransformSequence = transforms.Compose(train_list_transforms)
    testTransformSequence = transforms.Compose(test_list_transforms)
    
    return sets_of_images, preTransformSequence, trainTransformSequence, testTransformSequence, num_ftrs

def get_all_images(opt):
    #list of name of files containing filepaths for all images in the dataset, separated by \r\n
    files_with_image_filenames = [ 'images' + str(dataset) + '.txt' for dataset in ['2012-2016', '2017']]
    num_ftrs = None
    list_pretransforms =[]
    normalize = transforms.Normalize(normalization_mean,
                                        normalization_std)
    resize_size = 256
    
    list_pretransforms = list_pretransforms + [image_preprocessing.Convert16BitToFloat(),
              image_preprocessing.CropBiggestCenteredInscribedSquare(),
              transforms.Resize(size=(resize_size)),
              transforms.ToTensor(),
              normalize]
    
    all_images = []
    for file_with_image_filenames in files_with_image_filenames:
        all_images = all_images + read_filenames(opt.COPD_lists_location + '/' + file_with_image_filenames)
    all_images = pd.DataFrame(all_images)
    all_images['image_index'] = all_images.index
    return all_images, num_ftrs, list_pretransforms

def read_filenames(pathFileTrain):
    listImage = []
    fileDescriptor = open(pathFileTrain, "r")
    
    line = True
    while line:
      line = fileDescriptor.readline()
      if line:
          thisimage = read_filename(line)
          if thisimage is not None:
              listImage.append(thisimage)
    fileDescriptor.close()
    return listImage

#get dictionary of xray information from the filename
def read_filename(line):
    thisimage = {}
    
    lineItems = line.split()
    thisimage['filepath'] = lineItems[0]
    splitted_filepath = thisimage['filepath'].replace('\\','/').split("/")
    if splitted_filepath[-1] in [ 'PFT_000263_CRStudy_15_ScanID_1-W_Chest_Lat.png', 'PFT_000264_CRStudy_05_ScanID_2-W_Chest_Lat.png']:
        return None
    splitted_ids = splitted_filepath[-1].replace('-','_').replace('.','_').split('_')
    thisimage['subjectid'] = int(splitted_ids[1])
    thisimage['crstudy'] = int(splitted_ids[3])
    try:
        thisimage['scanid'] = int(splitted_ids[5])
    except ValueError:
        return None
    position = splitted_ids[-2].upper()
    if 'LARGE' in position:
        return None
    if 'STANDING' in position:
        position = splitted_ids[-3].upper()
    elif 'LAT' in position:
        position = 'LAT'
    elif 'PA' in position:
        position = 'PA'
    elif 'AP' in position:
        position = 'AP'
    elif 'SUPINE' in position:
        return None
    elif 'CHEST' in position:
        return None
    elif 'P' == position and  splitted_ids[-3].upper() == 'A':
        position = 'AP'
    elif 'PORTRAIT' in position:
        return None
    elif 'SID' in position:
        return None
    else:
        raise ValueError('Unknown position: '+position + ', for file: ' +  lineItems[0])
    thisimage['position'] = position
    return thisimage

#dataset pairing longitudinal data of subjects that have images labeled with both COPD and no COPD
# registration is performed between image and their difference is used as groundtruth
# for difference between healthy features and COPD features
class ValXRay(Dataset):
    def __init__(self, xray_val_dataset, metadata_val, opt, hash_):
        super().__init__()
        assert(len(metadata_val)==len(xray_val_dataset))
        metadata_val = metadata_val.copy()
        metadata_val['original_indexes'] = metadata_val.index
        
        #apply filter_min_max function to metadata_val
        metadata_val['data_to_filter'] = list(zip(metadata_val['fev1fvc_predrug'], metadata_val['PFT_DATE_MonthOnly'], metadata_val['PFT_DATE_YearOnly']))
        metadata_val_temp = (copy.deepcopy(metadata_val)[['subjectid', 'data_to_filter']].groupby('subjectid').aggregate(self.filter_min_max)).reset_index()
        metadata_val_temp.columns = ['subjectid', 'repeat']
        metadata_val_temp = metadata_val_temp[metadata_val_temp['repeat']]
        metadata_val = pd.merge(metadata_val, metadata_val_temp)
        
        index_pairs = []
        self.total_pairs_this_subject = []
        self.x_ids = []
        #create list with all pair of indexes of same subjects
        self.total_subjects = 0
        for sid in metadata_val['subjectid'].unique():
            #removing problematic subjects
            if sid in [1317, 136, 1934, 335, 1948, 1437, 5758, 1663]:
                continue
            this_metadata = metadata_val[metadata_val['subjectid']==sid]
            repeated_crtudies = list(set(this_metadata[this_metadata['fev1fvc_predrug']>=0.7]['crstudy'].values).intersection(set(this_metadata[this_metadata['fev1fvc_predrug']<0.7]['crstudy'].values)))
            this_metadata = this_metadata[~this_metadata['crstudy'].isin(repeated_crtudies)]
            max_metadata = this_metadata[this_metadata['fev1fvc_predrug']>=0.7]['original_indexes'].values
            min_metadata = this_metadata[this_metadata['fev1fvc_predrug']<0.7]['original_indexes'].values
            this_index_pairs = list(itertools.product(min_metadata, max_metadata))
            if len(this_index_pairs)>0:
                index_pairs = index_pairs + this_index_pairs
                self.total_subjects += 1
                self.total_pairs_this_subject += [len(this_index_pairs)]*len(this_index_pairs)
                self.x_ids += [int(sid)]*len(this_index_pairs)
        self.index_pairs = index_pairs
        self.xray_val_dataset = xray_val_dataset
        self.im_diff_dataset = H5Dataset(ValImDiffDataset(self.xray_val_dataset, self.index_pairs, opt), opt.folder_dataset, "valxraydataset_sitk_invertclasses_" + hash_)
        assert(len(self.im_diff_dataset) == len(self))
    
    #filter out subjects that do not have cases both with and without copd
    @staticmethod
    def filter_min_max(values):
        values = np.array([*values])
        if not np.max(values[:,0])>=0.7:
            return False
        if not np.min(values[:,0])<0.7:
            return False
        return (np.max(values[:,0])-np.min(values[:, 0]))>0
    
    def __len__(self):
        return len(self.index_pairs)
        
    def __getitem__(self,index):
        out0 = self.xray_val_dataset[self.index_pairs[index][0]]
        im0 = out0[0].cuda()
        pft0 = out0[1].cuda()
        out1 = self.xray_val_dataset[self.index_pairs[index][1]]
        pft1 = out1[1].cuda()
        im_diff, moved_mask = self.im_diff_dataset[index]
        
        #masks are used to only calculate the NCC scores inside the region where
        # both x-rays have non-padded pixels, after registration
        mask_im_diff = moved_mask
        
        #weight is used to have each subject have the same weight in the calculation of 
        #NCC scores, and to prevent subjects with several images from dominating it
        weight = 1./self.total_subjects/self.total_pairs_this_subject[index]
        
        x_id = self.x_ids[index]
        return im0, pft0, torch.tensor(im_diff), pft1, weight, x_id, torch.tensor(mask_im_diff)