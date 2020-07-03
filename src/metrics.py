"""DeFI-GAN metrics

File that contain functions for calculating the normalized cross-correlation 
between two images and a class Metrics for storing losses and metrics during 
mini batch inferences, so that you can get an epoch summary after the epoch 
is complete.

by Ricardo Bigolin Lanfredi
-Last modified: 2020-06-26
Project: DeFI-GAN
GNU General Public License v3.0
"""

import collections
import torch
import numpy as np

# Defining eq. 7 from the paper, masked normalized cross-correlation.
#The mask is important for both datasets from the paper:
# for ADNI, the mask assures that only regions of the brain are used for calculating ncc, 
# eliminating background pixels from the calculation
# for the COPD dataset, it assures that regions that were padded with zeros during registration
# of the x-ray are not used for calculations
def normalized_cross_correlation_one_example(a,v, mask):
    if mask.ndim>2:
        assert(mask.size(0)==1)
    assert(a.size()==v.size())
    assert(a.size(0)==1)
    a = a.squeeze(0)
    v = v.squeeze(0)  
    flattened_mask = mask.view([-1])
    masked_a = a.view([-1])[flattened_mask.bool()]
    masked_v = v.view([-1])[flattened_mask.bool()]
    norm_std = torch.std(masked_a, dim = 0)*torch.std(masked_v, dim = 0)
    step1a = (a - torch.mean(masked_a, dim = 0)[(None, ) * (len(a.size())-1)])
    step1v = (v - torch.mean(masked_v, dim = 0)[(None, ) * (len(v.size())-1)])
    step2 = torch.sum((step1a*step1v*mask).view([-1]), dim = 0)
    step3 = step2/norm_std
    step3 = step3/(torch.sum(flattened_mask, dim=0)-1)
    return step3

def normalized_cross_correlation(a,v, mask):
    assert(a.size(0)==v.size(0))
    assert(a.size(0)==mask.size(0))
    to_return =  torch.stack([normalized_cross_correlation_one_example(a[i,...],v[i,...], mask[i,...]) for i in range(a.size(0))])
    return to_return

class Metrics():
    def __init__(self, opt):
        self.values = collections.defaultdict(list)
    
    def add_ncc(self, gt_toy, delta_x, weight, mask):
        gt_toy = gt_toy.detach()
        delta_x = delta_x.detach()
        mask = mask.detach()
        self.add_list('ncc', normalized_cross_correlation(gt_toy, delta_x, mask))
        
        #save weights given for each of the batch examples, so that a weighted average of the ncc is calculated
        # This way, each subject has the same weight in the calculation of the final ncc.
        self.add_list('ncc_weights', weight)
    
    def add_list(self, key, value):
        value = value.detach().cpu().tolist()
        self.values[key] += value
        
    def add_value(self, key, value):
        if value is not None:
            value = value.detach().cpu()
            self.values[key].append( value)
    
    def calculate_average(self):
        self.average = {}
        for key, element in self.values.items():
            if ('_weights' in key):
                continue
            #if the current key has a complementary key with respective weights saved,
            #calculate the weighted average instead of a simple average
            if (key+'_weights') in self.values.keys():
                self.average[key] = np.sum(np.array(element)*np.array(self.values[key+'_weights']))
            else:
                n_values = len(element)
                if n_values == 0:
                    self.average[key] = 0
                    continue
                sum_values = sum(element)
                self.average[key] = sum_values/float(n_values)
        self.values = collections.defaultdict(list)
    
    def get_average(self):
        self.calculate_average()
        return self.average
        
    def get_last_added_value(self,key):
        return self.values[key][-1]