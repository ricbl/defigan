"""Dataset preprocessing functions

by Ricardo Bigolin Lanfredi
Last modified: 2020-06-26
Project: DeFI-GAN
GNU General Public License v3.0
"""

import numpy as np
import skimage
import skimage.exposure
from random import randint
import numbers
from PIL import ImageMath
import torch

class BatchNormalizeTensorMinMax01(object):
    def __init__(self):
        pass
    def __call__(self, tensor):
        maxes, _ = torch.max(tensor.view([tensor.size(0), -1]), dim = 1)
        maxes = maxes[(..., ) + (None, ) * (len(tensor.size())-1)].expand(tensor.size())
        mins, _ = torch.min(tensor.view([tensor.size(0), -1]), dim = 1)
        mins = mins[(..., ) + (None, ) * (len(tensor.size())-1)].expand(tensor.size())
        assert(tensor.size()==mins.size())
        assert(tensor.size()==maxes.size())
        to_return = (tensor - mins)/(maxes - mins)
        return to_return

class BatchNormalizeTensor(object):
    def __init__(self, mean, std, invert_batchnormalized_tensors = False):
        self.mean = mean
        self.std = std
        self.invert_batchnormalized_tensors = invert_batchnormalized_tensors

    def __call__(self, tensor):
        assert(tensor.shape[1]==self.std.shape[1])
        assert(self.mean.shape[1]==tensor.shape[1])
        assert(tensor.ndim==self.std.ndim)
        assert(tensor.ndim==self.mean.ndim)
        if self.invert_batchnormalized_tensors:
            to_return = (tensor+self.mean-1)/self.std
        else:
            to_return = (tensor-self.mean)/self.std
        return to_return
        
class HistogramEqualization(object):
    def __init__(self, nbins = 256 ):
        self.nbins = nbins

    def __call__(self, img):
        img2 = np.copy(img)
        assert(img2.ndim==3)
        assert(img2.shape[0]==3 or img2.shape[0]==1)
        for channel in range(img.shape[0]):  # equalizing each channel
            img2[channel, :, :] = skimage.exposure.equalize_hist(img[channel, :, :], nbins = self.nbins)
        return img2

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ExtractFirstChannel(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return img[0:1,...]

    def __repr__(self):
        return self.__class__.__name__ + '()'
        
def crop_center(img,sh, sw, th, tw):
    assert(img.ndim==3)
    assert(img.shape[0]==3 or img.shape[0]==1)
    return img[:,sh:sh+th,sw:sw+tw]

def pad(array, padding):
    assert(array.ndim==3)
    assert(array.shape[0]==3 or array.shape[0]==1)
    if isinstance(padding, numbers.Number):
        padding = (int(padding))
    if len(padding)==4: #left, top, right, bottom
        pass
    elif len(padding)==2:
        padding = padding*2
    elif len(padding)==1:
        padding = padding*4
    else:
        raise ValueError('padding has an invalid value in function pad. expected list or tuple of length 1, 2 or 4 or a number. received: '+str(padding))
    result = np.zeros((array.shape[0], array.shape[1]+ padding[1]+ padding[3], array.shape[2]+ padding[0]+ padding[2]))
    result[:,padding[1]:(array.shape[1]+padding[1]), padding[0]:(array.shape[2]+padding[0])] = array
    return result

class RandomCropNumpy():
    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        assert(img.ndim==3)
        assert(img.shape[0]==3 or img.shape[0]==1)
        c, h , w = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = randint(0, h - th)
        j = randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        assert(img.ndim==3)
        assert(img.shape[0]==3 or img.shape[0]==1)
        if self.padding > 0:
            img = pad(img, self.padding)
        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))
        sh, sw, th, tw = self.get_params(img, self.size)
        return crop_center(img,sh, sw, th, tw).copy()

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

class CenterCropNumpy(RandomCropNumpy):
    @staticmethod
    def get_params(img, output_size):
        assert(img.ndim==3)
        assert(img.shape[0]==3 or img.shape[0]==1)
        c, h, w = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = (h - th)//2
        j = (w - tw)//2
        return i, j, th, tw
        
class ToNumpy(object):
    def __init__(self):
        pass
    def __call__(self, tensor):
        return tensor.numpy()
    def __repr__(self):
        return self.__class__.__name__ + '()'
        
class castTensor(object):
    def __call__(self, image):
        return torch.FloatTensor(np.array(image))
    def __repr__(self):
        return self.__class__.__name__ + '()'
        
class Convert16BitToFloat(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        tensor.mode = 'I'
        return ImageMath.eval('im/256', {'im':tensor}).convert('RGB')

    def __repr__(self):
        return self.__class__.__name__ + '()'
        
class CropBiggestCenteredInscribedSquare(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        assert(len(tensor.size)==2)
        longer_side = min(tensor.size)
        horizontal_padding = (longer_side - tensor.size[0]) / 2
        vertical_padding = (longer_side - tensor.size[1]) / 2
        return tensor.crop(
            (
                -horizontal_padding,
                -vertical_padding,
                tensor.size[0] + horizontal_padding,
                tensor.size[1] + vertical_padding
            )
        )

    def __repr__(self):
        return self.__class__.__name__ + '()'

class NormalizeTensorMinMax01(object):
    def __init__(self):
        pass

    def __call__(self, img):
        assert(img.shape[0]==3 or img.shape[0]==1)
        img = np.copy(img)
        for channel in range(img.shape[0]): 
            image_min = np.min(img[channel, ...])
            image_max = np.max(img[channel, ...])
            img[channel, ...] = (img[channel, ...] - image_min)/(image_max-image_min)
        return img
        
    def __repr__(self):
        return self.__class__.__name__ + '()'
        
class NormalizeTensorMinMax01Torch(object):
    def __init__(self):
        pass
        
    def __call__(self, img):
        assert(img.shape[0]==3 or img.shape[0]==1)
        assert(img.shape[1]>3)
        img = img.clone()
        for channel in range(img.shape[0]): 
            image_min = torch.min(img[channel, ...])
            image_max = torch.max(img[channel, ...])
            img[channel, ...] = (img[channel, ...] - image_min)/(image_max-image_min)
        return img
        
    def __repr__(self):
        return self.__class__.__name__ + '()'
        
class Range01To11(object):
    def __init__(self):
        pass
    def __call__(self, img):
        to_return = (img - 0.5)*2 
        return to_return
        
    def __repr__(self):
        return self.__class__.__name__ + '()'