"""Registration functions

Provides functions that can register one image/volume to another one

by Ricardo Bigolin Lanfredi
Last modified: 2020-06-26
Project: DeFI-GAN
GNU General Public License v3.0
"""

import SimpleITK as sitk
import numpy as np
from . import flow_layer
import torch

def permute_channel_last(x):
    return x.clone().permute((0,) + tuple(range(2,len(x.size()))) + (1,))
    
def permute_channel_second(x):
    return x.clone().permute((0,) + (len(x.size())-1,) + tuple(range(1,len(x.size())-1)))
    
def sitk_registration(im_fixed, im_moving, fixed_mask = None, moving_mask = None, mode = "affine"):
    assert(mode in ["rigid", "affine", "nonrigid"])
    im_fixed = (im_fixed[0,...])
    im_fixed = im_fixed.detach().cpu().numpy()
    im_moving = (im_moving[0,...])
    im_moving = im_moving.detach().cpu().numpy()
    for dim_index in range(im_fixed.ndim):
        assert(im_fixed.shape[dim_index]==im_moving.shape[dim_index])
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.LogToConsoleOff()
    elastixImageFilter.LogToFileOff()
    elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(im_fixed))
    elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(im_moving))
    all_pixels = np.ones_like(im_moving)
    if moving_mask is None:
        moving_mask = all_pixels
    else:
        moving_mask = (moving_mask[0,...])
        moving_mask = moving_mask.detach().cpu().numpy()
    if fixed_mask is None:
        fixed_mask = all_pixels
    else:
        fixed_mask = (fixed_mask[0,...])
        fixed_mask = fixed_mask.detach().cpu().numpy()
    elastixImageFilter.SetFixedMask(sitk.GetImageFromArray(fixed_mask.astype('uint8')))
    elastixImageFilter.SetMovingMask(sitk.GetImageFromArray(moving_mask.astype('uint8')))
    
    parameterMapVector = sitk.VectorOfParameterMap()

    parameterMapVector.append(sitk.GetDefaultParameterMap("affine" if mode=='nonrigid' else mode))
    if mode=="nonrigid":
        parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
    elastixImageFilter.SetParameterMap(parameterMapVector)

    elastixImageFilter.Execute()
    im_moving_deformed = sitk.GetArrayFromImage(elastixImageFilter.GetResultImage())
    
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.LogToConsoleOff()
    transformixImageFilter.LogToFileOff()
    transformixImageFilter.SetMovingImage(sitk.GetImageFromArray(im_moving))
    transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
    transformixImageFilter.ComputeDeformationFieldOn()
    transformixImageFilter.Execute()
    flow = transformixImageFilter.GetDeformationField()
    np_flow = sitk.GetArrayFromImage(flow)
    np_flow[np.where(np_flow<-1000)] = 0
    np_flow[np.where(np_flow>1000)] = 0
    
    moved_moving_mask = apply_flow(moving_mask, np_flow)
    moved_moving_mask = np.nan_to_num(moved_moving_mask)
    moved_moving_mask[moved_moving_mask<0.95]=0
    moved_moving_mask[moved_moving_mask>=0.95]=1
    difference_ = (im_moving_deformed - im_fixed )*moved_moving_mask*fixed_mask
    return difference_, im_moving_deformed, np_flow, moved_moving_mask

def apply_flow(numpy_to_deform, flow):
    return permute_channel_last(flow_layer.deform_to_grid(torch.tensor(numpy_to_deform).cuda().float()[None,None,...],permute_channel_second(torch.tensor(flow[None])).cuda())).detach().cpu()[(0,) + (slice(None),)*(numpy_to_deform.ndim) + (0,)]