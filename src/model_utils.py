"""Auxiliary functions for unet.py and critics.py

Originally from https://github.com/orobix/Visual-Feature-Attribution-Using-Wasserstein-GANs-Pytorch
Modified by Ricardo Bigolin Lanfredi
Last modified: 2020-06-26
Project: DeFI-GAN
GNU General Public License v3.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    def forward(self, x):
        return x

ACTIVATION = nn.ReLU


def crop_and_concat(upsampled, bypass, crop=False):
    if crop:
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = F.pad(bypass, (-c, -c, -c, -c))
    return torch.cat((upsampled, bypass), 1)


def conv2d_bn_block(in_channels, out_channels, opt, activation=ACTIVATION):
    '''
    returns a block conv-bn-activation
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels, eps = 1e-3, momentum = 0.01),
        activation(),
    )


def deconv2d_bn_block(in_channels, out_channels, opt, use_upsample=False, kernel=4, stride=2, padding=1, activation=ACTIVATION):
    '''
    returns a block deconv-bn-activation
    use_upsample = True helps to remove chessboard artifacts:
    https://distill.pub/2016/deconv-checkerboard/
    '''
    if use_upsample:
        up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        )
    else:
        up = nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=stride, padding=padding)
    return nn.Sequential(
        up,
        nn.BatchNorm2d(out_channels, eps = 1e-3, momentum = 0.01),
        activation(),
    )

def conv2d_block(in_channels, out_channels, opt, kernel=3, stride=1, padding=1, activation=ACTIVATION):
    '''
    returns a block conv-activation
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding),
        activation(),
    )


def conv3d_block(in_channels, out_channels, opt, kernel=3, stride=1, padding=1, activation=ACTIVATION):
    '''
    returns a block 3D conv-activation
    '''
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel, stride=stride, padding=padding),
        activation(),
    )
