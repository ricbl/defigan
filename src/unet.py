"""U-net generator
This module provides a UNet class to be used as the generator model in the algorithm

by Ricardo Bigolin Lanfredi
Last modified: 2020-07-01
Project: DeFI-GAN
GNU General Public License v3.0
"""

import torch.nn as nn
import torch

ACTIVATION = nn.ReLU

class Identity(nn.Module):
    def forward(self, x):
        return x

def crop_and_concat(upsampled, bypass, crop=False):
    return torch.cat((upsampled, bypass), 1)

def conv2d_bn_block(in_channels, out_channels, opt, activation=ACTIVATION, dimensions = 2):
    '''
    returns a block conv-bn-activation
    '''
    conv_operation = nn.Conv2d if (dimensions == 2) else nn.Conv3d
    bn_operation = nn.BatchNorm2d if (dimensions == 2) else nn.BatchNorm3d
    return nn.Sequential(
        conv_operation(in_channels, out_channels, 3, padding=1),
        bn_operation(out_channels, eps = 1e-3, momentum = 0.01),
        activation(),
    )

class Upsample(nn.Module):
    def __init__(self, scale_factor=2, dimensions=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = 'trilinear' if (dimensions==3) else 'bilinear'
    
    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor = self.scale_factor, mode=self.mode, align_corners=True)
    
def deconv2d_bn_block(in_channels, out_channels, opt, use_upsample=True, kernel=4, stride=2, padding=0, activation=ACTIVATION, dimensions = 2):
    '''
    returns a block deconv-bn-activation
    use_upsample = True helps to remove chessboard artifacts:
    https://distill.pub/2016/deconv-checkerboard/
    '''
    if use_upsample:
        if dimensions == 2:
            conv_layer = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        elif dimensions == 3:
            conv_layer = nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1)
        up = nn.Sequential(
            Upsample(scale_factor=2, dimensions = dimensions),
            conv_layer
        )
    else:
        if dimensions == 2:
            up = nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=stride, padding=padding)
        elif dimensions == 3:
            up = nn.ConvTranspose3d(in_channels, out_channels, kernel, stride=stride, padding=padding)
    if dimensions == 2:
        bn_layer = nn.BatchNorm2d(out_channels, eps = 1e-3, momentum = 0.01)
    elif dimensions == 3:
        bn_layer = nn.BatchNorm3d(out_channels, eps = 1e-3, momentum = 0.01)
    return nn.Sequential(
        up,
        bn_layer,
        activation(),
    )
    
def conv2d_block(in_channels, out_channels, opt, kernel=3, stride=1, padding=1, activation=ACTIVATION, dimensions = 2):
    '''
    returns a block conv-activation
    '''
    conv_operation = nn.Conv2d if (dimensions == 2) else nn.Conv3d
    return nn.Sequential(
        conv_operation(in_channels, out_channels, kernel, stride=stride, padding=padding),
        activation(),
    )

#used for 2D datasets. It has 4 downsampling operations
class UNet(nn.Module):
    def __init__(self, opt, n_channels=1, n_classes=1, nf=16, batch_norm=True):
        super().__init__()

        def output_flow_delta(x14):
            delta_x = x14[:,:-opt.n_dimensions_data,...]
            flow_x = opt.constant_to_multiply_flow*x14[:,-opt.n_dimensions_data:,...]
            return delta_x, flow_x

        def output_delta(x14):
            return x14, None

        self.dimensions = opt.n_dimensions_data
        conv_block = conv2d_bn_block if batch_norm else conv2d_block
        max_pool = nn.MaxPool2d(2) if int(self.dimensions)==2 else nn.MaxPool3d(2)
        act = torch.nn.ReLU
        self.down0 = nn.Sequential(
            conv_block(n_channels, nf, opt, activation=act, dimensions = opt.n_dimensions_data),
            conv_block(nf, nf, opt, activation=act, dimensions = opt.n_dimensions_data)
        )
        self.down1 = nn.Sequential(
            max_pool,
            conv_block(nf, 2*nf, opt,activation=act, dimensions = opt.n_dimensions_data),
            conv_block(2*nf, 2*nf, opt,activation=act, dimensions = opt.n_dimensions_data),
        )
        self.down2 = nn.Sequential(
            max_pool,
            conv_block(2*nf, 4*nf, opt,activation=act, dimensions = opt.n_dimensions_data),
            conv_block(4*nf, 4*nf, opt,activation=act, dimensions = opt.n_dimensions_data),
        )
        self.down3 = nn.Sequential(
            max_pool,
            conv_block(4*nf, 8*nf, opt,activation=act, dimensions = opt.n_dimensions_data),
            conv_block(8*nf, 8*nf, opt,activation=act, dimensions = opt.n_dimensions_data),
        )

        multiplier_channels_up = 2

        self.down4 = nn.Sequential(
            max_pool,
        )

        self.bottleneck = nn.Sequential(
            conv_block(8*nf+3, 16*nf, opt, activation=act, dimensions = opt.n_dimensions_data),
            conv_block(16*nf, 16*nf, opt, activation=act, dimensions = opt.n_dimensions_data),
        )
        self.up4 = deconv2d_bn_block(multiplier_channels_up*8*nf, 8*nf, opt, activation=act, dimensions = opt.n_dimensions_data)
        
        self.conv4 = nn.Sequential(
            conv_block(2*8*nf, 8*nf, opt, activation=act, dimensions = opt.n_dimensions_data),
            conv_block(8*nf, 8*nf, opt, activation=act, dimensions = opt.n_dimensions_data),
        )
        self.up3 = deconv2d_bn_block(multiplier_channels_up*4*nf, 4*nf, opt, activation=act, dimensions = opt.n_dimensions_data)

        self.conv5 = nn.Sequential(
            conv_block(2*4*nf, 4*nf, opt, activation=act, dimensions = opt.n_dimensions_data),
            conv_block(4*nf, 4*nf, opt, activation=act, dimensions = opt.n_dimensions_data),
        )
        self.up2 = deconv2d_bn_block(multiplier_channels_up*2*nf, 2*nf, opt, activation=act, dimensions = opt.n_dimensions_data)

        self.conv6 = nn.Sequential(
            conv_block(2*2*nf, 2*nf, opt, activation=act, dimensions = opt.n_dimensions_data),
            conv_block(2*nf, 2*nf, opt, activation=act, dimensions = opt.n_dimensions_data),
        )
        self.up1 = deconv2d_bn_block(multiplier_channels_up*nf, nf, opt, activation=act, dimensions = opt.n_dimensions_data)

        self.conv7 = nn.Sequential(
            conv_block(2*nf, nf, opt, activation=act, dimensions = opt.n_dimensions_data),
            conv_block(nf, n_classes, opt, activation=Identity, dimensions = opt.n_dimensions_data),
        )
        self.preprocess_function =  (lambda x: x.float())
        self.output_function = output_flow_delta if opt.generator_output=='flow' else output_delta

    def forward(self, x, desired_output, groundtruth_output):
        assert(desired_output.size()==groundtruth_output.size())
        desired_output = self.preprocess_function(desired_output)
        groundtruth_output = self.preprocess_function(groundtruth_output)

        x0 = self.down0(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        desired_output = desired_output.view((-1,1) + (1,)*(len(x4.size())-2)).expand((x4.size(0),1)+x4.size()[2:])
        groundtruth_output = groundtruth_output.view((-1,1) + (1,)*(len(x4.size())-2)).expand((x4.size(0),1)+x4.size()[2:])
        assert(desired_output.shape[0]==x4.shape[0])
        assert(desired_output.shape[2]==x4.shape[2])
        assert(desired_output.shape[3]==x4.shape[3])
        x5 = torch.cat([x4, desired_output, groundtruth_output, (groundtruth_output - desired_output)], dim = 1)

        x6 = self.bottleneck(x5)

        xc6 = x6

        xu4 = self.up4(xc6)
        cat3 = crop_and_concat(xu4, x3)
        x10 = self.conv4(cat3)
        xc10 = x10
        xu3 = self.up3(xc10)
        cat3 = crop_and_concat(xu3, x2)
        x11 = self.conv5(cat3)
        xc11 = x11
        xu2 = self.up2(xc11)
        cat2 = crop_and_concat(xu2, x1)
        x12 = self.conv6(cat2)
        xc12 = x12
        xu1 = self.up1(xc12)
        cat1 = crop_and_concat(xu1, x0)
        x13 = self.conv7(cat1)
        x14 = self.output_function(x13)
        return x14

#used for 3D datasets, to limit the GPU memory use. It has 3 downsampling operations
class smallUNet_2(nn.Module):
    def __init__(self, opt, n_channels=1, n_classes=1, nf=16, batch_norm=True):
        super().__init__()

        def output_flow_delta(x14):
            delta_x = x14[:,:-opt.n_dimensions_data,...]
            flow_x = opt.constant_to_multiply_flow*x14[:,-opt.n_dimensions_data:,...]
            return delta_x, flow_x

        def output_delta(x14):
            return x14, None

        self.dimensions = opt.n_dimensions_data
        conv_block = conv2d_bn_block if batch_norm else conv2d_block
        max_pool = nn.MaxPool2d(2) if int(self.dimensions)==2 else nn.MaxPool3d(2)
        act = torch.nn.ReLU

        self.down0 = nn.Sequential(
            conv_block(n_channels, nf, opt, activation=act, dimensions = opt.n_dimensions_data),
            conv_block(nf, nf, opt, activation=act, dimensions = opt.n_dimensions_data)
        )
        self.down1 = nn.Sequential(
            max_pool,
            conv_block(nf, 2*nf, opt, activation=act, dimensions = opt.n_dimensions_data),
            conv_block(2*nf, 2*nf, opt, activation=act, dimensions = opt.n_dimensions_data),
        )
        self.down2 = nn.Sequential(
            max_pool,
            conv_block(2*nf, 4*nf, opt, activation=act, dimensions = opt.n_dimensions_data),
            conv_block(4*nf, 4*nf, opt, activation=act, dimensions = opt.n_dimensions_data),
        )

        multiplier_channels_up = 2
        up_block =deconv2d_bn_block
        self.down3 = nn.Sequential(
            max_pool,
        )

        self.bottleneck = nn.Sequential(
            conv_block(4*nf+3, 8*nf, opt, activation=act, dimensions = opt.n_dimensions_data),
            conv_block(8*nf, 8*nf, opt, activation=act, dimensions = opt.n_dimensions_data),
        )
        self.up3 = up_block(multiplier_channels_up*4*nf, 4*nf, opt, activation=act, dimensions = opt.n_dimensions_data)

        self.conv5 = nn.Sequential(
            conv_block(8*nf, 4*nf, opt, activation=act, dimensions = opt.n_dimensions_data),
            conv_block(4*nf, 4*nf, opt, activation=act, dimensions = opt.n_dimensions_data),
        )
        self.up2 = up_block(multiplier_channels_up*2*nf, 2*nf, opt, activation=act, dimensions = opt.n_dimensions_data)

        self.conv6 = nn.Sequential(
            conv_block(4*nf, 2*nf, opt, activation=act, dimensions = opt.n_dimensions_data),
            conv_block(2*nf, 2*nf, opt, activation=act, dimensions = opt.n_dimensions_data),
        )
        self.up1 = up_block(multiplier_channels_up*nf, nf, opt, activation=act, dimensions = opt.n_dimensions_data)

        self.conv7 = nn.Sequential(
            conv_block(2*nf, nf, opt, activation=act, dimensions = opt.n_dimensions_data),
            conv_block(nf, n_classes, opt, activation=Identity, dimensions = opt.n_dimensions_data),
        )
        self.preprocess_function = lambda x: x.float()
        self.output_function = output_flow_delta if opt.generator_output=='flow' else output_delta

    def forward(self, x, desired_output, groundtruth_output):
        assert(desired_output.size()==groundtruth_output.size())
        desired_output = self.preprocess_function(desired_output)
        groundtruth_output = self.preprocess_function(groundtruth_output)

        x0 = self.down0(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = x3
        desired_output = desired_output.view((-1,1) + (1,)*(len(x4.size())-2)).expand((x4.size(0),1)+x4.size()[2:])
        groundtruth_output = groundtruth_output.view((-1,1) + (1,)*(len(x4.size())-2)).expand((x4.size(0),1)+x4.size()[2:])
        assert(desired_output.shape[0]==x4.shape[0])
        assert(desired_output.shape[2]==x4.shape[2])
        assert(desired_output.shape[3]==x4.shape[3])
        x5 = torch.cat([x4, desired_output, groundtruth_output, (groundtruth_output - desired_output)], dim = 1)

        x6 = self.bottleneck(x5)

        xu3 = self.up3(x6)
        cat3 = crop_and_concat(xu3, x2)
        x11 = self.conv5(cat3)
        xu2 = self.up2(x11)
        cat2 = crop_and_concat(xu2, x1)
        x12 = self.conv6(cat2)
        xu1 = self.up1(x12)
        cat1 = crop_and_concat(xu1, x0)
        x13 = self.conv7(cat1)
        x14 = self.output_function(x13)

        return x14