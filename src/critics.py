"""Critic Architectures

Originally from https://github.com/orobix/Visual-Feature-Attribution-Using-Wasserstein-GANs-Pytorch
Modified by Ricardo Bigolin Lanfredi
Last modified: 2020-07-01
Project: DeFI-GAN
GNU General Public License v3.0
"""

import torch.nn as nn
from .model_utils import conv2d_block, conv2d_bn_block, conv3d_block, Identity
import torchvision
import torch
import types
from .image_preprocessing import BatchNormalizeTensor
from collections import OrderedDict

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x.view(x.size(0), -1)

class Average(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x.mean(1)[:,None]

#critic for the 3D dataset. Relatively shallow to not occupy to much GPU memory
class C3DFCN(nn.Module):
    def __init__(self, opt, n_channels=1, init_filters=16, dimensions=2, batch_norm=False):
        super(C3DFCN, self).__init__()
        nf = init_filters
        if dimensions == 2:
            conv_block = conv2d_bn_block if batch_norm else conv2d_block
        else:
            conv_block = conv3d_block
        max_pool = nn.MaxPool2d if (int(dimensions) == 2) else nn.MaxPool3d
        self.encoder = nn.Sequential(
            conv_block(n_channels, nf, opt),
            max_pool(2),
            conv_block(nf, 2*nf, opt),
            max_pool(2),
            conv_block(2*nf, 4*nf, opt),
            conv_block(4*nf, 4*nf, opt),
            max_pool(2),
            conv_block(4*nf, 8*nf, opt),
            conv_block(8*nf, 8*nf, opt),
            max_pool(2),
            conv_block(8*nf, 16*nf, opt),
            conv_block(16*nf, 16*nf, opt),
            conv_block(16*nf, 16*nf, opt),
        )
        self.classifier = nn.Sequential(
            conv_block(16*nf, 1, opt, kernel=1, activation=Identity),
            Flatten(),
            Average()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

#normalize input images with imagenet normalization values before using them 
# with the imagenet pre-trained critic
class CriticInputs(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.resnet = (not opt.n_dimensions_data==3)
        self.expand_size = [-1,3] + [-1] * opt.n_dimensions_data
        self.view_norm_size = [1,3] + [1] * opt.n_dimensions_data
        
        
    def forward(self, x):
        x = x/2+0.5
        if self.resnet:
            x = x.expand(self.expand_size).clone()
            self.bnt = BatchNormalizeTensor(torch.FloatTensor([0.485, 0.456, 0.406]).to(x.device).view(self.view_norm_size), 
                    torch.FloatTensor([0.229, 0.224, 0.225]).to(x.device).view(self.view_norm_size))
            x = self.bnt(x)
        return x

class CriticWithPreprocessing(torch.nn.Module):
    def __init__(self, original_model, preprocessing_model):
        super().__init__()
        self.original_model = original_model
        self.preprocessing_model = preprocessing_model
    
    def forward(self, x):
        x = self.preprocessing_model(x)
        x = self.original_model(x)
        return x

def get_critic(opt):
    if opt.n_dimensions_data==3:
        net_d = C3DFCN(opt, dimensions=opt.n_dimensions_data, n_channels = (1))
    else:
        net_d = torchvision.models.resnet18(pretrained = True)
        net_d.fc = torch.nn.Linear(in_features = net_d.fc.in_features, out_features = 1)
    if opt.load_checkpoint_d is not None:
        state_dict = torch.load(opt.load_checkpoint_d)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.original_model.','')
            new_state_dict[name]=v
        net_d.load_state_dict(new_state_dict)
    net_d_preprocessing = CriticWithPreprocessing(net_d, CriticInputs(opt))
    #following Baumgartner et al. (2018), turning off batch normalization 
    # on the critic
    def train(self, mode = True):
        print('Turning off batch normalization')
        super(type(net_d_preprocessing), self).train(mode)
        for module in self.modules():
            if isinstance(module, torch.nn.modules.BatchNorm2d) or \
            isinstance(module, torch.nn.modules.BatchNorm3d):
                module.eval()
    net_d_preprocessing.train = types.MethodType(train, net_d_preprocessing)
    return net_d_preprocessing