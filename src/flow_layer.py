"""Deformation field layer
Defines the pytorch layer used to apply a deformation field to an image

by Ricardo Bigolin Lanfredi
-Last modified: 2020-07-01
Project: DeFI-GAN
GNU General Public License v3.0
"""

import torch
import torch.nn as nn

def permute_channel_last(x):
    return x.clone().permute((0,) + tuple(range(2,len(x.size()))) + (1,))

#Defining eq. 1  from the paper
def deform_to_grid(x, flow):
    flow_ = flow.clone()
    
    #constant multiplication so that flows of -1 and 1 represent a change of one pixel
    n_dimensions = (x.ndim-2)
    for index_dimension in range(n_dimensions):
        flow_[:,index_dimension,...] *= 2/(x.size(n_dimensions-index_dimension+1)-1) 
    
    #create identity grid (each pixel filled with its respective index p)
    if n_dimensions==2:
        theta = torch.tensor([[[1,0,0],[0,1,0]]])
    elif n_dimensions==3:
        theta = torch.tensor([[[1,0,0,0],[0,1,0,0],[0,0,1,0]]])
    no_change_grid = torch.nn.functional.affine_grid(theta.float().to(x.device).expand((x.size(0),) + (-1,)*2), x.size())
    assert(no_change_grid.size()==permute_channel_last(flow_).size())
    
    #in each p, sampling the coordinate, given by p+phi(p), and using bilinear sampling for non-whole coordinates
    grid = no_change_grid+permute_channel_last(flow_)
    x = torch.nn.functional.grid_sample(x, grid)
    return x

# wrapper for the generator, including the u-net and the operation to unit x and xhat
class JoinModifications(nn.Module):
    def __init__(self, opt, unet, addl, stl):
        super().__init__()
        self.first_op = addl
        self.second_op = stl
        self.unet = unet
    
    def forward(self, x, y_prime, y):
        delta_x, flow_x = self.unet(x, y_prime, y)
        x_prime = self.first_op(x, delta_x)
        delta_x = x_prime - x
        x_prime_2 = self.second_op(x_prime, flow_x)
        return x_prime_2, delta_x, flow_x

#define the operation to calculate the modified image for DeFI-GAN
class SpatialTransformLayer(nn.Module):
    def __init__(self, use_flow):
        super().__init__()
        if use_flow:
            self.forward_f = deform_to_grid
        else:
            self.forward_f = lambda x,flow :x
    
    def forward(self, x, flow):
        return self.forward_f(x,flow)

#define the operation to calculate the modified image for the baseline
class AdditionLayer(nn.Module):
    def __init__(self, use_addition):
        super().__init__()
        if use_addition:
            self.forward_f = lambda x,map :x+map
        else:
            self.forward_f = lambda x,map :x
    
    def forward(self, x, map):
        assert(x.size()==map.size())
        return self.forward_f(x,map)


