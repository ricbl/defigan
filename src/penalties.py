"""Penalty functions
Functions for calculating penalties for the generator and the critic

by Ricardo Bigolin Lanfredi
Last modified: 2020-07-01
Project: DeFI-GAN
GNU General Public License v3.0
"""

import torch

# adapted from https://github.com/caogang/wgan-gp
# provides L_{RegD}, eq. 3 in the paper
def calc_gradient_penalty(critic, fake_data, real_data):
    device = 'cuda:0' if torch.cuda.is_available() and real_data.is_cuda else 'cpu'
    batch_size = real_data.size(0)

    alpha = torch.rand(batch_size, 1, device = device).float()
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous().view(real_data.size())
    
    assert(real_data.size()==fake_data.size())
    assert(alpha.size()==fake_data.size())
    interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).clone().detach().requires_grad_(True)
    interpolates = interpolates.to(device)
    output_interpolates = critic(interpolates)
    gradient_penalty = []
    for i in range(output_interpolates.size(1)):
        gradients = torch.autograd.grad(outputs=output_interpolates[:, i], inputs=interpolates,
                                  grad_outputs=torch.ones(output_interpolates[:, i].size()).float().to(device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty.append(((gradients.norm(2, dim=1) - 1) ** 2).mean())
    return sum(gradient_penalty)

def get_staetv(flow_x):
    # from Spatially Transformed Adversarial Examples
    l_reg_flow = torch.tensor(0.).cuda().float()
    neighbors = []
    total_dimensions = len(flow_x.size())-2
    for index_dim in range(total_dimensions):
        to_the_end = (total_dimensions-1-index_dim)
        slices_by_side = [(slice(0,-1),), (slice(1,None),)]
        paddings_by_side = [(1,0), (0,1)]
        for side in range(2):
            flow_slicing_tuple = (slice(None),)*(2+index_dim) + slices_by_side[side] + (slice(None),)*to_the_end
            paddings = (0,0)*to_the_end + paddings_by_side[side] + (0,0)*index_dim
            neighbors.append(torch.nn.functional.pad(flow_x[flow_slicing_tuple], paddings))
    for neighbor in neighbors:
        flow_gradients = torch.abs(flow_x - neighbor)
        l_reg_flow += (((flow_gradients+torch.tensor(1e-10).cuda().float())**2).sum(dim = 1)**0.5).view([flow_gradients.size(0), -1]).mean(dim = 1).mean()
    return l_reg_flow
            
def get_penalties(delta_x, flow_x, opt, metric):
    if opt.generator_output=='flow':
        #Eq. 5 in the paper, L_{regG}
        l_regg = get_staetv(flow_x)
    elif opt.generator_output=='residual':
        # in the case of training the baseline,
        # penalizes the norm of the difference map
        l_regg = (torch.abs(delta_x)).mean()
    metric.add_value('l_regg', l_regg)
    return l_regg