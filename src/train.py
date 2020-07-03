"""Training a DeFI-GAN
Use this file to train and validate a DeFI-GAN model

by Ricardo Bigolin Lanfredi
Last modified: 2020-06-26
Project: DeFI-GAN
GNU General Public License v3.0
"""

import os
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_default_dtype(torch.float)
import torch.optim as optim

from .unet import UNet, smallUNet_2
from . import critics
from . import opts
from . import outputs
from . import metrics
from .flow_layer import SpatialTransformLayer, AdditionLayer, JoinModifications
from . import schedule
from . import penalties

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, mean=0.1, std=0.01)
        m.bias.data.fill_(0)

def get_unet(opt, checkpoint, output = None):
    if opt.n_dimensions_data==2:
        unet_ = UNet(opt, nf=64, n_channels = opt.n_channels, n_classes=opt.n_classes)
    elif opt.n_dimensions_data==3:
        unet_ = smallUNet_2(opt, nf=16, n_channels = opt.n_channels, n_classes=opt.n_classes)
    if checkpoint is not None:
        unet_.load_state_dict(torch.load(checkpoint))
    else:
        unet_.apply(weights_init)
    return unet_

def init_model(opt, output = None):
    net_g = get_unet(opt, opt.load_checkpoint_g, output)
    net_d = critics.get_critic(opt)
    stl = SpatialTransformLayer(opt.generator_output=='flow')
    addl = AdditionLayer(opt.generator_output=='residual')
    add_net_g = JoinModifications(opt, net_g, addl, stl)
    add_net_g = torch.nn.DataParallel(add_net_g).cuda().float()
    net_d = torch.nn.DataParallel(net_d).cuda().float()
    return add_net_g, net_d

def init_optimizer(opt, net_g, net_d):
    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.learning_rate_g, betas=(
        0.0, 0.9), weight_decay=0)
    optimizer_d = optim.Adam(net_d.parameters(), lr=opt.learning_rate_d, betas=(
            0.0, 0.9), weight_decay=0)
    return optimizer_g, optimizer_d

def train_vagan(training_selection, metric, x1, x0, y1, y0, net_g, optim_g, net_d, optim_d, opt):
    #Calculate the modified image xhat0 from Eq. 1. If doing the addition baseline, flow_x will be None. If doing DeFI-GAN, delta_x will be all zeros
    xhat0, delta_x, flow_x = net_g(x1[:opt.batch_size,...], y0[:opt.batch_size,...], y1[:opt.batch_size,...])
    
    #calculating critic score for the modified image
    d_xhat0 = net_d(xhat0)
    
    if training_selection['step_critic']:
        #calculating critic score for the original image of class 0
        d_x0 = net_d(x0)
        
        #calculate L_D from Eq. 2
        l_dx0 = d_x0[:,0].mean() 
        l_dxhat0 = - d_xhat0[:,0].mean()
        
        #calculate the Lipschitz constraint penalty from the WGAN formulation, and given in Eq. 3 in the paper
        l_regd = penalties.calc_gradient_penalty(net_d, xhat0, x0)
        
        #calculate the critic part of Eq. 6
        critic_loss = l_dx0 + l_dxhat0 + opt.lambda_regd * l_regd
        
        #update the critic
        optim_d.zero_grad()
        critic_loss.backward(retain_graph = training_selection['step_generator'])
        optim_d.step()
        
        #log the loss values
        metric.add_value('l_dx0', l_dx0)
        metric.add_value('l_dxhat0', l_dxhat0)
        metric.add_value('l_regd', l_regd)
        metric.add_value('critic_loss', critic_loss)
        
    if training_selection['step_generator']:
        #calculate L_G from Eq. 4
        l_g = d_xhat0[:,0].mean()
        
        #calculate the generator part of Eq. 6
        gen_loss = opt.lambda_regg * penalties.get_penalties(delta_x, flow_x, opt, metric) + l_g
        
        #update the generator
        optim_g.zero_grad()
        gen_loss.backward()
        optim_g.step()
        
        #log the loss values
        metric.add_value('gen_loss', gen_loss)
        metric.add_value('l_g', l_g)

def train(opt, train_loader, val_loader_disease, net_g, net_d, 
    optim_g, optim_d, output, metric, train_schedule):
    
    last_best_ncc = float("-inf")
    
    #getting a fixed set of validation images to follow the evolution of the
    # correponding output over several epochs
    fixed_x1, fixed_y1, fixed_gt_delta_x ,fixed_y0, _,_,_ = iter(val_loader_disease).next()
    output.log_fixed(fixed_x1, fixed_y1)
    fixed_x1, fixed_y1, fixed_y0, fixed_gt_delta_x = [var.cuda() for var in [fixed_x1, fixed_y1, fixed_y0, fixed_gt_delta_x]]
    output.log_delta_x_gt(fixed_gt_delta_x)
    
    for epoch_index in range(opt.nepochs):
        #advance training schedule to next epoch
        epoch_schedule = train_schedule.next()
        
        #training
        if not opt.skip_train:
            net_d.train()
            net_g.train()
            
            #iterate through all the batches in this epoch
            for batch_index, batch_example in enumerate(train_loader):
                print("Current batch index: " + str(batch_index))
                
                #get the selection of training to perform in this epoch (generator/critic)
                training_selection = epoch_schedule.next()
                
                #get inputs and labels for this batch. x is from class 1, real_x from class 0
                (x1, y1), (x0, y0) = batch_example
                x1, y1, x0, y0 = [var.cuda() for var in [x1, y1, x0, y0]]
                x1.requires_grad = True
                
                #do all the inferences and updates
                train_vagan(training_selection, metric, x1, x0, y1, y0, net_g, optim_g, net_d, optim_d, opt)
            output.log_batch(epoch_index, batch_index, metric)
        
        #validation
        if not opt.skip_val:
            with torch.no_grad():
                net_d.eval()
                net_g.eval()
                torch.set_grad_enabled(False)
                
                #saving the outputs of the fixed images (always the same images every epoch) and models
                xhat0, delta_x, flow_x = net_g(fixed_x1, fixed_y0, fixed_y1)
                output.log_images(epoch_index, net_g, net_d, xhat0, delta_x, fixed_x1, flow_x, opt)
                
                #iterating through the full validation set
                for batch_index, batch_example in enumerate(val_loader_disease):
                    #get validation data
                    x1, y1, gt_delta_x , y0, weight, _, mask_im_diff = batch_example
                    x1, y1, gt_delta_x , y0, mask_im_diff = [var.cuda() for var in [x1, y1, gt_delta_x , y0, mask_im_diff]]
                    
                    #get validation output
                    xhat0, _, _ = net_g(x1, y0, y1)
                    
                    #calculate normalized cross correlation
                    metric.add_ncc(gt_delta_x, xhat0 - x1, weight, mask_im_diff)
                torch.set_grad_enabled(True)
                net_d.train()
                net_g.train()
        
        #save the current model as the best model, if the validation score is the best so far
        this_ncc = output.log_added_values(epoch_index, metric)['ncc']
        if opt.save_best_model:
            if this_ncc>=last_best_ncc:
                output.save_models(net_g, net_d, 'best_epoch')
                last_best_ncc = this_ncc

def main():
    #get user options/configurations
    opt = opts.get_opt()
    
    #get the correct dataset/dataloader
    if opt.dataset_to_use=='xray':
        from . import xray_loader as dataloaders
    elif opt.dataset_to_use=='toy':
        from . import synth_dataset as dataloaders
    elif opt.dataset_to_use=='adni':
        from . import brain_loader as dataloaders
    get_dataloaders = dataloaders.get_dataloaders
    loader_train = get_dataloaders(opt, mode='train')
    loader_val_disease, abnormal_dataset = get_dataloaders(opt, mode=opt.split_validation)
    
    #load Outputs class to save metrics, images and models to disk
    output = outputs.Outputs(opt)
    output.save_run_state(os.path.dirname(__file__))
    
    #load class to store metrics and losses values
    metric = metrics.Metrics(opt)
    
    #load the deep learning architecture for the critic and the generator
    net_g, net_d = init_model(opt, output)
    net_g = net_g.cuda()
    net_d = net_d.cuda()
    
    #load the optimizer and training schedules
    optim_g, optim_d = init_optimizer(opt, net_g=net_g, net_d=net_d)
    train_schedule = schedule.get_schedule(opt)
    
    #if opt.do_visual_validation is true, only run an inference for image indexed by opt.index_vis and save the resulting outputs.
    #if false, executing training and validation
    if opt.do_visual_validation:
        # Load the image, label and save them to disk
        fixed_image, fixed_y = abnormal_dataset[opt.index_vis]
        fixed_image = fixed_image[None].cuda()
        fixed_y = fixed_y[None].cuda()
        output.log_fixed(fixed_image, fixed_y)
        
        #Execute one inference and save the outputs to disk
        with torch.no_grad():
            net_g.eval()
            torch.set_grad_enabled(False)
            x_prime, delta_x, flow_x = net_g(fixed_image, 1-fixed_y, fixed_y)
            output.log_images(0, net_g, net_d, x_prime, delta_x, fixed_image, flow_x, opt)
    else:
        train(opt, loader_train, loader_val_disease,
          net_g=net_g, net_d=net_d, optim_g=optim_g, optim_d=optim_d, 
          output=output, metric=metric, train_schedule = train_schedule)

if __name__ == '__main__':
    main()
