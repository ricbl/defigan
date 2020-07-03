"""Test for baseline models trained with Tensorflow
used to get the final scores and images

by Ricardo Bigolin Lanfredi
Last modified: 2020-06-26
Project: DeFI-GAN
GNU General Public License v3.0
"""
from . import adni_experiment as exp_config

import torch
import os
import numpy as np
import logging
from .. import opts 
from .. import brain_loader as dataloaders
from .. import outputs
from .. import metrics
from .model_vagan_inference_only import vagan

#calculate the ncc values of a tensorflow model for the whole validation set
def main(opt, gan_log_dir, vagan_model = None, epoch=0):
    if opt.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus
    get_dataloaders = dataloaders.get_dataloaders
    loader_val_disease, abnormal_dataset = get_dataloaders(opt, mode=opt.split_validation)
    output = outputs.Outputs(opt)
    metric = metrics.Metrics(opt)
    output.save_command()
    if vagan_model is None:
        vagan_model = vagan(exp_config=exp_config)
        vagan_model.load_weights(gan_log_dir, type='best_ncc')
    if not opt.do_visual_validation:
        fixed_x, fixed_y, fixed_gt_delta_x ,_, _,_,_ = iter(loader_val_disease).next()
    else:
        fixed_x, fixed_y = abnormal_dataset[opt.index_vis]
        fixed_x = fixed_x[None].cuda()
        fixed_y = fixed_y[None].cuda()
    output.log_fixed(fixed_x, fixed_y, 'fixed')
    mask = vagan_model.predict_mask(fixed_x.permute([0,2,3,4,1]).cpu().numpy())
    x_prime = fixed_x.permute([0,2,3,4,1]).cpu().numpy() + mask
    
    if exp_config.use_tanh:
        x_prime = np.tanh(x_prime)
    
    output.log_images(epoch, None, None, torch.tensor(x_prime).permute([0,4,1,2,3]).cuda(), torch.tensor(x_prime - fixed_x.permute([0,2,3,4,1]).cpu().numpy()).permute([0,4,1,2,3]).cuda(), fixed_x.cuda(), None, opt)
    if not opt.do_visual_validation:
        for batch_index, batch_example in enumerate(loader_val_disease):
              print(batch_index)
              #get validation data
              x, y, gt_delta_x , _, weight, x_id, mask_im_diff = batch_example
              mask = vagan_model.predict_mask(x.permute([0,2,3,4,1]).cpu().numpy())
              x_prime = x.permute([0,2,3,4,1]).cpu().numpy() + mask
              if exp_config.use_tanh:
                  x_prime = np.tanh(x_prime)
              #calculate normalized cross correlation
              metric.add_ncc(gt_delta_x.cpu(), torch.tensor(x_prime - x.permute([0,2,3,4,1]).cpu().numpy()).permute([0,4,1,2,3]).cpu(), weight, mask_im_diff)
        to_return = output.log_added_values(epoch, metric)['ncc']
        return to_return

def get_opt_from_command(command):
    opt = opts.get_opt(command.split(' '))
    output_folder = opt.save_folder+'/'+opt.experiment+'_'+opt.timestamp
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    logging.basicConfig(filename = output_folder +'/log.txt' ,level = logging.INFO)
    return opt

if __name__=='__main__':
    opt = opts.get_opt()
    output_folder = opt.save_folder+'/'+opt.experiment+'_'+opt.timestamp
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    logging.basicConfig(filename = output_folder +'/log.txt' ,level = logging.INFO)
    
    gan_log_dir = os.path.split(opt.load_checkpoint_g)[0]
    main(opt,gan_log_dir, epoch=0)