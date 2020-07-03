"""Output to disk management

This module provides a class that manages all the disk outputs that the DeFI-GAN 
training requires. This includes saving images, models, tensorboard files and logs.

by Ricardo Bigolin Lanfredi
Last modified: 2020-07-01
Project: DeFI-GAN
GNU General Public License v3.0
"""

import math
from tensorboardX import SummaryWriter
import logging
import os
from PIL import Image
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import shutil
from .image_preprocessing import BatchNormalizeTensorMinMax01
import nibabel
import glob 

def permute_channel_last(x):
    return x.clone().permute((0,) + tuple(range(2,len(x.size()))) + (1,))

def save_volume(filepath, numpy_array):
    nib_volume = nibabel.Nifti1Image(numpy_array, np.eye(4))
    nibabel.save(nib_volume, filepath)

def append_id(filename, suffix):
    return "{0}_{2}{1}".format(*os.path.splitext(filename) + (suffix,))

def change_extension(filename, new_extension):
    return "{0}.{2}".format(*os.path.splitext(filename) + (new_extension,))
  
def save_image(filepath, numpy_array):
    numpy_array = np.clip(numpy_array, -1, 1)
    im = Image.fromarray(((numpy_array*0.5 + 0.5)*255).astype('uint8'))
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save(filepath)

class Outputs():
    def __init__(self, opt):
        if not os.path.exists(opt.save_folder):
            os.mkdir(opt.save_folder)
        output_folder = opt.save_folder+'/'+opt.experiment+'_'+opt.timestamp
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        logging.basicConfig(filename = output_folder +'/log.txt' ,level = logging.INFO)
        self.log_configs(opt)
        self.writer = SummaryWriter(output_folder + '/tensorboard/')
        
        #get number of rows for the saved images as approximally the sqrt of the number of batch examples to save in the same image
        self.nrows_fixed = int(math.sqrt(opt.batch_size))-[(opt.batch_size%i==0) for i in range(int(math.sqrt(opt.batch_size)),0,-1)].index(True)
        self.save_model = opt.save_model
        self.arrow_scale = opt.scale_flow_arrow
        self.gap_between_arrows = opt.gap_between_arrows
        
    def log_configs(self, opt):
        logging.info('-------------------------------used configs-------------------------------')
        for key, value in sorted(vars(opt).items()):
            logging.info(key + ': ' + str(value).replace('\n', ' ').replace('\r', ''))
        logging.info('-----------------------------end used configs-----------------------------')
    
    def save_plt(self, filepath, garbage):
        plt.savefig(filepath, bbox_inches = 'tight', pad_inches = 0)        
        plt.show()
    
    def log_fixed(self,fixed_x, fixed_y, epoch =0):
        fmi = permute_channel_last(fixed_x.detach()).squeeze(-1).cpu()
        self.save_batch(self.output_folder+'/x1_'+str(epoch)+'.png', fmi)
        with open(self.output_folder+'/x1_label_'+str(epoch)+'.txt', 'w') as f:
            out_gt = np.vstack(np.hsplit(np.hstack(fixed_y.cpu()), self.nrows_fixed))
            for i in range(out_gt.shape[0]):
                f.write(str( out_gt[i,:]))
                f.write('\n')
    
    # activate the average calculation for all metric values and save them to log and tensorboard
    def log_added_values(self, epoch, metrics):
        averages = metrics.get_average()
        logging.info('Metrics for epoch: ' + str(epoch))
        for key, average in averages.items():
            self.writer.add_scalar(key, average, epoch)
            logging.info(key + ': ' + str(average))
        if ('gen_loss' in averages.keys()) and ('critic_loss' in averages.keys()):
            self.writer.add_scalar('total_loss', averages['gen_loss']+averages['critic_loss'],epoch) 
        return averages
    
    def save_models(self, net_g, net_d, suffix):
        torch.save(net_g.module.unet.state_dict(), '{:}/generator_state_dict_'.format(self.output_folder) + str(suffix)) 
        torch.save(net_d.module.state_dict(), '{:}/critic_state_dict_'.format(self.output_folder) + str(suffix))
    
    #plot the quiver plots with arrows representing deformation field directions and magnitude, in pixels
    def plot_quiver(self, flow, x):
        flow = np.vstack(np.hsplit(np.hstack(flow), self.nrows_fixed))
        x = np.vstack(np.hsplit(np.hstack(x), self.nrows_fixed))
        plt.style.use('./plotstule.mplstyle')
        plt.close('all')
        
        #creates a really big figure so that the arrow details can be seen
        fig = plt.figure(figsize=(0.054*flow.shape[1],0.054*flow.shape[0] ), dpi=560)
        fig.tight_layout(pad=0)
        fig.add_subplot(111)
        fig.canvas.draw()
        plt.gca().invert_yaxis()
        plt.axis('off')
        
        X,Y = np.meshgrid(np.arange(0, flow.shape[1]), np.arange(0, flow.shape[0]))
        U,V = flow[:,:,0], flow[:,:,1]
        C = np.hypot(U, V)
        
        # plot arrows only every 4 pixels
        gap = self.gap_between_arrows
        plt.quiver(X[::gap,::gap],Y[::gap,::gap], -U[::gap,::gap], -V[::gap,::gap], C[::gap,::gap], pivot = 'tip', units='xy', angles='xy', scale_units='xy', scale=1/self.arrow_scale, width=0.4, headlength = 3, headwidth = 9/5., headaxislength = 13.5/5)
        plt.imshow(x, cmap='gray')
    
    #save colored difference map over original image
    def plot_diff_overlay(self, diff, x):
        _, s1, s2 = x.shape
        diff = np.vstack(np.hsplit(np.hstack(diff), self.nrows_fixed))
        x = np.vstack(np.hsplit(np.hstack(x), self.nrows_fixed))
        plt.style.use('./plotstule.mplstyle')
        plt.close('all')
        fig = plt.figure(figsize=(s2/10., s1/10), dpi=100)
        fig.tight_layout(pad=0)
        fig.add_subplot(111)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        fig.canvas.draw()
        plt.gca().invert_yaxis()
        plt.axis('off')
        
        plt.imshow(x, cmap='gray')
        
        #set high transparency for regions with low difference map values 
        alphas = np.clip(np.abs(diff*4),0,1)
        
        #sets the color to be green and pink for negative and positive values, respectively
        cmap = plt.get_cmap('PiYG')
        rgba_img = cmap(1-(np.clip(diff*2.5,-1,1)*0.5+0.5))
        
        rgba_img[:,:,3] = alphas
        plt.imshow(rgba_img)
    
    def log_images(self, epoch, net_g, net_d, x_prime, delta_x, x, flow, opt):
        if self.save_model:
            self.save_models(net_g, net_d, '{:05d}'.format(epoch))
        if flow is not None:
            flow = flow.detach()
        x_img = permute_channel_last(x.detach()).squeeze(-1).cpu()
        x_prime_img = permute_channel_last(x_prime.detach()).squeeze(-1).cpu()
        if flow is not None:
            flow = permute_channel_last(flow).cpu()
        delta_x_img = permute_channel_last(delta_x).squeeze(-1).detach().cpu()
        
        if flow is not None:
            path = '{:}/flow_quiver_{:05d}.png'.format(self.output_folder, epoch)
            if len(flow.size())>4:
                self._3d_to_2d_slices([flow, x_img], path, self.save_plt, self.plot_quiver)
            else:
                self.save_plt(path, self.plot_quiver(flow, x_img))
            dimension_name = {2:'x', 1:'y', 0:'z'} if len(flow.size())>4 else {1:'x', 0:'y'}
            path = self.output_folder+'/flow_' + dimension_name[1] + '_{:05d}.png'.format(epoch)
            self.save_batch(path, BatchNormalizeTensorMinMax01()(flow[...,1]))
            path = self.output_folder+'/flow_' + dimension_name[0] + '_{:05d}.png'.format(epoch)
            self.save_batch(path, BatchNormalizeTensorMinMax01()(flow[...,0]))
            if len(flow.size())>4:
                path = self.output_folder+'/flow_' + dimension_name[2] + '_{:05d}.png'.format(epoch)
                self.save_batch(path, BatchNormalizeTensorMinMax01()(flow[...,2]))
        path = '{:}/delta_x_{:05d}.png'.format(self.output_folder, epoch)
        self.save_batch(path, delta_x_img)
        path = '{:}/difference_{:05d}.png'.format(self.output_folder, epoch)
        self.save_batch(path, permute_channel_last(x_prime.detach() - x.detach()).squeeze(-1).cpu())
        
        path = '{:}/difference_overlaid_{:05d}.png'.format(self.output_folder, epoch)
        if len(x.size())>4:
            self._3d_to_2d_slices([permute_channel_last(x_prime.detach() - x.detach()).squeeze(-1).cpu(), x_img], path, self.save_plt, self.plot_diff_overlay)
        else:
            self.save_plt(path, self.plot_diff_overlay(permute_channel_last(x_prime.detach() - x.detach()).squeeze(-1).cpu(), x_img))
                    
        path = '{:}/xhat0_{:05d}.png'.format(self.output_folder, epoch)
        self.save_batch(path, x_prime_img)

    def log_batch(self, epoch, batch_index, metric):
        try:
            gen_loss = metric.get_last_added_value('gen_loss').item()
            gen_loss_defined = True
        except IndexError:
            gen_loss = 0
            gen_loss_defined = False
        try: 
            critic_loss = metric.get_last_added_value('critic_loss').item()
            critic_loss_defined = True
        except IndexError:
            critic_loss = 0
            critic_loss_defined = False
        loss_string = {(True, True):"Total", (True, False):"Generator", (False, True):"Critic", (False, False): "Undefined"}[(gen_loss_defined, critic_loss_defined)]
        loss_string = "; " + loss_string + " loss: "
        logging.info('Epoch: ' + str(epoch) + '; Batch ' + str(batch_index) + loss_string + str(gen_loss + critic_loss))
        
    def log_delta_x_gt(self, delta_x_gt):
        path = '{:}/difference_gt'.format(self.output_folder)+'.png'
        self.save_batch(path, permute_channel_last(delta_x_gt).squeeze(-1).detach().cpu())
    
    #save the source files used to run this experiment
    def save_run_state(self, py_folder_to_save):
        if not os.path.exists('{:}/src/'.format(self.output_folder)):
            os.mkdir('{:}/src/'.format(self.output_folder))
        [shutil.copy(filename, ('{:}/src/').format(self.output_folder)) for filename in glob.glob(py_folder_to_save + '/*.py')]
        self.save_command()
    
    #saves the command line command used to run these experiments
    def save_command(self, command = None):
        if command is None:
            command = ' '.join(sys.argv)
        with open("{:}/command.txt".format(self.output_folder), "w") as text_file:
            text_file.write(command)
    
    #saves 2d slices for  the 3D volumes of the ADNI dataset
    def _3d_to_2d_slices(self, datas_, filepath, main_fn, aux_fn = lambda x: x):
        parameters_ = {'xslice':{'slice':40, 'index_dimension':1,'perpendicular_flow_channel_index':[0,1]}, 
                        'yslice':{'slice':83,'index_dimension':2,'perpendicular_flow_channel_index':[0,2]}, 
                        'zslice':{'slice':56,'index_dimension':3,'perpendicular_flow_channel_index':[1,2]}}

        for key, element in parameters_.items(): 
            arguments_aux_fn = [data_[(slice(None),)*(element['index_dimension']) + (element['slice'],) + (slice(None),)*(3-element['index_dimension']) + ((element['perpendicular_flow_channel_index'],) if len(data_.size())>4 else tuple())] for data_ in datas_]
            main_fn(append_id(filepath, key) , aux_fn(*arguments_aux_fn))
                    
    def save_batch(self, filepath, tensor):
        numpy_array = tensor
        if len(numpy_array.shape)>3 and numpy_array.shape[3]>3:
            [save_volume(append_id(change_extension(filepath, 'nii'), 'example_' + str(i)) , numpy_array[i,...].detach().cpu().numpy()) for i in range(1)]#range(numpy_array.shape[0])]
            self._3d_to_2d_slices([numpy_array], filepath, self.save_batch)
            return
        try:
            numpy_array = numpy_array.numpy()
        except:
            pass
        numpy_array = np.vstack(np.hsplit(np.hstack(numpy_array), self.nrows_fixed))
        save_image(filepath, numpy_array)
    