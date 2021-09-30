# The MIT License (MIT)
#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import argparse
from datetime import datetime
import glob
import os
import subprocess
import sys
import pprint
import logging as log
import multiprocessing

import matplotlib.pyplot
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from lib.datasets import *
from lib.diffutils import positional_encoding, gradient
from lib.models import *
from lib.renderer import Renderer
from lib.tracer import *
from lib.utils import PerfTimer, image_to_np, suppress_output
from lib.validator import *

class Trainer(object):
    """
    Base class for the trainer.

    The default overall flow of things:

    init()
    |- set_dataset()
    |- set_network()
    |- set_optimizer()
    |- set_renderer()
    |- set_logger()

    train():
        for every epoch:
            pre_epoch()

            iterate()
                step()

            post_epoch()
            |- log_tb()
            |- render_tb()
            |- save_model()
            |- resample()

            validate()

    Each of these submodules can be overriden, or extended with super().

    """

    #######################
    # __init__
    #######################
    
    def __init__(self, args, args_str):
        """Constructor.
        
        Args:
            args (Namespace): parameters
            args_str (str): string representation of all parameters
            model_name (str): model nametag
        """
        #torch.multiprocessing.set_start_method('spawn')
        multiprocessing.set_start_method('spawn')

        self.args = args 
        self.args_str = args_str
        
        self.args.epochs += 1

        self.timer = PerfTimer(activate=self.args.perf)
        self.timer.reset()
        
        # Set device to use
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        device_name = torch.cuda.get_device_name(device=self.device)
        log.info(f'Using {device_name} with CUDA v{torch.version.cuda}')

        self.latents = None
        
        # In-training variables
        self.train_data_loader = None
        self.val_data_loader = None
        self.dataset_size = None
        self.log_dict = {}

        # Initialize
        self.set_dataset()
        self.timer.check('set_dataset')
        self.set_network()
        self.timer.check('set_network')
        #self.set_dataset()
        #self.timer.check('set_dataset')
        self.set_optimizer()
        self.timer.check('set_optimizer')
        self.set_renderer()
        self.timer.check('set_renderer')
        self.set_logger()
        self.timer.check('set_logger')
        self.set_validator()
        self.timer.check('set_validator')
        
    #######################
    # __init__ helper functions
    #######################

    def set_dataset(self):
        """
        Override this function if using a custom dataset.  
        By default, it provides 2 datasets: 

            AnalyticDataset
            MeshDataset
        
        The code uses the mesh dataset by default, unless --analytic is specified in CLI.
        """

        self.train_dataset = globals()[self.args.mesh_dataset](self.args)

        log.info("Dataset Size: {}".format(len(self.train_dataset)))
        
        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, 
                                            shuffle=True, pin_memory=True, num_workers=0)
        self.timer.check('create_dataloader')
        log.info("Loaded mesh dataset")
            
    def set_network(self):
        """
        Override this function if using a custom network, that does not use the default args based
        initialization, or if you need a custom network initialization scheme.
        """
        self.net = globals()[self.args.net](self.args)
        if self.args.jit:
            self.net = torch.jit.script(self.net)

        if self.args.pretrained:
            self.net.load_state_dict(torch.load(self.args.pretrained))

        self.net.to(self.device)

        log.info("Total number of parameters: {}".format(sum(p.numel() for p in self.net.parameters())))

    def set_optimizer(self):
        """
        Override this function to use custom optimizers. (Or, just add things to this switch)
        """

        # Set geometry optimizer
        if self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=0.8)
        else:
            raise ValueError('Invalid optimizer.')

    def set_renderer(self):
        """
        Override this function to use custom renderers.
        """
        # Renderer for logging
        self.log_tracer = globals()[self.args.tracer](self.args)
        self.renderer = Renderer(self.log_tracer, args=self.args)

    def set_logger(self):
        """
        Override this function to use custom loggers.
        """
        if self.args.exp_name:
            self.log_fname = self.args.exp_name
        else:
            self.log_fname = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        self.log_dir = os.path.join(self.args.logs, self.log_fname)
        self.writer = SummaryWriter(self.log_dir, purge_step=0)
        self.writer.add_text('Parameters', self.args_str)

        log.info('Model configured and ready to go')

    def set_validator(self):
        """
        Override this function to use custom validators.
        """
        if self.args.validator is not None:
            self.validator = globals()[self.args.validator](self.args, self.device, self.net)

    #######################
    # pre_epoch
    #######################

    def pre_epoch(self, epoch):
        """
        Override this function to change the pre-epoch preprocessing.
        This function runs once before the epoch.
        """

        # The DataLoader is refreshed befored every epoch, because by default, the dataset refreshes
        # (resamples) after every epoch.

        self.loss_lods = list(range(0, self.args.num_lods))
        if self.args.grow_every > 0:
            self.grow(epoch)
        
        if self.args.only_last:
            self.loss_lods = self.loss_lods[-1:]

        if epoch % self.args.resample_every == 0:
            self.resample(epoch)
            log.info("Reset DataLoader")
            self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, 
                                                    shuffle=True, pin_memory=True, num_workers=0)
            self.timer.check('create_dataloader')

        if epoch == self.args.freeze:
            log.info('Freezing network...')
            log.info("Total number of parameters: {}".format(sum(p.numel() for p in self.net.parameters())))
            self.net.freeze()
            self.net.to(self.device)

        self.net.train()
        
        # Initialize the dict for logging
        self.log_dict['l2_loss'] = 0
        self.log_dict['total_loss'] = 0
        self.log_dict['total_iter_count'] = 0

        self.timer.check('pre_epoch done')

    def grow(self, epoch):
        stage = min(self.args.num_lods, (epoch // self.args.grow_every) + 1) # 1 indexed

        if self.args.growth_strategy == 'onebyone':
            self.loss_lods = [stage-1]
        elif self.args.growth_strategy == 'increase':
            self.loss_lods = list(range(0, stage))
        elif self.args.growth_strategy == 'shrink':
            self.loss_lods = list(range(0, self.args.num_lods))[stage-1:] 
        elif self.args.growth_strategy == 'finetocoarse':
            self.loss_lods = list(range(0, self.args.num_lods))[self.args.num_lods-stage:] 
        elif self.args.growth_strategy == 'onlylast':
            self.loss_lods = list(range(0, self.args.num_lods))[-1:] 
        else:
            raise NotImplementedError


    #######################
    # iterate
    #######################b

    def iterate(self, epoch):
        """
        Override this if there is a need to override the dataset iteration.
        """
        for n_iter, data in enumerate(self.train_data_loader):
            self.step_geometry(epoch, n_iter, data)
    
    #######################
    # step
    #######################

    def step_geometry(self, epoch, n_iter, data):
        """
        Override this function to change the per-iteration behaviour.
        """
        idx = n_iter + (epoch * self.dataset_size)
        log_iter = (idx % 100 == 0)

        # Map to device

        pts = data[0].to(self.device)
        gts = data[1].to(self.device)
        nrm = data[2].to(self.device) if self.args.get_normals else None

        # Prepare for inference
        batch_size = pts.shape[0]
        self.net.zero_grad()

        # Calculate loss
        loss = 0

        l2_loss = 0.0
        _l2_loss = 0.0

        preds = []
        if self.args.return_lst:
            preds = self.net.sdf(pts, return_lst=self.args.return_lst)
            preds = [preds[i] for i in self.loss_lods]
        else:
            for i, lod in enumerate(self.loss_lods):
                preds.append(self.net.sdf(pts, lod=lod))

        for pred in preds:
            _l2_loss = ((pred - gts)**2).sum()
            l2_loss += _l2_loss

        loss += l2_loss

        # Update logs
        self.log_dict['l2_loss'] += _l2_loss.item()
        self.log_dict['total_loss'] += loss.item()
        self.log_dict['total_iter_count'] += batch_size

        loss /= batch_size

        # Backpropagate
        loss.backward()
        self.optimizer.step()
    
    #######################
    # post_epoch
    #######################
    
    def post_epoch(self, epoch):
        """
        Override this function to change the post-epoch post processing.

        By default, this function logs to Tensorboard, renders images to Tensorboard, saves the model,
        and resamples the dataset.

        To keep default behaviour but also augment with other features, do 
          
          super().post_epoch(self, epoch)

        in the derived method.
        """
        self.net.eval()
            
        self.log_tb(epoch)
        if epoch % self.args.save_every == 0:
            self.save_model(epoch)
        if epoch % self.args.render_every == 0:
            self.render_tb(epoch)

        self.timer.check('post_epoch done')
    
    #######################
    # post_epoch helper functions
    #######################

    def log_tb(self, epoch):
        """
        Override this function to change loss logging.
        """
        # Average over iterations

        log_text = 'EPOCH {}/{}'.format(epoch+1, self.args.epochs)
        self.log_dict['total_loss'] /= self.log_dict['total_iter_count'] + 1e-6
        log_text += ' | total loss: {:>.3E}'.format(self.log_dict['total_loss'])

        self.log_dict['l2_loss'] /= self.log_dict['total_iter_count'] + 1e-6
        log_text += ' | l2 loss: {:>.3E}'.format(self.log_dict['l2_loss'])
        
        self.writer.add_scalar('Loss/l2_loss', self.log_dict['l2_loss'], epoch)

        log.info(log_text)

        # Log losses
        self.writer.add_scalar('Loss/total_loss', self.log_dict['total_loss'], epoch)

    def render_tb(self, epoch):
        """
        Override this function to change render logging.
        """
        self.net.eval()
        for d in range(self.args.num_lods):
            self.net.lod = d
            out = self.renderer.shade_images(self.net,
                                             f=self.args.camera_origin, 
                                             t=self.args.camera_lookat,
                                             fov=self.args.camera_fov).image().byte().numpy()
            self.writer.add_image(f'Depth/{d}', out.depth.transpose(2,0,1), epoch)
            self.writer.add_image(f'Hit/{d}', out.hit.transpose(2,0,1), epoch)
            self.writer.add_image(f'Normal/{d}', out.normal.transpose(2,0,1), epoch)
            self.writer.add_image(f'RGB/{d}', out.rgb.transpose(2,0,1), epoch)
            out_x = self.renderer.sdf_slice(self.net, dim=0)
            out_y = self.renderer.sdf_slice(self.net, dim=1)
            out_z = self.renderer.sdf_slice(self.net, dim=2)
            self.writer.add_image(f'Cross-section/X/{d}', image_to_np(out_x), epoch)
            self.writer.add_image(f'Cross-section/Y/{d}', image_to_np(out_y), epoch)
            self.writer.add_image(f'Cross-section/Z/{d}', image_to_np(out_z), epoch)
            self.net.lod = None
                
    def save_model(self, epoch):
        """
        Override this function to change model saving.
        """
        log_comps = self.log_fname.split('/')
        if len(log_comps) > 1:
            _path = os.path.join(self.args.model_path, *log_comps[:-1])
            if not os.path.exists(_path):
                os.makedirs(_path)

        if not os.path.exists(self.args.model_path):
            os.makedirs(self.args.model_path)

        if self.args.save_as_new:
            model_fname = os.path.join(self.args.model_path, f'{self.log_fname}-{epoch}.pth')
        else:
            model_fname = os.path.join(self.args.model_path, f'{self.log_fname}.pth')
        
        log.info(f'Saving model checkpoint to: {model_fname}')
        if self.args.save_all:
            torch.save(self.net, model_fname)
        else:
            torch.save(self.net.state_dict(), model_fname)

        if self.latents is not None:
            model_fname = os.path.join(self.args.model_path, f'{self.log_fname}_latents.pth')
            torch.save(self.latents.state_dict(), model_fname)
        
    def resample(self, epoch):
        """
        Override this function to change resampling.
        """
        self.train_dataset.resample()

    #######################
    # train
    #######################
    
    def train(self):
        """
        Override this if some very specific training procedure is needed.
        """

        if self.args.validator is not None and self.args.valid_only:
            self.validate(0)
            return

        for epoch in range(self.args.epochs):    
            self.timer.check('new epoch...')
            
            self.pre_epoch(epoch)

            if self.train_data_loader is not None:
                self.dataset_size = len(self.train_data_loader)
            
            self.timer.check('iteration start')

            self.iterate(epoch)

            self.timer.check('iterations done')

            self.post_epoch(epoch)

            if self.args.validator is not None and epoch % self.args.valid_every == 0:
                self.validate(epoch)
                self.timer.check('validate')

        self.writer.close()
    
    #######################
    # validate
    #######################

    def validate(self, epoch):
        
        val_dict = self.validator.validate(epoch, self.loss_lods)
        
        log_text = 'EPOCH {}/{}'.format(epoch, self.args.epochs)

        for k, v in val_dict.items():
            score_total = 0.0
            for lod, score in zip(self.loss_lods, v):
                self.writer.add_scalar(f'Validation/{k}/{lod}', score, epoch)
                score_total += score
            log_text += ' | {}: {:.2f}'.format(k, score_total / len(v))
        log.info(log_text)

