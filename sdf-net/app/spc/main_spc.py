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

import os
import sys
import numpy as np
import torch

from NeuralSPC import NeuralSPC
from SPCDataset import SPCDataset
from SPCTracer import SPCTracer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import argparse
import glob
import logging as log
import multiprocessing

import torch.optim as optim
from torch.utils.data import DataLoader

from lib.trainer import Trainer
from lib.options import parse_options, argparse_to_str
from lib.utils import image_to_np, PerfTimer
from lib.renderer import Renderer

class SPCTrainer(Trainer):
    
    def __init__(self, args, args_str):
        
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
        self.set_network()
        self.timer.check('set_network')
        self.set_dataset()
        self.timer.check('set_dataset')
        self.set_optimizer()
        self.timer.check('set_optimizer')
        self.set_renderer()
        self.timer.check('set_renderer')
        self.set_logger()
        self.timer.check('set_logger')
        self.set_validator()
        self.timer.check('set_validator')
    
        
    def pre_epoch(self, epoch):
        self.epoch_per_block = self.args.epochs // len(self.block_idxes)
        self.active_block_idx = min(epoch // self.epoch_per_block, len(self.block_idxes)-1)
        
        active_block = self.block_idxes[self.active_block_idx]
        if epoch % self.epoch_per_block == 0:
            self.train_dataset.init(active_block)

        log.info(f"Active Block IDX: {active_block}")

        super().pre_epoch(epoch)

        if self.args.grow_every > 0 and \
            epoch % self.args.grow_every == 0 and \
            (epoch // self.args.grow_every) < self.args.num_lods and \
            epoch > 0:

            self.set_optimizer()

    def set_dataset(self):
        self.train_dataset = SPCDataset(self.net, args=self.args)
        self.block_idxes = self.train_dataset.get_block_idxes(lod=self.args.num_lods-1)
        log.info(f"Block Indices: {self.block_idxes}")
    
    def set_network(self):
        self.net = NeuralSPC(self.args)

        if self.args.pretrained:
            self.net.load_state_dict(torch.load(self.args.pretrained))

        self.net.to(self.device)

        log.info("Total number of parameters: {}".format(sum(p.numel() for p in self.net.parameters())))
    
    def set_renderer(self):
        self.log_tracer = SPCTracer(self.args)
        self.renderer = Renderer(self.log_tracer, args=self.args)
                                
    def step_geometry(self, epoch, n_iter, data):
        idx = n_iter + (epoch * self.dataset_size)
        log_iter = (idx % 100 == 0)

        # Map to device

        pts = data[0].to(self.device)
        gts = data[1].to(self.device)

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
        else:
            preds = [self.net.sdf(pts, lod=lod) for lod in self.loss_lods]

        for i, pred in enumerate(preds):
            res = 2**(self.args.base_lod + i)
            _l2_loss = ((pred - res * gts)**2).sum()
            l2_loss += _l2_loss

        loss += l2_loss * self.args.l2_loss
        
        loss /= batch_size

        # Update logs
        self.log_dict['l2_loss'] += _l2_loss.item()
        self.log_dict['total_loss'] += l2_loss.item()
        self.log_dict['total_iter_count'] += batch_size

        # Backpropagate
        loss.backward()
        self.optimizer.step()

    def log_tb(self, epoch):
        log_text = 'EPOCH {}/{}'.format(epoch+1, self.args.epochs)
        self.log_dict['total_loss'] /= self.log_dict['total_iter_count'] + 1e-6
        log_text += ' | total loss: {:>.3E}'.format(self.log_dict['total_loss'])
        self.log_dict['l2_loss'] /= self.log_dict['total_iter_count'] + 1e-6
        log_text += ' | l2 loss: {:>.3E}'.format(self.log_dict['l2_loss'])

        self.writer.add_scalar('Loss/l2_loss', self.log_dict['l2_loss'], epoch)
        log.info(log_text)

        # Log losses
        self.writer.add_scalar('Loss/total_loss', self.log_dict['total_loss'], epoch)

    def resample(self, epoch):
        self.train_dataset.resample(max(self.loss_lods), self.block_idxes[self.active_block_idx])
        log.info("Dataset Size: {}".format(len(self.train_dataset)))

# Set logger display format
log.basicConfig(format='[%(asctime)s] [INFO] %(message)s', 
                datefmt='%d/%m %H:%M:%S',
                level=log.INFO)


if __name__ == "__main__":
    """Main program."""
    parser = parse_options(return_parser=True)
    app_group = parser.add_argument_group('app')
    app_group.add_argument('--l2-loss', type=float, default=1.0, 
                            help='Weight of standard L2 loss')
    app_group.add_argument('--mesh-path', type=str,
                            help='Path of SPC mesh')
    app_group.add_argument('--normalize-mesh', action='store_true',
                            help='Normalize the mesh')
    app_group.add_argument('--feature-std', type=float, default=0.01,
                            help='Feature initialization distribution')
    args, args_str = argparse_to_str(parser)
    log.info(f'Parameters: \n{args_str}')

    log.info(f'Training on {args.dataset_path}')
    model = SPCTrainer(args, args_str)
    model.train()

