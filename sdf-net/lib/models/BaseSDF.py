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

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import *

from .BasicDecoder import BasicDecoder

from typing import Optional

from .Embedder import positional_encoding

class BaseSDF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.latents = None
        self.active_latent = None

        self.input_dim = 3
        self.out_dim = 1

        if args.latent:
            self.latents = nn.Embedding(args.mesh_subset_size, args.latent_dim)
            nn.init.normal_(self.latents.weight.data, 0.0, 0.01)

            self.input_dim += args.latent_dim

        self.pos_enc = args.pos_enc
        self.ff_dim = args.ff_dim
        if args.ff_dim > 0:
            self.gauss_matrix = nn.Parameter(torch.randn([args.ff_dim, 3]) * args.ff_width)
            self.gauss_matrix.requires_grad_(False)

            self.input_dim += (args.ff_dim * 2) - 3

        self.inputs : Optional[torch.Tensor] = torch.zeros(0)
        self.ids : Optional[torch.Tensor] = torch.zeros(0)
        self.preds : Optional[torch.Tensor] = torch.zeros(0)
        self.gts : Optional[torch.Tensor] = torch.zeros(0)
        self.grad : Optional[torch.Tensor] = torch.zeros(0)

    def forward(self, 
            x    : torch.Tensor,  
            gts  : Optional[torch.Tensor] = None, 
            grad : Optional[torch.Tensor] = None, 
            ids  : Optional[torch.Tensor] = None):
        if gts is not None:
            self.gts = gts
        if grad is not None:
            self.grad = grad
        if ids is not None:
            self.ids = ids
        #if self.active_latent is not None:
            #l = self.active_latent.unsqueeze(0)
            #l = l.expand(x.size()[0], self.active_latent.size()[0])
            #x = torch.cat([x, l], dim=1)
        self.inputs = x
        x = self.encode(x)
        self.preds = self.sdf(x)
        return self.preds
 
    def freeze(self):
        for k, v in self.named_parameters():
            v.requires_grad_(False)

    def setlatent(self, latent_id):
        # Sets the latent vector in use
        if latent_id is None:
            self.active_latent = None
        else:
            self.active_latent = self.latents(latent_id)

    def encode(self, x):
        if self.ff_dim > 0:
            x = F.linear(x, self.gauss_matrix)
            x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        
        elif self.pos_enc:
            x = positional_encoding(x)
        return x

    def sdf(self, x, ids=None):
        return None

    def loss(self, writer=None, epoch=None, l2=True):
        loss_val = torch.zeros_like(self.gts).squeeze()
        loss_dict = {}
        if l2:
            loss_dict['_l2_loss'] = l2_loss(self.preds, self.gts).squeeze()

        for l in self.args.loss:
            if l == 'gradient_loss':
                loss_dict[l] = globals()[l](self.inputs, self, self.grad).squeeze()
            elif l == 'eikonal_loss':
                loss_dict[l] = globals()[l](self.inputs, self, self.grad).squeeze()
            elif l == 'eikonal_loss_pre':
                loss_dict[l] = globals()[l](self.inputs, self, self.grad).squeeze()
            else:
                loss_dict[l] = globals()[l](self.preds, self.gts).squeeze()
            #if writer is not None:
            #    writer.add_scalar('Loss/{}'.format(l), _loss, epoch)
        return loss_dict

