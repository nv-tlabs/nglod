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

import math 

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.BaseLOD import BaseLOD
from lib.models.BasicDecoder import BasicDecoder
from lib.utils import PerfTimer

class MyActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class FeatureVolume(nn.Module):
    def __init__(self, fdim, fsize):
        super().__init__()
        self.fsize = fsize
        self.fdim = fdim
        self.fm = nn.Parameter(torch.randn(1, fdim, fsize+1, fsize+1, fsize+1) * 0.01)
        self.sparse = None

    def forward(self, x):
        N = x.shape[0]
        if x.shape[1] == 3:
            sample_coords = x.reshape(1, N, 1, 1, 3) # [N, 1, 1, 3]    
            sample = F.grid_sample(self.fm, sample_coords, 
                                   align_corners=True, padding_mode='border')[0,:,:,0,0].transpose(0,1)
        else:
            sample_coords = x.reshape(1, N, x.shape[1], 1, 3) # [N, 1, 1, 3]    
            sample = F.grid_sample(self.fm, sample_coords, 
                                   align_corners=True, padding_mode='border')[0,:,:,:,0].permute([1,2,0])
        
        return sample

class OctreeSDF(BaseLOD):
    def __init__(self, args, init=None):
        super().__init__(args)

        self.fdim = self.args.feature_dim
        self.fsize = self.args.feature_size
        self.hidden_dim = self.args.hidden_dim
        self.pos_invariant = self.args.pos_invariant

        self.features = nn.ModuleList([])
        for i in range(self.args.num_lods):
            self.features.append(FeatureVolume(self.fdim, (2**(i+self.args.base_lod))))
        self.interpolate = self.args.interpolate

        self.louts = nn.ModuleList([])

        self.sdf_input_dim = self.fdim
        if not self.pos_invariant:
            self.sdf_input_dim += self.input_dim

        self.num_decoder = 1 if args.joint_decoder else self.args.num_lods 

        for i in range(self.num_decoder):
            self.louts.append(
                nn.Sequential(
                    nn.Linear(self.sdf_input_dim, self.hidden_dim, bias=True),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, 1, bias=True),
                )
            )
        
    def encode(self, x):
        # Disable encoding
        return x

    def sdf(self, x, lod=None, return_lst=False):
        if lod is None:
            lod = self.lod
        
        # Query
        l = []
        samples = []

        for i in range(self.num_lods):
            
            # Query features
            sample = self.features[i](x)
            samples.append(sample)
            
            # Sum queried features
            if i > 0:
                samples[i] += samples[i-1]
            
            # Concatenate xyz
            ex_sample = samples[i]
            if not self.pos_invariant:
                ex_sample = torch.cat([x, ex_sample], dim=-1)

            if self.num_decoder == 1:
                prev_decoder = self.louts[0]
                curr_decoder = self.louts[0]
            else:
                prev_decoder = self.louts[i-1]
                curr_decoder = self.louts[i]
            
            d = curr_decoder(ex_sample)

            # Interpolation mode
            if self.interpolate is not None and lod is not None:
                
                if i == len(self.louts) - 1:
                    return d

                if lod+1 == i:
                    _ex_sample = samples[i-1]
                    if not self.pos_invariant:
                        _ex_sample = torch.cat([x, _ex_sample], dim=-1)
                    _d = prev_decoder(_ex_sample)

                    return (1.0 - self.interpolate) * _l + self.interpolate * d
            
            # Get distance
            else: 
                d = curr_decoder(ex_sample)

                # Return distance if in prediction mode
                if lod is not None and lod == i:
                    return d

                l.append(d)
        if self.training:
            self.loss_preds = l

        if return_lst:
            return l
        else:
            return l[-1]
    
