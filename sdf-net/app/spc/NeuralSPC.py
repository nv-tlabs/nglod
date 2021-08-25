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

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import logging as log
import time

from lib.utils import PerfTimer, setparam
from lib.torchgp import load_obj, normalize
from lib.PsDebugger import PsDebugger

from lib.models.BasicDecoder import BasicDecoder

import kaolin.ops.spc as spc_ops
import kaolin.render.spc as spc_render
import spc_utils as spc_utils
from SPC import SPC

class NeuralSPC(SPC):

    def __init__(self, 
        args                   = None,
        mesh_path      : str   = None,
        normalize_mesh : bool  = None,
        feature_dim    : int   = None,
        feature_std    : float = None,
        hidden_dim     : int   = None,
        num_lods       : int   = None, 
        base_lod       : int   = None,
        sample_tex     : bool  = None,
        joint_decoder  : bool  = None,
        feat_sum       : bool  = None,
        num_layers     : int   = None
    ):
        self.args = args
        self.mesh_path = setparam(args, mesh_path, 'mesh_path')
        self.normalize_mesh = setparam(args, normalize_mesh, 'normalize_mesh')
        self.feature_dim = setparam(args, feature_dim, 'feature_dim')
        self.feature_std = setparam(args, feature_std, 'feature_std')
        self.hidden_dim = setparam(args, hidden_dim, 'hidden_dim')
        self.num_lods = setparam(args, num_lods, 'num_lods') 
        self.base_lod = setparam(args, base_lod, 'base_lod')
        self.sample_tex = setparam(args, sample_tex, 'sample_tex')
        self.joint_decoder = setparam(args, joint_decoder, 'joint_decoder')
        self.feat_sum = setparam(args, feat_sum, 'feat_sum')
        self.num_layers = setparam(args, num_layers, 'num_layers')
        
        self.active_lods = [self.base_lod] + [self.base_lod + x for x in range(self.num_lods-1)]
        self.max_lod = self.num_lods + self.base_lod - 1

        if self.sample_tex:
            out = load_obj(self.mesh_path, load_materials=True)
            self.V, self.F, self.texv, self.texf, self.mats = out
        else:
            self.V, self.F = load_obj(self.mesh_path)
        if self.normalize_mesh:
            self.V, self.F = normalize(self.V, self.F)
        octree = spc_utils.mesh_to_octree(self.V, self.F, self.max_lod)

        super().__init__(octree, self.feature_dim, self.base_lod, 
                         num_lods=self.num_lods, feature_std=self.feature_std)

        self.louts = nn.ModuleList([])

        if self.joint_decoder:
            self.num_decoders = 1
        else:
            self.num_decoders = self.num_lods

        for i in range(self.num_decoders):
            self.louts.append(
                BasicDecoder(self.feature_dim, 1, F.relu, True, 
                             num_layers=self.num_layers, hidden_dim=self.hidden_dim, skip=False)
            )
        
        self.lod = None
        self.feat = None
        torch.cuda.empty_cache()
        
    def forward(self, x):
        return self.sdf(x)

    def sdf(self, x, lod=None, pidx=None, return_lst=False):
        if x.shape[0] == 0:
            return torch.zeros_like(x)[...,0:1]

        if self.lod is None:
            self.lod = self.num_lods - 1
        if lod is None:
            lod = self.lod
        if pidx is None:
            pidx = self.query(x, lod)

        # Get features for the hits.
        feat = self.interpolate(x, lod, pidx)
        self.feats = []
        self.feats.append(feat)

        # Climb the tree back to the base lod
        
        if self.feat_sum or return_lst:
            parent = self.parents[pidx]
            for i in range(lod):
                feat = self.interpolate(x, lod-i-1, parent)
                self.feats.append(feat)
                parent = self.parents[parent]
            self.feats = self.feats[::-1]

        if self.feat_sum:
            for i in range(1, len(self.feats)):
                self.feats[i] += self.feats[i-1]

        decoder = lambda i, j : self.louts[i](self.feats[j][...,:self.feature_dim])[...,0:1]

        if return_lst:
            if self.joint_decoder:
                return [decoder(0, j) for j in range(self.num_lods)]
            else:
                return [decoder(j, j) for j in range(self.num_lods)]
        else:
            didx = 0 if self.joint_decoder else lod
            fidx = 0 if not self.feat_sum else lod
            sdf = decoder(didx, fidx)
            return sdf

