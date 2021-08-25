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
import math

from lib.utils import PerfTimer, setparam
from lib.PsDebugger import PsDebugger

import spc_utils as spc_utils

import kaolin.ops.spc as spc_ops
import kaolin.render.spc as spc_render

class SPC(nn.Module):

    def __init__(self, 
        octree         : torch.Tensor,
        feature_dim    : int,
        base_lod       : int,
        num_lods       : int   = 1, 
        feature_std    : float = 0.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.base_lod = base_lod
        self.num_lods = num_lods
        self.feature_std = feature_std

        # List of octree levels which are optimized.
        self.active_lods = [self.base_lod] + [self.base_lod + x for x in range(self.num_lods-1)]
        self.max_lod = self.num_lods + self.base_lod - 1

        log.info(f"Active LODs: {self.active_lods}")
        
        self.build(octree)

        # SPC initialization uses lots of spare memory, so clear cache
        torch.cuda.empty_cache()

    def build(self, octree):
        self.octree = octree
        self.points, self.pyramid, self.prefix = spc_utils.octree_to_spc(self.octree)
        
        self.points_dual, self.pyramid_dual = spc_utils.create_dual(self.points, self.pyramid)

        self.trinkets, self.parents = spc_utils.create_trinkets(self.points, self.pyramid, 
                                                                self.points_dual, self.pyramid_dual)
        log.info("Built dual octree and trinkets")
        
        # Create features.
        self.features = nn.ParameterList([])
        for al in self.active_lods:
            fts = torch.zeros(self.pyramid_dual[0,al]+1, self.feature_dim)
            fts += torch.randn_like(fts) * self.feature_std
            self.features.append(nn.Parameter(fts))

        num_feat = 0
        for feature in self.features:
            num_feat += feature.shape[0]
        log.info(f"# Feature Vectors: {num_feat}")

    def query(self, x, lod):
        qpts = spc_ops.quantize_points(x, lod+self.base_lod)
        return spc_ops.unbatched_query(self.octree, self.points, self.pyramid, self.prefix, 
                                       qpts, lod+self.base_lod).long()
 
    def interpolate(self, x, lod, pidx=None):
        if pidx is None:
            pidx = self.query(x, lod)
        coeffs = spc_ops.points_to_coeffs(spc_utils.points_to_coords(x, lod+self.base_lod), 
                                          self.points[pidx])
        coeffs = torch.clamp(coeffs, 0.0, 1.0)
        return self._interpolate(coeffs, self.features[lod][self.trinkets[pidx]])

    def _interpolate(self, coeffs, feats):
        shape = feats.shape[2:]
        feats = (coeffs.view(*coeffs.shape, 1) * feats.view(*feats.shape[:2], -1)).sum(-2)
        return feats.reshape(-1, *shape)

    def raytrace(self, ray_o, ray_d, lod):
        nugs = spc_render.unbatched_raytrace(self.octree, self.points, self.pyramid, self.prefix,
                                             ray_o, ray_d, lod+self.base_lod)
        return nugs

