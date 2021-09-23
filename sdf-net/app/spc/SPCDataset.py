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

import torch
from torch.utils.data import Dataset

import numpy as np
import logging as log

import multiprocessing

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from lib.torchgp import load_obj, point_sample, sample_spc, sample_surface, sample_near_surface, compute_sdf, normalize
from lib.PsDebugger import PsDebugger
from lib.utils import PerfTimer, setparam

import contextlib
import sys

import kaolin
import kaolin.ops.spc as spc_ops
import kaolin.render.spc as spc_render
from spc_utils import get_level_points

def csdf(proc, pts, V, F, return_dict):
    return_dict[proc] = compute_sdf(V.cuda(), F.cuda(), pts.cuda()).cpu()

class SPCDataset(Dataset):
    """Base class for single mesh datasets."""

    def __init__(self, 
        net,
        args=None, 
        dataset_path = None,
        raw_obj_path = None,
        sample_mode = None,
        get_normals = None,
        seed = None,
        num_samples = None,
        trim = None,
        samples_per_voxel = None,
        block_res = None
    ):
        self.dataset_path = setparam(args, dataset_path, 'dataset_path')
        self.sample_mode = setparam(args, sample_mode, 'sample_mode')
        self.get_normals = setparam(args, get_normals, 'get_normals')
        self.num_samples = setparam(args, num_samples, 'num_samples')
        self.raw_obj_path = setparam(args, raw_obj_path, 'raw_obj_path')
        self.samples_per_voxel = setparam(args, samples_per_voxel, 'samples_per_voxel')
        self.block_res = setparam(args, block_res, 'block_res')
        
        self.block_size = 2**(self.block_res * 3)
        self.net = net

    def get_block(self, block_idx, mortons, subsample_res):
        """
        Returns the mask to the corners of the specified block index        
        """
        subsample_size = 2**subsample_res
        morton_range = (self.block_size * min(block_idx, subsample_size-1))
        morton_range = [morton_range, morton_range + self.block_size]
        return torch.stack([mortons >= morton_range[0], mortons < morton_range[1]]).all(0)

    def get_block_idxes(self, lod=0):
        """
        Returns a list of active subsample block indices.
        """

        point_range = [self.net.pyramid[1, lod+self.net.base_lod],
                       self.net.pyramid[1, lod+self.net.base_lod+1]]
        mortons = spc_ops.points_to_morton(self.net.points[point_range[0]:point_range[1]])
        subsample_res = 3 * max(0, lod+self.net.base_lod - self.block_res)
        subsample_size = 2**subsample_res
        block = 2**(self.block_res * 3)

        block_idxes = []
        for i in range(subsample_size):
            active_idx = self.get_block(i, mortons, subsample_res)
            if active_idx.sum() > 0:
                block_idxes.append(i)
        return block_idxes

    def init(self, block_idx=15):
        """
        Initialize the dataset.
        """

        log.info("Initializing dataset...")

        level = self.net.base_lod + self.net.num_lods - 1
        
        # Here, corners mean "the bottom left corner of the voxel to sample from"
        corners = get_level_points(self.net.points, self.net.pyramid, level)

        # Block Resolution determines the size of the subsampling region. If the region is 
        # smaller than the entire region, then use the idx to get the correct region.
        subsample_res = 3 * max(0, level - self.block_res)
        if subsample_res > 0:
            mortons = spc_ops.points_to_morton(corners)
            corners = corners[self.get_block(block_idx, mortons, subsample_res)]
        
        self.pts_ = sample_spc(corners, level, self.samples_per_voxel).cpu()
        
        # Sample from the near and surface distributions
        def sample_aux(size):
            near = sample_near_surface(self.net.V.cuda(), 
                                    self.net.F.cuda(), 
                                    size, 
                                    variance=1.0/(2**level))
            trace = sample_surface(self.net.V.cuda(),
                                self.net.F.cuda(),
                                size)[0]
            return torch.cat([near, trace], dim=0).cpu()

        surface_samples = []
        for i in range(self.pts_.shape[0] // (10**7)):
            surface_samples.append(sample_aux(10**7))
        remainder_samples = self.pts_.shape[0] % (10**7)
        if remainder_samples > 0:
            surface_samples.append(sample_aux(remainder_samples))

        self.pts_ = torch.cat([self.pts_, torch.cat(surface_samples, dim=0)], dim=0)
        
        # This is a hack to prevent memory errors when external libraries are used for SDF computation
        manager = multiprocessing.Manager()
        work = torch.split(self.pts_, 10**7)
        return_dict = manager.dict()
        jobs = []
        for i, w in enumerate(work):
            p = multiprocessing.Process(target=csdf, args=(i, w, self.net.V, self.net.F, return_dict))    
            p.start()
            p.join()
        p.close()
        self.d_ = torch.cat(return_dict.values(), dim=0)
        self.d_ = self.d_[...,None]

        # Only valid if the samples are correctly being filtered
        qpts = spc_ops.quantize_points(self.pts_, level).cuda()
        self.pidx = spc_ops.unbatched_query(self.net.octree, self.net.points, self.net.pyramid, 
                                            self.net.prefix, qpts, level)
        
        self.pts_ = self.pts_[self.pidx>-1]
        self.d_ = self.d_[self.pidx>-1]


    def resample(self, lod=0, idx=0):
        """Resample SDF samples."""

        log.info(f"Resampling...")
        
        # Find residual past 128 (2^7)
        level = lod + self.net.base_lod
        subsample_res = 3 * max(0, level - self.block_res)
        if subsample_res > 0:
            block_size = 2**(self.block_res * 3) # 2** (7 * 3) for 128^3
            subsample_size = 2**subsample_res
            morton_range = (block_size * min(idx, subsample_size-1))
            morton_range = [morton_range, morton_range + block_size]
            valid_pidx = self.pidx[self.pidx>-1]
            mortons = spc_ops.points_to_morton(self.net.points[valid_pidx.long()])
            active_idx = torch.stack([mortons >= morton_range[0], mortons < morton_range[1]]).all(0)
            valid_pidx[~active_idx] = -1

            self.pts = self.pts_[valid_pidx>-1]
            self.d = self.d_[valid_pidx>-1]

            log.info(f"Total ROI Samples: {self.pts.shape[0]}")
        else:
            self.pts = self.pts_
            self.d = self.d_

        _idx = torch.randperm(self.pts.shape[0], device='cuda')

        self.pts = self.pts[_idx]
        self.d = self.d[_idx]
        
        log.info(f"Permuted Samples")

        total_samples = self.num_samples
        self.pts = self.pts[:total_samples]
        self.d = self.d[:total_samples]

        self.d = self.d.cpu()
        self.pts = self.pts.cpu()

    def __getitem__(self, idx: int):
        """Retrieve point sample."""
        if self.get_normals:
            return self.pts[idx], self.d[idx], self.nrm[idx]
        else:
            return self.pts[idx], self.d[idx]
            
    def __len__(self):
        """Return length of dataset (number of _samples_)."""

        return self.pts.shape[0]

    def num_shapes(self):
        """Return length of dataset (number of _mesh models_)."""

        return 1

