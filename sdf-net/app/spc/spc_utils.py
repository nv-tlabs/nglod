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
import numpy as np
import pandas as pd

import kaolin.ops.spc as spc_ops
from lib.torchgp import sample_surface

def create_dense_octree(level):
    """Creates a dense SPC model"""
    coords = np.arange(2**level)
    points = np.array(np.meshgrid(coords, coords, coords, indexing='xy'))
    points = points.transpose(3,2,1,0).reshape(-1, 3)
    points = torch.from_numpy(points).short().cuda()
    octree = spc_ops.unbatched_points_to_octree(points, level)
    return octree

def points_to_coords(points, level):
    """Transform from [-1, 1] to [0, 2^l]"""
    return (2**level) * (points*0.5 + 0.5)

def normalize_points(points, level):
    """Trasnform from [0, 2^l] to [-1, 1]"""
    return (points.float() / (2**level)) * 2.0 - 1.0

def point2coeff(x, pts, lod):
    res = 2**lod
    x_ = (res * ((x + 1.0)/2.0)) - pts
    _x = 1.0 - x_

    coeffs = torch.zeros(x.shape[0], 8, device=x.device)
    coeffs[:,:4] = _x[...,0:1]
    coeffs[:,4:] = x_[...,0:1]
    coeffs[:,(0,1,4,5)] *= _x[...,1:2]
    coeffs[:,(2,3,6,7)] *= x_[...,1:2]
    coeffs[:,::2] *= _x[...,2:3]
    coeffs[:,1::2] *= x_[...,2:3]

    return coeffs

def get_level_points(points, pyramid, level):
    return points[pyramid[1, level]:pyramid[1, level+1]]

def octree_to_spc(octree):
    lengths = torch.tensor([len(octree)], dtype=torch.int32)
    _, pyramid, prefix = spc_ops.scan_octrees(octree, lengths)
    points = spc_ops.generate_points(octree, pyramid, prefix)
    pyramid = pyramid[0]
    return points, pyramid, prefix

def mesh_to_octree(vertices, faces, level):
    samples = sample_surface(vertices.cuda(), faces.cuda(), 100000000)[0]
    # Augment samples... may be a hack that isn't actually needed
    samples = torch.cat([samples, 
        samples + (torch.rand_like(samples) * 2.0 - 1.0) * (1.0/(2**(level+1)))], dim=0)
    samples = spc_ops.quantize_points(samples, level)
        
    morton = torch.sort(spc_ops.points_to_morton(torch.unique(samples.contiguous(), dim=0).contiguous()))[0]
    points = spc_ops.morton_to_points(morton)
    octree = spc_ops.unbatched_points_to_octree(points, level)
    return octree

def mesh_to_spc(vertices, faces, level):
    octree = mesh_to_octree(vertices, faces, level)
    points, pyramid, prefix = octree_to_spc(octree)
    return octree, points, pyramid, prefix

def create_dual(point_hierarchy, pyramid):
    pyramid_dual = torch.zeros_like(pyramid)
    point_hierarchy_dual = []
    for i in range(pyramid.shape[1]-1):
        corners = spc_ops.points_to_corners(get_level_points(point_hierarchy, pyramid, i)).reshape(-1, 3)
        points_dual = torch.unique(corners, dim=0)
        sort_idxes = spc_ops.points_to_morton(points_dual).sort()[1]
        points_dual = points_dual[sort_idxes]
        point_hierarchy_dual.append(points_dual)
        pyramid_dual[0, i] = len(point_hierarchy_dual[i])
        if i > 0:
            pyramid_dual[1, i] += pyramid_dual[:, i-1].sum()
    pyramid_dual[1, pyramid.shape[1]-1] += pyramid_dual[:, pyramid.shape[1]-2].sum()
    point_hierarchy_dual = torch.cat(point_hierarchy_dual, dim=0)
    return point_hierarchy_dual, pyramid_dual

def create_trinkets(point_hierarchy, pyramid, point_hierarchy_dual, pyramid_dual):
    trinkets = []
    parents = []
    luts = []

    # At a high level... the goal of this algorithm is to create a table which maps from the primary
    # octree of voxels to the dual octree of corners, while also keeping track of parents. 
    # It does so by constructing a lookup table which maps morton codes of the source octree corners
    # to the index of the destination (dual), then using pandas to do table lookups. It's a silly
    # solution that would be much faster with a GPU but works well enough.
    for i in range(pyramid_dual.shape[1]-1):
        
        # The source (primary octree) is sorted in morton order by construction
        points = get_level_points(point_hierarchy, pyramid, i)
        corners = spc_ops.points_to_corners(points)
        mt_src = spc_ops.points_to_morton(corners.reshape(-1, 3))

        # The destination (dual octree) needs to be sorted too
        points_dual = get_level_points(point_hierarchy_dual, pyramid_dual, i)
        mt_dest = spc_ops.points_to_morton(points_dual)

        # Uses arange to associate from the morton codes to the point index. The point index is indexed from 0.
        luts.append(dict(zip(mt_dest.cpu().numpy(), np.arange(mt_dest.shape[0]))))
        
        if i == 0:
            parents.append(torch.LongTensor([-1]).cuda())
        else:
            # Dividing by 2 will yield the morton code of the parent
            pc = torch.floor(points / 2.0).short()
            mt_pc = spc_ops.points_to_morton(pc.contiguous())
            mt_pc_dest = spc_ops.points_to_morton(points)
            plut = dict(zip(mt_pc_dest.cpu().numpy(), np.arange(mt_pc_dest.shape[0])))
            pc_idx = pd.Series(plut).reindex(mt_pc.cpu().numpy()).values
            parents.append(torch.LongTensor(pc_idx).cuda())

        idx = pd.Series(luts[i]).reindex(mt_src.cpu().numpy()).values
        #trinkets.append(torch.LongTensor(idx).cuda().reshape(-1, 8) + pyramid_dual[1, i])
        trinkets.append(torch.LongTensor(idx).cuda().reshape(-1, 8))

    # Trinkets are relative to the beginning of each pyramid base
    trinkets = torch.cat(trinkets, dim=0)
    parents = torch.cat(parents, dim=0)
    trinkets = torch.cat([trinkets, -1 * torch.ones([1,8], device=trinkets.device)], dim=0).long()
    return trinkets, parents

