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

import torch
from torch.utils.data import Dataset

import mesh2sdf

from ..meshutils import trim_obj_to_file, load_obj, convert_to_nvc
from ..torchgp import point_sample, sample_surface

def setparam(param, argsparam):
    return param if param is not None else argsparam

class MeshDataset(Dataset):
    """Base class for single mesh datasets."""

    def __init__(self, 
        args=None, 
        dataset_path = None,
        raw_obj_path = None,
        sample_mode = None,
        get_normals = None,
        seed = None,
        num_samples = None,
        trim = False
    ):
        if args is None:
            self.dataset_path = dataset_path
            self.sample_mode = sample_mode
            self.get_normals = get_normals
            self.num_samples = num_samples
            self.raw_obj_path = raw_obj_path
            self.trim = trim
        else:
            self.dataset_path = setparam(dataset_path, args.dataset_path)
            self.sample_mode = setparam(sample_mode, args.sample_mode)
            self.get_normals = setparam(get_normals, args.get_normals)
            self.num_samples = setparam(num_samples, args.num_samples)
            self.raw_obj_path = setparam(raw_obj_path, args.raw_obj_path)
            self.trim = setparam(trim, args.trim)
        
        if self.raw_obj_path is not None and not os.path.exists(self.dataset_path):
            self.mesh = trim_obj_to_file(self.raw_obj_path, self.dataset_path)
        elif not os.path.exists(self.dataset_path):
            assert False and "Data does not exist and raw obj file not specified"
        else:
            V, F = load_obj(self.dataset_path)
            self.mesh = convert_to_nvc(V, F)

        self.ids, self.p, self.d, self.nrm = self._sample()
        
    def _sample(self):
        """Sample from a random selection of meshes."""

        nrm = None
        if self.get_normals:
            pts, nrm = sample_surface(self.mesh, self.num_samples*5)
            nrm = nrm.cpu()
        else:
            pts = point_sample(self.mesh, self.sample_mode, self.num_samples)
        d = self.evaluate_distance(pts.cuda(), self.mesh.cuda()).unsqueeze(1)
        ids = torch.zeros_like(d)
        d = d.cpu()
        pts = pts.cpu()
        ids = ids.cpu()
        # Do I really want to flip the points?
        return ids, pts, d, nrm

    def resample(self):
        """Resample SDF samples."""

        self.ids, self.p, self.d, self.nrm = self._sample()
    
    def __getitem__(self, idx: int):
        """Retrieve point sample."""
        if self.get_normals:
            return self.ids[idx], self.p[idx], self.d[idx], self.nrm[idx]
        else:
            return self.ids[idx], self.p[idx], self.d[idx]
            
    def __len__(self):
        """Return length of dataset (number of _samples_)."""

        return self.p.size()[0]

    def num_shapes(self):
        """Return length of dataset (number of _mesh models_)."""

        return 1

    def evaluate_distance(self, points: torch.Tensor, mesh: torch.Tensor):
        """
        Args:
            points (torch.Tensor): 3D sample points
            mesh (torch.Tensor): triangle mesh

        Returns:
            torch.Tensor: Signed distance for all input points
        """

        dist = mesh2sdf.mesh2sdf_gpu(points.contiguous(), mesh)[0]
        return dist

