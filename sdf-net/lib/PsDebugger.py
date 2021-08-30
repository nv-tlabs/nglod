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

import pdb

import torch
import polyscope as ps

from lib.torchgp import load_obj

class PsDebugger:
    def __init__(self):
        ps.init()
        self.pcls = {}

    def register_point_cloud(self, name, tensor, **kwargs):
        if 'cpu' not in str(tensor.device):
            tensor = tensor.cpu().detach()
        self.pcls[name] = ps.register_point_cloud(name, tensor.reshape(-1, 3).numpy(), **kwargs)

    def add_vector_quantity(self, pcl_name, vec_name, tensor, **kwargs):
        if 'cpu' not in str(tensor.device):
            tensor = tensor.cpu().detach()
        self.pcls[pcl_name].add_vector_quantity(vec_name, tensor.reshape(-1, 3).numpy(), **kwargs)
    
    def add_scalar_quantity(self, pcl_name, s_name, tensor, **kwargs):
        if 'cpu' not in str(tensor.device):
            tensor = tensor.cpu().detach()
        self.pcls[pcl_name].add_scalar_quantity(s_name, tensor.reshape(-1).numpy(), **kwargs)
    
    def add_color_quantity(self, pcl_name, c_name, tensor, **kwargs):
        if 'cpu' not in str(tensor.device):
            tensor = tensor.cpu().detach()
        self.pcls[pcl_name].add_color_quantity(c_name, tensor.reshape(-1, 3).numpy(), **kwargs)

    def add_surface_mesh(self, name, obj_path, **kwargs):
        verts, faces = load_obj(obj_path)
        ps.register_surface_mesh(name, verts.numpy(), faces.numpy(), **kwargs)
    
    def show(self):
        ps.show()
        pdb.set_trace()

