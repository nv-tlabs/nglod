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
from .random_face import random_face
from .area_weighted_distribution import area_weighted_distribution

def sample_surface(
    mesh        : torch.Tensor,
    num_samples : int,
    distrib            = None,
):
    """Sample points and their normals on mesh surface.

    Args:
        mesh (torch.Tensor): triangle mesh
        num_samples (int): number of surface samples
        distrib: distribution to use. By default, area-weighted distribution is used
    """
    if distrib is None:
        distrib = area_weighted_distribution(mesh)

    # Select faces & sample their surface
    f, normals = random_face(mesh, num_samples, distrib)

    u = torch.sqrt(torch.rand(num_samples)).to(mesh.device).unsqueeze(-1)
    v = torch.rand(num_samples).to(mesh.device).unsqueeze(-1)

    samples = (1 - u) * f[:,0,:] + (u * (1 - v)) * f[:,1,:] + u * v * f[:,2,:]
    
    return samples, normals

