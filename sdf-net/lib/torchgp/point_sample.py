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
from .sample_near_surface import sample_near_surface
from .sample_surface import sample_surface
from .sample_uniform import sample_uniform
from .random_face import random_face
from .area_weighted_distribution import area_weighted_distribution

def point_sample(mesh : torch.Tensor, techniques : list, num_samples : int):
    """Sample points from a mesh.

    Args:
        mesh (torch.Tensor): #F, 3, 3 array of vertices
        techniques (list[str]): list of techniques to sample with
        num_samples (int): points to sample per technique
    """
    if 'trace' in techniques or 'near' in techniques:
        # Precompute face distribution
        distrib = area_weighted_distribution(mesh)

    samples = []
    for technique in techniques:
        if technique =='trace':
            samples.append(sample_surface(mesh, num_samples, distrib)[0])
        elif technique == 'near':
            samples.append(sample_near_surface(mesh, num_samples, distrib))
        elif technique == 'rand':
            samples.append(sample_uniform(num_samples).to(mesh.device))
    samples = torch.cat(samples, dim=0)
    return samples

