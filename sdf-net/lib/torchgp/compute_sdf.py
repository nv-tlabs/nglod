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
import numpy as np
import mesh2sdf

def compute_sdf(
    V : torch.Tensor,
    F : torch.Tensor,
    points : torch.Tensor):
    """Given a [N,3] list of points, returns a [N] list of SDFs for a mesh."""

    mesh = V[F]

    points_cpu = points.cpu().numpy().reshape(-1).astype(np.float64)
    mesh_cpu = mesh.cpu().numpy().reshape(-1).astype(np.float64)

    # Legacy, open source mesh2sdf code
    dist = mesh2sdf.mesh2sdf_gpu(points.contiguous(), mesh)[0]
    
    return dist
