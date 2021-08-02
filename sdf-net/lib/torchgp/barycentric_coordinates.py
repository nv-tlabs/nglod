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

# Same API as https://github.com/libigl/libigl/blob/main/include/igl/barycentric_coordinates.cpp

def barycentric_coordinates(
    points : torch.Tensor, 
    A : torch.Tensor,
    B : torch.Tensor,
    C : torch.Tensor):
    """
    Return barycentric coordinates for a given set of points and triangle vertices

    Args:
        points: [N, 3]
        A: [N, 3] vertex0
        B: [N, 3] vertex1
        C: [N, 3] vertex2
    """

    v0 = B-A
    v1 = C-A
    v2 = points-A
    d00 = (v0*v0).sum(dim=-1)
    d01 = (v0*v1).sum(dim=-1)
    d11 = (v1*v1).sum(dim=-1)
    d20 = (v2*v0).sum(dim=-1)
    d21 = (v2*v1).sum(dim=-1)
    denom = d00*d11 - d01*d01
    L = torch.zeros(points.shape[0], 3, device=points.device)
    # Warning: This clipping may cause undesired behaviour
    L[...,1] = torch.clamp((d11*d20 - d01*d21)/denom, 0.0, 1.0)
    L[...,2] = torch.clamp((d00*d21 - d01*d20)/denom, 0.0, 1.0)
    L[...,0] = torch.clamp(1.0 - (L[...,1] + L[...,2]), 0.0, 1.0)
    return L
