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

import numpy as np
import torch
from lib.renderer import SphereTracer
from lib.geoutils import sample_unif_sphere, voxel_corners
from lib.utils import PerfTimer

# Utils that use rendering features

def sample_surface(n, net, sol=False, device='cuda'):
    
    timer = PerfTimer(activate=False)
    tracer = SphereTracer(device, sol=sol)

    # Sample surface using random tracing (resample until num_samples is reached)
    i = 0
    while i < 1000:
        ray_o = torch.rand((n, 3), device=device) * 2.0 - 1.0
        # this really should just return a torch array in the first place
        ray_d = torch.from_numpy(sample_unif_sphere(n)).float().to(device)
        
        rb = tracer(net, ray_o, ray_d)
        pts = rb.x
        hit = rb.hit

        pts_pr = pts[hit] if i == 0 else torch.cat([pts_pr, pts[hit]], dim=0)
        if pts_pr.shape[0] >= n:
            break
    
        i += 1
        if i == 50:
            print('Taking an unusually long time to sample desired # of points.')
    timer.check(f"done in {i} iterations")

    return pts_pr

def voxel_sparsify(n, net, lod, sol=False, device='cuda'):
    
    #lod = 5

    _lod = net.lod
    net.lod = lod
    surface = sample_surface(n, net, sol=sol, device=device)[:n]    
    
    vs = []

    for i in range(lod+1):
        res = 2 ** (i+2)
        uniq = torch.unique( ( ((surface+1.0) / 2.0) * res).floor().long(), dim=0)
        vs.append(uniq)

    net.lod = _lod
    return vs

