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

from ..utils import PerfTimer
from ..diffutils import gradient
from ..geoutils import sample_unif_sphere
from .RenderBuffer import RenderBuffer

from ..PsDebugger import PsDebugger

from sol_nglod import aabb

class SphereTracer(nn.Module):
    
    def __init__(self, 
        device      : torch.device = 'cuda', 
        clamp       : list         = [-5.0,5.0], 
        sol         : bool         = False, 
        step_size   : float        = 1.0, 
        grad_method : str          = 'finitediff'
    ):
        """
        Args:
            device      : device
            clamp       : near far plane
            sol         : use optimized version (only works for some things)
            step_size    : scale the SDF for non-metric SDF
            grad_method : method for gradient
        """
        super().__init__()
        self.device = device
        self.sol = sol

        self._MARCH_ITER = 200 # Number of ray marching steps
        self._MIN_DIS = 0.0003 # Convergence threshold
        self.clamp = clamp
        self.step_size = step_size
        self.grad_method = grad_method


    def forward(self, net, ray_o, ray_d):
        """Performs sphere tracing.

        Args:
            ray_o (torch.Tensor): ray origin
            ray_d (torch.Tensor): ray direction
        """

        return self.forward_python(net, ray_o, ray_d)

    def forward_python(self, net, ray_o, ray_d):
        """Native implementation of sphere tracing."""

        timer = PerfTimer(activate=False)
        nettimer = PerfTimer(activate=False)

        if self.sol:
            return net.st(ray_o, ray_d)

        # Distanace from ray origin
        t = torch.zeros(ray_o.shape[0], 1, device=self.device)

        # Position in model space
        x = torch.addcmul(ray_o, ray_d, t)

        cond = torch.ones_like(t).bool()[:,0]
        #x, t, cond = aabb(ray_o, ray_d)

        normal = torch.zeros_like(x)

        # This function is in fact differentiable, but we treat it as if it's not, because
        # it evaluates a very long chain of recursive neural networks (essentially a NN with depth of
        # ~1600 layers or so). This is not sustainable in terms of memory use, so we return the final hit
        # locations, where additional quantities (normal, depth, segmentation) can be determined. The
        # gradients will propagate only to these locations. 
        with torch.no_grad():

            d = net(x)
            
            dprev = d.clone()

            # If cond is TRUE, then the corresponding ray has not hit yet.
            # OR, the corresponding ray has exit the clipping plane.
            #cond = torch.ones_like(d).bool()[:,0]

            # If miss is TRUE, then the corresponding ray has missed entirely.
            hit = torch.zeros_like(d).byte()
            
            for i in range(self._MARCH_ITER):
                timer.check("start")
                hit = (torch.abs(t) < self.clamp[1])[:,0]
                
                # 1. not hit surface
                cond = cond & (torch.abs(d) > self._MIN_DIS)[:,0] 

                # 2. not oscillating
                cond = cond & (torch.abs((d + dprev) / 2.0) > self._MIN_DIS * 3)[:,0]
                
                # 3. not a hit
                cond = cond & hit
                
                #cond = cond & ~hit
                
                # If the sum is 0, that means that all rays have hit, or missed.
                if not cond.any():
                    break;

                # Advance the x, by updating with a new t
                x = torch.where(cond.view(cond.shape[0], 1), torch.addcmul(ray_o, ray_d, t), x)
                
                # Store the previous distance
                dprev = torch.where(cond.unsqueeze(1), d, dprev)

                nettimer.check("nstart")
                # Update the distance to surface at x
                d[cond] = net(x[cond]) * self.step_size

                nettimer.check("nend")
                
                # Update the distance from origin 
                t = torch.where(cond.view(cond.shape[0], 1), t+d, t)
                timer.check("end")

        # AABB cull 

        hit = hit & ~(torch.abs(x) > 1.0).any(dim=-1)
        #hit = torch.ones_like(d).byte()[...,0]
        
        # The function will return 
        #  x: the final model-space coordinate of the render
        #  t: the final distance from origin
        #  d: the final distance value from
        #  miss: a vector containing bools of whether each ray was a hit or miss
        #_normal = F.normalize(gradient(x[hit], net, method='finitediff'), p=2, dim=-1, eps=1e-5)
        
        grad = gradient(x[hit], net, method=self.grad_method)
        _normal = F.normalize(grad, p=2, dim=-1, eps=1e-5)
        normal[hit] = _normal

        return RenderBuffer(x=x, depth=t, hit=hit, normal=normal)
    
    def sample_surface(self, n, net):
        
        # Sample surface using random tracing (resample until num_samples is reached)
        
        timer = PerfTimer(activate=False)
        
        with torch.no_grad():
            i = 0
            while i < 1000:
                ray_o = torch.rand((n, 3), device=self.device) * 2.0 - 1.0
                ray_d = torch.from_numpy(sample_unif_sphere(n)).float().to(self.device)
                rb = self.forward(net, ray_o, ray_d)

                pts_pr = rb.x[rb.hit] if i == 0 else torch.cat([pts_pr, rb.x[rb.hit]], dim=0)
                if pts_pr.shape[0] >= n:
                    break
                i += 1
                if i == 50:
                    print('Taking an unusually long time to sample desired # of points.')
        
        return pts_pr

