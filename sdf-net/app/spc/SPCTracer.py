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
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from lib.utils import PerfTimer
from lib.diffutils import gradient
from lib.PsDebugger import PsDebugger
from lib.tracer.RenderBuffer import RenderBuffer
from lib.tracer.BaseTracer import BaseTracer

import kaolin
import kaolin.ops.spc as spc_ops
import kaolin.render.spc as spc_render

class SPCTracer(BaseTracer):

    def forward(self, net, ray_o, ray_d):
        timer = PerfTimer(activate=False)

        N = ray_o.shape[0]
        t = torch.zeros(N, 1, device=ray_o.device)
        x = torch.addcmul(ray_o, ray_d, t)
        timer.check("generate rays")

        # Trace SPC
        nugs = net.raytrace(ray_o.cuda(), ray_d.cuda(), net.lod)

        timer.check("get nugs")
        info = spc_render.mark_first_hit(nugs)
        info_idxes = torch.nonzero(info).int()

        timer.check("info")

        # Trace against the SPC
        d, pidx, cond = spc_render.unbatched_ray_aabb(nugs, net.points, ray_o, ray_d, net.lod+net.base_lod, 
                                                      info, info_idxes)
        pidx = pidx.long()
        t += d
        x = torch.addcmul(ray_o, ray_d, t)
        timer.check("ray_aabb")

        # Initialize variables
        hit = torch.zeros_like(cond)
        normal = torch.zeros_like(x)

        res = float(2**(net.lod+net.base_lod))

        with torch.no_grad():
            _d = net.sdf(x[cond], net.lod, pidx[cond]) / res
            d[cond] = _d * self.step_size
            d[~cond] = 20
            dprev = d.clone()

            for i in range(self.num_steps):
                timer.check("start iter")
                t += d

                x = torch.where(cond.view(cond.shape[0], 1), torch.addcmul(ray_o, ray_d, t), x)
                hit = torch.where(cond, torch.abs(d)[...,0] < self.min_dis / res, hit)
                hit |= torch.where(cond, torch.abs(d+dprev)[...,0] * 0.5 < (self.min_dis*5) / res,  hit)
                cond = torch.where(cond, (t < self.camera_clamp[1])[...,0], cond)
                cond &= ~hit

                timer.check("step")

                if not cond.any():
                    break

                x = torch.where(cond.view(cond.shape[0], 1), torch.addcmul(ray_o, ray_d, t), x)
                dprev = torch.where(cond.view(cond.shape[0], 1), d, dprev)

                _d, _pidx, cond = spc_render.unbatched_ray_aabb(nugs, net.points, x, ray_d, 
                        net.lod+net.base_lod, info, info_idxes, cond)
                _pidx = _pidx.long()
                pidx = torch.where(cond, _pidx, pidx)
                t += _d
                x = torch.where(cond.view(cond.shape[0], 1), torch.addcmul(ray_o, ray_d, t), x)
                timer.check("everything but sdf")
                _d = net.sdf(x[cond], net.lod, pidx[cond]) / res
                d[cond] = _d * self.step_size
                timer.check("sdf")

        grad = gradient(x[hit], net, method=self.grad_method)
        normal[hit] = F.normalize(grad, p=2, dim=-1, eps=1e-5)

        return RenderBuffer(x=x, depth=t, hit=hit, normal=normal)

