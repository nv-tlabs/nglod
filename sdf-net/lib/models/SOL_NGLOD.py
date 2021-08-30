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
import numpy as np
import torch
import sol_nglod

from lib.spc3d import SPC3D, to_morton
from lib.renderutils import voxel_sparsify, sample_surface
from lib.geoutils import unique_corners

class SOL_NGLOD(object):
    def __init__(self, net):
        self.feat = []
        for f in net.features:
            self.feat.append(f.fm.detach())
        
        self.w0 = []
        self.b0 = []
        self.w1 = []
        self.b1 = []
        for l in net.louts:
            self.w0.append(l[0].weight.half())
            self.b0.append(l[0].bias.half())
            self.w1.append(l[2].weight.half())
            self.b1.append(l[2].bias.half())
        
        self.lod = len(net.louts)-1

        # Brute force function to get all occupied voxels
        self.vs = voxel_sparsify(2000000, net, self.lod, sol=False)

        _vs = self.vs[self.lod].cpu().numpy().astype(np.uint16)
        # Convert to SPC
        self.spc = SPC3D(self.lod+2)
        for i in range(_vs.shape[0]):
            self.spc.mdata[i] = to_morton(_vs[i])
            self.spc.psize += 1
        _sorted = np.sort(self.spc.mdata[:self.spc.psize])
        self.spc.mdata[:self.spc.psize] = _sorted
        self.spc.morton_to_point(self.spc.psize, self.spc.mdata, self.spc.pdata)
        self.spc.points_to_nodes()

        self.vc = []
        self.cc = []
        self.cf = []
        self.pyramid = []

        for i in range(self.lod+1):
            res = 2**(i+2)
            #self.vc.append((((self.vs[i].float() + 0.5) / float(res)) * 2.0 - 1.0).half())
            corners, features = unique_corners(self.vs[i], self.feat[i])
            self.pyramid.append(corners.shape[0])
            self.cc.append(corners.byte())
            self.cf.append(features.half())
        self.pyramid = np.array(self.pyramid)

    def __call__(self, x):
        return self.sdf(x)

    def save(self, path):
        print(f"Saving to {path}")

        cc = torch.cat(self.cc, dim=0).cpu().detach()
        cf = torch.cat(self.cf, dim=0).cpu().detach()
        w0 = torch.stack(self.w0, dim=0).cpu().detach()
        b0 = torch.stack(self.b0, dim=0).cpu().detach()
        w1 = torch.stack(self.w1, dim=0).cpu().detach()
        b1 = torch.stack(self.b1, dim=0).cpu().detach()
        pyramid = self.pyramid

        np.savez_compressed(os.path.join(path), 
                octree=self.spc.oroot,
                cc=cc,
                cf=cf,
                w0=w0,
                b0=b0,
                w1=w1,
                b1=b1,
                pyramid=pyramid
        )

