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

def sample_tex(
    Tp : torch.Tensor, # points [N ,2] 
    TM : torch.Tensor, # material indices [N]
    materials):

    max_idx = TM.max()
    assert(max_idx > -1 and "No materials detected! Check the material definiton on your mesh.")

    rgb = torch.zeros(Tp.shape[0], 3, device=Tp.device)

    Tp = (Tp * 2.0) - 1.0
    # The y axis is flipped from what UV maps generally expects vs in PyTorch
    Tp[...,1] *= -1

    for i in range(max_idx+1):
        mask = (TM == i)
        if mask.sum() == 0:
            continue
        if 'diffuse_texname' not in materials[i]:
            if 'diffuse' in materials[i]:
                rgb[mask] = materials[i]['diffuse'].to(Tp.device)
            continue

        map = materials[i]['diffuse_texname'][...,:3].permute(2, 0, 1)[None].to(Tp.device)
        grid = Tp[mask]
        grid = grid.reshape(1, grid.shape[0], 1, grid.shape[1])
        _rgb = F.grid_sample(map, grid, mode='bilinear', padding_mode='reflection', align_corners=True)
        _rgb = _rgb[0,:,:,0].permute(1,0)
        rgb[mask] = _rgb

    return rgb


