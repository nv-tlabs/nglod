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
import torch.nn as nn
import torch.nn.functional as F

from lib.models.Embedder import positional_encoding
from lib.models.BasicDecoder import BasicDecoder
from lib.utils import setparam

class BaseSDF(nn.Module):
    def __init__(self,
        args             = None,
        pos_enc  : bool  = None,
        ff_dim   : int   = None,
        ff_width : float = None
    ):
        super().__init__()
        self.args = args
        self.pos_enc = setparam(args, pos_enc, 'pos_enc')
        self.ff_dim = setparam(args, ff_dim, 'ff_dim')
        self.ff_width = setparam(args, ff_width, 'ff_width')
        
        self.input_dim = 3
        self.out_dim = 1

        if self.ff_dim > 0:
            mat = torch.randn([self.ff_dim, 3]) * self.ff_width
            self.gauss_matrix = nn.Parameter(mat)
            self.gauss_matrix.requires_grad_(False)
            self.input_dim += (self.ff_dim * 2) - 3
        elif self.pos_enc:
            self.input_dim = self.input_dim * 13

    def forward(self, x, lod=None):
        x = self.encode(x)
        return self.sdf(x)
 
    def freeze(self):
        for k, v in self.named_parameters():
            v.requires_grad_(False)

    def encode(self, x):
        if self.ff_dim > 0:
            x = F.linear(x, self.gauss_matrix)
            x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        elif self.pos_enc:
            x = positional_encoding(x)
        return x

    def sdf(self, x, lod=None):
        return None
