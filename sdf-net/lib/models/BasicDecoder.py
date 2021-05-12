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

from .utils import init_decoder

class BasicDecoder(nn.Module):
    def __init__(self, args, input_dim, num_layers=None, hidden_dim=None, output_dim=1, skip=None):
        super().__init__()

        self.args = args
        self.hidden_dim = hidden_dim if hidden_dim is not None else self.args.hidden_dim
        self.num_layers = num_layers if num_layers is not None else self.args.num_layers
        self.output_dim = output_dim
        if self.args.periodic:
            self.activation = torch.sin
        else:
            self.activation = F.relu

        self.skip = skip if skip is not None else not self.args.noskip

        self.bias = True
        
        layers = []
        for i in range(self.num_layers):
            if i == 0: 
                layers.append(nn.Linear(input_dim, self.hidden_dim, bias=self.bias))
            elif i == 2 and self.skip:
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim-input_dim, bias=self.bias))
            #elif i == 3 and self.skip:
            #    layers.append(nn.Linear(self.hidden_dim+input_dim, self.hidden_dim, bias=self.bias))
            else:
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias))

        self.layers = nn.ModuleList(layers)
        self.lout = nn.Linear(self.hidden_dim, self.output_dim, bias=self.bias)

        init_decoder(self, args)

    def forward(self, x):
        N = x.shape[0]

        for i, l in enumerate(self.layers):
            if i == 0:
                h = self.activation(l(x))
            elif i == 2 and self.skip:
                h = self.activation(l(h))
                h = torch.cat([x, h], dim=-1)
            #elif i == 3 and self.skip:
            #    h = torch.cat([x, h], dim=-1)
            #    h = self.activation(l(h))
            else:
                h = self.activation(l(h))
        
        out = self.lout(h)
        
        return out

