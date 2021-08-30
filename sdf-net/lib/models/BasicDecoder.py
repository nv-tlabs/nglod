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

from lib.utils import setparam

class BasicDecoder(nn.Module):
    def __init__(self, 
        input_dim, 
        output_dim, 
        activation,
        bias, 
        args       = None,
        num_layers = None, 
        hidden_dim = None, 
        skip       = None
    ):
        super().__init__()

        self.args = args
        self.num_layers = setparam(args, num_layers, 'num_layers')
        self.hidden_dim = setparam(args, hidden_dim, 'hidden_dim')
        self.skip = setparam(args, skip, 'skip')

        self.input_dim = input_dim
        self.output_dim = output_dim        
        self.activation = activation
        self.bias = bias
        
        layers = []
        for i in range(self.num_layers):
            if i == 0: 
                layers.append(nn.Linear(self.input_dim, self.hidden_dim, bias=self.bias))
            elif i == self.skip:
                layers.append(nn.Linear(self.hidden_dim+input_dim, self.hidden_dim, bias=self.bias))
            else:
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias))
        self.layers = nn.ModuleList(layers)
        self.lout = nn.Linear(self.hidden_dim, self.output_dim, bias=self.bias)
        
    def forward(self, x, return_h=False):
        N = x.shape[0]

        for i, l in enumerate(self.layers):
            if i == 0:
                h = self.activation(l(x))
            elif i == self.skip:
                h = self.activation(l(h))
                h = torch.cat([x, h], dim=-1)
            else:
                h = self.activation(l(h))
        
        out = self.lout(h)
        
        if return_h:
            return out, h
        else:
            return out

    def initialize(self, get_weight):
        ms = []
        for i, w in enumerate(self.layers):
            m = get_weight(w.weight)
            ms.append(m)
        for i in range(len(self.layers)):
            self.layers[i].weight = nn.Parameter(ms[i])
        m = get_weight(self.lout.weight)
        self.lout.weight = nn.Parameter(m)
        
