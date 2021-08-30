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

from lib.utils import setparam

class BaseTracer(object):
    """Virtual base class for tracer"""

    def __init__(self,
        args                 = None,
        camera_clamp : list  = None,
        step_size    : float = None,
        grad_method  : str   = None,
        num_steps    : int   = None, # samples for raymaching, iterations for sphere trace
        min_dis      : float = None): 

        self.args = args
        self.camera_clamp = setparam(args, camera_clamp, 'camera_clamp')
        self.step_size = setparam(args, step_size, 'step_size')
        self.grad_method = setparam(args, grad_method, 'grad_method')
        self.num_steps = setparam(args, num_steps, 'num_steps')
        self.min_dis = setparam(args, min_dis, 'min_dis')

        self.inv_num_steps = 1.0 / self.num_steps
        self.diagonal = np.sqrt(3) * 2.0
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, net, ray_o, ray_d):
        """Base implementation for forward"""
        raise NotImplementedError
