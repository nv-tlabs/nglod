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
import logging as log

import torch
import torch.nn as nn
import torch.nn.functional as F

from .BaseSDF import BaseSDF
from .utils import init_decoder

from .losses import *

from ..diffutils import gradient

class BaseLOD(BaseSDF):
    def __init__(self, args):
        super().__init__(args)
        self.num_lods = args.num_lods
        self.lod = None
        self.shrink_idx = (self.num_lods - 1)
        self.bias = True


    def grow(self):
        if self.shrink_idx > 0:
            self.shrink_idx -= 1
            log.info(f'Shrinking network by one layer: {self.shrink_idx + 1} to {self.shrink_idx}')

    def loss(self, writer=None, epoch=None):
        loss_val = torch.zeros_like(self.gts).squeeze()
        loss_dict = {}

        loss_dict['_l2_loss'] = 0
        for pred in self.loss_preds:
            loss_dict['_l2_loss'] = l2_loss(pred, self.gts).squeeze()

            for l in self.args.loss:
                if l not in loss_dict:
                    loss_dict[l] = 0
                if l == 'gradient_loss':
                    raise NotImplementedError
                    loss_dict[l] += globals()[l](self.inputs, self, self.grad).squeeze()
                elif l == 'eikonal_loss':
                    loss_dict[l] = globals()[l](self.inputs, self, self.grad).squeeze()
                else:
                    loss_dict[l] += globals()[l](pred, self.gts).squeeze()
                #if writer is not None:
                #    writer.add_scalar('Loss/{}'.format(l), _loss, epoch)
        return loss_dict

