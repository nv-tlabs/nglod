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
import torch.nn as nn
import torch.nn.functional as F
from lib.diffutils import gradient

def l1_loss(pred, gts):
    return torch.abs(pred - gts)

def l2_loss(pred, gts):
    return F.mse_loss(pred, gts, reduction='none')

def abs_l2_loss(pred, gts):
    return F.mse_loss(torch.abs(pred), torch.abs(gts), reduction='none')

def thresh_l2_loss(pred, gts):
    pred = torch.max(torch.cat([pred - 0.0, torch.zeros_like(pred)], dim=0), dim=0)[0]
    gts = torch.max(torch.cat([gts - 0.0, torch.zeros_like(gts)], dim=0), dim=0)[0]
    return F.mse_loss(pred, gts, reduction='none')
    
    #l2_loss = F.mse_loss(pred, gts, reduction='none').unsqueeze(0) - 0.0003
    #return torch.max(torch.cat([l2_loss, torch.zeros_like(l2_loss)], dim=0), dim=0)[0]

def reverse_l2_loss(pred, gts):
    return F.mse_loss(pred, -gts, reduction='none')

def relative_l2_loss(pred, gts):
    return l2_loss(pred, gts) / (torch.abs(gts) + 1e-3)

def relative_l2_lossv2(pred, gts):
    return l2_loss(pred, gts) / (torch.abs(gts) + 1e-6)

def multi_loss(preds, gts, f):
    # preds = [pred, pred, ... pred]
    loss = torch.zeros_like(gts)
    for pred in preds:
        loss += f(pred, gts)
    return loss

def log_loss(pred, gts, temp=1e-30):
    temp = 1e-30
    log_loss = -torch.log(torch.abs(gts)*(1.0 - temp) + temp) \
        * l2_loss(pred, gts)
    return log_loss

def eikonal_loss(x, f, gts, normal_mode='autodiff'):
    # Generate points between [-1,1]
    sample_pts = torch.rand([x.shape[0], x.shape[-1]], device=x.device) * 3.0 - 1.5
    #sample_pts = torch.rand_like(x, device=x.device) * 2.0 - 1.0
    sample_pts = sample_pts.requires_grad_(True)
    sample_grad = gradient(sample_pts, f, normal_mode)
    #return (sample_grad.norm(dim=-1) - 1.0)**2.0 * 0.01
    return 0.1 * (sample_grad.norm(dim=-1) - 1.0)**2.0
    #return 0.1 * (torch.clamp(sample_grad.norm(dim=-1) - 1.0, min=0.0))**2.0

def sparsity_loss(x, f, gts):
    sample_pts = torch.rand([x.shape[0], 3], device=x.device) * 3.0 - 1.5
    d = f(sample_pts) 
    return torch.exp(-100.0 * torch.abs(d))

def eikonal_loss_pre(x, f, grad, normal_mode='autodiff'):
    return 0.1 * (grad.norm(dim=-1) - 1.0)**2.0
    #import pdb; pdb.set_trace()
    #return 0.1 * (torch.clamp(grad.norm(dim=-1) - 1.0, min=0.0))**2.0


def multilayer_eikonal_loss(x, f, gts):
    sample_pts = torch.rand_like(x, device=x.device) * 2.0 - 1.0
    sample_pts = sample_pts.requires_grad_(True)
    sample_grads = gradient(sample_pts, f, method='multilayer')
    loss = 0.0
    for sample_grad in sample_grads:
        loss += 0.1 * (sample_grad.norm(dim=-1) - 1.0)**2.0
        #loss += 0.1 * F.mse_loss(sample_grad, F.normalize(gts, dim=-1))
    return loss

def zero_loss(pred, gts):
    zero_idx = torch.abs(gts[:, 0]) < 3e-5
    return pred[zero_idx] ** 2

def sign_loss(pred, gts):
    # Negative if there is diff, positive if signs match
    hinge = -(d[~zero_idx] * gts[~zero_idx])
    sign_loss = torch.max(torch.cat(
        [torch.zeros_like(hinge), hinge], dim=1), dim=1)[0]

def gradient_loss(x, f, gts, normal_mode='autodiff'):
    n = gradient(x, f, normal_mode)
    #return (1 - F.cosine_similarity(n, F.normalize(gts))) * 1.0
    return F.mse_loss(n, F.normalize(gts, dim=-1)) * 1.0


