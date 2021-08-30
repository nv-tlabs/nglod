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

from lib.utils import PerfTimer

# Differentiable Operators for General Functions

def gradient(x, f, method='autodiff'):
    """Compute gradient.
    """
    if method == 'autodiff':
        with torch.enable_grad():
            x = x.requires_grad_(True)
            y = f(x)
            grad = torch.autograd.grad(y, x, 
                                       grad_outputs=torch.ones_like(y), create_graph=True)[0]
    elif method == 'tetrahedron':
        h = 1.0 / (64.0 * 3.0)
        k0 = torch.tensor([ 1.0, -1.0, -1.0], device=x.device, requires_grad=False)
        k1 = torch.tensor([-1.0, -1.0,  1.0], device=x.device, requires_grad=False)
        k2 = torch.tensor([-1.0,  1.0, -1.0], device=x.device, requires_grad=False)
        k3 = torch.tensor([ 1.0,  1.0,  1.0], device=x.device, requires_grad=False)
        h0 = torch.tensor([ h, -h, -h], device=x.device, requires_grad=False)
        h1 = torch.tensor([-h, -h,  h], device=x.device, requires_grad=False)
        h2 = torch.tensor([-h,  h, -h], device=x.device, requires_grad=False)
        h3 = torch.tensor([ h,  h,  h], device=x.device, requires_grad=False)
        h0 = x + h0
        h1 = x + h1
        h2 = x + h2
        h3 = x + h3
        h0 = h0.detach()
        h1 = h1.detach()
        h2 = h2.detach()
        h3 = h3.detach()
        h0 = k0 * f(h0)
        h1 = k1 * f(h1)
        h2 = k2 * f(h2)
        h3 = k3 * f(h3)
        grad = (h0+h1+h2+h3) / (h*4.0)
    elif method == 'finitediff':
        min_dist = 1.0/(64.0 * 3.0)
        eps_x = torch.tensor([min_dist, 0.0, 0.0], device=x.device)
        eps_y = torch.tensor([0.0, min_dist, 0.0], device=x.device)
        eps_z = torch.tensor([0.0, 0.0, min_dist], device=x.device)

        grad = torch.cat([f(x + eps_x) - f(x - eps_x),
                          f(x + eps_y) - f(x - eps_y),
                          f(x + eps_z) - f(x - eps_z)], dim=-1)
        grad = grad / (min_dist*2.0)
    elif method == 'multilayer':
        # TODO: Probably remove this
        grad = []
        with torch.enable_grad():
            _y = f.sdf(x, return_lst=True)
            for i in range(len(_y)):
                _grad = torch.autograd.grad(_y[i], x, 
                                           grad_outputs=torch.ones_like(_y[i]), create_graph=True)[0]
                grad.append(_grad)
        return grad
    else:
        raise NotImplementedError

    return grad

# from https://github.com/krrish94/nerf-pytorch
def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)

