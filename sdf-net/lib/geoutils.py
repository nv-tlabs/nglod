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
import torch.nn.functional as F
from PIL import Image
from scipy.interpolate import RegularGridInterpolator
import cv2


def sample_unif_sphere(n):
    """
    Sample uniformly random points on a sphere.
    """
    u = np.random.rand(2, n)
    z = 1 - 2*u[0,:]
    r = np.sqrt(1. - z * z)
    phi = 2 * np.pi * u[1,:]
    xyz = np.array([r * np.cos(phi), r * np.sin(phi), z]).transpose()
    return xyz


def sample_fib_sphere(n):
    """
    Evenly distributed points on sphere using Fibonnaci sequence.
    From <http://extremelearning.com.au/evenly-distributing-points-on-a-sphere>
    WARNING: Order is not randomized.
    """

    i = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2*i/n)
    golden_ratio = (1 + 5**0.5)/2
    theta = 2. * np.pi * i / golden_ratio
    xyz = np.array([np.cos(theta) * np.sin(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(phi)]).transpose()
    return xyz


def unique_corners(centers, feat):
    """
    Given a [N, 3] list of points, return the list of unique corners.
    """
    pts = centers
    N = pts.shape[0]
    _vs = torch.zeros(8, N, 3, device=pts.device).long()
    _vs[0] = torch.stack([pts[...,0]+0, pts[...,1]+0, pts[...,2]+0], dim=-1)
    _vs[1] = torch.stack([pts[...,0]+0, pts[...,1]+0, pts[...,2]+1], dim=-1)
    _vs[2] = torch.stack([pts[...,0]+0, pts[...,1]+1, pts[...,2]+0], dim=-1)
    _vs[3] = torch.stack([pts[...,0]+0, pts[...,1]+1, pts[...,2]+1], dim=-1)
    _vs[4] = torch.stack([pts[...,0]+1, pts[...,1]+0, pts[...,2]+0], dim=-1)
    _vs[5] = torch.stack([pts[...,0]+1, pts[...,1]+0, pts[...,2]+1], dim=-1)
    _vs[6] = torch.stack([pts[...,0]+1, pts[...,1]+1, pts[...,2]+0], dim=-1)
    _vs[7] = torch.stack([pts[...,0]+1, pts[...,1]+1, pts[...,2]+1], dim=-1)
    vs = torch.unique(_vs.reshape(-1, 3), dim=0)
    cfeats = feat[0, :, vs[:,2], vs[:,1], vs[:,0]].transpose(1,0)
    return vs, cfeats


def voxel_corners(pts, res=64):
    """
    Given a [N, 3] list of points, returns the [8, N, 3] list of voxel coordinates.
    """

    # Maps 0 -> 0, 1->res
    idx = ((pts + 1) / 2) * res

    idx_0 = idx.floor().int()
    idx_1 = idx.ceil().int()
    N = idx.shape[0]
    
    _vs = torch.zeros(8, N, 3, device=pts.device).long()
    _vs[0] = torch.stack([idx_0[...,0], idx_0[...,1], idx_0[...,2]], dim=-1)
    _vs[1] = torch.stack([idx_0[...,0], idx_0[...,1], idx_1[...,2]], dim=-1)
    _vs[2] = torch.stack([idx_0[...,0], idx_1[...,1], idx_0[...,2]], dim=-1)
    _vs[3] = torch.stack([idx_0[...,0], idx_1[...,1], idx_1[...,2]], dim=-1)
    _vs[4] = torch.stack([idx_1[...,0], idx_0[...,1], idx_0[...,2]], dim=-1)
    _vs[5] = torch.stack([idx_1[...,0], idx_0[...,1], idx_1[...,2]], dim=-1)
    _vs[6] = torch.stack([idx_1[...,0], idx_1[...,1], idx_0[...,2]], dim=-1)
    _vs[7] = torch.stack([idx_1[...,0], idx_1[...,1], idx_1[...,2]], dim=-1)
 
    xyzd = idx % 1.0

    return _vs, xyzd


def normalize(v, axis=-1, order=2):
    """Normalizes an arbitrary dimension of an array.
    More info: https://stackoverflow.com/questions/21030391/how-to-normalize-an-array-in-numpy

    Args:
        v (torch.Tensor): tensor
        axis (int): axis along which to normalize (default: last)\
        order (int): order of the norm to use (default: 2)
    """

    l2 = np.atleast_1d(np.linalg.norm(v, order, axis))
    l2[l2 == 0] = 1
    return v / np.expand_dims(l2, axis)


def normalized_grid(width, height):
    """Returns grid[x,y] -> coordinates for a normalized window.
    
    Args:
        width, height (int): grid resolution
    """

    # These are normalized coordinates
    # i.e. equivalent to 2.0 * (fragCoord / iResolution.xy) - 1.0
    window_x = np.linspace(-1, 1, num=width) * (width / height)
    window_x += np.random.rand(*window_x.shape) * (1. / width)
    window_y = np.linspace(1, -1, num=height)
    window_y += np.random.rand(*window_y.shape) * (1. / height)
    coord = np.array(np.meshgrid(window_x, window_y, indexing='xy')).transpose(2,1,0)

    return coord


def normalized_grid(width, height, device='cuda'):
    """Returns grid[x,y] -> coordinates for a normalized window.
    
    Args:
        width, height (int): grid resolution
    """

    # These are normalized coordinates
    # i.e. equivalent to 2.0 * (fragCoord / iResolution.xy) - 1.0
    window_x = torch.linspace(-1, 1, steps=width, device=device) * (width / height)
    window_x += torch.rand(*window_x.shape, device=device) * (1. / width)
    window_y = torch.linspace(1,- 1, steps=height, device=device)
    window_y += torch.rand(*window_y.shape, device=device) * (1. / height)
    coord = torch.stack(torch.meshgrid(window_x, window_y)).permute(1,2,0)
    return coord


def normalized_slice(width, height, dim=0, depth=0.0, device='cuda'):
    """Returns grid[x,y] -> coordinates for a normalized slice for some dim at some depth."""
    window = normalized_grid(width, height, device)
    depth_pts = torch.ones(width, height, 1, device=device) * depth

    if dim==0:
        pts = torch.cat([depth_pts, window[...,0:1], window[...,1:2]], dim=-1)
    elif dim==1:
        pts = torch.cat([window[...,0:1], depth_pts, window[...,1:2]], dim=-1)
    elif dim==2:
        pts = torch.cat([window[...,0:1], window[...,1:2], depth_pts], dim=-1)
    else:
        assert(False, "dim is invalid!")
    pts[...,1] *= -1
    return pts


def unnormalized_grid(width, height, device='cuda'):
    uv = np.mgrid[0:width, 0:height].astype(np.int32)
    uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
    return uv


def look_at(f, t, width, height, mode='ortho', fov=90.0, device='cuda'):
    """Vectorized look-at function, returns an array of ray origins and directions
    URL: https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
    """

    camera_origin = torch.FloatTensor(f).to(device)
    camera_view = F.normalize(torch.FloatTensor(t).to(device) - camera_origin, dim=0)
    camera_right = F.normalize(torch.cross(camera_view, torch.FloatTensor([0,1,0]).to(device)), dim=0)
    camera_up = F.normalize(torch.cross(camera_right, camera_view), dim=0)

    coord = normalized_grid(width, height, device=device)
    ray_origin = camera_right * coord[...,0,np.newaxis] * np.tan(np.radians(fov/2)) + \
                 camera_up * coord[...,1,np.newaxis] * np.tan(np.radians(fov/2)) + \
                 camera_origin + camera_view
    ray_origin = ray_origin.reshape(-1, 3)
    ray_offset = camera_view.unsqueeze(0).repeat(ray_origin.shape[0], 1)
    
    if mode == 'ortho': # Orthographic camera
        ray_dir = F.normalize(ray_offset, dim=-1)
    elif mode == 'persp': # Perspective camera
        ray_dir = F.normalize(ray_origin - camera_origin, dim=-1)
        ray_origin = camera_origin.repeat(ray_dir.shape[0], 1)
    else:
        raise ValueError('Invalid camera mode!')


    return ray_origin, ray_dir


def compute_normal(net, x, pred=None):
    """Computes surface normal at a point.
    
    Args:
        net (nn.Module): model
        x (torch.Tensor): 3D point where to compute the normal
        pred (bool): TODO
    """

    MIN_DIST = 0.0003
    zeros = torch.zeros_like(x[:,0,np.newaxis])
    eps = torch.ones_like(x[:,0,np.newaxis]) * MIN_DIST * 5
    eps_x = torch.cat((eps, zeros, zeros), axis=1)
    eps_y = torch.cat((zeros, eps, zeros), axis=1)
    eps_z = torch.cat((zeros, zeros, eps), axis=1)
    if pred is None:
        pred = lambda x : x
        
    with torch.no_grad():
        normal_x = net(pred(x + eps_x)) - net(pred(x - eps_x))
        normal_y = net(pred(x + eps_y)) - net(pred(x - eps_y))
        normal_z = net(pred(x + eps_z)) - net(pred(x - eps_z)) 

    # Probably should normalize here
    # TODO: PyTorch normalization
    return torch.cat((normal_x, normal_y, normal_z), axis=1)


def matcap_sampler(path, interpolate=True):
    """Fetches MatCap texture & converts to a interpolation function (if needed).
    
    Args:
        path (str): path to MatCap texture
        interpolate (bool): perform interpolation (default: True)
    """

    matcap = np.array(Image.open(path)).transpose(1,0,2)
    if interpolate:
        return RegularGridInterpolator((np.linspace(0, 1, matcap.shape[0]),
                                        np.linspace(0, 1, matcap.shape[1])), matcap)
    else:
        return matcap


def spherical_envmap(ray_dir, normal):
    """Computes UV-coordinates.
    
    Args:
        ray_dir (torch.Tensor): incoming ray direction
        normal (torch.Tensor): surface normal
    """
    # Input should be size [...,3]
    # Returns [N,2] # Might want to make this [...,2]
    
    # Probably should implement all this on GPU
    ray_dir_screen = ray_dir.clone()
    ray_dir_screen[...,2] *= -1
    ray_dir_normal_dot = torch.sum(normal * ray_dir_screen, dim=-1, keepdim=True)
    r = ray_dir_screen - 2.0 * ray_dir_normal_dot * normal
    r[...,2] -= 1.0
    m = 2.0 * torch.sqrt(torch.sum(r**2, dim=-1, keepdim=True))
    vN = (r[...,:2] / m) + 0.5
    vN = 1.0 - vN
    vN = vN[...,:2].reshape(-1, 2)
    vN = torch.clamp(vN, 0.0, 1.0)
    vN[torch.isnan(vN)] = 0
    return vN


def spherical_envmap_numpy(ray_dir, normal):
    """Computes UV-coordinates."""
    ray_dir_screen = ray_dir * np.array([1,1,-1])
    # Calculate reflection
    ray_dir_normal_dot = np.sum(normal * ray_dir_screen, axis=-1)[...,np.newaxis]
    r = ray_dir_screen - 2.0 * ray_dir_normal_dot * normal
    m = 2.0 * np.sqrt(r[...,0]**2 + r[...,1] **2 + (r[...,2]-1)**2)
    vN = (r[...,:2] / m[...,np.newaxis]) + 0.5
    vN = 1.0 - vN
    vN = vN[...,:2].reshape(-1, 2)
    vN = np.clip(vN, 0, 1)
    vN[np.isnan(vN)] = 0
    return vN

