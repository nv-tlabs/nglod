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

import time

import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import matplotlib.pyplot
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from PIL import Image

from .utils import PerfTimer
from .diffutils import positional_encoding
from .diffutils import gradient
from .geoutils import normalize, look_at, compute_normal, matcap_sampler, spherical_envmap
from .geoutils import sample_unif_sphere
from .geoutils import normalized_slice
from .PsDebugger import PsDebugger
from sol_nglod import aabb

from .tracer import *

### Renderer

class Renderer(nn.Module):
    """Main renderer class."""

    def __init__(self, args, device, sdf_net=None, res=None, camera_proj=None):
        """Constructor.

        Args:
            args (Namespace): CLI arguments
            device (torch.device): 'cpu' or 'gpu'
            sdf_net (nn.Module): SDF network
        """

        super().__init__()
        self.args = args
        self.device = device
        self.matcap_path = args.matcap_path
        self.width = args.render_res[0]
        self.height = args.render_res[1]
        if res is not None:
            self.width, self.height = res
        self.perf = args.perf
        self.clamp = args.camera_clamp
        self.camera_proj = args.camera_proj
        if camera_proj is not None:
            self.camera_proj = camera_proj
        self.MIN_DIST = 0.0003
        self.step_size = args.step_size

        # Configure
        self.sdf_net = None
        if sdf_net:
            self.sdf_net = sdf_net
        self.tracer = globals()[self.args.tracer](clamp=self.clamp, sol=args.sol, step_size=args.step_size)
        
        if self.args.shadow:
            if self.args.ground_height:
                self.min_y = self.args.ground_height
            else:
               self.surface = self.tracer.sample_surface(1000000, self.sdf_net)
               self.min_x   = torch.min(self.surface[:,0]).item()
               self.min_y   = torch.min(self.surface[:,1]).item()
               self.min_z   = torch.min(self.surface[:,2]).item()
               self.max_x   = torch.max(self.surface[:,0]).item()
               self.max_y   = torch.max(self.surface[:,1]).item()
               self.max_z   = torch.max(self.surface[:,2]).item()
        
    def forward(self, net,  # Probably should make a camera class to make this simpler
            f=[0,0,1], 
            t=[0,0,0], 
            fv=30.0, 
            mm=None,
            ray_o=None, # If rays are passed in, the lookat is ignored
            ray_d=None,
            wh=None
        ):
        # Differentiable Renderer Dispatcher

        # Important note:
        # The tensors for depth, relative_depth, segmentation, view aren't actually differentiable.
        # Only the normal tensor is differentiable, because the normal information relies on the
        # net, whereas other quantities like depth / segmentation rely on the rendering process, which 
        # in itself isn't actually differentiable.
        # Instead, the depth and segmentation cues are used by another loss function, which can then
        # impose a sort of a "unsupervised" loss based on the segmentation cues.
     
        if ray_o is None:
            # Generate the ray origins and directions, from camera parameters
            ray_o, ray_d = look_at(f, t, self.width, self.height, 
                                     fv=fv, mode=self.camera_proj, device=self.device)
            # Rotate the camera into model space
            if mm is not None:
                mm = mm.to(self.device)
                ray_o = torch.mm(ray_o, mm)
                ray_d = torch.mm(ray_d, mm)
        else:
            assert ray_d is not None and "Ray direction missing"
    
        if wh is not None:
            self.width, self.height = wh

        return self.render(net, ray_o, ray_d)

    def render(self, net, ray_o, ray_d):
        # Differentiable Renderer

        timer = PerfTimer(activate=self.args.perf)
        if self.args.perf:
            _time = time.time()

        # Perform sphere tracing, obtain the final hit points
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                if self.args.render_batch > 0:
                    ray_os = torch.split(ray_o, self.args.render_batch)
                    ray_ds = torch.split(ray_d, self.args.render_batch)
                    rb = RenderBuffer()
                    for origin, direction in zip(ray_os, ray_ds):
                        rb  += self.tracer(net, origin, direction)
                else:
                    rb = self.tracer(net, ray_o, ray_d)

        ######################
        # Shadow Rendering
        ######################

        if self.args.shadow:
            rb.shadow = torch.zeros_like(rb.depth)[:,0].bool().to(self.device)
            with torch.no_grad():
                plane_hit = torch.zeros_like(rb.depth)[:,0].bool().to(self.device)
                rate = -ray_d[:,1] # check negative sign probably lol
                plane_hit[torch.abs(rate) < 0.00001] = False
                delta = ray_o[:,1] - (self.min_y)
                plane_t = delta / rate
                plane_hit[(plane_t > 0) & (plane_t < 500)] = True
                plane_hit = plane_hit & (plane_t < rb.depth[...,0])

                rb.hit = rb.hit & ~plane_hit

                rb.depth[plane_hit] = plane_t[plane_hit].unsqueeze(1)
                rb.x[plane_hit] = ray_o[plane_hit] + ray_d[plane_hit] * plane_t[plane_hit].unsqueeze(1)
                rb.normal[plane_hit] = 0
                rb.normal[plane_hit, 1] = 1

                # x is shadow ray origin
                light_o = torch.FloatTensor([[-1.5,4.5,-1.5]]).to(self.device)
                shadow_ray_o = rb.x + 0.1 * rb.normal
                shadow_ray_d = torch.zeros_like(rb.x).normal_(0.0, 0.01) + \
                        light_o - shadow_ray_o
                shadow_ray_d = F.normalize(shadow_ray_d, dim=1)
                
                light_hit = ((shadow_ray_d * rb.normal).sum(-1) > 0.0)

                rb.shadow = self.tracer(net, shadow_ray_o, shadow_ray_d).hit
                #rb.shadow[~plane_hit] = 0.0
                rb.shadow[~light_hit] = 0.0
                #rb.hit = rb.hit | plane_hit


        ######################
        # Relative Depth
        ######################
        rb.relative_depth = torch.clip(rb.depth, 0.0, self.clamp[1]) / self.clamp[1]

        ######################
        # Shading Rendering
        ######################
        
        if self.args.shading_mode == 'rb' and rb.rgb is None:
            net(rb.x)
            rb.rgb = net.render(rb.x, -ray_d, F.normalize(rb.normal))

        def _vis():
            psd = PsDebugger()
            #psd.register_point_cloud("ray_o", rb.ray_o)
            #psd.add_vector_quantity("ray_o", "ray_d", rb.ray_d)
            psd.register_point_cloud("x", rb.x)
            psd.add_vector_quantity("x", "normal", rb.normal)
            psd.add_color_quantity("x", "rgb", rb.rgb)
            psd.add_surface_mesh("obj", "/home/ttakikawa/datasets/pineapple_mesh/models/pineapple.obj")
            psd.show()
        #_vis()

        ######################
        # Ambient Occlusion
        ######################

        #noise = torch.rand_like(t);
        if self.args.ao:
            acc = torch.zeros_like(rb.depth).to(self.device)
            r = torch.zeros_like(rb.depth).to(self.device)
            with torch.no_grad():
                weight = 3.5
                for i in range(40):
                    
                    # Visual constants
                    ao_width = 0.1
                    _d = ao_width * 0.25 * (float(i+1) / float(40+1)) ** 1.6
                    q = rb.x + rb.normal * _d

                    # AO for surface
                    with torch.no_grad():
                        r[rb.hit] = net(q[rb.hit])

                    if self.args.shadow:
                        net_d = net(q[plane_hit])
                        plane_d = torch.zeros_like(net_d) + _d
                        r[plane_hit] = torch.min(torch.cat([net_d, plane_d], dim=-1), dim=-1, keepdim=True)[0]
                        acc[plane_hit] += 3.5 * F.relu(_d - r[plane_hit] - 0.0015)
                    acc[rb.hit] += 3.5 * F.relu(_d - r[rb.hit] - 0.0015)
                    weight *= 0.84
        
            rb.ao = torch.clamp(1.0 - acc,  0.1, 1.0)
            rb.ao = rb.ao * rb.ao

        rb.view = ray_d
        rb = rb.reshape(self.width, self.height, -1) 

        if self.args.perf:
            print("Time Elapsed:{:.4f}".format(time.time() - _time))
        
        return rb
    
    def normal_slice(self, dim=0, depth=0.0):
        pts = normalized_slice(self.width, self.height, dim=dim, depth=depth, device=self.device).reshape(-1,3)
        normal = (F.normalize(gradient(pts, self.sdf_net).detach()) + 1.0) / 2.0
        normal = normal.reshape(self.width, self.height, 3).cpu().numpy()
        return normal

    def rgb_slice(self, dim=0, depth=0.0):
        pts = normalized_slice(self.width, self.height, dim=dim, depth=depth, device=self.device).reshape(-1,3)
        normal = F.normalize(gradient(pts, self.sdf_net).detach())
        view = torch.FloatTensor([1.0,0.0,0.0]).to(self.device).expand(normal.shape)
        with torch.no_grad():
            rgb = self.sdf_net.render(pts.reshape(-1,3), view, normal)
        rgb = rgb.reshape(self.width, self.height, 3).cpu().numpy()
        return rgb
    
    def sdf_slice(self, dim=0, depth=0):
        pts = normalized_slice(self.width, self.height, dim=dim, depth=depth, device=self.device)

        with torch.no_grad():
            d = self.sdf_net(pts.reshape(-1,3))
        d = d.reshape(self.width, self.height, 1)

        d = d.squeeze().cpu().numpy()
        dpred = d
        d = np.clip((d + 1.0) / 2.0, 0.0, 1.0)
        blue = np.clip((d - 0.5)*2.0, 0.0, 1.0)
        yellow = 1.0 - blue
        vis = np.zeros([*d.shape, 3])
        vis[...,2] = blue
        vis += yellow[...,np.newaxis] * np.array([0.4, 0.3, 0.0])
        vis += 0.2
        vis[d - 0.5 < 0] = np.array([1.0, 0.38, 0.0])
        for i in range(50):
            vis[np.abs(d - 0.02*i) < 0.0015] = 0.8
        vis[np.abs(d - 0.5) < 0.004] = 0.0
        return vis
    
    def shade_tensor(self, f=[0,0,1], t=[0,0,0], fv=30.0, mm=None):
        """Non-differentiable shading for visualization.
        
        Args:
            f (list[f,f,f]): camera from
            t (list[f,f,f]): camera to
            fv: field of view
            mm: model transformation matrix
        """

        rb = self(self.sdf_net, f, t, fv, mm)
        rb = rb.detach()
        
        # Shade the image
        if self.args.shading_mode == 'matcap':
            matcap = matcap_sampler(self.matcap_path)
            matcap_normal = rb.normal.clone()
            matcap_view = rb.view.clone()
            if mm is not None:
                mm = mm.to(self.device)
                #matcap_normal = torch.mm(matcap_normal.reshape(-1, 3), mm.transpose(1,0))
                #matcap_normal = matcap_normal.reshape(self.width, self.height, 3)
                matcap_view = torch.mm(matcap_view.reshape(-1, 3), mm.transpose(1,0))
                matcap_view = matcap_view.reshape(self.width, self.height, 3)
            vN = spherical_envmap(matcap_view, matcap_normal).cpu().numpy()
            rb.rgb = torch.FloatTensor(matcap(vN).reshape(self.width, self.height, -1))[...,:3].cuda() / 255.0
        elif self.args.shading_mode == 'rb':
            assert rb.rgb is not None and "No rgb in buffer; change shading-mode"
            pass
        else:
            raise NotImplementedError
        
        # Use segmentation
        rb.normal[~rb.hit[...,0]] = 1.0
        rb.rgb[~rb.hit[...,0]] = 1.0

        # Add secondary effects
        if self.args.shadow:
            shadow_map = torch.clamp((1.0 - rb.shadow.float() + 0.9), 0.0, 1.0).cpu().numpy()[...,0]
            shadow_map = torch.from_numpy(gaussian_filter(shadow_map, sigma=2)).unsqueeze(-1)
            rb.rgb[...,:3] *= shadow_map.cuda()

        if self.args.ao:        
            rb.rgb[...,:3] *= rb.ao        

        return rb

    def shade_images(self, f=[0,0,1], t=[0,0,0], fv=30.0, aa=False, mm=None):
        """
        Invokes the renderer and outputs images.

        Args:
            f (list[f,f,f]): camera from
            t (list[f,f,f]): camera to
            fv: field of view
            mm: model transformation matrix
        """

        if mm is None:
            mm = torch.eye(3)
        
        if aa:
            rblst = [] 
            for _ in range(4):
                rblst.append(self.shade_tensor(f=f, t=t, fv=fv, mm=mm))
            rb = RenderBuffer.mean(*rblst)
        else:
            rb = self.shade_tensor(f=f, t=t, fv=fv, mm=mm)
        
        rb = rb.cpu().transpose()

        return rb

