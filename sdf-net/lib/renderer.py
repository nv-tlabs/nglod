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
from scipy.ndimage import gaussian_filter
from PIL import Image

from lib.utils import PerfTimer, setparam
from lib.diffutils import gradient
from lib.geoutils import normalize, look_at, compute_normal, matcap_sampler, spherical_envmap
from lib.geoutils import normalized_slice
from lib.PsDebugger import PsDebugger

from lib.tracer import *

class Renderer():
    """Main renderer class."""

    def __init__(self, 
        tracer, # Required to set a tracer
        args                = None,
        render_res   : list = None, # [w, h]
        camera_clamp : list = None, # [near, far]
        camera_proj  : str  = None, # one of ['persp', 'ortho']
        render_batch : int  = None, # -1 for no batching
        shading_mode : str  = None, # one of ['matcap', 'rb']
        matcap_path  : str  = None, # set if shadming mode = matcap
        shadow       : bool = None, 
        ao           : bool = None, 
        perf         : bool = None,
        device              = None
    ):
        self.args = args
        self.camera_clamp = tracer.camera_clamp
        self.render_res = setparam(args, render_res, 'render_res')
        self.camera_proj = setparam(args, camera_proj, 'camera_proj')
        self.render_batch = setparam(args, render_batch, 'render_batch')
        self.shading_mode = setparam(args, shading_mode, 'shading_mode')
        self.matcap_path = setparam(args, matcap_path, 'matcap_path')
        self.shadow = setparam(args, shadow, 'shadow')
        self.ao = setparam(args, ao, 'ao')
        self.perf = setparam(args, perf, 'perf')
        self.device = setparam(args, device, 'device')
        
        if self.device is None:
            self.device = 'cuda'

        self.width, self.height = self.render_res

        self.tracer = tracer
        
        # TODO: Fix this
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

    def render_lookat(self, 
            net, 
            f           = [0,0,1], 
            t           = [0,0,0], 
            fov          = 30.0, 
            camera_proj = 'persp',  
            device      = 'cuda', 
            mm          = None
        ):
        # Generate the ray origins and directions, from camera parameters
        ray_o, ray_d = look_at(f, t, self.width, self.height, 
                               fov=fov, mode=camera_proj, device=device)
        # Rotate the camera into model space
        if mm is not None:
            mm = mm.to('cuda')
            ray_o = torch.mm(ray_o, mm)
            ray_d = torch.mm(ray_d, mm)
        return self.render(net, ray_o, ray_d)


    def render(self, net, ray_o, ray_d):
        # Differentiable Renderer
        timer = PerfTimer(activate=self.perf)
        if self.perf:
            _time = time.time()

        # Perform sphere tracing, obtain the final hit points
        with torch.no_grad():
            #with torch.cuda.amp.autocast():
            if self.render_batch > 0:
                ray_os = torch.split(ray_o, self.render_batch)
                ray_ds = torch.split(ray_d, self.render_batch)
                rb = RenderBuffer()
                for origin, direction in zip(ray_os, ray_ds):
                    rb  += self.tracer(net, origin, direction)
            else:
                rb = self.tracer(net, ray_o, ray_d)

        ######################
        # Shadow Rendering
        ######################

        if self.shadow:
            rb.shadow = torch.zeros_like(rb.depth)[:,0].bool().to(ray_o.device)
            with torch.no_grad():
                plane_hit = torch.zeros_like(rb.depth)[:,0].bool().to(ray_o.device)
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
                light_o = torch.FloatTensor([[-1.5,4.5,-1.5]]).to(ray_o.device)
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
        rb.relative_depth = torch.clamp(rb.depth, 0.0, self.camera_clamp[1]) / self.camera_clamp[1]

        ######################
        # Shading Rendering
        ######################
        
        # This is executed if the tracer does not handle RGB
        if self.shading_mode == 'rb' and rb.rgb is None: 
            net(rb.x)
            #rb.rgb = net.render(rb.x, -ray_d, F.normalize(rb.normal))
            rb.rgb = net.render(rb.x, ray_d, F.normalize(rb.normal))[...,:3]

        ######################
        # Ambient Occlusion
        ######################

        #noise = torch.rand_like(t);
        if self.ao:
            acc = torch.zeros_like(rb.depth).to(ray_o.device)
            r = torch.zeros_like(rb.depth).to(ray_o.device)
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

                    if self.shadow:
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

        if self.perf:
            print("Time Elapsed:{:.4f}".format(time.time() - _time))
        
        return rb
    
    def normal_slice(self, net, dim=0, depth=0.0):
        pts = normalized_slice(self.width, self.height, dim=dim, depth=depth, device=self.device).reshape(-1,3)
        normal = (F.normalize(gradient(pts, net).detach()) + 1.0) / 2.0
        normal = normal.reshape(self.width, self.height, 3).cpu().numpy()
        return normal

    def rgb_slice(self, net, dim=0, depth=0.0):
        pts = normalized_slice(self.width, self.height, dim=dim, depth=depth, device=self.device).reshape(-1,3)
        normal = F.normalize(gradient(pts, net).detach())
        view = torch.FloatTensor([1.0,0.0,0.0]).to(self.device).expand(normal.shape)
        with torch.no_grad():
            rgb = net.render(pts.reshape(-1,3), view, normal)[...,:3]
        rgb = rgb.reshape(self.width, self.height, 3).cpu().numpy()
        return rgb
    
    def sdf_slice(self, net, dim=0, depth=0):
        pts = normalized_slice(self.width, self.height, dim=dim, depth=depth, device=self.device)

        d = torch.zeros(self.width * self.height, 1, device=pts.device)
        with torch.no_grad():
            d = net.sdf(pts.reshape(-1,3))

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
    
    def sdf_grad_slice(self, net, dim=0, depth=0):
        pts = normalized_slice(self.width, self.height, dim=dim, depth=depth, device=self.device)

        d = torch.zeros(self.width * self.height, 1, device=pts.device)

        g = gradient(pts.reshape(-1,3), net).detach()
        d = g.norm(2, dim=-1)
        d = d.reshape(self.width, self.height, 1)

        return d.cpu().numpy()
    
    def shade_tensor(self, net, f=[0,0,1], t=[0,0,0], fov=30.0, mm=None):
        """Non-differentiable shading for visualization.
        
        Args:
            f (list[f,f,f]): camera from
            t (list[f,f,f]): camera to
            fov: field of view
            mm: model transformation matrix
        """
        rb = self.render_lookat(net, f=f, t=t, fov=fov, mm=mm)
        # Shade the image
        if self.shading_mode == 'matcap':
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
        elif self.shading_mode == 'rb':
            assert rb.rgb is not None and "No rgb in buffer; change shading-mode"
            pass
        else:
            raise NotImplementedError
        # Use segmentation
        rb.normal[~rb.hit[...,0]] = 1.0
        rb.rgb[~rb.hit[...,0]] = 1.0

        # Add secondary effects
        if self.shadow:
            shadow_map = torch.clamp((1.0 - rb.shadow.float() + 0.9), 0.0, 1.0).cpu().numpy()[...,0]
            shadow_map = torch.from_numpy(gaussian_filter(shadow_map, sigma=2)).unsqueeze(-1)
            rb.rgb[...,:3] *= shadow_map.cuda()

        if self.ao:
            rb.rgb[...,:3] *= rb.ao        
        return rb

    def shade_images(self, net, f=[0,0,1], t=[0,0,0], fov=30.0, aa=1, mm=None):
        """
        Invokes the renderer and outputs images.

        Args:
            f (list[f,f,f]): camera from
            t (list[f,f,f]): camera to
            fov: field of view
            mm: model transformation matrix
        """
        if mm is None:
            mm = torch.eye(3)
        
        if aa > 1:
            rblst = [] 
            for _ in range(aa):
                rblst.append(self.shade_tensor(net, f=f, t=t, fov=fov, mm=mm))
            rb = RenderBuffer.mean(*rblst)
        else:
            rb = self.shade_tensor(net, f=f, t=t, fov=fov, mm=mm)
        rb = rb.cpu().transpose()
        return rb

