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


import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import time

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import moviepy.editor as mpy
from scipy.spatial.transform import Rotation as R
import pyexr

from lib.renderer import Renderer
from lib.models import *
from lib.tracer import *
from lib.options import parse_options
from lib.geoutils import sample_unif_sphere, sample_fib_sphere, normalized_slice


def write_exr(path, data):
    pyexr.write(path, data,
                channel_names={'normal': ['X', 'Y', 'Z'],
                               'x': ['X', 'Y', 'Z'],
                               'view': ['X', 'Y', 'Z']},
                precision=pyexr.HALF)


if __name__ == '__main__':

    # Parse
    parser = parse_options(return_parser=True)
    app_group = parser.add_argument_group('app')
    app_group.add_argument('--img-dir', type=str, default='_results/render_app/imgs',
                           help='Directory to output the rendered images')
    app_group.add_argument('--render-2d', action='store_true',
                           help='Render in 2D instead of 3D')
    app_group.add_argument('--exr', action='store_true',
                           help='Write to EXR')
    app_group.add_argument('--r360', action='store_true',
                           help='Render a sequence of spinning images.')
    app_group.add_argument('--rsphere', action='store_true',
                           help='Render around a sphere.')
    app_group.add_argument('--nb-poses', type=int, default=64,
                           help='Number of poses to render for sphere rendering.')
    app_group.add_argument('--cam-radius', type=float, default=4.0,
                           help='Camera radius to use for sphere rendering.')
    app_group.add_argument('--disable-aa', action='store_true',
                           help='Disable anti aliasing.')
    app_group.add_argument('--export', type=str, default=None,
                           help='Export model to C++ compatible format.')
    app_group.add_argument('--rotate', type=float, default=None,
                           help='Rotation in degrees.')
    app_group.add_argument('--depth', type=float, default=0.0,
                           help='Depth of 2D slice.')
    args = parser.parse_args()

    # Pick device
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Get model name
    if args.pretrained is not None:
        name = args.pretrained.split('/')[-1].split('.')[0]
    else:
        assert False and "No network weights specified!"

    net = globals()[args.net](args)
    if args.jit:
        net = torch.jit.script(net)

    net.load_state_dict(torch.load(args.pretrained))

    net.to(device)
    net.eval()

    print("Total number of parameters: {}".format(sum(p.numel() for p in net.parameters())))

    if args.export is not None:
        net = SOL_NGLOD(net)

        net.save(args.export)
        sys.exit()

    if args.sol:
        net = SOL_NGLOD(net)

    if args.lod is not None:
        net.lod = args.lod

    # Make output directory
    ins_dir = os.path.join(args.img_dir, name)
    if not os.path.exists(ins_dir):
        os.makedirs(ins_dir)

    for t in ['normal', 'rgb', 'exr']:
        _dir = os.path.join(ins_dir, t)
        if not os.path.exists(_dir):
            os.makedirs(_dir)

    tracer = globals()[args.tracer](args)
    renderer = Renderer(tracer, args=args, device=device)

    if args.rotate is not None:
        rad = np.radians(args.rotate)
        model_matrix = torch.FloatTensor(R.from_rotvec(rad * np.array([0, 1, 0])).as_matrix())
    else:
        model_matrix = torch.eye(3)

    if args.r360:
        for angle in np.arange(0, 360, 2):
            rad = np.radians(angle)
            model_matrix = torch.FloatTensor(R.from_rotvec(rad * np.array([0, 1, 0])).as_matrix())

            out = renderer.shade_images(net=net,
                                        f=args.camera_origin,
                                        t=args.camera_lookat,
                                        fov=args.camera_fov,
                                        aa=not args.disable_aa,
                                        mm=model_matrix)

            data = out.float().numpy().exrdict()

            idx = int(math.floor(100 * angle))

            if args.exr:
                write_exr('{}/exr/{:06d}.exr'.format(ins_dir, idx), data)

            img_out = out.image().byte().numpy()
            Image.fromarray(img_out.rgb).save('{}/rgb/{:06d}.png'.format(ins_dir, idx), mode='RGB')
            Image.fromarray(img_out.normal).save('{}/normal/{:06d}.png'.format(ins_dir, idx), mode='RGB')

    elif args.rsphere:
        views = sample_fib_sphere(args.nb_poses)
        cam_origins = args.cam_radius * views
        for p, cam_origin in enumerate(cam_origins):
            out = renderer.shade_images(net=net,
                                        f=cam_origin,
                                        t=args.camera_lookat,
                                        fov=args.camera_fov,
                                        aa=not args.disable_aa,
                                        mm=model_matrix)

            data = out.float().numpy().exrdict()

            if args.exr:
                write_exr('{}/exr/{:06d}.exr'.format(ins_dir, p), data)

            img_out = out.image().byte().numpy()
            Image.fromarray(img_out.rgb).save('{}/rgb/{:06d}.png'.format(ins_dir, p), mode='RGB')
            Image.fromarray(img_out.normal).save('{}/normal/{:06d}.png'.format(ins_dir, p), mode='RGB')

    else:

        out = renderer.shade_images(net=net,
                                    f=args.camera_origin,
                                    t=args.camera_lookat,
                                    fov=args.camera_fov,
                                    aa=not args.disable_aa,
                                    mm=model_matrix)

        data = out.float().numpy().exrdict()

        if args.render_2d:
            depth = args.depth
            data['sdf_slice'] = renderer.sdf_slice(depth=depth)
            data['rgb_slice'] = renderer.rgb_slice(depth=depth)
            data['normal_slice'] = renderer.normal_slice(depth=depth)

        if args.exr:
            write_exr(f'{ins_dir}/out.exr', data)

        img_out = out.image().byte().numpy()

        Image.fromarray(img_out.rgb).save('{}/{}_rgb.png'.format(ins_dir, name), mode='RGB')
        Image.fromarray(img_out.depth).save('{}/{}_depth.png'.format(ins_dir, name), mode='RGB')
        Image.fromarray(img_out.normal).save('{}/{}_normal.png'.format(ins_dir, name), mode='RGB')
        Image.fromarray(img_out.hit).save('{}/{}_hit.png'.format(ins_dir, name), mode='L')
