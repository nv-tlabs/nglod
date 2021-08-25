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


import argparse
import pprint

### Default CLI options
# Apps should use these CLI options, and then 
# extend using parser.add_argument_group('app')

def parse_options(return_parser=False):
    # New CLI parser
    parser = argparse.ArgumentParser(description='Train deep implicit 3D geometry representations.')
    
    # Global arguments
    global_group = parser.add_argument_group('global')
    global_group.add_argument('--exp-name', type=str,
                              help='Experiment name.')
    global_group.add_argument('--perf', action='store_true',
                              help='Use profiling.')
    global_group.add_argument('--validator', type=str, default=None,
                              help='Run validation.')
    global_group.add_argument('--valid-only', action='store_true',
                              help='Run validation (and do not run training).')
    global_group.add_argument('--valid-every', type=int, default=1,
                             help='Frequency of running validation.')
    global_group.add_argument('--debug', action='store_true',
                              help='Utility argument for debug output and viz.')
    global_group.add_argument('--seed', type=int,
                              help='NumPy random seed.')
    global_group.add_argument('--ngc', action='store_true',
                              help='Use NGC arguments.')

    # Architecture for network
    net_group = parser.add_argument_group('net')
    net_group.add_argument('--net', type=str, default='OverfitSDF', 
                          help='The network architecture to be used.')
    net_group.add_argument('--jit', action='store_true',
                          help='Use JIT.')
    net_group.add_argument('--pos-enc', action='store_true', 
                          help='Use positional encoding.')
    net_group.add_argument('--feature-dim', type=int, default=32,
                          help='Feature map dimension')
    net_group.add_argument('--feature-size', type=int, default=4,
                          help='Feature map size (w/h)')
    net_group.add_argument('--joint-feature', action='store_true',
                          help='Use joint features')
    net_group.add_argument('--num-layers', type=int, default=1,
                          help='Number of layers for the decoder')
    net_group.add_argument('--num-lods', type=int, default=1,
                          help='Number of LODs')
    net_group.add_argument('--base-lod', type=int, default=2,
                          help='Base level LOD')
    net_group.add_argument('--ff-dim', type=int, default=-1,
                          help='Fourier feature dimension.')
    net_group.add_argument('--ff-width', type=float, default='16.0',
                          help='Fourier feature width.')
    net_group.add_argument('--hidden-dim', type=int, default=128,
                          help='Network width')
    net_group.add_argument('--pretrained', type=str,
                          help='Path to pretrained model weights.')
    net_group.add_argument('--periodic', action='store_true',
                          help='Use periodic activations.')
    net_group.add_argument('--skip', type=int, default=None,
                          help='Layer to have skip connection.')
    net_group.add_argument('--freeze', type=int, default=-1,
                          help='Freeze the network at the specified epoch.')
    net_group.add_argument('--pos-invariant', action='store_true',
                          help='Use a position invariant network.')
    net_group.add_argument('--joint-decoder', action='store_true',
                          help='Use a single joint decoder.')
    net_group.add_argument('--feat-sum', action='store_true',
                          help='Sum the features.')

    # Arguments for dataset
    data_group = parser.add_argument_group('dataset')

    # Mesh Dataset
    data_group.add_argument('--dataset-path', type=str,
                            help='Path of dataset')
    data_group.add_argument('--analytic', action='store_true',
                            help='Use analytic dataset')
    data_group.add_argument('--mesh-dataset', type=str, default='MeshDataset',
                            help='Mesh dataset class')
    data_group.add_argument('--raw-obj-path', type=str, default=None,
                            help='Raw mesh root directory to be preprocessed')
    data_group.add_argument('--mesh-batch', action='store_true',
                            help='Batch meshes together')
    data_group.add_argument('--mesh-subset-size', type=int, default=-1,
                            help='Mesh dataset subset (e.g. for ShapeNet, per category); default uses all')
    data_group.add_argument('--train-valid-split', type=str, default=None,
                            help='Path to train/valid dataset split dictionary (JSON)')
    data_group.add_argument('--num-samples', type=int, default=100000,
                            help='Number of samples per mode (or per epoch for SPC)')
    data_group.add_argument('--samples-per-voxel', type=int, default=256,
                            help='Number of samples per voxel (for SPC)')
    data_group.add_argument('--sample-mode', type=str, nargs='*', 
                            default=['rand', 'near', 'near', 'trace', 'trace'],
                            help='The sampling scheme to be used.')
    data_group.add_argument('--trim', action='store_true',
                            help='Trim inner triangles (will destroy UVs!).')
    data_group.add_argument('--sample-tex', action='store_true',
                            help='Sample textures')
    data_group.add_argument('--block-res', type=int, default=7,
                            help='Resolution of blocks')

    # Analytic Dataset
    data_group.add_argument('--include', nargs='*', 
                            help='Shapes to include (all shapes are included by default).')
    data_group.add_argument('--exclude', nargs='*', 
                            help='Shapes to exclude.')
    data_group.add_argument('--glsl-path', type=str, default='../sdf-viewer/data-files/sdf', 
                            help='Path to the GLSL shaders to sample.')
    data_group.add_argument('--viewer-path', type=str, default='../sdf-viewer', 
                            help='Path to the viewer.')
    data_group.add_argument('--get-normals', action='store_true',
                            help='Sample the normals.')
    data_group.add_argument('--build-dataset', action='store_true',
                            help='Builds the dataset.')

    # Arguments for optimizer
    optim_group = parser.add_argument_group('optimizer')
    optim_group.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], 
                             help='Optimizer to be used.')
    optim_group.add_argument('--lr', type=float, default=0.001, 
                             help='Learning rate.')
    optim_group.add_argument('--loss', nargs='+', type=str, 
                             default=['l2_loss'], help='Objective function/loss.')
    optim_group.add_argument('--grad-method', type=str, choices=['autodiff', 'finitediff'], 
                             default='finitediff', help='Mode of gradient computations.')
 
    # Arguments for training
    train_group = parser.add_argument_group('trainer')
    train_group.add_argument('--epochs', type=int, default=250, 
                             help='Number of epochs to run the training.')
    train_group.add_argument('--batch-size', type=int, default=512, 
                             help='Batch size for the training.')
    train_group.add_argument('--only-last', action='store_true', 
                             help='Train only last LOD.')
    train_group.add_argument('--resample-every', type=int, default=10,
                             help='Resample every N epochs')
    train_group.add_argument('--model-path', type=str, default='_results/models', 
                             help='Path to save the trained models.')
    train_group.add_argument('--save-as-new', action='store_true', 
                             help='Save the model at every epoch (no overwrite).')
    train_group.add_argument('--save-every', type=int, default=1, 
                             help='Save the model at every N epoch.')
    train_group.add_argument('--save-all', action='store_true', 
                             help='Save the entire model')
    train_group.add_argument('--latent', action='store_true', 
                             help='Train latent space.')
    train_group.add_argument('--return-lst', action='store_true', 
                             help='Returns a list of predictions (optimization).')
    train_group.add_argument('--latent-dim', type=int, default=128, 
                             help='Latent vector dimension.')
    train_group.add_argument('--logs', type=str, default='_results/logs/runs/',
                             help='Log file directory for checkpoints.')
    train_group.add_argument('--grow-every', type=int, default=-1,
                             help='Grow network every X epochs')
    train_group.add_argument('--loss-sample', type=int, default=-1,
                             help='Sample Nx points for loss importance sampling')
    # One by one trains one level at a time. 
    # Increase starts from [0] and ends up at [0,...,N]
    # Shrink strats from [0,...,N] and ends up at [N]
    # Fine to coarse starts from [N] and ends up at [0,...,N]
    # Only last starts and ends at [N]
    train_group.add_argument('--growth-strategy', type=str, default='increase',
                             choices=['onebyone','increase','shrink', 'finetocoarse', 'onlylast'],
                             help='Strategy for coarse-to-fine training')
            
    # Arguments for renderer
    renderer_group = parser.add_argument_group('renderer')
    renderer_group.add_argument('--sol', action='store_true',
                                help='Use the SOL mode renderer.')
    renderer_group.add_argument('--render-res', type=int, nargs=2, default=[512, 512], 
                                help='Width/height to render at.')
    renderer_group.add_argument('--render-batch', type=int, default=0, 
                                help='Batch size for batched rendering.')
    renderer_group.add_argument('--matcap-path', type=str, 
                                default='data/matcap/green.png', 
                                help='Path to the matcap texture to render with.')
    renderer_group.add_argument('--camera-origin', type=float, nargs=3, default=[-2.8, 2.8, -2.8], 
                                help='Camera origin.')
    renderer_group.add_argument('--camera-lookat', type=float, nargs=3, default=[0, 0, 0], 
                                help='Camera look-at/target point.')
    renderer_group.add_argument('--camera-fov', type=float, default=30, 
                                help='Camera field of view (FOV).')
    renderer_group.add_argument('--camera-proj', type=str, choices=['ortho', 'persp'], default='persp', 
                                help='Camera projection.')
    renderer_group.add_argument('--camera-clamp', nargs=2, type=float, default=[-5, 10], 
                                help='Camera clipping bounds.')
    renderer_group.add_argument('--lod', type=int, default=None, 
                                help='LOD level to use.')
    renderer_group.add_argument('--interpolate', type=float, default=None,
                                help='LOD interpolation value')
    renderer_group.add_argument('--render-every', type=int, default=1,
                                help='Render every N epochs')
    renderer_group.add_argument('--num-steps', type=int, default=256,
                                help='Number of steps')
    renderer_group.add_argument('--step-size', type=float, default=1.0,
                                help='Scale of step size')
    renderer_group.add_argument('--min-dis', type=float, default=0.0003,
                                help='Minimum distance away from surface')
    renderer_group.add_argument('--ground-height', type=float,
                                help='Ground plane y coords')
    renderer_group.add_argument('--tracer', type=str, default='SphereTracer', 
                                help='The tracer to be used.')
    renderer_group.add_argument('--ao', action='store_true',
                                help='Use ambient occlusion.')
    renderer_group.add_argument('--shadow', action='store_true',
                                help='Use shadowing.')
    renderer_group.add_argument('--shading-mode', type=str, default='matcap',
                                help='Shading mode.')

    # Parse and run
    if return_parser:
        return parser
    else:
        return argparse_to_str(parser)


def argparse_to_str(parser):
    """Convert parser to string representation for Tensorboard logging.

    Args:
        parser (argparse.parser): CLI parser
    """

    args = parser.parse_args()

    args_dict = {}
    for group in parser._action_groups:
        group_dict = {a.dest:getattr(args, a.dest, None) for a in group._group_actions}
        args_dict[group.title] = vars(argparse.Namespace(**group_dict))

    pp = pprint.PrettyPrinter(indent=2)
    args_str = pp.pformat(args_dict)
    args_str = f'```{args_str}```'

    return args, args_str

