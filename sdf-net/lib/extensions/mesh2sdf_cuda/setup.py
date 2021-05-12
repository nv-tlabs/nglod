from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

PACKAGE_NAME = 'mesh2sdf'
VERSION = '0.1.0'
DESCRIPTION = 'Fast CUDA kernel for computing SDF of triangle mesh'
AUTHOR = 'Zekun Hao et al.'
URL = 'https://github.com/zekunhao1995/DualSDF'
LICENSE = 'MIT'

# Standalone package (old way)
setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    url=URL,
    license=LICENSE,
    ext_modules=[
        CUDAExtension(
            name='mesh2sdf',
            sources=['mesh2sdf_kernel.cu'],
            extra_compile_args={'cxx': ['-std=c++14', '-ffast-math'], 'nvcc': ['-std=c++14']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
