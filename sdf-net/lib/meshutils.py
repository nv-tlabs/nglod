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

import os
import time
import logging as log
import numpy as np
import plyfile
import skimage.measure

import tinyobjloader
import torch
import mesh2sdf as m2s

# Utilities for Mesh Processing

def load_obj(fname: str):
    """Load .obj file using TinyOBJ and extract info.
    This is more robust since it can triangulate polygon meshes 
    with up to 255 sides per face.
    
    Args:
        fname (str): path to Wavefront .obj file
    """

    assert os.path.exists(fname), \
        'Invalid file path and/or format, must be an existing Wavefront .obj'

    reader = tinyobjloader.ObjReader()
    config = tinyobjloader.ObjReaderConfig()
    config.triangulate = True # Ensure we don't have any polygons

    ret = reader.ParseFromFile(fname, config)

    # Get vertices
    attrib = reader.GetAttrib()
    vertices = torch.FloatTensor(attrib.vertices).reshape(-1, 3)

    # Get triangle face indices
    shapes = reader.GetShapes()
    faces = []
    for shape in shapes:
        faces += [idx.vertex_index for idx in shape.mesh.indices] 
    faces = torch.LongTensor(faces).reshape(-1, 3)
    
    return vertices, faces


def convert_to_nvc(vertices, faces):
    """Convert into format expected by CUDA kernel.
    Nb of triangles x 3 (vertices) x 3 (xyz-coordinate/vertex)
    
    WARNING: Will destroy any resemblance of UVs

    Args:
        vertices (torch.Tensor): tensor of all vertices
        faces (torch.Tensor): tensor of all triangle faces
    """

    mesh = vertices[faces.flatten()].reshape(faces.size()[0], 3, 3)
    return mesh.contiguous()


def convert_to_obj(mesh):
    """Convert into vertices and faces from NVC format.
    
    Args:
        mesh (torch.Tensor): NVC-formatted tensor
    """
    
    mesh = mesh.cpu()
    unique_v, idx = np.unique(mesh.view(-1,3), axis=0, return_inverse=True)
    vertices = torch.from_numpy(unique_v)
    faces = torch.from_numpy(idx).view(-1,3)
    
    return vertices, faces


def save_obj(fname, vertices, faces):
    """Save to Wavefront .OBJ.
    
    Args:
        fname (str): filename
        vertices (torch.Tensor): [N_vertices, 3]
        vertices (torch.Tensor): [N_faces, 3]
    """

    assert fname.endswith('.obj'), 'Filename must end with .obj'
    with open(fname, 'w') as f:
        for vert in vertices:
            f.write('v %f %f %f\n' % tuple(vert))
        for face in faces:
            f.write('f %d %d %d\n' % tuple(face + 1))


def preprocess_mesh(mesh, nvc=True):
    """Preprocess and normalize within bounding sphere.
    Adapted from DualSDF implementation <https://github.com/zekunhao1995/DualSDF>

    Args:
        mesh (torch.Tensor): mesh in appropriate tensor format (NxVxC)

    """
    # Flips Y by default
    # TODO: Should we? 
    mesh[..., 1] *= -1

    # Normalize mesh
    mesh = mesh.reshape(-1, 3)
    mesh_max, _ = torch.max(mesh, dim=0)
    mesh_min, _ = torch.min(mesh, dim=0)
    mesh_center = (mesh_max + mesh_min) / 2.
    mesh = mesh - mesh_center

    # Find the max distance to origin
    max_dist = torch.sqrt(torch.max(torch.sum(mesh**2, dim=-1)))
    mesh_scale = 1. / max_dist
    mesh *= mesh_scale
    if nvc:
        mesh = mesh.reshape(-1, 3, 3)

    return mesh


def compute_trimmesh(mesh: torch.Tensor, residual: bool = False):
    """Convert mesh to TrimMesh (with trimmed interior triangles).
    Adapted from DualSDF implementation <https://github.com/zekunhao1995/DualSDF>

    Args:
        mesh (torch.Tensor): trimmed triangle mesh
        residual (bool): return the trims
    """

    if not torch.cuda.is_available():
        raise IOError('Cannot run mesh trimmer without CUDA.')

    mesh = mesh.to('cuda:0')
    valid_triangles = m2s.trimmesh_gpu(mesh)
    if residual:
        valid_triangles = ~valid_triangles
    mesh = mesh[valid_triangles, ...].contiguous()

    return mesh


def trim_obj(obj_fname):
    """Preprocess mesh
    
    Args:
        obj_fname (str): path to Wavefront .obj input file
    """

    vertices, faces = load_obj(obj_fname)
    mesh = convert_to_nvc(vertices, faces)
    mesh = preprocess_mesh(mesh)
    mesh = compute_trimmesh(mesh)
    return mesh


def trim_obj_to_file(obj_fname: str, out_fname: str):
    """Convert OBJ mesh to trimmed Torch tensor and save it to file.
    
    Args:
        obj_fname (str): path to Wavefront .obj input file
        pt_fname (str): path to Torch .pt output file (default: use obj_fname)
    """

    mesh = trim_obj(obj_fname)

    nobj_fname = f'{out_fname.split(".")[0]}.obj'
    V, F = convert_to_obj(mesh)
    save_obj(nobj_fname, V, F)
    return mesh

