/******************************************************************************
 * The MIT License (MIT)

 * Copyright (c) 2021, NVIDIA CORPORATION.

 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 ******************************************************************************/

#pragma once

#include <torch/torch.h>

#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <string>

#include <solr/util_structs.cuh>

typedef unsigned char  uchar;
typedef unsigned short ushort;
typedef unsigned int   uint;

class SDF {

public:
    SDF(void);
    ~SDF(void);

    void loadWeights(std::string);
    void initTrinkets(uint num_pts, uint num_levels, uint* pyramid, ushort4* points);
    torch::Tensor forward(
        const torch::Tensor  & x,
        const int lod);
    
    torch::Tensor getNormal(
        const torch::Tensor & x,
        const torch::Tensor & pidx,
        const torch::Tensor & hit,
        const int lod);

    std::vector<torch::Tensor> sphereTrace(
        const torch::Tensor & ray_o,
        const torch::Tensor & ray_d,
        const torch::Tensor & nuggets,
        const torch::Tensor & points,
        const torch::Tensor & info,
        const int lod);

private:
    // Features
    std::vector<torch::Tensor> feat;

    // Decoder Params
    std::vector<torch::Tensor> w0;
    std::vector<torch::Tensor> b0;
    std::vector<torch::Tensor> w1;
    std::vector<torch::Tensor> b1;

    // Corner Coordinates
    torch::Tensor cc;

    // Corner Features
    torch::Tensor cf;
    
    // Pyramid of # sparse voxels per level 
    // (note this is in increasing order of levels, unlike SPC pyramid)
    // (it also only contains Levels - 2 levels compared to SPC pyramid)
    uint* pyramid_cf; 
    std::vector<uint> pyramid_cf_cpu;

    // SPC point (voxel) idx -> 8 cf idx
    solr::Trinket* trinkets;
    ushort4* m_points;
    uint* m_pyramid;
    uint* m_res;
};

