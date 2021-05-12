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

namespace solr {

// Allocates 6 tap positions for finite differences
__global__ void finitediff_allocate_kernel(
    const float* __restrict__ x,
    const int* __restrict__ pidx,
    const int* __restrict__ idxes,
    float* __restrict__ _x,
    int* __restrict__ _pidx,
    const int n,
    const float eps) {

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    if (idx > n) return;
    
    for (int _i=idx; _i<n; _i+=stride) {
        int i = idxes[_i];
        _x[_i*3  ]         = x[i*3  ] + eps;
        _x[_i*3+1]         = x[i*3+1];
        _x[_i*3+2]         = x[i*3+2];
        _x[_i*3   + n]     = x[i*3  ] - eps;
        _x[_i*3+1 + n]     = x[i*3+1];
        _x[_i*3+2 + n]     = x[i*3+2];
        _x[_i*3   + (n*2)] = x[i*3  ];
        _x[_i*3+1 + (n*2)] = x[i*3+1] + eps;
        _x[_i*3+2 + (n*2)] = x[i*3+2];
        _x[_i*3   + (n*3)] = x[i*3  ];
        _x[_i*3+1 + (n*3)] = x[i*3+1] - eps;
        _x[_i*3+2 + (n*3)] = x[i*3+2];
        _x[_i*3   + (n*4)] = x[i*3  ];
        _x[_i*3+1 + (n*4)] = x[i*3+1];
        _x[_i*3+2 + (n*4)] = x[i*3+2] + eps;
        _x[_i*3   + (n*5)] = x[i*3  ];
        _x[_i*3+1 + (n*5)] = x[i*3+1];
        _x[_i*3+2 + (n*5)] = x[i*3+2] - eps;
        
        #pragma unroll
        for (int j=0; j<6; ++j) {
            _pidx[_i + (n*j)] = pidx[i];
        }

    }

}

// 6 tap finite differences
__global__ void finitediff_kernel(
    const float* __restrict__ d, // precomputed Nx6 array of scalars
    const int* __restrict__ idxes,
    float* __restrict__ normal,
    const int n) {

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    if (idx > n) return;
    
    for (int _i=idx; _i<n; _i+=stride) {
        int i = idxes[_i];
        float diffx = d[_i] - d[_i+n];
        float diffy = d[_i+(n*2)] - d[_i+(n*3)];
        float diffz = d[_i+(n*4)] - d[_i+(n*5)];
        float nrm   = norm3df(diffx, diffy, diffz);
        normal[i*3  ] = diffx / nrm;
        normal[i*3+1] = diffy / nrm;
        normal[i*3+2] = diffz / nrm;
    }
}

}

