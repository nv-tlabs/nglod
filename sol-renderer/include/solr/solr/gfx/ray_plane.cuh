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

// Ray-plane intersection
__global__ void ray_plane_kernel(
    const float* __restrict__ ray_o, 
    const float* __restrict__ ray_d, 
    bool* __restrict__ model_hit, 
    float* __restrict__ x,
    float* __restrict__ t,
    float* __restrict__ normal,
    //bool* __restrict__ plane_hit,
    const float height,
    const int n
) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    if (idx > n) return;

    for (int i=idx; i<n; i+=stride) { 
        
        if (model_hit[i]) continue;
        
        float rate = -ray_d[3*i+1];
        
        if (fabs(rate) < 0.00001) continue;

        float delta = ray_o[3*i+1] - height;
        float _t = delta / rate;
        if (_t > 0 && _t < 500) {
            normal[3*i  ] = 0.0;
            normal[3*i+1] = 1.0;
            normal[3*i+2] = 0.0;

            t[i] = _t;
            x[3*i  ] = ray_o[3*i  ] + ray_d[3*i  ] * _t;
            x[3*i+1] = ray_o[3*i+1] + ray_d[3*i+1] * _t;
            x[3*i+2] = ray_o[3*i+2] + ray_d[3*i+2] * _t;

            model_hit[i] = true;
        }

    }
}



}

