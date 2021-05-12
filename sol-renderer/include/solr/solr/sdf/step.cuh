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

#include "../common/array.cuh"
#include "../util_structs.cuh"

namespace solr {

__global__ void step_kernel(
    const float* __restrict__ ray_o, // ray origin array 
    const float* __restrict__ ray_d, // ray direction array
    const int* __restrict__ idxes,   // active ray indices
    const float* __restrict__ _d,    // temp distance (fuse this..)
    float* __restrict__ x,           // position
    float* __restrict__ t,           // ray param
    float* __restrict__ d,           // distnace
    float* __restrict__ dprev,       // previous distnace
    bool* __restrict__ cond,         // hit array
    bool* __restrict__ hit,          // is there a diff?
    const int n                      // num rays
){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    if (idx > n) return;
    
    for (int _i=idx; _i<n; _i+=stride) {
        
        int i = idxes[_i];
        
        d[i] = _d[_i];
        
        //if (cond[i] == false) {
        //    continue;
        //}
        
        // First, update conditions 

        t[i] += d[i];

        // Explicitly hits surface?
        hit[i]  = (fabs(d[i]) < 0.0003);
        // Or is oscillating?
        hit[i] |= ((fabs(d[i]+dprev[i]) * 0.5) < 0.0015);

        // Deactivate by far clipping
        //cond[i] = (fabs(t[i]) < 5.0);
        cond[i] = (t[i] < 5.0);
        //cond[i] = (fabs(t[i]) < 2.0);
        // Not a hit yet?
        cond[i] &= !hit[i];

        // Outside the AABB
        //float x0 = fabs(x[i*3]);
        //float x1 = fabs(x[i*3+1]);
        //float x2 = fabs(x[i*3+2]);
        //cond[i] &= (x0 > -1) & (x0 < 1) & (x1 > -1) & (x1 < 1) & (x2 > -1) & (x2 < 1);

        // Otherwise, advance
        x[3*i]   = fmaf(ray_d[3*i],   t[i], ray_o[3*i]  );
        x[3*i+1] = fmaf(ray_d[3*i+1], t[i], ray_o[3*i+1]);
        x[3*i+2] = fmaf(ray_d[3*i+2], t[i], ray_o[3*i+2]);
        dprev[i] = d[i];
    }
}

}

