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

__global__ void sparse_grid_sample_kernel(
    const float* __restrict__ x, // sample locations [n, 3]
    const int* __restrict__ pidx, // sampled voxel idxes [n]
    const int* __restrict__ idxes, // sampled voxel idxes [n]
    const Trinket* __restrict__ trinkets, // going from pidx to corners (cidx) and parents (vidx)
    //const uint* __restrict__ res, // resolution per layer
    const float* __restrict__ feats_in, // features [m, dim]
    const uint* __restrict__ pyramid, // of nums
    const uint* __restrict__ resolutions, // of nums
    float* __restrict__ feats_out, // interpolated features [n, dim]
    const int n, // num sample locations
    const int m, // num features
    const int dim, // num dims
    const int nl, // num layers
    const int lod,
    const int* __restrict__ cc
) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    if (idx > n) return;

    for (int _i=idx; _i<n; _i+=stride) { // sample coords / voxel idx
        int i = idxes[_i];
        feats_out[_i*(dim+3)+0] = x[i*3]  ;
        feats_out[_i*(dim+3)+1] = x[i*3+1];
        feats_out[_i*(dim+3)+2] = x[i*3+2];
        
        //int offset = 0;
        //int offset = pyramid[lod];

        int offset = pyramid[lod+1];
        
        //for (int j=0; j < lod+2; ++j) {
        //    offset += pyramid[j];
        //}
        
        int cidx = pidx[i] + offset;
        float nx = fmaf(x[i*3]  , 0.5f, 0.5f);
        float ny = fmaf(x[i*3+1], 0.5f, 0.5f);
        float nz = fmaf(x[i*3+2], 0.5f, 0.5f);

        for (int j=nl-lod-1; j < nl; ++j) { // lod (top to bottom)
            if (j != (nl-lod-1)) {
                cidx = trinkets[cidx].parent;
            }
            // Calculate remainders
            // res should be precalculated, will also result in -6 ops per thread
            //float res = powf(2, nl-j+1);
            float res = resolutions[nl-j-1];
            float iz = nx * res; 
            float iy = ny * res;
            float ix = nz * res;
            
            // Get the feature coords
            // note: pidx[i] = voxel idx
            // trinkets maps from voxel idx to cf idx
            uint v000 = trinkets[cidx].v[0]; // note: this is in xyz order
            uint v100 = trinkets[cidx].v[1];
            uint v010 = trinkets[cidx].v[2];
            uint v110 = trinkets[cidx].v[3];
            uint v001 = trinkets[cidx].v[4];
            uint v101 = trinkets[cidx].v[5];
            uint v011 = trinkets[cidx].v[6];
            uint v111 = trinkets[cidx].v[7];

            #pragma unroll
            for (int k=0; k<32; ++k) {
                feats_out[_i*(dim+3)+k+3] += feats_in[v000*dim+k] * (cc[v111*3+2]-ix) * (cc[v111*3+1]-iy) * (cc[v111*3]-iz);
                feats_out[_i*(dim+3)+k+3] += feats_in[v001*dim+k] * (cc[v110*3+2]-ix) * (cc[v110*3+1]-iy) * (iz-cc[v110*3]);
                feats_out[_i*(dim+3)+k+3] += feats_in[v010*dim+k] * (cc[v101*3+2]-ix) * (iy-cc[v101*3+1]) * (cc[v101*3]-iz);
                feats_out[_i*(dim+3)+k+3] += feats_in[v011*dim+k] * (cc[v100*3+2]-ix) * (iy-cc[v100*3+1]) * (iz-cc[v100*3]);
                feats_out[_i*(dim+3)+k+3] += feats_in[v100*dim+k] * (ix-cc[v011*3+2]) * (cc[v011*3+1]-iy) * (cc[v011*3]-iz);
                feats_out[_i*(dim+3)+k+3] += feats_in[v101*dim+k] * (ix-cc[v010*3+2]) * (cc[v010*3+1]-iy) * (iz-cc[v010*3]);
                feats_out[_i*(dim+3)+k+3] += feats_in[v110*dim+k] * (ix-cc[v001*3+2]) * (iy-cc[v001*3+1]) * (cc[v001*3]-iz);
                feats_out[_i*(dim+3)+k+3] += feats_in[v111*dim+k] * (ix-cc[v000*3+2]) * (iy-cc[v000*3+1]) * (iz-cc[v000*3]);
            } 
        }
    }
}

}

