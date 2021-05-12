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

__global__ void index_trinket_kernel(
    const ushort4* __restrict__ points,   // lower left coords
    const int* __restrict__ coords,       // [N,3] tensor of coords
    const float* __restrict__ feats,      // [N,32] tensor of features
    solr::Trinket* __restrict__ trinkets, // idx correspondence per point
    const int num_coords,                 // N 
    const int offset_cf,     // offset (starting pointer) of the coords tensor
    const int n,             // PSize of current layer
    const int offset,        // offset (starting pointer) of parent layer
    const int parent_n,      // PSize of parent layer
    const int parent_offset, // offset (starting pointer) of parent layer
    const int level          // current level
){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    if (idx > n) return;
    
    for (int i=idx; i<n; i+=stride) {
        // Move pointer to start of current layer
        int iidx = i + offset;
        
        // Initialize parent to -1
        trinkets[iidx].parent = -1;

        // If not the first level, search through the array of the parent
        // voxels to find the parent voxel
        if (parent_offset != offset) {
            for (int j=0; j<parent_n; ++j) {
                // Move pointer to start of parent layer
                int jidx = j + parent_offset;
                // Get lower left coord of current layer
                ushort p0 = floorf(float(points[iidx].x) / 2.0f);
                ushort p1 = floorf(float(points[iidx].y) / 2.0f);
                ushort p2 = floorf(float(points[iidx].z) / 2.0f);

                // If they match, that is the parent
                if (p0 == points[jidx].x &&
                    p1 == points[jidx].y && 
                    p2 == points[jidx].z) {
                    trinkets[iidx].parent = jidx;
                }
            }
        }

        // Set the trinkets up
        for (int j=0; j<8; ++j) {

            //trinkets[i].v[j] = -1; // Initialize 

            // Getting the 8 corners with bitwise magic...
            int c0 = points[iidx].x + ((j & 4) >> 2);
            int c1 = points[iidx].y + ((j & 2) >> 1);
            int c2 = points[iidx].z +  (j & 1);

            // Search the [N,3] tensor of coords
            for (int k=0; k<num_coords; ++k) {
                int kidx = k + offset_cf;
                int v0 = coords[kidx*3]  ;
                int v1 = coords[kidx*3+1];
                int v2 = coords[kidx*3+2];
                if (v0 == c0 && v1 == c1 && v2 == c2) {
                    trinkets[iidx].v[j] = kidx; // casting?
                    break;
                }    
            }
        }
    }
}

}

