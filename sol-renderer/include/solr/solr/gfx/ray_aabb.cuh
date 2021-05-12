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

// Get component sign for the direction ray
__device__ float3 ray_sgn(
    const float3 dir     // ray direction
) {
    return make_float3(
            signbit(dir.x) ? 1.0f : -1.0f,
            signbit(dir.y) ? 1.0f : -1.0f,
            signbit(dir.z) ? 1.0f : -1.0f);
}

// Device primitive for a single ray-AABB intersection
__device__ float ray_aabb(
    const float3 query,  // query point (or ray origin)
    const float3 dir,    // ray direction
    const float3 invdir, // ray inverse direction
    const float3 sgn,    // sgn bits
    const float3 origin, // origin of aabb
    const float  r       // radius of aabb
) {
    // From Majercik et. al 2018

    float3 o = make_float3(query.x-origin.x, query.y-origin.y, query.z-origin.z);
    float cmax = fmaxf(fmaxf(fabs(o.x), fabs(o.y)), fabs(o.z));    

    float winding = cmax < r ? -1.0f : 1.0f;
    winding *= r;
    if (winding < 0) {
        return winding;
    }
    
    float d0 = fmaf(winding, sgn.x, - o.x) * invdir.x;
    float d1 = fmaf(winding, sgn.y, - o.y) * invdir.y;
    float d2 = fmaf(winding, sgn.z, - o.z) * invdir.z;
    float ltxy = fmaf(dir.y, d0, o.y);
    float ltxz = fmaf(dir.z, d0, o.z);
    float ltyx = fmaf(dir.x, d1, o.x);
    float ltyz = fmaf(dir.z, d1, o.z);
    float ltzx = fmaf(dir.x, d2, o.x);
    float ltzy = fmaf(dir.y, d2, o.y);
    bool test0 = (d0 >= 0.0f) && (fabs(ltxy) < r) && (fabs(ltxz) < r);
    bool test1 = (d1 >= 0.0f) && (fabs(ltyx) < r) && (fabs(ltyz) < r);
    bool test2 = (d2 >= 0.0f) && (fabs(ltzx) < r) && (fabs(ltzy) < r);

    float3 _sgn = make_float3(0.0f, 0.0f, 0.0f);

    if (test0) { _sgn.x = sgn.x; }
    else if (test1) { _sgn.y = sgn.y; }
    else if (test2) { _sgn.z = sgn.z; }

    float d = 0.0f;
    if (_sgn.x != 0.0f) { d = d0; } 
    else if (_sgn.y != 0.0f) { d = d1; }
    else if (_sgn.z != 0.0f) { d = d2; }
    if (d != 0.0f) {
        return d;
    }

    return 0.0;
}

// Calls sgn if sgn is not precomputed
__device__ float ray_aabb(
    const float3 query,  // query point (or ray origin)
    const float3 dir,    // ray direction
    const float3 invdir, // ray inverse direction
    const float3 origin, // origin of aabb
    const float  r       // radius of aabb
) {
    float3 sgn = ray_sgn(dir);
    return ray_aabb(query, dir, invdir, sgn, origin, r);
}

// This kernel will iterate over Nuggets, instead of iterating over rays
__global__ void ray_aabb_kernel(
    const float* __restrict__ ray_o,     // ray origin array
    const float* __restrict__ ray_d,     // ray direction array
    const float* __restrict__ ray_inv,   // inverse ray direction array
    const float* __restrict__ query,     // ray query array
    const Nugget* __restrict__ nuggets,  // nugget array (ray-aabb correspondences)
    const short* __restrict__ points,    // 3d coord array
    const int* __restrict__ info,        // binary array denoting end of nugget group
    const int* __restrict__  info_idxes, // array of active nugget indices
    const float r,                       // radius of aabb
    const bool init,                     // first run?
    float* __restrict__ x,               // xyz position of intersection
    float* __restrict__ t,               // parameter of ray intersection
    bool* __restrict__ cond,             // true if hit
    int* __restrict__ pidx,              // index of 3d coord array
    const int num_rays,                  // # of nugget indices
    const int n                          // # of active nugget indices
){
    
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    if (idx > n) return;

    for (int _i=idx; _i<n; _i+=stride) {
        // Get index of corresponding nugget
        int i = info_idxes[_i];
        
        // Get index of ray
        uint ridx = nuggets[i].x;

        // If this ray is already terminated, continue
        if (!cond[ridx] && !init) continue;
        
        bool _hit = false;

        // Get the direction and inverse direction
        const float3 dir = solr::xyz(ray_d, ridx);
        const float3 invdir = solr::xyz(ray_inv, ridx);
        const float3 query_pt = solr::xyz(query, ridx);
        
        // Sign bit
        const float3 sgn = solr::ray_sgn(dir);
            
        bool _cond = false;
        
        int j = 0;
        // In order traversal of the voxels
        do {
            // Get the vc from the nugget
            uint _pidx = nuggets[i].y; // Index of compressed 3D coords array
            short3 p = solr::xyz(points, _pidx);

            // Center of voxel
            const float3 vc = make_float3(
                fmaf(r, fmaf(2.0, float(p.x), 1.0), -1.0f),
                fmaf(r, fmaf(2.0, float(p.y), 1.0), -1.0f),
                fmaf(r, fmaf(2.0, float(p.z), 1.0), -1.0f));

            float d = solr::ray_aabb(query_pt, dir, invdir, sgn, vc, r);

            if (d != 0.0) {
                _hit = true;
                pidx[ridx] = _pidx;
                cond[ridx] = _hit;
                if (d > 0.0) {
                    t[ridx] += d;
                    x[3*ridx]   = fmaf(ray_d[3*ridx],   t[ridx], ray_o[3*ridx]  );
                    x[3*ridx+1] = fmaf(ray_d[3*ridx+1], t[ridx], ray_o[3*ridx+1]);
                    x[3*ridx+2] = fmaf(ray_d[3*ridx+2], t[ridx], ray_o[3*ridx+2]);
                }
            } 
           
            ++i;
            ++j;
            
        } while (i < num_rays && info[i] != 1 && _hit == false);
        

        if (!_hit) {
            // Should only reach here if it misses
            cond[ridx] = false;
            t[ridx] = 100;
            x[3*ridx]   = fmaf(ray_d[3*ridx],   t[ridx], ray_o[3*ridx]  );
            x[3*ridx+1] = fmaf(ray_d[3*ridx+1], t[ridx], ray_o[3*ridx+1]);
            x[3*ridx+2] = fmaf(ray_d[3*ridx+2], t[ridx], ray_o[3*ridx+2]);
        }
        
    }
}

__global__ void ray_aabb_kernel(
    const float* __restrict__ ray_o,
    const float* __restrict__ ray_d,
    const float* __restrict__ ray_inv,
    const float* __restrict__ vc,
    const int nvc,
    const float r,
    bool* __restrict__ hit,
    const int n
){
    // From Majercik et. al 2018
    // This kernel, for now, assumes the AABB is 0 centered with radius 1
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    if (idx > n) return;
    
    for (int i=idx; i<n; i+=stride) {
        const float3 dir = solr::xyz(ray_d, i);
        const float3 invdir = solr::xyz(ray_inv, i);
        const float3 query_pt = solr::xyz(ray_o, i);
        
        const float3 sgn = solr::ray_sgn(dir);

        bool _hit = false;
        int j = 0; 
        do { 
            const float3 _vc = make_float3(vc[3*j], vc[3*j+1], vc[3*j+2]);

            float d = solr::ray_aabb(query_pt, dir, invdir, sgn, _vc, r);
            
            if (d > 0.0) {
                _hit = true;
                hit[i] = true;
            }
            j += 1;
        } while (j < nvc && _hit == false);
    }
}

}


