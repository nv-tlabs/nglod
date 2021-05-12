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

#define CUB_NS_PREFIX namespace kaolin {
#define CUB_NS_POSTFIX }
    
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <vector>
#include <iostream>

#include <helper_cuda.h>
#include <helper_math.h>
#include <vector_types.h>

#include <GL/glew.h>

#include <torch/torch.h>

#include <spc/SPC.h>
#include "SDF.h"
#include "nvmath.h"

#include  <solr/util_timer.cuh>

#define CUB_STDERR
#include <cub/util_allocator.cuh>
#include <cub/device/device_scan.cuh>

#define CUDA_PRINT_ERROR() cudaPrintError(__FILE__, __LINE__)

// #define DEBUG

#define STRINGIFY2(X) #X
#define STRINGIFY(X) STRINGIFY2(X)

#ifdef DEBUG
#   define TIMER PerfTimer timer = PerfTimer()
#   define TIMER_CHECK(x) timer.check(x) 
#   define DEBUG_PRINT(x) std::cout << STRINGIFY(x) ":" << x << std::endl
#else
#   define TIMER
#   define TIMER_CHECK(x)
#   define DEBUG_PRINT(x)
#endif 

// These are for hot testing
#   define PROBE PerfTimer probe_timer = PerfTimer()
#   define PROBE_CHECK(x) probe_timer.check(x) 

using namespace solr;

namespace I = torch::indexing;

GLuint                          pbo; // OpenGL pixel buffer object
struct cudaGraphicsResource*    cuda_pbo_resource = NULL; // CUDA Graphics Resource (to transfer PBO)
uint*                           d_output = NULL;
float                           g_milliseconds;
uint                            g_TargetLevel = 0;
uint                            g_Renderer = 0;

extern inline void              cudaPrintError(const char* file, const int line);


torch::Tensor d_ray_o;
torch::Tensor d_ray_d;
torch::Tensor d_x;
torch::Tensor d_t;
torch::Tensor d_normal;

Nugget*     d_Nuggets[2];
uint*       d_Info;
uint*       d_InfoA;
uint*       d_PrefixSum;
void*       d_temp_storage = NULL;
size_t      temp_storage_bytes = 0;
torch::Tensor tInfo;

__device__ __inline__ uchar4 to_uchar4(float4 vec)
{
    return make_uchar4((uchar)vec.x, (uchar)vec.y, (uchar)vec.z, (uchar)vec.w);
}


__global__ void
d_MarkUniqueRays(uint num, Nugget* nuggets, uint* info)
{
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num)
    {
        if (tidx == 0)
            info[tidx] = 1;
        else
            info[tidx] = nuggets[tidx - 1].x == nuggets[tidx].x ? 0 : 1;
    }
}


__global__ void
d_renderHit(uint num, float* ray_o, float* ray_d, bool* hit, uchar4 *d_output)
{
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num)
    {
        float4 color = make_float4(
            float(hit[tidx]),
            float(hit[tidx]),
            float(hit[tidx]),
            1.0);
        d_output[tidx] = to_uchar4(255.0 * color);
    }
}


__global__ void
d_renderDepth(uint num, float* ray_o, float* ray_d, float* depth, uchar4 *d_output)
{
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num)
    {
        float4 color = make_float4(
            clamp(depth[tidx] * 0.25f, 0.0f, 1.0f),
            clamp(depth[tidx] * 0.25f, 0.0f, 1.0f),
            clamp(depth[tidx] * 0.25f, 0.0f, 1.0f),
            1.0);
        d_output[tidx] = to_uchar4(255.0 * color);
    }
}


__global__ void
d_renderNormal(uint num, float* ray_o, float* ray_d, float* normal, uchar4 *d_output)
{
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num)
    {
        float4 color = make_float4(
            (normal[tidx*3  ]+1) * 0.5,
            (normal[tidx*3+1]+1) * 0.5,
            (normal[tidx*3+2]+1) * 0.5,
            1.0);
        if (color.x + color.y + color. z < 2.9) {
            d_output[tidx] = to_uchar4(255.0 * color);
        }
    }
}


extern "C" 
uint RenderImage(
    uchar4 *d_output, 
    uint imageW, 
    uint imageH, 
    torch::Tensor Org, 
    torch::Tensor Dir, 
    torch::Tensor Nuggets, 
    SPC* spc, 
    SDF* sdf) 
{
    CUDA_PRINT_ERROR();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    uint num_rays = imageW*imageH;
    
    CUDA_PRINT_ERROR();

    // map PBO to get CUDA device pointer
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes, cuda_pbo_resource);
    cudaMemset(d_output, g_Renderer!=0?0:-1, num_rays * sizeof(color_type)); //clear image buffer
    CUDA_PRINT_ERROR();

    TIMER;

    torch::Tensor Info = torch::zeros({SCAN_MAX_VOXELS}, torch::device(torch::kCUDA).dtype(torch::kInt32));
    int num_nuggets = Nuggets.size(0);
    d_MarkUniqueRays << <(num_nuggets + 1023) / 1024, 1024 >> > (
            num_nuggets, 
            reinterpret_cast<Nugget*>(Nuggets.data_ptr<int>()),
            reinterpret_cast<uint*>(Info.data_ptr<int>()));
    
    TIMER_CHECK("Postprocess SPC  ");

    CUDA_PRINT_ERROR();
    TIMER_CHECK("generate rays    ");

    int lod = std::max(0, (int) g_TargetLevel - 2);
    
    torch::Tensor Points = spc->GetPoints(g_TargetLevel).index({I::Slice(), torch::tensor({0,1,2})});

    auto out = sdf->sphereTrace(Org, Dir, Nuggets, Points, Info, lod);

    CUDA_PRINT_ERROR();

    TIMER_CHECK("st               ");
    
    switch (g_Renderer)
    {
    case 0:
        d_renderNormal << <(num_rays + 1023) / 1024, 1024 >> >(
            num_rays, Org.data_ptr<float>(), Dir.data_ptr<float>(), out[3].data_ptr<float>(), d_output);
        break;
    case 1:
        d_renderDepth << <(num_rays + 1023) / 1024, 1024 >> >(
            num_rays, Org.data_ptr<float>(), Dir.data_ptr<float>(), out[1].data_ptr<float>(), d_output);
        break;
    case 2:
        d_renderHit << <(num_rays + 1023) / 1024, 1024 >> >(
            num_rays, Org.data_ptr<float>(), Dir.data_ptr<float>(), out[2].data_ptr<bool>(), d_output);
        break;

    }

    TIMER_CHECK("write buffer     ");
    CUDA_PRINT_ERROR();

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    g_milliseconds = 0;
    cudaEventElapsedTime(&g_milliseconds, start, stop);
    CUDA_PRINT_ERROR();

    return num_rays;
}


