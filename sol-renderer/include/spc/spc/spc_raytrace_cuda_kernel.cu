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
    
#include <torch/torch.h>
#include <stdio.h>

#define CUB_STDERR
#include "cub/device/device_scan.cuh"

#include "spc_math.h"

using namespace std;
using namespace torch::indexing;  


__constant__ uint Order[8][8] = {
    { 0, 1, 2, 4, 3, 5, 6, 7 },
    { 1, 0, 3, 5, 2, 4, 7, 6 },
    { 2, 0, 3, 6, 1, 4, 7, 5 },
    { 3, 1, 2, 7, 0, 5, 6, 4 },
    { 4, 0, 5, 6, 1, 2, 7, 3 },
    { 5, 1, 4, 7, 0, 3, 6, 2 },
    { 6, 2, 4, 7, 0, 3, 5, 1 },
    { 7, 3, 5, 6, 1, 2, 4, 0 } };



__global__ void 
d_ScanNodesA(
    const uint numBytes,
    const uchar *d_octree,
    uint *d_Info)
{
    uint tidx = blockIdx.x * 1024 + threadIdx.x;

    if (tidx < numBytes)
        d_Info[tidx] = __popc(d_octree[tidx]);
}


ulong GetStorageBytes(void* d_temp_storage, uint* d_Info, uint* d_PrefixSum, uint max_total_points)
{
    ulong       temp_storage_bytes = 0;
    kaolin::cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_Info, d_PrefixSum, max_total_points); 
    return temp_storage_bytes;
}


__global__ void
d_InitNuggets(uint num, uint2* nuggets)
{
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num)
    {
        nuggets[tidx].x = tidx; //ray idx
        nuggets[tidx].y = 0;
    }
}


__device__ bool 
d_FaceEval(ushort i, ushort j, float a, float b, float c)
{
    float result[4];

    result[0] = a*i + b*j + c;
    result[1] = result[0] + a;
    result[2] = result[0] + b;
    result[3] = result[1] + b;

    float min = 1;
    float max = -1;

    for (int i = 0; i < 4; i++)
    {
        if (result[i] < min) min = result[i];
        if (result[i] > max) max = result[i];
    }

    return (min <= 0.0f && max >= 0.0f);
}


__global__ void
d_Decide(uint num, point_data* points, float3* rorg, float3* rdir, uint2* nuggets, uint* info, uint* D, uint Level, uint offset, uint notDone)
{
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num)
    {
        uint ridx = nuggets[tidx].x;
        uint pidx = nuggets[tidx].y;
        point_data p = points[pidx];

        float3 o = 0.5f*rorg[ridx];
        float3 d = 0.5f*rdir[ridx];
        o.x += 0.5f;
        o.y += 0.5f;
        o.z += 0.5f;

        float3 oxd = crs3(o, d);
        float s1 = 1.0 / ((float)(0x1 << Level));
        float s2 = s1 * s1;
        uchar dd = D[offset + pidx];

        if (d_FaceEval(p.y, p.z, -s2 * d.z,  s2 * d.y, s1 * oxd.x) &&
            d_FaceEval(p.x, p.z,  s2 * d.z, -s2 * d.x, s1 * oxd.y) &&
            d_FaceEval(p.x, p.y, -s2 * d.y,  s2 * d.x, s1 * oxd.z)
            )
            info[tidx] = notDone ? dd : 1;
        else
            info[tidx] = 0;
    }
}


__global__ void
d_Subdivide(uint num, uint2* nuggetsIn, uint2* nuggetsOut, float3* rorg, point_data* points, uchar* O, uint* S, uint* info, uint* prefix_sum, uint Level, uint offset0, uint offset1)
{
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num && info[tidx])
    {
        uint ridx = nuggetsIn[tidx].x;
        int pidx = nuggetsIn[tidx].y;
        point_data p = points[pidx];

        uint IdxBase = prefix_sum[tidx];
        
        uchar o = O[offset0 + pidx];
        uint s = S[offset0 + pidx];

        float scale = 1.0 / ((float)(0x1 << Level));
        float3 org = rorg[ridx];

        float x = (0.5f*org.x+0.5f) - scale*((float)p.x + 0.5);
        float y = (0.5f*org.y+0.5f) - scale*((float)p.y + 0.5);
        float z = (0.5f*org.z+0.5f) - scale*((float)p.z + 0.5);

        uint code = 0;
        if (x > 0) code = 4;
        if (y > 0) code += 2;
        if (z > 0) code += 1;

        for (uint i = 0; i < 8; i++)
        {
            uint j = Order[code][i];
            if (o&(0x1 << j))
            {
                uint cnt = __popc(o&((0x2 << j) - 1)); // count set bits up to child - inclusive sum
                nuggetsOut[IdxBase].y = s + cnt - offset1;
                nuggetsOut[IdxBase++].x = ridx;
            }
        }
    }
}


__global__ void
d_Compactify(uint num, uint2* nuggetsIn, uint2* nuggetsOut, uint* info, uint* prefix_sum)
{
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num && info[tidx])
        nuggetsOut[prefix_sum[tidx]] = nuggetsIn[tidx];
}



uint spc_raytrace_cuda( 
    uchar* d_octree,
    uint Level,
    uint targetLevel,
    point_data* d_points,
    uint* h_pyramid,
    uint*   d_D,
    uint*   d_S, 
    uint num,
    float3* d_Org,
    float3* d_Dir,
    uint2*  d_NuggetBuffers,
    uint*   d_Info,
    uint*   d_PrefixSum, 
    void* d_temp_storage, 
    ulong temp_storage_bytes)
{
#ifdef VERBOSE
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif

    uint* PyramidSum = h_pyramid + Level + 2;

    uint2*  d_Nuggets[2];
    d_Nuggets[0] = d_NuggetBuffers;
    d_Nuggets[1] = d_NuggetBuffers + MAX_TOTAL_POINTS;

    int osize = PyramidSum[Level];

    d_ScanNodesA << < (osize + 1023) / 1024, 1024 >> >(osize, d_octree, d_D);
    kaolin::cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_D, d_S, osize); //NOTE: ExclusiveSum

    d_InitNuggets << <(num + 1023) / 1024, 1024 >> > (num, d_Nuggets[0]);

    uint cnt, buffer = 0;

    // set first element to zero
    cudaMemcpy(d_PrefixSum, &buffer, sizeof(uint), cudaMemcpyHostToDevice);

    for (uint l = 0; l <= targetLevel; l++)
    {
        point_data* proot = d_points + PyramidSum[l];
        d_Decide << <(num + 1023) / 1024, 1024 >> > (num, proot, d_Org, d_Dir, d_Nuggets[buffer], d_Info, d_D, l, PyramidSum[l], targetLevel - l);
        kaolin::cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_Info, d_PrefixSum + 1, num);//start sum on second element
        cudaMemcpy(&cnt, d_PrefixSum + num, sizeof(uint), cudaMemcpyDeviceToHost);

        if (cnt == 0 || cnt > MAX_TOTAL_POINTS) break; // either miss everything, or exceed memory allocation

        if (l < targetLevel)
            d_Subdivide << <(num + 1023) / 1024, 1024 >> > (num, d_Nuggets[buffer], d_Nuggets[(buffer + 1) % 2], d_Org, proot, d_octree, d_S, d_Info, d_PrefixSum, l, PyramidSum[l], PyramidSum[l+1]);
        else
            d_Compactify << <(num + 1023) / 1024, 1024 >> > (num, d_Nuggets[buffer], d_Nuggets[(buffer + 1) % 2], d_Info, d_PrefixSum);

        cudaGetLastError();

        buffer = (buffer + 1) % 2;
        num = cnt;
    }

#ifdef VERBOSE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("\nspc_raytrace_cuda: %d  voxels hit in %f ms\n", num, milliseconds);
#endif

    return cnt;
}


////////// generate rays //////////////////////////////////////////////////////////////////////////


__global__ void
d_generate_rays(uint num, uint imageW, uint imageH, float4x4 mM, float3* rayorg, float3* raydir)
{
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num)
    {
        uint px = tidx % imageW;
        uint py = tidx / imageW;

        float4 a = mul4x4(make_float4(0.0f, 0.0f, 1.0f, 0.0f), mM);
        float4 b = mul4x4(make_float4(px, py, 0.0f, 1.0f), mM);

        rayorg[tidx] = make_float3(a.x, a.y, a.z);
        raydir[tidx] = make_float3(b.x, b.y, b.z);
    }
}


uint generate_primary_rays_cuda(uint imageW, uint imageH, float4x4& mM, float3* d_Org, float3* d_Dir)
{
    uint num = imageW*imageH;

    d_generate_rays << <(num + 1023) / 1024, 1024 >> > (num, imageW, imageH, mM, d_Org, d_Dir);

    return num;
}
