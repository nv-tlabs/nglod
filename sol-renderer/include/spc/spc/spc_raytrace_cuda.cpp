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

#include <stdlib.h>
#include <stdio.h>
#include <torch/torch.h>
#include <cuda_runtime.h>

#include "spc_math.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(x.device().is_cpu(), #x " must be a cpu tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_TRIPLE(x) TORCH_CHECK(x.dim() == 2 && x.size(1) == 3, "input is not Nx3")
#define CHECK_QUAD(x) TORCH_CHECK(x.dim() == 2 && x.size(1) == 4, "input is not Nx4")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, "input is not Float")
#define CHECK_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, "input is not Int")
#define CHECK_SHORT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Short, "input is not Short")
#define CHECK_BYTE(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Byte, "input is not Byte")

#define CHECK_PACKED_FLOAT3(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x) CHECK_TRIPLE(x)
#define CHECK_PACKED_SHORT3(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_SHORT(x) CHECK_TRIPLE(x)
#define CHECK_PACKED_SHORT4(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_SHORT(x) CHECK_QUAD(x)
#define CHECK_CPU_INT(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x); CHECK_INT(x)

using namespace std;
using namespace torch::indexing;


extern ulong GetStorageBytes(
  void* d_temp_storage, 
  uint* d_Info, 
  uint* d_PrefixSum, 
  uint max_total_points);

extern void generate_primary_rays_cuda(
  uint imageW, 
  uint imageH, 
  float4x4& nM, 
  float3* d_org, 
  float3* d_dir);
  

std::vector<at::Tensor>  spc_generate_primary_rays(
    torch::Tensor Eye, torch::Tensor At, torch::Tensor Up,
    uint imageW, uint imageH, float fov, torch::Tensor World) 
{
    uint num = imageW * imageH;
    torch::Tensor Org = torch::zeros({num, 3}, torch::device(torch::kCUDA).dtype(torch::kFloat));
    torch::Tensor Dir = torch::zeros({num, 3}, torch::device(torch::kCUDA).dtype(torch::kFloat));

    float3* d_org = reinterpret_cast<float3*>(Org.data_ptr<float>());
    float3* d_dir = reinterpret_cast<float3*>(Dir.data_ptr<float>());

    float3 eye = *reinterpret_cast<float3*>(Eye.data_ptr<float>());
    float3 at = *reinterpret_cast<float3*>(At.data_ptr<float>());
    float3 up = *reinterpret_cast<float3*>(Up.data_ptr<float>());

    float4x4 world = *reinterpret_cast<float4x4*>(World.data_ptr<float>());
    float4x4 mWorldInv = transpose(world);

    float ar = (float)imageW / (float)imageH;
    float const rad = 0.01745329f * fov;
    float tanHalfFov = tanf(0.5f * rad);    

    float4x4 mPvpInv = make_float4x4(
        2.0f*ar*tanHalfFov/ imageW, 0.0f, 0.0f, 0.0f,
        0.0f, 2.0f*tanHalfFov / imageH, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
        ar*tanHalfFov*(1.0f-imageW)/imageW, tanHalfFov*(1.0f-imageH)/imageH, -1.0f, 0.0f);

    float3 z = normalize(at - eye);
    float3 x = normalize(crs3(z, up));
    float3 y = crs3(x, z);

    float4x4 mViewInv = make_float4x4(
        x.x, x.y, x.z, 0.0f,
        y.x, y.y, y.z, 0.0f,
        -z.x, -z.y, -z.z, 0.0f,
        eye.x, eye.y, eye.z, 1.0f);

    float4x4 mWVPInv = mPvpInv*mViewInv*mWorldInv;

    generate_primary_rays_cuda(imageW, imageH, mWVPInv, d_org, d_dir);

    // assemble output tensors
    std::vector<at::Tensor> result;
    result.push_back(Org);
    result.push_back(Dir);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();

    if ( err != cudaSuccess )
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    return result;
}


extern uint spc_raytrace_cuda( 
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
    uint2* d_Nuggets,
    uint*   d_Info,
    uint*   d_PrefixSum, 
    void* d_temp_storage, 
    ulong temp_storage_bytes);

torch::Tensor spc_raytrace(
    torch::Tensor octree,
    torch::Tensor points,
    torch::Tensor pyramid, 
    torch::Tensor Org,
    torch::Tensor Dir,
    uint targetLevel) 
{
    CHECK_BYTE(octree);
    CHECK_PACKED_SHORT4(points);
    CHECK_CPU_INT(pyramid);
    TORCH_CHECK(pyramid.dim() == 2, "bad spc table0");
    TORCH_CHECK(pyramid.size(0) == 2, "bad spc table1");
    uint Level = pyramid.size(1)-2;
    TORCH_CHECK(Level < MAX_LEVELS, "bad spc table2");
    uint* h_pyramid = (uint*)pyramid.data_ptr<int>();
    uint osize = h_pyramid[2*Level+2];
    uint psize = h_pyramid[2*Level+3];
    TORCH_CHECK(octree.size(0) == osize, "bad spc octree size");
    TORCH_CHECK(points.size(0) == psize, "bad spc points size");
    TORCH_CHECK(h_pyramid[Level+1] == 0 && h_pyramid[Level+2] == 0, "bad spc table3");

    //check Org and Dir better... for now
    uint num = Org.size(0);

    // allocate local GPU storage
    torch::Tensor Nuggets = torch::zeros({2 * MAX_TOTAL_POINTS, 2}, torch::device(torch::kCUDA).dtype(torch::kInt32));
    torch::Tensor Info = torch::zeros({MAX_TOTAL_POINTS}, torch::device(torch::kCUDA).dtype(torch::kInt32));
    torch::Tensor PrefixSum = torch::zeros({MAX_TOTAL_POINTS}, torch::device(torch::kCUDA).dtype(torch::kInt32));
    torch::Tensor D = torch::zeros({osize}, torch::device(torch::kCUDA).dtype(torch::kInt32));
    torch::Tensor S = torch::zeros({osize}, torch::device(torch::kCUDA).dtype(torch::kInt32));

    // get tensor data pointers
    float3* d_org = reinterpret_cast<float3*>(Org.data_ptr<float>());
    float3* d_dir = reinterpret_cast<float3*>(Dir.data_ptr<float>());

    uint2*    d_Nuggets = reinterpret_cast<uint2*>(Nuggets.data_ptr<int>());
    uint* d_Info = reinterpret_cast<uint*>(Info.data_ptr<int>());
    uint* d_PrefixSum = reinterpret_cast<uint*>(PrefixSum.data_ptr<int>());
    uint* d_D = reinterpret_cast<uint*>(D.data_ptr<int>());
    uint* d_S = reinterpret_cast<uint*>(S.data_ptr<int>());
    uchar*    d_octree = octree.data_ptr<uchar>();
    point_data* d_points = reinterpret_cast<point_data*>(points.data_ptr<short>());

    // set up memory for DeviceScan calls
    void* d_temp_storage = NULL;
    ulong temp_storage_bytes = GetStorageBytes(d_temp_storage, d_Info, d_PrefixSum, MAX_TOTAL_POINTS);
    torch::Tensor temp_storage = torch::zeros({(long)temp_storage_bytes}, torch::device(torch::kCUDA).dtype(torch::kByte));
    d_temp_storage = (void*)temp_storage.data_ptr<uchar>();

    // do cuda
    num = spc_raytrace_cuda(d_octree, Level, targetLevel, d_points, h_pyramid, 
                        d_D, d_S, num, d_org, d_dir, d_Nuggets, 
                        d_Info, d_PrefixSum, d_temp_storage, temp_storage_bytes);

    uint pad = ((targetLevel+1) % 2) * MAX_TOTAL_POINTS;

    return Nuggets.index({Slice(pad, pad+num)});
}
