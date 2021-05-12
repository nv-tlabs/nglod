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
    
// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR
#include "SPC.h"

#include <cub/device/device_scan.cuh>

using namespace torch::indexing;  

extern inline void              cudaPrintError(const char* file, const int line);

#define CUDA_PRINT_ERROR() cudaPrintError(__FILE__, __LINE__)


SPC::SPC()
{}

SPC::~SPC()
{}

void SPC::SaveNPZ(std::string filename)
{
    torch::Tensor OctreeCPU = m_Octree.to(torch::kCPU);
    uint m_Osize = GetOSize();
    cnpy::npz_save(filename, "octree", reinterpret_cast<uchar*>(OctreeCPU.data_ptr<uchar>()), { m_Osize }, "w");
}
    

void SPC::LoadNPZ(std::string filename)
{
    cnpy::npz_t F = cnpy::npz_load(filename);

    cnpy::NpyArray O = F["octree"];
    uchar* octree = O.data<uchar>();
    
    m_Osize = O.num_vals;

    m_Octree = torch::zeros({m_Osize}, torch::device(torch::kCUDA).dtype(torch::kByte));
    uchar* octreeT = reinterpret_cast<uchar*>(m_Octree.data_ptr<uchar>());
    cudaMemcpy(octreeT, octree, m_Osize, cudaMemcpyHostToDevice);

    std::vector<at::Tensor> tmp;   
    tmp = SetGeometry(m_Octree);

    m_Points = tmp[0];
    m_Pyramid = tmp[1];

    uint* h_pyramid = reinterpret_cast<uint*>(m_Pyramid.data_ptr<int>());
   
    m_Level = m_Pyramid.size(1) - 2;
    m_Psize = h_pyramid[m_Level];
}
    
        
__global__ void MortonToPoint(
    const uint Psize,
    morton_code *DataIn,
    point_data *DataOut)
{
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < Psize)
        DataOut[tidx] = ToPoint(DataIn[tidx]);
}


__global__ void NodesToMorton(
    const uint Psize,
    const uchar *d_Odata,
    const uint * d_PrefixSum,
    const morton_code *d_MdataIn,
    morton_code *d_MdataOut)
{
    uint tidx = blockIdx.x * 1024 + threadIdx.x;

    if (tidx < Psize)
    {
        uchar bits = d_Odata[tidx];
        morton_code code = d_MdataIn[tidx];
        int addr = d_PrefixSum[tidx];

        for (int i = 7; i >= 0; i--)
            if (bits&(0x1 << i))
                d_MdataOut[addr--] = 8 * code + i;
    }
}


point_data* SPC::GetProotGPU(uint l)
{
    point_data* Pdata = reinterpret_cast<point_data*>(m_Points.data_ptr<short>());
    uint offset = 0;

    auto pyramid_a = m_Pyramid.accessor<int, 2>();
    offset = pyramid_a[1][l];

    return Pdata + offset;
}


torch::Tensor SPC::GetPoints(uint l)
{
    auto pyramid_a = m_Pyramid.accessor<int, 2>();
    uint offset = pyramid_a[1][l];

    return m_Points.index({Slice(offset, None, None)});
}


__global__ void 
d_ScanNodes(
    const uint numBytes,
    const uchar *d_octree,
    uint *d_Info)
{
    uint tidx = blockIdx.x * 1024 + threadIdx.x;

    if (tidx < numBytes)
        d_Info[tidx] = __popc(d_octree[tidx]);
}


std::vector<at::Tensor> SPC::SetGeometry(torch::Tensor Octree)
{
    // CHECK_INPUT(odata);
    uchar* Odata = Octree.data_ptr<uchar>();
    m_Osize = Octree.size(0);

    m_Info = torch::zeros({m_Osize+1}, torch::device(torch::kCUDA).dtype(torch::kInt32));
    m_PrefixSum = torch::zeros({m_Osize+1}, torch::device(torch::kCUDA).dtype(torch::kInt32));
    torch::Tensor Pyramid = torch::zeros({2, MAX_LEVELS+2}, torch::device(torch::kCPU).dtype(torch::kInt32));
  
    uint*   d_Info = reinterpret_cast<uint*>(m_Info.data_ptr<int>());
    uint*   d_PrefixSum = reinterpret_cast<uint*>(m_PrefixSum.data_ptr<int>());
    int*    h_Pyramid = Pyramid.data_ptr<int>();
  
    void*           d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;
    kaolin::cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_Info, d_PrefixSum, m_Osize+1);

    torch::Tensor temp_storage = torch::zeros({(long)temp_storage_bytes}, torch::device(torch::kCUDA).dtype(torch::kByte));
    d_temp_storage = (void*)temp_storage.data_ptr<uchar>();

    // compute exclusive sum 1 element beyond end of list to get inclusive sum starting at d_PrefixSum+1
    d_ScanNodes << < (m_Osize + 1023) / 1024, 1024 >> >(m_Osize, Odata, d_Info);
    kaolin::cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_Info, d_PrefixSum, m_Osize+1); // carful with the +1

    uint psize;
    cudaMemcpy(&psize, d_PrefixSum+m_Osize, sizeof(uint), cudaMemcpyDeviceToHost);
    psize++; //plus one for root

    torch::Tensor Points = torch::zeros({psize, 4}, torch::device(torch::kCUDA).dtype(torch::kInt16));
    point_data* Pdata = reinterpret_cast<point_data*>(Points.data_ptr<short>());
 
    //TODO: share this memory with Points
    torch::Tensor Mortons = torch::zeros({psize}, torch::device(torch::kCUDA).dtype(torch::kInt64));
    morton_code* Mdata = reinterpret_cast<morton_code*>(Mortons.data_ptr<long>());
    
    int* pyramid = h_Pyramid;
    int* pyramidSum = h_Pyramid + MAX_LEVELS + 2;

    uint*           S = d_PrefixSum + 1; // this shouldn't matter
    morton_code*    M = Mdata;
    uchar*          O = Odata;

    morton_code m0 = 0;
    cudaMemcpy(M, &m0, sizeof(morton_code), cudaMemcpyHostToDevice);

    int Lsize = 1;
    uint currSum, prevSum = 0;

    uint sum = pyramid[0] = Lsize;
    pyramidSum[0] = 0;
    pyramidSum[1] = sum;

    int Level = 0;
    while (sum <= m_Osize)
    {
        NodesToMorton << <(Lsize + 1023) / 1024, 1024 >> >(Lsize, O, S, M, Mdata);
        O += Lsize;
        S += Lsize;
        M += Lsize;

        cudaMemcpy(&currSum, d_PrefixSum + prevSum + 1, sizeof(uint), cudaMemcpyDeviceToHost);

        Lsize = currSum - prevSum;
        prevSum = currSum;

        pyramid[++Level] = Lsize;
        sum += Lsize;
        pyramidSum[Level+1] = sum;
    }

    uint totalPoints = pyramidSum[Level+1];

    MortonToPoint << <(totalPoints + 1023) / 1024, 1024 >> >(totalPoints, Mdata, Pdata);
    cudaGetLastError();

    // assemble output tensors
    std::vector<at::Tensor> result;
    result.push_back(Points);
    result.push_back(Pyramid.index({Slice(None), Slice(None, Level+2)}).contiguous());

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();

    if ( err != cudaSuccess )
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    return result;
}


