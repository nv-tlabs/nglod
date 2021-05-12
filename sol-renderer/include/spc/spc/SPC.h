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

#include <torch/torch.h>

#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_runtime.h>
#include <string>

#include <cnpy/cnpy.h>
#include "vector_types.h"


#define MAX_LEVELS          16
#define SCAN_MAX_VOXELS     (0x1 << 27) 


typedef unsigned char       uchar;
typedef unsigned short      ushort;
typedef unsigned int        uint;

typedef unsigned long long  morton_code;
typedef ushort4             point_data;

typedef uchar4              color_type;
typedef char3               normal_type;


static __inline__ __host__ __device__ point_data make_point_data(ushort x, ushort y, ushort z)
{
    point_data p;
    p.x = x; p.y = y; p.z = z; p.w = 0;
    return p;
}

static __inline__ __host__ __device__ color_type make_color_type(float x, float y, float z, float w)
{
    return make_uchar4(ushort(255.0f*x), ushort(255.0f*y), ushort(255.0f*z), ushort(255.0f*w));
}

static __inline__ __host__ __device__ morton_code ToMorton(point_data V)
{
    morton_code mcode = 0;

    for (uint i = 0; i < 16; i++)
    {
        uint i2 = i + i;
        morton_code x = V.x;
        morton_code y = V.y;
        morton_code z = V.z;

        mcode |= (z&(0x1 << i)) << i2;
        mcode |= (y&(0x1 << i)) << ++i2;
        mcode |= (x&(0x1 << i)) << ++i2;
    }

    return mcode;
}


static __inline__ __host__ __device__ point_data ToPoint(morton_code mcode)
{
    point_data p = make_point_data(0, 0, 0);

    for (int i = 0; i < 16; i++)
    {
        p.x |= (mcode&(0x1ull << (3 * i + 2))) >> (2 * i + 2);
        p.y |= (mcode&(0x1ull << (3 * i + 1))) >> (2 * i + 1);
        p.z |= (mcode&(0x1ull << (3 * i + 0))) >> (2 * i + 0);
    }

    return p;
}


class SPC
{
private:
    torch::Tensor m_Octree;
    torch::Tensor m_Points;
    torch::Tensor m_Info;
    torch::Tensor m_PrefixSum;
    torch::Tensor m_Pyramid;

    uint          m_Level = 0;
    uint          m_Psize = 0;
    uint          m_Osize = 0;

    std::vector<at::Tensor> SetGeometry(torch::Tensor Octree);

public:
    SPC();
    ~SPC();

    torch::Tensor GetOctree() { return m_Octree; }
    torch::Tensor GetPoints() { return m_Points; }
    torch::Tensor GetPoints(uint l);
    torch::Tensor GetPyramid() { return m_Pyramid; }

    uint        GetPSize() { return m_Psize; }
    uint        GetOSize() { return m_Osize; }
    uint        GetLevel() { return m_Level; }

    point_data* GetProotGPU(uint l);

    point_data* GetProotGPU() { return reinterpret_cast<point_data*>(m_Points.data_ptr<short>()); }
    uchar*      GetOrootGPU() { return reinterpret_cast<uchar*>(m_Octree.data_ptr<uchar>()); }
    uint*       GetInfo() { return reinterpret_cast<uint*>(m_Info.data_ptr<int>()); }
    uint*       GetPrefixSum() { return reinterpret_cast<uint*>(m_PrefixSum.data_ptr<int>()); }
    uint*       GetPyramidPtr() { return reinterpret_cast<uint*>(m_Pyramid.data_ptr<int>()); }

    void SaveNPZ(std::string filename);
    void LoadNPZ(std::string filename);
};












