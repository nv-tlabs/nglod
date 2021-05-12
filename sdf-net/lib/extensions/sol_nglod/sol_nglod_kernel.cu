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

#include <torch/extension.h>

#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <string>

//#define DEBUG

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

namespace F = torch::nn::functional;
namespace I = torch::indexing;

class PerfTimer {

    cudaStream_t m_stream;
    std::chrono::time_point<std::chrono::system_clock> m_curr;

public:
    
    PerfTimer() {
        m_stream = at::cuda::getCurrentCUDAStream();    
        cudaStreamSynchronize(m_stream);
        m_curr = std::chrono::system_clock::now();
    }

    void check(std::string checkpoint) {
        cudaStreamSynchronize(m_stream);
        auto end = std::chrono::system_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end-m_curr);
        std::cout << checkpoint << ": " << elapsed_seconds.count() << " us" << std::endl;
        m_curr = end;
    }
};

template <typename scalar_t>
__global__ void aabb_kernel(
    const scalar_t* __restrict__ ray_o,
    const scalar_t* __restrict__ ray_d,
    scalar_t* __restrict__ x,
    scalar_t* __restrict__ t,
    bool* __restrict__ hit,
    const int n
){
    // From Majercik et. al 2018
    // This kernel, for now, assumes the AABB is 0 centered with radius 1
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    if (idx > n) return;
    
    for (int i=idx; i<n; i+=stride) {
        const float3 dir = make_float3(ray_d[3*i], ray_d[3*i+1], ray_d[3*i+2]);
        const float3 invdir = make_float3(1.0/ray_d[3*i], 1.0/ray_d[3*i+1], 1.0/ray_d[3*i+2]);
        
        const float sgn0 = signbit(dir.x) ? 1.0 : -1.0;
        const float sgn1 = signbit(dir.y) ? 1.0 : -1.0;
        const float sgn2 = signbit(dir.z) ? 1.0 : -1.0;
        
        float _t = 500.0;
        bool _hit = false;
        float3 _normal = make_float3(0.0, 0.0, 0.0);

        const float3 o = make_float3(ray_o[3*i], ray_o[3*i+1], ray_o[3*i+2]);
        float cmax = fmaxf(fmaxf(fabs(o.x), fabs(o.y)), fabs(o.z));    

        float r = 1.0;

        float winding = cmax < r ? -1.0 : 1.0;
        if (winding < 0) {
            continue; // check perf implications here if slow
        }
        
        float d0 = (r * winding * sgn0 - o.x) * invdir.x;
        float d1 = (r * winding * sgn1 - o.y) * invdir.y;
        float d2 = (r * winding * sgn2 - o.z) * invdir.z;

        float ltxy = fmaf(dir.y, d0, o.y);
        float ltxz = fmaf(dir.z, d0, o.z);
        
        float ltyx = fmaf(dir.x, d1, o.x);
        float ltyz = fmaf(dir.z, d1, o.z);
        
        float ltzx = fmaf(dir.x, d2, o.x);
        float ltzy = fmaf(dir.y, d2, o.y);
        
        bool test0 = (d0 >= 0.0) && (fabs(ltxy) < r) && (fabs(ltxz) < r);
        bool test1 = (d1 >= 0.0) && (fabs(ltyx) < r) && (fabs(ltyz) < r);
        bool test2 = (d2 >= 0.0) && (fabs(ltzx) < r) && (fabs(ltzy) < r);

        float3 sgn = make_float3(0.0, 0.0, 0.0);

        if (test0) { sgn.x = sgn0; }
        else if (test1) { sgn.y = sgn1; }
        else if (test2) { sgn.z = sgn2; }

        float d = 0.0;
        if (sgn.x != 0.0) { d = d0; } 
        else if (sgn.y != 0.0) { d = d1; }
        else if (sgn.z != 0.0) { d = d2; }

        bool __hit = (sgn.x != 0.0) | (sgn.y != 0.0) | (sgn.z != 0.0);
        if (__hit && d < _t) {
            _t = d;
            hit[i] = __hit;
            x[3*i]   = fmaf(ray_d[3*i],   _t, ray_o[3*i]  );
            x[3*i+1] = fmaf(ray_d[3*i+1], _t, ray_o[3*i+1]);
            x[3*i+2] = fmaf(ray_d[3*i+2], _t, ray_o[3*i+2]);
            t[i] = _t;
        }
        //normal[3*i]   = _normal.x;
        //normal[3*i+1] = _normal.y;
        //normal[3*i+2] = _normal.z;
        //x[3*i]   = fmaf(ray_d[3*i],   t[i], ray_o[3*i]  );
        //x[3*i+1] = fmaf(ray_d[3*i+1], t[i], ray_o[3*i+1]);
        //x[3*i+2] = fmaf(ray_d[3*i+2], t[i], ray_o[3*i+2]);
    
    }
}

std::vector<torch::Tensor> f_aabb(
    const torch::Tensor ray_o,
    const torch::Tensor ray_d) {

    int n = ray_o.size(0);
    //torch::Tensor x = torch::zeros_like(ray_o);
    torch::Tensor x = ray_o.clone();
    
    auto b_opt = torch::TensorOptions().dtype(torch::kBool).device(ray_o.device());
    torch::Tensor hit = torch::zeros({n}, b_opt);
    
    auto f_opt = torch::TensorOptions().dtype(torch::kF32).device(ray_o.device());
    torch::Tensor t = torch::zeros({n, 1}, f_opt);

    const int _aabb_threads = 256;
    const int _aabb_blocks = (n + _aabb_threads - 1) / _aabb_threads;
    AT_DISPATCH_FLOATING_TYPES(ray_o.scalar_type(), "sol_nglod_kernel", ([&] {
        aabb_kernel<scalar_t><<<_aabb_blocks, _aabb_threads>>>(
            ray_o.data_ptr<scalar_t>(),
            ray_d.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            t.data_ptr<scalar_t>(),
            hit.data_ptr<bool>(),
            n); 
    }));
    return { x, t, hit };

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("aabb", &f_aabb, "Ray-AABB intersect");
}


