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

#include "SDF.h"

#include <torch/script.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <solr/solr.cuh>
#include <cnpy/cnpy.h>

extern inline void              cudaPrintError(const char* file, const int line);
#define CUDA_PRINT_ERROR() cudaPrintError(__FILE__, __LINE__)

//#define DEBUG

#ifdef DEBUG
#   define STRINGIFY2(X) #X
#   define STRINGIFY(X) STRINGIFY2(X)
#   define TIMER solr::PerfTimer timer = solr::PerfTimer()
#   define TIMER_CHECK(x) timer.check(x) 
#   define DEBUG_PRINT(x) std::cout << STRINGIFY(x) ":" << x << std::endl
#else
#   define TIMER
#   define TIMER_CHECK(x)
#   define DEBUG_PRINT(x)
#endif 

#define PROBE_CHECK(x) timer.check(x) 

namespace F = torch::nn::functional;
namespace I = torch::indexing;

SDF::SDF(void) {
}

SDF::~SDF(void) {
    cudaFree(pyramid_cf);
    cudaFree(m_pyramid);
    cudaFree(m_res);
}

void SDF::loadWeights(std::string filename) {
    
    cnpy::npz_t F = cnpy::npz_load(filename);
    
    auto h_opt = torch::TensorOptions().dtype(torch::kHalf);
    cnpy::NpyArray w0_npz = F["w0"];
    long w0_shape[3] = { static_cast<long>(w0_npz.shape[0]), 
                         static_cast<long>(w0_npz.shape[1]), 
                         static_cast<long>(w0_npz.shape[2]) };
    torch::Tensor _w0 = torch::from_blob(
            w0_npz.data<short>(), w0_shape, h_opt);

    for (long i=0; i<w0_shape[0]; ++i) {
        w0.push_back(_w0.index({i}));
        w0[i] = w0[i].to(torch::kFloat);
        w0[i] = w0[i].to(torch::kCUDA).transpose(0,1);
    }

    cnpy::NpyArray w1_npz = F["w1"];
    long w1_shape[3] = { static_cast<long>(w1_npz.shape[0]), 
                         static_cast<long>(w1_npz.shape[1]), 
                         static_cast<long>(w1_npz.shape[2]) };
    torch::Tensor _w1 = torch::from_blob(
            w1_npz.data<short>(), w1_shape, h_opt);

    for (long i=0; i<w1_shape[0]; ++i) {
        w1.push_back(_w1.index({i}));
        w1[i] = w1[i].to(torch::kFloat);
        w1[i] = w1[i].to(torch::kCUDA).transpose(0,1);
    }

    cnpy::NpyArray b0_npz = F["b0"];
    long b0_shape[2] = { static_cast<long>(b0_npz.shape[0]), 
                         static_cast<long>(b0_npz.shape[1]) };
    torch::Tensor _b0 = torch::from_blob(
            b0_npz.data<short>(), b0_shape, h_opt);

    for (long i=0; i<b0_shape[0]; ++i) {
        b0.push_back(_b0.index({i}));
        b0[i] = b0[i].to(torch::kFloat);
        b0[i] = b0[i].to(torch::kCUDA);
    }

    cnpy::NpyArray b1_npz = F["b1"];
    long b1_shape[2] = { static_cast<long>(b1_npz.shape[0]), 
                         static_cast<long>(b1_npz.shape[1]) };
    torch::Tensor _b1 = torch::from_blob(
            b1_npz.data<short>(), b1_shape, h_opt);

    for (long i=0; i<b1_shape[0]; ++i) {
        b1.push_back(_b1.index({i}));
        b1[i] = b1[i].to(torch::kFloat);
        b1[i] = b1[i].to(torch::kCUDA);
    }

    cnpy::NpyArray cc_npz = F["cc"];
    auto b_opt = torch::TensorOptions().dtype(torch::kByte);
    long cc_shape[2] = { static_cast<long>(cc_npz.shape[0]), 
                         static_cast<long>(cc_npz.shape[1]) };
    cc = torch::from_blob(cc_npz.data<char>(), cc_shape, b_opt);
    cc = cc.to(torch::kInt);
    cc = cc.to(torch::kCUDA);

    cnpy::NpyArray cf_npz = F["cf"];
    long cf_shape[2] = { static_cast<long>(cf_npz.shape[0]), 
                         static_cast<long>(cf_npz.shape[1]) };
    cf = torch::from_blob(cf_npz.data<char>(), cf_shape, h_opt);
    cf = cf.to(torch::kFloat);
    cf = cf.to(torch::kCUDA);

    cnpy::NpyArray pyramid_npz = F["pyramid"];
    for (long i=0; i<w0_shape[0]; ++i) {
        pyramid_cf_cpu.push_back(pyramid_npz.data<long>()[i]);
    }
}

void SDF::initTrinkets(uint num_pts, uint num_levels, uint* pyramid, ushort4* points) {
    m_points = points;
        
    cudaMalloc<solr::Trinket>(&trinkets, num_pts * sizeof(solr::Trinket));
    size_t sz = (num_levels+1) * sizeof(uint);
    cudaMallocManaged((void **)&m_pyramid, sz);
    for (int i=0; i<num_levels+1; ++i) {
        if (i == 0) {
            m_pyramid[i] = pyramid[i];
        } else {
            m_pyramid[i] = pyramid[i] + m_pyramid[i-1];
        }
    }

    size_t sz_res = (num_levels-1) * sizeof(uint);
    cudaMallocManaged((void **)&m_res, sz_res);
    for (int i=0; i<num_levels-1; ++i) {
        m_res[i] = 4 * std::pow(2, i);
    }

    uint offset_cf = 0;
    for (int i=2; i<num_levels+1; ++i) {
        uint offset = 0;
        uint parent_offset = 0;
        int parent_i = std::max(i-1, 2);
        
        for (int j=0; j<parent_i; j++) {
            parent_offset = m_pyramid[j];
        }

        for (int j=0; j<i; j++) {
            offset = m_pyramid[j];
        }
        
#       ifdef VERBOSE
        printf("offset on cf level %d            : %d\n", i-2, offset_cf);
        printf("# elem in cf level %d            : %d\n", i-2, pyramid_cf_cpu[i-2]);
        printf("offset on parent nuggets array %d: %d\n", parent_i, parent_offset);
        printf("# elem in parent nuggets array %d: %d\n", parent_i, pyramid[parent_i]);
        printf("offset on nuggets array %d       : %d\n", i, offset);
        printf("# elem in nuggets level %d       : %d\n", i, pyramid[i]);
        printf("\n");
#       endif
        
        const int threads = 1024;
        const int blocks = (pyramid[i] + threads - 1) / threads;
        // generalize this to multiple levels
        solr::index_trinket_kernel<<<blocks, threads>>>(
            points, 
            cc.data_ptr<int>(), 
            cf.data_ptr<float>(), 
            trinkets, 
            pyramid_cf_cpu[i-2],
            offset_cf,
            pyramid[i],
            offset,
            pyramid[parent_i],
            parent_offset,
            i);
        
        offset_cf += pyramid_cf_cpu[i-2];
        
#       ifdef DEBUG
            const int _tnum = 50;
            solr::Trinket* hosttrinkets = new solr::Trinket[_tnum];
            cudaMemcpy(hosttrinkets, trinkets+offset, _tnum*sizeof(solr::Trinket), cudaMemcpyDeviceToHost);
            for (int i=0; i<_tnum; ++i) {
                solr::Trinket _t = hosttrinkets[i];
                printf("%d: %d %d %d %d %d %d %d %d %d\n", 
                        i+offset, _t.v[0], _t.v[1], _t.v[2], _t.v[3],
                                  _t.v[4], _t.v[5], _t.v[6], _t.v[7], _t.parent);
            }
#       endif

    }
}

torch::Tensor SDF::getNormal(
    const torch::Tensor & x,
    const torch::Tensor & pidx,
    const torch::Tensor & hit,
    const int lod) {
    
    int nr = x.size(0);
    int nf = cf.size(0);
    int nl = w0.size();
    int fdim = cf.size(1);
    
    torch::Tensor active_x  = x.index({ hit });
    torch::Tensor active_pidx  = pidx.index({ hit });
    
    torch::Tensor active_idxes = torch::nonzero(hit).to(torch::kInt);
    int active_n = active_idxes.size(0);
    
    const int threads = 128;
    const int blocks = (active_n + threads - 1) / threads;
    
    auto f_opt = torch::TensorOptions().dtype(torch::kF32).device(x.device());
    torch::Tensor normal = torch::ones({ nr, 3 }, f_opt);

    std::vector<torch::Tensor> eps = {
        torch::tensor({0.001f, 0.0f, 0.0f}, f_opt),
        torch::tensor({0.0f, 0.001f, 0.0f}, f_opt),
        torch::tensor({0.0f, 0.0f, 0.001f}, f_opt)
    };


    #pragma unroll
    for (int i=0; i<3; ++i) {
        torch::Tensor x_fwd = x + eps[i];
        torch::Tensor x_bck = x - eps[i];
        torch::Tensor xs_fwd = torch::zeros({ active_n, fdim+3 }, f_opt);
        torch::Tensor xs_bck = torch::zeros({ active_n, fdim+3 }, f_opt);
        sparse_grid_sample_kernel<<<blocks, threads>>>(
            x_fwd.data_ptr<float>(),
            pidx.data_ptr<int>(),
            active_idxes.data_ptr<int>(),
            trinkets,
            cf.data_ptr<float>(),
            m_pyramid,
            m_res,
            xs_fwd.data_ptr<float>(),
            active_n, nf, fdim, nl, lod,
            // debug stuff
            cc.data_ptr<int>()
        );
        sparse_grid_sample_kernel<<<blocks, threads>>>(
            x_bck.data_ptr<float>(),
            pidx.data_ptr<int>(),
            active_idxes.data_ptr<int>(),
            trinkets,
            cf.data_ptr<float>(),
            m_pyramid,
            m_res,
            xs_bck.data_ptr<float>(),
            active_n, nf, fdim, nl, lod,
            // debug stuff
            cc.data_ptr<int>()
        );
        auto xsw0_fwd = torch::relu(torch::addmm(b0[lod], xs_fwd, w0[lod]));
        auto xsw0_bck = torch::relu(torch::addmm(b0[lod], xs_bck, w0[lod]));
        auto d_fwd =  torch::addmm(b1[lod], xsw0_fwd, w1[lod]);
        auto d_bck =  torch::addmm(b1[lod], xsw0_bck, w1[lod]);
        auto d = d_fwd - d_bck;
        normal.index_put_({ hit, i }, d.index({ I::Slice(), 0 }));
    }
    
    solr::normalize_kernel<<<blocks, threads>>>(
            active_idxes.data_ptr<int>(), 
            normal.data_ptr<float>(), 
            active_n);

    return normal;
    
}

std::vector<torch::Tensor> SDF::sphereTrace(
    const torch::Tensor & ray_o,
    const torch::Tensor & ray_d,
    const torch::Tensor & nuggets,
    const torch::Tensor & points,
    const torch::Tensor & info,
    const int lod)
{
    TIMER;

    // Convert to solr::Nuggets 
    solr::Nugget* nuggets_ptr = reinterpret_cast<solr::Nugget*>(nuggets.contiguous().data_ptr<int>());
    int nn = nuggets.size(0);

    // Rendering Parameters
    const int MARCH_ITER = 50;
    const float MIN_DIS = 0.0003;
    const float far =   5.0;

    // Tensor sizes
    int nr = ray_o.size(0); // # rays
    int nf = cf.size(0);    // # feats
    int nl = w0.size();     // # lods
    
    int fdim = cf.size(1);  // feat dim
    
    auto f_opt = torch::TensorOptions().dtype(torch::kF32).device(ray_o.device());
    torch::Tensor x = ray_o.clone();
    torch::Tensor t = torch::zeros({ nr, 1 }, f_opt);
    torch::Tensor d = torch::zeros({ nr, 1 }, f_opt);
    torch::Tensor dprev = torch::zeros({ nr, 1 }, f_opt);
    torch::Tensor normal = torch::ones({ nr, 3 }, f_opt);
    torch::Tensor ray_inv = 1.0 / ray_d;

    auto i_opt = torch::TensorOptions().dtype(torch::kInt32).device(ray_o.device());
    
    // Tensor to store the hit-point index of each ray
    torch::Tensor pidx = torch::zeros({ nr, 1 }, i_opt) - 1;

    // Indices of beginnings of ray-nugget lists
    torch::Tensor info_idxes = torch::nonzero(info).index({I::Slice(), 0}).to(torch::kInt);

    // # ray-nugget hits
    int n_iidx = info_idxes.size(0); 

    // Voxel size
    int voxel_res = pow(2, lod+2);
    float voxel_radius = (1.0 / voxel_res);
    
    // cond is the active rays
    // hit is the rays that have hit a surface
    auto b_opt = torch::TensorOptions().dtype(torch::kBool).device(ray_o.device());
    torch::Tensor cond = torch::zeros({nr}, b_opt); // by default, no rays are active
    torch::Tensor hit = torch::zeros({nr}, b_opt);

    TIMER_CHECK(" init  ");
    const int _aabb_threads = 256;
    const int _aabb_blocks = (n_iidx + _aabb_threads - 1) / _aabb_threads;
    solr::ray_aabb_kernel<<<_aabb_blocks, _aabb_threads>>>(
        ray_o.data_ptr<float>(),
        ray_d.data_ptr<float>(),
        ray_inv.data_ptr<float>(),
        ray_o.data_ptr<float>(),
        nuggets_ptr,
        points.data_ptr<short>(),
        info.data_ptr<int>(),
        info_idxes.data_ptr<int>(),
        voxel_radius,
        true,
        x.data_ptr<float>(),
        t.data_ptr<float>(),
        cond.data_ptr<bool>(),
        pidx.data_ptr<int>(),
        nn,
        n_iidx);
    TIMER_CHECK(" trace ");
    
    // # of voxel centers
    //int nvc = vc[lod].size(0);

    //return {x, t, cond, normal}; // uncomment to return voxels
    
    TIMER_CHECK(" post  ");

    CUDA_PRINT_ERROR();
    
    for (int i=0; i<MARCH_ITER; ++i) {
        
        // probably write a cuda kernel here... first a sum kernel, allocate, then populate?
        torch::Tensor active_idxes = torch::nonzero(cond).index({I::Slice(), 0}).to(torch::kInt);
        int n_active = active_idxes.size(0); // # active
        
        TIMER_CHECK("  get sizes");
        
        if (n_active == 0) {
            DEBUG_PRINT(i);
            break;
        }
        TIMER_CHECK("  done?    ");

        // Concat [x, f]
        torch::Tensor xs = torch::zeros({ n_active, fdim+3 }, f_opt);
        TIMER_CHECK("  allocate ");

        const int _sparse_threads = 128;
        const int _sparse_blocks = (n_active + _sparse_threads - 1) / _sparse_threads;
        sparse_grid_sample_kernel<<<_sparse_blocks, _sparse_threads>>>(
            x.data_ptr<float>(),
            pidx.data_ptr<int>(),
            active_idxes.data_ptr<int>(),
            trinkets,
            cf.data_ptr<float>(),
            m_pyramid,
            m_res,
            xs.data_ptr<float>(),
            n_active, nf, fdim, nl, lod,
            // debug stuff
            cc.data_ptr<int>()
        );
        TIMER_CHECK("  sparse   ");

        CUDA_PRINT_ERROR();

        auto xsw0 = torch::relu(torch::addmm(b0[lod], xs, w0[lod]));
        auto _d =  torch::addmm(b1[lod], xsw0, w1[lod]);
        TIMER_CHECK("  d        ");
        
        int _step_threads = 128;
        int _step_blocks = (n_active + _step_threads - 1) / _step_threads;
        solr::step_kernel<<<_step_blocks, _step_threads>>>(
            ray_o.data_ptr<float>(),
            ray_d.data_ptr<float>(),
            active_idxes.data_ptr<int>(),
            _d.data_ptr<float>(),
            x.data_ptr<float>(),
            t.data_ptr<float>(),
            d.data_ptr<float>(),
            dprev.data_ptr<float>(),
            cond.data_ptr<bool>(),
            hit.data_ptr<bool>(),
            n_active);
        TIMER_CHECK("  step     ");
    
        CUDA_PRINT_ERROR();

        const int _sample_threads = 128;
        const int _sample_blocks = (n_iidx + _sample_threads - 1) / _sample_threads;
        solr::ray_aabb_kernel<<<_sample_blocks, _sample_threads>>>(
            ray_o.data_ptr<float>(),
            ray_d.data_ptr<float>(),
            ray_inv.data_ptr<float>(),
            x.data_ptr<float>(),
            nuggets_ptr,
            points.data_ptr<short>(),
            info.data_ptr<int>(),
            info_idxes.data_ptr<int>(),
            voxel_radius,
            false,
            x.data_ptr<float>(),
            t.data_ptr<float>(),
            cond.data_ptr<bool>(),
            pidx.data_ptr<int>(),
            nn,
            n_iidx);
        
        TIMER_CHECK("  sample   ");
        CUDA_PRINT_ERROR();
    }
    TIMER_CHECK(" st    ");

    normal = getNormal(x, pidx, hit, lod);

    CUDA_PRINT_ERROR();

    return {x, t, hit, normal};
}

