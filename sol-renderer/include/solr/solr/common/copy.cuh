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

namespace solr {

    // Copies source array into a target array
    template<
        typename IN, 
        typename OUT>
    __global__ void copy(
        const IN* __restrict__ src,
        OUT* __restrict__ tgt,
        const int n
    ){
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        int stride = blockDim.x*gridDim.x;
        if (idx > n) return;
        for (int i=idx; i<n; i+=stride) {
            tgt[i] = src[i];
        }
    }

    // Copies source array into a target array,
    // for indices specified by idxes
    template<
        typename IN, 
        typename OUT>
    __global__ void copy(
        const IN* __restrict__ src,
        const int* __restrict__ idxes,
        OUT* __restrict__ tgt,
        const int n
    ){
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        int stride = blockDim.x*gridDim.x;
        if (idx > n) return;
        for (int _i=idx; _i<n; _i+=stride) {
            int i = idxes[_i];
            tgt[i] = src[_i];
        }
    }
}


