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

}

