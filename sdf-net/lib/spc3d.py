# The MIT License (MIT)
#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np

def to_morton(p):
    mcode = np.uint64(0)
    for i in range(16):
        mcode = np.bitwise_or(mcode, np.uint64(\
            np.bitwise_and(p[0], 0x1 << i) << (2 * i + 2) | \
            np.bitwise_and(p[1], 0x1 << i) << (2 * i + 1) | \
            np.bitwise_and(p[2], 0x1 << i) << (2 * i + 0)))      
    return mcode

def to_point(mcode):
    p = np.zeros(3, dtype=np.uint16)
    for i in range(16):
        branch = np.uint16(np.bitwise_and(mcode, np.uint64((0x7 << 3*i)))>>np.uint64(3*i))
        p[0] |= ((branch&0x4)>>2)<<i
        p[1] |= ((branch&0x2)>>1)<<i
        p[2] |= ((branch&0x1)>>0)<<i
    return p

class Subdivide:
    def __init__(self, p, start=0):
        self.i = start
        self.max = 8
        self.p = 2*p

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.max:
            result = np.array([self.p[0]+(self.i >> 2), 
                               self.p[1]+((self.i >> 1)&0x1), 
                               self.p[2]+(self.i & 0x1)], dtype=np.uint16)
            self.i += 1
            return result
        else:
            raise StopIteration

class SPC3D:
    def __init__(self, level, nodebytes=None):
        self.level = level
        self.max_levels = level+10
        self.max_size = 0x1<<(self.max_levels)
        self.pdata = np.zeros((self.max_size, 3), dtype=np.uint16)
        self.proot = self.pdata       
        self.psize = 0
        self.mdata = np.zeros(self.max_size, dtype=np.uint64)
        self.mroot = self.mdata 

        if nodebytes is None:
            self.osize = 0
            self.odata = np.zeros(self.max_size, dtype=np.uint8)
        else:
            self.osize = nodebytes.size
            self.odata = nodebytes

        self.oroot = self.odata 

        self.ddata = np.zeros(self.max_size, dtype=np.uint32)
        self.sdata = np.zeros(self.max_size, dtype=np.uint32)
        self.ndata = np.zeros(self.max_size, dtype=np.int32)
        self.nroot = self.ndata
        self.pyramid = np.zeros(self.max_levels, dtype=np.uint32)
        self.pyramid_sum = np.zeros(self.max_levels, dtype=np.uint32)

        # lookup table to count ones in byte
        self.num_ones = np.array([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,\
                                  1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,\
                                  1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,\
                                  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,\
                                  1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,\
                                  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,\
                                  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,\
                                  3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,\
                                  1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,\
                                  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,\
                                  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,\
                                  3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,\
                                  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,\
                                  3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,\
                                  3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,\
                                  4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8 ])

    # contains --------------------------------------------------------------------------                               
    def contains_surface(self, f, p, level):
        if level < 1:
            return True

        minval = 1.0
        maxval = -1.0

        for i in range(8):
            invL = 1.0/(0x1<<(level-1))
            x = invL*(p[0] + (i >> 2)) - 1.0
            y = invL*(p[1] + ((i >> 1)&0x1)) - 1.0
            z = invL*(p[2] + (i & 0x1)) - 1.0
            val = f(x, y, z)
            minval = min(minval, val)
            maxval = max(maxval, val)

        return 1 if minval <= 0.0 and maxval >= 0.0 else 0 

    # recursive construction --------------------------------------------------------------------------
    def construct(self, f, p = np.array([0, 0, 0], dtype=np.uint16), level = 0):
        if self.contains_surface(f, p, level):
            if level == self.level:
                self.pdata[self.psize] = p
                self.psize += 1
            else:
                for q in Subdivide(p):
                    self.construct(f, q, level+1)


    # breadth-first construction --------------------------------------------------------------------------
    def inclusive_sum(self, num):
        for t in range(num):
            self.sdata[t] = self.ddata[t] if t==0 else self.sdata[t-1] + self.ddata[t]
        return self.sdata[num-1]

    def exclusive_sum(self, num):
        for t in range(num):
            self.sdata[t] = 0 if t==0 else self.sdata[t-1] + self.ddata[t-1]
        return self.sdata[num-1]

    def oracle(self, f, pdata, level, num):
        for k in range(num):
            self.ddata[k] = self.contains_surface(f, pdata[k], level)

    def subdivide(self, pdata_in, num):
        pdata_out = pdata_in[num:]
        for k in range(num):
            if self.ddata[k]:
                idx = 8 * (self.sdata[k]-1)
                for q in Subdivide(pdata_in[k]):
                    pdata_out[idx] = q
                    idx += 1

    def compactify(self, pdata_in, num):
        pdata_out = pdata_in[num:]
        for k in range(num):
            if self.ddata[k]:
                pdata_out[self.sdata[k] - 1] = pdata_in[k]

    # interative breadth-first version ----------------------------------------------------------------------
    def breadth_first(self, f):
        self.proot = self.pdata

        num = 1
        for i in range(self.level+1):
            self.oracle(f, self.proot, i, num)
            self.psize = self.inclusive_sum(num)
 
            if i < self.level:
                self.subdivide(self.proot, num)
            else:
                self.compactify(self.proot, num)

            self.proot = self.proot[num:]
            num = 8 * self.psize



    # binary encoding --------------------------------------------------------------------------   
    def set_pyramid_sum(self):
        sum = 0
        for i in range(self.level+1):
            sum += self.pyramid[i]
            self.pyramid_sum[i] = sum

    def scan_nodes(self, num, o):
        for t in range(num):
            self.ddata[t] = self.num_ones[o[t]]

    def point_to_morton(self, num, pdata, mdata):
        for t in range(num):
            mdata[t] = to_morton(pdata[t])

    def common_parent(self, num, mdata):
        for t in range(num):
            if t == 0:
                self.ddata[t] = 1
            else:
                self.ddata[t] = 0 if (mdata[t-1]>>np.uint64(3)) == (mdata[t]>>np.uint64(3)) else 1
 

    def compacify_nodes(self, num, m_in, m_out, odata):
        for t in range(num):
            if self.ddata[t]:
                i = self.sdata[t] - 1
                m_out[i] = m_in[t] >> np.uint64(3)

                code = 0
                v = t
                while True:
                    child_idx = np.uint32(m_in[v] & np.uint64(7))
                    code = code | 0x1 << (child_idx)
                    v = v + 1
                    if (v == num) or self.ddata[v] == 1:
                        break
                odata[i] = code

    def points_to_nodes(self):
        m = self.mdata
        oroot = self.max_size

        self.point_to_morton(self.psize, self.proot, m)

        prev = self.pyramid[self.level] = self.psize
        for i in range(self.level, 0, -1):
            self.common_parent(prev, m)
            curr = self.inclusive_sum(prev)

            self.pyramid[i-1] = curr

            oroot = oroot - curr

            self.compacify_nodes(prev, m, m[prev:], self.odata[oroot:])

            m = m[prev:]
            prev = curr

        sum = 0
        for i in range(self.level+1):
            sum += self.pyramid[i]
            self.pyramid_sum[i] = sum

        self.oroot = self.odata[oroot:]
        self.osize = self.max_size - oroot

        self.scan_nodes(self.osize, self.oroot)
        self.exclusive_sum(self.osize)

        return self.osize




    # binary decoding --------------------------------------------------------------------------
    def morton_to_point(self, num, mdata, pdata):
        for t in range(num):
           pdata[t] = to_point(mdata[t])

    def nodes_to_morton(self, num, o, p, mi, mo):
        for t in range(num):
            addr = p[t]
            for i in range(7,-1,-1):
                if o[t]&(0x1 << i):
                    mo[addr] = 8 * mi[t] + i
                    addr = addr - 1

    def nodes_to_points(self):
        o = self.oroot
        p = self.sdata
        m = self.mdata
        m[0] = np.uint64(0)

        self.scan_nodes(self.osize, o)
        self.inclusive_sum(self.osize)

        self.pyramid[0] = num = 1
        prev = 0
        #for i in range(self.level):
        i = 0
        while o.size > 0:
            self.nodes_to_morton(num, o, p, m, self.mdata)
            o = o[num:]
            p = p[num:]
            m = m[num:]

            curr = self.sdata[prev]
            num = curr - prev
            prev = curr
            self.pyramid[i+1] = num
            i += 1

        self.set_pyramid_sum()
        self.morton_to_point(num, m, self.pdata)

        self.scan_nodes(self.osize, self.oroot)
        self.exclusive_sum(self.osize)

        return num



    # # neighbor finding --------------------------------------------------------------------------
    def Identify(self, p):
        o = self.oroot
        psum = self.sdata

        # print(0x1<<self.level - 1)

        maxval = (0x1<<self.level) - 1
        for i in range(3):
            if p[i] < 0 or p[i] > maxval:
                return -1

        ord = 0
        for l in range(0, self.level):
            depth = self.level - l - 1
            mask = 0x1<<depth
            child_idx = ((mask&p[0]) << 2 | (mask&p[1]) << 1 | (mask&p[2])) >> depth

            bits = o[ord]

            if bits&(0x1 << child_idx) != 0:
                cnt = self.num_ones[bits&((0x2 << child_idx) - 1)]

                ord = psum[ord] + cnt

                if depth == 0:
                    return ord - self.pyramid_sum[l]
            else:
                return -1

        return -1 # should not happen


    # file output
    def write_to_file(self, filename):
        with open(filename, "wb") as f:
            d = np.array([self.level, self.psize], dtype=np.uint32)
            d.tofile(f)
            d = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            d.tofile(f)
            h = 0x1<<self.level
            d = np.array([0, 0, 0, h, h, h], dtype=np.uint16)
            d.tofile(f)
            d = np.array([self.osize], dtype=np.uint32)
            d.tofile(f)
            self.oroot.tofile(f)

            colors = np.zeros(self.psize, dtype=np.uint32)
            for k in range(self.psize):
                r = np.uint8(255.0*self.proot[k,0]/h)
                g = np.uint8(255.0*self.proot[k,1]/h)
                b = np.uint8(255.0*self.proot[k,2]/h)
                colors[k] = np.uint32(r | g << 16 | b << 24) 

            colors.tofile(f)

    def write_to_torch_input(self, filename, Labels):
        NodeBytes = self.oroot
        np.savez(filename, NodeBytes, Labels)

