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

from dataclasses import asdict, astuple, fields, dataclass
from typing import Union, List

import torch
import numpy as np

@dataclass
class RenderBuffer:
    x : Union[torch.Tensor, None] = None
    min_x : Union[torch.Tensor, None] = None
    hit : Union[torch.Tensor, None] = None
    depth : Union[torch.Tensor, None] = None
    relative_depth : Union[torch.Tensor, None] = None
    normal : Union[torch.Tensor, None] = None
    rgb : Union[torch.Tensor, None] = None
    shadow : Union[torch.Tensor, None] = None
    ao : Union[torch.Tensor, None] = None
    view : Union[torch.Tensor, None] = None
    err : Union[torch.Tensor, None] = None
    albedo : Union[torch.Tensor, None] = None

    def __iter__(self): 
        return iter(astuple(self))

    def __add__(self, other):
        def _proc(pair):
            if None not in pair:
                return torch.cat(pair)
            elif pair[0] is not None and pair[1] is None:
                return pair[0]
            elif pair[0] is None and pair[1] is not None:
                return pair[1]
            else:
                return None

        return self.__class__(*(map(_proc, zip(self, other))))

    def _apply(self, fn):
        data = {} 
        for f in fields(self):
            attr = getattr(self, f.name)
            data[f.name] = None if attr is None else fn(attr)
        return self.__class__(**data)

    def cuda(self):
        fn = lambda x : x.cuda()
        return self._apply(fn)
         
    def cpu(self):
        fn = lambda x : x.cpu()
        return self._apply(fn)
    
    def detach(self):
        fn = lambda x : x.detach()
        return self._apply(fn)
    
    def byte(self):
        fn = lambda x : x.byte()
        return self._apply(fn)

    def reshape(self, *dims : List[int]):
        fn = lambda x : x.reshape(*dims)
        return self._apply(fn)
    
    def transpose(self):
        fn = lambda x : x.permute(1,0,2)
        return self._apply(fn)
    
    def numpy(self):
        fn = lambda x : x.numpy()
        return self._apply(fn)
    
    def float(self):
        fn = lambda x : x.float()
        return self._apply(fn)

    def exrdict(self):
        _dict = asdict(self)
        _dict = {k:v for k,v in _dict.items() if v is not None}
        if 'rgb' in _dict:
            _dict['default'] = _dict['rgb']
            del _dict['rgb']
        return _dict

    def image(self):
        # Unfinished
        norm = lambda arr : ((arr + 1.0) / 2.0) if arr is not None else None
        bwrgb = lambda arr : torch.cat([arr]*3, dim=-1) if arr is not None else None
        rgb8 = lambda arr : (arr * 255.0) if arr is not None else None

        #x = rgb8(norm(self.x))
        #min_x = rgb8(norm(self.min_x))
        hit = rgb8(bwrgb(self.hit))
        depth = rgb8(bwrgb(self.relative_depth))
        normal = rgb8(norm(self.normal))
        #ao = 1.0 - rgb8(bwrgb(self.ao))
        rgb = rgb8(self.rgb)
        #return self.__class__(x=x, min_x=min_x, hit=hit, depth=depth, normal=normal, ao=ao, rgb=rgb)
        return self.__class__(hit=hit, normal=normal, rgb=rgb, depth=depth)

    @staticmethod
    def mean(*rblst):
        rb = RenderBuffer()
        n = len(rblst)
        for _rb in rblst:
            for f in fields(_rb):
                attr = getattr(_rb, f.name)
                dest = getattr(rb, f.name)
                if attr is not None and dest is not None:
                    setattr(rb, f.name, dest + attr)
                elif dest is None:
                    setattr(rb, f.name, attr)

        for f in fields(rb):
            dest = getattr(rb, f.name)
            if dest is not None:
                setattr(rb, f.name, dest / float(n))
        return rb

if __name__ == '__main__':
    hw = 1024
    hit = torch.zeros(hw, hw, 1).bool()
    rgb = torch.zeros(hw, hw, 3).float()

    rb0 = RenderBuffer(hit=hit.clone(), rgb=rgb.clone())
    rb1 = RenderBuffer(hit=hit, rgb=rgb)
    rb1.rgb += 1.0

    avg = RenderBuffer.mean(rb0, rb1)

    import pdb; pdb.set_trace()
