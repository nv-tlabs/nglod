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

import os
import sys
import time
import numpy as np
import pprint
import argparse
import torch

# General utilities

# This is the bridge between an argparse based approach and a non-argparse one
def setparam(args, param, paramstr):
    argsparam = getattr(args, paramstr, None)
    if param is not None or argsparam is None:
        return param
    else:
        return argsparam


# from https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
# Define a context manager to suppress stdout and stderr.
class suppress_output(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

def image_to_np(img):
    return np.array(img).transpose(2,0,1)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def colorize_time(elapsed):
    if elapsed > 1e-3:
        return bcolors.FAIL + "{:.3e}".format(elapsed) + bcolors.ENDC
    elif elapsed > 1e-4:
        return bcolors.WARNING + "{:.3e}".format(elapsed) + bcolors.ENDC
    elif elapsed > 1e-5:
        return bcolors.OKBLUE + "{:.3e}".format(elapsed) + bcolors.ENDC
    else:
        return "{:.3e}".format(elapsed)

class PerfTimer():
    def __init__(self, activate=False):
        self.prev_time = time.process_time()
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.prev_time_gpu = self.start.record()
        self.counter = 0
        self.activate = activate

    def reset(self):
        self.counter = 0
        self.prev_time = time.process_time()
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.prev_time_gpu = self.start.record()

    def check(self, name=None):
        if self.activate:
            cpu_time = time.process_time() - self.prev_time
            cpu_time = colorize_time(cpu_time)
          
            self.end.record()
            torch.cuda.synchronize()

            gpu_time = self.start.elapsed_time(self.end) / 1e3
            gpu_time = colorize_time(gpu_time)
            if name:
                print("CPU Checkpoint {}: {} s".format(name, cpu_time))
                print("GPU Checkpoint {}: {} s".format(name, gpu_time))
            else:
                print("CPU Checkpoint {}: {} s".format(self.counter, cpu_time))
                print("GPU Checkpoint {}: {} s".format(self.counter, gpu_time))

            self.prev_time = time.process_time()
            self.prev_time_gpu = self.start.record()
            self.counter += 1
            return cpu_time, gpu_time


