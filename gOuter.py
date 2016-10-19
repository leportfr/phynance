#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

""" 
Multiples two square matrices together using multiple blocks and shared memory. 
Each thread block is assigned a "tile" of the resulting matrix and is responsible
for generating the elements in that tile.  Each thread in a block computes one element 
of the tile.
"""

import numpy as np
import time
from numpy import linalg as la
from pycuda import driver, compiler, gpuarray, tools

# -- initialize the device
import pycuda.autoinit

class gOuter:
    def __init__(self, bsizeA, bsizeB):
        self.kernel_code_template = """
        __global__ void MatrixMulKernel(float *A, float *B, float *C, const int vSL)
        {
          // Block index
          const unsigned int bA = blockIdx.x;
          const unsigned int bB = blockIdx.y;
        
          // Thread index
          const unsigned int tx = threadIdx.x;
          const unsigned int ty = threadIdx.y;
        
          // Index of the first sub-matrix of A processed by the block
          const unsigned int aBegin = %(BLOCK_SIZE_A)s * bA;
        
          // Index of the first sub-matrix of B processed by the block
          const unsigned int bBegin = %(BLOCK_SIZE_B)s * bB;
        
          // Shared memory for the sub-matrix of A
          __shared__ float As[%(BLOCK_SIZE_A)s];
          // Shared memory for the sub-matrix of B
          __shared__ float Bs[%(BLOCK_SIZE_B)s];
        
          // Load the matrices from global memory to shared memory
          // each thread loads one element of each matrix
          As[tx] = A[aBegin + tx];
          Bs[ty] = B[bBegin + ty];
          // Synchronize to make sure the matrices are loaded
          __syncthreads();
        
          // Multiply the two matrices together;
          // each thread computes one element
          // of the block sub-matrix
          C[(aBegin + tx)*vSL + bBegin + ty] = As[tx] * Bs[ty];
        
          // Synchronize to make sure that the preceding
          // computation is done before loading two new
          // sub-matrices of A and B in the next iteration
          __syncthreads();
        }
        """
        
        # define size of blocks and tiles sub-matrix 
        # (we assume that the block size is same as tile size)
        self.BLOCK_SIZE_A = bsizeA
        self.BLOCK_SIZE_B = bsizeB
        
    def gCompile(self):
        # get the kernel code from the template 
        # by specifying the constants MATRIX_SIZE and BLOCK_SIZE
        kernel_code = self.kernel_code_template % { 
            'BLOCK_SIZE_A': self.BLOCK_SIZE_A,
            'BLOCK_SIZE_B': self.BLOCK_SIZE_B
            }
        
        # compile the kernel code
        mod = compiler.SourceModule(kernel_code)
        
        # get the kernel function from the compiled module
        self.matrixmul = mod.get_function("MatrixMulKernel")

    def gOuter(self, a_gpu, b_gpu, c_gpu):
        self.matrixmul(a_gpu, b_gpu, c_gpu, np.int32(b_gpu.size),
            # grid of multiple blocks
            grid = (a_gpu.size // self.BLOCK_SIZE_A, b_gpu.size // self.BLOCK_SIZE_B),
            # block of multiple threads
            block = (self.BLOCK_SIZE_A, self.BLOCK_SIZE_B, 1), 
        )
    
if __name__ == '__main__':
    # define the (square) matrix size
    VECTOR_SIZE_N = 50
    VECTOR_SIZE_L = 100
    numvectors = 100
    
    gOuter = gOuter(50,100)
    gOuter.gCompile()    
    
    # create two random square matrices
    a_cpu = np.array([np.random.randn(VECTOR_SIZE_N).astype(np.float32) for i in range(numvectors)])
    b_cpu = np.array([np.random.randn(VECTOR_SIZE_L).astype(np.float32) for i in range(numvectors)])
#    a_cpu = np.reshape(np.arange(0,MATRIX_SIZE*MATRIX_SIZE),(MATRIX_SIZE,MATRIX_SIZE)).astype(np.float32)
#    b_cpu = np.reshape(np.arange(0,MATRIX_SIZE*MATRIX_SIZE),(MATRIX_SIZE,MATRIX_SIZE)).astype(np.float32)
    
    # compute reference on the CPU to verify GPU computation
    c_cpu = np.zeros((numvectors,VECTOR_SIZE_N,VECTOR_SIZE_L), np.float32)
    t0 = time.clock()
    [np.outer(a_cpu[i], b_cpu[i], out=c_cpu[i]) for i in range(numvectors)]
    cputime = time.clock() - t0
    print 'cpu time', cputime
    
    # create empty gpu array for the result (C = A * B)
    c_gpu = gpuarray.empty((numvectors,VECTOR_SIZE_N,VECTOR_SIZE_L), np.float32)    
    
    t0 = time.clock()
    # transfer host (CPU) memory to device (GPU) memory 
    a_gpu = gpuarray.to_gpu(a_cpu) 
    b_gpu = gpuarray.to_gpu(b_cpu)
    print 'gpu transfer', time.clock()-t0

    t1 = time.clock()
    [gOuter.gOuter(a_gpu[i], b_gpu[i], c_gpu[i]) for i in range(numvectors)]
    print 'gpu calc time', time.clock() - t1
    gpu_gotten = c_gpu.get()
    gputime = time.clock() - t0
        
    print 'gpu time', gputime
    print cputime/gputime
    
    #print the results
#    print "-" * 80
#    print "Matrix A (GPU):"
#    print a_gpu.get()
#    
#    print "-" * 80
#    print "Matrix B (GPU):"
#    print b_gpu.get()
#    
#    print "-" * 80
#    print "Matrix C (GPU):"
#    print c_gpu.get()
    
    print "-" * 80
#    print "CPU-GPU difference:"
#    print c_cpu - c_gpu.get()
    print "L2 norm:", np.average([la.norm(c_cpu[i] - gpu_gotten[i]) for i in range(numvectors)])
    np.allclose(c_cpu, c_gpu.get())