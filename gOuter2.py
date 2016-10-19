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
    def __init__(self, bsizeA, bsizeB, numvectors):
        self.kernel_code_template = """
        __global__ void MatrixMulKernel(float *A, float *B, float *C)
        {
          // Block index
          const unsigned int bA = blockIdx.x;
        
          // Thread index
          const unsigned int tx = threadIdx.x;
        
          // Index of the first sub-matrix of A processed by the block
          const unsigned int aBegin = %(BLOCK_SIZE_A)s * bA;
        
          // Index of the first sub-matrix of B processed by the block
          const unsigned int bBegin = %(BLOCK_SIZE_B)s * bA;
        
          // Shared memory for the sub-matrix of A
          __shared__ float As[%(BLOCK_SIZE_A)s * %(NUMVECTORS)s];
          // Shared memory for the sub-matrix of B
          __shared__ float Bs[%(BLOCK_SIZE_B)s * %(NUMVECTORS)s];
        
          // Load the matrices from global memory to shared memory
          // each thread loads one element of each matrix
          for (int i = 0;
              i < %(BLOCK_SIZE_A)s;
              i += 1) 
              {
                  As[tx * %(BLOCK_SIZE_A)s + i] = A[aBegin + i];
              }
           for (int j = 0;
              j < %(BLOCK_SIZE_B)s;
              j += 1) 
              {
                  Bs[tx * %(BLOCK_SIZE_B)s + j] = B[bBegin + j];
              }
          // Synchronize to make sure the matrices are loaded
          //__syncthreads();
        
          // Multiply the two matrices together;
          // each thread computes one element
          // of the block sub-matrix
          for (int i = 0;
              i < %(BLOCK_SIZE_A)s;
              i += 1) 
              {
              for (int j = 0;
              j < %(BLOCK_SIZE_B)s;
              j += 1) 
                  {
                  C[tx * %(BLOCK_SIZE_A)s * %(BLOCK_SIZE_B)s + i * %(BLOCK_SIZE_B)s + j] = As[tx * %(BLOCK_SIZE_A)s + i] * Bs[tx * %(BLOCK_SIZE_B)s + j];
                  }
              }
        
          // Synchronize to make sure that the preceding
          // computation is done before loading two new
          // sub-matrices of A and B in the next iteration
          //__syncthreads();
        }
        """
        
        # define size of blocks and tiles sub-matrix 
        # (we assume that the block size is same as tile size)
        self.BLOCK_SIZE_A = bsizeA
        self.BLOCK_SIZE_B = bsizeB
        self.NUMVECTORS = numvectors
        
    def gCompile(self):
        # get the kernel code from the template 
        # by specifying the constants MATRIX_SIZE and BLOCK_SIZE
        kernel_code = self.kernel_code_template % { 
            'BLOCK_SIZE_A': self.BLOCK_SIZE_A,
            'BLOCK_SIZE_B': self.BLOCK_SIZE_B,
            'NUMVECTORS' : self.NUMVECTORS
            }
        
        # compile the kernel code
        mod = compiler.SourceModule(kernel_code)
        
        # get the kernel function from the compiled module
        self.matrixmul = mod.get_function("MatrixMulKernel")

    def gOuter(self, a_gpu, b_gpu, c_gpu):
        self.matrixmul(a_gpu, b_gpu, c_gpu,
            # grid of multiple blocks
            grid = (self.NUMVECTORS,1),
            # block of multiple threads
            block = (self.NUMVECTORS,1,1), 
        )
    
if __name__ == '__main__':
    # define the (square) matrix size
    VECTOR_SIZE_N = 50
    VECTOR_SIZE_L = 100
    NUMVECTORS = 100
    
    gOuter = gOuter(VECTOR_SIZE_N,VECTOR_SIZE_L,NUMVECTORS)
    gOuter.gCompile()    
    
    # create two random square matrices
#    a_cpu = np.concatenate([np.random.randn(VECTOR_SIZE_N).astype(np.float32) for i in range(NUMVECTORS)])
#    b_cpu = np.concatenate([np.random.randn(VECTOR_SIZE_L).astype(np.float32) for i in range(NUMVECTORS)])
    a_cpu = np.concatenate([np.arange(VECTOR_SIZE_N).astype(np.float32) for i in range(NUMVECTORS)])
    b_cpu = np.concatenate([np.arange(VECTOR_SIZE_L).astype(np.float32) for i in range(NUMVECTORS)])
    
    # compute reference on the CPU to verify GPU computation
    c_cpu = np.zeros((NUMVECTORS*VECTOR_SIZE_N,VECTOR_SIZE_L), np.float32)
    t0 = time.clock()
    [np.outer(a_cpu[i*VECTOR_SIZE_N:(i+1)*VECTOR_SIZE_N], b_cpu[i*VECTOR_SIZE_L:(i+1)*VECTOR_SIZE_L], out=c_cpu[i*VECTOR_SIZE_N:(i+1)*VECTOR_SIZE_N]) for i in range(NUMVECTORS)]
    cputime = time.clock() - t0
    print 'cpu time', cputime
    
    # create empty gpu array for the result (C = A * B)
    c_gpu = gpuarray.empty((NUMVECTORS*VECTOR_SIZE_N,VECTOR_SIZE_L), np.float32)    
    
    t0 = time.clock()
    # transfer host (CPU) memory to device (GPU) memory 
    a_gpu = gpuarray.to_gpu(a_cpu) 
    b_gpu = gpuarray.to_gpu(b_cpu)
    print 'gpu transfer', time.clock()-t0

    t1 = time.clock()
    gOuter.gOuter(a_gpu, b_gpu, c_gpu)
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
#    print c_cpu
#    print gpu_gotten
    print "L2 norm:", np.average([la.norm(c_cpu[i*VECTOR_SIZE_N:(i+1)*VECTOR_SIZE_N] - gpu_gotten[i*VECTOR_SIZE_N:(i+1)*VECTOR_SIZE_N]) for i in range(NUMVECTORS)])
    np.allclose(c_cpu, c_gpu.get())