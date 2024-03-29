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

kernel_code_template = """
__global__ void MatrixMulKernel(float *A, float *B, float *C)
{

  const unsigned int wA = %(MATRIX_SIZE_K)s;
  const unsigned int wB = %(MATRIX_SIZE_N)s;  
  
  // Block index
  const unsigned int bx = blockIdx.x;
  const unsigned int by = blockIdx.y;

  // Thread index
  const unsigned int tx = threadIdx.x;
  const unsigned int ty = threadIdx.y;

  // Index of the first sub-matrix of A processed by the block
  const unsigned int aBegin = wA * %(BLOCK_SIZE)s * by;
  // Index of the last sub-matrix of A processed by the block
  const unsigned int aEnd = aBegin + wA - 1;
  // Step size used to iterate through the sub-matrices of A
  const unsigned int aStep = %(BLOCK_SIZE)s;

  // Index of the first sub-matrix of B processed by the block
  const unsigned int bBegin = %(BLOCK_SIZE)s * bx;
  // Step size used to iterate through the sub-matrices of B
  const unsigned int bStep = %(BLOCK_SIZE)s * wB;

  // The element of the block sub-matrix that is computed
  // by the thread
  float Csub = 0;
  // Loop over all the sub-matrices of A and B required to
  // compute the block sub-matrix
  for (int a = aBegin, b = bBegin;
       a <= aEnd;
       a += aStep, b += bStep) 
    {
      // Shared memory for the sub-matrix of A
      __shared__ float As[%(BLOCK_SIZE)s][%(BLOCK_SIZE)s];
      // Shared memory for the sub-matrix of B
      __shared__ float Bs[%(BLOCK_SIZE)s][%(BLOCK_SIZE)s];

      // Load the matrices from global memory to shared memory
      // each thread loads one element of each matrix
      As[ty][tx] = A[a + wA * ty + tx];
      Bs[ty][tx] = B[b + wB * ty + tx];
      // Synchronize to make sure the matrices are loaded
      __syncthreads();

      // Multiply the two matrices together;
      // each thread computes one element
      // of the block sub-matrix
      for (int k = 0; k < %(BLOCK_SIZE)s; ++k)
        Csub += As[ty][k] * Bs[k][tx];

      // Synchronize to make sure that the preceding
      // computation is done before loading two new
      // sub-matrices of A and B in the next iteration
      __syncthreads();
    }

  // Write the block sub-matrix to global memory;
  // each thread writes one element
  const unsigned int c = wB * %(BLOCK_SIZE)s * by + %(BLOCK_SIZE)s * bx;
  C[c + wB * ty + tx] = Csub;
}
"""

# define the (square) matrix size
MATRIX_SIZE_M = 1000
MATRIX_SIZE_K = 100
MATRIX_SIZE_N = 200

# define size of blocks and tiles sub-matrix 
# (we assume that the block size is same as tile size)
BLOCK_SIZE = 10

# get the kernel code from the template 
# by specifying the constants MATRIX_SIZE and BLOCK_SIZE
kernel_code = kernel_code_template % { 
    'MATRIX_SIZE_M': MATRIX_SIZE_M,
    'MATRIX_SIZE_K': MATRIX_SIZE_K,
    'MATRIX_SIZE_N': MATRIX_SIZE_N,
    'BLOCK_SIZE': BLOCK_SIZE
    }

# compile the kernel code
mod = compiler.SourceModule(kernel_code)

# get the kernel function from the compiled module
matrixmul = mod.get_function("MatrixMulKernel")

def gDot(a_gpu, b_gpu, c_gpu):
    matrixmul(
        # inputs
        a_gpu, b_gpu, 
        # output
        c_gpu, 
        # grid of multiple blocks
        grid = (MATRIX_SIZE_N // BLOCK_SIZE, MATRIX_SIZE_M // BLOCK_SIZE),
        # block of multiple threads
        block = (BLOCK_SIZE, BLOCK_SIZE, 1), 
    )
    
if __name__ == '__main__':
    # create two random square matrices
    a_cpu = np.random.randn(MATRIX_SIZE_M, MATRIX_SIZE_K).astype(np.float32)
    b_cpu = np.random.randn(MATRIX_SIZE_K, MATRIX_SIZE_N).astype(np.float32)
#    a_cpu = np.reshape(np.arange(0,MATRIX_SIZE*MATRIX_SIZE),(MATRIX_SIZE,MATRIX_SIZE)).astype(np.float32)
#    b_cpu = np.reshape(np.arange(0,MATRIX_SIZE*MATRIX_SIZE),(MATRIX_SIZE,MATRIX_SIZE)).astype(np.float32)
    
    # compute reference on the CPU to verify GPU computation
    c_cpu = np.zeros((MATRIX_SIZE_M,MATRIX_SIZE_N), np.float32)
    t0 = time.clock()
    np.dot(a_cpu, b_cpu, out=c_cpu)
    cputime = time.clock() - t0
    print cputime
    
    # create empty gpu array for the result (C = A * B)
    c_gpu = gpuarray.empty((MATRIX_SIZE_M, MATRIX_SIZE_N), np.float32)    
    
    # transfer host (CPU) memory to device (GPU) memory 
    t0 = time.clock()
    a_gpu = gpuarray.to_gpu(a_cpu) 
    b_gpu = gpuarray.to_gpu(b_cpu)
    
    # call the kernel on the card
    gDot(a_gpu, b_gpu, c_gpu)
    c_gpu_out = c_gpu.get()
    gputime = time.clock() - t0
    print gputime
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
    print "L2 norm:", la.norm(c_cpu - c_gpu_out)
    np.allclose(c_cpu, c_gpu_out)