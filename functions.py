import numpy as np
from time import clock
import numba
from copy import copy
import math
from math import tanh
np.random.seed(63)

@numba.jit(nopython=True)
def dot_add(a, b, c):
    for i in range(len(a)):
        for j,val in enumerate(a[0]):
            for k,val2 in enumerate(b[0]):
                c[i,j]+=a[i,j]*b[j,k]
            
@numba.jit(nopython=True)
def sigmoid(x):
    if x>100.0:
        return 1.0
    elif x<-100.0:
        return 0.0
    else:
        return 1./(1. + math.exp(-x))
    
@numba.jit(nopython=True)
def sigprime(x):
    return (1-x)*x
    
@numba.jit(nopython=True)
def hstack(a, b, c):
    for j in range(len(a)):
        for i,val in enumerate(a[j]):
            c[j,i] = val
        l=len(a[j])
        for i,val in enumerate(b[j]):
            c[j,i+l] = val

@numba.jit(nopython=True)
def bottom_data_is_func(inptc, dropout_list, out_dim, w, b, h_prev, s_prev, g, i, f, o, s, h):
    hstack(dropout_list,  h_prev, inptc)
    
    dotprod = np.dot(w, inptc.T).T
    for y in range(len(g)):
        for x in range(out_dim):
            g[y,x] = tanh(dotprod[y,x]+b[x])
            i[y,x] = sigmoid(dotprod[y,x+out_dim]+b[x+out_dim])
            f[y,x] = sigmoid(dotprod[y,x+2*out_dim]+b[x+2*out_dim])
            o[y,x] = sigmoid(dotprod[y,x+3*out_dim]+b[x+3*out_dim]) 
            
            s[y,x] = g[y,x] * i[y,x] + s_prev[y,x] * f[y,x]
            h[y,x] = tanh(s[y,x]) * o[y,x]

@numba.jit(nopython=True)
def top_diff_is_func(inpt,out_dim, g, f, i, o, s_prev, tdh, tds, dsf, s):
    for y in range(len(inpt)):
        for x in range(out_dim):
            ds=(o[y,x] * (1. - tanh(s[y,x])**2) * tdh[y,x] + tds[y,x])
            inpt[y,x] = (1. - g[y,x]**2) * (i[y,x] * ds)
            inpt[y,x+out_dim] = sigprime(i[y,x]) * (g[y,x] * ds) 
            inpt[y,x+2*out_dim] = sigprime(f[y,x]) * (s_prev[y,x] * ds) 
            inpt[y,x+3*out_dim] = sigprime(o[y,x]) * (tanh(s[y,x]) * tdh[y,x]) 
            dsf[y,x] = ds * f[y,x]

@numba.jit(nopython=True)
def outer_add(a, b, c, d):
    for y in range(len(a)):
        for i,val in enumerate(a[y]):
            for j,val2 in enumerate(b[y]):
                c[i,j] += val*val2
            d[i] += val

def rand_arr_w(*args): 
    return np.random.normal(loc=0.0, scale=1/np.sqrt(args[-1]+1), size=args)
    
def rand_arr_b(*args): 
    return np.random.normal(loc=0.0, scale=1.0, size=args)
    
def loss_func(pred, label):
    return (pred - label) ** 2
#    return -(label*np.log(pred) + (1-label)*np.log(1-pred))

def bottom_diff(pred, label):
    return 2 * (pred - label)
#    return pred - label    

def movingaverage(values, window):
#    weights = np.repeat(1.0, window)/window
    return np.convolve(values, [1.0/window]*window, 'valid')    

if __name__ == '__main__':
    a1 = np.random.rand(10,100)
    b1 = np.random.rand(10,200)
    c1 = np.zeros((10,100,200))
    d1 = np.zeros((10,100))
    
    a2 = copy(a1)
    b2 = copy(b1)
    c2 = copy(c1)
    d2 = copy(d1)
    
    t0 = clock()
    for i in range(10000):
        outer_add2(a2,b2,c2,d2)
    print clock() - t0
    
    t0 = clock()
    for i in range(10000):
        outer_add(a1,b1,c1,d1)
    print clock() - t0
    
    print np.average(c2-c1)