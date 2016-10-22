import numpy as np
from time import clock
import numba
from copy import copy
import math
from math import tanh
np.random.seed(0)

#@numba.jit(nopython=True)
#def dot_add(a, b, c):
#    for j,val2 in enumerate(b):
#        for i,val in enumerate(a.T[j]):
#            c[i]+=val*val2
            
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
def hstack(a,b,c):
    for i,val in enumerate(a):
        c[i] = val
    l=len(a)
    for i,val in enumerate(b):
        c[i+l] = val

@numba.jit(nopython=True)
def bottom_data_is_func(inptc, dropout_list, out_dim, w, b, h_prev, s_prev, g, i, f, o, s, h):
    hstack(dropout_list,  h_prev, inptc)
    
    dotprod = np.dot(w, inptc)
    for x in range(out_dim):
        g[x] = tanh(dotprod[x]+b[x])
        i[x] = sigmoid(dotprod[x+out_dim]+b[x+out_dim])
        f[x] = sigmoid(dotprod[x+2*out_dim]+b[x+2*out_dim])
        o[x] = sigmoid(dotprod[x+3*out_dim]+b[x+3*out_dim]) 
        
        s[x] = g[x] * i[x] + s_prev[x] * f[x]
        h[x] = tanh(s[x]) * o[x]

@numba.jit(nopython=True)
def top_diff_is_func(inpt,out_dim, g, f, i, o, s_prev, tdh, tds, dsf, s):
    for x in range(out_dim):
        ds=(o[x] * (1. - tanh(s[x])**2) * tdh[x] + tds[x])
        inpt[x] = (1. - g[x]**2) * (i[x] * ds)
        inpt[x+out_dim] = sigprime(i[x]) * (g[x] * ds) 
        inpt[x+2*out_dim] = sigprime(f[x]) * (s_prev[x] * ds) 
        inpt[x+3*out_dim] = sigprime(o[x]) * (tanh(s[x]) * tdh[x]) 
        dsf[x] = ds * f[x]

@numba.jit(nopython=True)
def outer_add(a, b, c, d):
    for i,val in enumerate(a):
        for j,val2 in enumerate(b):
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
#    return 2 * (pred - label)
    return pred - label        

if __name__ == '__main__':
    a = np.random.rand(100,200)
    b = np.random.rand(200)
    c1 = np.zeros(100)
    c2 = np.random.rand(100)
    c3 = copy(c2)
    
    t0 = clock()
    for i in range(100000):
        c2+=np.dot(a,b)
    print clock() - t0
    
    t0 = clock()
    for i in range(100000):
        dot_add(a,b,c3)
    print clock() - t0
    
    print np.average(c3-c2)