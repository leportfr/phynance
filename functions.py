import numpy as np
from time import clock
import numba
from copy import copy
np.random.seed(0)

#@numba.jit(nopython=True)
#def dot_add(a, b, c):
#    for j,val2 in enumerate(b):
#        x=0
#        for i,val in enumerate(a.T[j]):
#            x+=val*val2
#            c[i]=x
@numba.jit(nopython=True)
def sigmoid(x):
    return 1. / (1. + np.exp(-x))
    
@numba.jit(nopython=True)
def tanh(x):
    return (np.exp(2.*x) - 1)/(np.exp(2.*x) + 1)
    
@numba.jit(nopython=True)
def hstack(a,b,c):
    for i,val in enumerate(a):
        c[i] = val
    l=len(a)
    for i,val in enumerate(b):
        c[i+l] = val

@numba.jit(nopython=True)
def bottom_data_is_func(inptc, dropout_list, out_dim, w, b, h_prev, s_prev):
    hstack(dropout_list,  h_prev, inptc)
    
    dotprod = np.dot(w, inptc) + b
    g = tanh(dotprod[:out_dim])
    i = sigmoid(dotprod[out_dim:2*out_dim])
    f = sigmoid(dotprod[2*out_dim:3*out_dim])
    o = sigmoid(dotprod[3*out_dim:])       
    
    s = g * i + s_prev * f
    tanhs = tanh(s)
    h = tanhs * o
    
    return (g, i, f, o, s, tanhs, h)

@numba.jit(nopython=True)
def top_diff_is_func(inpt,out_dim, g, f, i, o, s_prev, tanhs, tdh, tds):
    ds = o * (1. - tanhs * tanhs) * tdh + tds    
    
    inpt[:out_dim] = (1. - g * g) * (i * ds)
    inpt[out_dim:2*out_dim] = (1. - i) * i * (g * ds) 
    inpt[2*out_dim:3*out_dim] = (1. - f) * f * (s_prev * ds) 
    inpt[3*out_dim:] = (1. - o) * o * (tanhs * tdh) 
    
    return ds * f

@numba.jit(nopython=True)
def sigprime(x):
    return (1-x)*x

@numba.jit(nopython=True)
def outer_add(a, b, c):
    for i,val in enumerate(a):
        for j,val2 in enumerate(b):
            c[i,j] += val*val2

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

if __name__ == '__main__':
    a = np.random.rand(100)
    b = np.random.rand(100)
    c2 = np.random.rand(200)
    
    t0=clock()
    for i in range(1000000):
        c1 = np.hstack((a,b))
    print clock()-t0

    t0=clock()
    for i in range(1000000):
        hstack(a,b,c2)
    print clock()-t0
    
    print np.average(c1-c2)
    
    
#    a = np.random.rand(100,200)
#    b = np.random.rand(200)
#    c1 = np.zeros(100)
#    c2 = np.random.rand(100)
#    c3 = copy(c1)
#    
#    t0 = clock()
#    for i in range(100000):
#        np.dot(a,b,out=c1)
#        c2=c1
#    print clock() - t0
#    
#    t0 = clock()
#    for i in range(100000):
#        dot_add(a,b,c3)
#    print clock() - t0
#    
#    print np.average(c3-c2)