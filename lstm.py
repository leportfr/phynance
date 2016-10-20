import numpy as np
import time
import sys
import gOuter
from numba import autojit
from pycuda import gpuarray

np.random.seed(0)

@autojit
def sigmoid(x):
#    if any(x<-100):
#        print 'xvals',x
#        sys.exit()
    return 1. / (1. + np.exp(-x))

# create uniform random array w/ values in [a,b) and shape args
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
    
class OutParam:
    def __init__(self, out_dim, in_dim, learn_rate, ema_rate):
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.learn_rate = learn_rate
        self.ema_rate = ema_rate
        # weight matrices        
        self.wy = rand_arr_w(out_dim, in_dim)
        # bias terms
        self.by = rand_arr_b(out_dim)
        # diffs (derivative of loss function w.r.t. all parameters)
        self.wy_diff = np.zeros((out_dim, in_dim))
        self.by_diff = np.zeros(out_dim)
        # emas of diffs
        self.wy_diff_ema = np.zeros((out_dim, in_dim))
        self.by_diff_ema = np.zeros(out_dim)
        
        self.wy_diff_ema2 = np.zeros((out_dim, in_dim))# + learn_rate
        self.by_diff_ema2 = np.zeros(out_dim)# + learn_rate
        #learning rates
        self.wy_lr = np.zeros((out_dim, in_dim)) + learn_rate
        self.by_lr = np.zeros(out_dim) + learn_rate
        
    def apply_diff(self, l2, lr):
        #update ema2s
        self.wy_diff_ema2 *= self.ema_rate
        self.wy_diff_ema2 += (1. - self.ema_rate) * self.wy_diff * self.wy_diff + 1.e-12
        self.by_diff_ema2 *= self.ema_rate
        self.by_diff_ema2 += (1. - self.ema_rate) * self.by_diff * self.by_diff + 1.e-12
        #update learn rates   
        self.wy_lr *= np.clip(1.0 + lr * self.wy_diff * self.wy_diff_ema / self.wy_diff_ema2, 0.5, 2.0)
        np.clip(self.wy_lr, 1.e-12, 0.1, out=self.wy_lr)
        self.by_lr *= np.clip(1.0 + lr * self.by_diff * self.by_diff_ema / self.by_diff_ema2, 0.5, 2.0)
        np.clip(self.by_lr, 1.e-12, 0.1, out=self.by_lr)
        #update emas
        self.wy_diff_ema *= self.ema_rate
        self.wy_diff_ema += (1. - self.ema_rate) * self.wy_diff
        self.by_diff_ema *= self.ema_rate
        self.by_diff_ema += (1. - self.ema_rate) * self.by_diff
        #update weights      
        self.wy *= (1. - self.wy_lr*l2)
        self.wy -= self.wy_lr * self.wy_diff
        self.by -= self.by_lr * self.by_diff  
        # reset diffs to zero
        self.wy_diff = np.zeros_like(self.wy) 
        self.by_diff = np.zeros_like(self.by)
        
    def getParams(self):
        return (self.wy, self.by)
        
    def getLearnRateStatsW(self):
        array = [np.amin(self.wy_lr),np.average(self.wy_lr),np.amax(self.wy_lr),np.std(self.wy_lr),np.median(self.wy_lr)]
        return [array[0],array[4],array[1],array[1]+array[3],array[2]]

    def getLearnRateStatsB(self):        
        array = [np.amin(self.by_lr),np.average(self.by_lr),np.amax(self.by_lr),np.std(self.by_lr),np.median(self.by_lr)]
        return [array[0],array[4],array[1],array[1]+array[3],array[2]]
        
class LstmParam:
    def __init__(self, out_dim, in_dim, learn_rate, ema_rate):
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.learn_rate = learn_rate
        self.ema_rate = ema_rate
        # weight matrices
        self.w = rand_arr_w(4*out_dim, in_dim+out_dim)
        # bias terms
        bg = rand_arr_b(out_dim) 
        bi = rand_arr_b(out_dim) 
        bf = rand_arr_b(out_dim) + 2.0 
        bo = rand_arr_b(out_dim) 
        self.b = np.concatenate([bg,bi,bf,bo])
        # diffs (derivative of loss function w.r.t. all parameters)
        self.w_diff = np.zeros((4*out_dim, in_dim+out_dim))
        self.b_diff = np.zeros(4*out_dim) 
        # temp_diffs
        self.d_input = np.zeros(4*out_dim)
        self.w_tdiff = np.zeros((4*out_dim, in_dim+out_dim))
        self.b_tdiff = np.zeros(4*out_dim) 
        # emas of diffs
        self.w_diff_ema = np.zeros((4*out_dim, in_dim+out_dim)) 
        self.b_diff_ema = np.zeros(4*out_dim) 
        
        self.w_diff_ema2 = np.zeros((4*out_dim, in_dim+out_dim))# + learn_rate
        self.b_diff_ema2 = np.zeros(4*out_dim)# + learn_rate 
        # learning rates
        self.w_lr = np.zeros((4*out_dim, in_dim+out_dim)) + learn_rate
        self.b_lr = np.zeros(4*out_dim) + learn_rate   

    def apply_diff(self, l2, lr):
        #update ema2s
#        self.w_diff_ema2 *= self.ema_rate
#        self.w_diff_ema2 += (1. - self.ema_rate) * self.w_diff * self.w_diff + 1.e-12
#        self.b_diff_ema2 *= self.ema_rate
#        self.b_diff_ema2 += (1. - self.ema_rate) * self.b_diff * self.b_diff + 1.e-12
        #update learn rates        
#        self.w_lr *= np.clip(1.0 + lr * self.w_diff * self.w_diff_ema / self.w_diff_ema2, 0.5, 2.0)
#        np.clip(self.w_lr, 1.e-12, 0.1, out=self.w_lr)
#        self.b_lr *= np.clip(1.0 + lr * self.b_diff * self.b_diff_ema / self.b_diff_ema2, 0.5, 2.0)
#        np.clip(self.b_lr, 1.e-12, 0.1, out=self.b_lr)
        #update emas
#        self.w_diff_ema *= self.ema_rate
#        self.w_diff_ema += (1. - self.ema_rate) * self.w_diff
#        self.b_diff_ema *= self.ema_rate
#        self.b_diff_ema += (1. - self.ema_rate) * self.b_diff
        #update weights
        self.w *= (1. - lr*l2) 
        self.w -= lr * self.w_diff
        self.b -= lr * self.b_diff
        # reset diffs to zero
        self.w_diff = np.zeros_like(self.w)
        self.b_diff = np.zeros_like(self.b)
        
    def getParams(self):
        return (self.wg, self.wi, self.wf, self.wo, self.bg, self.bi, self.bf, self.bo)
        
    def getLearnRateStatsW(self):
        conc_lrs = self.w_lr
        array = [np.amin(conc_lrs), np.average(conc_lrs), np.amax(conc_lrs), np.std(conc_lrs), np.median(conc_lrs)]
        return [array[0],array[4],array[1],array[1]+array[3],array[2]]
                
    def getLearnRateStatsB(self):
        conc_lrs = self.b_lr
        array = [np.amin(conc_lrs), np.average(conc_lrs), np.amax(conc_lrs), np.std(conc_lrs), np.median(conc_lrs)]
        return [array[0],array[4],array[1],array[1]+array[3],array[2]]
        
class OutState:
    def __init__(self, out_dim, in_dim):
        self.y = np.zeros(out_dim)
        self.bottom_diff_h = np.zeros_like(in_dim)
        
class LstmState:
    def __init__(self, out_dim, in_dim):
        self.g = np.zeros(out_dim)
        self.i = np.zeros(out_dim)
        self.f = np.zeros(out_dim)
        self.o = np.zeros(out_dim)
        self.s = np.zeros(out_dim)
        self.tanhs = np.zeros(out_dim)
        self.h = np.zeros(out_dim)
        self.bottom_diff_h = np.zeros_like(self.h)
        self.bottom_diff_s = np.zeros_like(self.s)
        self.bottom_diff_x = np.zeros(in_dim)
        
class OutNode:
    def __init__(self, out_param, out_state):
        self.param = out_param
        self.state = out_state
        self.h = None
        
    def bottom_data_is(self, h, dropout_rate):
        dropout_list=[0]*int(len(h)*dropout_rate)
        dropout_list+=[1]*(len(h)-len(dropout_list))
        np.random.shuffle(dropout_list)
        self.h = h*dropout_list
        
        self.state.y = np.tanh(np.dot(self.param.wy, self.h) + self.param.by)
#        self.state.y = sigmoid(np.dot(self.param.wy, h) + self.param.by)
        
    def top_diff_is(self, top_diff_y):
#        dy_input = top_diff_y
        dy_input = (1. - self.state.y * self.state.y) * top_diff_y
#        dy_input = (1. - self.state.y) * self.state.y * top_diff_y

        self.param.wy_diff += np.outer(dy_input, self.h)
        self.param.by_diff += dy_input
        
        self.state.bottom_diff_h = np.dot(self.param.wy.T, dy_input)     
    
class LstmNode:
    def __init__(self, lstm_param, lstm_state):#, gOuter):
        # store reference to parameters and to activations
        self.state = lstm_state
        self.param = lstm_param
        # non-recurrent input to node
        self.inpt = None
        # non-recurrent input concatenated with recurrent input
        self.inptc = None
        
#        self.gOuter = gOuter
#        self.b_gpu = None
#        self.c_gpu = None
        
    def bottom_data_is(self, inpt, s_prev, h_prev, dropout_rate):
        # save data for use in backprop
        self.s_prev = s_prev

        # concatenate inpt(t) and h(t-1)
        dropout_list=[0]*int(len(inpt)*dropout_rate)
        dropout_list+=[1]*(len(inpt)-len(dropout_list))
        np.random.shuffle(dropout_list)
        self.inptc = np.hstack((inpt*dropout_list,  h_prev))

        dotprod = np.dot(self.param.w, self.inptc) + self.param.b
        self.state.g = np.tanh(dotprod[:self.param.out_dim])
#        self.state.g = sigmoid(np.dot(self.param.wg, xc) + self.param.bg)
        self.state.i = sigmoid(dotprod[self.param.out_dim:2*self.param.out_dim])
        self.state.f = sigmoid(dotprod[2*self.param.out_dim:3*self.param.out_dim])
        self.state.o = sigmoid(dotprod[3*self.param.out_dim:])       
        
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f
        self.state.tanhs = np.tanh(self.state.s)
        self.state.h = self.state.tanhs * self.state.o
#        self.state.h = self.state.s * self.state.o
        
#        self.b_gpu = gpuarray.to_gpu(self.inptc.astype(np.float32))
#        self.c_gpu = gpuarray.empty((len(self.state.i),len(self.inptc)), np.float32)
    
    def top_diff_is(self, top_diff_h, top_diff_s):
#        t0 = time.clock()
        ds = self.state.o * (1. - self.state.tanhs * self.state.tanhs) * top_diff_h + top_diff_s
#        ds = self.state.o * top_diff_h + top_diff_s
        
        self.param.d_input[:self.param.out_dim] = (1. - self.state.g * self.state.g) * (self.state.i * ds)
        self.param.d_input[self.param.out_dim:2*self.param.out_dim] = (1. - self.state.i) * self.state.i * (self.state.g * ds) 
        self.param.d_input[2*self.param.out_dim:3*self.param.out_dim] = (1. - self.state.f) * self.state.f * (self.s_prev * ds) 
        self.param.d_input[3*self.param.out_dim:] = (1. - self.state.o) * self.state.o * (self.state.tanhs * top_diff_h) 
#        do_input = (1. - self.state.o) * self.state.o * (self.state.s * top_diff_h)
#        dg_input = (1. - self.state.g) * self.state.g * dg
#        print 'top_diff_is 0', time.clock() - t0

#        t0 = time.clock()   
#        t0 = time.clock()
#        a_gpu = gpuarray.to_gpu(di_input.astype(np.float32))  
#        print 'hi',time.clock()-t0
#        t0 = time.clock()
#        self.gOuter.gOuter(a_gpu, self.b_gpu, self.c_gpu) 
#        print 'hi2',time.clock()-t0
#        t0 = time.clock()
#        self.param.wi_tdiff = self.c_gpu.get()
#        print 'hi3',time.clock()-t0
#        
#        a_gpu = gpuarray.to_gpu(df_input.astype(np.float32))        
#        self.gOuter.gOuter(a_gpu, self.b_gpu, self.c_gpu) 
#        self.param.wf_tdiff = self.c_gpu.get()
#        a_gpu = gpuarray.to_gpu(do_input.astype(np.float32))        
#        self.gOuter.gOuter(a_gpu, self.b_gpu, self.c_gpu) 
#        self.param.wo_tdiff = self.c_gpu.get()
#        a_gpu = gpuarray.to_gpu(dg_input.astype(np.float32))        
#        self.gOuter.gOuter(a_gpu, self.b_gpu, self.c_gpu) 
#        self.param.wg_tdiff = self.c_gpu.get()
        
#        t0 = time.clock()
        np.outer(self.param.d_input, self.inptc, self.param.w_tdiff)
#        print 'top_diff_is 1', time.clock() - t0
        
#        t0 = time.clock()
        self.param.w_diff += self.param.w_tdiff
        self.param.b_diff += self.param.d_input      
#        print 'top_diff_is 2', time.clock() - t0
        
#        t0 = time.clock()    
        dinptc = np.dot(self.param.w.T, self.param.d_input)
#        print 'top_diff_is 3', time.clock() - t0

#        t0 = time.clock() 
        self.state.bottom_diff_s = ds * self.state.f
        self.state.bottom_diff_x = dinptc[:self.param.in_dim]
        self.state.bottom_diff_h = dinptc[self.param.in_dim:]
#        print 'top_diff_is 4', time.clock() - t0

class LstmNetwork():
    def __init__(self, layer_dims, learn_rate, ema_rate):
        # layer_dims represents dimensions of all layers, where [0] is x_dim and [-1] is y_dim
        self.num_layers = len(layer_dims)
        self.lstm_params = list()
        self.lstm_params.append(LstmParam(layer_dims[1], layer_dims[0], learn_rate, ema_rate))
        for lyr in range(self.num_layers-3):
            self.lstm_params.append(LstmParam(layer_dims[lyr+2], layer_dims[lyr+1], learn_rate, ema_rate))
        self.lstm_params.append(OutParam(layer_dims[-1], layer_dims[-2], learn_rate, ema_rate))
        self.lstm_node_list = []
        self.out_node_list = []
        # input sequence
        self.x_list = []
        
#        self.gOuter1 = gOuter.gOuter(layer_dims[1]/10,3)
#        self.gOuter2 = gOuter.gOuter(layer_dims[1]/25,layer_dims[1]/25)
#        self.gOuter1.gCompile()
#        self.gOuter2.gCompile()

    def y_list_is(self, y_list):
        """
        Updates diffs by setting target sequence 
        with corresponding loss layer. 
        Will *NOT* update parameters.  To update parameters,
        call self.lstm_param.apply_diff()
        """
        assert len(y_list) <= len(self.x_list)
        # here s is not affecting loss due to h(t+1), hence we set equal to zero
        idy = len(y_list) - 1
        idx = len(self.x_list) - 1
        # calculate loss from out_node and backpropagate
        loss = loss_func(self.out_node_list[idx].state.y, y_list[idy])
        diff_y = bottom_diff(self.out_node_list[idx].state.y, y_list[idy]) 
#        print diff_y
        self.out_node_list[idx].top_diff_is(diff_y)   
        # calculate diff for lstm nodes and backpropagate
        diff_h = self.out_node_list[idx].state.bottom_diff_h
        diff_s = np.zeros_like(self.lstm_node_list[idx][-1].state.s)
        self.lstm_node_list[idx][-1].top_diff_is(diff_h, diff_s)
        for lyr in range(self.num_layers-3):
            diff_h = self.lstm_node_list[idx][-lyr-1].state.bottom_diff_x
            diff_s = np.zeros_like(self.lstm_node_list[idx][-lyr-2].state.s)
            self.lstm_node_list[idx][-lyr-2].top_diff_is(diff_h, diff_s)
        idx -= 1
        idy -= 1

        ### ... following nodes also get diffs from next nodes, hence we add diffs to diff_h
        ### we also propagate error along constant error carousel using diff_s
        while idx >= 0:
            if idy >= 0:
                loss += loss_func(self.out_node_list[idx].state.y, y_list[idy])
                diff_y = bottom_diff(self.out_node_list[idx].state.y, y_list[idy])  
                self.out_node_list[idx].top_diff_is(diff_y)
                diff_h = self.out_node_list[idx].state.bottom_diff_h
            else:
                diff_h = 0.0
            diff_h += self.lstm_node_list[idx + 1][-1].state.bottom_diff_h
            diff_s = self.lstm_node_list[idx + 1][-1].state.bottom_diff_s
            self.lstm_node_list[idx][-1].top_diff_is(diff_h, diff_s)
            for lyr in range(self.num_layers-3):
                diff_h = self.lstm_node_list[idx][-lyr-1].state.bottom_diff_x
                diff_h += self.lstm_node_list[idx + 1][-lyr-2].state.bottom_diff_h
                diff_s = self.lstm_node_list[idx + 1][-lyr-2].state.bottom_diff_s
                self.lstm_node_list[idx][-lyr-2].top_diff_is(diff_h, diff_s)
            idx -= 1 
            idy -= 1
#            print diff_h2[np.where(diff_h2>1.)], diff_s2[np.where(diff_s2>1.)], diff_h[np.where(diff_h>1.)], diff_s[np.where(diff_s>1.)]

        return loss

    def x_list_clear(self):
        self.x_list = []

    def x_list_add(self, x, dropout_rate):
        self.x_list.append(x)
        if len(self.x_list) > len(self.lstm_node_list):
            lstm_states=list()
            # need to add new lstm node, create new state mem
            for lyr in range(self.num_layers-2):
                lstm_states.append(LstmState(self.lstm_params[lyr].out_dim, self.lstm_params[lyr].in_dim))
            lstm_states.append(OutState(self.lstm_params[-1].out_dim, self.lstm_params[-1].in_dim))
            lstm_nodes=list()
            for lyr in range(self.num_layers-2):
                lstm_nodes.append(LstmNode(self.lstm_params[lyr], lstm_states[lyr]))#, self.gOuter1 if lyr==0 else self.gOuter2))
            self.lstm_node_list.append(lstm_nodes)
            self.out_node_list.append(OutNode(self.lstm_params[-1], lstm_states[-1]))

        # get index of most recent x input
        idx = len(self.x_list) - 1
        if idx == 0:
#            print 'idx', idx
            # no recurrent inputs yet
            self.lstm_node_list[idx][0].bottom_data_is(x, np.zeros_like(self.lstm_node_list[idx][0].state.s), np.zeros_like(self.lstm_node_list[idx][0].state.h), 0.0)
            for lyr in range(self.num_layers-3):
                self.lstm_node_list[idx][lyr+1].bottom_data_is(self.lstm_node_list[idx][lyr].state.h, np.zeros_like(self.lstm_node_list[idx][lyr+1].state.s),
                                                               np.zeros_like(self.lstm_node_list[idx][lyr+1].state.h), dropout_rate)
            self.out_node_list[idx].bottom_data_is(self.lstm_node_list[idx][-1].state.h, dropout_rate)
        else:
#            print 'idx', idx
            s_prevs=list()
            h_prevs=list()
            for lyr in range(self.num_layers-2):
                s_prevs.append(self.lstm_node_list[idx - 1][lyr].state.s)
                h_prevs.append(self.lstm_node_list[idx - 1][lyr].state.h)
#            print x, s_prevs[0], h_prevs[0]
            self.lstm_node_list[idx][0].bottom_data_is(x, s_prevs[0], h_prevs[0], 0.0)
            for lyr in range(self.num_layers-3):
                self.lstm_node_list[idx][lyr+1].bottom_data_is(self.lstm_node_list[idx][lyr].state.h, s_prevs[lyr+1], h_prevs[lyr+1], dropout_rate)
            self.out_node_list[idx].bottom_data_is(self.lstm_node_list[idx][-1].state.h, dropout_rate)
            
    def getOutData(self):
        outData = list()
        for outNode in self.out_node_list:
            outData.append(outNode.state.y)
        return np.array(outData)
        
    def getParams(self):
        paramlist=list()
        for lstm_param in self.lstm_params:
            paramlist.append(lstm_param.getParams())
        return paramlist
        
    def getLearnRateStatsW(self):
        return np.array([par.getLearnRateStatsW() for par in self.lstm_params])
    
    def getLearnRateStatsB(self):
        return np.array([par.getLearnRateStatsB() for par in self.lstm_params])
        

