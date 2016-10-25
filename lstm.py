import numpy as np
from time import clock
import sys
from copy import copy
from functions import rand_arr_w, rand_arr_b, loss_func, bottom_diff
from functions import outer_add, top_diff_is_func, bottom_data_is_func
from collections import deque
from scipy.special import expit
#import gOuter
#from pycuda import gpuarray

LRMAX = 0.1
LRMIN = 1.e-12
    
class OutParam:
    def __init__(self, out_dim, in_dim, mb_size, learn_rate, ema_rate, trainMBratio):
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.mb_size = mb_size
        self.learn_rate = learn_rate
        self.ema_rate = ema_rate
        # weight matrices        
        self.w = rand_arr_w(out_dim, in_dim)#/10.0
        # bias terms
        self.b = np.zeros(out_dim)#/10.0
        # diffs (derivative of loss function w.r.t. all parameters)
        self.w_diff = np.zeros((out_dim, in_dim))
        self.b_diff = np.zeros(out_dim)
        # emas of diffs
        self.w_diff_ema = np.zeros((out_dim, in_dim))
        self.b_diff_ema = np.zeros(out_dim)
        
        self.w_diff_ema2 = np.zeros((out_dim, in_dim))# + learn_rate
        self.b_diff_ema2 = np.zeros(out_dim)# + learn_rate
        
        self.w_diff_deque = deque([np.zeros_like(self.w)]*(trainMBratio))
        self.b_diff_deque = deque([np.zeros_like(self.b)]*(trainMBratio))
        #learning rates
        self.w_lr = np.zeros((out_dim, in_dim)) + learn_rate
        self.b_lr = np.zeros(out_dim) + learn_rate
        
    def apply_diff(self, l2, lr):
#        w_diff_sum = self.w_diff#np.sum(self.w_diff, axis=0)
#        b_diff_sum = self.b_diff#np.sum(self.b_diff, axis=0)          
        
#        self.w_diff_deque.appendleft(copy(w_diff_sum))
#        self.b_diff_deque.appendleft(copy(b_diff_sum)) 
#        w_diff_ma = np.average(self.w_diff_deque, axis=0)
#        b_diff_ma = np.average(self.b_diff_deque, axis=0)
#        w_diff_ma =self.w_diff_deque.pop()
#        b_diff_ma =self.b_diff_deque.pop()
        #update ema2s
        self.w_diff_ema2 *= self.ema_rate
        self.w_diff_ema2 += (1. - self.ema_rate) * self.w_diff * self.w_diff + 1.e-12
        self.b_diff_ema2 *= self.ema_rate
        self.b_diff_ema2 += (1. - self.ema_rate) * self.b_diff * self.b_diff + 1.e-12
        #update learn rates   
        self.w_lr *= np.clip(1.0 + lr * self.w_diff * self.w_diff_ema / self.w_diff_ema2, 0.5, 2.0)
#        self.w_lr *=  (w_diff_sum * self.w_diff_deque.pop() > 0).astype(int)*(lr-1/lr)+1/lr
        np.clip(self.w_lr, LRMIN, LRMAX, out=self.w_lr)
        self.b_lr *= np.clip(1.0 + lr * self.b_diff * self.b_diff_ema / self.b_diff_ema2, 0.5, 2.0)
#        self.b_lr *= (b_diff_sum * self.b_diff_deque.pop() > 0).astype(int)*(lr-1/lr)+1/lr
        np.clip(self.b_lr, LRMIN, LRMAX, out=self.b_lr)  
        #update emas
        self.w_diff_ema *= self.ema_rate
        self.w_diff_ema += (1. - self.ema_rate) * self.w_diff
        self.b_diff_ema *= self.ema_rate
        self.b_diff_ema += (1. - self.ema_rate) * self.b_diff
        #update weights      
        self.w *= (1. - self.w_lr*l2)
        self.w -= self.w_lr * self.w_diff
        self.b -= self.b_lr * self.b_diff 
        #reset diffs
        self.w_diff = np.zeros_like(self.w_diff) 
        self.b_diff = np.zeros_like(self.b_diff)
        
    def getParams(self):
        return (self.w, self.b)
    def getWeightStatsW(self):
        arr = np.abs(self.w)
        return [np.amin(arr),np.percentile(arr,10),np.percentile(arr,25),np.percentile(arr,50),np.percentile(arr,75),np.percentile(arr,90),np.amax(arr)]
    def getWeightStatsB(self):
        arr = np.abs(self.b)
        return [np.amin(arr),np.percentile(arr,10),np.percentile(arr,25),np.percentile(arr,50),np.percentile(arr,75),np.percentile(arr,90),np.amax(arr)]
    def getGradStatsW(self):
        arr = np.abs(self.w_diff)
        return [np.amin(arr),np.percentile(arr,10),np.percentile(arr,25),np.percentile(arr,50),np.percentile(arr,75),np.percentile(arr,90),np.amax(arr)]
    def getGradStatsB(self):
        arr = np.abs(self.b_diff)
        return [np.amin(arr),np.percentile(arr,10),np.percentile(arr,25),np.percentile(arr,50),np.percentile(arr,75),np.percentile(arr,90),np.amax(arr)]   
    def getLearnRateStatsW(self):
        arr = self.w_lr
        return [np.amin(arr),np.percentile(arr,10),np.percentile(arr,25),np.percentile(arr,50),np.percentile(arr,75),np.percentile(arr,90),np.amax(arr)]
    def getLearnRateStatsB(self):        
        arr = self.b_lr
        return [np.amin(arr),np.percentile(arr,10),np.percentile(arr,25),np.percentile(arr,50),np.percentile(arr,75),np.percentile(arr,90),np.amax(arr)]

        
class LstmParam:
    def __init__(self, out_dim, in_dim, mb_size, learn_rate, ema_rate, trainMBratio):
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.mb_size = mb_size
        self.learn_rate = learn_rate
        self.ema_rate = ema_rate
        # weight matrices
        self.w = rand_arr_w(4*out_dim, in_dim+out_dim)#*10.0
        # bias terms
        bg = rand_arr_b(out_dim) 
        bi = rand_arr_b(out_dim) 
        bf = rand_arr_b(out_dim) + 2.0 
        bo = rand_arr_b(out_dim) 
        self.b = np.concatenate([bg,bi,bf,bo])#*10.0
        # diffs (derivative of loss function w.r.t. all parameters)
        self.w_diff = np.zeros((4*out_dim, in_dim+out_dim))
        self.b_diff = np.zeros(4*out_dim) 
        # temp_diffs
        self.d_input = np.zeros((mb_size, 4*out_dim))
        # emas of diffs
        self.w_diff_ema = np.zeros((4*out_dim, in_dim+out_dim)) 
        self.b_diff_ema = np.zeros(4*out_dim) 
        
        self.w_diff_ema2 = np.zeros((4*out_dim, in_dim+out_dim))# + learn_rate
        self.b_diff_ema2 = np.zeros(4*out_dim)# + learn_rate 
        
        self.w_diff_deque = deque([np.zeros_like(self.w)]*(trainMBratio))
        self.b_diff_deque = deque([np.zeros_like(self.b)]*(trainMBratio))
        # learning rates
        self.w_lr = np.zeros((4*out_dim, in_dim+out_dim)) + learn_rate * out_dim
        self.b_lr = np.zeros(4*out_dim) + learn_rate * out_dim

    def apply_diff(self, l2, lr):
#        w_diff_sum = self.w_diff#np.sum(self.w_diff, axis=0)
#        b_diff_sum = self.b_diff#np.sum(self.b_diff, axis=0)            
        
#        self.w_diff_deque.appendleft(copy(w_diff_sum))
#        self.b_diff_deque.appendleft(copy(b_diff_sum)) 
#        w_diff_ma = np.average(self.w_diff_deque, axis=0)
#        b_diff_ma = np.average(self.b_diff_deque, axis=0)
#        w_diff_prev = self.w_diff_deque.pop()
#        b_diff_prev = self.b_diff_deque.pop()
        #update ema2s
        self.w_diff_ema2 *= self.ema_rate
        self.w_diff_ema2 += (1. - self.ema_rate) * self.w_diff * self.w_diff + 1.e-12
        self.b_diff_ema2 *= self.ema_rate
        self.b_diff_ema2 += (1. - self.ema_rate) * self.b_diff * self.b_diff + 1.e-12   
        #update learn rates        
        self.w_lr *= np.clip(1.0 + lr * self.w_diff * self.w_diff_ema / self.w_diff_ema2, 0.5, 2.0)
#        self.w_lr *= (w_diff_sum * self.w_diff_deque.pop() > 0).astype(int)*(lr-1/lr)+1/lr
        np.clip(self.w_lr, LRMIN, LRMAX, out=self.w_lr)
        self.b_lr *= np.clip(1.0 + lr * self.b_diff * self.b_diff_ema / self.b_diff_ema2, 0.5, 2.0)
#        self.b_lr *=  (b_diff_sum * self.b_diff_deque.pop() > 0).astype(int)*(lr-1/lr)+1/lr
        np.clip(self.b_lr, LRMIN, LRMAX, out=self.b_lr)
        #update emas
        self.w_diff_ema *= self.ema_rate
        self.w_diff_ema += (1. - self.ema_rate) * self.w_diff
        self.b_diff_ema *= self.ema_rate
        self.b_diff_ema += (1. - self.ema_rate) * self.b_diff
        #update weights
        self.w *= (1. - self.w_lr*l2) 
        self.w -= self.w_lr * self.w_diff
        self.b -= self.b_lr * self.b_diff
        #reset diffs
        self.w_diff = np.zeros_like(self.w_diff)
        self.b_diff = np.zeros_like(self.b_diff)
        
    def getParams(self):
        return (self.w, self.b)
    def getWeightStatsW(self):
        arr = np.abs(self.w)
        return [np.amin(arr),np.percentile(arr,10),np.percentile(arr,25),np.percentile(arr,50),np.percentile(arr,75),np.percentile(arr,90),np.amax(arr)]
    def getWeightStatsB(self):
        arr = np.abs(self.b)
        return [np.amin(arr),np.percentile(arr,10),np.percentile(arr,25),np.percentile(arr,50),np.percentile(arr,75),np.percentile(arr,90),np.amax(arr)]
    def getGradStatsW(self):
        arr = np.abs(self.w_diff)
        return [np.amin(arr),np.percentile(arr,10),np.percentile(arr,25),np.percentile(arr,50),np.percentile(arr,75),np.percentile(arr,90),np.amax(arr)]
    def getGradStatsB(self):
        arr = np.abs(self.b_diff)
        return [np.amin(arr),np.percentile(arr,10),np.percentile(arr,25),np.percentile(arr,50),np.percentile(arr,75),np.percentile(arr,90),np.amax(arr)]
    def getLearnRateStatsW(self):
        arr = self.w_lr
        return [np.amin(arr),np.percentile(arr,10),np.percentile(arr,25),np.percentile(arr,50),np.percentile(arr,75),np.percentile(arr,90),np.amax(arr)]        
    def getLearnRateStatsB(self):
        arr = self.b_lr
        return [np.amin(arr),np.percentile(arr,10),np.percentile(arr,25),np.percentile(arr,50),np.percentile(arr,75),np.percentile(arr,90),np.amax(arr)]
        
class OutState:
    def __init__(self, out_dim, in_dim, mb_size):
        self.y = np.zeros((mb_size, out_dim))
        self.bottom_diff_h = np.zeros((mb_size, in_dim))
        
class LstmState:
    def __init__(self, out_dim, in_dim, mb_size):
        self.g = np.zeros((mb_size, out_dim))
        self.i = np.zeros((mb_size, out_dim))
        self.f = np.zeros((mb_size, out_dim))
        self.o = np.zeros((mb_size, out_dim))
        self.s = np.zeros((mb_size, out_dim))
        self.h = np.zeros((mb_size, out_dim))
        self.bottom_diff_h = np.zeros_like(self.h)
        self.bottom_diff_s = np.zeros_like(self.s)
        self.bottom_diff_x = np.zeros((mb_size, in_dim))
        
class OutNode:
    def __init__(self, out_param, out_state):
        self.param = out_param
        self.state = out_state
        self.h = None
        
    def bottom_data_is(self, h, dropout_rate):
        dropout_list=[0]*int(self.param.in_dim*dropout_rate)
        dropout_list+=[1]*(self.param.in_dim-len(dropout_list))
        np.random.shuffle(dropout_list)
        self.h = h*dropout_list
        
#        self.state.y = np.tanh(np.dot(self.param.w, self.h) + self.param.b)
        self.state.y = expit(np.dot(self.param.w, self.h.T).T + self.param.b)
        
    def top_diff_is(self, top_diff_y):
        dy_input = top_diff_y
#        dy_input = (1. - self.state.y * self.state.y) * top_diff_y
#        dy_input = (1. - self.state.y) * self.state.y * top_diff_y

        outer_add(dy_input, self.h, self.param.w_diff, self.param.b_diff)
        
        self.state.bottom_diff_h = np.dot(self.param.w.T, dy_input.T).T  
    
class LstmNode:
    def __init__(self, lstm_param, lstm_state):#, gOuter):
        # store reference to parameters and to activations
        self.state = lstm_state
        self.param = lstm_param
        # non-recurrent input concatenated with recurrent input
        self.inptc = np.zeros((self.param.mb_size, self.param.in_dim + self.param.out_dim))
        
#        self.gOuter = gOuter
#        self.b_gpu = None
#        self.c_gpu = None
        
    def bottom_data_is(self, inpt, s_prev, h_prev, dropout_rate):
        # save data for use in backprop
        self.s_prev = s_prev       
        
#        t0=clock()
        dropout_list=[0]*int(self.param.in_dim*dropout_rate)
        dropout_list+=[1]*(self.param.in_dim-len(dropout_list))
        np.random.shuffle(dropout_list)
#        print 'bott_data_is 0', clock()-t0
        
#        t0=clock()
        bottom_data_is_func(self.inptc, inpt*dropout_list, self.param.out_dim, 
                            self.param.w, self.param.b, h_prev, s_prev, self.state.g, 
                            self.state.i, self.state.f, self.state.o, self.state.s, 
                            self.state.h)
#        print 'bott_data_is 1', clock()-t0
    
    def top_diff_is(self, top_diff_h, top_diff_s):
        
#        t0 = clock()       
        top_diff_is_func(self.param.d_input, self.param.out_dim, self.state.g, 
                         self.state.f, self.state.i, self.state.o, self.s_prev, 
                         top_diff_h, top_diff_s, self.state.bottom_diff_s, self.state.s)
#        print 'top_diff_is 0', clock() - t0
        
#        t0 = clock()
        outer_add(self.param.d_input, self.inptc, self.param.w_diff, self.param.b_diff)
#        print 'top_diff_is 1', clock() - t0
        
#        t0 = clock()    
        dinptc = np.dot(self.param.w.T, self.param.d_input.T).T
#        print 'top_diff_is 3', clock() - t0

#        t0 = clock() 
        self.state.bottom_diff_x = dinptc[:,:self.param.in_dim]
        self.state.bottom_diff_h = dinptc[:,self.param.in_dim:]
#        print 'top_diff_is 4', clock() - t0

class LstmNetwork():
    def __init__(self, layer_dims, learn_rate, ema_rate, trainMBratio, mb_size):
        # layer_dims represents dimensions of all layers, where [0] is x_dim and [-1] is y_dim
        self.num_layers = len(layer_dims)
        self.lstm_params = list()
        self.lstm_params.append(LstmParam(layer_dims[1], layer_dims[0], mb_size, learn_rate, ema_rate, trainMBratio))
        for lyr in range(self.num_layers-3):
            self.lstm_params.append(LstmParam(layer_dims[lyr+2], layer_dims[lyr+1], mb_size, learn_rate, ema_rate, trainMBratio))
        self.lstm_params.append(OutParam(layer_dims[-1], layer_dims[-2], mb_size, learn_rate, ema_rate, trainMBratio))
        self.lstm_node_list = []
        self.out_node_list = []
        # input sequence
        self.x_list = []
        
    def setParameters(self, w, b):
        for i in range(self.num_layers-1):
            self.lstm_params[i].w = w[i]
            self.lstm_params[i].b = b[i]
        
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
        loss = np.average(loss_func(self.out_node_list[idx].state.y, y_list[idy]),axis=0)
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
                loss += np.average(loss_func(self.out_node_list[idx].state.y, y_list[idy]),axis=0)
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
                lstm_states.append(LstmState(self.lstm_params[lyr].out_dim, self.lstm_params[lyr].in_dim, self.lstm_params[lyr].mb_size))
            lstm_states.append(OutState(self.lstm_params[-1].out_dim, self.lstm_params[-1].in_dim, self.lstm_params[-1].mb_size))
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
        
    def getWeightStatsW(self):
        return np.array([par.getWeightStatsW() for par in self.lstm_params])         
    def getWeightStatsB(self):
        return np.array([par.getWeightStatsB() for par in self.lstm_params])
    def getGradStatsW(self):
        return np.array([par.getGradStatsW() for par in self.lstm_params])         
    def getGradStatsB(self):
        return np.array([par.getGradStatsB() for par in self.lstm_params]) 
    def getLearnRateStatsW(self):
        return np.array([par.getLearnRateStatsW() for par in self.lstm_params])    
    def getLearnRateStatsB(self):
        return np.array([par.getLearnRateStatsB() for par in self.lstm_params])
        

