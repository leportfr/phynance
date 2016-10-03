import numpy as np
import time
import gDot
import gOuter
import sys

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

# create uniform random array w/ values in [a,b) and shape args
def rand_arr(*args): 
    a=0.5
    b=-0.5
    np.random.seed(0)
    return np.random.rand(*args) * (b - a) + a
    
def loss_func(pred, label):
    return (pred - label) ** 2

def bottom_diff(pred, label):
    return 2 * (pred - label)
    
class OutParam:
    def __init__(self, out_dim, in_dim):
        self.out_dim = out_dim
        self.in_dim = in_dim
        # weight matrices        
        self.wy = rand_arr(out_dim, in_dim)
        # bias terms
        self.by = rand_arr(out_dim)
        # diffs (derivative of loss function w.r.t. all parameters)
        self.wy_diff = np.zeros((out_dim, in_dim))
        self.by_diff = np.zeros(out_dim)
        
    def apply_diff(self, lr = 1):
        self.wy -= lr * self.wy_diff
        self.by -= lr * self.by_diff  
        self.wy_diff = np.zeros_like(self.wy) 
        self.by_diff = np.zeros_like(self.by)
        
class LstmParam:
    def __init__(self, out_dim, in_dim):
        self.out_dim = out_dim
        self.in_dim = in_dim
        # weight matrices
        self.wg = rand_arr(out_dim, in_dim+out_dim)
        self.wi = rand_arr(out_dim, in_dim+out_dim) 
        self.wf = rand_arr(out_dim, in_dim+out_dim)
        self.wo = rand_arr(out_dim, in_dim+out_dim)
        # bias terms
        self.bg = rand_arr(out_dim) 
        self.bi = rand_arr(out_dim) 
        self.bf = rand_arr(out_dim) 
        self.bo = rand_arr(out_dim) 
        # diffs (derivative of loss function w.r.t. all parameters)
        self.wg_diff = np.zeros((out_dim, in_dim+out_dim)) 
        self.wi_diff = np.zeros((out_dim, in_dim+out_dim)) 
        self.wf_diff = np.zeros((out_dim, in_dim+out_dim)) 
        self.wo_diff = np.zeros((out_dim, in_dim+out_dim))
        self.bg_diff = np.zeros(out_dim) 
        self.bi_diff = np.zeros(out_dim) 
        self.bf_diff = np.zeros(out_dim) 
        self.bo_diff = np.zeros(out_dim)
        # temp_diffs
        self.wg_tdiff = np.zeros((out_dim, in_dim+out_dim)) 
        self.wi_tdiff = np.zeros((out_dim, in_dim+out_dim)) 
        self.wf_tdiff = np.zeros((out_dim, in_dim+out_dim)) 
        self.wo_tdiff = np.zeros((out_dim, in_dim+out_dim))
        self.bg_tdiff = np.zeros(out_dim) 
        self.bi_tdiff = np.zeros(out_dim) 
        self.bf_tdiff = np.zeros(out_dim) 
        self.bo_tdiff = np.zeros(out_dim)

    def apply_diff(self, lr = 1):
        self.wg -= lr * self.wg_diff
        self.wi -= lr * self.wi_diff
        self.wf -= lr * self.wf_diff
        self.wo -= lr * self.wo_diff
        
        self.bg -= lr * self.bg_diff
        self.bi -= lr * self.bi_diff
        self.bf -= lr * self.bf_diff
        self.bo -= lr * self.bo_diff
        # reset diffs to zero
        self.wg_diff = np.zeros_like(self.wg)
        self.wi_diff = np.zeros_like(self.wi) 
        self.wf_diff = np.zeros_like(self.wf) 
        self.wo_diff = np.zeros_like(self.wo)
        self.bg_diff = np.zeros_like(self.bg)
        self.bi_diff = np.zeros_like(self.bi) 
        self.bf_diff = np.zeros_like(self.bf) 
        self.bo_diff = np.zeros_like(self.bo)
        
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
        self.h = np.zeros(out_dim)
        self.bottom_diff_h = np.zeros_like(self.h)
        self.bottom_diff_s = np.zeros_like(self.s)
        self.bottom_diff_x = np.zeros(in_dim)
        
class OutNode:
    def __init__(self, out_param, out_state):
        self.param = out_param
        self.state = out_state
        self.h = None
        
    def bottom_data_is(self, h):
        self.state.y = np.tanh(np.dot(self.param.wy, h) + self.param.by)
#        self.state.y = sigmoid(np.dot(self.param.wy, h) + self.param.by)
        self.h = h
        
    def top_diff_is(self, top_diff_y):
        dy_input = (1. - self.state.y * self.state.y) * top_diff_y
#        dy_input = (1. - self.state.y) * self.state.y * top_diff_y

        self.param.wy_diff += np.outer(dy_input, self.h)
        self.param.by_diff += dy_input
        
        self.state.bottom_diff_h = np.dot(self.param.wy.T, dy_input)     
    
class LstmNode:
    def __init__(self, lstm_param, lstm_state):
        # store reference to parameters and to activations
        self.state = lstm_state
        self.param = lstm_param
        # non-recurrent input to node
        self.inpt = None
        # non-recurrent input concatenated with recurrent input
        self.inptc = None

    def bottom_data_is(self, inpt, s_prev, h_prev):
        # save data for use in backprop
        self.s_prev = s_prev

        # concatenate inpt(t) and h(t-1)
        inptc = np.hstack((inpt,  h_prev))

        self.state.g = np.tanh(np.dot(self.param.wg, inptc) + self.param.bg)
#        self.state.g = sigmoid(np.dot(self.param.wg, xc) + self.param.bg)
        self.state.i = sigmoid(np.dot(self.param.wi, inptc) + self.param.bi)
        self.state.f = sigmoid(np.dot(self.param.wf, inptc) + self.param.bf)
        self.state.o = sigmoid(np.dot(self.param.wo, inptc) + self.param.bo)
        self.state.tanhs = np.tanh(self.state.g * self.state.i + s_prev * self.state.f)
        self.state.h = self.state.tanhs * self.state.o
#        self.state.h = self.state.s * self.state.o

        self.inptc = inptc
    
    def top_diff_is(self, top_diff_h, top_diff_s):
#        t0 = time.clock()
        ds = self.state.o * (1. - self.state.tanhs * self.state.tanhs) * top_diff_h + top_diff_s
#        ds = self.state.o * top_diff_h + top_diff_s

        di_input = (1. - self.state.i) * self.state.i * (self.state.g * ds) 
        df_input = (1. - self.state.f) * self.state.f * (self.s_prev * ds) 
        do_input = (1. - self.state.o) * self.state.o * (self.state.tanhs * top_diff_h) 
#        do_input = (1. - self.state.o) * self.state.o * (self.state.s * top_diff_h)
        dg_input = (1. - self.state.g * self.state.g) * (self.state.i * ds)
#        dg_input = (1. - self.state.g) * self.state.g * dg
#        print 'top_diff_is 0', time.clock() - t0

#        t0 = time.clock()    
        np.outer(di_input, self.inptc, self.param.wi_tdiff)
        np.outer(df_input, self.inptc, self.param.wf_tdiff)
        np.outer(do_input, self.inptc, self.param.wo_tdiff)
        np.outer(dg_input, self.inptc, self.param.wg_tdiff)
#        print 'top_diff_is 1', time.clock() - t0
        
#        t0 = time.clock()
        self.param.wi_diff += self.param.wi_tdiff
        self.param.wf_diff += self.param.wf_tdiff
        self.param.wo_diff += self.param.wo_tdiff
        self.param.wg_diff += self.param.wg_tdiff
        self.param.bi_diff += di_input
        self.param.bf_diff += df_input       
        self.param.bo_diff += do_input
        self.param.bg_diff += dg_input       
#        print 'top_diff_is 2', time.clock() - t0
        
#        t0 = time.clock()    
        dinptc = np.dot(self.param.wi.T, di_input)
        dinptc += np.dot(self.param.wf.T, df_input)
        dinptc += np.dot(self.param.wo.T, do_input)
        dinptc += np.dot(self.param.wg.T, dg_input)
#        print 'top_diff_is 3', time.clock() - t0

#        t0 = time.clock() 
        self.state.bottom_diff_s = ds * self.state.f
        self.state.bottom_diff_x = dinptc[:self.param.in_dim]
        self.state.bottom_diff_h = dinptc[self.param.in_dim:]
#        print 'top_diff_is 4', time.clock() - t0

class LstmNetwork():
    def __init__(self, layer_dims):
        # layer_dims represents dimensions of all layers, where [0] is x_dim and [-1] is y_dim
        self.num_layers = len(layer_dims)
        self.lstm_params = list()
        self.lstm_params.append(LstmParam(layer_dims[1], layer_dims[0]))
        for lyr in range(self.num_layers-3):
            self.lstm_params.append(LstmParam(layer_dims[lyr+2], layer_dims[lyr+1]))
        self.lstm_params.append(OutParam(layer_dims[-1], layer_dims[-2]))
        self.lstm_node_list = []
        self.out_node_list = []
        # input sequence
        self.x_list = []

    def y_list_is(self, y_list):
        """
        Updates diffs by setting target sequence 
        with corresponding loss layer. 
        Will *NOT* update parameters.  To update parameters,
        call self.lstm_param.apply_diff()
        """
        assert len(y_list) == len(self.x_list)
        # here s is not affecting loss due to h(t+1), hence we set equal to zero
        idx = len(self.x_list) - 1
        # calculate loss from out_node and backpropagate
        loss = loss_func(self.out_node_list[idx].state.y, y_list[idx])
        diff_y = bottom_diff(self.out_node_list[idx].state.y, y_list[idx]) 
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

        ### ... following nodes also get diffs from next nodes, hence we add diffs to diff_h
        ### we also propagate error along constant error carousel using diff_s
        while idx >= 0:
            loss += loss_func(self.out_node_list[idx].state.y, y_list[idx])
            diff_y = bottom_diff(self.out_node_list[idx].state.y, y_list[idx])  
            self.out_node_list[idx].top_diff_is(diff_y)
            diff_h = self.out_node_list[idx].state.bottom_diff_h
            diff_h += self.lstm_node_list[idx + 1][-1].state.bottom_diff_h
            diff_s = self.lstm_node_list[idx + 1][-1].state.bottom_diff_s
            self.lstm_node_list[idx][-1].top_diff_is(diff_h, diff_s)
            for lyr in range(self.num_layers-3):
                diff_h = self.lstm_node_list[idx][-lyr-1].state.bottom_diff_x
                diff_h += self.lstm_node_list[idx + 1][-lyr-2].state.bottom_diff_h
                diff_s = self.lstm_node_list[idx + 1][-lyr-2].state.bottom_diff_s
                self.lstm_node_list[idx][-lyr-2].top_diff_is(diff_h, diff_s)
            idx -= 1 
#            print diff_h2[np.where(diff_h2>1.)], diff_s2[np.where(diff_s2>1.)], diff_h[np.where(diff_h>1.)], diff_s[np.where(diff_s>1.)]

        return loss

    def x_list_clear(self):
        self.x_list = []

    def x_list_add(self, x):
        self.x_list.append(x)
        lstm_states=list()
        if len(self.x_list) > len(self.lstm_node_list):
            # need to add new lstm node, create new state mem
            for lyr in range(self.num_layers-2):
                lstm_states.append(LstmState(self.lstm_params[lyr].out_dim, self.lstm_params[lyr].in_dim))
            lstm_states.append(OutState(self.lstm_params[-1].out_dim, self.lstm_params[-1].in_dim))
            lstm_nodes=list()
            for lyr in range(self.num_layers-2):
                lstm_nodes.append(LstmNode(self.lstm_params[lyr], lstm_states[lyr]))
            self.lstm_node_list.append(lstm_nodes)
            self.out_node_list.append(OutNode(self.lstm_params[-1], lstm_states[-1]))

        # get index of most recent x input
        idx = len(self.x_list) - 1
        if idx == 0:
            # no recurrent inputs yet
            self.lstm_node_list[idx][0].bottom_data_is(x, np.zeros_like(self.lstm_node_list[idx][0].state.s), np.zeros_like(self.lstm_node_list[idx][0].state.h))
            for lyr in range(self.num_layers-3):
                self.lstm_node_list[idx][lyr+1].bottom_data_is(self.lstm_node_list[idx][lyr].state.h, np.zeros_like(self.lstm_node_list[idx][lyr+1].state.s), np.zeros_like(self.lstm_node_list[idx][lyr+1].state.h))
            self.out_node_list[idx].bottom_data_is(self.lstm_node_list[idx][-1].state.h)
        else:
            s_prevs=list()
            h_prevs=list()
            for lyr in range(self.num_layers-2):
                s_prevs.append(self.lstm_node_list[idx - 1][lyr].state.s)
                h_prevs.append(self.lstm_node_list[idx - 1][lyr].state.h)
#            print x, s_prevs[0], h_prevs[0]
            self.lstm_node_list[idx][0].bottom_data_is(x, s_prevs[0], h_prevs[0])
            for lyr in range(self.num_layers-3):
                self.lstm_node_list[idx][lyr+1].bottom_data_is(self.lstm_node_list[idx][lyr].state.h, s_prevs[lyr+1], h_prevs[lyr+1])
            self.out_node_list[idx].bottom_data_is(self.lstm_node_list[idx][-1].state.h)
            
    def getOutData(self):
        outData = list()
        for outNode in self.out_node_list:
            outData.append(outNode.state.y)
        return np.array(outData)

