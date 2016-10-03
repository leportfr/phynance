import numpy as np
import time

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

# create uniform random array w/ values in [a,b) and shape args
def rand_arr(a, b, *args): 
    np.random.seed(0)
    return np.random.rand(*args) * (b - a) + a
    
def loss_func(pred, label):
    return (pred - label) ** 2

def bottom_diff(pred, label):
    return 2 * (pred - label)
    
class OutParam:
    def __init__(self, y_dim, mem_cell_ct2):
        self.y_dim = y_dim
        self.mem_cell_ct2 = mem_cell_ct2
        # weight matrices        
        self.wy = rand_arr(-0.1, 0.1, self.y_dim, mem_cell_ct2)
        # bias terms
        self.by = rand_arr(-0.1, 0.1, self.y_dim)
        # diffs (derivative of loss function w.r.t. all parameters)
        self.wy_diff = np.zeros((self.y_dim, mem_cell_ct2))
        self.by_diff = np.zeros(self.y_dim)
        
    def apply_diff(self, lr = 1):
        self.wy -= lr * self.wy_diff
        self.by -= lr * self.by_diff  
        self.wy_diff = np.zeros_like(self.wy) 
        self.by_diff = np.zeros_like(self.by)
        
class LstmParam:
    def __init__(self, mem_cell_ct, x_dim):
        self.mem_cell_ct = mem_cell_ct
        self.x_dim = x_dim
        # weight matrices
        self.wg = rand_arr(-0.1, 0.1, mem_cell_ct, x_dim+mem_cell_ct)
        self.wi = rand_arr(-0.1, 0.1, mem_cell_ct, x_dim+mem_cell_ct) 
        self.wf = rand_arr(-0.1, 0.1, mem_cell_ct, x_dim+mem_cell_ct)
        self.wo = rand_arr(-0.1, 0.1, mem_cell_ct, x_dim+mem_cell_ct)
        # bias terms
        self.bg = rand_arr(-0.1, 0.1, mem_cell_ct) 
        self.bi = rand_arr(-0.1, 0.1, mem_cell_ct) 
        self.bf = rand_arr(-0.1, 0.1, mem_cell_ct) 
        self.bo = rand_arr(-0.1, 0.1, mem_cell_ct) 
        # diffs (derivative of loss function w.r.t. all parameters)
        self.wg_diff = np.zeros((mem_cell_ct, x_dim+mem_cell_ct)) 
        self.wi_diff = np.zeros((mem_cell_ct, x_dim+mem_cell_ct)) 
        self.wf_diff = np.zeros((mem_cell_ct, x_dim+mem_cell_ct)) 
        self.wo_diff = np.zeros((mem_cell_ct, x_dim+mem_cell_ct))
        self.bg_diff = np.zeros(mem_cell_ct) 
        self.bi_diff = np.zeros(mem_cell_ct) 
        self.bf_diff = np.zeros(mem_cell_ct) 
        self.bo_diff = np.zeros(mem_cell_ct)

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
    def __init__(self, y_dim, mem_cell_ct):
        self.y = np.zeros(y_dim)
        self.bottom_diff_h = np.zeros_like(mem_cell_ct)
        
class LstmState:
    def __init__(self, mem_cell_ct, x_dim):
        self.g = np.zeros(mem_cell_ct)
        self.i = np.zeros(mem_cell_ct)
        self.f = np.zeros(mem_cell_ct)
        self.o = np.zeros(mem_cell_ct)
        self.s = np.zeros(mem_cell_ct)
        self.h = np.zeros(mem_cell_ct)
        self.bottom_diff_h = np.zeros_like(self.h)
        self.bottom_diff_s = np.zeros_like(self.s)
        self.bottom_diff_x = np.zeros(x_dim)
        
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
        self.x = None
        # non-recurrent input concatenated with recurrent input
        self.xc = None

    def bottom_data_is(self, x, s_prev, h_prev):
        # save data for use in backprop
        self.s_prev = s_prev

        # concatenate x(t) and h(t-1)
        xc = np.hstack((x,  h_prev))
        self.state.g = np.tanh(np.dot(self.param.wg, xc) + self.param.bg)
#        self.state.g = sigmoid(np.dot(self.param.wg, xc) + self.param.bg)
        self.state.i = sigmoid(np.dot(self.param.wi, xc) + self.param.bi)
        self.state.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf)
        self.state.o = sigmoid(np.dot(self.param.wo, xc) + self.param.bo)
        
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f
        self.state.h = np.tanh(self.state.s) * self.state.o
#        self.state.h = self.state.s * self.state.o

        self.xc = xc
    
    def top_diff_is(self, top_diff_h, top_diff_s):
        # notice that top_diff_s is carried along the constant error carousel
        ds = self.state.o * (1.0 - np.tanh(self.state.s) ** 2) * top_diff_h + top_diff_s
#        ds = self.state.o * top_diff_h + top_diff_s

        # diffs w.r.t. vector inside sigma / tanh function
        di_input = (1. - self.state.i) * self.state.i * (self.state.g * ds) 
        df_input = (1. - self.state.f) * self.state.f * (self.s_prev * ds) 
        do_input = (1. - self.state.o) * self.state.o * (self.state.s * top_diff_h) 
        dg_input = (1. - self.state.g * self.state.g) * (self.state.i * ds)
#        dg_input = (1. - self.state.g) * self.state.g * dg

        # diffs w.r.t. inputs
        self.param.wi_diff += np.outer(di_input, self.xc)
        self.param.wf_diff += np.outer(df_input, self.xc)
        self.param.wo_diff += np.outer(do_input, self.xc)
        self.param.wg_diff += np.outer(dg_input, self.xc)
        self.param.bi_diff += di_input
        self.param.bf_diff += df_input       
        self.param.bo_diff += do_input
        self.param.bg_diff += dg_input       

        # compute bottom diff
        dxc = np.dot(self.param.wi.T, di_input)
        dxc += np.dot(self.param.wf.T, df_input)
        dxc += np.dot(self.param.wo.T, do_input)
        dxc += np.dot(self.param.wg.T, dg_input)

        # save bottom diffs
        self.state.bottom_diff_s = ds * self.state.f
        self.state.bottom_diff_x = dxc[:self.param.x_dim]
        self.state.bottom_diff_h = dxc[self.param.x_dim:]

class LstmNetwork():
    def __init__(self, out_param, lstm_param2, lstm_param):
        self.lstm_param = lstm_param
        self.lstm_param2 = lstm_param2
        self.out_param = out_param
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
        idx = len(self.x_list) - 1
        # first node only gets diffs from label ...
        loss = loss_func(self.out_node_list[idx].state.y, y_list[idx])
        diff_y = bottom_diff(self.out_node_list[idx].state.y, y_list[idx]) 
#        print diff_y
        self.out_node_list[idx].top_diff_is(diff_y)        
        diff_h2 = self.out_node_list[idx].state.bottom_diff_h
        # here s is not affecting loss due to h(t+1), hence we set equal to zero
        diff_s2 = np.zeros_like(self.lstm_node_list[idx][1].state.s)
        self.lstm_node_list[idx][1].top_diff_is(diff_h2, diff_s2)
        diff_h = self.lstm_node_list[idx][1].state.bottom_diff_x
        # here s is not affecting loss due to h(t+1), hence we set equal to zero
        diff_s = np.zeros_like(self.lstm_node_list[idx][0].state.s)
        self.lstm_node_list[idx][0].top_diff_is(diff_h, diff_s)
        idx -= 1

        ### ... following nodes also get diffs from next nodes, hence we add diffs to diff_h
        ### we also propagate error along constant error carousel using diff_s
        while idx >= 0:
            loss += loss_func(self.out_node_list[idx].state.y, y_list[idx])
            diff_y = bottom_diff(self.out_node_list[idx].state.y, y_list[idx])        
            self.out_node_list[idx].top_diff_is(diff_y)
            diff_h2 = self.out_node_list[idx].state.bottom_diff_h
            diff_h2 += self.lstm_node_list[idx + 1][1].state.bottom_diff_h
            diff_s2 = self.lstm_node_list[idx + 1][1].state.bottom_diff_s
            self.lstm_node_list[idx][1].top_diff_is(diff_h2, diff_s2)
            diff_h = self.lstm_node_list[idx][1].state.bottom_diff_x
            diff_h += self.lstm_node_list[idx + 1][0].state.bottom_diff_h
            diff_s = self.lstm_node_list[idx + 1][0].state.bottom_diff_s
            self.lstm_node_list[idx][0].top_diff_is(diff_h, diff_s)
            idx -= 1 
#            print diff_h2[np.where(diff_h2>1.)], diff_s2[np.where(diff_s2>1.)], diff_h[np.where(diff_h>1.)], diff_s[np.where(diff_s>1.)]

        return loss

    def x_list_clear(self):
        self.x_list = []

    def x_list_add(self, x):
        self.x_list.append(x)
        if len(self.x_list) > len(self.lstm_node_list):
            # need to add new lstm node, create new state mem
            lstm_state = LstmState(self.lstm_param.mem_cell_ct, self.lstm_param.x_dim)
            lstm_state2 = LstmState(self.lstm_param2.mem_cell_ct, self.lstm_param2.x_dim)
            out_state = OutState(self.out_param.y_dim, self.out_param.mem_cell_ct2)
            self.lstm_node_list.append([LstmNode(self.lstm_param, lstm_state),LstmNode(self.lstm_param2, lstm_state2)])
            self.out_node_list.append(OutNode(self.out_param, out_state))

        # get index of most recent x input
        idx = len(self.x_list) - 1
        if idx == 0:
            # no recurrent inputs yet
            self.lstm_node_list[idx][0].bottom_data_is(x, np.zeros_like(self.lstm_node_list[idx][0].state.s), np.zeros_like(self.lstm_node_list[idx][0].state.h))
            self.lstm_node_list[idx][1].bottom_data_is(self.lstm_node_list[idx][0].state.h, np.zeros_like(self.lstm_node_list[idx][1].state.s), np.zeros_like(self.lstm_node_list[idx][1].state.h))
            self.out_node_list[idx].bottom_data_is(self.lstm_node_list[idx][1].state.h)
        else:
            s_prev = self.lstm_node_list[idx - 1][0].state.s
            h_prev = self.lstm_node_list[idx - 1][0].state.h
            s_prev2 = self.lstm_node_list[idx - 1][1].state.s
            h_prev2 = self.lstm_node_list[idx - 1][1].state.h
            self.lstm_node_list[idx][0].bottom_data_is(x, s_prev, h_prev)
            self.lstm_node_list[idx][1].bottom_data_is(self.lstm_node_list[idx][0].state.h, s_prev2, h_prev2)
            self.out_node_list[idx].bottom_data_is(self.lstm_node_list[idx][1].state.h)
            
    def getOutData(self):
        outData = list()
        for outNode in self.out_node_list:
            outData.append(outNode.state.y)
        return np.array(outData)

