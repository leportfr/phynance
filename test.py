import numpy as np

import lstm

def example_0():
    # learns to repeat simple sequence from random inputs
    np.random.seed(0)

    # parameters for input data dimension and lstm cell count 
    mem_cell_ct = 100
    mem_cell_ct2 = 100
    x_dim = 50
    y_dim = 1
    lstm_param = LstmParam(mem_cell_ct, x_dim) 
    lstm_param2 = LstmParam(mem_cell_ct2, mem_cell_ct)
    out_param = OutParam(y_dim, mem_cell_ct2)
    lstm_net = LstmNetwork(out_param, lstm_param2, lstm_param)
    y_list = [0.5,0.7,0.4, 0.5]
    input_val_arr = [np.random.random(x_dim) for _ in y_list]

    for cur_iter in range(1000):
        print "cur iter: ", cur_iter
        for ind in range(len(y_list)):
            lstm_net.x_list_add(input_val_arr[ind])
            print "y_pred[%d] : %f" % (ind, lstm_net.out_node_list[ind].state.y)

        loss = lstm_net.y_list_is(y_list)
        print "loss: ", loss
        lstm_param.apply_diff(lr=0.1)
        lstm_param2.apply_diff(lr=0.1)
        out_param.apply_diff(lr=0.1)
        lstm_net.x_list_clear()

if __name__ == "__main__":
    example_0()

