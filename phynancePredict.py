import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lstm
import time

def scale(a,b,x):
    return (a*(x.T-b)+0.2).T
    
def rescale(a,b,y):
    return b+(y-0.2)/a

#locate data files
curdir = os.path.dirname(__file__)
datadir = os.path.join(curdir, 'Data/quantquote_daily_sp500_83986/daily')
filelist = os.listdir(datadir)[2:3]

#load stock data
sp500dict = {}
for i,filename in enumerate(filelist):
    stockSymbol = filename[6:-4]
    stockData = pd.read_csv(os.path.join(datadir, filename),index_col=0,header=None,
                            names=('day','time','open','high','low','close','volume'))
    sp500dict[str(stockSymbol)] = stockData
sp500 = pd.Panel(sp500dict)
print sp500
df = sp500.loc[:,:,'close'].dropna()

#build and scale input and output arrays
exd = 50
inputData = np.array(df).T[:,-1000:]
scaleFactorB=np.min(inputData,axis=1)
scaleFactorA=0.6/(np.max(inputData,axis=1)-scaleFactorB)
scaledData = scale(scaleFactorA, scaleFactorB, inputData).T
x_list = scaledData[:500]
y_list = list()
for i in range(len(x_list)):
    y_list.append(scaledData[i+1:i+1+exd,0])
x_list_train = scaledData[500:500+exd]

#set RNN parameters
mem_cell_ct = 100
x_dim = x_list.shape[1]
y_dim = exd
lstm_param = lstm.LstmParam(mem_cell_ct, x_dim, y_dim) 
lstm_net = lstm.LstmNetwork(lstm_param)

#build plots
f,axarr = plt.subplots(2)
plt.ion()
#for j in range(inputData.shape[0]):
#    axarr[0].plot(inputData[j,1:])

loss_list = list()
for cur_iter in range(1000):
    t1 = time.clock()
    print "cur iter: ", cur_iter
    for val in x_list:
        lstm_net.x_list_add(val)
#        print "y_pred[%d] : %f" % (ind, lstm_net.out_node_list[ind].state.y)
    print time.clock() - t1    
    t0 = time.clock()
    loss_list.append(lstm_net.y_list_is(y_list))
    print time.clock() - t0
    print "loss: ", loss_list[-1]
    t2 = time.clock()
    lstm_param.apply_diff(lr=0.5e-03)
    print time.clock() - t2

    if cur_iter%10==0:
        axarr[0].cla()
        axarr[1].cla()
        axarr[0].set_yscale('log',nonposy='clip')
        axarr[1].set_yscale('log',nonposy='clip')
        axarr[1].set_ylim([0.1,100])
        axarr[0].plot(inputData[:,1:].T)
        for val in x_list_train:
            lstm_net.x_list_add(val)
        predList = np.concatenate((rescale(scaleFactorA, scaleFactorB, lstm_net.getOutData()[:,49]),rescale(scaleFactorA, scaleFactorB, lstm_net.out_node_list[-1].state.y)))
        axarr[0].plot(predList)
        axarr[1].plot(np.array(loss_list))
        plt.pause(0.01)
        
    lstm_net.x_list_clear()