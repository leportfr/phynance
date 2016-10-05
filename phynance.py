import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lstm
import time
import strategy
import cPickle as pickle

def scale(a,b,x):
    return (a*(x.T-b)-0.8).T
    
def rescale(a,b,y):
    return b+(y+0.8)/a

#locate data files
curdir = os.path.dirname(__file__)
datadir = os.path.join(curdir, 'Data/quantquote_daily_sp500_83986/daily')
filelist = os.listdir(datadir)[1:20]

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
train_len = 500
test_len = 100
inputData = np.array(df).T[:,-2*(train_len+test_len):]
#inputDataDiff = np.array(df.diff()[1:]).T[:,-2*(train_len+test_len):]
scaleFactorB=np.min(inputData,axis=1)
scaleFactorA=1.6/(np.max(inputData,axis=1)-scaleFactorB)
scaledData = scale(scaleFactorA, scaleFactorB, inputData).T
#scaledData = inputData.T
x_list = scaledData[:train_len]
x_list_test = scaledData[train_len:train_len+test_len]

#reward_list = x_list[1:]
#x_list = x_list[:-1]

#hzn = 14
#reward_list = list()
#for i in range(len(x_list)+len(x_list_test)):
#    reward_list.append(np.max(inputData.T[i:i+hzn,:],axis=0)-inputData.T[i,:])
#reward_list = np.array(reward_list).T
#scaleFactorAY=0.6/np.max(reward_list,axis=1)
#scaleFactorBY=np.zeros_like(scaleFactorAY)
#y_list_full = scale(scaleFactorY, np.zeros_like(scaleFactorY), np.array(reward_list)).T
#y_list_train = y_list_full[:len(x_list)]

scaledData_frame = pd.DataFrame(inputData.T[::-1])
averaged_data = scaledData_frame.ewm(halflife=5).mean()[::-1]
dv_data = averaged_data.diff()[1:]
y_list_full = np.array(dv_data[:len(x_list)+len(x_list_test)]).T
scaleFactorBY=np.min(y_list_full,axis=1)
scaleFactorAY=1.6/(np.max(y_list_full,axis=1)-scaleFactorBY)
y_list_full = scale(scaleFactorAY, scaleFactorBY, y_list_full).T
y_list_train = y_list_full[:len(x_list)]
rescaled_data=rescale(scaleFactorAY, scaleFactorBY, y_list_full)

#t0 = time.clock()
#print strategy.ewma_strategy(inputData[0],rescale(scaleFactorAY, scaleFactorBY, y_list_full)[:,0])[-1]
ideal_return = strategy.ewma_strategy(inputData[0,train_len:train_len+test_len],rescaled_data[-test_len:,0])[-1]
#print time.clock()-t0

#set RNN parameters
learn_factor = 2.e-4
ema_factor = 0.5
mem_cells = [100,100]
iterations = 100000
x_dim = x_list.shape[1]
y_dim = x_dim
layer_dims = [x_dim]+mem_cells+[y_dim]
lstm_net = lstm.LstmNetwork(layer_dims, learn_factor, ema_factor)

#build plots
f,axarr = plt.subplots(5)
f.canvas.set_window_title('lf=2.e-4,[100,100],500 hist,20stocks')
plt.ion()
axarr[2].set_yscale('log',nonposy='clip')
axarr[2].plot(inputData[:,:(train_len+test_len)].T)

loss_list = list()
#learnRate = list()
return_list = list()
loss_list.append(np.zeros(y_dim))
lost_list_ave=100
learnRateLim=0.25
loss_list*=lost_list_ave
maweights=np.exp(-np.arange(lost_list_ave-1)[::-1]/30.0)
#learnRate.append(0.05)

#outfile = 'C:\Users\leportfr\Desktop\Phynance\outPickle'
#openfile = open(outfile, 'wb')

for cur_iter in range(iterations):
    t1 = time.clock()
    print '\ncur iter: ', cur_iter
    for val in x_list:
        lstm_net.x_list_add(val)
#        print "y_pred[%d] : %f" % (ind, lstm_net.out_node_list[ind].state.y)
    print 'add x_val time: ', time.clock() - t1  
    
    t0 = time.clock()
    loss_list.append(lstm_net.y_list_is(y_list_train))
    print 'train time: ', time.clock() - t0
    
    t2 = time.clock()
#    loss_diff=np.ma.average(np.abs(np.diff(loss_list[-lost_list_ave:],axis=0)/loss_list[-1]),weights=maweights)
#    learnRate.append(np.clip(learn_factor/loss_diff,0.,learnRateLim))
    for lstm_param in lstm_net.lstm_params:
        lstm_param.apply_diff()
    print 'apply time: ', time.clock() - t2
    
    t3 = time.clock()
    for val in x_list_test:
        lstm_net.x_list_add(val)
    outdata=lstm_net.getOutData()
    return_list.append([ideal_return, strategy.ewma_strategy(inputData[0,train_len:train_len+test_len],outdata[-test_len:,0])[-1]])
    print 'return time: ', time.clock() - t3

    if cur_iter%500==0:
        axarr[0].cla()
        axarr[1].cla()
        axarr[3].cla()
        axarr[4].cla()
#        axarr[0].set_yscale('log',nonposy='clip')
        axarr[1].set_yscale('log',nonposy='clip')
        axarr[3].set_yscale('log',nonposy='clip')
#        axarr[1].set_ylim([.1,.3])
        axarr[0].plot(rescaled_data)
        predList = rescale(scaleFactorAY, scaleFactorBY, outdata)
        axarr[0].plot(predList)
        axarr[1].plot(np.array(loss_list)[lost_list_ave:])
#        axarr[3].plot(np.array(learnRate))
        axarr[4].plot(return_list)
        plt.pause(0.01)
        
    lstm_net.x_list_clear()
#    pickle.dump(lstm_net.getParams(), openfile)
    print 'totaltime', time.clock() - t1
    print "loss: ", loss_list[-1]
#openfile.close()
#    print 'learnRate: ', learnRate[-1]