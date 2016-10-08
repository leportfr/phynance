import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lstm
import time
import strategy
import cPickle as pickle
import sys

scalerange=2.0
def scale(a,b,x):
    return (a*(x.T-b)-scalerange/2.0).T
    
def rescale(a,b,y):
    return b+(y+scalerange/2.0)/a

#locate data files
curdir = os.path.dirname(__file__)
datadir = os.path.join(curdir, 'Data/quantquote_daily_sp500_83986/daily')
filelist = os.listdir(datadir)[1:2]

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
inputDataDiff = np.diff(inputData,axis=1)
#inputDataDiff = np.array(df.diff()[1:]).T[:,-2*(train_len+test_len):]
scaleFactorB=np.min(inputDataDiff,axis=1)
scaleFactorA=scalerange/(np.max(inputDataDiff,axis=1)-scaleFactorB)
scaledData = scale(scaleFactorA, scaleFactorB, inputDataDiff).T
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

(ideal_return_full, buySellList) = strategy.ideal_strategy(inputData[0,1:(train_len+test_len)+1])
y_list_full = np.zeros((train_len+test_len,1))
mult=1
for i in buySellList:
    if i<train_len+test_len:
        y_list_full[i] = mult
#        mult*=-1
ideal_return = strategy.trade_abs(inputData[0,train_len+1:(train_len+test_len)+1],y_list_full[-test_len:,0])[-1]

scaleFactorBY=np.min(y_list_full,axis=0)
scaleFactorAY=scalerange/(np.max(y_list_full,axis=0)-scaleFactorBY)
y_list_full = scale(scaleFactorAY, scaleFactorBY, y_list_full)
y_list_train = y_list_full[:len(x_list)]
rescaled_data=rescale(scaleFactorAY, scaleFactorBY, y_list_full)

#set RNN parameters
learn_factor = 10.e-4
ema_factor = 0.5
mem_cells = [10]
iterations = 100000
x_dim = x_list.shape[1]
y_dim = x_dim
layer_dims = [x_dim]+mem_cells+[y_dim]
lstm_net = lstm.LstmNetwork(layer_dims, learn_factor, ema_factor)

#build plots
f,axarr = plt.subplots(4)
#plt.ion()
axarr[1].set_yscale('log',nonposy='clip')
axarr[3].set_yscale('log',nonposy='clip')
axarr[0].set_xlim([0,train_len+test_len])
axarr[0].set_ylim([0,1])
axarr[0].plot(rescaled_data)
axarr[3].plot(inputData[:,:(train_len+test_len)].T)
graphs=list()
graphs.append(axarr[0].plot(np.zeros_like(rescaled_data)+1.0,animated=True)[0])
graphs.append(axarr[1].plot(np.zeros_like(rescaled_data)+1.0,animated=True)[0])
graphs.append(axarr[2].plot(np.zeros_like(rescaled_data)+1.0,animated=True)[0])
plt.show()
plt.draw()
plt.get_current_fig_manager().window.showMaximized()
f.canvas.set_window_title('lf=10.e-4,[10],500 hist,strategyOutAbsDiffIn,scalerange=2')
plt.pause(0.01)
backgrounds = [f.canvas.copy_from_bbox(ax.bbox) for ax in axarr]

loss_list = list()
#learnRate = list()
return_list = list()
#loss_list.append(np.zeros(y_dim))
#lost_list_ave=100
#learnRateLim=0.25
#loss_list*=lost_list_ave
#maweights=np.exp(-np.arange(lost_list_ave-1)[::-1]/30.0)
#learnRate.append(0.05)

#outfile = 'C:\Users\leportfr\Desktop\Phynance\outPickle'
#openfile = open(outfile, 'wb')

for cur_iter in range(iterations):
    t1 = time.clock()
    print '\ncur iter: ', cur_iter
    for val in x_list:
        lstm_net.x_list_add(val)
    for val in x_list_test:
        lstm_net.x_list_add(val)
    outdata=lstm_net.getOutData()
    predList = rescale(scaleFactorAY, scaleFactorBY, outdata)
    return_list.append(strategy.trade_abs(inputData[0,train_len+1:(train_len+test_len)+1],outdata[-test_len:,0])[-1])
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

    axes_update_period=50
    if cur_iter%axes_update_period==0:
        plt.pause(0.001)
        axarr[1].set_xlim([0,len(loss_list)+axes_update_period])
        axarr[1].set_ylim([1,10.**(int(np.amax(np.log10(loss_list)))+1)])
        axarr[2].set_xlim([0,len(return_list)+axes_update_period])
        axarr[2].set_ylim([np.amin(return_list),np.amax([np.amax(return_list),ideal_return])])

    if cur_iter%5==0:
        t3 = time.clock()
        f.canvas.restore_region(backgrounds[0])
        graphs[0].set_ydata(predList)
        axarr[0].draw_artist(graphs[0])
        f.canvas.blit(axarr[0].bbox)
        f.canvas.restore_region(backgrounds[1])
        graphs[1].set_data(range(cur_iter+1),loss_list)
        axarr[1].draw_artist(graphs[1])
        f.canvas.blit(axarr[1].bbox)
        f.canvas.restore_region(backgrounds[1])
        graphs[2].set_data(range(cur_iter+1),return_list)
        axarr[2].draw_artist(graphs[2])
        f.canvas.blit(axarr[2].bbox)
        print 'draw time: ', time.clock() - t3
        
    lstm_net.x_list_clear()
#    pickle.dump(lstm_net.getParams(), openfile)
    print 'totaltime', time.clock() - t1
    print "loss: ", loss_list[-1]
#openfile.close()
#    print 'learnRate: ', learnRate[-1]