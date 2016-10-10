import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lstm
import time
import strategy
import cPickle as pickle
import sys
from random import randint

scalerange=1.6
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
train_len = 365
test_len = 100

num_training_sets=10
inputData = np.array([np.array(df).T[0,i*300:i*300+train_len+test_len+1] for i in range(num_training_sets)])
inputDataDiff = np.diff(inputData,axis=1)
#inputDataDiff = np.array(df.diff()[1:]).T[:,-2*(train_len+test_len):]
scaleFactorB=np.min(inputDataDiff,axis=1)
scaleFactorA=scalerange/(np.max(inputDataDiff,axis=1)-scaleFactorB)
scaledData = scale(scaleFactorA, scaleFactorB, inputDataDiff).T
#scaledData = inputData.T
x_list = np.reshape(scaledData[:train_len].T,[num_training_sets,train_len,1])
x_list_test = np.reshape(scaledData[train_len:train_len+test_len],[num_training_sets,test_len,1])

(ideal_return_full, buySellList) = zip(*[strategy.ideal_strategy(inpt[-(train_len+test_len):]) for inpt in inputData])
y_list_full = np.zeros([num_training_sets,train_len+test_len])
mult=1
for j,sublist in enumerate(buySellList):
    for i in sublist:
        y_list_full[j,i] = mult
        mult*=-1
ideal_return = [strategy.trade(inputData[i,-test_len:],y_list_full[i,-test_len:])[-1] for i in range(num_training_sets)]

scaleFactorBY=np.min(y_list_full,axis=1)
scaleFactorAY=scalerange/(np.max(y_list_full,axis=1)-scaleFactorBY)
y_list_full = scale(scaleFactorAY, scaleFactorBY, y_list_full).T
y_list_train = np.reshape(y_list_full[:train_len].T,[num_training_sets,train_len,1])
rescaled_data = np.reshape(rescale(scaleFactorAY, scaleFactorBY, y_list_full).T,[num_training_sets,len(y_list_full),1])

#set RNN parameters
learn_factor = 5.e-4
ema_factor = 0.5
l2_factor = 0.02
mem_cells = [100,100,100]
iterations = int(1e6)
x_dim = 1#x_list.shape[1]
y_dim = x_dim
layer_dims = [x_dim]+mem_cells+[y_dim]
lstm_net = lstm.LstmNetwork(layer_dims, learn_factor, ema_factor)

#build plots
f,axarr = plt.subplots(4)
#plt.ion()
axarr[1].set_yscale('log',nonposy='clip')
axarr[3].set_yscale('log',nonposy='clip')
axarr[0].set_xlim([0,train_len+test_len])
axarr[0].set_ylim([-1,1])
axarr[3].plot(inputData[:,-(train_len+test_len):].T)
graphs=list()
graphs.append(axarr[0].plot(np.zeros_like(rescaled_data[0])+1.0,animated=True)[0])
graphs.append(axarr[0].plot(np.zeros_like(rescaled_data[0])+1.0,animated=True)[0])
for i in range(num_training_sets):
    graphs.append(axarr[1].plot(np.zeros_like(rescaled_data[0])+1.0,animated=True)[0])
graphs.append(axarr[2].plot(np.zeros_like(rescaled_data[0])+1.0,animated=True)[0])
plt.show()
plt.draw()
plt.get_current_fig_manager().window.showMaximized()
f.canvas.set_window_title('lf=5.e-4,[100,100,100],500 hist,strategyOutDiffIn,10_train_sets,l2=0.02')
plt.pause(0.01)
backgrounds = [f.canvas.copy_from_bbox(ax.bbox) for ax in axarr]

loss_list = list()
for i in range(num_training_sets):
    loss_list.append(list())
    loss_list[-1].append([100.])
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
    train_set = randint(0,num_training_sets-1)   
    
    t1 = time.clock()
    print '\ncur iter: ', cur_iter, train_set
    for val in x_list[train_set]:
        lstm_net.x_list_add(val)
    for val in x_list_test[train_set]:
        lstm_net.x_list_add(val)
    outdata=lstm_net.getOutData()
    predList = rescale(scaleFactorAY[train_set], scaleFactorBY[train_set], outdata)
    return_list.append((strategy.trade(inputData[train_set,-test_len:],outdata[-test_len:,0])[-1]-1.e5)/(ideal_return[train_set]-1.e5))
#        print "y_pred[%d] : %f" % (ind, lstm_net.out_node_list[ind].state.y)
    print 'add x_val time: ', time.clock() - t1  
    
    t0 = time.clock()
    loss_list[train_set].append(lstm_net.y_list_is(y_list_train[train_set]))
    print 'train time: ', time.clock() - t0
    
    t2 = time.clock()
#    loss_diff=np.ma.average(np.abs(np.diff(loss_list[-lost_list_ave:],axis=0)/loss_list[-1]),weights=maweights)
#    learnRate.append(np.clip(learn_factor/loss_diff,0.,learnRateLim))
    for lstm_param in lstm_net.lstm_params:
        lstm_param.apply_diff(l2=l2_factor)
    print 'apply time: ', time.clock() - t2

    t3 = time.clock()
    axes_update_period=50
    if cur_iter%axes_update_period==0:
        plt.pause(0.001)
        axarr[1].set_xlim([0,cur_iter/num_training_sets+axes_update_period])
        axarr[1].set_ylim([1,10.**(int(np.amax([np.amax(np.log10(loss_list[i])) for i in range(num_training_sets)]))+1)])
        axarr[2].set_xlim([0,cur_iter+axes_update_period])
        axarr[2].set_ylim([np.amin(return_list)-0.1,np.amax([np.amax(return_list),1.0])])

    if cur_iter%1==0:
        f.canvas.restore_region(backgrounds[0])
        graphs[0].set_ydata(rescaled_data[train_set])
        axarr[0].draw_artist(graphs[0])
        graphs[1].set_ydata(predList)
        axarr[0].draw_artist(graphs[1])
        f.canvas.blit(axarr[0].bbox)
     
    if cur_iter%50==0:
        f.canvas.restore_region(backgrounds[1])
        for i in np.arange(2,2+num_training_sets):
            graphs[i].set_data(range(len(loss_list[i-2])),loss_list[i-2])
            axarr[1].draw_artist(graphs[i])
        f.canvas.blit(axarr[1].bbox)
        
        f.canvas.restore_region(backgrounds[2])
        graphs[-1].set_data(range(cur_iter+1),return_list)
        axarr[2].draw_artist(graphs[-1])
        f.canvas.blit(axarr[2].bbox)
    print 'draw time: ', time.clock() - t3
        
    lstm_net.x_list_clear()
#    pickle.dump(lstm_net.getParams(), openfile)
    print 'totaltime', time.clock() - t1
    print "loss: ", [sub[-1] for sub in loss_list]
#openfile.close()
#    print 'learnRate: ', learnRate[-1]