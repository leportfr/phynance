import numpy as np
import time
import strategy
import cPickle as pickle
import sys
import pyqtgraph as pg 
from pyqtgraph.Qt import QtGui, QtCore

from dataload import loadData
import lstm

scalerangeX=1.6
def scaleX(a,b,x):
    return (a*(x.T-b)-scalerangeX/2.0).T
def rescaleX(a,b,y):
    return b+(y+scalerangeX/2.0)/a

scalerangeY=1.6  
def scaleY(a,b,x):
    return (a*(x.T-b)-scalerangeY/2.0).T 
def rescaleY(a,b,y):
    return b+(y+scalerangeY/2.0)/a

df = loadData()
datasize = df.shape[0]

### set RNN parameters ###

learn_factor = 0.0001
ema_factor = 0.1
l2_factor = 0.0
dropout_rate = 0.0
mem_cells = [20,20,20]

##array parameters
history_len = 365
train_len = 100

num_training_sets = 100
mini_batch_size = 10
random_batch = 1

num_test_sets = 100

test_train_cutoff = 1000
test_limit = 2500

###------ build and scale input and output arrays ------###
iterations = int(1e6)
x_dim = 1#x_list.shape[1]
y_dim = x_dim
layer_dims = [x_dim]+mem_cells+[y_dim]
lstm_net = lstm.LstmNetwork(layer_dims, learn_factor, ema_factor)

np.random.seed(10)
## build and scale input arrays
inputData = np.array([np.array(df).T[0,i:i+history_len+train_len+1] for i in np.random.choice(test_train_cutoff,size=num_training_sets,replace=False)])
inputDataDiff = np.diff(inputData,axis=1)
scaleFactorB=np.min(inputDataDiff,axis=1)
scaleFactorA=scalerangeX/(np.max(inputDataDiff,axis=1)-scaleFactorB)
scaledData = scaleX(scaleFactorA, scaleFactorB, inputDataDiff).T
x_list_train = np.reshape(scaledData.T,[num_training_sets,history_len+train_len,1])

## build and scale output arrays
buySellList = zip(*[strategy.ideal_strategy(inpt[-(history_len+train_len):]) for inpt in inputData])[1]
y_list_full = np.zeros([num_training_sets,history_len+train_len])
mult=1
for j,sublist in enumerate(buySellList):
    for i in sublist:
        y_list_full[j,i] = mult
        mult*=-1
ideal_return = [strategy.trade(inputData[i,-train_len:],y_list_full[i,-train_len:])[-1] for i in range(num_training_sets)]

scaleFactorBY=-1.0
scaleFactorAY=scalerangeY/(1.0-scaleFactorBY)
y_list_full = scaleY(scaleFactorAY, scaleFactorBY, y_list_full).T
y_list_train = np.reshape(y_list_full[-train_len:].T,[num_training_sets,train_len,1])
rescaled_data = np.reshape(rescaleY(scaleFactorAY, scaleFactorBY, y_list_full).T,[num_training_sets,len(y_list_full),1])

## build and scale test arrays
testData = np.array([np.array(df).T[0,i+test_train_cutoff+history_len+train_len+1:i+test_train_cutoff+2*history_len+2*train_len+2] for i in np.random.choice(test_limit-(test_train_cutoff+2*history_len+2*train_len+2),size=num_test_sets-1,replace=False)])
testData = np.concatenate([[np.array(df).T[0,test_train_cutoff+train_len:test_train_cutoff+history_len+2*train_len+1]],testData])
testDataDiff = np.diff(testData,axis=1)
scaleFactorBT=np.min(testDataDiff,axis=1)
scaleFactorAT=scalerangeX/(np.max(testDataDiff,axis=1)-scaleFactorBT)
scaledTestData = scaleX(scaleFactorAT, scaleFactorBT, testDataDiff).T
x_list_test = np.reshape(scaledTestData.T,[num_test_sets,history_len+train_len,1])

buySellList = zip(*[strategy.ideal_strategy(inpt[-(history_len+train_len):]) for inpt in testData])[1]
ytest_list_full = np.zeros([num_test_sets,history_len+train_len])
mult=1
for j,sublist in enumerate(buySellList):
    for i in sublist:
        ytest_list_full[j,i] = mult
        mult*=-1
ideal_test_return = [strategy.trade(testData[i,-train_len:],ytest_list_full[i,-train_len:])[-1] for i in range(num_test_sets)]

###------ build visualization window and execute training ------###
wintitle='rndiffin,lf='+str(learn_factor)+',mem='+str(mem_cells)+','+str(history_len)+'-'+str(train_len)+',ema_factor='+str(ema_factor)+',l2='+str(l2_factor)+',dr='+str(dropout_rate)+',samps='+str(num_training_sets)+',mbsize='+str(mini_batch_size)+'ran'+str(random_batch)
app = QtGui.QApplication([])
win = pg.GraphicsWindow(title=wintitle)
win.resize(1575,825)
pg.setConfigOptions(antialias=True)

plots=list()
curves=list()

plots.append(win.addPlot(title='visualize fit',colspan=2))
curves.append(plots[-1].plot(pen='g')) #curve 0
curves.append(plots[-1].plot(pen='r')) #curve 1
win.nextRow()

plots.append(win.addPlot(title='loss function',colspan=2))
curves.append(plots[-1].plot(pen='r')) #curve 2
plots[-1].setLogMode(x=False, y=True)
win.nextRow()

plots.append(win.addPlot(title='training return',colspan=2))
curves.append(plots[-1].plot(pen='r')) #curve 3
win.nextRow()

plots.append(win.addPlot(title='test set return'))
curves.append(plots[-1].plot(stepMode=True, fillLevel=0, brush=(0,0,255,150))) #curve 4
plots.append(win.addPlot(title='test set return history'))
curves.append(plots[-1].plot(pen='g')) #curve 5
curves.append(plots[-1].plot(pen='b')) #curve 6
curves.append(plots[-1].plot(pen='r')) #curve 7
curves.append(plots[-1].plot(pen='b')) #curve 8
curves.append(plots[-1].plot(pen='g')) #curve 9
curves.append(plots[-1].plot(pen='y')) #curve 10

#outfile = 'C:\Users\leportfr\Desktop\Phynance\outPickle'
#openfile = open(outfile, 'wb')
loss_list = list()
#learnRate = list()
return_list = list()
test_return_plot_list = list()
next_test_return = list()
cur_iter=0
def iterate():
    global curves, plots, cur_iter
    if random_batch:
        train_set = np.random.randint(num_training_sets) 
    else:
        train_set = cur_iter%num_training_sets    
    
    t1 = time.clock()
    print '\ncur iter: ', cur_iter, train_set
    for val in x_list_train[train_set]:
        lstm_net.x_list_add(val, dropout_rate)
    predList = rescaleY(scaleFactorAY, scaleFactorBY, lstm_net.getOutData())
    return_list.append((strategy.trade(inputData[train_set,-train_len:],predList[-train_len:,0])[-1]-1.e5)/(ideal_return[train_set]-1.e5))
    print 'add x_val time: ', time.clock() - t1  
    
    t0 = time.clock()
    loss_list.append(lstm_net.y_list_is(y_list_train[train_set])[0]/scaleFactorAY/scaleFactorAY)
    print 'train time: ', time.clock() - t0
    
    if cur_iter%mini_batch_size == 0:
        t2 = time.clock()
        for lstm_param in lstm_net.lstm_params:
            lstm_param.apply_diff(l2=l2_factor)
        print 'apply time: ', time.clock() - t2
    
    lstm_net.x_list_clear()
    
    
    if cur_iter%500==1:
        t3 = time.clock()
        test_return_list = []
        
        for test_set,test_setX in enumerate(x_list_test):
            for val in test_setX:
                lstm_net.x_list_add(val, 0.0)
            predTestList = rescaleY(scaleFactorAY, scaleFactorBY, lstm_net.getOutData())
            if test_set==0:
                next_test_return.append((strategy.trade(testData[test_set,-train_len:],predTestList[-train_len:,0])[-1]-1.e5)/(ideal_test_return[test_set]-1.e5))
            else:
                test_return_list.append((strategy.trade(testData[test_set,-train_len:],predTestList[-train_len:,0])[-1]-1.e5)/(ideal_test_return[test_set]-1.e5))
            lstm_net.x_list_clear()
        
        y_test_hist,x_test_hist = np.histogram(test_return_list, bins=np.linspace(-0.5, 1, 15))
        curves[4].setData(x_test_hist, y_test_hist)  
        
        ave=np.average(test_return_list)
        std=np.std(test_return_list)
        test_return_plot_list.append([np.amin(test_return_list),ave-std,ave,ave+std,np.amax(test_return_list)])
        for i in range(5):
            curves[5+i].setData(np.array(test_return_plot_list)[:,i])
        curves[10].setData(next_test_return)
        print 'test time: ', time.clock() - t3

    curves[0].setData(rescaled_data[train_set,:,0])
    curves[1].setData(predList[:,0])
    curves[2].setData(loss_list)
    curves[3].setData(return_list)
    
#    pickle.dump(lstm_net.getParams(), openfile)
    print 'totaltime', time.clock() - t1
    print "loss: ", loss_list[-1]
    cur_iter+=1
    
timer = QtCore.QTimer()
timer.timeout.connect(iterate)
timer.start(1)
#openfile.close()

###------ Start Qt event loop unless running in interactive mode or using pyside ------###
if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()