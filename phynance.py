import numpy as np
import time
import strategy
import cPickle as pickle
import sys
import pyqtgraph as pg 
from pyqtgraph.Qt import QtGui, QtCore

from dataload import loadData
import lstm

scalerangeX=10
def scaleX(a,b,x):
    return (a*(x.T-b)-scalerangeX/2.0).T
def rescaleX(a,b,y):
    return b+(y+scalerangeX/2.0)/a

scalerangeY=1.6 
def scaleY(a,b,x):
    return (a*(x.T-b)-scalerangeY/2.0).T 
def rescaleY(a,b,y):
    return b+(y+scalerangeY/2.0)/a
    
def movingaverage(values, window):
#    weights = np.repeat(1.0, window)/window
    return np.convolve(values, [1.0/window]*window, 'valid')

df = loadData()
datasize = df.shape[0]

### set RNN parameters ###

init_learn_rate = 3.e-3
learn_factor = 0.1
ema_factor = 0.8
l2_factor = 0.0
dropout_rate = 0.0
mem_cells = [10,10]

##array parameters
history_len = 365
train_len = 100

num_training_sets = 10
mini_batch_size = 3
random_batch = 1

num_test_sets = 100

test_train_cutoff = 2500
test_limit = datasize
sdol = 1.e5

###------ build and scale input and output arrays ------###
iterations = int(1e6)
x_dim = 1#x_list.shape[1]
y_dim = 1
layer_dims = [x_dim]+mem_cells+[y_dim]
lstm_net = lstm.LstmNetwork(layer_dims, init_learn_rate, ema_factor)

np.random.seed(10)
## build and scale input and test arrays
inputData = np.array([np.array(df).T[0,i:i+history_len+train_len+1] for i in np.random.choice(test_train_cutoff,size=num_training_sets,replace=False)])
testData = np.array([np.array(df).T[0,i+test_train_cutoff+history_len+train_len+1:i+test_train_cutoff+2*history_len+2*train_len+2] for i in np.random.choice(test_limit-(test_train_cutoff+2*history_len+2*train_len+2),size=num_test_sets-1,replace=False)])
testData = np.concatenate([[np.array(df).T[0,test_train_cutoff+train_len:test_train_cutoff+history_len+2*train_len+1]],testData])

inputDataDiffQuot = (inputData[:,1:]-inputData[:,:-1])/inputData[:,:-1]
testDataDiffQuot = (testData[:,1:]-testData[:,:-1])/testData[:,:-1]
minval = np.amin([np.amin(np.abs(inputDataDiffQuot[np.nonzero(inputDataDiffQuot)])),np.amin(np.abs(testDataDiffQuot[np.nonzero(testDataDiffQuot)]))])/10.0
inputDataDiff = np.array([np.sign(inputDataDiffQuot)*np.nan_to_num(np.log10(np.abs(inputDataDiffQuot/minval)))])
testDataDiff = np.array([np.sign(testDataDiffQuot)*np.nan_to_num(np.log10(np.abs(testDataDiffQuot/minval)))])
#inputDataDiff = np.concatenate([inputDataDiff,[np.diff(inputData,axis=1)]])
#testDataDiff = np.concatenate([testDataDiff,[np.diff(testData,axis=1)]])
#inputDataDiff = np.concatenate([inputDataDiff,[inputData[:,1:]]])
#testDataDiff = np.concatenate([testDataDiff,[testData[:,1:]]])

scaleFactorB=np.amin(np.amin(np.concatenate([inputDataDiff,testDataDiff],axis=1),axis=1),axis=1)
scaleFactorA=scalerangeX/(np.amax(np.amax(np.concatenate([inputDataDiff,testDataDiff],axis=1),axis=1),axis=1)-scaleFactorB)
#print np.average([np.average(inputDataDiff),np.average(testDataDiff)]), np.amin([np.amin(inputDataDiff),np.amin(testDataDiff)]), np.amax([np.amax(inputDataDiff),np.amax(testDataDiff)])
#sys.exit()

x_list_train = np.transpose(scaleX(scaleFactorA, scaleFactorB, inputDataDiff),(1,2,0))
x_list_test = np.transpose(scaleX(scaleFactorA, scaleFactorB, testDataDiff),(1,2,0))

## build and scale output and test arrays
y_list_full = np.clip(np.array([strategy.buysellVal(inpt[-(history_len+train_len):], sval=sdol) for inpt in inputData]),0.98,1.02)
ideal_return = [strategy.trade_val(inputData[i,-train_len:],y_list_full[i,-train_len:], sdol=sdol)[-1] for i in range(num_training_sets)]

ytest_list_full = np.clip(np.array([strategy.buysellVal(inpt[-(history_len+train_len):], sval=sdol) for inpt in testData]),0.98,1.02)
ideal_test_return = [strategy.trade_val(testData[i,-train_len:],ytest_list_full[i,-train_len:], sdol=sdol)[-1] for i in range(num_test_sets)]

scaleFactorBY=np.amin(np.concatenate([y_list_full,ytest_list_full]))
scaleFactorAY=scalerangeY/(np.amax(np.concatenate([y_list_full,ytest_list_full]))-scaleFactorBY)
y_list_full = scaleY(scaleFactorAY, scaleFactorBY, y_list_full).T
y_list_train = np.reshape(y_list_full[-train_len:].T,[num_training_sets,train_len,1])
rescaled_data = np.reshape(rescaleY(scaleFactorAY, scaleFactorBY, y_list_full).T,[num_training_sets,len(y_list_full),1])

###------ build visualization window and execute training ------###
wintitle='rnlogqotin1scl,valclip.02,xyrg='+str(scalerangeX)+','+str(scalerangeY)+',lf='+str(init_learn_rate)+','+str(learn_factor)+',mem='+str(mem_cells)+','+str(history_len)+'-'+str(train_len)+',ema_factor='+str(ema_factor)+',l2='+str(l2_factor)+',dr='+str(dropout_rate)+',samps='+str(num_training_sets)+',mbsize='+str(mini_batch_size)+'ran'+str(random_batch)
app = QtGui.QApplication([])
win = pg.GraphicsWindow(title=wintitle)
win.resize(1575,825)
pg.setConfigOptions(antialias=True)

plots=list()
curves=list()

plots.append(win.addPlot(title='visualize fit',colspan=len(layer_dims)-1))
curves.append(plots[-1].plot(pen='g')) #curve 0
curves.append(plots[-1].plot(pen='r')) #curve 1
win.nextRow()

plots.append(win.addPlot(title='loss function',colspan=len(layer_dims)-1))
plots[-1].setLogMode(x=False, y=True)
plots[-1].showGrid(x=False, y=True)
curves.append(plots[-1].plot(pen='r')) #curve 2
curves.append(plots[-1].plot(pen='g')) #curve 3
curves.append(plots[-1].plot(pen='y')) #curve 4
win.nextRow()

plots.append(win.addPlot(title='training return',colspan=len(layer_dims)-1))
plots[-1].showGrid(x=False, y=True)
curves.append(plots[-1].plot(pen='r')) #curve 5
curves.append(plots[-1].plot(pen='g')) #curve 6
win.nextRow()

plots.append(win.addPlot(title='test set return',colspan=1))
curves.append(plots[-1].plot(stepMode=True, fillLevel=0, brush=(0,0,255,150))) #curve 7
plots.append(win.addPlot(title='test set loss function',colspan=1))
curves.append(plots[-1].plot(stepMode=True, fillLevel=0, brush=(0,0,255,150))) #curve 8
plots.append(win.addPlot(title='test set return history',colspan=len(layer_dims)-2))
plots[-1].showGrid(x=False, y=True)
curves.append(plots[-1].plot(pen='g')) #curve 9
curves.append(plots[-1].plot(pen='b')) #curve 10
curves.append(plots[-1].plot(pen='r')) #curve 11
curves.append(plots[-1].plot(pen='b')) #curve 12
curves.append(plots[-1].plot(pen='g')) #curve 13
curves.append(plots[-1].plot(pen='y')) #curve 14
win.nextRow()

for i in range(len(layer_dims)-1):
    plots.append(win.addPlot(title='learning rate stats layer W'+str(i+1)))
    plots[-1].showGrid(x=False, y=True)
    plots[-1].setLogMode(x=False, y=True)
    curves.append(plots[-1].plot(pen='g')) #min
    curves.append(plots[-1].plot(pen='y')) #median
    curves.append(plots[-1].plot(pen='r')) #average
    curves.append(plots[-1].plot(pen='b')) #ave + std
    curves.append(plots[-1].plot(pen='g')) #max 
win.nextRow()

for i in range(len(layer_dims)-1):
    plots.append(win.addPlot(title='learning rate stats layer B'+str(i+1)))
    plots[-1].showGrid(x=False, y=True)
    plots[-1].setLogMode(x=False, y=True)
    curves.append(plots[-1].plot(pen='g')) #min
    curves.append(plots[-1].plot(pen='y')) #median
    curves.append(plots[-1].plot(pen='r')) #average
    curves.append(plots[-1].plot(pen='b')) #ave + std
    curves.append(plots[-1].plot(pen='g')) #max 
win.nextRow()

plots.append(win.addPlot(title='visualize test fit',colspan=len(layer_dims)-1))
curves.append(plots[-1].plot(pen='g'))
curves.append(plots[-1].plot(pen='r'))

#outfile = 'C:\Users\leportfr\Desktop\Phynance\outPickle'
#openfile = open(outfile, 'wb')
loss_list = list()
test_loss_list_ma = list()
loss_list_ma = list()
#learnRate = list()
return_list = list()
test_return_plot_list = list()
next_test_return = list()
learn_rate_listW = list()
learn_rate_listB = list()
predTestList = list()
cur_iter=0
t5=0
def iterate():
    global curves, plots, cur_iter, t5, predTestList
    if random_batch:
        train_set = np.random.randint(num_training_sets) 
    else:
        train_set = cur_iter%num_training_sets    
    
    t1 = time.clock()
    print '\ncur iter: ', cur_iter, train_set
    print 'iteration time: ', time.clock() - t5
    for val in x_list_train[train_set]:
        lstm_net.x_list_add(val, dropout_rate)
    predList = rescaleY(scaleFactorAY, scaleFactorBY, lstm_net.getOutData())
    return_list.append((strategy.trade_val(inputData[train_set,-train_len:],predList[-train_len:,0], sdol=sdol)[-1]-sdol)/(ideal_return[train_set]-sdol))
    return_list_ma=[]
    print 'add x_val time: ', time.clock() - t1  
    
    t0 = time.clock()
    loss_list.append(lstm_net.y_list_is(y_list_train[train_set])[0]/scaleFactorAY/scaleFactorAY)
    loss_list_ma=[]
    print 'train time: ', time.clock() - t0
    
    if cur_iter%mini_batch_size == 0:
        t2 = time.clock()
        for lstm_param in lstm_net.lstm_params:
            lstm_param.apply_diff(l2=l2_factor, lr=learn_factor)
        print 'apply time: ', time.clock() - t2
    
    lstm_net.x_list_clear()
    
    t4 = time.clock()
    curves[0].setData(rescaled_data[train_set,:,0])
    curves[1].setData(predList[:,0])
    curves[2].setData(loss_list)
    curves[5].setData(return_list)
    if cur_iter%100 == 0:
        loss_list_ma = movingaverage(loss_list,100)
        return_list_ma = movingaverage(return_list,100)
        curves[3].setData(np.arange(len(loss_list_ma))+100,loss_list_ma)
        curves[6].setData(np.arange(len(return_list_ma))+100,return_list_ma)
    if cur_iter%500 == 0:
        t3 = time.clock()
        test_return_list = []
        test_loss_list = []
        predTestList = []
        for test_set,test_setX in enumerate(x_list_test):
            for val in test_setX:
                lstm_net.x_list_add(val, 0.0)
            predTestList.append(rescaleY(scaleFactorAY, scaleFactorBY, lstm_net.getOutData()))
            test_loss_list.append(np.sum(lstm.loss_func(predTestList[-1][-train_len:,0],ytest_list_full[test_set,-train_len:])))
            if test_set==0:
                next_test_return.append((strategy.trade_val(testData[test_set,-train_len:],predTestList[-1][-train_len:,0], sdol=sdol)[-1]-sdol)/(ideal_test_return[test_set]-sdol))
            else:
                test_return_list.append((strategy.trade_val(testData[test_set,-train_len:],predTestList[-1][-train_len:,0], sdol=sdol)[-1]-sdol)/(ideal_test_return[test_set]-sdol))
            lstm_net.x_list_clear()
        test_loss_list_ma.append(np.average(test_loss_list))
        print 'test time: ', time.clock() - t3
        curves[4].setData(np.arange(len(test_loss_list_ma))*500,test_loss_list_ma)
        
        y_test_hist,x_test_hist = np.histogram(test_return_list, bins=np.linspace(-0.5, 1, 30))
        curves[7].setData(x_test_hist, y_test_hist)  
        y_test_loss_hist,x_test_loss_hist = np.histogram(test_loss_list, bins=np.linspace(0, 1, 100))
        curves[8].setData(x_test_loss_hist, y_test_loss_hist) 
        
        ave=np.average(test_return_list)
        std=np.std(test_return_list)
        test_return_plot_list.append([np.amin(test_return_list),ave-std,ave,ave+std,np.amax(test_return_list)])
        for i in range(5):
            curves[9+i].setData(np.arange(len(test_return_plot_list))*500,np.array(test_return_plot_list)[:,i])
        curves[14].setData(np.arange(len(test_return_plot_list))*500,next_test_return)
    if cur_iter%10 == 0:
        learn_rate_listW.append(lstm_net.getLearnRateStatsW())
        learn_rate_listB.append(lstm_net.getLearnRateStatsB())
        for i in range(len(layer_dims)-1):
            for j in range(5):
                curves[15+5*i+j].setData(np.arange(len(learn_rate_listW))*10,np.array(learn_rate_listW)[:,i,j])
                curves[(15+5*(len(layer_dims)-1))+5*i+j].setData(np.arange(len(learn_rate_listB))*10,np.array(learn_rate_listB)[:,i,j])   
    
    curves[15+10*(len(layer_dims)-1)].setData(ytest_list_full[cur_iter%num_test_sets])
    curves[15+10*(len(layer_dims)-1)+1].setData(predTestList[cur_iter%num_test_sets][:,0])
    print 'display time: ', time.clock() - t4
    
#    pickle.dump(lstm_net.getParams(), openfile)
    print 'totaltime', time.clock() - t1
    print 'loss: ', loss_list[-1]
    cur_iter+=1
    t5 = time.clock()
    
timer = QtCore.QTimer()
timer.timeout.connect(iterate)
timer.start(1)
#openfile.close()

###------ Start Qt event loop unless running in interactive mode or using pyside ------###
if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()