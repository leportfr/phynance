import numpy as np
from time import clock
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
    return x
#    return (a*(x.T-b)-scalerangeY/2.0).T 
def rescaleY(a,b,y):
    return y
#    return b+(y+scalerangeY/2.0)/a
    
def movingaverage(values, window):
#    weights = np.repeat(1.0, window)/window
    return np.convolve(values, [1.0/window]*window, 'valid')

df = loadData()
datasize = df.shape[0]

### set RNN parameters ###

init_learn_rate = 1.e-3
learn_factor = 0.1
ema_factor = 0.8
l2_factor = 0.0
dropout_rate = 0.0
mem_cells = [50,50,50]

##array parameters
history_len = 365
train_len = 100

num_training_sets = 1000
mini_batch_size = 1000
random_batch = 1

num_test_sets = 100

test_train_cutoff = 2500
test_limit = datasize
sdol = 1.e5

###------ build and scale input and output arrays ------###
iterations = int(1e6)
x_dim = 1#x_list.shape[1]
y_dim = 2
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
y_list_full = np.array([strategy.buysellVal(inpt[-(history_len+train_len):], sval=sdol) for inpt in inputData])
#print np.amax(y_list_full)
#print np.amin(y_list_full[y_list_full.nonzero()])
#print np.average(y_list_full[y_list_full.nonzero()])
#print np.median(y_list_full[y_list_full.nonzero()])
#sys.exit()
ideal_return = [strategy.trade_2val(inputData[i,-train_len:],y_list_full[i,-train_len:], sdol=sdol)[-1] for i in range(num_training_sets)]

ytest_list_full = np.array([strategy.buysellVal(inpt[-(history_len+train_len):], sval=sdol) for inpt in testData])
ideal_test_return = [strategy.trade_2val(testData[i,-train_len:],ytest_list_full[i,-train_len:], sdol=sdol)[-1] for i in range(num_test_sets)]

scaleFactorBY=np.amin(np.concatenate([y_list_full,ytest_list_full]))
scaleFactorAY=scalerangeY/(np.amax(np.concatenate([y_list_full,ytest_list_full]))-scaleFactorBY)
y_list_full = scaleY(scaleFactorAY, scaleFactorBY, y_list_full)
y_list_train = np.reshape(y_list_full[:,-train_len:],[num_training_sets,train_len,2])
rescaled_data = np.reshape(rescaleY(scaleFactorAY, scaleFactorBY, y_list_full),[num_training_sets,history_len+train_len,2])

###------ build visualization window and execute training ------###
wintitle='logqotin2valout,xrg='+str(scalerangeX)+',lf='+str(init_learn_rate)+','+str(learn_factor)+',mem='+str(mem_cells)+','+str(history_len)+'-'+str(train_len)+',ema_factor='+str(ema_factor)+',l2='+str(l2_factor)+',dr='+str(dropout_rate)+',samps='+str(num_training_sets)+',mbsize='+str(mini_batch_size)
app = QtGui.QApplication([])
win = pg.GraphicsWindow(title=wintitle)
win.resize(1575,825)
pg.setConfigOptions(antialias=True)

plots=list()
curves=list()

plots.append(win.addPlot(title='visualize fit Buy',colspan=len(layer_dims)-1))
curves.append(plots[-1].plot(pen='g')) #curve 0
curves.append(plots[-1].plot(pen='r')) #curve 1
plots.append(win.addPlot(title='visualize fit Sell',colspan=len(layer_dims)-1))
curves.append(plots[-1].plot(pen='g')) #curve 2
curves.append(plots[-1].plot(pen='r')) #curve 3
win.nextRow()

plots.append(win.addPlot(title='loss function Buy',colspan=len(layer_dims)-1))
plots[-1].setLogMode(x=False, y=True)
plots[-1].showGrid(x=False, y=True)
curves.append(plots[-1].plot(pen='r')) #curve 4
curves.append(plots[-1].plot(pen='g')) #curve 5
curves.append(plots[-1].plot(pen='y')) #curve 6
plots.append(win.addPlot(title='loss function Sell',colspan=len(layer_dims)-1))
plots[-1].setLogMode(x=False, y=True)
plots[-1].showGrid(x=False, y=True)
curves.append(plots[-1].plot(pen='r')) #curve 7
curves.append(plots[-1].plot(pen='g')) #curve 8
curves.append(plots[-1].plot(pen='y')) #curve 9
win.nextRow()

plots.append(win.addPlot(title='training return',colspan=2*len(layer_dims)-2))
plots[-1].showGrid(x=False, y=True)
curves.append(plots[-1].plot(pen='r')) #curve 10
curves.append(plots[-1].plot(pen='g')) #curve 11
win.nextRow()

plots.append(win.addPlot(title='test set return',colspan=2))
curves.append(plots[-1].plot(stepMode=True, fillLevel=0, brush=(0,0,255,150))) #curve 12
plots.append(win.addPlot(title='test set loss function buy',colspan=1))
curves.append(plots[-1].plot(stepMode=True, fillLevel=0, brush=(0,0,255,150))) #curve 13
plots.append(win.addPlot(title='test set loss function sell',colspan=1))
curves.append(plots[-1].plot(stepMode=True, fillLevel=0, brush=(0,0,255,150))) #curve 14
plots.append(win.addPlot(title='test set return history',colspan=2*len(layer_dims)-4))
plots[-1].showGrid(x=False, y=True)
curves.append(plots[-1].plot(pen='g')) #curve 15
curves.append(plots[-1].plot(pen='b')) #curve 16
curves.append(plots[-1].plot(pen='r')) #curve 17
curves.append(plots[-1].plot(pen='b')) #curve 18
curves.append(plots[-1].plot(pen='g')) #curve 19
curves.append(plots[-1].plot(pen='y')) #curve 20
win.nextRow()

for i in range(len(layer_dims)-1):
    plots.append(win.addPlot(title='learning rate stats layer W'+str(i+1),colspan=2))
    plots[-1].showGrid(x=False, y=True)
    plots[-1].setLogMode(x=False, y=True)
    curves.append(plots[-1].plot(pen='g')) #min
    curves.append(plots[-1].plot(pen='y')) #median
    curves.append(plots[-1].plot(pen='r')) #average
    curves.append(plots[-1].plot(pen='b')) #ave + std
    curves.append(plots[-1].plot(pen='g')) #max 
win.nextRow()

for i in range(len(layer_dims)-1):
    plots.append(win.addPlot(title='learning rate stats layer B'+str(i+1),colspan=2))
    plots[-1].showGrid(x=False, y=True)
    plots[-1].setLogMode(x=False, y=True)
    curves.append(plots[-1].plot(pen='g')) #min
    curves.append(plots[-1].plot(pen='y')) #median
    curves.append(plots[-1].plot(pen='r')) #average
    curves.append(plots[-1].plot(pen='b')) #ave + std
    curves.append(plots[-1].plot(pen='g')) #max 
win.nextRow()

plots.append(win.addPlot(title='visualize test fit buy',colspan=len(layer_dims)-1))
curves.append(plots[-1].plot(pen='g'))
curves.append(plots[-1].plot(pen='r'))
plots.append(win.addPlot(title='visualize test fit sell',colspan=len(layer_dims)-1))
curves.append(plots[-1].plot(pen='g'))
curves.append(plots[-1].plot(pen='r'))

#outfile = 'C:\Users\leportfr\Desktop\Phynance\outPickle'
#openfile = open(outfile, 'wb')
loss_list = list()
test_loss_list_ma_buy = list()
test_loss_list_ma_sell = list()
loss_list_ma_buy = list()
loss_list_ma_sell = list()
#learnRate = list()
return_list = list()
test_return_plot_list = list()
next_test_return = list()
learn_rate_listW = list()
learn_rate_listB = list()
predTestList = list()
random_set = []
cur_iter=0
t5=0
def iterate():
    global curves, plots, cur_iter, t5, predTestList, random_set
    if random_batch:
        if cur_iter%num_training_sets==0:
            random_set = np.random.choice(num_training_sets,size=num_training_sets, replace=False)
        train_set = random_set[cur_iter%num_training_sets] 
    else:
        train_set = cur_iter%num_training_sets    
    
    t1 = clock()
    print '\ncur iter: ', cur_iter, train_set
    print 'iteration time: ', clock() - t5
    for val in x_list_train[train_set]:
        lstm_net.x_list_add(val, dropout_rate)
    predList = rescaleY(scaleFactorAY, scaleFactorBY, lstm_net.getOutData())
    return_list.append((strategy.trade_2val(inputData[train_set,-train_len:],predList[-train_len:], sdol=sdol)[-1]-sdol)/(ideal_return[train_set]-sdol))
    return_list_ma=[]
    print 'add x_val time: ', clock() - t1  
    
    t0 = clock()
    loss_list.append(lstm_net.y_list_is(y_list_train[train_set]))#/scaleFactorAY/scaleFactorAY)
    loss_list_ma_buy=[]
    loss_list_ma_sell=[]
    print 'train time: ', clock() - t0
    
    if cur_iter%mini_batch_size == 0:
        t2 = clock()
        for lstm_param in lstm_net.lstm_params:
            lstm_param.apply_diff(l2=l2_factor, lr=learn_factor)
        print 'apply time: ', clock() - t2
    
    lstm_net.x_list_clear()
    
    t4 = clock()
    curves[0].setData(rescaled_data[train_set,:,0])
    curves[1].setData(predList[:,0])
    curves[2].setData(rescaled_data[train_set,:,1])
    curves[3].setData(predList[:,1])
    curves[4].setData(np.array(loss_list)[:,0])
    curves[7].setData(np.array(loss_list)[:,1])
    curves[10].setData(return_list)
    if cur_iter%100 == 0:
        loss_list_ma_buy = movingaverage(np.array(loss_list)[:,0],100)
        loss_list_ma_sell = movingaverage(np.array(loss_list)[:,1],100)
        return_list_ma = movingaverage(return_list,100)
        curves[5].setData(np.arange(len(loss_list_ma_buy))+100,loss_list_ma_buy)
        curves[8].setData(np.arange(len(loss_list_ma_sell))+100,loss_list_ma_sell)
        curves[11].setData(np.arange(len(return_list_ma))+100,return_list_ma)
    if cur_iter%500 == 0:
        t3 = clock()
        test_return_list = []
        test_loss_list_buy = []
        test_loss_list_sell = []
        predTestList = []
        for test_set,test_setX in enumerate(x_list_test):
            for val in test_setX:
                lstm_net.x_list_add(val, 0.0)
            predTestList.append(rescaleY(scaleFactorAY, scaleFactorBY, lstm_net.getOutData()))
            test_loss_list_buy.append(np.sum(lstm.loss_func(predTestList[-1][-train_len:,0],ytest_list_full[test_set,-train_len:,0])))
            test_loss_list_sell.append(np.sum(lstm.loss_func(predTestList[-1][-train_len:,1],ytest_list_full[test_set,-train_len:,1])))
            if test_set==0:
                next_test_return.append((strategy.trade_2val(testData[test_set,-train_len:],predTestList[-1][-train_len:], sdol=sdol)[-1]-sdol)/(ideal_test_return[test_set]-sdol))
            else:
                test_return_list.append((strategy.trade_2val(testData[test_set,-train_len:],predTestList[-1][-train_len:], sdol=sdol)[-1]-sdol)/(ideal_test_return[test_set]-sdol))
            lstm_net.x_list_clear()
        test_loss_list_ma_buy.append(np.average(test_loss_list_buy))
        test_loss_list_ma_sell.append(np.average(test_loss_list_sell))
        print 'test time: ', clock() - t3
        curves[6].setData(np.arange(len(test_loss_list_ma_buy))*500,test_loss_list_ma_buy)
        curves[9].setData(np.arange(len(test_loss_list_ma_sell))*500,test_loss_list_ma_sell)
        
        y_test_hist,x_test_hist = np.histogram(test_return_list, bins=np.linspace(-0.5, 1, 30))
        curves[12].setData(x_test_hist, y_test_hist)  
        y_test_loss_hist_buy,x_test_loss_hist_buy = np.histogram(test_loss_list_buy, bins=np.linspace(0, .02, 100))
        curves[13].setData(x_test_loss_hist_buy, y_test_loss_hist_buy) 
        y_test_loss_hist_sell,x_test_loss_hist_sell = np.histogram(test_loss_list_sell, bins=np.linspace(0, .02, 100))
        curves[14].setData(x_test_loss_hist_sell, y_test_loss_hist_sell) 
        
        ave=np.average(test_return_list)
        std=np.std(test_return_list)
        test_return_plot_list.append([np.amin(test_return_list),ave-std,ave,ave+std,np.amax(test_return_list)])
        for i in range(5):
            curves[15+i].setData(np.arange(len(test_return_plot_list))*500,np.array(test_return_plot_list)[:,i])
        curves[20].setData(np.arange(len(test_return_plot_list))*500,next_test_return)
    if cur_iter%10 == 0:
        learn_rate_listW.append(lstm_net.getLearnRateStatsW())
        learn_rate_listB.append(lstm_net.getLearnRateStatsB())
        for i in range(len(layer_dims)-1):
            for j in range(5):
                curves[21+5*i+j].setData(np.arange(len(learn_rate_listW))*10,np.array(learn_rate_listW)[:,i,j])
                curves[(21+5*(len(layer_dims)-1))+5*i+j].setData(np.arange(len(learn_rate_listB))*10,np.array(learn_rate_listB)[:,i,j])   
    
    curves[21+10*(len(layer_dims)-1)+0].setData(ytest_list_full[cur_iter%num_test_sets,:,0])
    curves[21+10*(len(layer_dims)-1)+1].setData(predTestList[cur_iter%num_test_sets][:,0])
    curves[21+10*(len(layer_dims)-1)+2].setData(ytest_list_full[cur_iter%num_test_sets,:,1])
    curves[21+10*(len(layer_dims)-1)+3].setData(predTestList[cur_iter%num_test_sets][:,1])
    print 'display time: ', clock() - t4
    
#    pickle.dump(lstm_net.getParams(), openfile)
    print 'totaltime', clock() - t1
    print 'loss: ', loss_list[-1]
    cur_iter+=1
    t5 = clock()
    
timer = QtCore.QTimer()
timer.timeout.connect(iterate)
timer.start(1)
#openfile.close()

###------ Start Qt event loop unless running in interactive mode or using pyside ------###
if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()