import numpy as np
from time import clock
import strategy
import cPickle as pickle
import sys
import pyqtgraph as pg 
from pyqtgraph.Qt import QtGui, QtCore
from copy import copy

from dataload import loadData, loadTrueTestData
import lstm

scalerangeX=2.0
def scaleX(a,b,x):
    return (a*(x.T-b)-scalerangeX/2.0).T
def rescaleX(a,b,y):
    return b+(y+scalerangeX/2.0)/a

scalerangeY=2.0
def scaleY(a,b,x):
    return (a*(x.T-b)-scalerangeY/2.0).T 
def rescaleY(a,b,y):
    return b+(y+scalerangeY/2.0)/a
    
def movingaverage(values, window):
#    weights = np.repeat(1.0, window)/window
    return np.convolve(values, [1.0/window]*window, 'valid')

df = loadData()
datasize = df.shape[1]

##array parameters
history_len = 365
train_len = 100

#fsf = 6
num_training_sets = 2000#(7-fsf)*342
mini_batch_size = 50
random_batch = 1
num_test_sets = 100

assert(num_training_sets%mini_batch_size == 0)
assert(num_test_sets%mini_batch_size == 0)

test_train_cutoff = 2500
test_limit = datasize

### set RNN parameters ###
init_learn_rate = 1.e-3
ema_factor = 0.9#(1.-1./num_training_sets*mini_batch_size)
l2_factor = 0.0
dropout_rate = 0.1
mem_cells = [50,50,50]

sdolinit = 1.0e5
bidaskinit = 0.005
cominit = 9.99

###------ build and scale input and output arrays ------###
iterations = int(1e2)
x_dim = 1#x_list.shape[1]
y_dim = 1
layer_dims = [x_dim]+mem_cells+[y_dim]
lstm_net = lstm.LstmNetwork(layer_dims, init_learn_rate/mini_batch_size, ema_factor**(float(mini_batch_size)/float(num_training_sets)), num_training_sets/mini_batch_size, mini_batch_size)

np.random.seed(85)
## build and scale input and test arrays
dfarrayclose = np.array(df.loc[:,:,'close']).T
dfarrayvol = np.array(df.loc[:,:,'volume']).T
#inputData = np.array([dfarrayclose[int(i/(7-fsf)),(fsf+i%(7-fsf))*(history_len+train_len+1):(fsf+i%(7-fsf)+1)*(history_len+train_len+1)] for i in range(num_training_sets)])
#testData = np.array([dfarrayclose[i,7*(history_len+train_len+1):8*(history_len+train_len+1)] for i in range(num_test_sets)])
inputData = np.array([dfarrayclose[0, i:i+(history_len+train_len+2)] for i in range(num_training_sets)])
testData = np.array([dfarrayclose[0, i+(history_len+train_len+2)+test_train_cutoff:i+2*(history_len+train_len+2)+test_train_cutoff] for i in range(num_test_sets)])

#inputDataVol = np.array([dfarrayvol[0, i:i+(history_len+train_len+1)] for i in range(num_training_sets)])
#testDataVol = np.array([dfarrayvol[0, i+(history_len+train_len+1)+test_train_cutoff:i+2*(history_len+train_len+1)+test_train_cutoff] for i in range(num_test_sets)])
#inputDataVol[inputDataVol == 0] = np.amin(inputDataVol[inputDataVol.nonzero()])
#testDataVol[testDataVol == 0] = np.amin(testDataVol[testDataVol.nonzero()])

inputDataDiffQuot = (inputData[:,1:]-inputData[:,:-1])/inputData[:,:-1]
testDataDiffQuot = (testData[:,1:]-testData[:,:-1])/testData[:,:-1]
minval = np.amin([np.amin(np.abs(inputDataDiffQuot[np.nonzero(inputDataDiffQuot)])),np.amin(np.abs(testDataDiffQuot[np.nonzero(testDataDiffQuot)]))])/10.0
inputDataDiff = np.array([np.sign(inputDataDiffQuot)*np.nan_to_num(np.log10(np.abs(inputDataDiffQuot/minval)))])
testDataDiff = np.array([np.sign(testDataDiffQuot)*np.nan_to_num(np.log10(np.abs(testDataDiffQuot/minval)))])
#inputDataDiff = np.concatenate([inputDataDiff,[np.log(inputDataVol[:,1:])]])
#testDataDiff = np.concatenate([testDataDiff,[np.log(testDataVol[:,1:])]])

#dfTrue = loadTrueTestData()    
#inputTrueData = np.array([np.array(dfTrue).astype(np.float64).T[0,-(history_len+train_len+1+mov1):] for i in range(mini_batch_size)])
#inputTrueDataDiffQuot = (inputTrueData[:,1:]-inputTrueData[:,:-1])/inputTrueData[:,:-1]
#minval = np.amin([np.amin(np.abs(inputTrueDataDiffQuot[np.nonzero(inputTrueDataDiffQuot)]))])/10.0
#inputTrueDataDiff = np.array([np.sign(inputTrueDataDiffQuot)*np.nan_to_num(np.log10(np.abs(inputTrueDataDiffQuot/minval)))])

scaleFactorB=np.amin(np.amin(np.concatenate([inputDataDiff,testDataDiff],axis=1),axis=1),axis=1)
scaleFactorA=scalerangeX/(np.amax(np.amax(np.concatenate([inputDataDiff,testDataDiff],axis=1),axis=1),axis=1)-scaleFactorB)
#print np.average([np.average(inputDataDiff),np.average(testDataDiff)]), np.amin([np.amin(inputDataDiff),np.amin(testDataDiff)]), np.amax([np.amax(inputDataDiff),np.amax(testDataDiff)])
#sys.exit()
x_list_train = np.transpose(scaleX(scaleFactorA, scaleFactorB, inputDataDiff[:,:,:(history_len+train_len)]),(1,2,0))
x_list_test = np.transpose(scaleX(scaleFactorA, scaleFactorB, testDataDiff[:,:,:(history_len+train_len)]),(1,2,0))
#    x_list_true_train = np.transpose(scaleX(scaleFactorA, scaleFactorB, inputTrueDataDiff),(1,2,0))    

y_list_full = np.transpose(inputDataDiff[:,:,-(history_len+train_len):],(1,2,0))
## build and scale output and test arrays
scaleFactorBY=scaleFactorB
scaleFactorAY=scaleFactorA
y_list_full = scaleY(scaleFactorAY, scaleFactorBY, y_list_full)
y_list_train = y_list_full[:,-train_len:]
rescaled_data = rescaleY(scaleFactorAY, scaleFactorBY, y_list_full)

ytest_list_full = np.transpose(testDataDiff[:,:,-(history_len+train_len):],(1,2,0))

###------ build visualization window and execute training ------###
wintitle='rnlogqotin1scl,xyrg='+str(scalerangeX)+','+str(scalerangeY)+',lf='+str(init_learn_rate)+',mem='+str(mem_cells)+','+str(history_len)+'-'+str(train_len)+',ema_factor='+str(ema_factor)+',l2='+str(l2_factor)+',dr='+str(dropout_rate)+',samps='+str(num_training_sets)+',mbsize='+str(mini_batch_size)+'ran'+str(random_batch)
app = QtGui.QApplication([])
win = pg.GraphicsWindow(title=wintitle)
win.resize(1575,825)
pg.setConfigOptions(antialias=True)

plots=list()
curves=list()

plots.append(win.addPlot(title='visualize fit',colspan=2*len(layer_dims)-2))
curves.append(plots[-1].plot(pen='g')) #curve 0
curves.append(plots[-1].plot(pen='r')) #curve 1
win.nextRow()

plots.append(win.addPlot(title='loss function',colspan=len(layer_dims)-1))
plots[-1].setLogMode(x=False, y=True)
plots[-1].showGrid(x=False, y=True)
curves.append(plots[-1].plot(pen='r')) #curve 2
curves.append(plots[-1].plot(pen='g')) #curve 3
curves.append(plots[-1].plot(pen='y')) #curve 4
#win.nextRow()

plots.append(win.addPlot(title='training return',colspan=(2*len(layer_dims)-2-(len(layer_dims)-1))))
plots[-1].showGrid(x=False, y=True)
curves.append(plots[-1].plot(pen='r')) #curve 5
curves.append(plots[-1].plot(pen='g')) #curve 6
win.nextRow()

plots.append(win.addPlot(title='test set return',colspan=1))
curves.append(plots[-1].plot(stepMode=True, fillLevel=0, brush=(0,0,255,150))) #curve 7
plots.append(win.addPlot(title='test set loss function',colspan=1))
curves.append(plots[-1].plot(stepMode=True, fillLevel=0, brush=(0,0,255,150))) #curve 8
plots.append(win.addPlot(title='test set return history',colspan=2*len(layer_dims)-2))
plots[-1].showGrid(x=False, y=True)
curves.append(plots[-1].plot(pen='g')) #curve 9
curves.append(plots[-1].plot(pen='b')) #curve 10
curves.append(plots[-1].plot(pen='r')) #curve 11
curves.append(plots[-1].plot(pen='b')) #curve 12
curves.append(plots[-1].plot(pen='g')) #curve 13
curves.append(plots[-1].plot(pen='y')) #curve 14
win.nextRow()

for i in range(len(layer_dims)-1):
    plots.append(win.addPlot(title='weights layer W'+str(i+1)))
    plots[-1].showGrid(x=False, y=True)
    plots[-1].setLogMode(x=False, y=True)
    curves.append(plots[-1].plot(pen='g')) #min
    curves.append(plots[-1].plot(pen=(100,100,0))) #10%
    curves.append(plots[-1].plot(pen=(100,100,0))) #25%
    curves.append(plots[-1].plot(pen=(255,255,0))) #50%
    curves.append(plots[-1].plot(pen=(100,100,0))) #75%
    curves.append(plots[-1].plot(pen=(100,100,0))) #90%
    curves.append(plots[-1].plot(pen='g')) #max 
    
for i in range(len(layer_dims)-1):
    plots.append(win.addPlot(title='weights layer B'+str(i+1)))
    plots[-1].showGrid(x=False, y=True)
    plots[-1].setLogMode(x=False, y=True)
    curves.append(plots[-1].plot(pen='g')) #min
    curves.append(plots[-1].plot(pen=(100,100,0))) #10%
    curves.append(plots[-1].plot(pen=(100,100,0))) #25%
    curves.append(plots[-1].plot(pen=(255,255,0))) #50%
    curves.append(plots[-1].plot(pen=(100,100,0))) #75%
    curves.append(plots[-1].plot(pen=(100,100,0))) #90%
    curves.append(plots[-1].plot(pen='g')) #max 
win.nextRow()

for i in range(len(layer_dims)-1):
    plots.append(win.addPlot(title='grads layer W'+str(i+1)))
    plots[-1].showGrid(x=False, y=True)
    plots[-1].setLogMode(x=False, y=True)
    curves.append(plots[-1].plot(pen='g')) #min
    curves.append(plots[-1].plot(pen=(100,100,0))) #10%
    curves.append(plots[-1].plot(pen=(100,100,0))) #25%
    curves.append(plots[-1].plot(pen=(255,255,0))) #50%
    curves.append(plots[-1].plot(pen=(100,100,0))) #75%
    curves.append(plots[-1].plot(pen=(100,100,0))) #90%
    curves.append(plots[-1].plot(pen='g')) #max 
    
for i in range(len(layer_dims)-1):
    plots.append(win.addPlot(title='grads layer B'+str(i+1)))
    plots[-1].showGrid(x=False, y=True)
    plots[-1].setLogMode(x=False, y=True)
    curves.append(plots[-1].plot(pen='g')) #min
    curves.append(plots[-1].plot(pen=(100,100,0))) #10%
    curves.append(plots[-1].plot(pen=(100,100,0))) #25%
    curves.append(plots[-1].plot(pen=(255,255,0))) #50%
    curves.append(plots[-1].plot(pen=(100,100,0))) #75%
    curves.append(plots[-1].plot(pen=(100,100,0))) #90%
    curves.append(plots[-1].plot(pen='g')) #max 
win.nextRow()

for i in range(len(layer_dims)-1):
    plots.append(win.addPlot(title='delta stats layer W'+str(i+1)))
    plots[-1].showGrid(x=False, y=True)
    plots[-1].setLogMode(x=False, y=True)
    curves.append(plots[-1].plot(pen='g')) #min
    curves.append(plots[-1].plot(pen=(100,100,0))) #10%
    curves.append(plots[-1].plot(pen=(100,100,0))) #25%
    curves.append(plots[-1].plot(pen=(255,255,0))) #50%
    curves.append(plots[-1].plot(pen=(100,100,0))) #75%
    curves.append(plots[-1].plot(pen=(100,100,0))) #90%
    curves.append(plots[-1].plot(pen='g')) #max 

for i in range(len(layer_dims)-1):
    plots.append(win.addPlot(title='delta stats layer B'+str(i+1)))
    plots[-1].showGrid(x=False, y=True)
    plots[-1].setLogMode(x=False, y=True)
    curves.append(plots[-1].plot(pen='g')) #min
    curves.append(plots[-1].plot(pen=(100,100,0))) #10%
    curves.append(plots[-1].plot(pen=(100,100,0))) #25%
    curves.append(plots[-1].plot(pen=(255,255,0))) #50%
    curves.append(plots[-1].plot(pen=(100,100,0))) #75%
    curves.append(plots[-1].plot(pen=(100,100,0))) #90%
    curves.append(plots[-1].plot(pen='g')) #max 
win.nextRow()

plots.append(win.addPlot(title='visualize test fit',colspan=2*len(layer_dims)-2))
curves.append(plots[-1].plot(pen='g'))
curves.append(plots[-1].plot(pen='r'))

#outfile = 'C:\Users\leportfr\Desktop\Phynance\outPickle'
#openfile = open(outfile, 'wb')
loss_list = list()
test_loss_list_ma = list()
#learnRate = list()
return_list = list()
test_return_plot_list = list()
next_test_return = list()
weight_listW = list()
weight_listB = list()
grad_listW = list()
grad_listB = list()
learn_rate_listW = list()
learn_rate_listB = list()
predTestList = list()
cur_iter=0
t5=0
train_set_list = np.arange(num_training_sets).reshape(num_training_sets/mini_batch_size,mini_batch_size)
stats_graph_factor = 20
test_graph_factor = 20
def iterate():
    global cur_iter, t5, predTestList, train_set_list
    
    #generate minibatches
    if random_batch: 
        if cur_iter % (num_training_sets/mini_batch_size) == 0:
            train_set_list = np.arange(num_training_sets)
            np.random.shuffle(train_set_list)
            train_set_list=train_set_list.reshape(num_training_sets/mini_batch_size,mini_batch_size)
        train_set = train_set_list[cur_iter%(num_training_sets/mini_batch_size)]
    else:
        train_set = train_set_list[cur_iter%(num_training_sets/mini_batch_size)]    
        
    #generate dropout lists
    dropout_list = [[1]*x_dim]
    dropout_list_test = [[1]*x_dim]
    for i in range(len(layer_dims)-2):
        dropout_list.append([0]*int(mem_cells[i]*dropout_rate))
        dropout_list[-1] += [1]*(mem_cells[i]-len(dropout_list[-1]))
        np.random.shuffle(dropout_list[-1])
        
        dropout_list_test.append([1]*mem_cells[i])
    
    #forward propagate the network
    t1 = clock()
    print '\ncur iter: ', cur_iter, train_set
    print 'iteration time: ', clock() - t5
    for val in np.transpose(x_list_train[train_set],(1,0,2)):
        lstm_net.x_list_add(val, dropout_list)
    predList = rescaleY(scaleFactorAY, scaleFactorBY, lstm_net.getOutData())
#    return_list.append(np.average([(strategy.trade_cont(inputData[ts,-train_len:],predList[-train_len:,i,0], 0, sdol=sdolinit, bidask=bidaskinit, com=cominit)[-1]-1.e5)/(ideal_return[ts]-1.e5) for i,ts in enumerate(train_set)]))
#    return_list_ma=[]
    print 'add x_val time: ', clock() - t1  
    
    #back propagate the network
    t0 = clock()
    loss_list.append(lstm_net.y_list_is(np.transpose(y_list_train[train_set],(1,0,2))))
    print 'train time: ', clock() - t0
    
    #apply weight updates
    t2 = clock()
    if cur_iter % stats_graph_factor == 0:
        grad_listW.append(lstm_net.getGradStatsW())
        grad_listB.append(lstm_net.getGradStatsB())
    
    for lyr,lstm_param in enumerate(lstm_net.lstm_params):
        lstm_param.apply_diff(l2=l2_factor)
    
    if cur_iter % stats_graph_factor == 0:
        weight_listW.append(lstm_net.getWeightStatsW())
        weight_listB.append(lstm_net.getWeightStatsB())
        learn_rate_listW.append(lstm_net.getLearnRateStatsW())
        learn_rate_listB.append(lstm_net.getLearnRateStatsB())
    print 'apply time: ', clock() - t2
    
    #clear network for next iteration
    lstm_net.x_list_clear()
    
    #update graphs
    t4 = clock()
    curves[0].setData(rescaled_data[train_set[0],:,0])
    curves[1].setData(predList[:,0,0])
    curves[2].setData(loss_list)
#    curves[5].setData(np.arange(len(return_list)),return_list)

    if cur_iter % test_graph_factor == 0:
        t3 = clock()
#        test_return_list = []
        test_loss_list = []
        predTestList = []
        for i in range(num_test_sets/mini_batch_size):
            for val in np.transpose(x_list_test[i:i+mini_batch_size],(1,0,2)):
                lstm_net.x_list_add(val, dropout_list_test)#*(((np.random.rand(mini_batch_size)+.5)/50.0).reshape((100,1))), 0.0)
            predTestList.append(rescaleY(scaleFactorAY, scaleFactorBY, lstm_net.getOutData()))
            [test_loss_list.append(np.sum(lstm.loss_func(predTestList[-1][-train_len:,j,0],ytest_list_full[test_set,-train_len:]))) for j,test_set in enumerate(np.arange(i*mini_batch_size,i*mini_batch_size+mini_batch_size))]
#            if i==0:
#                next_test_return.append((strategy.trade_cont(testData[0,-train_len:],predTestList[-1][-train_len:,0,0], 0, sdol=sdolinit, bidask=bidaskinit, com=cominit)[-1]-1.e5)/(ideal_test_return[0]-1.e5))
#                [test_return_list.append((strategy.trade_cont(testData[test_set,-train_len:],predTestList[-1][-train_len:,j,0], 0, sdol=sdolinit, bidask=bidaskinit, com=cominit)[-1]-1.e5)/(ideal_test_return[test_set]-1.e5)) for j,test_set in enumerate(np.arange(i*mini_batch_size,i*mini_batch_size+mini_batch_size))]
#            else:
#            [test_return_list.append((strategy.trade_cont(testData[test_set,-train_len:],predTestList[-1][-train_len:,j,0], 0, sdol=sdolinit, bidask=bidaskinit, com=cominit)[-1]-1.e5)/(ideal_test_return[test_set]-1.e5)) for j,test_set in enumerate(np.arange(i*mini_batch_size,i*mini_batch_size+mini_batch_size))]
            lstm_net.x_list_clear()
        test_loss_list_ma.append(np.average(test_loss_list))
        print 'test time: ', clock() - t3
        curves[4].setData(np.arange(len(test_loss_list_ma))*test_graph_factor,test_loss_list_ma)
        
#        y_test_hist,x_test_hist = np.histogram(test_return_list, bins=np.linspace(-0.5, 1, 30))
#        curves[7].setData(x_test_hist, y_test_hist)  
        y_test_loss_hist,x_test_loss_hist = np.histogram(test_loss_list, bins=np.linspace(0, 100, 50))
        curves[8].setData(x_test_loss_hist, y_test_loss_hist) 
        
#        ave=np.average(test_return_list)
#        std=np.std(test_return_list)
#        test_return_plot_list.append([np.amin(test_return_list),ave-std,ave,ave+std,np.amax(test_return_list)])
#        for i in range(5):
#            curves[9+i].setData(np.arange(len(test_return_plot_list))*test_graph_factor,np.array(test_return_plot_list)[:,i])
#        curves[14].setData(np.arange(len(test_return_plot_list))*test_graph_factor,next_test_return)
        
    if cur_iter % stats_graph_factor == 0:
        for i in range(len(layer_dims)-1):
            for j in range(7):
                curves[15+7*i+j].setData(np.arange(len(weight_listW))*stats_graph_factor,np.array(weight_listW)[:,i,j])
                curves[(15+7*(len(layer_dims)-1))+7*i+j].setData(np.arange(len(weight_listB))*stats_graph_factor,np.array(weight_listB)[:,i,j]) 
                curves[(15+14*(len(layer_dims)-1))+7*i+j].setData(np.arange(len(grad_listW))*stats_graph_factor,np.array(grad_listW)[:,i,j]) 
                curves[(15+21*(len(layer_dims)-1))+7*i+j].setData(np.arange(len(grad_listB))*stats_graph_factor,np.array(grad_listB)[:,i,j]) 
                curves[(15+28*(len(layer_dims)-1))+7*i+j].setData(np.arange(len(learn_rate_listW))*stats_graph_factor,np.array(learn_rate_listW)[:,i,j]) 
                curves[(15+35*(len(layer_dims)-1))+7*i+j].setData(np.arange(len(learn_rate_listB))*stats_graph_factor,np.array(learn_rate_listB)[:,i,j])  
    
    curves[15+42*(len(layer_dims)-1)].setData(ytest_list_full[cur_iter%num_test_sets,:,0])
    curves[15+42*(len(layer_dims)-1)+1].setData(predTestList[int(cur_iter/mini_batch_size)%(num_test_sets/mini_batch_size)][:,cur_iter%mini_batch_size,0])
    print 'display time: ', clock() - t4
    
#    pickle.dump(lstm_net.getParams(), openfile)
    print 'totaltime', clock() - t1
    print 'loss: ', loss_list[-1]
    cur_iter+=1
    t5 = clock()
    
#    predTrueTestList=0
#    if cur_iter>=0:
#        ## build and scale output and test arrays
#        buySellListTrue = zip(*[strategy.ideal_strategy(inpt[-(history_len+train_len):], sshares=0, sdol=sdolinit, bidask=bidaskinit, com=cominit) for inpt in inputTrueData])[1]
#        y_list_true_full = np.zeros([num_training_sets,history_len+train_len])
#        for j,sublist in enumerate(buySellListTrue):
#            mult = 1
#            for i in sublist:
#                y_list_true_full[:,i] = mult
#                mult*=-1
#        ideal_true_return = [strategy.trade(inputTrueData[i,-train_len:],y_list_true_full[i,-train_len:], sdol=sdolinit, bidask=bidaskinit, com=cominit)[-1] for i in range(num_training_sets)]
#        print 'ideal true test return', ideal_true_return[0]
#        
#        for i,val in enumerate(np.transpose(x_list_true_train[0:mini_batch_size],(1,0,2))):
#            lstm_net.x_list_add(val, 0.0)
#        predTrueTestList = rescaleY(scaleFactorAY, scaleFactorBY, lstm_net.getOutData())
#        print 'pred true test return factor', (strategy.trade(inputTrueData[0,-train_len:],predTrueTestList[-train_len:,0,0], sdol=sdolinit, bidask=bidaskinit, com=cominit)[-1]-1.e5)/(ideal_true_return[0]-1.e5)
#        lstm_net.x_list_clear()
    
timer = QtCore.QTimer()
timer.timeout.connect(iterate)
timer.start(1)
#openfile.close()

###------ Start Qt event loop unless running in interactive mode or using pyside ------###
if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()