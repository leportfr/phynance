import numpy as np
from time import clock
import strategy
import cPickle as pickle
import sys
import pyqtgraph as pg 
from pyqtgraph.Qt import QtGui, QtCore

from dataload import loadData, loadTrueTestData
import lstm

class train_lstm():
    def __init__(self):
        #initialize training and test set parameters
        np.random.seed(85)
    
        self.history_len = 365
        self.train_len = 100
        self.datalen = self.history_len + self.train_len
        
        self.num_training_sets = 100
        self.mini_batch_size = 100
        self.random_batch = 1
        self.num_test_sets = 100
        assert(self.num_training_sets % self.mini_batch_size == 0)
        assert(self.num_test_sets % self.mini_batch_size == 0)
        
        self.test_train_cutoff = 2500
        self.test_limit = None      
        
        #initialize lstm hyper-parameters
        self.iterations = int(40)
        self.init_learn_rate = 1.e-3
        self.learn_factor = 0.1
        self.ema_factor = 0.8#(1.-1./num_training_sets*mini_batch_size)
        self.l2_factor = 0.0
        self.dropout_rate = 0.0
        self.mem_cells = [50,50,50]
        
        #initialize scaling parameters
        self.scalerangeX = 10.0
        self.scalerangeY = 1.0
        self.scaleFactorA = None
        self.scaleFactorB = None
        self.scaleFactorBY = -1.0
        self.scaleFactorAY = self.scalerangeY/(1.0-self.scaleFactorBY)
        
        #initialize trading parameters
        self.sdolinit = 1.0e5
        self.bidaskinit = 0.005
        self.cominit = 9.99
        
    def scaleX(self,x):
        assert(self.scaleFactorA is not None)
        assert(self.scaleFactorB is not None)
        return (self.scaleFactorA * (x.T - self.scaleFactorB) - self.scalerangeX/2.0).T
    def rescaleX(self,y):
        assert(self.scaleFactorA is not None)
        assert(self.scaleFactorB is not None)
        return self.scaleFactorB + (y + self.scalerangeX/2.0) / self.scaleFactorA
        
    def scaleY(self,x):
        assert(self.scaleFactorAY is not None)
        assert(self.scaleFactorBY is not None)
        return (self.scaleFactorAY * (x.T - self.scaleFactorBY) - 0*self.scalerangeY/2.0).T 
    def rescaleY(self,y):
        assert(self.scaleFactorAY is not None)
        assert(self.scaleFactorBY is not None)
        return self.scaleFactorBY + (y + 0*self.scalerangeY/2.0) / self.scaleFactorAY
        
    def loadData(self):  
        self.df = loadData()
        self.test_limit = self.df.shape[0]
        
    def buildDataSets(self):
        self.x_dim = 1
        
        mov1 = 0
        self.inputData = np.array([np.array(self.df).T[0,i:i+self.datalen+1+mov1] for i in np.random.choice(self.test_train_cutoff, size=self.num_training_sets, replace=False)])
        self.testData = np.array([np.array(self.df).T[0,i+self.test_train_cutoff+self.datalen+1+mov1:i+self.test_train_cutoff+2*(self.datalen+1+mov1)] for i in np.random.choice(self.test_limit-(self.test_train_cutoff+2*(self.datalen+1+mov1)),size=self.num_test_sets-1,replace=False)])
        self.testData = np.concatenate([[np.array(self.df).T[0,self.test_train_cutoff+self.train_len:self.test_train_cutoff+self.datalen+self.train_len+1+mov1]] , self.testData])
        
        inputDataDiffQuot = (self.inputData[:,1:] - self.inputData[:,:-1]) / self.inputData[:,:-1]
        testDataDiffQuot = (self.testData[:,1:] - self.testData[:,:-1]) / self.testData[:,:-1]
        minval = np.amin([np.amin(np.abs(inputDataDiffQuot[np.nonzero(inputDataDiffQuot)])), np.amin(np.abs(testDataDiffQuot[np.nonzero(testDataDiffQuot)]))]) / 10.0
        inputDataDiff = np.array([np.sign(inputDataDiffQuot) * np.nan_to_num(np.log10(np.abs(inputDataDiffQuot/minval)))])
        testDataDiff = np.array([np.sign(testDataDiffQuot) * np.nan_to_num(np.log10(np.abs(testDataDiffQuot/minval)))])
        #inputDataDiff = np.concatenate([inputDataDiff,[np.diff(inputData,axis=1)]])
        #testDataDiff = np.concatenate([testDataDiff,[np.diff(testData,axis=1)]])
        
        self.scaleFactorB = np.amin(np.amin(np.concatenate([inputDataDiff,testDataDiff],axis=1),axis=1),axis=1)
        self.scaleFactorA = self.scalerangeX / (np.amax(np.amax(np.concatenate([inputDataDiff,testDataDiff],axis=1),axis=1),axis=1) - self.scaleFactorB)
        
        if mov1 == 1:
            self.x_list_train = np.transpose(self.scaleX(inputDataDiff[:,:,-(self.datalen+mov1):-mov1]),(1,2,0))
            self.x_list_test = np.transpose(self.scaleX(testDataDiff[:,:,-(self.datalen+mov1):-mov1]),(1,2,0))
        else:
            self.x_list_train = np.transpose(self.scaleX(inputDataDiff),(1,2,0))
            self.x_list_test = np.transpose(self.scaleX(testDataDiff),(1,2,0))        
    
    def buildOutputTrainingArrays(self):
        self.y_dim = 1        
        
        buySellList = zip(*[strategy.ideal_strategy(inpt[-self.datalen:], sshares=0, sdol=self.sdolinit, bidask=self.bidaskinit, com=self.cominit) for inpt in self.inputData])[1]
        self.y_list_full = np.zeros([self.num_training_sets,self.datalen,self.y_dim])
        for j,sublist in enumerate(buySellList):
            mult = 1
            for i in sublist:
                self.y_list_full[j,i,0] = mult
                mult*=-1

        self.ideal_return = [strategy.trade(self.inputData[i,-self.train_len:] , self.y_list_full[i,-self.train_len:], sdol=self.sdolinit, bidask=self.bidaskinit, com=self.cominit)[-1] for i in range(self.num_training_sets)]
        self.y_list_train = self.scaleY(self.y_list_full[:,-self.train_len:])

    def buildOutputTestArrays(self):
        buySellList = zip(*[strategy.ideal_strategy(inpt[-(self.datalen):], sshares=0, sdol=self.sdolinit, bidask=self.bidaskinit, com=self.cominit) for inpt in self.testData])[1]
        self.ytest_list_full = np.zeros([self.num_test_sets,self.datalen,self.y_dim])
        for j,sublist in enumerate(buySellList):
            mult=1
            for i in sublist:
                self.ytest_list_full[j,i,0] = mult
                mult*=-1

        self.ideal_test_return = [strategy.trade(self.testData[i,-self.train_len:] , self.ytest_list_full[i,-self.train_len:,0], sdol=self.sdolinit, bidask=self.bidaskinit, com=self.cominit)[-1] for i in range(self.num_test_sets)]
        print 'ave yearly test return', (np.average(self.ideal_test_return)/1.e5)**(365.0/self.train_len*5.0/7.0)

    def buildNetwork(self):
        layer_dims = [self.x_dim]+self.mem_cells+[self.y_dim]
        self.lstm_net = lstm.LstmNetwork(layer_dims, self.init_learn_rate/self.mini_batch_size, self.ema_factor**(float(self.mini_batch_size)/float(self.num_training_sets)), self.num_training_sets/self.mini_batch_size, self.mini_batch_size)

    def trainNetwork(self):
        def iterate(cur_iter):
            if self.random_batch:
        #        train_set = np.random.randint(num_training_sets) 
                if cur_iter % self.num_training_sets/self.mini_batch_size == 0:
                    train_set_list = np.arange(self.num_training_sets)
                    np.random.shuffle(train_set_list)
                    train_set_list=train_set_list.reshape(self.num_training_sets/self.mini_batch_size,self.mini_batch_size)
                train_set = train_set_list[cur_iter%(self.num_training_sets/self.mini_batch_size)]
            else:
                train_set = train_set_list[cur_iter%(self.num_training_sets/self.mini_batch_size)]    
            
            t1 = clock()
            print '\ncur iter: ', cur_iter#, train_set
            for val in np.transpose(self.x_list_train[train_set],(1,0,2)):
                self.lstm_net.x_list_add(val, self.dropout_rate)
#            predList = self.rescaleY(self.lstm_net.getOutData())
#            return_list.append(np.average([(strategy.trade(self.inputData[ts,-self.train_len:],predList[-self.train_len:,i,0])[-1]-1.e5)/(self.ideal_return[ts]-1.e5) for i,ts in enumerate(train_set)]))
        #    return_list_ma=[]
            print 'add x_val time: ', clock() - t1  
            
            t0 = clock()
            self.lstm_net.y_list_is(np.transpose(self.y_list_train[train_set],(1,0,2)))
            print 'train time: ', clock() - t0
            
            t2 = clock()
            for lyr,lstm_param in enumerate(self.lstm_net.lstm_params):
                lstm_param.apply_diff(l2=self.l2_factor, lr=self.learn_factor*self.mini_batch_size/self.num_training_sets)
            print 'apply time: ', clock() - t2
            
            self.lstm_net.x_list_clear()
        
            if cur_iter == self.iterations-1:
                t3 = clock()
                test_return_list = []
                predTestList = []
                for i in range(self.num_test_sets/self.mini_batch_size):
                    for val in np.transpose(self.x_list_test[i:i+self.mini_batch_size],(1,0,2)):
                        self.lstm_net.x_list_add(val, 0.0)#*(((np.random.rand(mini_batch_size)+.5)/50.0).reshape((100,1))), 0.0)
                    predTestList = self.rescaleY(self.lstm_net.getOutData())
                    [test_return_list.append((strategy.trade(self.testData[test_set,-self.train_len:], predTestList[-self.train_len:,j,0], sdol=self.sdolinit, bidask=self.bidaskinit, com=self.cominit)[-1]-1.e5)/(self.ideal_test_return[test_set]-1.e5)) for j,test_set in enumerate(np.arange(i*self.mini_batch_size,i*self.mini_batch_size+self.mini_batch_size))]
                    self.lstm_net.x_list_clear()
                print 'test time: ', clock() - t3
                print 'ave, min test return factor', np.average(test_return_list), np.amin(test_return_list)
            
        #    pickle.dump(lstm_net.getParams(), openfile)
            print 'totaltime', clock() - t1
            
        for cur_iter in range(self.iterations):
            iterate(cur_iter)
            
    def getFowardNetwork(self):
        w = list()
        b = list()
        for lstm_param in self.lstm_net.lstm_params:
            (ws,bs) = lstm_param.getParams()
            w.append(ws)
            b.append(bs)
        
        layer_dims = [self.x_dim]+self.mem_cells+[self.y_dim]
        lstm_forward_net = lstm.LstmNetwork(layer_dims, self.init_learn_rate/self.mini_batch_size, self.ema_factor**(float(self.mini_batch_size)/float(self.num_training_sets)), self.num_training_sets/self.mini_batch_size, 1)
        lstm_forward_net.setParameters(w,b)        
        
        return lstm_forward_net
            
if __name__ == '__main__':
    train = train_lstm()
    
    train.loadData()
    train.buildDataSets()
    train.buildOutputTrainingArrays()
    train.buildOutputTestArrays()
    train.buildNetwork()
    train.trainNetwork()
    fn = train.getFowardNetwork()
    
    dfTrue = loadTrueTestData()    
    inputTrueData = np.array([np.array(dfTrue).astype(np.float64).T[0,-(train.history_len+train.train_len+1):]])
    inputTrueDataDiffQuot = (inputTrueData[:,1:]-inputTrueData[:,:-1])/inputTrueData[:,:-1]
    minval = np.amin([np.amin(np.abs(inputTrueDataDiffQuot[np.nonzero(inputTrueDataDiffQuot)]))])/10.0
    inputTrueDataDiff = np.array([np.sign(inputTrueDataDiffQuot)*np.nan_to_num(np.log10(np.abs(inputTrueDataDiffQuot/minval)))])
    x_list_true_train = np.transpose(train.scaleX(inputTrueDataDiff),(1,2,0)) 
    
    for i,val in enumerate(np.transpose(x_list_true_train[0:1],(1,0,2))):
        fn.x_list_add(val, 0.0)
    predTrueTestList = train.rescaleY(fn.getOutData())
    fn.x_list_clear()
    
    buySellListTrue = zip(*[strategy.ideal_strategy(inpt[-(train.history_len+train.train_len):], sshares=0, sdol=train.sdolinit, bidask=train.bidaskinit, com=train.cominit) for inpt in inputTrueData])[1]
    y_list_true_full = np.zeros([train.num_training_sets,train.history_len+train.train_len])
    for j,sublist in enumerate(buySellListTrue):
        mult = 1
        for i in sublist:
            y_list_true_full[:,i] = mult
            mult*=-1
    ideal_true_return = [strategy.trade(inputTrueData[0,-train.train_len:],y_list_true_full[0,-train.train_len:], sdol=train.sdolinit, bidask=train.bidaskinit, com=train.cominit)[-1]]
    
    print 'ideal true test return', ideal_true_return[0]
    print 'pred true test return factor', (strategy.trade(inputTrueData[0,-train.train_len:],predTrueTestList[-train.train_len:,0,0], sdol=train.sdolinit, bidask=train.bidaskinit, com=train.cominit)[-1]-1.e5)/(ideal_true_return[0]-1.e5)