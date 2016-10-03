import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lstm

curdir = os.path.dirname(__file__)
datadir = os.path.join(curdir, 'Data/quantquote_daily_sp500_83986/daily')
filelist = os.listdir(datadir)[:10]

sp500 = pd.DataFrame(index=range(np.size(filelist)),columns=range(1))

for i,filename in enumerate(filelist):
    stockSymbol = filename[6:-4]
    print stockSymbol
    with open(os.path.join(datadir, filename), 'rb') as csvfile: 
        filereader = csv.reader(csvfile, delimiter=',', quotechar='|')
        
        numlines = len(csvfile.readlines())
        stockData = np.empty([numlines,7])
        csvfile.seek(0)
        
        for j,row in enumerate(filereader):
#            print j
            stockData[j]=row
    sp500.set_value(i,0,pd.DataFrame(stockData))

#inputData = np.array([np.array(sp500.loc[2,0])[:,5]])
inputData = np.array([np.array(sp500.loc[2,0])[:,5],np.array(sp500.loc[1,0])[:,5]])

lstm = LSTM.LSTM(alpha=.2,inputDim=inputData.shape[0],hiddenDim=50,hiddenLayers=5)
lstm.train(inputData,np.zeros(inputData.shape[0]),np.max(inputData,axis=1))

f,axarr = plt.subplots(2,sharex=True)
for j in range(inputData.shape[0]):
    axarr[0].plot(inputData[j,1:])
    axarr[0].plot(np.array(lstm.predictionList)[:,j])
    axarr[1].plot(np.array(lstm.errorList)[:,j]/np.array(lstm.predictionList)[:,j])
axarr[1].set_ylim([-1,1])