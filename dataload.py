import pandas as pd
import numpy as np
import os

np.random.seed(0)

def loadTrueTestData():
    #locate data files
    curdir = os.path.dirname(__file__)
    datadir = os.path.join(curdir, 'Data/')
    filelist = os.listdir(datadir)[0:1]
    
    #load stock data
    sp500dict = {}
    for i,filename in enumerate(filelist):
        stockSymbol = filename[6:-4]
        stockData = pd.read_csv(os.path.join(datadir, filename),index_col=0,header=None,
                                names=('Date','Open','High','Low','Close','Volume','Adj Close'))
        sp500dict[str(stockSymbol)] = stockData
    sp500 = pd.Panel(sp500dict)
#    print sp500
    df = sp500.loc[:,:,'Close'].dropna()
    
    return df[::-1][:-1]

def loadData():
    #locate data files
    curdir = os.path.dirname(__file__)
    datadir = os.path.join(curdir, 'Data/quantquote_daily_sp500_83986/daily')
    filelist = os.listdir(datadir)[:]
    
    #load stock data
    sp500dict = {}
    for i,filename in enumerate(filelist):
        stockSymbol = filename[6:-4]
        stockData = pd.read_csv(os.path.join(datadir, filename),index_col=0,header=None,
                                names=('day','time','open','high','low','close','volume'))
        sp500dict[str(stockSymbol)] = stockData
    sp500 = pd.Panel(sp500dict)
    print sp500
    df = sp500.loc[:,:,['close','volume']].dropna()
    
#    print 'df',df.loc[:,:,'volume']
    
    return df
    
def loadDataTest():
    #locate data files
    init = [np.random.rand() + 0.5 for i in range(100)]
    pos = np.random.choice(99,size=10)
    randarr = np.random.rand(99) + 1./3
    
    for i in range(3926-len(init)):
        x=np.average(np.array(init)[i+pos]) * randarr[pos[i%10]]
        init.append(x)
    
    df = init
    return np.array(df).reshape([len(df),1])
    
#def loadData2():
#    #locate data files
#    randmult = np.random.randint(5,size=50)
#    randarr2 = np.random.rand(5)
#    init = [np.random.rand() + 0.5 for i in range(100)]
#    init2 = 
#    
#    pos = np.random.choice(99,size=10)
#    randarr = np.random.rand(99) + 1./3
#    
#    for i in range(3926-len(init)):
#        x=np.average(np.array(init)[i+pos]) * randarr[pos[i%10]]
#        init.append(x)
#    
#    df = init
#    return np.array(df).reshape([len(df),1])
    
if __name__ == '__main__':
    df = loadData()
    print np.array(df.loc[:,:,'close']).shape