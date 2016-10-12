import pandas as pd
import os

def loadData():
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
    
    return df