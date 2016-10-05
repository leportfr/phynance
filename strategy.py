import numpy as np
from numba import autojit

#@autojit
def ewma_strategy(stock_price, pred_ewma, sdol=1.0e5, bidask=0.005, com=9.99):
    dollars=sdol
    shares=0
    value=list()
    
    askmul=1.0+bidask/2.0
    bidmul=1.0-bidask/2.0
    
    alpha=0
    for i,pred in enumerate(pred_ewma):
        if pred>alpha and dollars>(stock_price[i]*askmul+com):
            newshares=int((dollars-com)/(stock_price[i]*askmul))
            shares+=newshares
#            print 'if', shares, dollars
            dollars-=newshares*(stock_price[i]*askmul)+com
        elif pred<alpha and shares>0:
            dollars+=shares*(stock_price[i]*bidmul)-com
#            print 'elif', shares, dollars
            shares=0
        value.append(dollars+shares*stock_price[i])
#        print 'out', shares, dollars, value[-1], stock_price[i]
    
    return value
    
def ideal_strategy(stock_price, sdol=1.0e5, bidask=0.005, com=9.99):
    shifts=np.diff(np.array(np.diff(stock_price[:])>0,dtype=int))
    enumshifts=np.array(list(enumerate(shifts)))
    changePoints=enumshifts[enumshifts[:,1]!=0]
    if np.array_equal(changePoints[0],[0,-1]):
        changePoints=changePoints[1:]
    elif changePoints[0,1]==-1:
        changePoints=np.concatenate(([[0,1]],changePoints))
#    print stock_price
#    print shifts
    
    dollars=sdol
    shares=0  
     
    askmul=1.0+bidask/2.0
    bidmul=1.0-bidask/2.0
    
    buySellList=list()
    
    for i in range(len(changePoints)/2):
        pos1=changePoints[2*i,0]+1
        pos2=changePoints[2*i+1,0]+1
#        print pos1, pos2
        orgdollars=dollars
        shares=int((dollars-com)/(stock_price[pos1]*askmul))
        dollars-=shares*(stock_price[pos1]*askmul)+com
        dollars+=shares*(stock_price[pos2]*bidmul)-com
        if orgdollars>dollars:
#            print 'skip',pos1
            dollars=orgdollars
        else:
            buySellList.append(pos1)
            buySellList.append(pos2)
    
    return (dollars,buySellList)