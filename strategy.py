import numpy as np
from numba import autojit

sdolinit = 1.0e5
bidaskinit = 0.005
cominit = 9.99

def ewma_strategy(stock_price, pred_ewma, sdol=sdolinit, bidask=bidaskinit, com=cominit):
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
    
def ideal_strategy(stock_price, sdol=sdolinit, sshares=None, bidask=bidaskinit, com=cominit):
    if sshares is not None:
        flag=False
        shares=sshares
    else:
        flag=True
        shares=0
        
    askmul=1.0+bidask/2.0
    bidmul=1.0-bidask/2.0
    buySellList=list()
    
    diff=np.diff(stock_price[:])
    for i,val in enumerate(diff):
        if val > 0:
            diff[i] = 1
        elif val < 0:
            diff[i] = 0
        else:
            diff[i] = diff[i-1]
    shifts=np.diff(diff)
    enumshifts=np.array(list(enumerate(shifts)))
    changePoints=enumshifts[enumshifts[:,1]!=0]
    
    if len(changePoints)==0:
        return (sdol+shares*stock_price[0]*bidmul-com,[])
    
    if changePoints[0,1]==(-1 if flag else 1):
        changePoints=np.concatenate(([[-1,1] if flag else [-1,-1]],changePoints))
    
    if flag:
        dollars=sdol 
        shift=0
    else:
        pos2=changePoints[0,0]+1
        dollars=sdol
        dollars+=shares*(stock_price[pos2]*bidmul)-com
        buySellList.append(pos2)
        shares=0
        shift=1
        
    for i in np.arange((len(changePoints)-shift)/2):
        pos1=changePoints[2*i+shift,0]+1
        pos2=changePoints[2*i+1+shift,0]+1
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
            
    if (len(changePoints)-shift)%2==1:
        pos1=changePoints[-1,0]+1
        orgdollars=dollars
        
        shares=int((dollars-com)/(stock_price[pos1]*askmul))
        dollars-=shares*(stock_price[pos1]*askmul)+com
        dollars+=shares*(stock_price[-1]*bidmul)-com
        
        if orgdollars>dollars:
#            print 'skip',pos1
            dollars=orgdollars
        else:
            buySellList.append(pos1)
    
    return (dollars,buySellList)
    
def buysellVal(stock_price, sval=sdolinit, bidask=bidaskinit, com=cominit):
    print stock_price
    
    askmul=1.0+bidask/2.0
    bidmul=1.0-bidask/2.0
    
    buyval=list()
    for i in range(len(stock_price)-3):
        shares=int((sval-com)/(stock_price[i]*askmul))
        dollars=sval-shares*(stock_price[i]*askmul)-com
        buyval.append(ideal_strategy(stock_price[i+1:], sdol=dollars, sshares=shares)[0] / ideal_strategy(stock_price[i:], sdol=sval, sshares=None)[0])
        
    sellval=list()
    for i in range(len(stock_price)-3):
        shares=int((sval)/(stock_price[i]*bidmul))
        dollars=sval-shares*(stock_price[i]*bidmul)
        sellval.append(ideal_strategy(stock_price[i+1:], sdol=sval-com, sshares=None)[0] / ideal_strategy(stock_price[i:], sdol=dollars, sshares=shares)[0])
        
    combval=[(1./sellval[i] if val==1. else val) for i,val in enumerate(buyval)]
    return combval
    
#def ideal_strategyOld(stock_price, sdol=sdolinit, bidask=bidaskinit, com=cominit):
#    shifts=np.diff(np.array(np.diff(stock_price[:])>0,dtype=int))
#    enumshifts=np.array(list(enumerate(shifts)))
#    changePoints=enumshifts[enumshifts[:,1]!=0]
#    if np.array_equal(changePoints[0],[0,-1]):
#        changePoints=changePoints[1:]
#    elif changePoints[0,1]==-1:
#        changePoints=np.concatenate(([[0,1]],changePoints))
##    print stock_price
##    print shifts
#    
#    dollars=sdol
#    shares=0  
#     
#    askmul=1.0+bidask/2.0
#    bidmul=1.0-bidask/2.0
#    
#    buySellList=list()
#    
#    for i in range(len(changePoints)/2):
#        pos1=changePoints[2*i,0]+1
#        pos2=changePoints[2*i+1,0]+1
##        print pos1, pos2
#        orgdollars=dollars
#        shares=int((dollars-com)/(stock_price[pos1]*askmul))
#        dollars-=shares*(stock_price[pos1]*askmul)+com
#        dollars+=shares*(stock_price[pos2]*bidmul)-com
#        if orgdollars>dollars:
##            print 'skip',pos1
#            dollars=orgdollars
#        else:
#            buySellList.append(pos1)
#            buySellList.append(pos2)
#    
#    return (dollars,buySellList)
    
def trade_abs(stock_price, trades, sdol=sdolinit, bidask=bidaskinit, com=cominit):
    assert(len(stock_price)==len(trades))
    
    dollars=sdol
    shares=0
    value=list()   
    
    askmul=1.0+bidask/2.0
    bidmul=1.0-bidask/2.0
    
    buy=1
    for i,sp in enumerate(stock_price):
        if trades[i]*buy>0.5 and dollars>=(stock_price[i]*askmul+com):
            newshares=int((dollars-com)/(stock_price[i]*askmul))
            shares+=newshares
#            print 'if', shares, dollars
            dollars-=newshares*(stock_price[i]*askmul)+com
            buy*=-1
        elif trades[i]*buy<-0.5 and shares>0:
            dollars+=shares*(stock_price[i]*bidmul)-com
#            print 'elif', shares, dollars
            shares=0
            buy*=-1
        value.append(dollars+shares*stock_price[i])
#        print 'out', shares, dollars, value[-1], stock_price[i]
    
    return value
    
def trade(stock_price, trades, sdol=sdolinit, bidask=bidaskinit, com=cominit):
    dollars=sdol
    shares=0
    value=list()   
    
    askmul=1.0+bidask/2.0
    bidmul=1.0-bidask/2.0
    
    for i,sp in enumerate(stock_price):
        if trades[i]>0.5 and dollars>=(stock_price[i]*askmul+com):
            newshares=int((dollars-com)/(stock_price[i]*askmul))
            shares+=newshares
#            print 'if', shares, dollars
            dollars-=newshares*(stock_price[i]*askmul)+com
        elif trades[i]<-0.5 and shares>0:
            dollars+=shares*(stock_price[i]*bidmul)-com
#            print 'elif', shares, dollars
            shares=0
        value.append(dollars+shares*stock_price[i])
#        print 'out', shares, dollars, value[-1], stock_price[i]
    
    return value