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
    
def ideal_strategy(stock_price, sdol=sdolinit, sshares=0, bidask=bidaskinit, com=cominit):        
    askmul=1.0+bidask/2.0
    bidmul=1.0-bidask/2.0
    buySellList=list()
    dollars=sdol
    pos1=0
    pos2=0
    
    diff=np.diff(stock_price)
    diff[(diff==0).nonzero()]=diff[(diff==0).nonzero()[0]-1]
    diff[diff>0]=1
    diff[diff<0]=0
    shifts=np.diff(diff)
    nzshifts=shifts.nonzero()[0]
    
    changePoints=np.zeros(len(nzshifts)+1)
    changePoints[1:]=nzshifts
    
    if sshares > 0:
        shift=0
        if len(nzshifts)==0:
            return (dollars+sshares*stock_price[0]*bidmul-com,[])
        if shifts[nzshifts[0]]==1:
            changePoints[0]=-1
        else: shift+=1
            
        pos2=int(changePoints[0+shift]+1)
        dollars+=sshares*(stock_price[pos2]*bidmul)-com
        shift +=1
        buySellList.append(pos2)
    else:
        shift=0
        if len(nzshifts)==0:
            return (dollars,[])
        if shifts[nzshifts[0]]==-1:
            changePoints[0]=-1
        else: shift+=1
        
    for i in range((len(changePoints)-shift)/2):
        pos1=int(changePoints[2*i+shift]+1)
        pos2=int(changePoints[2*i+1+shift]+1)
        orgdollars=dollars
        
        shares=int((dollars-com)/(stock_price[pos1]*askmul))
        dollars-=shares*(stock_price[pos1]*askmul)+com
        dollars+=shares*(stock_price[pos2]*bidmul)-com
        
        if orgdollars>dollars:
            dollars=orgdollars
        else:
            buySellList.append(pos1)
            buySellList.append(pos2)
            
    if (len(changePoints)-shift)%2==1:
        pos1=int(changePoints[-1]+1)
        orgdollars=dollars
        
        shares=int((dollars-com)/(stock_price[pos1]*askmul))
        dollars-=shares*(stock_price[pos1]*askmul)+com
        dollars+=shares*(stock_price[-1]*bidmul)-com
        
        if orgdollars>dollars:
            dollars=orgdollars
        else:
            buySellList.append(pos1)
    
    return (dollars,buySellList)
    
def ideal_strategyOrg(stock_price, sdol=sdolinit, bidask=bidaskinit, com=cominit):
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