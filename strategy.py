import numpy as np
import numba

#sdolinit = 1.0e5
#bidaskinit = 0.005
#cominit = 9.99

def ewma_strategy(stock_price, pred_ewma, sdol, bidask, com):
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
    
#@numba.jit(nopython=True)    
def ideal_strategy(stock_price, sshares, sdol, bidask, com):        
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
    
@numba.jit(nopython = True)
def trade_cont(stock_price, trades, sshares, sdol, bidask, com):
    dollars=sdol
    shares=sshares
    value=np.zeros_like(stock_price)   
    
    askmul = 1.0+bidask/2.0
    bidmul = 1.0-bidask/2.0
    
    for i,sp in enumerate(stock_price):
        tr = trades[i]
        if np.abs(tr) > 1.0:
            tr = tr/np.abs(tr)
        
        if tr>0 and dollars>=(stock_price[i]*askmul+com):
            newshares = int((dollars-com)/(stock_price[i]*askmul))
            newshares = np.round(tr*newshares)
#            print 'newshares', newshares
            
            shares += newshares
            dollars -= newshares*(stock_price[i]*askmul)+com
        elif tr<0 and shares>0:
            soldshares = np.round(-tr*shares)
#            print 'soldshares', soldshares
            
            dollars += soldshares*(stock_price[i]*bidmul)-com
            shares -= soldshares
        value[i] = dollars + shares*stock_price[i]*bidmul
    
    return value
    
@numba.jit(nopython = True)    
def trade_cont_prime(stock_price, trades, sdol, bidask, com, delta):
    drv = np.zeros_like(trades)
    
    s_p = np.zeros(len(stock_price)+1)
    s_p[:-1] = stock_price
    s_p[-1] = stock_price[-1]
        
    temptrades = np.zeros(len(trades)+1)
    temptrades2 = np.zeros(len(trades)+1)
    for i,val in enumerate(trades):
        temptrades[i] = val
        temptrades2[i] = val
    temptrades[-1] = trades[-1]
    temptrades2[-1] = trades[-1]
    
    temptrades[np.abs(temptrades)>1.0] = temptrades[np.abs(temptrades)>1.0]/np.abs(temptrades[np.abs(temptrades)>1.0])
    temptrades2[np.abs(temptrades2)>1.0] = temptrades2[np.abs(temptrades2)>1.0]/np.abs(temptrades2[np.abs(temptrades2)>1.0])
    
    for i,val in enumerate(trades):
        shares = int(sdol/stock_price[i])        
        
        if trades[i] < 1.0-delta:
            temptrades[i] += delta
        if trades[i] > -1.0+delta:
            temptrades2[i] -= delta
            
        trd1 = trade_cont(s_p[i:], temptrades[i:], shares, sdol, bidask, com)[-1]
        trd2 = trade_cont(s_p[i:], temptrades2[i:], shares, sdol, bidask, com)[-1]
        drv[i] = (trd1 - trd2) / (2*abs(delta))
        
        if trades[i] > 1.0 and drv[i] > 0.0:
            drv[i] = 0.0
        if trades[i] < -1.0 and drv[i] < 0.0:
            drv[i] = 0.0
        
        if trades[i] < 1.0-delta:
            temptrades[i] -= delta
        if trades[i] > -1.0+delta:
            temptrades2[i] += delta
    
    return drv/sdol

#@numba.jit(nopython = True)    
#def trade_cont_prime(stock_price, trades, sdol, bidask, com, delta):
##    org = trade_cont(stock_price, trades, sshares, sdol, bidask, com)[-1]
#    drv = np.zeros_like(trades)
#    dollars = sdol
#    shares = 0
#    
#    if trades[0]<0:
#        bidmul = 1.0 - bidask/2.0
#        newshares = int(dollars/(stock_price[0]*bidmul))
#        
#        shares += newshares
#        dollars -= newshares*(stock_price[0]*bidmul) 
#        
#    temptrades = np.zeros_like(trades)
#    temptrades2 = np.zeros_like(trades)
#    for i,val in enumerate(trades):
#        temptrades[i] = val
#        temptrades2[i] = val
#    
#    for i,val in enumerate(trades):
#        temptrades[i] += delta
#        temptrades2[i] -= delta
#        trd1 = trade_cont(stock_price, temptrades, shares, sdol, bidask, com)[-1]
#        trd2 = trade_cont(stock_price, temptrades2, shares, sdol, bidask, com)[-1]
#        drv[i] = (trd1 - trd2) / (2*delta)
#        temptrades[i] -= delta
#        temptrades2[i] += delta
#    
#    return drv/sdol

@numba.jit(nopython=True)        
def buysellVal(stock_price, sval, bidask, com):
    s_p = np.zeros(len(stock_price)+3)
    s_p[:-3] = stock_price
    s_p[-3:] = stock_price[-1]
    
    askmul=1.0+bidask/2.0
    bidmul=1.0-bidask/2.0
    
    outar=np.zeros((len(s_p)-3,2))
    
#    buyval=np.zeros(len(s_p)-3)
    for i in range(len(s_p)-3):
        shares=int((sval-com)/(s_p[i]*askmul))
        dollars=sval-shares*(s_p[i]*askmul)-com
        outar[i,0] = ideal_strategy(s_p[i+1:], sdol=dollars, sshares=shares, bidask=bidask, com=com) / ideal_strategy(s_p[i:], sdol=sval, sshares=0, bidask=bidask, com=com)
        
#    sellval=np.zeros(len(s_p)-3)
    for i in range(len(s_p)-3):
        shares=int((sval)/(s_p[i]*bidmul))
        dollars=sval-shares*(s_p[i]*bidmul)
        outar[i,1] = ideal_strategy(s_p[i+1:], sdol=sval-com, sshares=0, bidask=bidask, com=com) / ideal_strategy(s_p[i:], sdol=dollars, sshares=shares, bidask=bidask, com=com)
    
#    #remove precision errors
#    buyval[buyval<1e-12]=0 
#    sellval[sellval<1e-12]=0 
#    
#    #regulate output (bring mean and median closer to max)
#    buyval=buyval/(20.0*buyval+1.0)
#    sellval=sellval/(20.0*sellval+1.0)
        
    combval=np.zeros(len(s_p)-3)
    for i,val in enumerate(outar[:,0]):
        if val==1.:
            combval[i] = 1./outar[i,1]
        elif outar[i,1]==1.:
            combval[i] = val
        else:
            combval[i] = 1.0
    return combval
#    return outar
        
def trade_val(stock_price, trades, sdol, bidask, com):
    dollars=sdol
    shares=0
    value=list()   
    
    askmul=1.0+bidask/2.0
    bidmul=1.0-bidask/2.0
    
    for i,sp in enumerate(stock_price):
        if trades[i]>1.0 and dollars>=(stock_price[i]*askmul+com):
            newshares=int((dollars-com)/(stock_price[i]*askmul))
            shares+=newshares
            dollars-=newshares*(stock_price[i]*askmul)+com
        elif trades[i]<1.0 and shares>0:
            dollars+=shares*(stock_price[i]*bidmul)-com
            shares=0
        value.append(dollars+shares*stock_price[i]*bidmul-np.sign(shares)*com)
    
    return value
    
def ideal_strategyOrg(stock_price, sdol, bidask, com):
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
    
def trade_abs(stock_price, trades, sdol, bidask, com):
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
    
#def trade(stock_price, trades, sdol, bidask, com):
#    dollars=sdol
#    shares=0
#    value=list()   
#    
#    askmul=1.0+bidask/2.0
#    bidmul=1.0-bidask/2.0
#    
#    for i,sp in enumerate(stock_price):
#        if i < len(stock_price)-1:
#            if trades[i]>0.1 and dollars>=(stock_price[i]*askmul+com) and trades[i+1]<0.1:
#                newshares=int((dollars-com)/(stock_price[i]*askmul))
#                shares+=newshares
#    #            print 'if', shares, dollars
#                dollars-=newshares*(stock_price[i]*askmul)+com
#            elif trades[i]<-0.1 and shares>0 and trades[i+1]>-0.1:
#                dollars+=shares*(stock_price[i]*bidmul)-com
#    #            print 'elif', shares, dollars
#                shares=0
#        else:
#            if trades[i]>0.1 and dollars>=(stock_price[i]*askmul+com):
#                newshares=int((dollars-com)/(stock_price[i]*askmul))
#                shares+=newshares
#    #            print 'if', shares, dollars
#                dollars-=newshares*(stock_price[i]*askmul)+com
#            elif trades[i]<-0.1 and shares>0:
#                dollars+=shares*(stock_price[i]*bidmul)-com
#    #            print 'elif', shares, dollars
#                shares=0
#        value.append(dollars+shares*stock_price[i])
##        print 'out', shares, dollars, value[-1], stock_price[i]
#    
#    return value
    
def trade(stock_price, trades, sdol, bidask, com):
    dollars=sdol
    shares=0
    value=list()   
    
    askmul=1.0+bidask/2.0
    bidmul=1.0-bidask/2.0
    
    for i,sp in enumerate(stock_price):
        if trades[i]>0.25 and dollars>=(stock_price[i]*askmul+com):
            newshares=int((dollars-com)/(stock_price[i]*askmul))
            shares+=newshares
#            print 'if', shares, dollars
            dollars-=newshares*(stock_price[i]*askmul)+com
        elif trades[i]<-0.25 and shares>0:
            dollars+=shares*(stock_price[i]*bidmul)-com
#            print 'elif', shares, dollars
            shares=0
        value.append(dollars+shares*stock_price[i])
#        print 'out', shares, dollars, value[-1], stock_price[i]
    
    return value