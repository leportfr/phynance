import numpy as np
from numba import autojit

#@autojit
def ewma_strategy(stock_price, pred_ewma, sdol=1.0e5, bidask=0.0, com=0.0):
    dollars=sdol
    shares=0
    value=list()
    
    for i,pred in enumerate(pred_ewma):
        if pred>0 and dollars>(stock_price[i]*(1.0+bidask/200.0)):
            shares=int(dollars/(stock_price[i]*(1.0+bidask/200.0)))
#            print 'if', shares, dollars
            dollars-=shares*(stock_price[i]*(1.0+bidask/200.0))
            dollars-=com
        elif pred<0 and shares>0:
            dollars+=shares*(stock_price[i]*(1.0-bidask/200.0))
#            print 'elif', shares, dollars
            shares=0
            dollars-=com
#        print 'out', shares, dollars
        value.append(dollars+shares*stock_price[i])
    
    return value