import cPickle as pickle
import lstm
import numpy as py

openfile = open('C:\Users\leportfr\Desktop\Phynance\outPickle','rb')

paramList = list()
try:
    while True: 
        paramList.append(pickle.load(openfile))
except EOFError:
    pass
#paramList.append(pickle.load(openfile))
print len(paramList)
