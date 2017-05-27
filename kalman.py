import numpy as np 
import numba
from dataload import loadData

#plt.rcParams['figure.figsize'] = (10, 8)

# intial parameters

@numba.jit(nopython=True)
def kalman(z, Rfact=0.05, Qfact=0.022):
    n_iter = len(z)
    sz = (n_iter,) # size of array
    
#    Q = 1e-5 # process variance
#    R = 0.1**2 # estimate of measurement variance, change to see effect

# allocate space for arrays
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    xhatminus=np.zeros(sz) # a priori estimate of x
    Pminus=np.zeros(sz)    # a priori error estimate
    K=np.zeros(sz)         # gain or blending factor

    # intial guesses
    xhat[0] = 0.0
    P[0] = 1.0

    for k in range(1,n_iter):
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1]+(Qfact*z[k])**2
    
        # measurement update
        K[k] = Pminus[k] / (Pminus[k]+(Rfact*z[k])**2)
        xhat[k] = xhatminus[k] + K[k] * (z[k]-xhatminus[k])
        P[k] = (1-K[k]) * Pminus[k]
        
    return xhat#, Pminus

#x = 100 # truth value (typo in example at top of p. 13 calls this z)
#z = np.random.normal(x,0.1,size=50) # observations (normal about x, sigma=0.1)

if __name__ == '__main__':
    import matplotlib.pyplot as plt      
    
    df = loadData()
    dfarrayclose = np.array(df.loc[:,:,'close']).T
    z = dfarrayclose[1,:].copy()
    dfarrayclose[1] = kalman(dfarrayclose[1])
    print len(z)
    
    xhat = kalman(z)
    
    plt.figure()
    plt.plot(z,'k+',label='noisy measurements')
    plt.plot(dfarrayclose[1],'b-',label='a posteri estimate')
    #plt.axhline(x,color='g',label='truth value')
    plt.legend()
    plt.title('Estimate vs. iteration step', fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('Voltage')
    
#    plt.figure()
#    valid_iter = range(1,len(z)) # Pminus not valid at step 0
#    plt.plot(valid_iter,Pminus[valid_iter],label='a priori error estimate')
#    plt.title('Estimated $\it{\mathbf{a \ priori}}$ error vs. iteration step', fontweight='bold')
#    plt.xlabel('Iteration')
#    plt.ylabel('$(Voltage)^2$')
#    plt.setp(plt.gca(),'ylim',[0,.01])
#    plt.show()