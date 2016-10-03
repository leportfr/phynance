import copy, numpy as np
import matplotlib.pyplot as plt
np.random.seed(10)

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)
    
def scale(a,b,x):
    return a*(x-b)+0.2
    
def rescale(a,b,y):
    return b+(y-0.2)/a

class LSTM:
    def __init__(self,alpha=0.5,hiddenDim=20):
        # input variables
        self.alpha = alpha
        self.inputDim = 1
        self.hiddenDim = hiddenDim
        self.outputDim = 1
        
        # initialize neural network weights
        self.synapse_0 = 2*np.random.random((self.inputDim,self.hiddenDim)) - 1
        self.synapse_1 = 2*np.random.random((self.hiddenDim,self.outputDim)) - 1
        self.synapse_h = 2*np.random.random((self.hiddenDim,self.hiddenDim)) - 1
        
        self.errorList = list()
        self.predictionList = list()
        
    def train(self,inputData,dataMin,dataMax):
        scaleFactorA=0.6/(dataMax-dataMin)
        scaleFactorB=dataMin
        
        layer_1_values = list()
        layer_1_values.append(np.zeros(self.hiddenDim))
        
        for j in np.arange(0,inputData.size-1)+1:
            # generate inputs and outputs
            a = scale(scaleFactorA, scaleFactorB, inputData[j-1:j])
            c = scale(scaleFactorA, scaleFactorB, inputData[j:j+1])
            
            #forward propagate the network
            #generate input and output
            X = np.array([[a[0]]])
            y = np.array([[c[0]]]).T
        
            #hidden layer (input ~+ prev_hidden)
            layer_1 = sigmoid(np.dot(X,self.synapse_0) + np.dot(layer_1_values[-1],self.synapse_h))
            #store hidden layer so we can use it in the next timestep
            layer_1_values.append(layer_1)
        
            #output layer (new binary representation)
            layer_2 = sigmoid(np.dot(layer_1,self.synapse_1))
        
            #did we miss?... if so, by how much?
            layer_2_error = y - layer_2
            overallError = np.abs(layer_2_error[0])
            self.errorList.append(overallError[0]/scaleFactorA)
        
            #store prediction
            self.predictionList.append(rescale(scaleFactorA, scaleFactorB, layer_2[0][0]))
            
#            future_layer_1_delta = np.zeros(self.hiddenDim)
            
            #backpropagate errors  
            layer_2_delta = (layer_2_error)*sigmoid_output_to_derivative(layer_2)
            layer_1_delta = (layer_2_delta.dot(self.synapse_1.T)) * sigmoid_output_to_derivative(layer_1)
            #update all our weights so we can try again
            synapse_1_update = np.atleast_2d(layer_1).T.dot(layer_2_delta)
            synapse_h_update = np.atleast_2d(layer_1_values[-2]).T.dot(layer_1_delta)
            synapse_0_update = X.T.dot(layer_1_delta)
            
#            future_layer_1_delta = layer_1_delta
            
            self.synapse_0 += synapse_0_update * self.alpha
            self.synapse_1 += synapse_1_update * self.alpha
            self.synapse_h += synapse_h_update * self.alpha    
            
            #    print out progress
            if(j % 100 == 0):
                print "Error: " + str(self.errorList[-1])
                print "Pred: " + str(self.predictionList[-1])
                print "True: " + str(rescale(scaleFactorA, scaleFactorB, c[0]))
    
if __name__ == "__main__":
    #training set
    inputData = list()
    inputData.append(1.0)
    inputData.append(2.0)
    for j in range(2000):
        inputData.append(inputData[-1]+j*j%1000-(j/2.0)%100)
    #    inputData.append(np.random.random(1))
    inputData=np.array(inputData)
    
    lstm = LSTM(alpha=0.5,hiddenDim=20)
    lstm.train(inputData,0,max(inputData))
    
    f,axarr = plt.subplots(2,sharex=True)
    axarr[0].plot(inputData)
    axarr[0].plot(lstm.predictionList)
    axarr[1].plot(np.divide(lstm.errorList,lstm.predictionList))
        