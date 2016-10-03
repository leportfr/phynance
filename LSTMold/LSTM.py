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
    return (a*(x.T-b)+0.2).T
    
def rescale(a,b,y):
    return b+(y-0.2)/a

class LSTM:
    def __init__(self,alpha=0.5,inputDim=1,hiddenDim=20,hiddenLayers=1):
        # input variables
        self.alpha = alpha
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.hiddenLayers = hiddenLayers
        
        # initialize neural network weights\
        self.synapseIn = 2*np.random.random((self.inputDim,self.hiddenDim)) - 1
        self.synapseOut = 2*np.random.random((self.hiddenDim,self.inputDim)) - 1
        self.synapseH = 2*np.random.random((self.hiddenLayers, self.hiddenDim,self.hiddenDim)) - 1
        self.synapseHidden = 2*np.random.random((self.hiddenLayers-1,self.hiddenDim,self.hiddenDim)) - 1
        
        self.errorList = list()
        self.predictionList = list()
        
    def train(self,inputData,dataMin,dataMax):
        scaleFactorA=0.6/(dataMax-dataMin)
        scaleFactorB=dataMin
        
        layerValues = list()
        for layer in range(self.hiddenLayers):
            layerValues.append(list())
            layerValues[-1].append(np.zeros(self.hiddenDim))
        
        for j in np.arange(0,inputData.shape[1]-1)+1:
            #forward propagate the network
            #generate input and output
            X = np.array([scale(scaleFactorA, scaleFactorB, inputData)[:,j-1]])
            y = np.array([scale(scaleFactorA, scaleFactorB, inputData)[:,j]])
        
            #hidden layer (input ~+ prev_hidden)
            for layer in range(self.hiddenLayers):
                if layer == 0:
                    layerValues[layer].append(sigmoid(np.dot(X,self.synapseIn) + np.dot(layerValues[layer][-1],self.synapseH[layer])))
                else:
                    layerValues[layer].append(sigmoid(np.dot(layerValues[layer-1][-1],self.synapseHidden[layer-1]) + np.dot(layerValues[layer][-1],self.synapseH[layer])))
        
            #output layer (new binary representation)
            outLayer = sigmoid(np.dot(layerValues[-1][-1],self.synapseOut))
        
            #did we miss?... if so, by how much? 
            outLayerError = y - outLayer
 
            self.errorList.append(outLayerError[0]/scaleFactorA)
        
            #store prediction
            self.predictionList.append(rescale(scaleFactorA, scaleFactorB, outLayer[0]))
            
            #backpropagate errors  
            outLayerDelta = (outLayerError)*sigmoid_output_to_derivative(outLayer)
            layerDeltas = list()
            for layer in range(self.hiddenLayers):
                if layer == 0:
                    layerDeltas.append((outLayerDelta.dot(self.synapseOut.T)) * sigmoid_output_to_derivative(layerValues[-1][-1]))
                else:
                    layerDeltas.append((layerDeltas[-1].dot(self.synapseHidden[self.hiddenLayers-layer-1].T)) * sigmoid_output_to_derivative(layerValues[self.hiddenLayers-layer][-1]))
            layerDeltas.reverse()
            #update all our weights so we can try again
            synapseOutUpdate = np.atleast_2d(layerValues[-1][-1]).T.dot(outLayerDelta)
            synapseHUpdate = np.zeros_like(self.synapseH)
            synapseHiddenUpdate = np.zeros_like(self.synapseHidden)
            for layer in range(self.hiddenLayers):
                if layer == 0:
                    synapseHUpdate[-layer] = np.atleast_2d(layerValues[-layer][-2]).T.dot(layerDeltas[-layer])
                else:
                    synapseHUpdate[-layer] = np.atleast_2d(layerValues[-layer][-2]).T.dot(layerDeltas[-layer])
                    synapseHiddenUpdate[-layer+1] = np.atleast_2d(layerValues[-layer][-2]).T.dot(layerDeltas[-layer])
            synapseInUpdate = X.T.dot(layerDeltas[0])
            
#            future_layer_1_delta = layer_1_delta
            
            self.synapseIn += synapseInUpdate * self.alpha
            self.synapseOut += synapseOutUpdate * self.alpha
            self.synapseH += synapseHUpdate * self.alpha
            self.synapseHidden += synapseHiddenUpdate * self.alpha
            
            #    print out progress
            if(j % 100 == 0):
                print "Error: " + str(self.errorList[-1])
                print "Pred: " + str(self.predictionList[-1])
                print "True: " + str(rescale(scaleFactorA, scaleFactorB, y[0,0]))
    
if __name__ == "__main__":
    #training set
    inputData = list()
    inputData.append(1.0)
    for j in range(2000):
        inputData.append(inputData[-1]+j*j%1000-(j/2.0)%100)
    inputData2 = list()
    for j in range(len(inputData)):
        inputData2.append(inputData[j]%100)
    inputData=np.array([np.array(inputData),np.array(inputData2)])
    
    lstm = LSTM(alpha=0.5,inputDim=inputData.shape[0],hiddenDim=50,hiddenLayers=5)
    lstm.train(inputData,np.zeros(inputData.shape[0]),np.max(inputData,axis=1))
    
    print np.array(lstm.errorList)[:,0]/np.array(lstm.predictionList)[:,0]
    print np.array(lstm.errorList)[:,1]/np.array(lstm.predictionList)[:,1]
    
    f,axarr = plt.subplots(2,sharex=True)
    for j in range(inputData.shape[0]):
        print np.array(lstm.predictionList)[:,j].shape
        axarr[0].plot(inputData[j,1:])
        axarr[0].plot(np.array(lstm.predictionList)[:,j])
        axarr[1].plot(np.array(lstm.errorList)[:,j]/np.array(lstm.predictionList)[:,j])
        