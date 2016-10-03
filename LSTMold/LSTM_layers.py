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

# input variables
alpha = .5
input_dim = 1
hidden_dim = 10
output_dim = 1


# initialize neural network weights
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

#training set
inputData = list()
inputData.append(1.0)
inputData.append(2.0)
for j in range(2000):
    inputData.append(inputData[-1]+j*j%1000-(j/2.0)%100)
#    inputData.append(np.random.random(1))
scaleFactor=1.0/max(inputData)/2.0
inputData=np.array(inputData)

# training logic
layer_1_values = list()
layer_1_values.append(np.zeros(hidden_dim))
overallErrorList = list()
for j in np.arange(0,inputData.size-1)+1:
    # generate a simple addition problem (a + b = c)
    a = inputData[j-1:j]*scaleFactor

    # true answer
    c = inputData[j:j+1]*scaleFactor
    
    # where we'll store our best guess (binary encoded)
    d = np.zeros_like(c)

    overallError = 0
    
    layer_2_deltas = list()
#    layer_1_values = list()
#    layer_1_values.append(np.zeros(hidden_dim))
    
    # moving along the positions in the binary encoding
    binary_dim = int(1)
    for position in range(binary_dim):
        # generate input and output
        X = np.array([[a[binary_dim - position - 1]]])
#        print 'X is',X
        y = np.array([[c[binary_dim - position - 1]]]).T
#        print 'Y is',y

        # hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))
#        print 'layer_1 is',layer_1
#        print 'layer_1_values is',layer_1_values

        # output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))
#        print 'layer_2 is',layer_2

        # did we miss?... if so, by how much?
        layer_2_error = y - layer_2
#        print 'layer_2_error is',layer_2_error
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])
        overallErrorList.append(overallError/scaleFactor)
    
        # decode estimate so we can print it out
        d[binary_dim - position - 1] = layer_2[0][0]
        
        # store hidden layer so we can use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1))
    
    future_layer_1_delta = np.zeros(hidden_dim)
    
    for position in range(binary_dim):
        
        X = np.array([[a[position]]])
        layer_1 = layer_1_values[-position-1]
        prev_layer_1 = layer_1_values[-position-2]
        
        # error at output layer
        layer_2_delta = layer_2_deltas[-position-1]
        # error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

        # let's update all our weights so we can try again
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)
        
        future_layer_1_delta = layer_1_delta
    

    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha    

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0
    
#    print out progress
    if(j % 100 == 0):
        print "Error:" + str(overallError/scaleFactor)
        print "Pred:" + str(d/scaleFactor)
        print "True:" + str(c/scaleFactor)
        
    plt.plot(overallErrorList)
        