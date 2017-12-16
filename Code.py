Jupyter Notebook
NNFL Project 2017

Python based Back Propogation Neural Network for Automatic Speech Recognition
Group No. 18


#Likhit Teja Valavala  -  2015A3PS0221P
#Shikhar Shiromani     -  2015A3PS0194P
#Pratyush Priyank      -  2015A3PS0188P

%%cmd
​
pip install jdc


# Library imports
import random
import numpy as np
import jdc
import sklearn
from datasets import *


from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import scipy


def pca(x,k):
    covar_x = np.dot(x.T,x)/x.shape[0]
    [U,S,V] = scipy.linalg.svd(covar_x)
    Z = np.dot(x,U[:,0:k])
    return Z


training_data = []
names = ["_jackson_","_theo_","_jason_"]
for k in range(0,2):
    for i in range(0,10):
        for j in range(0,40):
            string = str(i)+names[k]+str(j)+".wav"
            (rate,sig) = wav.read(string)
            mfcc_feat = mfcc(sig,rate,winlen=0.025,winstep=0.01,numcep=13,nfilt=26,nfft=1103,lowfreq=0,highfreq=None,preemph=0.97,
                    ceplifter=22,appendEnergy=True)  
            z = pca(mfcc_feat.T,1)
            training_data.append(z)
print(len(training_data))
800


outputs = []
for k in range(0,2):
    for i in range(0,10):
        temp = [0]*10
        temp[i] = 1
        temp = np.array([temp]).T
        for j in range(0,40):
            outputs.append(temp)
print(len(outputs))
800


test_data = []
names = ["_jackson_","_theo_","_jason_"]
for k in range(0,2):
    for i in range(0,10):
        for j in range(40,50):
            string = str(i)+names[k]+str(j)+".wav"
            (rate,sig) = wav.read(string)
            mfcc_feat = mfcc(sig,rate,winlen=0.025,winstep=0.01,numcep=13,nfilt=26,nfft=1103,lowfreq=0,highfreq=None,preemph=0.97,
                    ceplifter=22,appendEnergy=True)  
            z = pca(mfcc_feat.T,1)
            test_data.append(z)
print(len(test_data))

test_outputs = []
for k in range(0,2):
    for i in range(0,10):
        temp = [0]*10
        temp[i] = 1
        temp = np.array([temp]).T
        for j in range(0,10):
            test_outputs.append(temp)
print(len(test_outputs))

We define a generic neural network architecture as a python class which we would use in multiple exercies.

Note: We are using jdc to define each method of class Network in seperate cells. jdc follows the following syntax,

%%add_to #CLASS_NAME#
def dummy_method(self):


class Network(object):
​
    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network. For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.initialize_biases()
        self.initialize_weights()
Initialization
Initialize weights and biases
The biases and weights for the network are initialized randomly, using a Gaussian distribution with mean 0, and variance 1. Note that the first layer is assumed to be an input layer, and by convention we won't set any biases for those neurons, since biases are only ever used in computing the outputs from later layers.



%%add_to Network
def initialize_biases(self):
    
    self.biases = [np.random.uniform(-0.5,0.5,(self.sizes[b],1)) for b in range(1,self.num_layers)]
    self.delta_b = [np.zeros((self.sizes[b],1)) for b in range(1,self.num_layers)]


%%add_to Network
def initialize_weights(self):
    
    self.weights = [np.random.uniform(-0.5,0.5,(self.sizes[b],self.sizes[b-1])) for b in range(1,self.num_layers)]
    self.delta_w = [np.zeros((self.sizes[b],self.sizes[b-1])) for b in range(1,self.num_layers)]
Training
We shall implement backpropagation with stochastic mini-batch gradient descent to optimize our network.



%%add_to Network
def train(self, training_data, epochs, mini_batch_size, learning_rate,momentum):
    """Train the neural network using gradient descent.  
    ``training_data`` is a list of tuples ``(x, y)``
    representing the training inputs and the desired
    outputs.  The other parameters are self-explanatory."""
​
    # training_data is a list and is passed by reference
    # To prevernt affecting the original data we use 
    # this hack to create a copy of training_data
    # https://stackoverflow.com/a/2612815
    training_data = list(training_data)
    
    for i in range(epochs):
        # Get mini-batches    
        mini_batches = self.create_mini_batches(training_data, mini_batch_size)
        
        # Itterate over mini-batches to update pramaters   
        cost = sum(map(lambda mini_batch: self.update_params(mini_batch, learning_rate,momentum), mini_batches))
        
        # Find accuracy of the model at the end of epoch         
        acc = self.evaluate(training_data)
        
        if(i%100==0):
            print("Epoch {} complete. Total Accuracy: {}".format(i,acc))
Create mini-batches
Split the training data into mini-batches of size mini_batch_size and return a list of mini-batches.



%%add_to Network
def create_mini_batches(self, training_data, mini_batch_size):
    # Shuffling data helps a lot in mini-batch SGD
    random.shuffle(training_data)
    # YOUR CODE HERE
    mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,len(training_data),mini_batch_size)]
    return mini_batches
Update weights and biases
In [85]:

%%add_to Network
def update_params(self, mini_batch, learning_rate,momentum):
    """Update the network's weights and biases by applying
    gradient descent using backpropagation."""
    #print(mini_batch)
    # Initialize gradients     
    delta_b = [np.zeros(b.shape) for b in self.biases]
    delta_w = [np.zeros(w.shape) for w in self.weights]
    
    total_cost = 0
    
    for x,y in mini_batch:
        # Obtain the mean squared error and the gradients
        # with resepect to biases and weights 
        
        cost, del_b, del_w = self.backprop(x, y)
        
        # Add the gradients for each sample in mini-batch        
        delta_b = [nb + dnb for nb, dnb in zip(delta_b, del_b)]
        delta_w = [nw + dnw for nw, dnw in zip(delta_w, del_w)]
        
        total_cost += cost
​
    # Update self.biases and self.weights
    # using delta_b, delta_w and learning_rate 
    #Momentum
    self.delta_b = [(learning_rate*delta + momentum*db) for delta,db in zip(delta_b,self.delta_b)]
    self.biases = [b - (1 / len(mini_batch)) * db
                   for b, db in zip(self.biases, self.delta_b)]
    self.delta_w = [(learning_rate*delta + momentum*dw) for delta,dw in zip(delta_w,self.delta_w)]
    self.weights = [w - (1 / len(mini_batch)) * dw
                    for w, dw in zip(self.weights, self.delta_w)]
​
    return total_cost


%%add_to Network
def backprop(self, x, y):
    """Return arry containiing cost, del_b, del_w representing the
    cost function C(x) and gradient for cost function.  ``del_b`` and
    ``del_w`` are layer-by-layer lists of numpy arrays, similar
    to ``self.biases`` and ``self.weights``."""
    # Forward pass
    zs, activations = self.forward(x)
    
    # Backward pass     
    cost, del_b, del_w = self.backward(activations, zs, y)
​
    return cost, del_b, del_w
Activation Functions


%%add_to Network
def sigmoid(self, z):
    """The sigmoid function."""
    # YOUR CODE HERE
    return 1/(1+np.exp(-z))


%%add_to Network
def sigmoid_derivative(self, z):
    """Derivative of the sigmoid function."""
    # YOUR CODE HERE
    return self.sigmoid(z)*(1-self.sigmoid(z))
Forward propogration


%%add_to Network
def forward(self, x):
    """Compute Z and activation for each layer."""
    
    # list to store all the activations, layer by layer
    zs = []
    
    # current activation
    activation = x
    # list to store all the activations, layer by layer
    activations = [x]
    
    # Loop through each layer to compute activations and Zs  
    for b, w in zip(self.biases, self.weights):
        # YOUR CODE HERE
        # Calculate z
        # watch out for the dimensions of multiplying matrices
        #print(w)
        #print(activations[-1])
        z = np.dot(w,activations[-1])+b
        zs.append(z)
        # Calculate activation
        activation = self.sigmoid(z)
        activations.append(activation)
        
    return zs, activations
Loss Function
Logistic regression error and it's derivative



%%add_to Network
def lre(self, output_activations, y):
    """Returns mean square error."""
    return -(y*np.log(output_activations) + (1-y)*np.log(1-output_activations))


%%add_to Network
def lre_derivative(self, output_activations, y):
    """Return the vector of partial derivatives \partial C_x /
    \partial a for the output activations. """
    return -(y/output_activations - (1-y)/(1-output_activations))
Backward pass

%%add_to Network
def backward(self, activations, zs, y):
    """Compute and return cost funcation, gradients for 
    weights and biases for each layer."""
    # Initialize gradient arrays
    del_b = [np.zeros(b.shape) for b in self.biases]
    del_w = [np.zeros(w.shape) for w in self.weights]
    
    # Compute for last layer
    cost = self.lre(activations[-1], y)
    
    delta = self.lre_derivative(activations[-1],y)*self.sigmoid_derivative(zs[-1])
    #print(delta.shape)
    del_b[-1] = delta
    del_w[-1] = np.dot(delta, activations[-2].transpose())
    #print(del_w[-1].shape)
    
    # Loop through each layer in reverse direction to 
    # populate del_b and del_w   
    for l in range(2, self.num_layers):
        #print(delta.shape);print(self.sigmoid_derivative(activations[-l]).shape); print(np.dot(self.weights[-l+1].T,delta).shape)
        delta = np.dot(self.weights[-l+1].T,delta)*self.sigmoid_derivative(zs[-l])
        #print(delta.shape)
        del_b[-l] = delta
        del_w[-l] = np.dot(delta, activations[-l -1].transpose())
        #print(del_w[-l].shape)
    
    return cost, del_b, del_w


%%add_to Network
def evaluate(self, test_data):
    """Return the accuracy of Network. Note that the neural
    network's output is assumed to be the index of whichever
    neuron in the final layer has the highest activation."""
    test_results = [(np.argmax(self.forward(x)[1][-1]), np.argmax(y))
                    for (x, y) in test_data]
    return sum(int(x == y) for (x, y) in test_results) * 100 / len(test_results)

Showtime
Let's test our implementation on a bunch of datasets.



training_data = [sklearn.preprocessing.normalize(a) for a in training_data]
data = list(zip(training_data,outputs))
#print(data)
network = Network([13, 11,8, 10])
network.train(data,3001,100,1,0.2)
#network.evaluate(list(zip(test_data,test_outputs)))
predictions = list(map(lambda sample: np.argmax(network.forward(sample)[1][-1]), test_data))
#print(predictions)
Epoch 0 complete. Total Accuracy: 10.875
Epoch 100 complete. Total Accuracy: 71.625
Epoch 200 complete. Total Accuracy: 76.875
Epoch 300 complete. Total Accuracy: 77.75
Epoch 400 complete. Total Accuracy: 78.375
Epoch 500 complete. Total Accuracy: 78.125
Epoch 600 complete. Total Accuracy: 78.375
Epoch 700 complete. Total Accuracy: 78.5
Epoch 800 complete. Total Accuracy: 78.25
Epoch 900 complete. Total Accuracy: 79.5
Epoch 1000 complete. Total Accuracy: 78.5
Epoch 1100 complete. Total Accuracy: 79.625
Epoch 1200 complete. Total Accuracy: 79.625
Epoch 1300 complete. Total Accuracy: 79.125
Epoch 1400 complete. Total Accuracy: 79.625
Epoch 1500 complete. Total Accuracy: 79.625
Epoch 1600 complete. Total Accuracy: 79.5
Epoch 1700 complete. Total Accuracy: 79.25
Epoch 1800 complete. Total Accuracy: 79.125
Epoch 1900 complete. Total Accuracy: 79.625
Epoch 2000 complete. Total Accuracy: 79.125
Epoch 2100 complete. Total Accuracy: 79.0
Epoch 2200 complete. Total Accuracy: 79.5
Epoch 2300 complete. Total Accuracy: 79.0
Epoch 2400 complete. Total Accuracy: 79.5
Epoch 2500 complete. Total Accuracy: 79.125
Epoch 2600 complete. Total Accuracy: 79.5
Epoch 2700 complete. Total Accuracy: 79.5
Epoch 2800 complete. Total Accuracy: 79.5
Epoch 2900 complete. Total Accuracy: 79.75
Epoch 3000 complete. Total Accuracy: 79.625

Testing Accuracy : 
51.5


​
