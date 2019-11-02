################## NEURAL NET IMPORTS ##########################################
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from random import random, seed
from activation_functions import Activation
################## NEURAL NET IMPORTS ##########################################

np.set_printoptions(precision = 5)
np.set_printoptions(linewidth = 400)


class Neural_Network:
    '''
    ---------------------- INPUTS ---------------------------
    net_type: specify between 'CLASSIFICATION' or 'REGRESSION'

    data: preprocessed X_train, X_test, y_train, y_test

    activation: specify [activation 1, activation 2, ... output_activation]

    stochastic gradient decsent variables: epochs and batchs for SGD back propogation

    hidden_layers: specify number of hidden layers for net

    hidden_neurons: specify number of neurons in the hidden layers ex [a,b,c]
                    len(hidden_neurons) MUST match hidden_layers value

    hyper_parameters: learning rate and regularization parameter for SGD

    '''
    def __init__(
            self,
            net_type,
            X, y, #data
            act_func,
            epochs = None, batchsize = None, iterations = None, #GD variables
            hidden_layers = 1, hidden_neurons = [50],
            eta = 0.1, lambd = 0.001, #SGD hyperparameters
        ):

        if net_type not in ['SOFTMAX CROSS-ENTROPY', 'REGRESSION']:
            raise TypeError("net_type must be either 'SOFTMAX CROSS-ENTROPY', or 'REGRESSION'")
        self._type = net_type

        if (hidden_layers < 1):
            raise ValueError("Need at least 1 hidden layer. Use logistic_regression.py for no layers")
        if (len(hidden_neurons) != hidden_layers):
            raise ValueError("Need to specify number of neurons for each layer exactly, excluding output layer")
        if (len(act_func) != hidden_layers +1):
            raise ValueError("Need an activation function for each layer and an output activation function")

        self._type = net_type

        self._hidden_layers = hidden_layers
        self._hidden_neurons = hidden_neurons
        self._eta = eta
        self._lambd = lambd

        self._activate = Activation(act_func)
        self._act_functions = [a for a in self._activate.func.values()]
        self._act_derivatives = [a for a in self._activate.deriv.values()]

        if ((batchsize != None) and (epochs != None)):
            self._X_full = X
            self._y_full = y
            self._categories = self._y_full.shape[1]
            self._data_len = self._X_full.shape[0]
            self._epochs = epochs
            self._batch_size = batchsize #size of minibatches
            self._condition = 'SGD'
        elif (iterations != None):
            self._X = X
            self._y = y
            self._categories = self._y.shape[1]
            self._data_len = self._X.shape[0]
            self._n = iterations
            self._condition = 'brute force'

    def initialize(self):
        if self._condition == 'SGD':
            size_list = [self._X_full.shape[1]] + self._hidden_neurons + [self._categories]
        elif self._condition == 'brute force':
            size_list = [self._X.shape[1]] + self._hidden_neurons + [self._categories]
        # size list = [f,n_0,n_1,n_2,..., n_l-1, c]
        # makes w_0 = [f,n_0], w_1 = [n_0,n_1], ... w_l = [n_l-1,c] where l is output layer
        self._bias = [0.01*np.random.randn(s) for s in size_list[1:]]
        self._weights = [0.01*np.random.randn(x, y) for x, y in zip(size_list[:-1],size_list[1:])]


    def feed_forward_train(self, X):
        a = np.zeros(self._hidden_layers + 2, dtype = np.ndarray)
        z = np.zeros(self._hidden_layers + 2, dtype = np.ndarray)
        a[0] = X
        print('start', a[0].shape)
        z[0] = 0
        for i in range(self._hidden_layers +1):
            z[i+1] = a[i] @ self._weights[i] + self._bias[i]
            print('z', z[i+1].shape)
            a[i+1] = self._act_functions[i](z[i+1])
        return z, a

    def back_propogate(self, X, y):

        delta = np.zeros(self._hidden_layers +1, dtype = np.ndarray)
        # create list to hold gradients Weights and Bias
        grad_w = np.zeros_like(delta)
        grad_b = np.zeros_like(delta)

        z, a = self.feed_forward_train(X)
        # set up delta for prediction layer
        if (self._type == 'SOFTMAX CROSS-ENTROPY'):
            delta[-1] = a[-1] - y

        elif (self._type == 'REGRESSION'):
            deriv = self._act_derivatives[-1](z[-1])
            delta[-1] = (a[-1] - y) * deriv

        # output layer gradients
        grad_b[-1] = np.sum(delta[-1], axis = 0)
        grad_w[-1] = a[-2].T @ delta[-1]

        # hidden layer gradients
        for l in reversed(range(0, self._hidden_layers)):
            delta[l] = (delta[l+1] @ self._weights[l+1].T) * self._act_derivatives[l](z[l+1])
            grad_b[l] = np.sum(delta[l], axis=0)
            grad_w[l] = a[l].T @ delta[l]

        # update weights and biases
        for i in reversed(range(0, self._hidden_layers +1)):
            grad_w[i] += self._lambd * self._weights[i]
            self._weights[i] -= self._eta * grad_w[i]
            self._bias[i] -= self._eta * grad_b[i]

    def train(self, X, y):
        self.initialize()
        batches = self._data_len // self._batch_size
        if (self._condition == 'brute force'):
            X = self._X_full
            y = self._y_full
            for iter in range(self._n):
                self._eta = self._eta / self._data_len
                self.back_propogate(X, y)
        elif (self._condition == 'SGD'):
            data_indices = np.arange(self._data_len)
            for i in range(self._epochs):
                batch_idx = np.array_split(np.random.permutation(self._data_len), batches)
                for j in range(batches):
                    random_batch = np.random.randint(batches)
                    self._eta = self._eta / self._batch_size
                    X = self._X_full[batch_idx[random_batch]]
                    y = self._y_full[batch_idx[random_batch]]
                    self.back_propogate(X, y)

    def feed_out_softmax(self, X):
        # just need last a matrix which is probabilities from output layer
        #outputs probabilites and on_hot vec
        a = np.zeros(self._hidden_layers + 2, dtype = np.ndarray)
        z = np.zeros(self._hidden_layers + 2, dtype = np.ndarray)

        a[0] = X
        z[0] = 0

        for i in range(self._hidden_layers +1):
            z[i+1] = a[i] @ self._weights[i] + self._bias[i]
            a[i+1] = self._act_functions[i](z[i+1])
        probabilities = a[-1]
        idx = np.argmax(probabilities, axis = 1)
        prediction = np.zeros_like(probabilities)
        for i in range(prediction.shape[0]):
            prediction[i, idx[i]] = 1
        return probabilities, prediction


    def feed_out_regression(self, X):
        # just need last a matrix which is probabilities from output layer
        a = np.zeros(self._hidden_layers + 2, dtype = np.ndarray)
        z = np.zeros(self._hidden_layers + 2, dtype = np.ndarray)

        a[0] = X
        z[0] = 0
        for i in range(self._hidden_layers +1):
            z[i+1] = a[i] @ self._weights[i] + self._bias[i]
            a[i+1] = self._act_functions[i](z[i+1])
        return z[-1], a[-1]

def accuracy(a,b):
    return accuracy_score(a,b)

def score_binary(a):
    a[np.where(a > .5)] = 1
    a[np.where(a != 1)] = 0
    return a


########################## MNSIT DATA ################################################
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd

#import MNSIT DATA
digits = datasets.load_digits()
inputs = digits.images
labels = digits.target
labels = labels.reshape(len(labels),1)
n_inputs = len(inputs)
inputs = inputs.reshape(n_inputs, -1) #reshape len, 64 from len,8,8
enc = OneHotEncoder(sparse = 'False', categories = 'auto')
labels = enc.fit_transform(labels).toarray()

################################ BINARY TEST SET ################################
def load_data(path, header):
    marks_df = pd.read_csv(path, header=header)
    return marks_df
# load the data from the file
data = load_data("/Users/douglas/Fall_2019/filemovement/Machine-Learning/LogisticRegression/data/marks.txt", None)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
admitted = data.loc[y == 1]
not_admitted = data.loc[y == 0]
# data setup. DO NOT add 1's since we are using softmax
X = np.array(X)
y = np.array(y)
y = y.reshape(len(y),1)
y = enc.fit_transform(y).toarray()

scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,test_size=0.2)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


test = Neural_Network('SOFTMAX CROSS-ENTROPY', X_train, y_train,
                          ['SIGMOID',  'SOFTMAX'], epochs = 1000, batchsize = 10, iterations = None, hidden_layers =1, hidden_neurons = [3], eta =0.00001, lambd = 0.001)
test.train(X_train, y_train)
prob, pred = test.feed_out_softmax(X_train)
print(prob[:10])
print(pred)

print(accuracy(pred,y_train))

















    #
