################## NEURAL NET IMPORTS ##########################################
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from random import random, seed
from activation_functions import Activation
################## NEURAL NET IMPORTS ##########################################

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
            X_train, X_test, y_train, y_test, #data
            act_func,
            epochs = None, batchs = None, iterations = None, #GD variables
            hidden_layers = 1, hidden_neurons = [50],
            eta = 0.1, lambd = 0.001, #SGD hyperparameters
        ):

        if net_type not in ['CLASSIFICATION', 'REGRESSION']:
            raise TypeError("net_type must be either 'CLASSIFICATION', or 'REGRESSION'")
        self._type = net_type

        if (hidden_layers < 1):
            raise ValueError("Need at least 1 hidden layer. Use logistic_regression.py for no layers")
        if (len(hidden_neurons) != hidden_layers):
            raise ValueError("Need to specify number of neurons for each layer exactly, excluding output layer")
        if (len(act_func) != hidden_layers +1):
            raise ValueError("Need an activation function for each layer and an output activation function")

        self._type = type
        self._X = X_train
        self._y = y_train
        self._X_test = X_test
        self._y_test = y_test
        self._categories = self._y_train.shape[1]
        self._data_len = self._X.shape[0]

        self._hidden_layers = hidden_layers
        self._hidden_neurons = hidden_neurons
        self._eta = eta
        self._lambd = lambd

        self.activate = Activation(act_func)

        if ((batchs != None) and (epochs != None)):
            self._batchs = batchs
            self._epochs = epochs
            self._batch_size = self._data_len // self._batchs #size of minibatches
            self._condition = 'SGD'
        elif (iterations != None):
            self._n = iterations
            self._condition = 'brute force'

        def initialize(self):
            size_list = [self._X.shape[1]] + self._hidden_neurons + [self._categories]
            # size list = [f,n_0,n_1,n_2,..., n_l-1, c]
            # makes w_0 = [f,n_0], w_1 = [n_0,n_1], ... w_l = [n_l-1,c] where l is output layer
            self._bias = [np.random.randn(size,1) for size in size_list[1:]]
            self._weights = [np.random.randn(x, y) for x, y in zip(size_list[:-1],size_list[1:])]

        def feed_forward(self):

            z = np.matmul(self._X, self._weights[0]) + self._bias[0]
            a = self.activate.func[0](z)
            z_vals = [z]
            act_vals = [a]
            for i, b, w in zip(self.activate.func[1:], self._bias[1:], self._weights[1:]):
                z = np.matmul(a, w) + b
                a = self.activate.func[i](z)
                z_vals.append(z)
                act_vals.append(a)

        def back_propogate(self):

            # bias / weight for all layers
            grad_w = [np.zeros(w.shape) for w in self._weights]
            grad_b = [np.zeros(b.shape) for b in self._bias]

            # set up delta for prediction layer
            if (self._type == 'CLASSIFICATION'):
                delta = act_vals[-1] - self._y
            elif (self._type == 'REGRESSION'):
                deriv = self.activate.deriv[-1](z_vals[-1])
                delta = (act_vals[-1] - self._y) * deriv
            # output layer gradients
            grad_b[-1][:] = delta
            grad_w[-1][:] = np.matmul(act_vals[-2].T, delta)

            for l in reversed(range(0,self._hidden_layers)):
                deriv = self.activate.deriv[l](z_vals[l])
                temp = np.matmul(delta, self._weights[l+1].T)
                delta = np.multiply(temp, deriv)
                grad_b[l][:] = delta
            if (l > 0):
                grad_w[l][:] = np.matmul(act_vals[l-1].T, delta)
            elif (l = 0):
                grad_w[l][:] = np.matmul(self._X.T, delta)
            # UPDATE WEIGHTS AND BIASSSSSSSŠ

















    #
