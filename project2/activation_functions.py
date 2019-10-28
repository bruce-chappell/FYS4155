import numpy as np

class Activation:

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exps = np.exp(x - x.max(axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return np.maximum(x, 0)

    def d_sigmoid(self, x):
        return (1 - x) * x

    def d_softmax(self, x):
        dx_ds = np.diag(x) - np.dot(x, x.T)
        return dx_ds.sum(axis=0).reshape(-1, 1)

    def d_tanh(self, x):
        return 1 - x*x

    def d_relu(self, x):
        return 1 * (x > 0)

    def __init__(self, activation):
        funcs = {
            'SIGMOID' : self.sigmoid,
            'SOFTMAX' : self.softmax,
            'TANH'    : self.tanh,
            'RELU'    : self.relu,
        }
        derivs = {
            'SIGMOID' : self.d_sigmoid,
            'SOFTMAX' : self.d_softmax,
            'TANH'    : self.d_tanh,
            'RELU'    : self.d_relu,
        }
        # if activation not in funcs:
        #     raise TypeError("\nActivation function must be either:\n'SIGMOID'\n'SOFTMAX'\n'TANH'\n'RELU'")
        self.func = {}
        self.deriv = {}
        for i in range(len(activation)):
            self.func[i] = funcs[activation[i]]
            self.deriv[i] = derivs[activation[i]]
        return
















        #
