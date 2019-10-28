import numpy as np
from random import random,seed
from sklearn.metrics import accuracy_score
np.set_printoptions(precision=5)
class test:
    def addition(self,a,b):
        return a+b

    def subtraction(self,a,b):
        return a-b

    def __init__(self, a, b, activation):
        self._a = a
        self._b = b
        funcs = {
            'ADD' : self.addition,
            'SUB' : self.subtraction,
        }
        self.act = activation
        self.func = funcs[activation]

    def do_something(self):
        return self.func(self._a,self._b)

    def test(self):
        prob = self.do_something()
        return prob

# a = test(5,4,'ADD')
# print(a.test())
# a = test(5,4,'SUB')
# print(a.test())

def softmax(x):
    a = x - x.max(axis = 1, keepdims=True)
    print(a)
    exps = np.exp(x - x.max(axis = 1, keepdims=True))
    print(x)
    print(exps)
    return exps / np.sum(exps, axis = 1, keepdims = True)
a =np.array([[602828.9818, 700205.9454],
 [455042.4803, 524890.7496],
 [638261.6227, 734769.3836],
 [321969.1879, 369567.3693],
 [470256.2293, 547991.4172],
 [396282.0646, 455444.3794],
 [406223.6946, 476943.3211],
 [494795.1487, 566624.7932],
 [600901.1682, 694344.0898],
 [668588.7517, 769910.1888],
 [521516.9053, 602170.1022],
 [290539.3458, 332947.2701],
 [692838.3382, 800341.4385],
 [393571.4039, 453832.22  ],
 [415916.9111, 480570.2009],
 [493688.0698, 577279.7303],
 [628942.4998, 724183.851 ],
 [506616.7341, 573486.6801],
 [544163.1841, 617900.8305],
 [507506.1458, 595647.4453]])

# from activation_functions import Activation
# s = ['SIGMOID','TANH','RELU']
# a = Activation(s)
# print(a.func)
# for options in a.func:
#     b = a.func[options](2)
#     print(b)

# X=np.zeros((1000,64))
# hidden_neurons = 50
# hidden_layers = 3
# w = np.array([np.random.randn(X.shape[1],hidden_neurons) for i in range(hidden_layers+1)])
# print(w.shape)
#
# a = np.zeros((4,3,3))
# b =np.random.randn(1,3,3)
#
# a = np.r_[a,b]
# print(a)
sizes = [3,4,5,6]
a = [np.zeros((y,x)) for x, y in zip(sizes[:-1], sizes[1:])]
print(a)




#
