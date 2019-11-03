##################### IMPORTS FOR LOG REG CLASS ################################
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from random import random, seed
from activation_functions import Activation
##################### IMPORTS FOR LOG REG CLASS ################################

class LogisticRegression:
    '''
    ----------------- INPUTS -----------------
    X_train = N x M design matrix

    y_train = N x 1 results vector for sigmoid $BINARY FORM$
           = N x D for softmax (D is number of categories) $ONE HOT FORM$

    X_test = Q x M design matrix

    y_test = Q x 1 or Q x D target vector

    activation = activation function to be used:
                    ['SIGMOID', 'SOFTMAX', 'TANH', 'RELU']

    eta = static learning rate

    epochs = epochs for SGD

    batchs = number of minibatches

    Data is to be preprossed before creating Regression object (cleaning, normalizing, splitting)
    '''


    def __init__(self, X_train, y_train, X_test, y_test, act_func, iterations = None,
                 eta = 0.1, lambd = 0, epochs = None, batchs = None,):

        self._X = X_train
        self._y = y_train
        self._total_data_len = self._X.shape[0]
        self._X_test = X_test
        self._y_test = y_test
        self._beta = np.random.randn(self._X.shape[1], self._y.shape[1])
        self._lam = lambd
        self._eta = eta
        self.activation = Activation(act_func)

        if ((batchs != None) and (epochs != None)):
            self._batchs = batchs
            self._epochs = epochs
            self._batch_size = self._total_data_len // self._batchs #size of minibatches
            self._condition = 'SGD'
        else:
            self._n = iterations
            self._condition = 'brute force'

    def probability(self, a, b):
        return self.activation.func(np.matmul(a,b))

    def train(self):
        if (self._condition == 'brute force'):
            m = self._total_data_len
            for iter in range(self._n):
                prob = self.probability(self._X,self._beta)
                error = self._y - prob
                gradient = (-1/m)*(np.matmul(self._X.T, error))
                if (self._lam > 0.0):
                    gradient += self._lam*self._beta
                self._beta -= self._eta*gradient

        if (self._condition == 'SGD'):
            data_indices = np.arange(self._total_data_len)
            for i in range(self._epochs):
                for j in range(self._batchs):
                    pts_idx = np.random.choice(data_indices, self._batch_size, replace = False)
                    self._X_batch = self._X[pts_idx]
                    self._y_batch = self._y[pts_idx]

                    prob = self.probability(self._X_batch, self._beta)
                    error = self._y_batch - prob
                    gradient = (-1/self._batch_size)*np.matmul(self._X_batch.T, error)
                    if (self._lam >0):
                        gradient += self._lam*self._beta
                    self._beta -= self._eta * gradient


    def predict(self):
        self._predict = self.probability(self._X_test, self._beta)

    def score_one_hot(self):
        idx = np.argmax(self._predict, axis = 1)
        self._one_hot_score = np.zeros_like(self._predict)
        for i in range(self._one_hot_score.shape[0]):
            self._one_hot_score[i,idx[i]] = 1

    def score_binary(self):
        self._predict[np.where(self._predict > .5)] = 1
        self._predict[np.where(self._predict != 1)] = 0

def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    return onehot_vector

def accuracy_metric(a,b):
    return accuracy_score(a,b)


if __name__ == '__main__':
    ####################### DATA PROCESSING IMPORTS ################################
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler

    ####################### DATA PROCESSING IMPORTS ################################


    ######################################################### MULTI CLASS ##########################################################
    #import MNSIT DATA
    # digits = datasets.load_digits()
    #
    # inputs = digits.images
    # labels = digits.target
    #
    # n_inputs = len(inputs)
    # inputs = inputs.reshape(n_inputs, -1) #reshape len, 64 from len,8,8
    #
    # labels = to_categorical_numpy(labels)
    # X_train, X_test, y_train, y_test = train_test_split(inputs,labels,train_size=0.99,test_size=0.01)
    #
    # categorical = LogisticRegression(X_train, y_train, X_test, y_test,  'SOFTMAX', iterations = 100000,eta = 0.0001,)
    # categorical.train()
    # accuracy, onehot, pred = categorical.predict_softmax()
    # np.set_printoptions(precision=4)
    # print(pred)
    #
    # print(accuracy)

    ################################################ BINARY ###############################################################
    def load_data(path, header):
        marks_df = pd.read_csv(path, header=header)
        return marks_df
    # load the data from the file
    data = load_data("/Users/douglas/Fall_2019/filemovement/Machine-Learning/LogisticRegression/data/marks.txt", None)


    ##### THIS GIVES GOOD RESULTS SIGMOID#####
    # # X = feature values, all the columns except the last column
    # X = data.iloc[:, :-1]
    # # y = target values, last column of the data frame
    # y = data.iloc[:, -1]
    # # filter out the applicants that got admitted
    # admitted = data.loc[y == 1]
    # # filter out the applicants that din't get admission
    # not_admitted = data.loc[y == 0]
    # # data setup. Add column of ones to X
    # X = np.c_[np.ones((X.shape[0], 1)), X]
    # y = y[:, np.newaxis]
    # X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,test_size=0.2)

    # binary = LogisticRegression(X_train, y_train, X_test, y_test, 'SIGMOID', eta = 0.001, lambd = 0.001, epochs = 10000, batchs = 20)
    # binary.train()
    # accuracy, prediction = binary.predict_sigmoid()
    # results = np.c_[prediction, y_test]
    # print(accuracy)
    # print(results)

    ##################################

    # X = feature values, all the columns except the last column
    X = data.iloc[:, :-1]
    # y = target values, last column of the data frame
    y = data.iloc[:, -1]
    # filter out the applicants that got admitted
    admitted = data.loc[y == 1]
    # filter out the applicants that din't get admission
    not_admitted = data.loc[y == 0]

    # data setup. DO NOT add 1's since we are using softmax
    X = np.array(X)
    y = to_categorical_numpy(y)

    # TRAINING AND TEST DATA
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,test_size=0.2)

    # SCALE DATA
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    x_test = scaler.transform(X_test)

    eta_vec = np.logspace(-5, 1, 7)
    lam_vec = np.logspace(-5, 1, 7)
    test_accuracy = np.zeros((len(eta_vec), len(lam_vec)))

    for i, eta in enumerate(eta_vec):
        for j, lam in enumerate(lam_vec):
            binary = LogisticRegression(X_train, y_train, X_test, y_test, 'SOFTMAX', eta = eta,
                                        lambd = lam, epochs = 100, batchs = 20)
            binary.train()
            binary.predict()
            binary.score_one_hot()
            accuracy = accuracy_metric(binary._one_hot_score, y_test)
            test_accuracy[i][j] = accuracy

    import seaborn as sns
    sns.set()
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Test Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()

    # np.set_printoptions(precision=4)
    # print('BEFORE SOFTMAX= \n', binary._predict)
    # print('prediction accuracy= ', accuracy)
    # print('predicted catagories= \n', binary._one_hot_score)
    # print('true catgeories= \n', y_test)

    ################################################ BINARY ###############################################################























    # if __name__ == "__main__":
    #
    #
    #
    #     # plots
    #     plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
    #     plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')
    #     plt.legend()
    #     plt.show()
