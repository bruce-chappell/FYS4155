import numpy as np
from random import *
from franke import FrankeFunction
from itertools import *

def create_data(points = 10000, noise = .01):

    x = np.random.random(size = points)
    y = np.random.random(size = points)
    z = FrankeFunction(x, y)

    if noise:
        z_noise = z + np.random.normal(0, noise, size = z.shape[0])

    return x, y, z, z_noise

class Regression_methods(object):
    '''
    Create a class to create objects for ridge and OLS Regression
    The class takes a design matrix A and a vector y as inputs
    from the equation of form:
                            A*beta = y
    It also allow you to set lambda if you are doing ridge
    '''
    def __init__(self, A, y, lamb = 0):
        try:
            A.shape
        except AttributeError:
            A = np.array(A)
        try:
            y.shape
        except AttributeError:
            y = np.array(y)
        if A.shape[0] != y.shape[0]:
            msg = 'Matrix and Vector must have compatable dimensions for matrix math'
            raise Exception(msg)

        self._A = A
        self._y = y
        self._lamb = lamb
        self._hessian = A.T @ A

    @property
    def beta(self):
        '''
        solve linear system A*beta = y for beta
        '''
        try:
            return self._beta
        except:
            if (self._lamb == 0 or self._lamb == None):
                self._beta = np.linalg.pinv(self._hessian) @ self._A.T @ self._y
                return self._beta

            else:
                n = self._hessian.shape[0]
                self._beta = np.linalg.inv(self._hessian + self._lamb*np.eye(n)) @ self._A.T @ self._y
                return self._beta

    @property
    def y_tilde(self):
        '''
        make prediction of y_tilde using beta
        '''
        try:
            return self._y_tilde
        except:
            self._y_tilde = self._A @ self.beta
            return self._y_tilde

    @property
    def sigma_y_sqr(self):
        '''
        sigma^2 for calculating variance of beta
        '''
        n = self._y.size
        p = self._hessian.shape[0]
        self._sigma_y_sqr = 1/(n-p-1) * np.sum((self._y-self.y_tilde)**2)
        return self._sigma_y_sqr

    @property
    def beta_var(self):
        '''
        Used for calculating beta variance for OLS
        '''
        try:
            return self._beta_var
        except:
            self._beta_var = np.linalg.inv(self._hessian) * self.sigma_y_sqr
            return self._beta_var

    @property
    def hessian(self):
        return self._hessian

    def R2score(self):
        self.y_tilde
        return 1 - np.sum((self._y - self.y_tilde) ** 2) / np.sum((self._y - np.mean(self._y)) ** 2)

    def ms_error(self):
        self.y_tilde
        return np.sum((self._y-self.y_tilde)**2)/np.size(self.y_tilde)


def no_resample_analysis(regression_object, method = 'ols', lamb = 0):
    '''
    takes a regression object and does a fit analysis printing:
        beta +- confidence interval
        R2
        MSE
    '''
    n = regression_object._y.shape[0]

    if (method == 'ols'):
        betavar = np.diag(regression_object.beta_var)
        sigma = regression_object.sigma_y_sqr
        mse = regression_object.ms_error()
        r2 = regression_object.R2score()

    if (method == 'ridge'):
        sigma = regression_object.sigma_y_sqr
        x = regression_object.hessian
        W = np.linalg.inv(x+lamb*np.eye(x.shape[0]))@x
        betavar = np.diag(sigma*W@np.linalg.inv(x)@W.T)
        mse = regression_object.ms_error()
        r2 = regression_object.R2score()

    print('NO RESAMPLE ANALYSIS')
    print('MSE: ', mse)
    print('R2 Score: ', r2)
    print('----Confidence Intervals for Selected Beta----')
    for i in range(len(betavar)):
        if i in np.arange(0,len(betavar)+1,2):
            print('\u03B2{:} = {:.3} +- {:.3}'.format(i, regression_object.beta[i], 1.645*np.sqrt(betavar[i]/n)))

    return betavar, sigma, mse, r2

def lasso_analysis(X, zdata, lamb):
    '''
    function to perform lasso regression on a given data set
    '''
    reg = Lasso(alpha = lamb, fit_intercept=False)
    reg.fit(X,zdata)
    beta = reg.coef_
    zpred = X @ beta

    err = ms_error(zdata, zpred)
    r2 = R2score(zdata, zpred)

    return err, r2

def data_split(x_data, y_data, z_data, j):
    '''
    my version of train_test_split from sklearn
    '''
    x_train=np.delete(x_data,j)
    y_train=np.delete(y_data,j)
    z_train=np.delete(z_data,j)
    x_test=np.take(x_data,j)
    y_test=np.take(y_data,j)
    z_test=np.take(z_data,j)

    return x_train, y_train, z_train, x_test, y_test, z_test


def build_design_matrix(x1_data, x2_data, order):

    '''
    takes in sorted x and y data and creates a design matrix  of specified polynomial order
    x and y must be the same shape
    '''

    if x1_data.shape[0] != x2_data.shape[0]:
        msg = "\n\nArguments <x1> and <x2> in function <build_design_matrix> must be of "
        msg += f"the same shape."
        raise Exception(msg)

    if len(x1_data.shape) > 1:
        x1_data = np.ravel(x1_data)
        x2_data = np.ravel(x2_data)

    exponents = list(product(range(0,order+1), repeat=2)) #creates tupples of all combos from 0 to order
    expo_sum = np.sum(exponents, axis = 1)
    valid_index = np.where(np.less_equal(expo_sum, order))[0] #collects index of sum(exponents) <= order
    exponents = np.array(exponents)
    exponents = exponents[valid_index] #only take in valid exponents

    design_matrix = np.zeros((x1_data.shape[0], exponents.shape[0]))
    for row in range(len(exponents)):
        for i in range(x1_data.shape[0]):
            design_matrix[i,row] = (x1_data[i]**exponents[row][0])*(x2_data[i]**exponents[row][1])
    return design_matrix

def R2score(z, zhat):
    return 1 - np.sum((z - zhat) ** 2) / np.sum((z - np.mean(zhat)) ** 2)

def ms_error(z, zhat):
    return np.sum((z - zhat)**2)/np.size(zhat)

def bias(z,zhat):
    return np.mean((z - zhat)**2)

def my_kfold(xdata, ydata, zdata, model_type, lamb = 0, order = 5, k=5):
    '''
    take in x, y, z vectors from data set.
    use model_type = 'ols', 'ridge', or 'lasso'
    use poly_order to set order of polynomial fit

    '''

    z_out = []
    x_out = []
    y_out = []

    error_test_local = 0
    error_train_local = 0
    r2_test_local = 0
    r2_train_local = 0

    error_global = 0
    bias_global = 0
    var_global = 0

    xtr_global, xtst_global, ytr_global, ytst_global, ztr_global, ztst_global = train_test_split(xdata,
                                                                                                 ydata,
                                                                                                 zdata,
                                                                                                 test_size=0.19)
    d_tst_global = build_design_matrix(xtst_global, ytst_global, order)
    #Hold a set of data permanently out of the training set and call it
    #our 'test global' set. We always predict and test with this set.

    idx_vec = np.arange(len(xtr_global))
    L = int(len(idx_vec)/k)
    np.random.shuffle(idx_vec)
    #randomize index to split our 'global training' data into local training and local test data

    Z = np.zeros((k, len(ztst_global)))
    #create vector to put our global predictions in

    for i in range(k):
        x_train, y_train, z_train, xtst_local, ytst_local, ztst_local = data_split(xtr_global,
                                                                                   ytr_global,
                                                                                   ztr_global,
                                                                                   idx_vec[i*L:(i+1)*L])

        d_train = build_design_matrix(x_train, y_train, order)
        d_tst_local = build_design_matrix(xtst_local, ytst_local, order)

        if (model_type == 'ols' or model_type == 'ridge'):
            reg_object = Regression_methods(d_train, z_train, lamb = lamb)
            beta = reg_object.beta
        if (model_type == 'lasso'):
            reg = Lasso(alpha = lamb,  max_iter = 10e6, tol = 0.01, fit_intercept = False)
            reg.fit(d_train, z_train)
            beta = reg.coef_

        z_pred_train = d_train @ beta
        z_pred_local = d_tst_local @ beta
        Z[i] = d_tst_global @ beta

        #train / test error scores for our local test/train sets
        error_test_local += ms_error(ztst_local, z_pred_local)
        error_train_local += ms_error(z_train, z_pred_train)
        r2_test_local += R2score(ztst_local, z_pred_local)
        r2_train_local += R2score(z_train, z_pred_train)

        #output vectors with predictions for each of our local test data points
        z_out = np.concatenate((z_out, z_pred_local), axis = 0)
        x_out = np.concatenate((x_out, xtst_local), axis = 0)
        y_out = np.concatenate((y_out, ytst_local), axis = 0)

    #analysis of our prediction using the global design matrix compared agains the global z values
    error_global = np.mean( np.mean((ztst_global - Z)**2, axis=0, keepdims=True) )
    bias_global = np.mean( (ztst_global - np.mean(Z, axis=0, keepdims=True))**2 )
    var_global = np.mean(np.var(Z,axis=0))

    data = np.zeros((z_out.size,3))
    data[:,0] = x_out
    data[:,1] = y_out
    data[:,2] = z_out

    return data, error_test_local/k, error_train_local/k, r2_test_local/k, r2_train_local/k, error_global, bias_global, var_global


def bias_variance(xdata, ydata, zdata, model_type, lamb = 0, poly_order = 5, k=5, fitplot = False,
                 bvplot = False, testtrainplot = False, printval = False):
    '''
    This function performs a regression analysis specified by:

            model_type: 'ols', 'ridge', 'lasso'

    While doing k = 5 kfold cross validation resampling
    This is done for polynomial fits from:

            polynomial order = 1 to poly_order

    The bias variance tradeoff is then plotted vs polynomial order
    '''
    vec_size = poly_order + 1
    polynomial_vec = np.arange(vec_size)

    error_test = np.zeros(vec_size)
    error_train = np.zeros(vec_size)
    r2vec = np.zeros(vec_size)

    error_global = np.zeros(vec_size)
    bias_ = np.zeros(vec_size)
    variance = np.zeros(vec_size)

    for i in polynomial_vec:
        data_out, error_test_, error_train_, r2, _, error_global_, biasval, varianceval = my_kfold(xdata,
                                                                                                   ydata,
                                                                                                   zdata,
                                                                                                   model_type,
                                                                                                   lamb = lamb,
                                                                                                   order = i,
                                                                                                   k=k)

        error_test[i] = error_test_
        error_train[i] = error_train_
        r2vec[i] = r2

        error_global[i] = error_global_
        bias_[i] = biasval
        variance[i] = varianceval

        #toggle print analysis in the window
        if printval:
            print('\n')
            print("polynomial degree: ", i)
            print('R2 test is: ', r2vec[i])
            print('Error test is: ', error_test[i])
            print('-------------------------------')
            print('Global error is: ', error_global[i])
            print("bias is: ", bias_[i])
            print("variance is: ", variance[i])
            print('{:.5} >= {:.5} + {:.5} = {:.5}'.format(error_global[i], bias_[i], variance[i], bias_[i] + variance[i]))
            print('\n')

        #plot predicted values for the whole 'local' data set
        if fitplot:
            fig1 = plt.figure(1)
            plt.scatter(data_out[:,0],data_out[:,1],c=data_out[:,2], cmap=cm.plasma)
            plt.show()

    #bias variance decomp
    if bvplot:
        fig2 = plt.figure(2)
        plt.title('Error Decomposition Plot')
        plt.xlabel('Polynomial Degree')
        plt.ylabel('MSE')
        plt.semilogy(polynomial_vec, error_global, label='Error')
        plt.semilogy(polynomial_vec, bias_, label='bias')
        plt.semilogy(polynomial_vec, variance, label='Variance')
        plt.legend()
        plt.show()

    #train vs test error for the 'local'
    if testtrainplot:
        fig2 = plt.figure(3)
        plt.title('Test vs Training error')
        plt.xlabel('Polynomial Degree')
        plt.ylabel('MSE')
        plt.semilogy(polynomial_vec[2:], error_test[2:], label='Test Error')
        plt.semilogy(polynomial_vec[2:], error_train[2:], label='Training Error')
        plt.legend()
        plt.show()

    return polynomial_vec, error_test, error_train, error_global, bias_, variance

def lambda_error(model_type, xdata, ydata, zdata, order, lambmax = 1, lambvecsize = 100, plot = False):
    '''
    loops over lambda for ridge and lasso and returns lambda and its corresponding MSE
    '''
    lvec = np.logspace(-13, lambmax, num = lambvecsize)
    mse = np.zeros(lambvecsize)

    X = build_design_matrix(xdata, ydata, order)

    if (model_type == 'ridge'):
        for i in range(lvec.shape[0]):
            regr = Regression_methods( X, zdata, lamb = lvec[i])
            mse[i] = regr.ms_error()

        if (plot == True):
            fig = plt.figure(4)
            plt.title('MSE vs lambda Ridge')
            plt.xlabel('Lambda')
            plt.ylabel('MSE')
            plt.semilogx(lvec, mse)
            plt.show()

            fig = plt.figure(5)
            plt.title('R2 vs lambda Ridge')
            plt.xlabel('Lambda')
            plt.ylabel('R2')
            plt.semilogx(lvec, r2)
            plt.show()

    if (model_type == 'lasso'):
        for i in range(lvec.shape[0]):
            reg = Lasso(alpha = lvec[i],  max_iter = 10e5, tol = 0.1, fit_intercept = False)
            reg.fit(X,zdata)
            beta = reg.coef_
            zpred = reg.predict(X)
            mse[i] = ms_error(zdata, zpred)

        if (plot == True):
            fig = plt.figure(6)
            plt.title('MSE vs lambda lasso')
            plt.xlabel('Lambda')
            plt.ylabel('MSE')
            plt.semilogx(lvec, mse)
            plt.show()

    return mse, lvec

def lasso_fit(xdata, ydata, zdata, order, lamb):
    '''
    plot the lasso prediction just to see if things are reasonable
    '''
    X = build_design_matrix(xdata, ydata, order)
    reg = Lasso(alpha = lamb,  max_iter = 10e4, tol = 0.001, fit_intercept = False)
    reg.fit(X,zdata)
    beta = reg.coef_
    zpred = reg.predict(X)

    fig1 = plt.figure(8)
    plt.scatter(xdata, ydata, c = zpred, cmap=cm.plasma)
    plt.show()
