"""
Written by S. Tomin, 2017
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy import optimize
from copy import deepcopy
import time
import matplotlib.cm as cm

class GP:
    def __init__(self):
        self.x_search = np.array([])
        self.x_obs = np.array([])
        self.y_obs = np.array([])
        self.y_sigma_obs = np.array([])
        self.y_pred = np.array([])
        self.sigma_y = 0

        # RBF kernel
        self.rbf_length_scale = 1
        self.rbf_length_scale_bounds = (0.01, 100)
        # ConstantKernel
        self.ck_const_value = 1.0
        self.ck_const_value_bounds = (1e-05, 100000.0)
        self.n_restarts_optimizer = 10
        self.max_iter = 40
        self.pen_max = 100
        self.ytol = 0.001
        self.xtol = 0.001
        self.opt_ctrl = None

    def append_new_data(self, x_new, y_obs, sigma_y_obs):
        self.x_obs = np.append(self.x_obs, [x_new], axis=0)
        self.y_obs = np.append(self.y_obs, y_obs)
        self.y_sigma_obs = np.append(self.y_sigma_obs, sigma_y_obs)

    def fit(self):
        """
        RBF(length_scale=1.0, length_scale_boun ds=(1e-05, 100000.0))
        k(x_i, x_j) = exp(-1 / 2 d(x_i / length_scale, x_j / length_scale)^2)
        :return:
        """
        # Instanciate a Gaussian Process model
        #kernel = ConstantKernel(self.ck_const_value, self.ck_const_value_bounds)\
        #         * RBF(self.rbf_length_scale, self.rbf_length_scale_bounds)
        #kernel = ConstantKernel(self.ck_const_value, self.ck_const_value_bounds)* RBF(self.rbf_length_scale, self.rbf_length_scale_bounds)
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        # Instanciate a Gaussian Process model
        if self.sigma_y != 0:
            self.alpha = (self.y_sigma_obs / self.y_obs) ** 2
        else:
            self.alpha = 1e-10
        print('alpha is', self.alpha)
        #self.gp = GaussianProcessRegressor(kernel=kernel, alpha=self.alpha,
                                           #n_restarts_optimizer=self.n_restarts_optimizer)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha = self.alpha, n_restarts_optimizer=9)
        # Fit to data using Maximum Likelihood Estimation of the parameters
        #print('self.x and self.y', self.x_obs, self.y_obs)
        print('trying to fit', self.x_obs, self.y_obs)
        self.gp.fit(self.x_obs, self.y_obs)
        print('success')

    def acquire_simplex(self):
        # Make the prediction on the meshed x-axis (ask for MSE as well)
        print('acquire simplex')
        def func(x):
            for i, xi in enumerate(x):
                if self.x_search[0][i] > xi or xi > self.x_search[-1][i]:
                    print("exceed limits ")
                    return self.pen_max

            y_pred, sigma = self.gp.predict(np.atleast_2d(x), return_std=True)
            self.sigma = sigma
            return y_pred

        y_pred, sigma = self.gp.predict(self.x_obs, return_std=True)
        x = self.x_obs[np.argmin(y_pred)]
        res = optimize.fmin(func, x)
        return res

    def acquire(self):
        # Make the prediction on the meshed x-axis (ask for MSE as well)
        y_pred, sigma = self.gp.predict(self.x_search, return_std=True)
        x = self.x_search[np.argmin(y_pred)]
        return x

    def minimize(self, error_func, x):
        # weighting for exploration vs exploitation in the GP at the end of scan, alpha array goes from 1 to zero
        print('making it to fit')
        self.fit()
        print('made it to iteration')
        for i in range(self.max_iter):
            # get next point to try using acquisition function
            if self.opt_ctrl != None and self.opt_ctrl.kill == True:
                print('GP: Killed from external process')
                break
            print('made it further')
            start = time.time()
            x_next = self.acquire()
            print("acquire ", start - time.time(), " sec")

            y_new = error_func(x_next.flatten())

            self.append_new_data(x_next, y_new, sigma_y_obs=self.sigma_y)
            # update the model (may want to add noise if using testEI)
            self.fit()
            if i>3 and np.linalg.norm((self.x_obs[-3] - self.x_obs[-1])) <= self.xtol:
                break
        return self.x_obs[-1]
