# -*- coding: iso-8859-1 -*-
from __future__ import absolute_import, print_function
import os
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from GP.OnlineGP import OGP
from GP.DKL.dknet import NNRegressor
from GP.DKL.dknet.layers import Dense, CovMat
from GP.DKL.dknet.optimizers import Adam
#from GP.GP_utils import SPGP_likelihood_4scipy

# the DKLGP class
#        - uses neural networks to embed input data x -> z before feeding into GP
# dim: the dimension of the input data
# hidden_layers: a list of integers specifying the structure of the network
#        - if hidden_layers=[], then the embedding is linear, as nonlinearities are only applied
#                 in hidden layers
#        - if hidden_layers=[a,b,c], then there will be four total network layers, mapping to output dimensions as follows:
#                 dim -> a -> b -> c -> dim_z
# dim_z: the output dimension of the network embedding. This is the dimension that the final GP will operate on
# mask: if there are no hidden layers, this is used to reduce the number of parameters in the embedding
# alpha: the amplitude parameter used by the Online GP after training. This does not affect the embedding in any way (not logged)
# noise: the noise variance parameter used by the Online GP after training. This does not affect the embedding in any way (not logged)
# activations: the activation functions used as nonlinearities in the hidden layers. Need to be supported by DKL
#        - can be in ['relu', 'lrelu', 'linear', 'sigmoid', 'tanh', 'softplus', 'softmax', 'rbf']
#        - easy to add more if desired
# weight_dir: string specifying directory location. if specified, initializes the network and embedding function based
#                 on the parameters found in the directory
#        - it is essential that the network structure matches the architecture implied by the parameters in weight_dir

# simple training example:
#  dkl = DKLGP(dim, hidden_layers = [4,4])
#  dkl.train_embedding(X,Y)
#  dkl.save_embedding('weight_dir')
#  Z = dkl.embed(X_new)
#  LL = dkl.eval_LL(X_new,Y_new)

# simple linear training example:
#  lower_tri_mask = np.ones((dim,dim))
#  lower_tri_mask[0,1] = 0.
#  dkl = DKLGP(dim, mask=lower_tri_mask)  # might as well use lower-tri for linear map
#  dkl.train_embedding(X,Y)
#  dkl.save_embedding('linear_map')
#  Z = dkl.embed(X_new)
#  LL = dkl.eval_LL(X_new,Y_new)

# simple known param linear example
#  dkl = DKLGP(dim, alpha=known_alpha, noise=known_noise)
#  dkl.set_linear(known_linear_map)
#  Z = dkl.embed(X_new)
#  LL = dkl.eval_LL(X_new,Y_new)

# DKL.train_embedding does not vary alpha and noise

class DKLGP(object):
    def __init__(self, dim, hidden_layers=[], dim_z=None, mask=None, alpha=1.0, noise=0.1, activations='lrelu', weight_dir=None):
        self.dim = dim
        self.dim_z = dim_z or dim

        # initialize the OGP object we use to actually make our predictions
        OGP_params = (np.zeros((self.dim_z,)), np.log(alpha), np.log(noise)) # lengthscales of one (logged)
        self.ogp = OGP(self.dim_z, OGP_params)

        # our embedding function, initially the identity
        # if unchanged, the DKLGP should match the functionality of OGP
        self.embed = lambda x: x

        # build the neural network structure of the DKL
        self.layers = []
        for l in hidden_layers:
            self.layers.append(Dense(l, activation=activations))

        # add the linear output layer and the GP (used for likelihood training)
        if len(self.layers) > 0:
            self.layers.append(Dense(dim_z))
        else:
            self.mask = mask
            self.layers.append(Dense(dim_z, mask=mask))
        self.layers.append(CovMat(kernel='rbf', alpha_fixed=False)) # kernel should match the one used in OGP

        # if weight_dir is specified, we immediately initialize the embedding based on the specified neural network
        if weight_dir is not None:
            self.load_embedding(weight_dir)

    # sets up the DKL and trains the embedding. nullifies the effect of load_embedding if it was called previously
    # lr is the learning rate: reasonable deafult is 2e-4
    # maxiter is the number of iterations of the solver; scales the training time linearly
    # batch_size is the size of a mini batch; scales the training time ~quadratically
    # gp = True in NNRegressor() sets gp likelihood as optimization target
    def train_embedding(self, x, y, lr=2.e-4, batch_size=50, maxiter=4000):
        opt = Adam(lr)
        self.DKLmodel = NNRegressor(self.layers, opt=opt, batch_size=batch_size, maxiter=maxiter, gp=True, verbose=False)
        self.DKLmodel.fit(x,y)

        self.embed = self.DKLmodel.fast_forward # fast_forward gives mapping up to (but not including) gp (x -> z)
                                                # (something like) full_forward maps through the whole dkl + gp

    # loads the DKL and embedding from the specified directory. forgets any previous embedding
    # note that network structure and activations, etc. still need to be specified in __init__
    def load_embedding(self, dname):
        self.DKLmodel = NNRegressor(self.layers)
        self.DKLmodel.first_run(np.zeros((1,self.dim)), None, load_path=dname)

        self.embed = self.DKLmodel.fast_forward

    # saves the neural network parameters to specified directory, allowing the saved embedding to be replicated without re-training it
    def save_embedding(self, dname):
        if not os.path.isdir(dname):
            os.makedirs(dname)
        self.DKLmodel.save_weights(dname)

    # allows manually setting a linear transform. Make sure you get your tranpose stuff right (x_rows.shape is [npoints,ndim])
    def set_linear(self, matrix):
        self.linear_transform = matrix
        self.embed = lambda x_rows: np.dot(x_rows, self.linear_transform)

    # sets a linear transformation based on a given correlation matrix which is assumed to fit the data
    # NOTE: this isn't necessarily log-likelihood-optimal
    def linear_from_correlation(self, matrix): # multinormal covariance matrix
        center = np.linalg.inv(matrix)
        chol = np.linalg.cholesky(center)

        self.set_linear(chol)

    # computes the log-likelihood of the given data set using the current embedding
    # ASSUMES YOU'RE USING RBF KERNEL
    def eval_LL(self, X, Y):
        N = X.shape[0]
        Z = self.embed(X)
        diffs = euclidean_distances(Z, squared=True)

        alpha = np.exp(self.ogp.covar_params[1]) # kind of a hack
        rbf_K = alpha * np.exp(-diffs / 2.)
        K_full = rbf_K + (self.ogp.noise_var) * np.eye(N)

        L = np.linalg.cholesky(K_full)  # K = L * L.T
        Ly = np.linalg.solve(L, Y)  # finds inverse(L) * y
        log_lik = -0.5 * np.sum(Ly**2) # -1/2 * y.T * inverse(L * L.T) * y
        log_lik -= np.sum(np.log(np.diag(L)))  # equivalent to -1/2 * log(det(K))
        log_lik -= 0.5 * N * np.log(2 * np.pi)

        return float(log_lik)

    # allows passing custom alpha/noise
    # if compute_deriv is true, assumes that embedding is linear and returns derivative w.r.t. transform
    def custom_LL(self, X, Y, alpha, noise_variance, compute_deriv=False):
        N,dim = X.shape
        Z = self.embed(X)
        if not compute_deriv:
            diffs = euclidean_distances(Z, squared=True)

            rbf_K = alpha * np.exp(-diffs / 2.)
            K_full = rbf_K + noise_variance * np.eye(N)

            L = np.linalg.cholesky(K_full)  # K = L * L.T
            Ly = np.linalg.solve(L, Y)  # finds inverse(L) * y
            log_lik = -0.5 * np.sum(Ly**2) # -1/2 * y.T * inverse(L * L.T) * y
            log_lik -= np.sum(np.log(np.diag(L)))  # equivalent to -1/2 * log(det(K))
            log_lik -= 0.5 * N * np.log(2 * np.pi)

            return float(log_lik)

        lengths = [0. for d in range(dim)]
        params = lengths + [np.log(alpha)] + [np.log(noise_variance)]
        neglik, deriv = SPGP_likelihood_4scipy(params, Y, Z)

        deriv_noise = deriv[-1]
        deriv_coeff = deriv[-2]
        deriv_z = deriv[:self.dim_z*N].reshape((N,self.dim_z))

        deriv_transform = np.dot(X.T, deriv_z)
        mask = self.mask or np.ones((dim,dim_z))
        return -neglik, deriv_transform * mask, deriv_coeff, deriv_noise

    # takes an n x dim_z matrix Z and translates it to x, assuming the embedding is linear
    # currently requires that the model embedding was set via set_linear
    def inverse_embed(self, Z):
        assert ('linear_transform' in dir(self))
        transform = self.linear_transform

        # assumption is that z = x * transform
        column_x = np.linalg.solve(transform.T, Z.T)
        return column_x.T

    ##########
    # remaining functions mimic Online GP functionality, just embedding x -> z first
    ##########

    def fit(self, X, y):
        Z = self.embed(X)
        self.ogp.fit(Z, y)

    def update(self, x_new, y_new):
        z_new = self.embed(x_new)
        self.ogp.update(z_new, y_new)

    def predict(self, x):
        z = np.array(self.embed(x),ndmin=2)
        return self.ogp.predict(z)