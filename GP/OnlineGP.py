# -*- coding: iso-8859-1 -*-
"""
Designed by Lehel Csato for NETLAB, rewritten
for Python in 2016 by Mitchell McIntire

The Online Gaussian process class.

Initialization parameters:
    dim: the dimension of input data
    hyperparams: GP model hyperparameters. For RBF_ARD, a 3-tuple with entries:
        hyp_ARD: size (1 x dim) vector of ARD parameters
        hyp_coeff: the coefficient parameter of the RBF kernel
        hyp_noise: the model noise VARIANCE hyperparameter
        Note -- different hyperparams needed for different covariance functions
    For RBF_ARD:
        hyp_ARD = np.log(1./(length_scales**2))
        hyp_coeff = np.log(signal_peak_amplitude)
        hyp_noise = np.log(signal_variance) <== note: signal standard deviation squared
    covar: the covariance function to be used, currently only 'RBF_ARD'
        RBF_ARD: the radial basis function with ARD, i.e. a squared exponential
            with diagonal scaling matrix specified by hyp_ARD
    maxBV: the number of basis vectors to represent the model. Increasing this
        beyond 100 or so will increase runtime substantially
    prmean: either None, a number, or a callable function that gives the prior mean
    prmeanp: parameters to the prmean function
    proj: I'm not sure exactly. Setting this to false gives a different method
        of computing updates, but I haven't tested it or figured out what the
        difference is.
    weighted: whether or not to use weighted difference computations. Slower but may
        yield improved performance. Still testing.
    thresh: some low float value to specify how different a point has to be to
        add it to the model. Keeps matrices well-conditioned.

Methods:
    update(x_new, y_new): Runs an online GP iteration incorporating the new data.
    fit(X, Y): Calls update on multiple points for convenience. X is assumed to
        be a pandas DataFrame.
    predict(x): Computes GP prediction(s) for input point(s).
    scoreBVs(): Returns a vector with the (either weighted or unweighted) KL
        divergence-cost of removing each BV.
    deleteBV(index): Removes the selected BV from the GP and updates to minimize
        the (either weighted or unweighted) KL divergence-cost of the removal

Change log:
    2019-10-23 - Adi added Matern kernel
    2018-02-?? - Mitch fixed a bug where the noise parameter wasn't used
    2018-02-23 - Joe suggestions to make code more user friendly
                 1) Change hyper_noise input to stdev -- currently variance
                 2) Input regular length scales -- currently log(0.5/lengths^2)
                 3) Drop logs on input parameters. Nothing gained by logging and
                    then exponentiating right after.
                 4) Add option to have likelihood calculate gradients. We need to
                    check result against full GP likelihood and also should
                    probably train parameters on the OnlineGP if we're using it!
    2018-05-23 - Joe added variance to prior mean and fixed the posterior PDF
    2018-06-12 - Removed prior mean from GP likelihood calculation in update

                 Prior philosophy: GP and prior are independent models which are
                 combined in a Bayesian way within the predict function.

    2018-11-11 - Adding non-diagonal matrix elements to the RBF kernel. To use,
                 just replace hyperlengths with a matrix instead of a vector.
    2018-11-14 - Last step. Need option to give precision matrix unlogged
    2018-12-05 - Dylan fixed a problem with loading in data for fitting
    2018-12-06 - Joe added __setstate__ and __getstate__ for easy pickling
"""

import numpy as np
import numbers
from numpy.linalg import solve, inv


class OGP(object):
    def __init__(self, dim, hyperparams, covar=['RBF_ARD','MATERN32_ARD','MATERN52_ARD'][0], maxBV=200,
                 prmean=None, prmeanp=None, prvar=None, prvarp=None, proj=True, weighted=False, thresh=1e-6,
                 sparsityQ=True):
        self.nin = dim
        self.maxBV = maxBV
        self.numBV = 0
        self.proj = proj
        self.weighted = weighted
        self.sparsityQ = sparsityQ
        self.verboseQ = False
        self.nupdates = 0

        if (covar in ['RBF_ARD','MATERN32_ARD','MATERN52_ARD']):
            self.covar = covar
            self.covar_params = hyperparams[:2]
            self.precisionMatrix = None
            cps = np.shape(self.covar_params[0])
            if len(cps) == 2:
                if cps[0] == cps[1]:
                    self.precisionMatrix = self.covar_params[0]
        else:
            self.precisionMatrix = None
            print('Unknown covariance function')
            raise Exception("Unknown covariance function")
            
        self.noise_var = np.exp(hyperparams[2])  # variance -- not stdev

        # prior (mean and variance): function; parameters
        self.prmean = prmean;
        self.prmeanp = prmeanp
        self.prvar = prvar;
        self.prvarp = prvarp

        # initialize model state
        self.BV = np.zeros(shape=(0, self.nin))
        self.alpha = np.zeros(shape=(0, 1))
        self.C = np.zeros(shape=(0, 0))

        self.KB = np.zeros(shape=(0, 0))
        self.KBinv = np.zeros(shape=(0, 0))

        self.thresh = thresh

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance atributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()

        # Remove unpicklable entries (these would need to be recreated
        # in the __setstate__ function.
        # Example: del state['file'] # since the file handle isn't pickleable

        return state

    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)

        # Should also manually recreate unpicklable members.
        # Example: file = load(self.filename)

    def fit(self, X, Y, m=0):
        # just train on all the data in X. m is a dummy parameter
        for i in range(X.shape[0]):
            self.update(np.array(X[i], ndmin=2), np.array([Y[i]]))
            # self.update(x, Y[i])

    def update(self, x_new, y_new):
        # compute covariance with BVs
        k_x = self.computeCov(self.BV, x_new)
        k = self.computeCov(x_new, x_new, is_self=True)

        # compute mean and variance
        cM = np.dot(np.transpose(k_x), self.alpha)
        cV = k + np.dot(np.transpose(k_x), np.dot(self.C, k_x))
        # not needed if nout==1: cV = (cV + np.transpose(cV)) / 2
        # cV = np.max(cV, 1.e-12)
        cV = np.reshape(np.max(np.append(cV, 1.e-12)), cV.shape)

        pM = self.priorMean(x_new)  # Mitch's

        (logLik, K1, K2) = logLikelihood(self.noise_var, y_new, cM + pM, cV)  # Mitch's
        # (logLik, K1, K2) = logLikelihood(self.noise_var, y_new, cM, cV) # joe: i don't think that the GP likelihood should take the prior mean

        # compute gamma, a geometric measure of novelty
        if (self.KB.shape[0] > 0):
            hatE = solve(self.KB, k_x)
            gamma = k - np.dot(np.transpose(k_x), hatE)
        else:
            hatE = np.array([], ndmin=2).transpose()
            gamma = k

        if (self.sparsityQ and gamma < self.thresh * k):
            # not very novel, just tweak parameters
            if self.verboseQ: print("OGP - INFO: Just tweaking parameters")
            self._sparseParamUpdate(k_x, K1, K2, gamma, hatE)
            # if self.verboseQ: print("OGP - WARNING: Forcing full parameter update")
            # self._fullParamUpdate(x_new, k_x, k, K1, K2, gamma, hatE)
        else:
            # expand model
            if self.verboseQ:
                print("OGP - INFO: Expanding full model")
                print('self._fullParamUpdate(', x_new, k_x, k, K1, K2, gamma, hatE, ')')
            self._fullParamUpdate(x_new, k_x, k, K1, K2, gamma, hatE)

        # reduce model according to maxBV constraint
        if self.sparsityQ:
            if self.verboseQ: print("OGP - INFO: Cutting BVs")
            while (self.BV.shape[0] > self.maxBV):
                minBVind = self.scoreBVs()
                self.deleteBV(minBVind)
        else:
            pass

        # count number of updates (assuming one update per acquisition, this gives number of acquisitions for optimizer GP-UCB acquisition fcn)

    def predict(self, x_in):
        # reads in a (n x dim) vector and returns the (n x 1) vector
        #   of predictions along with predictive variance for each

        # GP regression
        k_x = self.computeCov(x_in, self.BV)
        k = self.computeCov(x_in, x_in, is_self=True)
        gpMean = np.dot(k_x, self.alpha)
        gpVar = k + np.dot(k_x, np.dot(self.C, k_x.transpose()))

        priorMean = 0.
        if (callable(self.prmean)):  # we have a prior
            # prior
            priorMean = self.priorMean(x_in)

        return gpMean + priorMean, gpVar

        # return posterior PDF

        # if(False and callable(self.prmean)): # we have a prior variance

        ## prior
        # priorMean = self.priorMean(x_in)
        # priorVar = self.priorVar(x_in)

        ## posterior
        # postMean = (priorMean * gpVar + gpMean * priorVar) / (gpVar + priorVar)
        # postVar = gpVar * priorVar / (gpVar + priorVar)

        ##print '[gpMean, gpVar] = [', gpMean, ', ', gpVar, ']', '\t', '[priorMean, priorVar] = [', priorMean, ', ', priorVar, ']', '\t', '[postMean, postVar] = [', postMean, ', ', postVar, ']'

        # return postMean, postVar

        # else:

        # return gpMean, gpVar

    def _sparseParamUpdate(self, k_x, K1, K2, gamma, hatE):
        # computes a sparse update to the model without expanding parameters

        eta = 1
        if (self.proj):
            eta += K2 * gamma

        CplusQk = np.dot(self.C, k_x) + hatE
        self.alpha = self.alpha + (K1 / eta) * CplusQk
        eta = K2 / eta
        self.C = self.C + eta * np.dot(CplusQk, CplusQk.transpose())
        self.C = stabilizeMatrix(self.C)

    def _fullParamUpdate(self, x_new, k_x, k, K1, K2, gamma, hatE):
        # expands parameters to incorporate new input

        # add new input to basis vectors
        oldnumBV = self.BV.shape[0]
        numBV = oldnumBV + 1

        if (self.BV.shape == (0,)):  # seems like self.BV and x_new have incompatible shapes for first call
            self.BV = x_new
        else:
            self.BV = np.concatenate((self.BV, x_new), axis=0)

        hatE = extendVector(hatE, val=-1)

        # update KBinv
        self.KBinv = extendMatrix(self.KBinv)
        self.KBinv = self.KBinv + (1 / gamma) * np.dot(hatE, hatE.transpose())

        # update Gram matrix
        self.KB = extendMatrix(self.KB)
        if (numBV > 1):
            self.KB[0:oldnumBV, [oldnumBV]] = k_x
            self.KB[[oldnumBV], 0:oldnumBV] = k_x.transpose()
        self.KB[oldnumBV, oldnumBV] = k

        Ck = extendVector(np.dot(self.C, k_x), val=1)

        self.alpha = extendVector(self.alpha)
        self.C = extendMatrix(self.C)
        self.alpha = self.alpha + K1 * Ck
        self.C = self.C + K2 * np.dot(Ck, Ck.transpose())

        # stabilize matrices for conditioning/reducing floating point errors?
        self.C = stabilizeMatrix(self.C)
        self.KB = stabilizeMatrix(self.KB)
        self.KBinv = stabilizeMatrix(self.KBinv)

    def scoreBVs(self):
        # measures the importance of each BV for model accuracy
        # currently quite slow for the weighted GP if numBV is much more than 50

        numBV = self.BV.shape[0]
        a = self.alpha
        if (not self.weighted):
            scores = ((a * a).reshape((numBV)) /
                      (self.C.diagonal() + self.KBinv.diagonal()))
        else:
            scores = np.zeros(shape=(numBV, 1))

            # This is slow, in particular the numBV calls to computeWeightedDiv
            for removed in range(numBV):
                (hatalpha, hatC) = self.getUpdatedParams(removed)

                scores[removed] = self.computeWeightedDiv(hatalpha, hatC, removed)

        return scores.argmin()

    def priorMean(self, x):
        if (callable(self.prmean)):
            if (self.prmeanp is not None):
                return self.prmean(x, self.prmeanp)
            else:
                return self.prmean(x)
        elif (isinstance(self.prmean, numbers.Number)):
            return self.prmean
        else:
            # if no prior mean function is supplied, assume zero
            return 0

    def priorVar(self, x):
        if (callable(self.prvar)):
            if (self.prvarp is not None):
                return self.prvar(x, self.prvarp)
            else:
                return self.prvar(x)
        elif (isinstance(self.prvar, numbers.Number)):
            return self.prvar
        else:
            # if no prior variance function is supplied, assume unit variance
            return 1

    def deleteBV(self, removeInd):
        # removes a BV from the model and modifies parameters to
        #   attempt to minimize the removal's impact

        numBV = self.BV.shape[0]
        keepInd = [i for i in range(numBV) if i != removeInd]

        # update alpha and C
        (self.alpha, self.C) = self.getUpdatedParams(removeInd)

        # stabilize C
        self.C = stabilizeMatrix(self.C)

        # update KB and KBinv
        q_star = self.KBinv[removeInd, removeInd]
        red_q = self.KBinv[keepInd][:, [removeInd]]
        self.KBinv = (self.KBinv[keepInd][:, keepInd] -
                      (1 / q_star) * np.dot(red_q, red_q.transpose()))
        self.KBinv = stabilizeMatrix(self.KBinv)

        self.KB = self.KB[keepInd][:, keepInd]
        self.BV = self.BV[keepInd]

    def computeWeightedDiv(self, hatalpha, hatC, removeInd):
        # computes the weighted divergence for removing a specific BV
        # currently uses matrix inversion and therefore somewhat slow

        hatalpha = extendVector(hatalpha, ind=removeInd)
        hatC = extendMatrix(hatC, ind=removeInd)

        diff = self.alpha - hatalpha
        scale = np.dot(self.alpha.transpose(), np.dot(self.KB, self.alpha))

        Gamma = np.eye(self.BV.shape[0]) + np.dot(self.KB, self.C)
        Gamma = Gamma.transpose() / scale + np.eye(self.BV.shape[0])
        M = 2 * np.dot(Gamma, self.alpha) - (self.alpha + hatalpha)

        hatV = inv(hatC + self.KBinv)
        (s, logdet) = np.linalg.slogdet(np.dot(self.C + self.KBinv, hatV))

        if (s == 1):
            w = np.trace(np.dot(self.C - hatC, hatV)) - logdet
        else:
            w = np.Inf

        return np.dot(M.transpose(), np.dot(hatV, diff)) + w

    def getUpdatedParams(self, removeInd):
        # computes updates for alpha and C after removing the given BV

        numBV = self.BV.shape[0]
        keepInd = [i for i in range(numBV) if i != removeInd]
        a = self.alpha

        if (not self.weighted):
            # compute auxiliary variables
            q_star = self.KBinv[removeInd, removeInd]
            red_q = self.KBinv[keepInd][:, [removeInd]]
            c_star = self.C[removeInd, removeInd]
            red_CQsum = red_q + self.C[keepInd][:, [removeInd]]

            if (self.proj):
                hatalpha = (a[keepInd] -
                            (a[removeInd] / (q_star + c_star)) * red_CQsum)
                hatC = (self.C[keepInd][:, keepInd] +
                        (1 / q_star) * np.dot(red_q, red_q.transpose()) -
                        (1 / (q_star + c_star)) * np.dot(red_CQsum, red_CQsum.transpose()))
            else:
                tempQ = red_q / q_star
                hatalpha = a[keepInd] - a[removeInd] * tempQ
                red_c = self.C[removeInd, [keepInd]]
                hatC = (self.C[keepInd][:, keepInd] +
                        c_star * np.dot(tempQ, tempQ.transpose()))
                tempQ = np.dot(tempQ, red_c)
                hatC = hatC - tempQ - tempQ.transpose()
        else:
            # compute auxiliary variables
            q_star = self.KBinv[removeInd, removeInd]
            red_q = self.KBinv[keepInd][:, [removeInd]]
            c_star = self.C[removeInd, removeInd]
            red_CQsum = red_q + self.C[keepInd][:, [removeInd]]
            Gamma = (np.eye(numBV) + np.dot(self.KB, self.C)).transpose()
            Gamma = (np.eye(numBV) +
                     Gamma / np.dot(a.transpose(), np.dot(self.KB, a)))

            hatalpha = (np.dot(Gamma[keepInd], a) -
                        np.dot(Gamma[removeInd], a) * red_q / q_star)

            # this isn't rigorous...
            # extend = extendVector(hatalpha, ind=removeInd)
            hatC = self.C  # + np.dot(2*np.dot(Gamma,a) - (a + extend),
            # (a - extend).transpose())
            hatC = (hatC[keepInd][:, keepInd] +
                    (1 / q_star) * np.dot(red_q, red_q.transpose()) -
                    (1 / (q_star + c_star)) * np.dot(red_CQsum, red_CQsum.transpose()))

        return hatalpha, hatC

    def computeCov(self, x1, x2, is_self=False):
        # computes covariance between inputs x1 and x2
        #   returns a matrix of size (n1 x n2)

        # calculate covariance with kernel
        if self.covar == 'MATERN32_ARD':
            K = self.computeMatern(x1, x2, nu=1.5)
        elif self.covar == 'MATERN52_ARD':
            K = self.computeMatern(x1, x2, nu=2.5)
        else: # default to rbf
            if np.size(np.shape(self.covar_params[0])) == 2:
                K = self.computeCBF(x1, x2)
            else:
                K = self.computeRBF(x1, x2)
                
        # add noise
        if (is_self):
            K = K + self.noise_var * np.eye(x1.shape[0])

        return K

    def computeRBF(self, x1, x2):  # radial basis functions
        (n1, dim) = x1.shape
        n2 = x2.shape[0]

        (hyp_ARD, hyp_coeff) = self.covar_params

        # turn to normal units
        b = np.exp(hyp_ARD)
        coeff = np.exp(hyp_coeff)

        # use ARD to scale
        b_sqrt = np.sqrt(b)
        x1 = x1 * b_sqrt
        x2 = x2 * b_sqrt

        x1_sum_sq = np.reshape(np.sum(x1 * x1, axis=1), (n1, 1))
        x2_sum_sq = np.reshape(np.sum(x2 * x2, axis=1), (1, n2))

        K = -2 * np.dot(x1, x2.transpose())
        K = K + x1_sum_sq + x2_sum_sq
        K = coeff * np.exp(-0.5 * K)

        return K

    # updated to allow non-diagonal kernel matrix
    def computeCBF(self, x1, x2):  # correlated basis functions
        (n1, dim) = x1.shape
        n2 = x2.shape[0]

        if n1 * n2 == 0:
            return np.array([])

        (hyp_ARD, hyp_coeff) = self.covar_params
        coeff = np.exp(hyp_coeff)

        if type(self.precisionMatrix) == type(None):

            # turn to normal units
            b = np.exp(hyp_ARD)

            # if kernel params are a vector, reshape to a diagonal matrix
            # print 'b = ', b
            # print 'coeff = ', coeff
            sk = np.shape(b)
            nk = np.size(sk)  # 0 if scalar; 1 if vector; 2 if matrix
            if nk < 2:
                b = np.diagflat(b)
            elif nk == 2 and sk[0] < sk[1]:
                b = np.diagflat(b)
            elif nk > 2:
                print('OnlineGP - WARNING: kernel is a strange shape.')
            # print 'b = ', b

        else:
            b = self.precisionMatrix

        # save duplicate computations
        bdotx1T = np.array([np.dot(b, x.transpose()).transpose() for x in x1])
        bdotx2T = np.array([np.dot(b, x.transpose()).transpose() for x in x2])

        # compute with

        x1_sum_sq = np.reshape(np.sum(x1 * bdotx1T, axis=1), (n1, 1))
        x2_sum_sq = np.reshape(np.sum(x2 * bdotx2T, axis=1), (1, n2))

        K = -2 * np.dot(x1, bdotx2T.transpose())
        K = K + x1_sum_sq + x2_sum_sq
        K = coeff * np.exp(-0.5 * K)

        return K

    def computeMatern(self, x1, x2, nu=2.5):
        (n1, dim) = x1.shape
        n2 = x2.shape[0]

        (hyp_ARD, hyp_coeff) = self.covar_params

        # turn to normal units
        b = np.exp(hyp_ARD)
        coeff = np.exp(hyp_coeff)

        # use ARD to scale
        b_sqrt = np.sqrt(b)
        x1 = x1 * b_sqrt
        x2 = x2 * b_sqrt

        x1_sum_sq = np.reshape(np.sum(x1 * x1, axis=1), (n1, 1))
        x2_sum_sq = np.reshape(np.sum(x2 * x2, axis=1), (1, n2))

        dist_sq = x1_sum_sq  -2 * np.dot(x1, x2.transpose()) + x2_sum_sq
        dist = np.sqrt(dist_sq + 1e-14)
       
        if (nu == 1.5):
            poly = 1 + np.sqrt(3.0) * dist
        elif (nu == 2.5):
            poly = 1 + np.sqrt(5.0) * dist + (5.0 / 3.0) * dist_sq
        else:
            print('Invalid nu (only 1.5 and 2.5 supported)')

        K = coeff * poly * np.exp(-np.sqrt(2 * nu) * dist)

        return K

    # end OGP class


# GP function prediction pdf
def logLikelihood(noise, y, mu, var):
    sigX2 = noise + var
    K2 = -1 / sigX2
    delta = (y - mu)
    K1 = -K2 * delta
    logLik = - (np.log(2 * np.pi * sigX2) + delta * K1) / 2

    return logLik, K1, K2


def stabilizeMatrix(M):
    return (M + M.transpose()) / 2


def extendMatrix(M, ind=-1):
    if (ind == -1):
        M = np.concatenate((M, np.zeros(shape=(M.shape[0], 1))), axis=1)
        M = np.concatenate((M, np.zeros(shape=(1, M.shape[1]))), axis=0)
    elif (ind == 0):
        M = np.concatenate((np.zeros(shape=(M.shape[0], 1)), M), axis=1)
        M = np.concatenate((np.zeros(shape=(1, M.shape[1])), M), axis=0)
    else:
        M = np.concatenate((M[:ind], np.zeros(shape=(1, M.shape[1])), M[ind:]), axis=0)
        M = np.concatenate((M[:, :ind], np.zeros(shape=(M.shape[0], 1)), M[:, ind:]), axis=1)
    return M


def extendVector(v, val=0, ind=-1):
    if not len(v):
        return np.array([[val]])
    if (ind == -1):
        return np.concatenate((v, [[val]]), axis=0)
    elif (ind == 0):
        return np.concatenate(([[val]], v), axis=0)
    else:
        return np.concatenate((v[:ind], [[val]], v[ind:]), axis=0)
