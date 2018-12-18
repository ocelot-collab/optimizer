# -*- coding: utf-8 -*-
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
	hyp_ARD = np.log(0.5/(length_scales**2))
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
    2018-06-12 - Removed prior mean from GP likelihood calculation
"""

import numpy as np
import numbers
from numpy.linalg import solve, inv

class OGP(object):
    def __init__(self, dim, hyperparams, covar='RBF_ARD', maxBV=20000, 
                 prmean=None, prmeanp=None, prvar=None, prvarp=None, proj=True, weighted=False, thresh=1e-6, sparsityQ = True):
        self.nin = dim
        self.maxBV = maxBV
        self.numBV = 0
        self.proj = proj
        self.weighted = weighted
        self.sparsityQ = sparsityQ
        self.verboseQ = True
        self.nupdates = 0
        
        if(covar in ['RBF_ARD']):
            self.covar = covar
            self.covar_params = hyperparams[:2]
        else:
            print 'Unknown covariance function'
            raise
            
        self.noise_var = np.exp(hyperparams[2]) # variance -- not stdev

        # prior (mean and variance): function; parameters
        self.prmean = prmean; self.prmeanp = prmeanp
        self.prvar = prvar; self.prvarp = prvarp
        
        # initialize model state
        self.BV = np.zeros(shape=(0,self.nin))
        self.alpha = np.zeros(shape=(0,1))
        self.C = np.zeros(shape=(0,0))
        
        self.KB = np.zeros(shape=(0,0))
        self.KBinv = np.zeros(shape=(0,0))
        
        self.thresh = thresh
        
    def fit(self, X, Y, m=0):
        # just train on all the data in X. m is a dummy parameter
        for i in range(X.shape[0]):
            self.update(np.array(X.iloc[i,:],ndmin=2),Y[i])
        
    def update(self, x_new, y_new):
        # compute covariance with BVs
        k_x = self.computeCov(self.BV, x_new)
        k = self.computeCov(x_new, x_new, is_self=True)

        # compute mean and variance
        cM = np.dot(np.transpose(k_x),self.alpha)
        cV = k + np.dot(np.transpose(k_x),np.dot(self.C,k_x))
        # not needed if nout==1: cV = (cV + np.transpose(cV)) / 2
        #cV = np.max(cV, 1.e-12)
        cV = np.reshape(np.max(np.append(cV,1.e-12)),cV.shape)
        
        #pM = self.priorMean(x_new) # Mitch's

        # calculate posterior PDF
        
        if(callable(self.prmean)):

            # GP
            gpMean = cM
            gpVar = cV

            # prior
            priorMean = self.priorMean(x_new); pM = priorMean
            priorVar = self.priorVar(x_new)

            # posterior
            postMean = (priorMean * gpVar + gpMean * priorVar) / (gpVar + priorVar)
            postVar = gpVar * priorVar / (gpVar + priorVar)

        else:

            # posterior PDF
            postMean = cM
            postVar = cV
            
        print 'cM = ', cM
        print 'cV = ', cV
        print 'postMean = ', postMean
        print 'postVar = ', postVar

        # end if/else

        #(logLik, K1, K2) = logLikelihood(self.noise_var, y_new, cM+pM, cV) # Mitch's
        (logLik, K1, K2) = logLikelihood(self.noise_var, y_new, cM, cV) # joe: i don't think that the GP likelihood should take the prior mean
        print ' (logLik, K1, K2) = ',  (logLik, K1, K2)
        #(logLik, K1, K2) = logLikelihood(self.noise_var, y_new, postMean, postVar)
        #print ' (logLik, K1, K2) = ',  (logLik, K1, K2)
        
        # compute gamma, a geometric measure of novelty
        if(self.KB.shape[0] > 0):
            hatE = solve(self.KB, k_x)
            gamma = k - np.dot(np.transpose(k_x),hatE)
        else:
            hatE = np.array([],ndmin=2).transpose()
            gamma = k
            
        if self.verboseQ: print("self.sparsityQ = " + str(self.sparsityQ))
        if self.sparsityQ:
            if(gamma < self.thresh*k):
                # not very novel, just tweak parameters
                if self.verboseQ: print("OGP - INFO: Just tweaking parameters")
                self._sparseParamUpdate(k_x, K1, K2, gamma, hatE)
                #if self.verboseQ: print("OGP - WARNING: Forcing full parameter update")
                #self._fullParamUpdate(x_new, k_x, k, K1, K2, gamma, hatE)
            else:
                # expand model
                if self.verboseQ: print("OGP - INFO: Expanding full model")
                print 'self._fullParamUpdate(',x_new, k_x, k, K1, K2, gamma, hatE,')'
                self._fullParamUpdate(x_new, k_x, k, K1, K2, gamma, hatE)
        else:
            # expand model
            if self.verboseQ: print("OGP - INFO: Expanding full model")
            self._fullParamUpdate(x_new, k_x, k, K1, K2, gamma, hatE)
        
        # reduce model according to maxBV constraint
        if self.sparsityQ:
            while(self.BV.shape[0] > self.maxBV):
                minBVind = self.scoreBVs()
                self.deleteBV(minBVind)
	else:
	    if self.verboseQ: print("OGP - INFO: Cutting BVs") 
	    
	# count number of updates (assuming one update per acquisition, this gives number of acquisitions for optimizer GP-UCB acquisition fcn)
            
    def predict(self, x_in):
        # reads in a (n x dim) vector and returns the (n x 1) vector 
        #   of predictions along with predictive variance for each

        # GP regression
        k_x = self.computeCov(x_in, self.BV)
        k = self.computeCov(x_in, x_in, is_self=True)
        gpMean = np.dot(k_x, self.alpha)
        gpVar = k + np.dot(k_x,np.dot(self.C,k_x.transpose()))
        
        print 'k_x = ', k_x
        print 'self.C = ', self.C
        print 'k_x = ', k_x

        # return posterior PDF

        if(callable(self.prmean)): # we have a prior
            print '[gpMean, gpVar, k] = [', gpMean, ', ', gpVar, ', ', k, ']'

            # prior
            priorMean = self.priorMean(x_in)
            priorVar = self.priorVar(x_in)
            
            #print '[priorMean, priorVar] = [', priorMean, ', ', priorVar, ']'

            # posterior
            postMean = (priorMean * gpVar + gpMean * priorVar) / (gpVar + priorVar)
            postVar = gpVar * priorVar / (gpVar + priorVar)

            #print '[postMean, postVar] = [', postMean, ', ', postVar, ']'

            return postMean, postVar

        else:

            return gpMean, gpVar
        
        #return gpMean + priorMean, var # this is wrong...
        #return postMean, postVar

        #(\[Mu]2 \[Sigma]1^2+\[Mu]1 \[Sigma]2^2)/(\[Sigma]1^2+\[Sigma]2^2),Sqrt[(\[Sigma]1^2 \[Sigma]2^2)/(\[Sigma]1^2+\[Sigma]2^2)] (*posterior mean and standard deviation*)
            
    def _sparseParamUpdate(self, k_x, K1, K2, gamma, hatE):
        # computes a sparse update to the model without expanding parameters
    
        eta = 1
        if(self.proj):
            eta += K2 * gamma
            
        CplusQk = np.dot(self.C, k_x) + hatE
        self.alpha = self.alpha + (K1 / eta) * CplusQk
        eta = K2 / eta
        self.C = self.C + eta * np.dot(CplusQk,CplusQk.transpose())
        self.C = stabilizeMatrix(self.C)
        
    def _fullParamUpdate(self, x_new, k_x, k, K1, K2, gamma, hatE):
        # expands parameters to incorporate new input
        
        # add new input to basis vectors
        oldnumBV = self.BV.shape[0]; print 'oldnumBV = ', oldnumBV
        numBV = oldnumBV + 1
        #print(self.BV)  ###############
        #print(x_new)  #################
        if(self.BV.shape == (0,)): # seems like self.BV and x_new have incompatible shapes for first call
            self.BV = x_new
        else:
            self.BV = np.concatenate((self.BV,x_new), axis=0)
        print 'self.BV = ', self.BV
        hatE = extendVector(hatE, val=-1)
        print 'hatE = ', hatE
        # update KBinv
        self.KBinv = extendMatrix(self.KBinv)
        self.KBinv = self.KBinv + (1 / gamma) * np.dot(hatE,hatE.transpose())
        print 'self.KBinv = ', self.KBinv
        # update Gram matrix
        self.KB = extendMatrix(self.KB)
        if(numBV > 1):
            self.KB[0:oldnumBV,[oldnumBV]] = k_x
            self.KB[[oldnumBV],0:oldnumBV] = k_x.transpose()
        self.KB[oldnumBV,oldnumBV] = k
        print 'self.KB = ', self.KB
        Ck = extendVector(np.dot(self.C, k_x), val=1)
        print 'Ck = ', Ck
        self.alpha = extendVector(self.alpha)
        self.C = extendMatrix(self.C)
        print 'self.alpha = ', self.alpha
        print 'self.C = ', self.C
        self.alpha = self.alpha + K1 * Ck
        self.C = self.C + K2 * np.dot(Ck, Ck.transpose())
        print 'self.alpha = ', self.alpha
        print 'self.C = ', self.C
        
        # stabilize matrices for conditioning/reducing floating point errors?
        self.C = stabilizeMatrix(self.C)
        self.KB = stabilizeMatrix(self.KB)
        self.KBinv = stabilizeMatrix(self.KBinv)
        print 'self.C = ', self.C
        print 'self.KB = ', self.KB
        print 'self.KBinv = ', self.KBinv
        
    def scoreBVs(self):
        # measures the importance of each BV for model accuracy  
        # currently quite slow for the weighted GP if numBV is much more than 50
        
        numBV = self.BV.shape[0]
        a = self.alpha
        if(not self.weighted):
            scores = ((a * a).reshape((numBV)) / 
                (self.C.diagonal() + self.KBinv.diagonal()))
        else:
            scores = np.zeros(shape=(numBV,1))
            
            # This is slow, in particular the numBV calls to computeWeightedDiv
            for removed in range(numBV):
                (hatalpha, hatC) = self.getUpdatedParams(removed)
                
                scores[removed] = self.computeWeightedDiv(hatalpha, hatC, removed)
                            
        return scores.argmin()

    def priorMean(self, x):
        if(callable(self.prmean)):
            if(self.prmeanp is not None):
                return self.prmean(x, self.prmeanp)
            else:
                return self.prmean(x)
        elif(isinstance(self.prmean,numbers.Number)):
            return self.prmean
        else:
            # if no prior mean function is supplied, assume zero
            return 0

    def priorVar(self, x):
        if(callable(self.prvar)):
            if(self.prvarp is not None):
                return self.prvar(x, self.prvarp)
            else:
                return self.prvar(x)
        elif(isinstance(self.prvar,numbers.Number)):
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
        q_star = self.KBinv[removeInd,removeInd]
        red_q = self.KBinv[keepInd][:,[removeInd]]
        self.KBinv = (self.KBinv[keepInd][:,keepInd] - 
            (1 / q_star) * np.dot(red_q, red_q.transpose()))
        self.KBinv = stabilizeMatrix(self.KBinv)
        
        self.KB = self.KB[keepInd][:,keepInd]
        self.BV = self.BV[keepInd]
        
    def computeWeightedDiv(self, hatalpha, hatC, removeInd):
        # computes the weighted divergence for removing a specific BV
        # currently uses matrix inversion and therefore somewhat slow
        
        hatalpha = extendVector(hatalpha, ind=removeInd)
        hatC = extendMatrix(hatC, ind=removeInd)        
        
        diff = self.alpha - hatalpha
        scale = np.dot(self.alpha.transpose(), np.dot(self.KB,self.alpha))
        
        Gamma = np.eye(self.BV.shape[0]) + np.dot(self.KB,self.C)
        Gamma = Gamma.transpose() / scale + np.eye(self.BV.shape[0])
        M = 2 * np.dot(Gamma,self.alpha) - (self.alpha + hatalpha)
        
        hatV = inv(hatC + self.KBinv)
        (s,logdet) = np.linalg.slogdet(np.dot(self.C + self.KBinv, hatV))
        
        if(s==1):
            w = np.trace(np.dot(self.C - hatC, hatV)) - logdet
        else:
            w = np.Inf
        
        return np.dot(M.transpose(), np.dot(hatV, diff)) + w
        
    def getUpdatedParams(self, removeInd):
        # computes updates for alpha and C after removing the given BV        
        
        numBV = self.BV.shape[0]
        keepInd = [i for i in range(numBV) if i != removeInd]
        a = self.alpha
        
        if(not self.weighted):
            # compute auxiliary variables
            q_star = self.KBinv[removeInd,removeInd]
            red_q = self.KBinv[keepInd][:,[removeInd]]
            c_star = self.C[removeInd,removeInd]
            red_CQsum = red_q + self.C[keepInd][:,[removeInd]]
        
            if(self.proj):
                hatalpha = (a[keepInd] - 
                    (a[removeInd] / (q_star + c_star)) * red_CQsum)
                hatC = (self.C[keepInd][:,keepInd] + 
                    (1 / q_star) * np.dot(red_q,red_q.transpose()) -
                    (1 / (q_star + c_star)) * np.dot(red_CQsum,red_CQsum.transpose()))
            else:
                tempQ = red_q / q_star
                hatalpha = a[keepInd] - a[removeInd] * tempQ
                red_c = self.C[removeInd,[keepInd]]
                hatC = (self.C[keepInd][:,keepInd] + 
                        c_star * np.dot(tempQ,tempQ.transpose()))
                tempQ = np.dot(tempQ, red_c)
                hatC = hatC - tempQ - tempQ.transpose()
        else:
            # compute auxiliary variables
            q_star = self.KBinv[removeInd,removeInd]
            red_q = self.KBinv[keepInd][:,[removeInd]]
            c_star = self.C[removeInd,removeInd]
            red_CQsum = red_q + self.C[keepInd][:,[removeInd]]
            Gamma = (np.eye(numBV) + np.dot(self.KB, self.C)).transpose()
            Gamma = (np.eye(numBV) + 
                Gamma / np.dot(a.transpose(), np.dot(self.KB, a)))
                
            hatalpha = (np.dot(Gamma[keepInd], a) - 
                np.dot(Gamma[removeInd], a) * red_q / q_star)
                    
            # this isn't rigorous...
            #extend = extendVector(hatalpha, ind=removeInd)
            hatC = self.C# + np.dot(2*np.dot(Gamma,a) - (a + extend),
                                       #(a - extend).transpose())
            hatC = (hatC[keepInd][:,keepInd] + 
                    (1 / q_star) * np.dot(red_q,red_q.transpose()) -
                    (1 / (q_star + c_star)) * np.dot(red_CQsum,red_CQsum.transpose()))
            
        return hatalpha, hatC
        
    def computeCov(self, x1, x2, is_self=False):
	# computes covariance between inputs x1 and x2
        #   returns a matrix of size (n1 x n2)
        K = self.computeRBF(x1, x2)
	if(is_self):
            K = K + self.noise_var * np.eye(x1.shape[0])
    
        return K

    def computeRBF(self, x1, x2):
        (n1, dim) = x1.shape
        n2 = x2.shape[0]
    
        (hyp_ARD, hyp_coeff) = self.covar_params
    
        b = np.exp(hyp_ARD)
        coeff = np.exp(hyp_coeff)
    
        # use ARD to scale
        b_sqrt = np.sqrt(b)
        x1 = x1 * b_sqrt
        x2 = x2 * b_sqrt
    

        x1_sum_sq = np.reshape(np.sum(x1 * x1, axis=1), (n1,1))
        x2_sum_sq = np.reshape(np.sum(x2 * x2, axis=1), (1,n2))

        K = -2 * np.dot(x1, x2.transpose())
        K = K + x1_sum_sq + x2_sum_sq
        K = coeff * np.exp(-.5 * K)
    
        return K

    def computeMatern(self, x1, x2, nu=2.5):
	(n1, dim) = x1.shape
        n2 = x2.shape[0]
    
        (hyp_ARD, hyp_coeff) = self.covar_params
    
        b = np.exp(hyp_ARD)
        coeff = np.exp(hyp_coeff)
    
        # use ARD to scale
        b_sqrt = np.sqrt(b)
        x1 = x1 * b_sqrt
        x2 = x2 * b_sqrt
    
	if(n1 != n2 or np.any(x1-x2)):
            x1_sum_sq = np.reshape(np.sum(x1 * x1, axis=1), (n1,1))
            x2_sum_sq = np.reshape(np.sum(x2 * x2, axis=1), (1,n2))

            dist_sq = -2 * np.dot(x1, x2.transpose())
            dist_sq = dist_sq + x1_sum_sq + x2_sum_sq
	    dist = np.sqrt(dist_sq)
	else:
	    dist = np.zeros((n1,n2))
	    dist_sq = np.zeros((n1,n2))
	
	if(nu==1.5):
	    poly = 1 + np.sqrt(3.0)*dist
	elif(nu==2.5):
	    poly = 1 + np.sqrt(5.0)*dist + (5.0/3.0)*dist_sq
	else:
	    print 'Invalid nu (only 1.5 and 2.5 supported)'

        K = coeff * poly * np.exp(-np.sqrt(2*nu)*dist)

	return K
    

        #logLikelihood(self.noise_var, y_new, cM, cV)
def logLikelihood(noise, y, mu, var):
    sigX2 = noise + var
    K2 = -1 / sigX2
    K1 = -K2 * (y - mu)
    logLik = - (np.log(2*np.pi*sigX2) + (y - mu) * K1) / 2
    
    return logLik, K1, K2
    
def stabilizeMatrix(M):
    return (M + M.transpose()) / 2
    
def extendMatrix(M, ind=-1):
    if(ind==-1):
        M = np.concatenate((M,np.zeros(shape=(M.shape[0],1))),axis=1)
        M = np.concatenate((M,np.zeros(shape=(1,M.shape[1]))),axis=0)
    elif(ind==0):
        M = np.concatenate((np.zeros(shape=(M.shape[0],1)),M),axis=1)
        M = np.concatenate((np.zeros(shape=(1,M.shape[1])),M),axis=0)
    else:
        M = np.concatenate((M[:ind], np.zeros(shape=(1,M.shape[1])), M[ind:]),axis=0)
        M = np.concatenate((M[:,:ind], np.zeros(shape=(M.shape[0],1)), M[:,ind:]),axis=1)
    return M
    
def extendVector(v, val=0, ind=-1):
    if(ind==-1):
        return np.concatenate((v,[[val]]),axis=0)
        #return v+[[[val]]]
    elif(ind==0):
        return np.concatenate(([[val]],v),axis=0)
        #return [[[val]]]+v
    else:
        return np.concatenate((v[:ind],[[val]],v[ind:]),axis=0)
        #return v[:ind]+[[[val]]]+v[ind:]
