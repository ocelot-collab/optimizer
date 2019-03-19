# -*- coding: iso-8859-1 -*-
"""
Contains the Bayes optimization class.
Initialization parameters:
    model: an object with methods 'predict', 'fit', and 'update'
    interface: an object which supplies the state of the system and
        allows for changing the system's x-value.
        Should have methods '(x,y) = intfc.getState()' and 'intfc.setX(x_new)'.
        Note that this interface system is rough, and used for testing and
            as a placeholder for the machine interface.
    acq_func: specifies how the optimizer should choose its next point.
        'PI': uses probability of improvement. The interface should supply y-values.
        'EI': uses expected improvement. The interface should supply y-values.
        'UCB': uses GP upper confidence bound. No y-values needed.
        'testEI': uses EI over a finite set of points. This set must be
            provided as alt_param, and the interface need not supply
            meaningful y-values.
    xi: exploration parameter suggested in some Bayesian opt. literature
    alt_param: currently only used when acq_func=='testEI'
    m: the maximum size of model; can be ignored unless passing an untrained
        SPGP or other model which doesn't already know its own size
    bounds: a tuple of (min,max) tuples specifying search bounds for each
        input dimension. Generally leads to better performance.
        Has a different interpretation when iter_bounds is True.
    iter_bounds: if True, bounds the distance that can be moved in a single
        iteration in terms of the length scale in each dimension. Uses the
        bounds variable as a multiple of the length scales, so bounds==2
        with iter_bounds==True limits movement per iteration to two length
        scales in each dimension. Generally a good idea for safety, etc.
    prior_data: input data to train the model on initially. For convenience,
        since the model can be trained externally as well.
        Assumed to be a pandas DataFrame of shape (n, dim+1) where the last
            column contains y-values.
Methods:
    acquire(): Returns the point that maximizes the acquisition function.
        For 'testEI', returns the index of the point instead.
        For normal acquisition, currently uses the bounded L-BFGS optimizer.
            Haven't tested alternatives much.
    best_seen(): Uses the model to make predictions at every observed point,
        returning the best-performing (x,y) pair. This is more robust to noise
        than returning the best observation, but could be replaced by other,
        faster methods.
    OptIter(): The main method for Bayesian optimization. Maximizes the
        acquisition function, then uses the interface to test this point and
        update the model.

# TODO callbacks or real-time acquisition needed: the minimizer for the acquisition fcn only looks for number of devices when loaded; not when devices change
2018-04-24: Need to improve hyperparam import
2018-05-23: Adding prior variance
2018-08-27: Removed prior variance
            Changed some hyper params to work more like 4/28 GOLD version
            Also updated parallelstuff with batch eval to prevent fork bombs

"""
from __future__ import absolute_import, print_function
import os # check os name
import operator as op
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.optimize import approx_fprime
#try:
    #from scipy.optimize import basinhopping
    #from parallelbasinhopping import *
    #basinhoppingQ = True
#except:
    #basinhoppingQ = False
    #pass
basinhoppingQ = False

from GP.parallelstuff import *

import time
#import math
from copy import deepcopy
import pandas as pd
import copy

from GP.heatmap import plotheatmap

def normVector(nparray):
    return nparray / np.linalg.norm(nparray)

class BayesOpt:
    def __init__(self, model, target_func, acq_func='EI', xi=0.0, alt_param=-1, m=200, bounds=None, iter_bound=False, prior_data=None, start_dev_vals=None, dev_ids=None, energy=None, hyper_file=None, corrmat=None, covarmat=None):
        self.model = model
        self.m = m
        self.bounds = bounds
        self.iter_bound = iter_bound
        self.prior_data = prior_data
        self.target_func = target_func
        try: 
            self.mi = self.target_func.mi
        except:
            self.mi = self.target_func
        print('Using ', self.mi.name) #LCLSMachineInterface, CorrplotInterface, #MultinormalInterface
        self.acq_func = (acq_func, xi, alt_param)
        self.ucb_params = [0.01, 2.]
        self.max_iter = 100
        self.check = None
        self.alpha = 1
        self.kill = False
        self.ndim = np.array(start_dev_vals).size
        self.multiprocessingQ = True # speed up acquisition function optimization
        # if os.name == 'nt': # multiprocessing doesn't work on windows
            # self.multiprocessingQ = False
        print("Bayesian optimizer set to use ", acq_func, " acquisition function")

        # DELETE AFTER PUSHING mint.GaussProcess.preprocess stuff into here
        if hyper_file == None:
            print('BayesOpt - WARNING: hyper_file = ', hyper_file)

        self.energy = energy
        self.dev_ids = dev_ids
        self.pvs = self.dev_ids
        self.pvs_ = [pv.replace(":","_") for pv in self.pvs]
        print('FIX ME?! pass mi instead of dev_ids. Might be a problem if you select GP, then change devices. Or maybe we just need to reinit BayesOpt upon device list change.')
        self.start_dev_vals = start_dev_vals

        try:
            # get initial state
            (x_init, y_init) = self.mi.getState()
            self.X_obs = np.array(x_init)
            self.Y_obs = [y_init]
            self.current_x = np.array(np.array(x_init).flatten(), ndmin=2)
            # self.ForcePoint(np.array(start_dev_vals, ndmin=2)) ??
        except:
            print('WARNING: problems in GP/BayesOptimization at line 117!!!!!!!!!!!!')
        # use identity correlation matrix if none passed
        if type(corrmat) is type(None):
            self.corrmat = np.eye(len(dev_ids))
        else:
            self.corrmat = corrmat
        
        # use diagonal covariance matrix if none passed
        if type(covarmat) is type(None):
            try:
                self.covarmat = self.model.linear_transform # DKL
            except:
                self.diaglens = np.diagflat(np.sqrt(0.5/np.exp(self.model.covar_params[0]))) # length scales (or principal widths)
                print('BO: diaglens = ', self.diaglens)
                self.covarmat = np.dot(self.diaglens,np.dot(self.corrmat,self.diaglens))
        else:
            self.covarmat = corrmat
        print('BO: self.covarmat = ', self.covarmat)
        self.invcovarmat = np.linalg.inv(self.covarmat)
        
        self.usePriorMean = False
        self.build_prior_mean()
        
        
    # call to reload the prior mean
    def build_prior_mean(self):
        if self.usePriorMean:
            self.invcovarmat = np.linalg.inv(self.covarmat)
            if self.mi.name == 'MultinormalInterface':
                self.build_prior_mean_sim()
            else:
                # prior created from fits to prior function
                try:
                    self.build_prior_mean_fitprior()
                except:
                    print('GP/BayesOptimization.py - ERROR: Could not build Bayesian prior mean for one or more devices. Make sure to select only quadrupole magnets for GP, and request additional priors if needed.')

    ## prior mean function definition (perhaps push unique copy into each interface
    #def multinormal_prior_mean(self, x, params):
        #[xpeak,dxpeak,peakFEL] = params
        #relfracdist = np.linalg.norm((x-xpeak)/dxpeak)
        #return peakFEL*np.exp(-0.5*(relfracdist**2.)) # Mitch might have flipped the sign on the

    # probably need to put the jacobian into the variance fcn.
    # consider reordering param order
    # consider passing mean and variance from same function (cleaner code but breaks compat)

    # prior mean function definition incorporating inverse covariance matrix
    def multinormal_prior_mean(self, x, params):
        [peakFEL,xpeak,invcovarmat] = params # unpack parameters
        dx = x - xpeak
        return peakFEL * np.exp(-0.5*np.dot(dx,np.dot(invcovarmat,dx.T)))

    # fit returns full covariance, so we need to pass
    # the full params in correct order + covariance matrix
    def multinormal_prior_mean_reverse(self, params, x):
        return self.multinormal_prior_mean(x, params)

    # def multinormal_prior_var(x,params,allparams,paramcovarmat,priorMeanFcnOnParams): # priorMeanOnParams is fcn of (params,x)
        # [xpeak,dxpeak,peakFEL,covar] = params # covariance is fit parameter covariance
        # ep = np.sqrt(np.finfo(float).eps) * np.sqrt(allparams)
        # jac = approx_fprime(allparam, priorMeanOnParams, ep, x)
        # fvar = np.dot(jac,np.dot(paramcovarmat,jac.T))
        # return fvar
    
    def multinormal_prior_var(self, x, params):
        return 1.
        
    def multinormal_prior_var_flat(self, x, params):
        return 1.
        
    def multinormal_prior_var_disable(self, x, params):
        return 1.e16 # huge variance => disable prior
        return np.inf
        
    def build_prior_mean_sim(self):

        # stuff for multinormal simulation interface
        if self.mi.name == 'MultinormalInterface':
            # hyperparams for multinormal simulation interface
            covar_params = np.array(np.log(0.5/(self.mi.sigmas**2)),ndmin=2)
            #noise_param = 2.*np.log((self.mi.bgNoise + self.mi.sigAmp * self.mi.sigNoiseScaleFactor) * (self.mi.noiseScaleFactor+1.e-15) / np.sqrt(self.mi.numSamples))
            noise_param = 2.*np.log(1.2*(self.mi.bgNoise + self.mi.sigAmp * self.mi.sigNoiseScaleFactor) * (self.mi.noiseScaleFactor+1.e-15))
            amp_param = np.log(1.2*self.mi.sigAmp)
            hyperparams = (covar_params, amp_param, noise_param)
            print("BO: changing hyperparams for multinormal sim ", hyperparams)
            self.model.covar_params = hyperparams[:2]
            self.model.noise_var = np.exp(hyperparams[2])

            # prior parameters for multinormal simulation interface
            #self.model.prmeanp = [startingpoint, 0.1*startingpoint, 1.] # naive prior
            #self.model.prmeanp = [startingpoint, 1.5*np.ones(ndim), 0.1]
            lengthscales = np.sqrt(0.5*np.exp(-self.model.covar_params[0]))
            prwidths = 2.*lengthscales
            ndim = lengthscales.size
            
            diaglens = np.diagflat(prwidths)
            covarmat = np.dot(diaglens,np.dot(self.corrmat,diaglens))
            prinvcovarmat = np.linalg.inv(covarmat)

            # grab centroid from offsets in simulation mode interface
            prcentroid = np.array(self.mi.offsets,ndmin=2)

        # stuff for CorrplotInterface simulation interface; should generally just use the scraped data params
        #overrideCorrplotPrior = False
        #if self.mi.name == 'CorrplotInterface' and overrideCorrplotPrior:
            #try: # prior mean centroid from corrplor sim interface
                #prcentroid = np.array(self.target_func.mi.pvs_optimum_value[0],ndmin=2)
            #except:
                #pass

        # kick prior mean centroid only in simulation modes
        usePriorKicks = True
        #if usePriorKicks and (self.mi.name == 'MultinormalInterface' or self.mi.name == 'CorrplotInterface'): # perturb centroid
        if usePriorKicks and (self.mi.name == 'MultinormalInterface'): # perturb centroid
            kick_nsigma = 1.#np.sqrt(ndim) # scales the magnitude of the distance between start and goal so that the distance has a zscore of nsigma
            kicks = np.random.randn(ndim) #1.*np.ones(ndim) # peak location is an array
            kicks = np.round(kicks*kick_nsigma/np.linalg.norm(kicks),2) #1.*np.ones(ndim) # peak location is an array
            kicks = kicks * lengthscales
            prcentroid = prcentroid + kicks
            print('BayesOpt - WARNING: using prior kicks with nsigma = ',kick_nsigma)

        # override the prior centroid with starting position
        #prcentroid = self.start_dev_vals
        #print 'WARNING: overriding prior centroid with current value'
        #print '         CONSIDER USING PRIOR CENTROID'

        # finally, set the prior amplitude
        #pramp = 0.1 * self.mi.sigAmp
        #pramp = self.mi.sigAmp
        pramp = 0.5 * self.mi.bgNoise

        # finally, compile all the prior params and LOAD THE MODEL
        print("Prior mean peak at point: ",prcentroid)
        print("Prior mean widths: ",prwidths)
        print("Prior mean covarmat: \n",covarmat)
        print("Prior mean amp: ",pramp)
        prcentroid = np.array(prcentroid,ndmin=2)
        prwidths = np.array(prwidths, ndmin=2)
        self.model.prmean = self.multinormal_prior_mean # prior mean fcn
        #self.model.prmeanp = [prcentroid, prwidths, pramp] # params of prmean fcn
        self.model.prmeanp = [pramp, prcentroid, prinvcovarmat] # params of prmean fcn
        self.model.prvar = self.multinormal_prior_var_flat
        #self.model.prvar = self.multinormal_prior_var_disable
        self.model.prvarp = []
        #print "self.model.prmean(prcentroid,self.model.prmeanp) = ",self.model.prmean(prcentroid,self.model.prmeanp)

        #self.model.prmean_name = 'multinormal_prior_mean(x,[xpeak,dxpeak,peakFEL]) =  peakFEL*np.exp(-0.5*(np.linalg.norm((x-xpeak)/dxpeak)**2.))'
        self.model.prmean_name = 'multinormal_prior_mean(x,[peakFEL,xpeak,invcovarmat]) = peakFEL * np.exp(-0.5*np.dot((x-xpeak),np.dot(invcovarmat,(x-xpeak).T)))'

    def build_prior_mean_fitprior(self):

        # prior mean function definition (perhaps push unique copy into each interface
        #self.model.prmean_name = 'multinormal_prior_mean(x,[xpeak,dxpeak,peakFEL]) =  peakFEL*np.exp(-0.5*(np.linalg.norm((x-xpeak)/dxpeak)**2.))'
        self.model.prmean_name = 'multinormal_prior_mean(x,[peakFEL,xpeak,invcovarmat]) = peakFEL * np.exp(-0.5*np.dot((x-xpeak),np.dot(invcovarmat,(x-xpeak).T)))'

        # new data reduction prior
        if self.mi.name != 'MultinormalInterface':
            self.prior_params_file = 'parameters/fit_params_2018-01_to_2018-01.pkl'
            self.prior_params_file_older = 'parameters/fit_params_2017-05_to_2018-01.pkl'

            print('Building prior from data in file ',self.prior_params_file)
            print('Filling in gaps with ',self.prior_params_file_older)

            filedata_recent = pd.read_pickle(self.prior_params_file) # recent fits
            filedata_older = pd.read_pickle(self.prior_params_file_older) # fill in sparsely scanned quads with more data from larger time range
            filedata = filedata_older
            names_recent = filedata_recent.T.keys()
            names_older = filedata_older.T.keys()
            pvs = self.pvs_

            # load in moments for prior mean from the data fits pickle
            pramps = np.array(self.start_dev_vals) * 0
            prcentroid = np.array(self.start_dev_vals) * 0
            prwidths = np.array(self.start_dev_vals) * 0
            self.model.prior_pv_info = [['i','pv','prior file','number of points fit for pv','pramp[i]','prcentroid[i]','prwidths[i]','peakFEL0','peakFEL1','kpeak0','kpeak1','dkpeak0','dkpeak1','ave_m','ave_b','width_m','width_b','ave_res_m','ave_res_b','width_res_m','width_res_b']]

            for i, pv in enumerate(pvs):
                # note: we pull data from most recent runs, but to fill in the gaps, we can use data from a larger time window
                #       it seems like the best configs change with time so we prefer recent data

                pvprlog = [i,pv]

                # use recent data unless too sparse (less than 10 points)
                if pv in names_recent and filedata_recent.get_value(pv, 'number of points fit')>10:
                    print('PRIOR STUFF: ' + pv + ' RECENT DATA LOOKS GOOD')
                    filedata = filedata_recent
                    pvprlog += [self.prior_params_file] # for logging
                    pvprlog += [filedata.get_value(pv, 'number of points fitted')]

                # fill in for sparse data with data from a larger time range
                #elif pv in names_recent and filedata.get_value(pv, 'number of points fitted')<=10:
                elif pv in names_older:
                    print('PRIOR STUFF: '+pv+' DATA TOO SPARSE <= 10 ################################################')
                    filedata = filedata_older
                    self.prior_params_file = self.prior_params_file_older
                    pvprlog += [self.prior_params_file_older] # for logging
                    pvprlog += [filedata.get_value(pv, 'number of points fitted')]
                elif pv in names_recent:
                    print('PRIOR STUFF: '+pv+' DATA TOO SPARSE <= 10 ################################################')
                    filedata = filedata_recent
                    pvprlog += [self.prior_params_file] # for logging
                    pvprlog += [filedata.get_value(pv, 'number of points fitted')]
                else:
                    print('PRIOR STUFF: WARNING WARNING WARNING WARNING ' + pv + ' NOT FOUND')
                    print('PRIOR STUFF: MIGHT WANT TO CONSIDER DEFAULT VALUES')
                    pvprlog += ['PV not found in '+self.prior_params_file+' or '+self.prior_params_file_older] # for logging
                    pvprlog += [0]


                # extract useful parameters

                peakFEL0 = filedata.get_value(pv, 'peakFEL0')
                peakFEL1 = filedata.get_value(pv, 'peakFEL1')
                kpeak0 = filedata.get_value(pv, 'kpeak0')
                kpeak1 = filedata.get_value(pv, 'kpeak1')
                dkpeak0 = filedata.get_value(pv, 'dkpeak0')
                dkpeak1 = filedata.get_value(pv, 'dkpeak1')

                # build prior parameters from above parameters

                # prior peak
                pramps[i] = (peakFEL0 + peakFEL1 * self.energy)

                # prior peak
                prcentroid[i] = (kpeak0 + kpeak1 * self.energy)

                # prior peak
                prwidths[i] = (dkpeak0 + dkpeak1 * self.energy)

                # logging prior stuff
                pvprlog += [pramps[i],prcentroid[i],prwidths[i],peakFEL0,peakFEL1,kpeak0,kpeak1,dkpeak0,dkpeak1]
                self.model.prior_pv_info += [[pvprlog]]


            # end data reduction prior

        # override the prior centroid with starting position
        #prcentroid = self.start_dev_vals
        #print 'WARNING: overriding prior centroid with current value'
        #print '         CONSIDER USING PRIOR CENTROID'

        # finally, set the prior amplitude relative to the expected
        #pramp = 0.05 * np.mean(pramps) # was 0.1
        pramp = np.mean(pramps)

        print("Prior amplitudes = ", pramps)
        print("Prior amp mean is ", np.mean(pramps), " and std is ", np.std(pramps))

        # finally, compile all the prior params and LOAD THE MODEL
        print ("Prior mean amp: ",pramp)
        print ("Prior mean centroid: ",prcentroid)
        print ("Prior mean widths: ",prwidths)
        prcentroid = np.array(prcentroid,ndmin=2)
        prwidths = np.array(prwidths, ndmin=2)
            
        # build covariance matrix with correlations
        diaglens = np.diagflat(prwidths)
        covarmat = np.dot(diaglens,np.dot(self.corrmat,diaglens))
        prinvcovarmat = np.linalg.inv(covarmat)
        print ("Prior mean covarmat: \n",covarmat)
        
        self.model.prmean = self.multinormal_prior_mean # prior mean fcn
        #self.model.prmeanp = [prcentroid, prwidths, pramp] # params of prmean fcn
        self.model.prmeanp = [pramp, prcentroid, prinvcovarmat] # params of prmean fcn
        self.model.prvar = self.multinormal_prior_var_flat
        self.model.prvarp = []
        #print "self.model.prmean(prcentroid,self.model.prmeanp) = ",self.model.prmean(prcentroid,self.model.prmeanp)


    def terminate(self, devices):
        """
        Sets the position back to the location that seems best in hindsight.
        It's a good idea to run this at the end of the optimization, since
        Bayesian optimization tries to explore and might not always end in
        a good place.
        """
        print("TERMINATE", self.x_best)
        if(self.acq_func[0] == 'EI'):
            # set position back to something reasonable
            for i, dev in enumerate(devices):
                dev.set_value(self.x_best[i])
            #error_func(self.x_best)
        if(self.acq_func[0] == 'UCB'):
            # UCB doesn't keep track of x_best, so find it
            (x_best, y_best) = self.best_seen()
            for i, dev in enumerate(devices):
                dev.set_value(x_best[i])


    def minimize(self, error_func, x):
        # weighting for exploration vs exploitation in the GP at the end of scan, alpha array goes from 1 to zero
        #alpha = [1.0 for i in range(40)]+[np.sqrt(50-i)/3.0 for i in range(41,51)]
        inverse_sign = -1
        self.current_x = np.array(np.array(x).flatten(), ndmin=2)
        #self.current_y = [np.array([[inverse_sign*error_func(x)]])]
        self.X_obs = np.array(self.current_x)
        self.Y_obs = [np.array([[inverse_sign*error_func(x)]])]
        # iterate though the GP method
        #print("GP minimize",  error_func, x, error_func(x))
        for i in range(self.max_iter):
            # get next point to try using acquisition function
            x_next = self.acquire(self.alpha)
            #check for problems with the beam
            if self.check != None: self.check.errorCheck()

            y_new = error_func(x_next.flatten())
            #if self.kill:
            if self.opt_ctrl.kill:
                print ('Killing Bayesian optimizer...')
                #disable so user does not start another scan while the data is being saved
                break
            y_new = np.array([[inverse_sign *y_new]])

            #advance the optimizer to the next iteration
            #self.opt.OptIter(alpha=alpha[i])
            #self.OptIter() # no alpha

            # change position of interface and get resulting y-value

            x_new = deepcopy(x_next)
            #(x_new, y_new) = self.mi.getState()
            self.current_x = x_new
            #self.current_y = y_new

            # add new entry to observed data
            self.X_obs = np.concatenate((self.X_obs, x_new), axis=0)
            self.Y_obs.append(y_new)

            # update the model (may want to add noise if using testEI)
            self.model.update(x_new, y_new)# + .5*np.random.randn())

    def OptIter(self,pause=0):
        # runs the optimizer for one iteration
    
        # get next point to try using acquisition function
        x_next = self.acquire()
        if(self.acq_func[0] == 'testEI'):
            ind = x_next
            x_next = np.array(self.acq_func[2].iloc[ind,:-1],ndmin=2)
            
        #print
        #print "BayesOpt.OptIter - setting next point to acquire"
        #print(self.X_obs)
        #print(self.Y_obs)
        
        # change position of interface and get resulting y-value
        self.mi.setX(x_next)
        if(self.acq_func[0] == 'testEI'):
            (x_new, y_new) = (x_next, self.acq_func[2].iloc[ind,-1])
        else:
            (x_new, y_new) = self.mi.getState()
        # add new entry to observed data
        self.X_obs = np.concatenate((self.X_obs,x_new),axis=0)
        self.Y_obs.append(y_new)
        #print("self.X_obs = " + str(self.X_obs))
        #print("self.Y_obs = " + str(self.Y_obs))
        
        #print(self.X_obs)
        #print(self.Y_obs)
        
        # update the model (may want to add noise if using testEI)
        self.model.update(x_new, y_new)# + .5*np.random.randn())
            
            
    def ForcePoint(self,x_next):
        # force it to take a point at my discretion and update the model
    
        # need x_next to adhere to the format self.acquire() produces
        
        # change position of interface and get resulting y-value
        self.mi.setX(x_next)
        if(self.acq_func[0] == 'testEI'):
            (x_new, y_new) = (x_next, self.acq_func[2].iloc[ind,-1])
        else:
            (x_new, y_new) = self.mi.getState()
        # add new entry to observed data
        self.X_obs = np.concatenate((self.X_obs,x_new),axis=0)
        self.Y_obs.append(y_new)
        #print("self.X_obs = " + str(self.X_obs))
        #print("self.Y_obs = " + str(self.Y_obs))
        
        # update the model (may want to add noise if using testEI)
        self.model.update(x_new, y_new) # + .5*np.random.randn())

    def best_seen(self):
        """
        Checks the observed points to see which is predicted to be best.
        Probably safer than just returning the maximum observed, since the
        model has noise. It takes longer this way, though; you could
        instead take the model's prediction at the x-value that has
        done best if this needs to be faster.

        Not needed for UCB so do it the fast way (return max obs)
        """
        if(self.acq_func[0] == 'UCB'):
            mu = self.Y_obs
        else:
            (mu, var) = self.model.predict(self.X_obs)
            mu = [self.model.predict(np.array(x,ndmin=2))[0] for x in self.X_obs]

        #(mu2, var2) = self.model.predict(self.X_obs)
        #print 'self.X_obs = ', self.X_obs
        #print 'mu2 = ', mu2
        #print 'mu = ', mu
        #print [len(m) for m in mu2]
        #print [self.model.predict(np.array(x,ndmin=2))[0] for x in self.X_obs]

        (ind_best, mu_best) = max(enumerate(mu), key=op.itemgetter(1))
        return (self.X_obs[ind_best], mu_best)
        #return (np.array(self.X_obs[ind_best], ndmin=2), mu_best)

    def acquire(self, alpha=1.):
                
        # print 'self.model.prmean = ', self.model.prmean
        # print 'self.model.prmeanp = ', self.model.prmeanp
        # print 'self.model.prvar = ', self.model.prvar
        # print 'self.model.prvarp = ', self.model.prvarp
        """
        Computes the next point for the optimizer to try by maximizing
        the acquisition function. If movement per iteration is bounded,
        starts search at current position.
        """
        # look from best positions
        (x_best, y_best) = self.best_seen() # sort of a misnomer (see function best_seen)
        self.x_best = x_best
        x_curr = self.current_x[-1]
        #y_curr = self.current_y[-1]
        #x_start = x_curr
        x_start = x_best

        # calculate length scales
        try:
            lengthscales = np.sqrt(0.5*np.exp(-self.model.covar_params[0][0])) # length scales from covar params
        except:
            lengthscales = np.sqrt(np.diag(self.covarmat))
        print ('lengthscales = ', lengthscales)
        ndim = x_curr.size # dimension of the feature space we're searching NEEDED FOR UCB
        try:
            nsteps = 1 + self.X_obs.shape[0] # acquisition number we're on  NEEDED FOR UCB
        except:
            nsteps = 1

        #print "self.x_best = " + str(x_best)
        #print "self.current_x = " + str(self.current_x)
        #print "self.current_x[-1] = " + str(self.current_x[-1])

        # check to see if this is bounding step sizes
        if(self.iter_bound or True):
            if(self.bounds is None): # looks like a scale factor
                self.bounds = 1.0

            bound_lengths = 3. * lengthscales # 3x hyperparam lengths
            relative_bounds = np.transpose(np.array([-bound_lengths, bound_lengths]))
            
            #iter_bounds = np.transpose(np.array([x_start - bound_lengths, x_start + bound_lengths]))
            iter_bounds = np.transpose(np.array([x_start - bound_lengths, x_start + bound_lengths]))

        else:
            iter_bounds = self.bounds

        #print "x_start = " + str(x_start)
        #print "BayesOpt.acquire - self.model.covar_params = " + str(self.model.covar_params)
        #print "self.model.covar_params[0] = " + str(self.model.covar_params[0])
        #print "iter_bounds = " + str(iter_bounds)

        # options for finding the peak of the acquisition function:
        optmethod = 'L-BFGS-B' # these 4 allow bounds
        #optmethod = 'BFGS'
        #optmethod = 'TNC'
        #optmethod = 'SLSQP'
        #optmethod = 'Powell' # these 2 don't
        #optmethod = 'COBYLA'
        maxiter = 1000 # max number of steps for one scipy.optimize.minimize call
        nproc = mp.cpu_count() # number of processes to launch minimizations on
        niter = 1 # max number of starting points for search
        niter_success = 1 # stop search if same minima for 10 steps
        tolerance = 1.e-4 # goal tolerance
        #nproc = 5*mp.cpu_count() # number of processes to launch minimization

        # perturb start to break symmetry
        #x_start += np.random.randn(lengthscales.size)*lengthscales*1e-6

        # probability of improvement acquisition function
        if(self.acq_func[0] == 'PI'):
            print ('Using PI')
            aqfcn = negProbImprove
            fargs=(self.model, y_best, self.acq_func[1])

        # expected improvement acquisition function
        elif(self.acq_func[0] == 'EI'):
            print ('Using EI')
            aqfcn = negExpImprove
            fargs = (self.model, y_best, self.acq_func[1], alpha)

        # gaussian process upper confidence bound acquisition function
        elif(self.acq_func[0] == 'UCB'):
            print ('Using UCB')
            aqfcn = negUCB
            fargs = (self.model, ndim, nsteps, self.ucb_params[0], self.ucb_params[1])

        # maybe something mitch was using once? (can probably remove)
        elif(self.acq_func[0] == 'testEI'):
            # collect all possible x values
            options = np.array(self.acq_func[2].iloc[:, :-1])
            (x_best, y_best) = self.best_seen()

            # find the option with best EI
            best_option_score = (-1,1e12)
            for i in range(options.shape[0]):
                result = negExpImprove(options[i],self.model,y_best,self.acq_func[1])
                if(result < best_option_score[1]):
                    best_option_score = (i, result)

            # return the index of the best option
            return best_option_score[0]

        else:
            print('Unknown acquisition function.')
            return 0

        try:
            # manual scan for diagnostics
            if False:
                nmax = 15.
                scale = 3.
                for i in scale*np.linspace(-1,1,nmax):
                    x = x_start + i * np.array(lengthscales,ndmin=2)
                    (y_mean, y_var) = self.model.predict(np.array(x, ndmin=2))
                    print (i,x,y_mean,y_var,negExpImprove(x,self.model, y_best, self.acq_func[1], alpha))

            print ('iter_bounds = ',iter_bounds)
            #print 'len(lengthscales) = ', len(lengthscales)

            # plot heatmaps
            if True and len(lengthscales) == 2:

                print('Plotting heat maps.')

                #center_point = self.x_start # moving view
                center_point = self.start_dev_vals # static view
                rangex = center_point[0] + 5 * lengthscales[0] * np.array([-1,1]) #+ x_start[0]
                rangey = center_point[1] + 5 * lengthscales[1] * np.array([-1,1]) #+ x_start[1]

                try:
                    plotheatmap(self.model.predict,(),rangex,rangey,series=self.model.BV)
                except Exception as e:
                    print ('Could not print prediction heatmap. Exception: ', e)
                    pass

                try:
                    plotheatmap(aqfcn,fargs,rangex,rangey,series=self.model.BV)
                except Exception as e:
                    print ('Could not print acquisition heatmap. Exception: ', e)
                    pass

            if(self.multiprocessingQ):

                neval = 2*int(10.*2.**(ndim/12.))
                nkeep = 2*min(4,neval)

                print ('neval = ', neval,'\t nkeep = ',nkeep)

                ## parallelgridsearch generates pseudo-random grid, then performs an ICDF transform
                ## to map to multinormal distrinbution centered on x_start and with widths given by hyper params
                #v0s = parallelgridsearch(aqfcn,x_start,0.6*lengthscales,fargs,neval,nkeep)
                #x0s = v0s[:,:-1] # for later testing if the minimize results are better than the best starting point
                #v0best = v0s[0]
                ##x0s = parallelgridsearch(aqfcn,x_start,lengthscales,fargs,max(1,int(neval/2)),max(1,int(nkeep/2)))
                ##x0s = np.vstack((x0s,parallelgridsearch(aqfcn,x_start,0.5*lengthscales,fargs,max(1,int(neval/2)),max(1,int(nkeep/2)))))

                # add the 10 best points seen so far (largest Y_obs)
                nbest = 3 # add the best points seen so far (largest Y_obs)
                nstart = 2 # make sure some starting points are there to prevent run away searches
                yobs = np.array([y[0][0] for y in self.Y_obs])
                isearch = yobs.argsort()[-nbest:]
                for i in range(min(nstart,len(self.Y_obs))): #
                    if np.sum(isearch == i) == 0: # not found in list
                        isearch = np.append(isearch, i)
                isearch.sort() # sort to bias searching near earlier steps
                #isearch = isearch[::-1]
                #print 'isearch = ', isearch
                #print 'self.X_obs = ', self.X_obs
                #print 'self.Y_obs = ', self.Y_obs
                v0s = None
                for i in isearch:
                    vs = parallelgridsearch(aqfcn,self.X_obs[i],0.6*lengthscales,fargs,neval,nkeep)
                    if type(v0s) == type(None):
                        v0s = copy.copy(vs)
                    else:
                        v0s = np.vstack((v0s,vs))

                #print 'v0s = ', v0s

                v0sort = v0s[:,-1].argsort()[:nkeep] # keep the nlargest
                #print 'v0sort = ', v0sort
                #print 'v0s[v0sort] = ', v0s[v0sort]
                v0s = v0s[v0sort]
                
                x0s = v0s[:,:-1] # for later testing if the minimize results are better than the best starting point
                v0best = v0s[0]
                #x0s = parallelgridsearch(aqfcn,x_start,lengthscales,fargs,max(1,int(neval/2)),max(1,int(nkeep/2)))
                #x0s = np.vstack((x0s,parallelgridsearch(aqfcn,x_start,0.5*lengthscales,fargs,max(1,int(neval/2)),max(1,int(nkeep/2)))))

                
                #print 'self.X_obs = \n', self.X_obs, '\n'
                #print 'self.Y_obs = \n', self.Y_obs, '\n'
                #yobs = np.array([y[0][0] for y in self.Y_obs])
                #print 'yobs = \n', yobs, '\n'
                #print 'yobs.argsort() = \n', yobs.argsort(), '\n'
                #print 'yobs.argsort()[-5:] = \n', yobs.argsort()[-5:], '\n'
                #print 'self.X_obs[0] = \n', self.X_obs[0], '\n'
                #print 'type(self.X_obs[0]) = \n', type(self.X_obs[0]), '\n'
                #xobs = [self.X_obs[i] for i in yobs.argsort()[-5:]] # take the best 5
                #print 'xobs = \n', xobs, '\n'
                
                # add more points to search from
                #x0s = np.vstack((x0s,np.array(x_curr))) # last point
                #x0s = np.vstack((x0s,np.array(x_best))) # best so far
                ## add the 10 best points seen so far (largest Y_obs)
                #yobs = np.array([y[0][0] for y in self.Y_obs])
                #for i in yobs.argsort()[-10:]:
                    #x0s = np.vstack((x0s,np.array(self.X_obs[i])))
                
                print ('x0s = ', x0s)
                print ('fargs = ', fargs)

                if basinhoppingQ:
                    # use basinhopping
                    bkwargs = dict(niter=niter,niter_success=niter_success, minimizer_kwargs={'method':optmethod,'args':fargs,'tol':tolerance,'bounds':iter_bounds,'options':{'maxiter':maxiter}}) # keyword args for basinhopping
                    res = parallelbasinhopping(aqfcn,x0s,bkwargs)

                else:
                    # use minimize
                    mkwargs = dict(bounds=iter_bounds, method=optmethod, options={'maxiter':maxiter}, tol=tolerance) # keyword args for scipy.optimize.minimize
                    res = parallelminimize(aqfcn,x0s,fargs,mkwargs,v0best,relative_bounds=relative_bounds)
                print ('mkwargs = ', mkwargs)
                print ('res = ', res)

            else: # single-processing
                if basinhoppingQ:
                    res = basinhopping(aqfcn, x_start,niter=niter,niter_success=niter_success, minimizer_kwargs={'method':optmethod,'args':(self.model, y_best, self.acq_func[1], alpha),'tol':tolerance,'bounds':iter_bounds,'options':{'maxiter':maxiter}})

                else:
                    res = minimize(aqfcn, x_start, args=(self.model, y_best, self.acq_func[1], alpha), method=optmethod,tol=tolerance,bounds=iter_bounds,options={'maxiter':maxiter})

                res = res.x
                # end else
            #print 'res = ',res
        except:
            raise
        return np.array(res,ndmin=2) # return resulting x value as a (1 x dim) vector

# why is this class declared in BayesOptimization.py???
class HyperParams:
    def __init__(self, pvs, filename, mi=None):
        self.pvs = pvs
        print ('HyperParams = ',self.pvs)
        self.filename = filename
        self.mi = mi
        pass

    def loadSeedData(self,filename, target):
        """ Load in the seed data from a matlab ocelot scan file.

        Input file should formated like OcelotInterface file format.
        ie. the datasets that are saved into the matlab data folder.

        Pulls out the vectors of data from the save file.
        Sorts them into the same order as this scanner objects pv list.
        The GP wont work if the data is in the wrong order and loaded data is not ordered.

        Args:
                filename (str): String for the .mat file directory.

        Returns:
                Matrix of ordered data for GP. [ len(num_iterations) x len(num_devices) ]
        """
        print()
        dout = []
        if type(filename) == type(''):
            print ("Loaded seed data from file:",filename)
            #stupid messy formating to unest matlab format
            din = scipy.io.loadmat(str(filename))['data']
            names = np.array(din.dtype.names)
            for pv in self.pvs:
                pv = pv.replace(":","_")
                if pv in names:
                    x = din[pv].flatten()[0]
                    x = list(chain.from_iterable(x))
                    dout.append(x)

            #check if the right number of PV were pulled from the file
            if len(self.pvs) != len(dout):
                print ("The seed data file device length unmatched with scan requested PVs!")
                print ('PV len         = ',len(self.pvs))
                print ('Seed data len = ',len(dout))
                self.parent.scanFinished()

            #add in the y values
            #ydata = din[self.objective_func_pv.replace(':','_')].flatten()[0]
            ydata = din[target.replace(':','_')].flatten()[0]
            dout.append(list(chain.from_iterable(ydata)))

        # If passing seed data from a seed scan
        else:
            print ("Loaded Seed Data from Seed Scan:")
            din = filename
            for pv in self.pvs:
                if pv in din.keys():
                    dout.append(din[pv])
            dout.append(din[target])
            #dout.append(din[target])
                    #dout.append(din[target])

        #transpose to format for the GP
        dout = np.array(dout).T

        #dout = dout.loc[~np.isnan(dout).any(axis=1),:]
        dout = dout[~np.isnan(dout).any(axis=1)]

        #prints for debug
        print ("[device_1, ..., device_N] detector")
        print (self.pvs,target)
        print (dout)
        print()

        return dout

    def extract_hypdata(self, energy):
        key = str(energy)
        f = np.load(str(self.filename), fix_imports=True, encoding='latin1')
        filedata = f[0][key]
        return filedata
        
    def loadSimHyperParams(self, filename, energy, detector, pvs, vals, multiplier = 1.):
            # hyperparams for multinormal simulation interface
            covar_params = np.array(np.log(0.5/(self.mi.sigmas**2)),ndmin=2)
            #noise_param = 2.*np.log((self.mi.bgNoise + self.mi.sigAmp * self.mi.sigNoiseScaleFactor) * (self.mi.noiseScaleFactor+1.e-15) / np.sqrt(self.mi.numSamples))
            noise_param = 2.*np.log((self.mi.bgNoise + self.mi.sigAmp * self.mi.sigNoiseScaleFactor) * (self.mi.noiseScaleFactor+1.e-15))
            amp_param = np.log(self.mi.sigAmp)
            hyperparams = (covar_params, amp_param, noise_param)
            
            return hyperparams

    #def loadHyperParams(self, filename, energy, detector, pvs, multiplier = 1):
    def loadHyperParams(self, filename, energy, detector, pvs, vals, multiplier = 1.):
        """
        Method to load in the hyperparameters from a .npy file.
        Sorts data, ordering parameters with this objects pv list.
        Formats data into tuple format that the GP model object can accept.
        ( [device_1, ..., device_N ], coefficent, noise)
        Args:
                filename (str): String for the file directory.
                energy:
        Returns:
                List of hyperparameters, ordered using the UI's "self.pvs" list.
        """
        
        # stuff for multinormal simulation interface
        if self.mi.name == 'MultinormalInterface':
            return self.loadSimHyperParams(filename, energy, detector, pvs, vals, multiplier = 1.)
        
        #Load in a npy file containing hyperparameters binned for every 1 GeV of beam energy
        #get current L3 beam energy
        # if len(energy) is 3: key = energy[0:1]
        # if len(energy) is 4: key = energy[0:2]
        #key = str(int(round(energy)))
        #print "Loading raw data for",key,"GeV from",filename
        #f = np.load(str(filename)); filedata0 = f[0][key]; names0 = filedata0.keys()
        #print energy, names0
        #filedata = filedata0

        # scrapes
        self.prior_params_file = 'parameters/fit_params_2018-01_to_2018-01.pkl'
        self.prior_params_file_older = 'parameters/fit_params_2017-05_to_2018-01.pkl'
        print ('Building hyper params from data in file ', self.prior_params_file)
        print ('Next, filling in gaps with ', self.prior_params_file_older)
        print ('Next, filling in gaps with ', filename)
        print ('Finally, filling in gaps with estimate from starting point and limits')
        filedata_recent = pd.read_pickle(self.prior_params_file) # recent fits
        filedata_older = pd.read_pickle(self.prior_params_file_older) # fill in sparsely scanned quads with more data from larger time range
        names_recent = filedata_recent.T.keys()
        names_older = filedata_older.T.keys()
        # pvs = [pv.replace(":","_") for pv in pvs]

        # store results
        hyps = [] # hyper params
        peakFELs = [] # for estimating the goal from historical values

        # calculate the length scales
        for i, pv in enumerate(pvs):
            # note: we pull data from most recent runs, but to fill in the gaps, we can use data from a larger time window
            #       it seems like the best configs change with time so we prefer recent data

            pv_ = pv.replace(":","_")

            # pv is in the data scrapes
            if pv_ in names_recent or pv_ in names_older:

                # use recent data unless too sparse (less than 10 points)
                if pv_ in names_recent and filedata_recent.get_value(pv_, 'number of points fitted')>10:
                    print('Hyperparams: ' + pv + ' RECENT DATA LOOKS GOOD')
                    filedata = filedata_recent
                # fill in for sparse data with data from a larger time range
                elif pv_ in names_older:
                    print('Hyperparams: '+pv+' DATA TOO SPARSE <= 10 ################################################')
                    filedata = filedata_older
                else:
                    print('Hyperparams: '+pv+' DATA TOO SPARSE <= 10 ################################################')
                    filedata = filedata_recent

                # calculate hyper width
                width_m = filedata.get_value(pv_, 'width slope')
                width_b = filedata.get_value(pv_, 'width intercept')
                # width_res_m = filedata.get_value(pv, 'width residual slope')
                # width_res_b = filedata.get_value(pv, 'width residual intercept')
                pvwidth = (width_m * energy + width_b) # prior widths
                hyp = np.log(0.5/(pvwidth**2))
                hyps.append(hyp)
                print ("calculated hyper param from scrapes:", pv, pvwidth, hyp)

                # estimate FEL response for amp param
                peakFEL0 = filedata.get_value(pv_, 'peakFEL0')
                peakFEL1 = filedata.get_value(pv_, 'peakFEL1')
                peakFELs += [peakFEL0 + peakFEL1 * energy] # store for average or max later

            # data is not in the scrapes so check if in the
            elif pv in names0:
                print ('WARNING: Using length scale from ', filename)
                ave = float(filedata[pv][0])
                std = float(filedata[pv][1])
                hyp = (self.calcLengthScaleHP(ave, std, multiplier = multiplier))
                hyps.append(hyp)
                print ("calculated hyper param from operator list:", pv, ave, std, hyp)

            # default to estimate from limits
            else:
                try:
                    print ('WARNING: for now, default length scale is calculated in some weird legacy way. Should calculate from range and starting value.')
                    ave = float(vals[i])
                    std = np.sqrt(abs(ave))
                    hyp = (self.calcLengthScaleHP(ave, std, multiplier = multiplier))
                    hyps.append(hyp)
                    print ("calculated hyper param from Mitch's function:", pv, ave, std, hyp)
                    print ('calculated from values: ', float(vals[i]))
                except:
                    print ('WARNING: Defaulting to 1 for now... (should estimate from starting value and limits in the future)')
                    hyp = np.log(0.5/1.**2)
                    hyps.append(hyp)
                    print ("calculated hyper param from default:", pv, 1, hyp)

        # estimate the amp and variance hyper params

        obj_func = detector.get_value() # get the current mean and std of the chosen detector
        print ('obj_func = ',obj_func)
        try:
            #std = np.std(  obj_func[(2799-5*120):-1])
            #ave = np.mean( obj_func[(2799-5*120):-1])
            #std = np.std(  obj_func[-120:])
            #ave = np.mean( obj_func[-120:])
            # SLACTarget.get_value() returns tuple with elements stat, stdev, ...
            ave = obj_func[0]
            std = obj_func[1]
        except:
            print ("Detector is not a waveform, Using scalar for hyperparameter calc")
            print ("Also check GP/BayesOptimization.py:HyperParams.loadHyperParams near line 722")
            ave = obj_func
            # Hard code in the std when obj func is a scalar
            # Not a great way to deal with this, should probably be fixed
            std = 0.1

        # print('WARNING: overriding amplitude and variance hyper params')
        # ave = 1.
        # std = 0.1

        try:
            ave = np.min([2,2.*np.max(peakFELs)]) # 100% more
            #ave = np.max(peakFELs)
            print ('INFO: using max of prior peakFELs for amp: max(', peakFELs, ')=', ave)
        except:
            ave = 6. # most mJ we've ever seen
            print ('WARNING: using ', ave, ' mJ (most weve ever seen) for amp')

        coeff = np.log(ave) # hyper amp
        noise = 2.*np.log(std) # hyper noise
        #noise = 2.*np.log(std/np.sqrt(120.))

        print ('DETECTOR AMP = ', ave, ' and hyper amp = ', coeff)
        print ('DETECTOR STD = ', std, ' and hyper variance = ', noise)
        
        dout = ( np.array([hyps]), coeff, noise )
        #prints for debug
        print()
        print ("Calculated Hyperparameters ( [device_1, ..., device_N ], amplitude coefficent, noise coefficent)")
        print()
        for i in range(len(hyps)):
            print(self.pvs[i], hyps[i])
        print ("AMP COEFF   = ", coeff)
        print ("NOISE COEFF = ", noise)
        print()
        return dout

    def calcLengthScaleHP(self, ave, std, c = 1.0, multiplier = 1, pv = None):
        """
        Method to calculate the GP length scale hyperparameters using history data
        Formula for hyperparameters are from Mitch and some papers he read on the GP.
        Args:
                ave (float): Mean of the device, binned around current machine energy
                std (float): Standard deviation of the device
                c   (float): Scaling factor to change the output to be larger or smaller, determined empirically
                pv  (str): PV input string to scale hyps depending on pv, not currently used
        Returns:
                Float of the calculated length scale hyperparameter
        """
        #for future use
        if pv is not None:
            #[pv,val]
            pass
        #+- 1 std around the mean
        #hi  = ave+std
        #lo  = ave-std
        #hyp = -2*np.log( ( ( multiplier*c*(hi-lo) ) / 4.0 ) + 0.01 )
        hyp = -2*np.log( ( ( multiplier*c*std ) / 2.0 ) + 0.01 )
        return hyp

    def calcAmpCoeffHP(self, ave, std, c = 0.5):
        """
        Method to calculate the GP amplitude hyperparameter
        Formula for hyperparameters are from Mitch and some papers he read on the GP.
        First we tried using the standard deviation to calc this but we found it needed to scale with mean instead
        Args:
                ave (float): Mean of of the objective function (GDET or something else)
                std (float): Standard deviation of the objective function
                c (float): Scaling factor to change the output to be larger or smaller, determined empirically
        Returns:
                Float of the calculated amplitude hyperparameter
        """
        #We would c = 0.5 to work well, could get changed at some point
        hyp2 = np.log( ( ((c*ave)**2) + 0.1 ) )
        #hyp2 = np.log( ave + 0.1 )
        return hyp2

    def calcNoiseHP(self, ave, std, c = 1.0):
        """
        Method to calculate the GP noise hyperparameter
        Formula for hyperparameters are from Mitch and some papers he read on the GP.
        Args:
                ave (float): Mean of of the objective function (GDET or something else)
                std (float): Standard deviation of the objective function
                c (float): Scaling factor to change the output to be larger or smaller, determined empirically
        Returns:
                Float of the calculated noise hyperparameter
        """
        hyp = np.log((c*std / 4.0) + 0.01)
        #hyp = np.log(std + 0.01)
        return hyp

    # end BayesOpt class

def negProbImprove(x_new, model, y_best, xi):
    """
    The probability of improvement acquisition function. Initial testing
    shows that it performs worse than expected improvement acquisition
    function for 2D scans (at least when alpha==1 in the fcn below). Alse
    performs worse than EI according to the literature.
    """
    (y_mean, y_var) = model.predict(np.array(x_new,ndmin=2))
    diff = y_mean - y_best - xi
    if(y_var == 0):
        return 0.
    else:
        Z = diff / np.sqrt(y_var)

    return -norm.cdf(Z)

def negExpImprove(x_new, model, y_best, xi, alpha=1.0):
    """
    The common acquisition function, expected improvement. Returns the
    negative for the minimizer (so that EI is maximized). Alpha attempts
    to control the ratio of exploration to exploitation, but seems to not
    work well in practice. The terminate() method is a better choice.
    """
    (y_mean, y_var) = model.predict(np.array(x_new, ndmin=2))
    diff = y_mean - y_best - xi

    # Nonvectorizable. Can prob use slicing to do the same.
    if(y_var == 0):
        return 0.
    else:
        Z = diff / np.sqrt(y_var)

    EI = diff * norm.cdf(Z) + np.sqrt(y_var) * norm.pdf(Z)
    #print(x_new, EI)
    return alpha * (-EI) + (1. - alpha) * (-y_mean)

# old version
#def negUCB(x_new, model, mult):
    #"""
    #The upper confidence bound acquisition function. Currently only partially
    #implemented. The mult parameter specifies how wide the confidence bound
    #should be, and there currently is no way to compute this parameter. This
    #acquisition function shouldn't be used until there is a proper mult.
    #"""
    #(y_new, var) = model.predict(np.array(x_new,ndmin=2))

    #UCB = y_new + mult * np.sqrt(var)
    #return -UCB

# GP upper confidence bound
# original paper: https://arxiv.org/pdf/0912.3995.pdf
# tutorial: http://www.cs.ubc.ca/~nando/540-2013/lectures/l7.pdf
def negUCB(x_new, model, ndim, nsteps, nu = 1., delta = 1.):
    """
    GPUCB: Gaussian process upper confidence bound aquisition function
    Default nu and delta hyperparameters theoretically yield "least regret".
    Works better than "expected improvement" (for alpha==1 above) in 2D.

    input params
    x_new: new point in the dim-dimensional space the GP is fitting
    model: OnlineGP object
    ndim: feature space dimensionality (how many devices are varied)
    nsteps: current step number counting from 1
    nu: nu in the tutorial (see above)
    delta: delta in the tutorial (see above)
    """

    #ndim = model.nin # problem space dimensionality
    #nsteps = model.nupdates + 1 # current step number
    if nsteps==0: nsteps += 1
    (y_mean, y_var) = model.predict(np.array(x_new,ndmin=2))

    tau = 2.*np.log(nsteps**(0.5*ndim+2.)*(np.pi**2.)/3./delta)
    #print 'y_mean = ', y_mean, '\tnu = ', nu, '\ttau = ', tau, '\ty_var = ', y_var
    GPUCB = y_mean + np.sqrt(nu * tau * y_var)

    return -GPUCB

# Thompson sampling
