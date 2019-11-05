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

# TODO callbacks or real-time acquisition needed: appears that the minimizer for the acquisition fcn only looks for number of devices when loaded; not when devices change
2018-04-24: Need to improve hyperparam import
2018-05-23: Adding prior variance
2018-08-27: Removed prior variance
            Changed some hyper params to work more like 4/28 GOLD version
            Also updated parallelstuff with batch eval to prevent fork bombs
2018-11-11: Added correlations; last step: need to strip the logs out of everything
2018-11-15: Hunted down factors of 0.5 & 2 in the hyperparameters and corrected for consistency. Also changed to regress using standard error of mean instead of standard deviation of the samples.
"""

import os # check os name
import operator as op
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.optimize import approx_fprime
try:
    from scipy.optimize import basinhopping
    basinhoppingQ = True
except:
    basinhoppingQ = False
try:
    from .parallelstuff import *
    multiprocessingQ = True
    basinhoppingQ = False
except:
    multiprocessingQ = False
import time
from copy import deepcopy, copy

def normVector(nparray):
    return nparray / np.linalg.norm(nparray)

class BayesOpt:
    def __init__(self, model, target_func, acq_func='EI', xi=0.0, alt_param=-1, m=200, bounds=None, iter_bound=False, prior_data=None, start_dev_vals=None, dev_ids=None, searchBoundScaleFactor=None):
        self.model = model
        self.m = m
        self.bounds = bounds
        self.searchBoundScaleFactor = 1.
        if type(searchBoundScaleFactor) is not type(None):
            try:
                self.searchBoundScaleFactor = abs(searchBoundScaleFactor)
            except:
                print(('BayesOpt - ERROR: ', searchBoundScaleFactor, ' is not a valid searchBoundScaleFactor (scaling coeff).'))
        self.iter_bound = iter_bound 
        self.prior_data = prior_data # for seeding the GP with data acquired by another optimizer
        self.target_func = target_func
        print('target_func = ', target_func)
        try: 
            self.mi = self.target_func.mi
            print('********* BO - self.mi = self.target_func.mi wORKED!')
        except:
            self.mi = self.target_func
            print('********* BO - self.mi = self.target_func wORKED!')
        self.acq_func = (acq_func, xi, alt_param)
        #self.ucb_params = [0.01, 2.] # [nu,delta]
        self.ucb_params = [0.002, 0.4] # [nu,delta] we like for lcls2
        self.ucb_params = [2.0, None] # [nu,delta] theortical values
        #self.ucb_params = [0.007, 1.0] # [nu,delta]
        self.max_iter = 100
        self.check = None
        self.alpha = 1
        self.kill = False
        self.ndim = np.array(start_dev_vals).size
        self.multiprocessingQ = multiprocessingQ # speed up acquisition function optimization

        #Post-edit
        if self.mi.name == 'MultinormalInterface':
            self.dev_ids = self.mi.pvs[:-1] # last pv is objective
            self.start_dev_vals = self.mi.x
        else:
            self.dev_ids = dev_ids
            self.start_dev_vals = start_dev_vals
        self.pvs = self.dev_ids
        self.pvs_ = [pv.replace(":","_") for pv in self.pvs]

        try:
            # get initial state
            (x_init, y_init) = self.getState()
            print('Supposed to be grabbing machine state...')
            print(x_init)
            print(y_init)
            self.X_obs = np.array(x_init)
            self.Y_obs = [y_init]
            self.current_x = np.array(np.array(x_init).flatten(), ndmin=2)
        except:
            print('BayesOpt - ERROR: Could not grab initial machine state')
        
        # calculate length scales
        try:
            # length scales from covar params
            cp = self.model.covar_params[0]
            cps = np.shape(cp)
            lengthscales = np.sqrt(1./np.exp(cp))
            if np.size(cps) == 2:
                if cps[0] < cps[1]: # vector of lengths
                    self.lengthscales = lengthscales.flatten()
                else: # matrix of lengths
                    self.lengthscales = np.diag(lengthscales)
        except:
            print('WARNING - GP.bayesian_optimization.BayesOpt: Using some unit length scales cause we messed up somehow...')
            self.lengthscales = np.ones(len(self.dev_ids))
        
        # initialize the prior 
        self.model.prmean = None # prior mean fcn
        self.model.prmeanp = None # params of prmean fcn
        self.model.prvar = None
        self.model.prvarp = None
        self.model.prmean_name = ''
        
    def getState(self):
        print('>>>>>>>> getState')
        x_vals = [self.mi.get_value(d) for d in self.dev_ids]
        print('>>>>>>>>>>>>>>>>>>>> invoking get_penalty')
        y_val = -self.target_func.get_penalty()
        print('>>>>>>>>>>>>> getState returning')
        return x_vals, y_val


    def terminate(self, devices):
        """
        Sets the position back to the location that seems best in hindsight.
        It's a good idea to run this at the end of the optimization, since
        Bayesian optimization tries to explore and might not always end in
        a good place.
        """
        print(("TERMINATE", self.x_best))
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
        inverse_sign = -1
        self.current_x = np.array(np.array(x).flatten(), ndmin=2)
        self.X_obs = np.array(self.current_x)
        self.Y_obs = [np.array([[inverse_sign*error_func(x)]])]
        # iterate though the GP method
        for i in range(self.max_iter):
            # get next point to try using acquisition function
            x_next = self.acquire(self.alpha)
            # check for problems with the beam
            if self.check != None: self.check.errorCheck()

            y_new = error_func(x_next.flatten())
            if self.opt_ctrl.kill:
                print('WARNING - BayesOpt: Killing Bayesian optimizer...')
                break
            y_new = np.array([[inverse_sign *y_new]])

            # change position of interface
            x_new = deepcopy(x_next)
            self.current_x = x_new

            # add new entry to observed data
            self.X_obs = np.concatenate((self.X_obs, x_new), axis=0)
            self.Y_obs.append(y_new)

            # update the model (may want to add noise if using testEI)
            self.model.update(x_new, y_new)

    def OptIter(self,pause=0):
        # runs the optimizer for one iteration
    
        # get next point to try using acquisition function
        x_next = self.acquire()
        if(self.acq_func[0] == 'testEI'):
            ind = x_next
            x_next = np.array(self.acq_func[2].iloc[ind,:-1],ndmin=2)
        
        # change position of interface and get resulting y-value
        self.mi.setX(x_next)
        if(self.acq_func[0] == 'testEI'):
            (x_new, y_new) = (x_next, self.acq_func[2].iloc[ind,-1])
        else:
            (x_new, y_new) = self.mi.getState()
        # add new entry to observed data
        self.X_obs = np.concatenate((self.X_obs,x_new),axis=0)
        self.Y_obs.append(y_new)
        
        # update the model (may want to add noise if using testEI)
        self.model.update(x_new, y_new)# + .5*np.random.randn())
            
            
    def ForcePoint(self,x_next):
        # force a point acquisition at our discretion and update the model
        
        # change position of interface and get resulting y-value
        self.mi.setX(x_next)
        if(self.acq_func[0] == 'testEI'):
            (x_new, y_new) = (x_next, self.acq_func[2].iloc[ind,-1])
        else:
            (x_new, y_new) = self.mi.getState()
        # add new entry to observed data
        self.X_obs = np.concatenate((self.X_obs,x_new),axis=0)
        self.Y_obs.append(y_new)
        
        # update the model (may want to add noise if using testEI)
        self.model.update(x_new, y_new)

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

        (ind_best, mu_best) = max(enumerate(mu), key=op.itemgetter(1))
        return (self.X_obs[ind_best], mu_best)

    def acquire(self, alpha=1.):
        """
        Computes the next point for the optimizer to try by maximizing
        the acquisition function. If movement per iteration is bounded,
        starts search at current position.
        """
        # look from best positions
        (x_best, y_best) = self.best_seen()
        self.x_best = x_best
        x_curr = self.current_x[-1]
        x_start = x_best
            
        ndim = x_curr.size # dimension of the feature space we're searching NEEDED FOR UCB
        try:
            nsteps = 1 + self.X_obs.shape[0] # acquisition number we're on  NEEDED FOR UCB
        except:
            nsteps = 1

        # check to see if this is bounding step sizes
        if(self.iter_bound or True):
            if(self.bounds is None): # looks like a scale factor
                self.bounds = 1.0

            bound_lengths = self.searchBoundScaleFactor * 3. * self.lengthscales # 3x hyperparam lengths
            relative_bounds = np.transpose(np.array([-bound_lengths, bound_lengths]))
            
            #iter_bounds = np.transpose(np.array([x_start - bound_lengths, x_start + bound_lengths]))
            iter_bounds = np.transpose(np.array([x_start - bound_lengths, x_start + bound_lengths]))

        else:
            iter_bounds = self.bounds

        # options for finding the peak of the acquisition function:
        optmethod = 'L-BFGS-B' # L-BFGS-B, BFGS, TNC, and SLSQP allow bounds whereas Powell and COBYLA don't
        maxiter = 1000 # max number of steps for one scipy.optimize.minimize call
        try:
            nproc = mp.cpu_count() # number of processes to launch minimizations on
        except:
            nproc = 1
        niter = 1 # max number of starting points for search
        niter_success = 1 # stop search if same minima for 10 steps
        tolerance = 1.e-4 # goal tolerance

        # perturb start to break symmetry?
        #x_start += np.random.randn(lengthscales.size)*lengthscales*1e-6

        # probability of improvement acquisition function
        if(self.acq_func[0] == 'PI'):
            aqfcn = negProbImprove
            fargs=(self.model, y_best, self.acq_func[1])

        # expected improvement acquisition function
        elif(self.acq_func[0] == 'EI'):
            aqfcn = negExpImprove
            fargs = (self.model, y_best, self.acq_func[1], alpha)

        # gaussian process upper confidence bound acquisition function
        elif(self.acq_func[0] == 'UCB'):
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
            print('WARNING - BayesOpt: Unknown acquisition function.')
            return 0

        try:

            if(self.multiprocessingQ): # multi-processing to speed search

                neval = 2*int(10.*2.**(ndim/12.))
                nkeep = 2*min(8,neval)

                # parallelgridsearch generates pseudo-random grid, then performs an ICDF transform
                # to map to multinormal distrinbution centered on x_start and with widths given by hyper params

                # add the 10 best points seen so far (largest Y_obs)
                nbest = 3 # add the best points seen so far (largest Y_obs)
                nstart = 2 # make sure some starting points are there to prevent run away searches
                yobs = np.array([y[0][0] for y in self.Y_obs])
                isearch = yobs.argsort()[-nbest:]
                for i in range(min(nstart,len(self.Y_obs))): #
                    if np.sum(isearch == i) == 0: # not found in list
                        isearch = np.append(isearch, i)
                        isearch.sort() # sort to bias searching near earlier steps

                v0s = None
                for i in isearch:
                    vs = parallelgridsearch(aqfcn,self.X_obs[i],self.searchBoundScaleFactor * 0.6*self.lengthscales,fargs,neval,nkeep)
                    if type(v0s) == type(None):
                        v0s = copy(vs)
                    else:
                        v0s = np.vstack((v0s,vs))

                v0sort = v0s[:,-1].argsort()[:nkeep] # keep the nlargest
                v0s = v0s[v0sort]
                
                x0s = v0s[:,:-1] # for later testing if the minimize results are better than the best starting point
                v0best = v0s[0]

                if basinhoppingQ:
                    # use basinhopping
                    bkwargs = dict(niter=niter,niter_success=niter_success, minimizer_kwargs={'method':optmethod,'args':fargs,'tol':tolerance,'bounds':iter_bounds,'options':{'maxiter':maxiter}}) # keyword args for basinhopping
                    res = parallelbasinhopping(aqfcn,x0s,bkwargs)

                else:
                    # use minimize
                    mkwargs = dict(bounds=iter_bounds, method=optmethod, options={'maxiter':maxiter}, tol=tolerance) # keyword args for scipy.optimize.minimize
                    res = parallelminimize(aqfcn,x0s,fargs,mkwargs,v0best,relative_bounds=relative_bounds)

            else: # single-processing
                if basinhoppingQ:
                    res = basinhopping(aqfcn, x_start,niter=niter,niter_success=niter_success, minimizer_kwargs={'method':optmethod,'args':(self.model, y_best, self.acq_func[1], alpha),'tol':tolerance,'bounds':iter_bounds,'options':{'maxiter':maxiter}})

                else:
                    res = minimize(aqfcn, x_start, args=(self.model, y_best, self.acq_func[1], alpha), method=optmethod,tol=tolerance,bounds=iter_bounds,options={'maxiter':maxiter})

                res = res.x
                
        except:
            raise
        return np.array(res,ndmin=2) # return resulting x value as a (1 x dim) vector
        

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
    return alpha * (-EI) + (1. - alpha) * (-y_mean)

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

    if nsteps==0: nsteps += 1
    (y_mean, y_var) = model.predict(np.array(x_new,ndmin=2))

    tau = 2.*np.log(nsteps**(0.5*ndim+2.)*(np.pi**2.)/3./delta)
    GPUCB = y_mean + np.sqrt(nu * tau * y_var)

    return -GPUCB

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

# Thompson sampling
