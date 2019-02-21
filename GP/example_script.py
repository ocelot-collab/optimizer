# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 16:19:03 2016

@author: Mitch

Script to show optimization results on toy problem.

Currently more sensitive to initial conditions than expected. Also currently
uses hyperparameters that are clearly suboptimal - results are inconsistent
based on initial sampling and whether bounds are given for acquisition.

Probably could find a better toy problem that is nonnegative, which
might imporove consistency.

"""

import numpy as np
import pandas as pd
from GPtools import *
from OnlineGP import OGP
from SPGPmodel import SPGP
from BasicInterfaces import fint
from numpy.random import rand
import BayesOptimization as BOpt

runs = 20
num_iter = 60
num_init = 10
numBV = 5

np.random.seed(1)

# use this just to compute hyperparams
fullGP = SPGP()
rand_sample = (rand(100,1) - .5) * 4
function = fint(rand_sample[0])
fullGP.fit(pd.DataFrame(rand_sample), function.f(rand_sample), 100)
hyps = fullGP.hyps


## setup for data collection
#res1 = range(runs)
#res2 = range(runs)
#x1 = range(runs)
#x2 = range(runs)
#model1 = range(runs)
#model2 = range(runs)
#
#for j in range(runs):
#    
#    # generate initial data
#    init_x = (rand(1,1) - .5) * 4
#    p_data = np.zeros(shape=(num_init,2))
#    int1 = fint(init_x)
#    int2 = fint(init_x)
#    
#    p_data[:,[0]] = (rand(num_init,1) - .5)*4
#    p_data[:,[1]] = int1.f(p_data[:,[0]])    
#    
#    
#    # initialize optimizers
#    #bnds = tuple([(-1.9,1.9)])
#    bnds = None
#    model1[j] = OGP(1,hyps, maxBV=numBV)
#    opt1 = BOpt.BayesOpt(model1[j], int1, xi=0.01, acq_func='EI', bounds=bnds, prior_data=pd.DataFrame(p_data))
#    model2[j] = OGP(1,hyps, weighted=True, maxBV=numBV)
#    opt2 = BOpt.BayesOpt(model2[j], int2, xi=0.01, acq_func='EI', bounds=bnds, prior_data=pd.DataFrame(p_data))
#    
#    # iterate, do optimization, collect data
#    res1[j] = []
#    res2[j] = []
#    x1[j] = []
#    x2[j] = []
#    for i in range(num_iter):
#        opt1.OptIter()
#        #opt2.OptIter()
#        x1[j].append(opt1.acquire())
#        #x2[j].append(opt2.acquire())
#        res1[j].append(int1.f(x1[j][-1])[0][0])
#        #res2[j].append(int1.f(x2[j][-1])[0][0])
#        
## performance plot
##errplot(res1,res2)
#errplot(res1)

# can plot individual GP models as well:
# BVplot(model1[0], function.f)


