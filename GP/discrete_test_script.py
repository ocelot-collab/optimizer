# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 21:19:02 2016

@author: Mitch

Imports LCLS data and does trial optimization.
"""

import numpy as np
import pandas as pd
from GPtools import *
import OnlineGP
from BasicInterfaces import TestInterface, GPint
from numpy.random import randn
import BayesOpt_oldcopy as BOpt

np.random.seed(1)

# load data
data_file = './d1.csv'
data = pd.read_csv(data_file)
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

# bound the acquisition: typically leads to better performance
mins = X.min(axis=0)
maxs = X.max(axis=0)
#bnds = tuple([(mins[i],maxs[i]) for i in range(len(mins))])
bnds = None

# guess at hyperparameters for now
data_hyps = (-2 * np.array(np.log((X.max() - X.min()) / 4.0),ndmin=2), np.log(y.var() + .1), np.log(y.var() / 4 + .01))

# set up run parameters
runs = 1
num_iter = 50
num_train = 0
numBV = 5
noise = 0.0
xi = .8

# initialize for data collection
model1 = range(runs)
model2 = range(runs)
opt1 = range(runs)
opt2 = range(runs)
res1 = range(runs)
res2 = range(runs)


for i in range(runs):
    model1[i] = OnlineGP.OGP(X.shape[1],data_hyps,weighted=False, maxBV=numBV, prmean=0)
    model2[i] = OnlineGP.OGP(X.shape[1],data_hyps, weighted=True, maxBV=numBV, prmean=0)

    # initial training
    train_data = data.copy()
    train_data.apply(np.random.shuffle,axis=0)
    train_data = train_data.iloc[:num_train,:]
    model1[i].fit(train_data.iloc[:,:-1],train_data.iloc[:,-1])
    model2[i].fit(train_data.iloc[:,:-1],train_data.iloc[:,-1])

    # mock machine interfaces
    intfc1 = TestInterface(vify(X,0))
    intfc2 = TestInterface(vify(X,0))

    # initialize optimizers
    opt1[i] = BOpt.BayesOpt(model1[i], intfc1, acq_func='testEI', xi=xi, bounds=bnds, alt_param=data)
    opt2[i] = BOpt.BayesOpt(model2[i], intfc2, acq_func='testEI', xi=xi, bounds=bnds, alt_param=data)

    # do optimization
    for j in range(num_iter):
        opt1[i].OptIter()
        opt2[i].OptIter()
    
    # collect data
    res1[i] = np.reshape(opt1[i].Y_obs[1:],(num_iter))
    res2[i] = np.reshape(opt2[i].Y_obs[1:],(num_iter))

# plot results
errplot(res1,res2)
