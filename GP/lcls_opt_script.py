# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 21:19:02 2016

@author: Mitch

Imports LCLS data and does trial optimization with unweighted and weighted
online GPs. 
"""

import numpy as np
import pandas as pd
from GPtools import *
import OnlineGP
import SPGPmodel
from BasicInterfaces import TestInterface, GPint
from numpy.random import randn
import BayesOptimization as BOpt

def run():

    np.random.seed(1)

    # load data
    data = pd.read_csv('../data.csv')
    dt = data[(data.iloc[:,1] > 0.2) & (data.iloc[:,1] < 6.0)]
    ctrl = [x for x in data.columns if x[-5:] == 'BCTRL']
    X = dt[ctrl]
    y = dt.iloc[:,1]

    # clear unneeded stuff from memory
    del data

    # set up data from a given event
    event_energy = 11.45
    Xsm = X.loc[X.iloc[:,0]==event_energy,:]
    Xsm = Xsm.iloc[50:2050,:]
    Ysm = np.array(dt.loc[X.iloc[:,0]==event_energy,dt.columns[1]])
    Ysm = Ysm[50:2050]
    XYsm = Xsm.copy()
    XYsm['y'] = Ysm
    mins = Xsm.min(axis=0)
    maxs = Xsm.max(axis=0)

    # bound the acquisition: leads to better performance and lessens
    #   the improvements from weighting
    #bnds = tuple([(mins[i],maxs[i]) for i in range(len(mins))])
    bnds = None

    # build a sparse GP and optimize its hyperparameters to use for online GP
    hprior = SPGPmodel.SPGP()
    hprior.fit(Xsm, Ysm, 300)
    data_hyps = hprior.hyps

    # train truth model, the high-res GP that stands in for a real-world machine
    prior = OnlineGP.OGP(17, data_hyps, weighted=False, maxBV=600, prmean=1)
    prior.fit(Xsm,Ysm)

    # set up run parameters
    runs = 40
    num_iter = 60
    numBV = 30
    noise = 0.0
    num_train = 100

    # initialize for data collection
    model1 = range(runs)
    model2 = range(runs)
    opt1 = range(runs)
    opt2 = range(runs)
    res1 = range(runs)
    res2 = range(runs)
    preds1 = range(runs)
    preds2 = range(runs)


    for i in range(runs):
        model1[i] = OnlineGP.OGP(17,data_hyps,weighted=False, maxBV=numBV, prmean=1)# prmean=prior_func, prmeanp=(mod,poly))
        model2[i] = OnlineGP.OGP(17,data_hyps, weighted=True, maxBV=numBV, prmean=1)

        # mock machine interfaces using the big GP to supply y-values
        intfc1 = GPint(vify(Xsm,0),prior)
        intfc2 = GPint(vify(Xsm,0),prior)

        # need initial training or complex prior mean function to guide optimization
        train = XYsm.iloc[-1000:].sample(n=num_train)
        train.iloc[:,-1] += noise*randn(num_train)
        model1[i].fit(train.iloc[:,:-1],np.array(train.iloc[:,-1]))
        model2[i].fit(train.iloc[:,:-1],np.array(train.iloc[:,-1]))

        # initialize optimizers
        opt1[i] = BOpt.BayesOpt(model1[i], intfc1, acq_func='EI', xi=0, bounds=bnds)#, alt_param=XYsm)
        opt2[i] = BOpt.BayesOpt(model2[i], intfc2, acq_func='EI', xi=0, bounds=bnds)#, alt_param=XYsm)

    # do optimization
    for j in range(num_iter):
        opt1[i].OptIter()
        opt2[i].OptIter()

    # collect data
    res1[i] = np.reshape(opt1[i].Y_obs[1:],(num_iter))
    res2[i] = np.reshape(opt2[i].Y_obs[1:],(num_iter))
    preds1[i] = opt1[i].model.predict(np.array(Xsm))[0]
    preds2[i] = opt2[i].model.predict(np.array(Xsm))[0]

    # plot results

if __name__ == "__main__":

    errplot(res1,res2)
