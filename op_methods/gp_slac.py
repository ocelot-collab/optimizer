from __future__ import print_function, absolute_import
from mint.mint import *
from GP.bayes_optimization import *
from GP.OnlineGP import OGP
from GP.DKLmodel import DKLGP
from op_methods.simplex import *

try:
    from matrixmodel.beamconfig import Beamconfig
except:
    for i in range(5):
        print('WARNING: could not import Beamconfig from matrixmodel.beamconfig')
import pandas as pd

from mint import normscales

class GaussProcess(Minimizer):
    def __init__(self, correlationsQ = False, searchBoundScaleFactor = None, bounds= None):
        super(GaussProcess,self).__init__()
        self.seed_timeout = 1
        self.target = None
        self.devices = []
        self.energy = 4
        self.seed_iter = 0
        self.numBV = 30
        self.xi = 0.01
        self.bounds = bounds
        self.acq_func = ['PI','EI','UCB'][-1]
        self.alt_param = -1
        self.m = 200
        self.iter_bound = False
        self.max_iter = 50
        self.norm_coef = 0.1
        self.multiplier = 1
        self.simQ = False
        self.seedScanBool = True
        self.prior_data = None
        self.correlationsQ = correlationsQ
        self.searchBoundScaleFactor = searchBoundScaleFactor

    def seed_simplex(self):
        opt_smx = Optimizer()
        opt_smx.normalization = True
        opt_smx.norm_coef = self.norm_coef
        opt_smx.timeout = self.seed_timeout
        minimizer = Simplex()
        minimizer.max_iter = self.seed_iter
        opt_smx.minimizer = minimizer
        # opt.debug = True
        seq = [Action(func=opt_smx.max_target_func, args=[self.target, self.devices])]
        opt_smx.eval(seq)

        seed_data = np.append(np.vstack(opt_smx.opt_ctrl.dev_sets), np.transpose(-np.array([opt_smx.opt_ctrl.penalty])), axis=1)
        import pandas as pd
        self.prior_data = pd.DataFrame(seed_data)
        self.seed_y_data = opt_smx.opt_ctrl.penalty

    def gp_offset_prior_mean(self,x, params): # params should be a float: offset
        return np.array([params for myx in x], ndmin=2)

    def preprocess(self):
        self.target.mi.target = self.target
        
    # assemble hyper parameters
        self.length_scales, self.amp_variance, self.single_noise_variance, self.mean_noise_variance, self.precision_matrix, self.offset = normscales.normscales(self.target.mi, self.devices, correlationsQ=self.correlationsQ)

        # build precision_matrix if not returned
        print('Precision before',  self.precision_matrix ) 
        if self.precision_matrix is None:
            self.covarmat = np.diag(self.length_scales)**2
            print('Covariance',  self.covarmat ) 
            self.precision_matrix = np.linalg.inv(self.covarmat)
        print('Precision',  self.precision_matrix ) 
        print('Length Scales',  self.length_scales ) 
        # create OnlineGP model
        dim = len(self.devices)
        hyperparams = (self.precision_matrix, np.log(self.amp_variance), np.log(self.mean_noise_variance))
        self.model = OGP(dim, hyperparams, maxBV=self.numBV, covar=['RBF_ARD','MATERN32_ARD','MATERN52_ARD'][0], weighted=False)

        # initialize model on prior data if available
        if(self.prior_data is not None):
            p_X = self.prior_data.iloc[:, :-1]
            p_Y = self.prior_data.iloc[:, -1]
            num = p_X.shape[0]
            self.model.fit(p_X, p_Y, min(self.m, num))

        # create Bayesian optimizer
        dev_ids = [dev.eid for dev in self.devices]
        dev_vals = [dev.get_value() for dev in self.devices]
        self.scanner = BayesOpt(model=self.model, target_func=self.target, acq_func=self.acq_func, xi=self.xi, alt_param=self.alt_param, m=self.m, bounds=self.bounds, iter_bound=self.iter_bound, prior_data=self.prior_data, start_dev_vals=dev_vals, dev_ids=dev_ids, searchBoundScaleFactor=self.searchBoundScaleFactor)
        self.scanner.max_iter = self.max_iter
        self.scanner.opt_ctrl = self.opt_ctrl

        # overwrite the default prior mean
        if self.offset is not None:
            self.model.prmean = self.gp_offset_prior_mean
            self.model.prmeanp = self.offset
            print('***Warnning: overwrite the default prior mean (None) *** Using prior mean of ', self.offset)

    def minimize(self,  error_func, x):
        self.energy = self.mi.get_energy()
        if self.seedScanBool: self.seed_simplex()
        self.preprocess()
        # x = [dev.get_value() for dev in self.devices] # is this needed?
        self.scanner.minimize(error_func, x)
        self.saveModel()
        return

    def saveModel(self):
        """
        Add GP model parameters to the save file.
        """
        # add in extra GP model data to save
        try:
            self.mi.data
        except:
            self.mi.data = {}
        self.mi.data["acq_fcn"] = self.acq_func
        # OnlineGP stuff
        try:
            self.mi.data["alpha"] = self.model.alpha
        except:
            pass
        try:
            self.mi.data["C"] = self.model.C
        except:
            pass
        try:
            self.mi.data["BV"] = self.model.BV
        except:
            pass
        try:
            self.mi.data["covar_params"] = self.model.covar_params
        except:
            pass
        try:
            self.mi.data["KB"] = self.model.KB
        except:
            pass
        try:
            self.mi.data["KBinv"] = self.model.KBinv
        except:
            pass
        try:
            self.mi.data["weighted"] = self.model.weighted
        except:
            pass
        try:
            self.mi.data["noise_var"] = self.model.noise_var
        except:
            pass
        try:
            self.mi.data["corrmat"] = self.corrmat
        except:
            pass
        try:
            self.mi.data["covarmat"] = self.covarmat
        except:
            pass
        try:
            self.mi.data["length_scales"] = self.length_scales
        except:
            pass
        try:
            self.mi.data["amp_variance"] = self.amp_variance
        except:
            pass
        try:
            self.mi.data["single_noise_variance"] = self.single_noise_variance
        except:
            pass
        try:
            self.mi.data["mean_noise_variance"] = self.mean_noise_variance
        except:
            pass
        try:
            self.mi.data["precision_matrix"] = self.precision_matrix
        except:
            pass
        self.mi.data["seedScanBool"] = self.seedScanBool
        if self.seedScanBool:
            self.mi.data["nseed"] = self.prior_data.shape[0]
        else:
            self.mi.data["nseed"] = 0
        if type(self.model.prmeanp) is type(None):
            self.mi.data["prmean_params_amp"] = "None"
            self.mi.data["prmean_params_centroid"] = "None"
            self.mi.data["prmean_params_invcovarmat"] = "None"
        else:
            self.mi.data["prmean_params_amp"] = self.model.prmeanp[0]
            self.mi.data["prmean_params_centroid"] = self.model.prmeanp[1]
            self.mi.data["prmean_params_invcovarmat"] = self.model.prmeanp[2]
        if type(self.model.prvarp) is type(None):
            self.mi.data["prvar_params"] = "None"
        else:
            self.mi.data["prvar_params"] = self.model.prvarp
        try:
            self.mi.data["prmean_name"] = self.model.prmean_name
        except:
            pass
        try:
            self.mi.data["prior_pv_info"] = self.model.prior_pv_info
        except:
            pass
