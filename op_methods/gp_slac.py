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


class GaussProcess(Minimizer):
    def __init__(self):
        super(GaussProcess,self).__init__()
        self.seed_timeout = 1
        self.target = None
        self.devices = []
        self.energy = 3
        self.seed_iter = 0
        self.numBV = 30
        self.xi = 0.01
        self.bounds = None
        #self.acq_func = 'PI'
        self.acq_func = 'EI'
        #self.acq_func = 'UCB'
        self.alt_param = -1
        self.m = 200
        self.iter_bound = False
        self.hyper_file = None
        self.max_iter = 50
        self.norm_coef = 0.1
        self.multiplier = 1
        self.simQ = False
        self.seedScanBool = True
        self.prior_data = None

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
        self.prior_data = pd.DataFrame(seed_data)
        self.seed_y_data = opt_smx.opt_ctrl.penalty


    def preprocess(self):
        self.energy = self.mi.get_energy()
        hyp_params = HyperParams(pvs=self.devices, filename=self.hyper_file, mi=self.mi)
        dev_ids = [dev.eid for dev in self.devices]
        print('devids = ', dev_ids)
        dev_vals = [dev.get_value() for dev in self.devices]
        print("mintGP: dev_vals = ",dev_vals)
        hyps1 = hyp_params.loadHyperParams(self.hyper_file, self.energy, self.target, dev_ids, dev_vals, self.multiplier)
        dim = len(self.devices)
        print("mintGP: initializing with hyperparams ", hyps1)

        # load correlations
        if self.mi.name == 'MultinormalInterface':
            corrmat = self.mi.corrmat
            covarmat = self.mi.covarmat
        else:
            #try:
                ##hardcoded for now
                #rho = 0.
                #corrmat = np.array([  [1., rho], [rho, 1.]  ])

            #except:
                #print('WARNING: There was an error importing a correlation matrix from the matrix model. Using an identity matrix instead.')
                #corrmat = np.eye(len(dev_ids))

            corrmat = np.eye(len(dev_ids))

            # build covariance matrix from correlation matrix and length scales
            diaglens = np.diagflat(np.sqrt(0.5/np.exp(hyps1[0]))) # length scales (or principal widths)
            print('diaglens = ', diaglens)
            covarmat = np.dot(diaglens, np.dot(corrmat,diaglens))

        print('corrmat = ', corrmat)
        print('covarmat = ', covarmat)
        self.corrmat = corrmat
        self.covarmat = covarmat

        # create model
        #self.model = OGP(dim, hyps1, maxBV=self.numBV, weighted=False)
        amp_param = np.exp(hyps1[1]); print('amp_param = ', amp_param)
        noise_variance = np.exp(hyps1[2]); print('noise_variance = ', noise_variance)
        self.model = DKLGP(dim, dim_z=dim, alpha=amp_param, noise=noise_variance)
        self.model.linear_from_correlation(covarmat)

        # initialize model on prior data if available
        if(self.prior_data is not None):
            print("mintGP: Seeding GP with self.prior_data = ",self.prior_data)
            p_X = self.prior_data.iloc[:, :-1]
            p_Y = self.prior_data.iloc[:, -1]
            num = len(self.prior_data.index)
            #self.model.fit(p_X, p_Y, min(self.m, num))
            self.model.fit(p_X, p_Y)


        print("mintGP: self.prior_data = ", self.prior_data)
        print("mintGP: self.bounds = ", self.bounds)
        print("mintGP: self.iter_bound = ", self.iter_bound)

        # create Bayesian optimizer
        #self.scanner = BayesOpt(model=self.model, target_func=self.target, acq_func=self.acq_func, xi=self.xi, alt_param=self.alt_param, m=self.m, bounds=self.bounds, iter_bound=self.iter_bound, prior_data=self.prior_data, start_dev_vals=dev_vals)
        self.scanner = BayesOpt(model=self.model, target_func=self.target, acq_func=self.acq_func, xi=self.xi, alt_param=self.alt_param, m=self.m, bounds=self.bounds, iter_bound=self.iter_bound, prior_data=self.prior_data, start_dev_vals=dev_vals, dev_ids=dev_ids, energy=self.energy, hyper_file=self.hyper_file,corrmat=corrmat,covarmat=covarmat)
        self.scanner.max_iter = self.max_iter
        self.scanner.opt_ctrl = self.opt_ctrl

    def minimize(self,  error_func, x):
        self.energy = self.mi.get_energy()
        print('Energy is ', self.energy, ' GeV')
        if self.seedScanBool: self.seed_simplex()
        self.preprocess()
        x = [dev.get_value() for dev in self.devices]
        print("start GP")
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
        # DKLmodel stuff
        try:
            self.mi.data["dim"] = self.model.dim
        except:
            pass
        try:
            self.mi.data["hidden_layers"] = self.model.hidden_layers
        except:
            pass
        try:
            self.mi.data["dim_z"] = self.model.dim_z
        except:
            pass
        if type(self.model.mask) is not type(None):
            self.mi.data["mask"]        = self.model.mask
        try:
            self.mi.data["alpha"] = self.model.alpha
        except:
            pass
        try:
            self.mi.data["noise"] = self.model.noise
        except:
            pass
        try:
            self.mi.data["activations"] = self.model.activations
        except:
            pass
        try:
            self.mi.data["weight_dir"] = self.model.weight_dir
        except:
            pass
        self.mi.data["corrmat"] = self.corrmat
        self.mi.data["covarmat"] = self.covarmat
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
            # print 'SUCCESS self.mi.data[prior_pv_info] = ', self.mi.data["prior_pv_info"]
        except:
            # print 'FAILURE self.mi.data[prior_pv_info]'
            pass
