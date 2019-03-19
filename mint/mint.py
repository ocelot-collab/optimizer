"""
Main Ocelot optimization file
Contains the setup for using the scipy.optimize package run simplex and other algorothms
Modified for use at LCLS from Ilya's version

The file was modified and were introduced new Objects and methods.
S. Tomin, 2017

"""
from __future__ import print_function, absolute_import
import time
from scipy.optimize import OptimizeResult
import scipy
import numpy as np
from mint.opt_objects import *
from scipy import optimize
from GP.bayes_optimization import *
from GP.OnlineGP import OGP
from GP.DKLmodel import DKLGP
try:
    from matrixmodel.beamconfig import Beamconfig
except:
    for i in range(5):
        print('WARNING: could not import Beamconfig from matrixmodel.beamconfig')
import pandas as pd
from threading import Thread
import sklearn
from op_methods.es import ES_min

from mint import normscales

sklearn_version = sklearn.__version__
if sklearn_version >= "0.18":
    from GP import gaussian_process as gp_sklearn


class Logger(object):
    def __init__(self, log_file):
        self.log_file = log_file
        self.f = open(self.log_file, 'a')

    def log_start(self, dev_ids, method, x_init, target_ref):
        self.f.write('\n*** optimization step ***\n')
        self.f.write(str(dev_ids) + '\n')
        self.f.write(method + '\n')
        self.f.write('x_init =' + str(x_init) + '\n')
        self.f.write('target_ref =' + str(target_ref) + '\n')

    def log(self, data):
        self.f.write(data)

    def log_fin(self, target_new):
        self.f.write('target_new=' + str(target_new) + '\n')
        self.f.close()


class Minimizer(object):
    def __init__(self):
        self.mi = None
        self.max_iter = 100
        self.maximize = False

    def minimize(self, error_func, x):
        pass


class ESMin(Minimizer):
    def __init__(self):
        super(ESMin, self).__init__()
        self.ES = ES_min()

    def minimize(self, error_func, x):
        self.ES.bounds = self.bounds
        self.ES.max_iter = self.max_iter
        self.ES.norm_coef = self.norm_coef
        self.ES.minimize(error_func, x)
        return


class Simplex(Minimizer):
    def __init__(self):
        super(Simplex, self).__init__()
        self.xtol = 1e-5
        self.dev_steps = None

    def minimize(self,  error_func, x):
        #print("start seed", np.count_nonzero(self.dev_steps))
        if self.dev_steps == None or len(self.dev_steps) != len(x):
            print("initial simplex is None")
            isim = None
        elif np.count_nonzero(self.dev_steps) != len(x):
            print("There is zero step. Initial simplex is None")
            isim = None
        else:
            #step = np.ones(len(x))*0.05
            isim = np.zeros((len(x) + 1, len(x)))
            isim[0, :] = x
            for i in range(len(x)):
                vertex = np.zeros(len(x))
                vertex[i] = self.dev_steps[i]
                isim[i + 1, :] = x + vertex
            print("ISIM = ", isim)
        #res = optimize.minimize(error_func, x, method='Nelder-Mead',  tol=self.xtol,
        #                        options = {'disp': False, 'initial_simplex': [0.05, 0.05], 'maxiter': self.max_iter})
        if scipy.__version__ < "0.18":
            res = optimize.fmin(error_func, x, maxiter=self.max_iter, maxfun=self.max_iter, xtol=self.xtol)
        else:
            res = optimize.fmin(error_func, x, maxiter=self.max_iter, maxfun=self.max_iter, xtol=self.xtol, initial_simplex=isim)

        #print("finish seed")
        return res


class Powell(Minimizer):
    def __init__(self):
        super(Powell, self).__init__()
        self.xtol = 1e-5
        self.dev_steps = None

    def minimize(self,  error_func, x):
        res = optimize.minimize(error_func, x, method='Powell', tol=self.xtol)
        return res


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
            self.model.fit(p_X, p_Y, min(self.m, num))

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
        #add in extra GP model data to save
        try:
            self.mi.data
        except:
            self.mi.data = {}
        self.mi.data["acq_fcn"]      = self.acq_func
        # OnlineGP stuff
        try:
            self.mi.data["alpha"]        = self.model.alpha
        except:
            pass
        try:
            self.mi.data["C"]            = self.model.C
        except:
            pass
        try:
            self.mi.data["BV"]           = self.model.BV
        except:
            pass
        try:
            self.mi.data["covar_params"] = self.model.covar_params
        except:
            pass
        try:
            self.mi.data["KB"]           = self.model.KB
        except:
            pass
        try:
            self.mi.data["KBinv"]        = self.model.KBinv
        except:
            pass
        try:
            self.mi.data["weighted"]     = self.model.weighted
        except:
            pass
        try:
            self.mi.data["noise_var"]    = self.model.noise_var
        except:
            pass
        # DKLmodel stuff
        try:
            self.mi.data["dim"]        = self.model.dim
        except:
            pass
        try:
            self.mi.data["hidden_layers"]        = self.model.hidden_layers
        except:
            pass
        try:
            self.mi.data["dim_z"]        = self.model.dim_z
        except:
            pass
        if type(self.model.mask) is not type(None):
            self.mi.data["mask"]        = self.model.mask
        try:
            self.mi.data["alpha"]        = self.model.alpha
        except:
            pass
        try:
            self.mi.data["noise"]        = self.model.noise
        except:
            pass
        try:
            self.mi.data["activations"]        = self.model.activations
        except:
            pass
        try:
            self.mi.data["weight_dir"]        = self.model.weight_dir
        except:
            pass
        self.mi.data["corrmat"]      = self.corrmat
        self.mi.data["covarmat"]     = self.covarmat
        self.mi.data["seedScanBool"] = self.seedScanBool
        if self.seedScanBool:
            self.mi.data["nseed"]    = self.prior_data.shape[0]
        else:
            self.mi.data["nseed"]    = 0

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
            #print 'SUCCESS self.mi.data[prior_pv_info] = ', self.mi.data["prior_pv_info"]
        except:
            #print 'FAILURE self.mi.data[prior_pv_info]'
            pass


class GaussProcessSKLearn(Minimizer):
    def __init__(self):
        super(GaussProcessSKLearn, self).__init__()
        self.seed_iter = 5
        self.seed_timeout = 0.1

        self.target = None
        self.devices = []

        self.x_obs = []
        self.y_obs = []
        #GP parameters

        self.max_iter = 50
        self.norm_coef = 0.1
        self.kill = False
        self.opt_ctrl = None

    def seed_simplex(self):
        opt_smx = Optimizer()
        opt_smx.normalization = True
        opt_smx.maximization = self.maximize
        opt_smx.norm_coef = self.norm_coef
        opt_smx.timeout = self.seed_timeout
        opt_smx.opt_ctrl = self.opt_ctrl
        minimizer = Simplex()
        minimizer.max_iter = self.seed_iter
        opt_smx.minimizer = minimizer
        # opt.debug = True
        seq = [Action(func=opt_smx.max_target_func, args=[self.target, self.devices])]
        opt_smx.eval(seq)
        print(opt_smx.opt_ctrl.dev_sets)
        self.x_obs = np.vstack(opt_smx.opt_ctrl.dev_sets)
        self.y_obs = np.array(opt_smx.opt_ctrl.penalty)
        self.y_sigma_obs = np.zeros(len(self.y_obs))

    def load_seed(self, x_sets, penalty, sigma_pen=None):

        self.x_obs = np.vstack(x_sets)
        self.y_obs = np.array(penalty)
        if sigma_pen == None:
            self.y_sigma_obs = np.zeros(len(self.y_obs))
        else:
            self.y_sigma_obs = sigma_pen

    def preprocess(self):

        self.scanner = gp_sklearn.GP()
        self.scanner.opt_ctrl = self.opt_ctrl
        devs_std = []
        devs_search_area = []
        for dev in self.devices:
            lims = dev.get_limits()
            devs_std.append((lims[-1] - lims[0])/3.)
            x_vec = np.atleast_2d(np.linspace(lims[0], lims[-1], num=50)).T
            devs_search_area.append(x_vec)

        self.scanner.x_search = np.hstack(devs_search_area)
        self.scanner.x_obs = self.x_obs
        self.scanner.y_obs = self.y_obs
        self.scanner.y_sigma_obs = self.y_sigma_obs

        self.scanner.ck_const_value = (0.5*np.mean(self.scanner.y_obs))**2 + 0.1
        #self.scanner.ck_const_value_bounds = (self.scanner.ck_const_value,self.scanner.ck_const_value)
        self.scanner.rbf_length_scale = np.array(devs_std)/2. + 0.01
        #self.scanner.rbf_length_scale_bounds = (self.scanner.rbf_length_scale, self.scanner.rbf_length_scale)
        self.scanner.max_iter = self.max_iter

    def minimize(self,  error_func, x):
        #self.target_func = error_func

        self.seed_simplex()
        if self.opt_ctrl.kill:
            return
        self.preprocess()
        x = [dev.get_value() for dev in self.devices]
        print("start GP")
        self.scanner.minimize(error_func, x)
        print("finish GP")
        return


class CustomMinimizer(Minimizer):
    def __init__(self):
        super(CustomMinimizer, self).__init__()
        self.dev_steps = [0.05]

    def minimize(self,  error_func, x):
        def custmin(fun, x0, args=(), maxfev=None, stepsize=[0.1],
                    maxiter=self.max_iter, callback=None, **options):

            print("inside ", stepsize)

            if np.size(stepsize) != np.size(x0):
                stepsize = np.ones(np.size(x0))*stepsize[0]
            print("inside ", stepsize)
            bestx = x0
            besty = fun(x0)
            print("BEST", bestx, besty)
            funcalls = 1
            niter = 0
            improved = True
            stop = False

            while improved and not stop and niter < maxiter:
                improved = False
                niter += 1
                for dim in range(np.size(x0)):
                    for s in [bestx[dim] - stepsize[dim], bestx[dim] + stepsize[dim]]:
                        print("custom", niter, dim, s)
                        testx = np.copy(bestx)
                        testx[dim] = s
                        testy = fun(testx, *args)
                        funcalls += 1
                        if testy < besty:
                            besty = testy
                            bestx = testx
                            improved = True
                    if callback is not None:
                        callback(bestx)
                    if maxfev is not None and funcalls >= maxfev:
                        stop = True
                        break

            return OptimizeResult(fun=besty, x=bestx, nit=niter,
                                  nfev=funcalls, success=(niter > 1))
        res = optimize.minimize(error_func, x, method=custmin, options=dict(stepsize=self.dev_steps))
        return res


class MachineStatus:
    def __init__(self):
        pass

    def is_ok(self):
        return True


class OptControl:
    """
    Optimization control

    :param m_status: MachineStatus (Device class), indicator of the machine state (beam on/off)
    :param timeot: 0.1, timeout between machine status (m_status) readings
    :param alarm_timeout: timeout between Machine status is again OK and optimization continuation

    """
    def __init__(self):
        self.penalty = []
        self.dev_sets = []
        self.devices = []
        self.nsteps = 0
        self.m_status = MachineStatus()
        self.pause = False
        self.kill = False
        self.is_ok = True
        self.timeout = 0.1
        self.alarm_timeout = 0.

    def wait(self):
        """
        check if the machine is OK. If it is not the infinite loop is launched with checking of the machine state

        :return:
        """

        if self.m_status.is_ok():
            return 1
        else:
            while 1:
                if self.m_status.is_ok():
                    self.is_ok = True
                    time.sleep(self.alarm_timeout)
                    return 1
                time.sleep(self.timeout)
                self.is_ok = False
                print(".",)

    def stop(self):
        self.kill = True

    def start(self):
        self.kill = False

    def back_nsteps(self, n):
        if n <= self.nsteps:
            n = -1 - n
        else:
            print("OptControl: back_nsteps n > nsteps. return last step")
            n = -1
        return self.dev_sets[-n]

    def save_step(self, pen, x):
        self.penalty.append(pen)
        self.dev_sets.append(x)
        self.nsteps = len(self.penalty)

    def best_step(self):
        #if len(self.penalty)== 0:
        #    print("No ")
        x = self.dev_sets[np.argmin(self.penalty)]
        return x

    def clean(self):
        self.penalty = []
        self.dev_sets = []
        self.nsteps = 0


class Optimizer(Thread):
    def __init__(self):
        super(Optimizer, self).__init__()
        self.debug = False
        self.minimizer = Simplex()
        self.logging = False
        # self.kill = False #intructed by tmc to terminate thread of this class
        self.log_file = "log.txt"
        self.logger = Logger(self.log_file)
        self.devices = []
        self.target = None
        self.timeout = 0
        self.opt_ctrl = OptControl()
        self.seq = []
        self.set_best_solution = True
        self.normalization = False
        self.norm_coef = 0.05
        self.maximization = True
        self.scaling_coef = 1.0

    def eval(self, seq=None, logging=False, log_file=None):
        """
        Run the sequence of tuning events
        """
        if seq is not None:
            self.seq = seq
        for s in self.seq:
            s.apply()

    def exceed_limits(self, x):
        for i in range(len(x)):
            if self.devices[i].check_limits(x[i]):
                return True
        return False

    def get_values(self):
        for i in range(len(self.devices)):
            print('reading ', self.devices[i].id)
            self.devices[i].get_value(save=True)

    def set_values(self, x):
        for i in range(len(self.devices)):
            print('setting', self.devices[i].id, '->', x[i])
            self.devices[i].set_value(x[i])

    def set_triggers(self):
        for i in range(len(self.devices)):
            print('triggering ', self.devices[i].id)
            self.devices[i].trigger()

    def do_wait(self):
        for i in range(len(self.devices)):
            print('waiting ', self.devices[i].id)
            self.devices[i].wait()

    def calc_scales(self):
        """
        calculate scales for normalized simplex

        :return: np.array() - device_delta_limits * norm_coef
        """
        self.norm_scales = normscales.normscales(self.target.mi, self.devices)
        if self.norm_scales is None:
            self.norm_scales = [None] * np.size(self.devices)

        for idx, dev in enumerate(self.devices):
            if self.norm_scales[idx] is not None:
                continue
            delta = dev.get_delta()
            self.norm_scales[idx] = delta*self.norm_coef

        self.norm_scales = np.array(self.norm_scales)
        
        # Randomize the initial steps of simplex - Talk to Joe if it fails
        if isinstance(self.minimizer, Simplex):
            self.norm_scales *= np.sign(np.random.randn(self.norm_scales.size))
        return self.norm_scales

    def error_func(self, x):
        # 0.00025 is used for Simplex because of the fmin steps.
        delta_x = x*self.scaling_coef

        if self.normalization:
            delta_x_scaled = delta_x/0.00025*self.norm_scales
            x = self.x_init + delta_x_scaled
            print("X Init: ", self.x_init)
            print("X: ", x)

        
        if self.opt_ctrl.kill:
            #self.minimizer.kill = self.opt_ctrl.kill
            print('Killed from external process')
            # NEW CODE - to kill if run from outside thread
            return self.target.pen_max

        self.opt_ctrl.wait()

        # check limits
        if self.exceed_limits(x):
            return self.target.pen_max
        # set values
        self.set_values(x)
        self.set_triggers()
        self.do_wait()
        self.get_values()

        print('sleeping ' + str(self.timeout))
        time.sleep(self.timeout)

        coef = -1
        if self.maximization:
            coef = 1

        pen = coef*self.target.get_penalty()
        print('penalty:', pen)
        if self.debug:
            print('penalty:', pen)

        self.opt_ctrl.save_step(pen, x)
        return pen

    def max_target_func(self, target, devices, params = {}):
        """
        Direct target function optimization with simplex/GP, using Devices as a multiknob
        """
        [dev.clean() for dev in devices]
        target.clean()
        self.target = target
        #print(self.target)
        self.devices = devices
        # testing
        self.minimizer.devices = devices
        self.minimizer.maximize = self.maximization
        self.minimizer.target = target
        self.minimizer.opt_ctrl = self.opt_ctrl
        self.target.devices = self.devices
        dev_ids = [dev.eid for dev in self.devices]
        if self.debug: print('starting multiknob optimization, devices = ', dev_ids)

        target_ref = self.target.get_penalty()

        x = [dev.get_value(save=True) for dev in self.devices]
        x_init = x

        if self.logging:
            self.logger.log_start(dev_ids, method=self.minimizer.__class__.__name__, x_init=x_init, target_ref=target_ref)

        self.x_init = x_init
        if self.normalization:
            x = np.zeros_like(x)
            self.calc_scales()

        res = self.minimizer.minimize(self.error_func, x)
        print("result", res)

        # set best solution
        if self.set_best_solution:
            print("SET the best solution", x)
            x = self.opt_ctrl.best_step()
            if self.exceed_limits(x):
                return self.target.pen_max
            self.set_values(x)

        target_new = self.target.get_penalty()

        print ('step ended changing sase from/to', target_ref, target_new)

        if self.logging:
            self.logger.log_fin(target_new=target_new)


    def run(self):
        self.opt_ctrl.start()
        self.eval(self.seq)
        print("FINISHED")
        #self.opt_ctrl.stop()
        return 0


class Action:
    def __init__(self, func, args = None, id = None):
        self.func = func
        self.args = args

        self.id = id

    def apply(self):
        print ('applying...', self.id)
        self.func(*self.args)
    #def to_JSON(self):
        #print "hoo"
    #def __repr__(self):
        #return json.dumps(self.__dict__)




def test_simplex():
    """
    test simplex method
    :return:
    """
    d1 = TestDevice(eid="d1")
    d2 = TestDevice(eid="d2")
    d3 = TestDevice(eid="d3")

    def get_limits():
        return [-100, 100]

    d1.get_limits = get_limits
    d2.get_limits = get_limits
    d3.get_limits = get_limits

    devices = [d1, d2, d3]
    target = Target_test()

    # init Optimizer
    opt = Optimizer()
    opt.timeout = 0

    # init Minimizer
    minimizer = Simplex()
    minimizer.max_iter = 300

    opt.minimizer = minimizer

    seq = [Action(func=opt.max_target_func, args=[target, devices])]
    opt.eval(seq)


def test_gauss_process():
    """
    test simplex method
    :return:
    """
    d1 = TestDevice(eid="d1")
    d2 = TestDevice(eid="d2")
    d3 = TestDevice(eid="d3")

    def get_limits():
        return [-100, 100]

    d1.get_limits = get_limits
    d2.get_limits = get_limits
    d3.get_limits = get_limits

    devices = [d1, d2, d3]
    target = TestTarget()

    # init Optimizer
    opt = Optimizer()
    opt.timeout = 0

    # init Minimizer
    minimizer = GaussProcess()
    minimizer.seed_iter = 3
    minimizer.max_iter = 300

    opt.minimizer = minimizer

    seq = [Action(func=opt.max_target_func, args=[ target, devices])]
    opt.eval(seq)


#from itertools import chain
#import scipy
#from ocelot.optimizer.GP.OnlineGP import OGP
#from ocelot.optimizer.GP.bayes_optimization import BayesOpt, HyperParams


def test_GP():
    """
    test GP method
    :return:
    """

    def get_limits():
        return [-100, 100]
    d1 = TestDevice(eid="d1")
    d1.get_limits = get_limits
    d2 = TestDevice(eid="d2")
    d2.get_limits = get_limits
    d3 = TestDevice(eid="d3")
    d3.get_limits = get_limits

    devices = [d1, d2]
    target = TestTarget()

    opt = Optimizer()
    opt.timeout = 0

    opt_smx = Optimizer()
    opt_smx.timeout = 0
    minimizer = Simplex()
    minimizer.max_iter = 3
    opt_smx.minimizer = minimizer
    #opt.debug = True

    seq = [Action(func=opt_smx.max_target_func, args=[target, devices])]
    opt_smx.eval(seq)
    s_data = np.append(np.vstack(opt_smx.opt_ctrl.dev_sets), np.transpose(-np.array([opt_smx.opt_ctrl.penalty])), axis=1)
    print(s_data)

    # -------------- GP config setup -------------- #
    #GP parameters
    numBV = 30
    xi = 0.01
    #no input bounds on GP selection for now

    pvs = [dev.eid for dev in devices]
    hyp_params = HyperParams(pvs=pvs, filename="../parameters/hyperparameters.npy")
    ave = np.mean(-np.array(opt_smx.opt_ctrl.penalty))
    std = np.std(-np.array(opt_smx.opt_ctrl.penalty))
    noise = hyp_params.calcNoiseHP(ave, std=0.)
    coeff = hyp_params.calcAmpCoeffHP(ave, std=0.)
    len_sc_hyps = []
    for dev in devices:
        ave = 10
        std = 3
        len_sc_hyps.append(hyp_params.calcLengthScaleHP(ave, std))
    print("y_data", opt_smx.opt_ctrl.penalty)
    print("pd.DataFrame(s_data)", pd.DataFrame(s_data))
    print("len_sc_hyps", len_sc_hyps )

    bnds = None
    #hyps = hyp_params.loadHyperParams(energy=3, detector_stat_params=target.get_stat_params())
    hyps1 = (np.array([len_sc_hyps]), coeff, noise) #(np.array([hyps]), coeff, noise)
    print("hyps1", hyps1)
    #exit(0)
    #init model
    dim = len(pvs)

    model = OGP(dim, hyps1, maxBV=numBV, weighted=False)

    minimizer = BayesOpt(model, target_func=target, xi=0.01, acq_func='EI', bounds=bnds, prior_data=pd.DataFrame(s_data))
    minimizer.devices = devices
    minimizer.max_iter = 300
    opt.minimizer = minimizer

    seq = [Action(func=opt.max_target_func, args=[ target, devices])]
    opt.eval(seq)


if __name__ == "__main__":
    test_simplex()
    #test_gauss_process()
    #test_GP()

