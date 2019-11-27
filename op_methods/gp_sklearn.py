from __future__ import print_function, absolute_import
from  mint.mint import *
from op_methods.simplex import Simplex
import sklearn
sklearn_version = sklearn.__version__
if sklearn_version >= "0.18":
    from GP import gaussian_process as gp_sklearn


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
