from __future__ import print_function, absolute_import
import GPy
from mint.mint import *
from op_methods.simplex import *
from scipy.optimize import *

class GPgpy(Minimizer):
    def __init__(self):
        super(GPgpy, self).__init__()
        self.xtol = 1e-5
        self.dev_steps = None

    def seed_simplex(self):
        opt_smx = Optimizer()
        opt_smx.maximization = self.maximize
        opt_smx.norm_coef = self.norm_coef
        opt_smx.timeout = self.seed_timeout
        opt_smx.opt_ctrl = self.opt_ctrl
        minimizer = Simplex()
        minimizer.max_iter = 20
        minimizer.dev_steps = self.dev_steps
        #print("MAX iter", self.seed_iter)
        opt_smx.minimizer = minimizer
        # opt.debug = True
        seq = [Action(func=opt_smx.max_target_func, args=[self.target, self.devices])]
        opt_smx.eval(seq)
        #print(opt_smx.opt_ctrl.dev_sets)
        self.x_obs = np.vstack(opt_smx.opt_ctrl.dev_sets)
        self.y_obs = np.array(opt_smx.opt_ctrl.penalty)
        self.y_sigma_obs = np.mean(self.target.std_dev)

    def gp_unnormalize(self, xnorm):
        ll = np.array([dev.get_limits()[0] for dev in self.devices])
        hl = np.array([dev.get_limits()[1] for dev in self.devices])
        c = (hl + ll)/2
        d = hl - ll
        x = xnorm * d/2 +c
        return x

    def gp_normalize(self, x):
        ll = np.array([dev.get_limits()[0] for dev in self.devices])
        hl = np.array([dev.get_limits()[1] for dev in self.devices])
        c = (hl + ll)/2
        d = hl - ll
        xnorm = 2*(x - c)/d
        return xnorm


    def init_gp(self, X, Y):
        ndim = np.shape(X)[1]
        self.bounds = Bounds(np.ones(ndim)*-0.99, np.ones(ndim)*0.99)
        self.bounds = [(-0.99, 0.99) for dev in self.devices]
        self.kernel = GPy.kern.RBF(input_dim=ndim, variance=np.mean(self.target.std_dev)**2, lengthscale=0.5)
        self.model = GPy.models.GPRegression(X, Y, self.kernel)

        # optimize and plot
        self.model.optimize(messages=True, max_f_eval=1000)
        self.model.Gaussian_noise = np.mean(self.target.std_dev)


    def gp_predict(self, x, model):
        ndim = np.shape(self.Xnorm)[1]
        x = np.reshape(x, (-1, ndim))
        # print(np.shape(x))
        f, v = model.predict(x)
        # print(f, v)
        return (f)

    def one_step(self, error_func, x):
        x = [dev.get_value() for dev in self.devices]
        print("start GP")

        #res = minimize(self.gp_predict, np.array(x), args=(self.model,), bounds=self.bounds, method='L-BFGS-B')
        #res = differential_evolution(self.gp_predict,  args=(self.model,), bounds=self.bounds)
        xnew = fmin(self.gp_predict, np.array(x), args=(self.model,))
        #xnew = res.x
        xnew_unnorm = self.gp_unnormalize(xnew)
        ynew = error_func(xnew_unnorm)

        self.Xnorm = np.append(self.Xnorm, np.array([xnew, ]), axis=0)
        #X = np.array([d.values for d in self.devices])
        #Y = np.array(self.target.penalties).reshape(-1, 1)
        self.Y = np.append(self.Y , np.array([ynew,]).reshape(-1, 1), axis=0)
        #print("NEW = ", self.Y, self.Xnorm )
        self.model.set_XY(X=self.Xnorm, Y=self.Y )
        self.model.optimize(messages=True, max_f_eval=1000)
        self.model.Gaussian_noise = np.mean(self.target.std_dev)

    def minimize(self,  error_func, x):
        #self.target_func = error_func

        self.seed_simplex()

        if self.opt_ctrl.kill:
            return
        self.Y = np.array(self.target.penalties[1:]).reshape(-1, 1)
        X = np.array([d.values for d in self.devices]).T
        self.Xnorm = self.gp_normalize(X)

        self.init_gp(self.Xnorm, self.Y )

        for i in range(self.max_iter):
            if self.opt_ctrl.kill:
                return
            self.one_step(error_func, x)

        print("finish GP")
        return
