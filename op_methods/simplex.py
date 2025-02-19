from __future__ import print_function, absolute_import
from mint.mint import *
from scipy import optimize
from mint import normscales

class Simplex(Minimizer):
    def __init__(self):
        super(Simplex, self).__init__()
        self.xtol = 1e-5
        self.dev_steps = None
    
    def preprocess(self):
        """
        defining attribute self.dev_steps

        :return:
        """
        if self.dev_steps is not None:
            return 
        self.dev_steps = []
        for dev in self.devices:
            if "istep" not in dev.__dict__:
                self.dev_steps = None
                return
            elif dev.istep is None or dev.istep == 0:
                self.dev_steps = None
                return
            else:
                self.dev_steps.append(dev.istep)
    
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


class SimplexNorm(Simplex):
    def __init__(self):
        super(SimplexNorm, self).__init__()
        self.xtol = 1e-5
        self.dev_steps = None

    def calc_scales(self):
        """
        calculate scales for normalized simplex

        :return: np.array() - device_delta_limits * norm_coef
        """
        # TODO: normscales.normscales() after last upgrade was broken. Fix or delete
        #self.norm_scales = normscales.normscales(self.target.mi, self.devices)

        self.norm_scales = None

        if self.norm_scales is None:
            self.norm_scales = [None] * np.size(self.devices)

        for idx, dev in enumerate(self.devices):
            if self.norm_scales[idx] is not None:
                continue
            delta = dev.get_delta()
            self.norm_scales[idx] = delta
        self.norm_scales = np.array(self.norm_scales)

        # Randomize the initial steps of simplex - Talk to Joe if it fails
        #if isinstance(self.minimizer, Simplex):
        self.norm_scales *= np.sign(np.random.randn(self.norm_scales.size))
        return self.norm_scales

    def unnormalize(self, xnorm):
        # 0.00025 is used for Simplex because of the fmin steps.

        delta_x = np.array(xnorm)*self.scaling_coef
        delta_x_scaled = delta_x/0.00025*self.norm_scales * self.norm_coef
        x = self.x_init + delta_x_scaled
        print("norm_scales = ", self.norm_scales )
        print("norm_coef = ", self.norm_coef)
        print("scaling_coef = ", self.scaling_coef)
        print("delta_x = ", delta_x)
        print("X Init: ", self.x_init)
        print("X: ", x)
        return x

    def normalize(self, x):
        xnorm = np.zeros_like(x)
        return xnorm

    def preprocess(self):
        self.calc_scales()


