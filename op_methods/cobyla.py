from __future__ import print_function, absolute_import
from mint.mint import *
from scipy import optimize


class Cobyla(Minimizer):
    def __init__(self):
        super(Cobyla, self).__init__()
        self.xtol = 1e-5
        self.dev_steps = None
        self.norm_scales = None
        self.cons = None

    def calc_scales(self):
        """
        calculate scales for normalized simplex

        :return: np.array() - device_delta_limits * norm_coef
        """

        if self.norm_scales is None:
            self.norm_scales = [None] * np.size(self.devices)

        for idx, dev in enumerate(self.devices):
            if self.norm_scales[idx] is not None:
                continue
            delta = dev.get_delta()
            if delta == 0:
                delta = 1
            self.norm_scales[idx] = delta
        self.norm_scales = np.array(self.norm_scales)

        return self.norm_scales

    def calc_constraints(self):

        def make_lambda(indx, b):
            return lambda x: x[indx] + b

        cons = []
        for idx, dev in enumerate(self.devices):
            if dev.get_delta() == 0:
                continue
            high = (dev.high_limit - self.x_init[idx])/self.norm_scales[idx]/self.norm_coef/self.scaling_coef
            con = {'type': 'ineq', 'fun': make_lambda(idx, -high)}
            cons.append(con)
            low = (dev.low_limit - self.x_init[idx]) / self.norm_scales[idx]/self.norm_coef/self.scaling_coef
            con = {'type': 'ineq', 'fun': make_lambda(idx, -low)}
            cons.append(con)

        return cons

    def unnormalize(self, xnorm):
        # 1.0 is used because of the 'rhobeg': 1.0.

        delta_x = np.array(xnorm)*self.scaling_coef
        delta_x_scaled = delta_x/1.0 * self.norm_scales * self.norm_coef
        x = self.x_init + delta_x_scaled
        print("xnorm = ", xnorm)
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
        """
        defining attribute self.dev_steps

        :return:
        """
        self.calc_scales()
        self.cons = self.calc_constraints()

    def minimize(self, error_func, x):
        if self.cons is None:
            self.cons = ()

        res = optimize.minimize(error_func, x, tol=self.xtol, method='COBYLA', constraints=list(self.cons),
                                options={'rhobeg': 1.0, 'maxiter': self.max_iter, 'disp': False, 'catol': 0.0002})

        return res
