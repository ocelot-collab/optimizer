from __future__ import print_function, absolute_import
from mint.mint import *
from scipy import optimize

class Powell(Minimizer):
    def __init__(self):
        super(Powell, self).__init__()
        self.xtol = 1e-5
        self.dev_steps = None

    def minimize(self,  error_func, x):
        res = optimize.minimize(error_func, x, method='Powell', tol=self.xtol)
        return res
