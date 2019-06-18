from __future__ import print_function, absolute_import
from mint.mint import *

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