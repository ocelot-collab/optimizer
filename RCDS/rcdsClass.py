import math
import numpy as np


# Created by X. Huang, SLAC, 10/6/2016
# Disclaimer: The RCDS algorithm or the Matlab RCDS code come with absolutely
# NO warranty. The author of the RCDS method and the Matlab RCDS code does not
# take any responsibility for any damage to equipments or personnel injury
# that may result from the use of the algorithm or the code.

class RCDS:
    def __init__(self, func, g_noise=0.1, g_cnt=0, Nvar=6, g_vrange=None, g_data=None, Imat=None):
        self.g_noise = g_noise
        self.g_cnt = g_cnt
        self.Nvar = Nvar
        self.g_vrange = g_vrange
        self.g_data = g_data
        self.Imat = Imat
        self.objfunc = func

    def powellmain(self, x0, step, Dmat0, tol=1.0E-5, maxIt=100, maxEval=1500):
        '''RCDS main self.func_objtion, implementing Powell's direction set update method
        Created by X. Huang, 10/5/2016
        Input:
                 self.func_obj is self.func_objtion handle,
                 step, tol : floating number
                 x0: NumPy vector
                        Dmat0: a matrix
                        maxIt, maxEval: Integer
        Output:
                 x1, f1,
                        nf: integer, number of evaluations
        '''
        self.Nvar = len(x0)
        f0 = self.func_obj(x0)
        nf = 1

        xm = x0
        fm = f0

        it = 0
        Dmat = Dmat0
        Npmin = 6  # number of points for fitting
        while it < maxIt:
            print('iteration {}'.format(it))
            it += 1
            step /= 1.2

            k = 1
            dl = 0
            for ii in range(self.Nvar):
                dv = Dmat[:, ii]
                # print('bracketmin', xm,fm,dv,step)
                (x1, f1, a1, a2, xflist, ndf) = self.bracketmin(xm, fm, dv, step)
                nf += ndf
                # print([it, ii, a1,a2, f1])

                print("iter %d, dir %d: begin\t%d\t%f" % (it, ii, self.g_cnt, f1))
                (x1, f1, ndf) = self.linescan(x1, f1, dv, a1, a2, Npmin, xflist)
                nf += ndf

                if (fm - f1) > dl:
                    dl = (fm - f1)
                    k = ii
                    print("iteration %d, var %d: del = %f updated\n" % (it, ii, dl))
                fm = f1
                xm = x1

            xt = 2 * xm - x0
            print('evaluating self.func_obj')
            ft = self.func_obj(xt)
            print('done')
            nf += 1

            if f0 <= ft or 2 * (f0 - 2 * fm + ft) * ((f0 - fm - dl) / (ft - f0)) ** 2 >= dl:
                print("   , dir %d not replaced: %d, %d\n" % (
                k, f0 <= ft, 2 * (f0 - 2 * fm + ft) * ((f0 - fm - dl) / (ft - f0)) ** 2 >= dl))
            else:
                ndv = (xm - x0) / np.linalg.norm(xm - x0)
                dotp = np.zeros([self.Nvar])
                print(dotp)
                for jj in range(self.Nvar):
                    dotp[jj] = abs(np.dot(ndv.transpose(), Dmat[:, jj]))

                if max(dotp) < 0.9:
                    for jj in range(k, self.Nvar - 1):
                        Dmat[:, jj] = Dmat[:, jj + 1]
                    Dmat[:, -1] = ndv

                    # move to the minimum of the new direction
                    dv = Dmat[:, -1]
                    (x1, f1, a1, a2, xflist, ndf) = self.bracketmin(xm, fm, dv, step)
                    nf += ndf
                    print("iter %d, new dir %d: begin\t%d\t%f " % (it, k, self.g_cnt, f1))
                    (x1, f1, ndf) = self.linescan(x1, f1, dv, a1, a2, Npmin, xflist)
                    print("end\t%d : %f\n" % (self.g_cnt, f1))
                    nf = nf + ndf
                    fm = f1
                    xm = x1
                else:
                    print("    , skipped new direction %d, max dot product %f\n" % (k, max(dotp)))

            print('g count is ', self.g_cnt, 'and maxEval is ', maxEval)
            # termination
            if self.g_cnt > maxEval:
                print("terminated, reaching self.func_objtion evaluation limit: %d > %d\n" % (self.g_cnt, maxEval))
                break

            if 2.0 * abs(f0 - fm) < tol * (abs(f0) + abs(fm)) and tol > 0:
                print("terminated: f0=%4.2e\t, fm=%4.2e, f0-fm=%4.2e\n" % (f0, fm, f0 - fm))
                break;

            f0 = fm;
            x0 = xm;

        return xm, fm, nf

    def bracketmin(self, x0, f0, dv, step):
        '''bracket the minimum
        Created by X. Huang, 10/5/2016
        Input:
                 self.func_obj is self.func_objtion handle,
                 f0,step : floating number
                 x0, dv: NumPy vector
        Output:
                 xm, fm
                        a1, a2: floating
                        xflist: Nx2 array
                        nf: integer, number of evaluations
        '''
        # global g_noise

        nf = 0
        if math.isnan(f0):
            f0 = self.func_obj(x0)
            nf += 1

        xflist = np.array([[0, f0]])
        fm = f0
        am = 0
        xm = x0

        step_init = step

        x1 = x0 + dv * step
        f1 = self.func_obj(x1)
        nf += 1

        xflist = np.concatenate((xflist, np.array([[step, f1]])), axis=0)
        if f1 < fm:
            fm = f1
            am = step
            xm = x1

        gold_r = 1.618
        while f1 < fm + self.g_noise * 3:
            step0 = step
            if abs(step) < 0.1:  # maximum step
                step = step * (1.0 + gold_r)
            else:
                step = step + 0.1
            x1 = x0 + dv * step
            f1 = self.func_obj(x1)
            nf += 1

            if math.isnan(f1):
                step = step0
                break
            else:
                xflist = np.concatenate((xflist, np.array([[step, f1]])), axis=0)
                if f1 < fm:
                    fm = f1
                    am = step
                    xm = x1

        a2 = step
        if f0 > fm + self.g_noise * 3:  # no need to go in the negative direction
            a1 = 0
            a1 = a1 - am
            a2 = a2 - am
            xflist[:, 0] -= am
            return xm, fm, a1, a2, xflist, nf

        # go in the negative direction
        step = -step_init
        x2 = x0 + dv * step
        f2 = self.func_obj(x2)
        nf += 1
        xflist = np.concatenate((xflist, np.array([[step, f2]])), axis=0)
        if f2 < fm:
            fm = f2
            am = step
            xm = x2

        while f2 < fm + self.g_noise * 3:
            step0 = step
            if abs(step) < 0.1:
                step = step * (1.0 + gold_r)
            else:
                step -= 0.1

            x2 = x0 + dv * step
            f2 = self.func_obj(x2)
            nf += 1
            if math.isnan(f2):
                step = step0
                break
            else:
                xflist = np.concatenate((xflist, np.array([[step, f2]])), axis=0)
            if f2 < fm:
                fm = f2
                am = step
                xm = x2

        a1 = step
        if a1 > a2:
            a1, a2 = a2, a1

        a1 -= am
        a2 -= am
        xflist[:, 0] -= am
        # sort by alpha
        # print(xflist)
        xflist = xflist[np.argsort(xflist[:, 0])]

        return xm, fm, a1, a2, xflist, nf

    def linescan(self, x0, f0, dv, alo, ahi, Np, xflist):
        '''Line optimizer for RCDS
        Created by X. Huang, 10/3/2016
        Input:
                 self.func_obj is self.func_objtion handle,
                 f0, alo, ahi: floating number
                 x0, dv: NumPy vector
                 xflist: Nx2 array
        Output:
                 x1, f1, nf
        '''
        # global g_noise
        nf = 0
        if math.isnan(f0):
            f0 = self.func_obj(x0)
            nf += 1

        if alo >= ahi:
            print('Error: bracket upper bound equal to or lower than lower bound')
            return x0, f0, nf

        V = len(x0)
        if len(x0) != len(dv):
            print('Error: x0 and dv dimension do not match.')
            return x0, f0, nf

        if math.isnan(Np) | (Np < 6):
            Np = 6
        delta = (ahi - alo) / (Np - 1.0)

        alist = np.arange(alo, ahi, (ahi - alo) / (Np - 1))
        flist = alist * float('nan')
        Nlist = np.shape(xflist)[0]
        for ii in range(Nlist):
            if xflist[ii, 1] >= alo and xflist[ii, 1] <= ahi:
                ik = math.round((xflist[ii, 1] - alo) / delta)
                # print('test', ik, ii, len(alist),len(xflist),xflist[ii,0])
                alist[ik] = xflist[ii, 0]
                flist[ik] = xflist[ii, 1]

        mask = np.ones(len(alist), dtype=bool)
        for ii in range(len(alist)):
            if math.isnan(flist[ii]):
                alpha = alist[ii]
                flist[ii] = self.func_obj(x0 + alpha * dv)
                nf += 1
            if math.isnan(flist[ii]):
                mask[ii] = False

        # filter out NaNs
        alist = alist[mask]
        flist = flist[mask]
        if len(alist) <= 0:
            return x0, f0, nf
        elif len(alist) < 5:
            imin = flist.argmin()
            xm = x0 + alist[imin] * dv
            fm = flist[imin]
            return xm, fm, nf
        else:
            # print(np.c_[alist,flist])
            (p) = np.polyfit(alist, flist, 2)
            pf = np.poly1d(p)

            # remove outlier and re-fit here, to be done later

            MP = 101
            av = np.linspace(alist[0], alist[-1], MP - 1)
            yv = pf(av)
            imin = yv.argmin()
            xm = x0 + av[imin] * dv
            fm = yv[imin]
            # print(x0, xm, fm)
            return xm, fm, nf

    def func_obj(self, x):
        '''Objective self.func_objtion for test
        Input:
                x : a column vector
        Output:
                obj : an floating number
        '''
        # global g_cnt, g_data, g_vrange
        # global g_noise
        self.Nvar = len(x)
        # print(x)
        # print(self.g_vrange[:,0])
        p = self.g_vrange[:, 0] + np.multiply((self.g_vrange[:, 1] - self.g_vrange[:, 0]), x)
        # print(p)
        if min(x) < 0 or max(x) > 1:
            obj = float('NaN')
        else:
            obj = self.objfunc(p)
        self.g_cnt += 1

        return obj

