from RCDS.rcdsClass import *
import numpy as np

def func_obj(p):
    '''Objective function for test
    Input:
        p : a column vector
    Output:
        obj : an floating number
    '''

    obj = 0
    for ii in range(len(p) - 1):
        obj -= 10 * math.exp(-0.2 * math.sqrt(p[ii] ** 2 + p[ii + 1] ** 2))

    return obj


#	obj += np.random.randn()*g_noise

def fuc_test0(x):
    return np.linalg.norm(x)


def test0():
    Nvar = 6
    g_vrange = np.matrix(np.ones((Nvar, 2))) * 150
    g_vrange[:, 0] *= -1

    p0 = np.matrix(np.ones([Nvar, 1])) * 10.0
    x0 = np.divide(p0 - g_vrange[:, 0], g_vrange[:, 1] - g_vrange[:, 0])

    y0 = fuc_test0(x0)
    print('x0', x0)
    print('y0', y0)

    step = 0.01
    g_noise = 0.001
    g_cnt = 0
    g_data = np.zeros([1, Nvar + 2])
    Imat = np.matrix(np.identity(Nvar))

    rcds = RCDS(func_obj, g_noise, g_cnt, Nvar, g_vrange, g_data, Imat)
    # rcds = RCDS(fuc_test0, g_noise, g_cnt, Nvar, g_vrange, g_data, Imat)
    (xm, fm, nf) = rcds.powellmain(x0, step, Imat, maxIt=1000, maxEval=10000)
    print('start', x0)
    print('end', xm)
    print('start', y0)
    print('end', fm)
    print('end', fuc_test0(xm))


test0()