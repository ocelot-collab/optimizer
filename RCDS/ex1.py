from rcdsClass import RCDS
import numpy as np

if 1:
    g_noise = 0.1
    g_cnt = 0
    Nvar = 6
    g_vrange = np.matrix(np.ones((Nvar,2)))*150
    g_vrange[:,0] *= -1
    g_data = np.zeros([1,Nvar+2])
    Imat = np.matrix(np.identity(Nvar))

    rcds = RCDS(g_noise, g_cnt, Nvar, g_vrange, g_data, Imat)
    p0 = np.matrix(np.ones([Nvar,1]))*15.0
    x0 = np.divide(p0-g_vrange[:,0],g_vrange[:,1]-g_vrange[:,0])

    y0 = rcds.func_obj(x0)

    step = 0.01
    (xm,fm,nf)=rcds.powellmain(rcds.func_obj,x0,step,Imat)
    print([x0,xm])
    print(y0,fm)
    for ii in range(g_cnt):
        print(g_data[ii,:])
