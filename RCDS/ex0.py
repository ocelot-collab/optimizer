
from rcds import *

if 1:
    p0 = np.matrix(np.ones([Nvar,1]))*15.0
    x0 = np.divide(p0-g_vrange[:,0],g_vrange[:,1]-g_vrange[:,0])

    y0 = func_obj(x0)

    step = 0.01
    (xm,fm,nf)=powellmain(func_obj,x0,step,Imat)
    print([x0,xm])
    print(y0,fm)
    #for ii in range(g_cnt):
    #       print(g_data[ii,:])
