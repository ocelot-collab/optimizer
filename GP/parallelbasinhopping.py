
import numpy as np
from scipy.optimize import basinhopping
import multiprocessing as mp

# handle 'IOError: [Errno 4] Interrupted system call' errors from multiprocessing.Queue.get
#https://stackoverflow.com/questions/14136195/what-is-the-proper-way-to-handle-in-python-ioerror-errno-4-interrupted-syst
import errno
def my_queue_get(queue, block=True, timeout=None):
    while True:
        try:
            return queue.get(block, timeout)
        except IOError, e:
            if e.errno != errno.EINTR:
                raise
# Now replace instances of queue.get() with my_queue_get(queue), with other
# parameters passed as usual.
                    
# see here https://eli.thegreenplace.net/2012/01/16/python-parallelizing-cpu-bound-tasks-with-multiprocessing/
# and here https://stackoverflow.com/questions/37060091/multiprocessing-inside-function

def bworker(f,x0,bkwargs,out_q):
    # worker invoked in a process puts the results in the output queue out_q
    res = basinhopping(f, x0, **bkwargs)
    out_q.put([[res.x, res.fun[0][0]]])

# parallelize minimizations using different starting positions using multiprocessing, scipy.optimize.minimize
def parallelbasinhopping(f,x0s,bkwargs):
    # f is fcn to minimize
    # x0s are positions to start search from
    # fargs are arguments to pass to f
    # margs are arguments to pass to scipy.optimize.minimize
    
    
    # Each process will get a queue to put its result in
    out_q = mp.Queue()
        
    # arguments to loop over
    args = [(f,x0,bkwargs,out_q) for x0 in x0s]

    # https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop#9786225
    # also could try concurrent futures
#    import multiprocessing
#    pool = multiprocessing.Pool()
#    res = np.array(pool.map(minimizeone, args))
#    res = np.array(pool.map(l, range(10)))
    
    # seems like this maybe be needed 
    # https://stackoverflow.com/questions/37060091/multiprocessing-inside-function
    
    nprocs = len(x0s)
    procs = []

    for i in range(nprocs):
        p = mp.Process(
                target=bworker,
                args=args[i])
        procs.append(p)
        p.start()

    res = [];
    for i in range(nprocs):
        #res += out_q.get()
        res += my_queue_get(out_q)

    for p in procs:
        p.join()
        
    res = np.array(res)
    #print 'res = ', res
    res = res[res[:,1]==np.min(res[:,1])][0]
    #print 'selected min is ',res
    res = np.array(res[0])

    return res
