
import matplotlib
import numpy as np
import time
from datetime import datetime as dt
import os
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

#matplotlib.rcParams['xtick.direction'] = 'out'
#matplotlib.rcParams['ytick.direction'] = 'out'


def plotheatmap(function, fargs, rangex, rangey, ngrid=25, xlabel='Device 1', ylabel='Device 2', description='', series = None):
    #function - pointer to the function to evaluate
    #rangex & rangey - [xlow,xhigh]
    #ngrid - number of points to plot
    #xlabel & ylabel - axes labels
    #description - plot title
    #series - list of [x,y] pairs to plot showing the optimization path 
    
    # evaluate function
    xs = np.linspace(min(rangex),max(rangex),ngrid)
    ys = np.linspace(min(rangey),max(rangey),ngrid)
    
    ## for vectorizable functions
    #xys = np.zeros([ngrid,ngrid,2])
    #for i in range(ngrid):
        #for j in range(ngrid):
            #xys[i][j] = [xs[i],ys[j]]
    #xys = xys.reshape([ngrid*ngrid,2])
    #zs = function(xys,*fargs)
    #Z = zs.reshape([ngrid,ngrid])
    
    # for non-vectorizable functions
    Z = np.zeros([ngrid,ngrid])
    for i in range(ngrid):
        for j in range(ngrid):
            Z[i][j] = function(np.array([[xs[i],ys[j]]],ndmin=2),*fargs)[0]
            
    X, Y = np.meshgrid(xs, ys)

    # bug in matplotlib prevents plotting Z with all zeros so perturb if this is the case

    if(np.sum(Z)==0):
        Zshape = np.shape(Z)
        Z = Z + 1.e-6 * np.random.randn(Zshape[0], Zshape[1])
        print ('WARNING: z-values are all zero so adding a small random field to it')
    
    # Create a simple contour plot with labels using default colors.  The
    # inline argument to clabel will control whether the labels are draw
    # over the line segments of the contour, removing the lines beneath
    # the label
    plt.figure()
    #print 'np.shape(X) = ',np.shape(X)
    #print 'np.shape(Y) = ',np.shape(Y)
    #print 'np.shape(Z.T) = ',np.shape(Z.T)
    #CS = plt.contour(X, Y, Z.T, cmap='viridis')
    CS = plt.contour(X, Y, Z.T, cmap='jet')
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(description)
    #plt.show() # doesn't work with ocelot's threading
    
    # Ocelot path series
    try:
        if series is not None:
            series = np.array(series)
            #print 'series = ', series
            ptsx = series[:,0]
            ptsy = series[:,1]
            #print 'ptsx = ', ptsx
            #print 'ptsy = ', ptsy
            plt.plot(ptsx, ptsy, 'k--')
            plt.plot(ptsx, ptsy, 'k.', ms=8)
    except:
        pass
    
    # plot path
    try: # if running under a profile, save to profile directory
        #username = os.environ['PHYSICS_USER']
        #if username == 'none':
            #username = 'Ocelot'
        #basepath = '/home/physics/' + username + '/OcelotPlots/'

        # save to a directory under the user's home directory
        homepath = os.environ['HOME']
        basepath = homepath + '/ocelot/plots/'

        year = str(dt.fromtimestamp(time.time()).strftime('%Y'))
        month = str(dt.fromtimestamp(time.time()).strftime('%m'))
        day = str(dt.fromtimestamp(time.time()).strftime('%d'))
        basepath = str(basepath+year+'/'+year+'-'+month+'/'+year+'-'+month+'-'+day+'/')
        
    except:
        basepath = os.environ['PWD']+'/'

    try:
        os.makedirs(basepath) # make it if it doesn't exist
    except:
        pass

    # plot file path
    timestr = time.strftime("%Y%m%d-%H%M%S") + str(round(time.time()%1*1000)/1000)[1:]
    fpath = basepath + 'heatmap-' + function.__name__ + '-' + timestr + '.png'

    # save plot
    plt.savefig(fpath, bbox_inches='tight')
    plt.close('all') # close all open figures to prevent memory leaks
    print('Saved contour plot for ' + function.__name__ + ' to ' + fpath)

    #ax1.scatter(ptsx,ptsy,s=50,c=data.iloc[:moment+1,-1])
    #ax1.set_xlim(axes[:2])
    #ax1.set_ylim(axes[2:])
