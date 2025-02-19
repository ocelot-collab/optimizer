# -*- coding: iso-8859-1 -*-
import sys
try:
    sys.path.append('./matrixmodel/')
except:
    pass
import numpy as np
#from scipy.optimize import minimize
#import numdifftools as nd
import matplotlib.pyplot as plt
from numpy.linalg import inv
import re
import pandas as pd
import time as steve
from parallelstuff import parallelmap
from archive_stuff import *
from scipy import special
from genesis_tools import *
from scipy.optimize import curve_fit as fit

try:
    import epics
    global_disable_epics = False
except:
    global_disable_epics = True
#global_disable_epics = True

class Beamconfig:
    def __init__(self, config_type, arc_time = None, mquadpvs = None, beamEnergyMeV = 3370, beamCurrent=1200, emitx = None, emity = None, und_Ks = None, undQuadGrad = 10):

        self.disable_epics = global_disable_epics
        if self.disable_epics:
            print '\n', '\n', 'Epics disabled. Will not be able to get current device values.'
        else:
            print '\n', '\n', 'CAUTION!! Epics is enabled. DO NOT MAKE CHANGES TO MACHINE (via caput commands, which should never appear in this script).'
        if config_type == 'current':
            if self.disable_epics:
                print 'Epics disabled (in beamconfig.py). Cannot get current config. Terminating scipt.'
                sys.exit()
            else:
                self.quads, self.und_quads, self.mquad_names = self.get_model()[0:3]
                self.beamEnergyMeV = 1000.*epics.caget('BEND:DMP1:400:BDES') ##################### Check units! Should this factor of 1000 be here?
                self.emitx = epics.caget('WIRE:IN20:561:EMITN_X')
                self.emity = epics.caget('WIRE:IN20:561:EMITN_Y')
        if config_type == 'archive':
            self.disable_epics = True
            self.quads, self.und_quads, self.mquad_names = self.get_model()[0:3]
            self.quads[:,0], self.beamEnergyMeV = self.get_arc_config(arc_time)
        if config_type == 'user':
            self.disable_epics = True
            self.quads, self.und_quads, self.mquad_names = self.get_model()[0:3]
            self.quads[:,0] = mquadpvs
            self.beamEnergyMeV = beamEnergyMeV
            self.und_quads[0,0:2] = [-1.*undQuadGrad*10*.04, .04] #################Change me????????????????
            self.und_quads[1:,0] = [undQuadGrad*10*.08*(-1)**i for i in range(len(self.und_quads[1:,0])) ]

        #print self.und_quads
        #print self.mquad_names
        self.mquad_dict = {}
        for i, mquad_name in enumerate(self.mquad_names):
            self.mquad_dict[mquad_name] = i
        self.zend = sum(self.quads[:,1]) + sum(self.quads[:,2])
        self.gamma_rel = self.beamEnergyMeV / 0.511 #relatvistic gamma, not to be confused with the twiss parameter 'gamma'
        self.beamCurrent = beamCurrent
        self.delgam = (0.629 + 0.477 * self.beamCurrent / 1000.) / 0.511
        if type(emitx) is not type(None):
            self.emitx = emitx #normalized emittance
        else:
            try: type(self.emitx)
            except:
                print 'Failed to specify or acquire emitx. Using default value of 0.4e-6'
                self.emitx = 0.4e-6
        if type(emity) is not type(None):
            self.emity = emity #normalized emittance
        else:
            try: type(self.emity)
            except:
                print 'Failed to specify or acquire emity. Using default value of 0.4e-6'
                self.emity = 0.4e-6
        self.emitx_u = self.emitx/self.gamma_rel #un-normalized emittance
        self.emity_u = self.emity/self.gamma_rel #un-normalized emittance
        if type(und_Ks) is not type(None):
            self.und_Ks = und_Ks
        else:
            und_Ks=np.array([pullData('USEG:UND1:'+str(1+i)+'50:KACT','2016-09-21T18:50:34')[1][0] for i in range(33)])
        self.xkx = 0
        self.xky = 1
        self.npart = 2048
        self.a = 3.e8/(beamEnergyMeV*1.e6) # electron charge divided by relativistic momentum, SI units m/(V*s)
        self.matched_twiss_x, self.matched_twiss_y = self.calcmatch_FODO_genesis(beamEnergyMeV=beamEnergyMeV, undQuadGrad=10) #matched_twiss_x, matched_twiss_y are the twiss parameters that are required halfway through the first undulator quad in order to produce a match through the undulator.
        self.matched_twiss_x_matrixmodel, self.matched_twiss_y_matrixmodel = self.calcmatch_FODO_matrixmodel(beamEnergyMeV=beamEnergyMeV,undQuadGrad=37.5)
        self.input_twiss_x, self.input_twiss_y = self.back_prop() #input_twiss_x, input_twiss_y are the twiss parameters that are required at the beginning of beamline in order to produce the matched_twiss params at the undulator.
        #self.quad_names = [' Unfinished...
        print 'Beamline successfully configured.'

    def output_twiss(self, quadx_id=0, quady_id=1, dquadx = 0, dquady = 0):
        #quadx, quady are the indeces of the quads you wish to vary
        #dquadx, dquady are the amounts that you wish to vary. Units are the same as the quad PVs, i.e. field integrals: gradient(kG/m)*length(m)
        quads = np.copy(self.quads) #make a copy so you don't alter the master copy (matched quad values)
        quads[quadx_id,0] = quads[quadx_id,0] + dquadx
        quads[quady_id,0] = quads[quady_id,0] + dquady
        twissx, twissy = self.twiss(self.zend, self.input_twiss_x, self.input_twiss_y, quads)
        return twissx, twissy


    def varied_twiss(self, quadx_id = 0, quady_id = 1, rel_rangex=[-1,3], rel_rangey=[-2.5,1.5], resolution=[11,11]):
        #this method outputs a list of twiss parameters (halfway through the first undulator quad) for various configurations close to the match.
        #quadx and quady are the indeces of the quads you wish to vary. 0 corresponds to the first quad in the beam line (currently LTU620), and so on.
        #rangex and rangey are the ranges relative to the matched values for the selected quads.
        #the output is a nested list is in the form of a grid, with a list of two (2x2) np arrays for each entry, one for the x-twiss and one for the y-twiss. For example, twiss_grid[0][0] will return a list with two np arrays, corresponding to the uppermost, leftmost configuration in the range.
        #
        numrow = resolution[0]
        numcol = resolution[1]
        left, right = rel_rangex[0], rel_rangex[1]
        down, up = rel_rangey[0], rel_rangey[1]
        sx = np.linspace(left, right, numcol) #range of 'scan' (around the matched quad values), and resolution
        sy = np.linspace(down, up, numrow)
        rangex = [self.quads[quadx_id,0]+left, self.quads[quadx_id,0]+right]
        rangey = [self.quads[quady_id,0]+down, self.quads[quady_id,0]+up]
        twiss_grid = []
        quadval_grid = []
        for i, dquady in enumerate(sy):
            twiss_row = []
            quadval_row = []
            for j, dquadx in enumerate(sx):
                twissx, twissy = self.output_twiss(quadx_id=quadx_id, quady_id=quady_id, dquadx=dquadx, dquady=dquady)
                this_quadval = [self.quads[quadx_id,0] + dquadx, self.quads[quady_id,0] + dquady]
                quadval_row += [this_quadval]
                this_twiss = [twissx, twissy]
                twiss_row += [this_twiss]
            twiss_grid = [twiss_row] + twiss_grid
            quadval_grid = [quadval_row] + quadval_grid
        return twiss_grid, rangex, rangey, quadval_grid

    def corrmat(self, pvnames, fastQ = False, corrScale = 1.):
        try:
            np.exp(corrScale * 1. + 0.) # can we do math with corrScale?
            corr_scale_factor = max(-1,min(1,corrScale))
        except:
            print 'matrixmodel.beamconfig.corrmat - WARNING: ', corrScale, ' is an invalid choice for corrScale (should be a float between 0 and 1)'
            corr_scale_factor = 1.
        corrparams = []
        for i in range(len(pvnames)-1):
            rho = self.corrparam(pvnames[i], pvnames[i+1], fastQ = fastQ)[0]
            print 'correlation param (rho) for ', pvnames[i], pvnames[i+1], ' = ', rho
            corrparams += [corr_scale_factor * rho]
        mat = np.identity(len(corrparams)+1)
        for i, rho in enumerate(corrparams):
            mat[i,i+1], mat[i+1,i] = rho, rho
        print 'correlation matrix for given PVs = ', mat
        return mat

    def corrmat_full(self, pvnames, fastQ = False):
        mat = np.identity(len(pvnames))
        for od in range(len(pvnames)-1): # off diagonal rows
            for i in range(len(pvnames)-1-od): # elements along one off-diag row
                loc = i+1+od
                rho = self.corrparam(pvnames[i], pvnames[loc], fastQ = fastQ)[0]
                print 'correlation param (rho) for ', pvnames[i], pvnames[loc], ' = ', rho
                mat[i,loc], mat[loc,i] = rho, rho
        print 'correlation matrix for given PVs = ', mat
        return mat


    def corrparam(self, pvname1, pvname2, fastQ = False, eps = 1.e-5):
        quadx_id, quady_id = self.mquad_dict[pvname1], self.mquad_dict[pvname2]
        #eps = np.finfo(float).eps
        #eps = 1.e-6 #1.e-16 is too small??
        #corrplot, quadval_grid = self.beam_size_heatmap(quadx_id=quadx_id, quady_id=quady_id, rel_rangex=[-eps, eps], rel_rangey=[-eps, eps], resolution=[3,3], showplot=False, filename=None)
        corrplot, quadval_grid = self.beam_size_heatmap_parallel(quadx_id=quadx_id, quady_id=quady_id, rel_rangex=[-eps, eps], rel_rangey=[-eps, eps], resolution=[3,3], showplot=False, filename=None, fastQ=fastQ)
        #print 'corrplot = ', corrplot
        hess = np.zeros([2, 2])
        hess[0,0] = 1/eps*( (corrplot[1,2]-corrplot[1,1])/eps - (corrplot[1,1]-corrplot[1,0])/eps )
        hess[1,0] = 1/eps*( (corrplot[0,2]-corrplot[0,1])/eps - (corrplot[1,2]-corrplot[1,1])/eps )
        hess[0,1] = 1/eps*( (corrplot[0,2]-corrplot[1,2])/eps - (corrplot[0,1]-corrplot[1,1])/eps )
        hess[1,1] = 1/eps*( (corrplot[0,1]-corrplot[1,1])/eps - (corrplot[1,1]-corrplot[2,1])/eps )
        print 'hess = \n', hess
        hess[0,0] = 1/eps*( (corrplot[1,2]-corrplot[1,1])/eps - (corrplot[1,1]-corrplot[1,0])/eps )
        hess[1,0] = 1/(2*eps)*( (corrplot[0,2]-corrplot[2,2])/eps - (corrplot[0,0]-corrplot[2,0])/eps )
        hess[0,1] = 1/(2*eps)*( (corrplot[2,2]-corrplot[2,0])/eps - (corrplot[0,2]-corrplot[0,0])/eps )
        hess[1,1] = 1/eps*( (corrplot[2,1]-corrplot[1,1])/eps - (corrplot[1,1]-corrplot[0,1])/eps )
        print 'hess = \n', hess
        rho = -1*hess[0,1]/np.sqrt(hess[0,0]*hess[1,1]) #should I have this negative 1 here????
        covar = hess
        return rho, covar, corrplot

    def beam_size_heatmap_parallel(self, quadx_id=0, quady_id=1, rel_rangex=[-1,3], rel_rangey=[-2.5,1.5], resolution=[11,11], showplot = True, filename = None, fastQ=False):
        #method to generate heatmap of average beam sizes in undulator using the simple matrix model.

        #zs, sizes, sigma_xs, sigma_ys  = self.beam_size_undulator_only(self.matched_twiss_x_matrixmodel, self.matched_twiss_y_matrixmodel, self.und_quads)[1:5]
        #plt.plot(zs, sizes)
        #plt.plot(zs, sigma_xs)
        #plt.plot(zs, sigma_ys)
        #plt.show()
        #plt.close()
        #print zs, sizes
        print  '\n', '\n', 'Generating beam size heatmap...'
        twiss_grid1, rangex, rangey, quadval_grid = self.varied_twiss(quadx_id=quadx_id, quady_id=quady_id, rel_rangex=rel_rangex, rel_rangey=rel_rangey, resolution=resolution)
        corrplot = np.zeros([resolution[0],resolution[1]])

        #evaluate avg beam size on grid
        iterator = 0
        argslist = []
        for row in twiss_grid1:
            for twiss in row:
                twissx = twiss[0]
                twissy = twiss[1]
                #betax, alphax = twissx[0,0], twissx[0,1]
                #betay, alphay = twissy[0,0], twissy[0,1]
                argslist += [(twissx, twissy)]
                iterator += 1
        if fastQ == True:
            resultslist = parallelmap(self.beam_size_undulator_only_onearg_fast, argslist, ([self.und_quads]) )
        else:
            resultslist = parallelmap(self.beam_size_undulator_only_onearg, argslist, ([self.und_quads]) )
        #print 'Beam size evaluated ', len(argslist), ' times.'
        #print 'length of results list = ', len(resultslist)
        #print 'resultslist[0][0][0] = ', resultslist[0][0][0]

        iterator = 0
        for i, row in enumerate(twiss_grid1):
            for j, twiss in enumerate(row):
                #print resultslist[iterator][0][0]
                corrplot[i,j] = resultslist[iterator][0][0]
                iterator += 1
        #assign plot range to axes
        top = rangey[1]
        bottom = rangey[0]
        left = rangex[0]
        right = rangex[1]
        extent = [left, right, bottom, top] #left, right, bottom, top

        plt.imshow(corrplot, cmap='hot', interpolation='nearest', extent=extent)
        plt.title('Avg Beam Size in Undulator')
        plt.xlabel('QUAD:LTU1:620 (kG)')
        plt.ylabel('QUAD:LTU1:640 (kG)')
        cbar = plt.colorbar()
        cbar.set_label('Meters')
        if type(filename) != type(None):
            plt.savefig(filename + '.png') # save figure
        if showplot == True:
            plt.show()
        plt.close()
        print 'Finished beam size heatmap.'
        return corrplot, quadval_grid


    def beam_size_heatmap(self, pv_names = None, quadx_id=0, quady_id=1, rel_rangex=[-1,3], rel_rangey=[-2.5,1.5], resolution=[11,11], showplot = False, filename = None):
        #method to generate heatmap of average beam sizes in undulator using the simple matrix model.

        #zs, sizes, sigma_xs, sigma_ys  = self.beam_size_undulator_only(self.matched_twiss_x_matrixmodel, self.matched_twiss_y_matrixmodel, self.und_quads)[1:5]
        #plt.plot(zs, sizes)
        #plt.plot(zs, sigma_xs)
        #plt.plot(zs, sigma_ys)
        #plt.show()
        #plt.close()
        #print zs, sizes
        if type(pv_names) is not type(None):
            if len(pv_names) > 2:
                print 'WARNING: More than 2 pv names were passed to beam_size_heatmap function. Only the first two devices in pv_names will be varied to produce the heatmap.'
            if len(pv_names) < 2:
                print 'WARNING: Not enough pv names passed to beam_size_heatmap. This function requires len(pv_names) = 2.'
            quadx_id, quady_id = self.mquad_dict[pv_names[0]], self.mquad_dict[pv_names[1]]
        print  '\n', '\n', 'Generating beam size heatmap...'
        twiss_grid1, rangex, rangey, quadval_grid = self.varied_twiss(quadx_id=quadx_id, quady_id=quady_id, rel_rangex=rel_rangex, rel_rangey=rel_rangey, resolution=resolution)
        corrplot = np.zeros([resolution[0],resolution[1]])

        #evaluate avg beam size on grid
        iterator = 1
        runtimes = []
        max_iter = resolution[0]*resolution[1]
        for i, row in enumerate(twiss_grid1):
            for j, twiss in enumerate(row):
                t0 = steve.time()
                twissx = twiss[0]
                twissy = twiss[1]
                #betax, alphax = twissx[0,0], twissx[0,1]
                #betay, alphay = twissy[0,0], twissy[0,1]
                size = self.beam_size_undulator_only(twissx, twissy, self.und_quads)[0]
                corrplot[i,j] = size
                print iterator, '/', max_iter, ' complete.'
                iterator += 1

        #assign plot range to axes
        top = rangey[1]
        bottom = rangey[0]
        left = rangex[0]
        right = rangex[1]
        extent = [left, right, bottom, top] #left, right, bottom, top

        plt.imshow(corrplot, cmap='hot', interpolation='nearest', extent=extent)
        plt.title('Avg Beam Size in Undulator')
        if type(pv_names) is not type(None):
            plt.xlabel(pv_names[0] + ' (kG)')
            plt.ylabel(pv_names[1] + ' (kG)')
        else:
            plt.xlabel('quadx_id = ' + str(quadx_id) + ' (kG)')
            plt.ylabel('quady_id = ' + str(quady_id) + ' (kG)')
        cbar = plt.colorbar()
        cbar.set_label('Meters')
        if type(filename) != type(None):
            if filename[-3:] == 'png':
                plt.savefig(filename)
            else:
                plt.savefig(filename + '.png') # save figure
        if showplot == True:
            plt.show()
        plt.close()
        print 'Finished beam size heatmap.'
        return corrplot, quadval_grid

    def genesis_heatmap(self, quadx_id=0, quady_id=1, rel_rangex=[-1,3], rel_rangey=[-2.5,1.5], resolution=[11,11], reoptimize = False, showplot = True, filename = None):
        #ONLY PLOTS FEL HEATMAP... DOES NOT PLOT: taper, detuning optimization, bunching factor vs z, radphase vs z, bunching phase vs z, power vs z, log(power) vs z, and beam size vs z
        print  '\n', '\n', 'Generating FEL heatmap...'
        twiss_grid1, rangex, rangey, quadval_grid = self.varied_twiss(quadx_id=quadx_id, quady_id=quady_id, rel_rangex=rel_rangex, rel_rangey=rel_rangey, resolution=resolution)
        corrplot = np.zeros([resolution[0],resolution[1]])

        #evaluate genesis on grid
        iterator = 1
        runtimes = []
        max_iter = resolution[0]*resolution[1]
        s = serpent()
        s.input_params['emitx'] = 2.34556e-6
        self.configure_serpent(s) #configures input params and optimizes detuning using the matched case
        if s.input_params['emitx'] == 2.34556e-6:
            print 'WARNING WARNING: WE FUCKED UP'
        for i, row in enumerate(twiss_grid1):
            for j, twiss in enumerate(row):
                t0 = steve.time()
                twissx = twiss[0]
                twissy = twiss[1]
                #betax, alphax = twissx[0,0], twissx[0,1]
                #betay, alphay = twissy[0,0], twissy[0,1]
                self.configure_serpent(s, input_twiss_x = twissx, input_twiss_y = twissy, optimize_detuning=reoptimize, plotQ=False) #updates the configuration for this (ith, jth) square of the heatmap. Reoptimizes detuning if specified.
                zs, powers = s.run_genesis_for_twiss(plotQ=False)[0:2]
                end_power = powers[-1]
                corrplot[i,j] = end_power
                runtime = steve.time()-t0
                runtimes += [runtime]
                avg_runtime = np.mean(np.array(runtimes))
                time_remaining = (max_iter - iterator)*avg_runtime
                print 'Genesis run ' + str(iterator) + '/' + str(max_iter) +' complete. Power at zstop = ' + str(end_power)
                print 'Took ', round(runtime,1), ' seconds'
                print 'Estimated time remaining: ', str(int(time_remaining/60.)) + 'min' + str(int(round(np.mod(time_remaining/60.,1.)*60,0))) + 'sec.'

                iterator += 1

        #assign plot range to axes
        top = rangey[1]
        bottom = rangey[0]
        left = rangex[0]
        right = rangex[1]
        extent = [left, right, bottom, top] #left, right, bottom, top

        plt.imshow(corrplot, cmap='hot', interpolation='nearest', extent=extent)
        plt.title('Est FEL Powers (Genesis)')
        plt.xlabel('QUAD:LTU1:620 (kG)')
        plt.ylabel('QUAD:LTU1:640 (kG)')
        cbar = plt.colorbar()
        cbar.set_label('Watts')
        if type(filename) != type(None):
            plt.savefig(filename + '.png') # save figure
        if showplot == True:
            plt.show()
        plt.close()
        print 'Finished generating FEL heatmap. Took ', np.sum(runtimes)/60., ' minutes.',  '\n', '\n'

    def genesis_heatmap_along_z(self, quadx_id=0, quady_id=1, rel_rangex=[-1,3], rel_rangey=[-2.5,1.5], resolution=[11,11], reoptimize = False, showplot = True, filename = None):
        #ONLY PLOTS FEL HEATMAP... DOES NOT PLOT: taper, detuning optimization, bunching factor vs z, radphase vs z, bunching phase vs z, power vs z, log(power) vs z, and beam size vs z
        print  '\n', '\n', 'Generating FEL heatmap...'
        twiss_grid1, rangex, rangey, quadval_grid= self.varied_twiss(quadx_id=quadx_id, quady_id=quady_id, rel_rangex=rel_rangex, rel_rangey=rel_rangey, resolution=resolution)
        corrplot_list = []
        #evaluate genesis on grid
        iterator = 1
        runtimes = []
        max_iter = resolution[0]*resolution[1]
        s = serpent()
        s.input_params['emitx'] = 2.34556e-6
        self.configure_serpent(s, detuning_zstop = 1.e6) #configures input params and optimizes detuning using the matched case
        zs = s.run_genesis_for_twiss(plotQ=False)[0] # run once just to get list of z values from genesis.
        for z in zs:
            corrplot_list = corrplot_list + [ [z, np.zeros([resolution[0],resolution[1]])] ] #make a list of ordered pairs formatted: [z value, corrplot]

        if s.input_params['emitx'] == 2.34556e-6:
            print 'WARNING WARNING: WE FUCKED UP'
        for i, row in enumerate(twiss_grid1):
            for j, twiss in enumerate(row):
                t0 = steve.time()
                twissx = twiss[0]
                twissy = twiss[1]
                #betax, alphax = twissx[0,0], twissx[0,1]
                #betay, alphay = twissy[0,0], twissy[0,1]
                self.configure_serpent(s, input_twiss_x = twissx, input_twiss_y = twissy, optimize_detuning=reoptimize, plotQ=False) #updates the configuration for this (ith, jth) square of the heatmap. Reoptimizes detuning if specified.
                zs, powers = s.run_genesis_for_twiss(plotQ=False)[0:2]
                for k in range(len(zs)):
                    corrplot_list[k][1][i,j] = powers[k]
                runtime = steve.time()-t0
                runtimes += [runtime]
                avg_runtime = np.mean(np.array(runtimes))
                time_remaining = (max_iter - iterator)*avg_runtime
                print 'Genesis run ' + str(iterator) + '/' + str(max_iter) +' complete.'
                print 'Took ', round(runtime,1), ' seconds'
                print 'Estimated time remaining: ', str(int(time_remaining/60.)) + 'min' + str(int(round(np.mod(time_remaining/60.,1.)*60,0))) + 'sec.'

                iterator += 1


        #assign plot range to axes
        top = rangey[1]
        bottom = rangey[0]
        left = rangex[0]
        right = rangex[1]
        extent = [left, right, bottom, top] #left, right, bottom, top

        for corrplot in corrplot_list:
            plt.imshow(corrplot[1], cmap='hot', interpolation='nearest', extent=extent)
            plt.title('Est FEL Powers (Genesis)')
            plt.xlabel('QUAD:LTU1:620 (kG)')
            plt.ylabel('QUAD:LTU1:640 (kG)')
            cbar = plt.colorbar()
            cbar.set_label('Watts')
            if type(filename) != type(None):
                plt.savefig(filename + '_z' + str(corrplot[0]) + '.png') # save figure
            if showplot == True:
                plt.show()
            plt.close()
        print 'Finished generating FEL heatmaps. Took ', np.sum(runtimes)/60., ' minutes.',  '\n', '\n'


        ################# below function is work in progress
    def genesis_FEL_vs_betas(self, quadx_id=0, quady_id=1, rel_frac_rangex=[-1,3], rel_frac_rangey=[-2.5,1.5], resolution=[11,11], reoptimize = False, showplot = True, filename = None):
        #ONLY PLOTS FEL HEATMAP... DOES NOT PLOT: taper, detuning optimization, bunching factor vs z, radphase vs z, bunching phase vs z, power vs z, log(power) vs z, and beam size vs z
        print  '\n', '\n', 'Generating FEL heatmap...'
        twiss_grid1, rangex, rangey, quadval_grid = self.varied_twiss(quadx_id=quadx_id, quady_id=quady_id, rel_rangex=rel_rangex, rel_rangey=rel_rangey, resolution=resolution)
        corrplot = np.zeros([resolution[0],resolution[1]])

        #evaluate genesis on grid
        iterator = 1
        runtimes = []
        max_iter = resolution[0]*resolution[1]
        s = serpent()
        self.configure_serpent(s) #configures input params and optimizes detuning using the matched case
        for i, row in enumerate(twiss_grid1):
            for j, twiss in enumerate(row):
                t0 = steve.time()
                twissx = twiss[0]
                twissy = twiss[1]
                #betax, alphax = twissx[0,0], twissx[0,1]
                #betay, alphay = twissy[0,0], twissy[0,1]
                self.configure_serpent(s, input_twiss_x = twissx, input_twiss_y = twissy, optimize_detuning=reoptimize, plotQ=False) #updates the configuration for this (ith, jth) square of the heatmap. Reoptimizes detuning if specified.
                zs, powers = s.run_genesis_for_twiss(plotQ=False)[0:2]
                end_power = powers[-1]
                corrplot[i,j] = end_power
                runtime = steve.time()-t0
                runtimes += [runtime]
                avg_runtime = np.mean(np.array(runtimes))
                time_remaining = (max_iter - iterator)*avg_runtime
                print 'Genesis run ' + str(iterator) + '/' + str(max_iter) +' complete. Power at zstop = ' + str(end_power)
                print 'Took ', round(runtime,1), ' seconds'
                print 'Estimated time remaining: ', str(int(time_remaining/60.)) + 'min' + str(int(round(np.mod(time_remaining/60.,1.)*60,0))) + 'sec.'

                iterator += 1

        #assign plot range to axes
        top = rangey[1]
        bottom = rangey[0]
        left = rangex[0]
        right = rangex[1]
        extent = [left, right, bottom, top] #left, right, bottom, top

        plt.imshow(corrplot, cmap='hot', interpolation='nearest', extent=extent)
        plt.title('Est FEL Powers (Genesis)')
        plt.xlabel('QUAD:LTU1:620 (kG)')
        plt.ylabel('QUAD:LTU1:640 (kG)')
        cbar = plt.colorbar()
        cbar.set_label('Watts')
        if type(filename) != type(None):
            plt.savefig(filename + '.png') # save figure
        if showplot == True:
            plt.show()
        plt.close()
        print 'Finished generating FEL heatmap. Took ', np.sum(runtimes)/60., ' minutes.',  '\n', '\n'


    def genesis_single_run(self, quadx_id = 0, quady_id = 1, dquadx = 0, dquady = 0, detuning_zstop = None, plotQ = False, filename = None):
        #plots taper, detuning optimization, bunching factor vs z, radphase vs z, bunching phase vs z, power vs z, log(power) vs z, and beam size vs z, FOR CHOSEN CONFIGURATION. Defaults to matched config.
        print '\n', '\n', 'Running single genesis simulation with specified settings...'
        twissx, twissy = self.output_twiss(quadx_id=quadx_id, quady_id=quady_id, dquadx=dquadx, dquady=dquady)
        betax, alphax = twissx[0,0], twissx[0,1]
        betay, alphay = twissy[0,0], twissy[0,1]
        print 'Twiss used for genesis run:'
        print 'twissx = ', '\n', twissx, '\n', 'twissy = ', '\n', twissy
        print 'For reference, the matched twiss are: '
        print 'matched twissx = ', '\n', self.matched_twiss_x, '\n', 'matched twissy = ', '\n', self.matched_twiss_y

        s = serpent()
        self.configure_serpent(s, input_twiss_x = twissx, input_twiss_y = twissy, optimize_detuning = True, detuning_zstop = detuning_zstop, plotQ = plotQ)

        #s.run_genesis_for_twiss(betax=betax, betay=betay, alphax=alphax, alphay=alphay, plotQ=True, plotName = filename)
        s.run_genesis_for_twiss(plotQ=plotQ, plotName = filename)
        print 'Finished running single genesis sim.' '\n', '\n'


    def get_arc_config(self, ttime): #time format '2017-07-01T00:00:01'
        pvlist = ['QUAD:LTU1:620:BCTRL', 'QUAD:LTU1:640:BCTRL', 'QUAD:LTU1:660:BCTRL', 'QUAD:LTU1:665:BCTRL', 'QUAD:LTU1:680:BCTRL', 'QUAD:LTU1:720:BCTRL', 'QUAD:LTU1:730:BCTRL', 'QUAD:LTU1:740:BCTRL', 'QUAD:LTU1:750:BCTRL', 'QUAD:LTU1:760:BCTRL', 'QUAD:LTU1:770:BCTRL', 'QUAD:LTU1:820:BCTRL', 'QUAD:LTU1:840:BCTRL', 'QUAD:LTU1:860:BCTRL', 'QUAD:LTU1:880:BCTRL', 'QUAD:UND1:180:BCTRL']
        quad_vals = []
        for pv in pvlist:
            val = pullData(pv, ttime, ttime )[1][0]
            quad_vals.append(val)
        energy = pullData('BEND:DMP1:400:BDES', ttime, ttime)[1][0]
        return np.array(quad_vals), energy


    def my_hess(self, f):
        #returns finite difference approximation to hessian of -log(FEL/A) response. FEL response appears to be well-behaved around optimum; shouldn't encounter numerical problems(...?)
        #evaluates FEL at grid points 0-12, where point 6 is the matched coordinate and the point for which the hessian is calculated. Uses these FEL values to calculate finite-different approximation to the derivatives.
        #X  X  0  X  X
        #X  1  2  3  X
        #4  5  6  7  8
        #X  9  10 11 X
        #X  X  12 X  X
        eps = np.finfo(float).eps
        #eps = 0.1
        deltas = [ [0, 2*eps], [-eps, eps], [0, eps], [eps, eps], [-2*eps, 0], [-eps, 0], [0, 0], [eps, 0], [2*eps, 0], [-eps, -eps], [0, -eps], [eps, -eps], [0, -2*eps] ]
        resultslist = parallelmap(f, deltas, () )
        FEL = resultslist

        hess = np.zeros([2, 2])
        hess[0,0] = ((FEL[8]-FEL[6])/(2*eps) - (FEL[6]-FEL[4])/(2*eps))/(2*eps)
        hess[0,1] = ((FEL[3]-FEL[1])/(2*eps) - (FEL[11]-FEL[9])/(2*eps))/(2*eps)
        hess[1,0] = ((FEL[3]-FEL[11])/(2*eps) - (FEL[1]-FEL[9])/(2*eps))/(2*eps)
        hess[1,1] = ((FEL[0]-FEL[6])/(2*eps) - (FEL[6]-FEL[12])/(2*eps))/(2*eps)
        return hess


    # solves for the input twiss parameters at the beginning of the lattice that, when forward propagated through the lattice, will result in the matched twiss params half through the first undulator quad.
    def back_prop(self):
        zmax = sum(self.quads[:,1]) + sum(self.quads[:,2])

        Ms = self.calc_Ms(zmax, self.quads, dim = 'x')
        M = self.Mprod(Ms)
        Mt = M.transpose()
        Minv = inv(M)
        Mtinv = inv(Mt) # it looks like the transpose and inverse commute here
        input_twiss_x= Minv.dot(self.matched_twiss_x).dot(Mtinv)

        Ms = self.calc_Ms(zmax, self.quads, dim = 'y')
        M = self.Mprod(Ms)
        Mt = M.transpose()
        Minv = inv(M)
        Mtinv = inv(Mt)
        input_twiss_y= Minv.dot(self.matched_twiss_y).dot(Mtinv)
        return input_twiss_x, input_twiss_y

    def calcmatch_FODO_genesis(self, beamEnergyMeV = 3370., undQuadGrad = 10):
        # note that for genesis simulation (where the undulator magnets are 30 cm), undQuadGrad = gradient integral / length = (3 T) / (0.3 m) = 10 T/m
        # for the actual accelerator lattice, the undQuadGrad = gradient integral / length = (3 T) / (0.08 m) = 37.5 T/m

        a = 3.e8/(beamEnergyMeV*1.e6) # electron charge divided by relativistic momentum, SI units m/(V*s)
        k = a*undQuadGrad
        l = 0.3 #quad length
        L = 3.6 #drift length
        MF = np.array([  [np.cos(0.5*np.sqrt(k)*l),1/np.sqrt(k)*np.sin(0.5*np.sqrt(k)*l)],
                            [-np.sqrt(k)*np.sin(0.5*np.sqrt(k)*l),np.cos(0.5*np.sqrt(k)*l)]  ])
        MD = np.array([  [np.cosh(0.5*np.sqrt(k)*l),1/np.sqrt(k)*np.sinh(0.5*np.sqrt(k)*l)],
                            [np.sqrt(k)*np.sinh(0.5*np.sqrt(k)*l),np.cosh(0.5*np.sqrt(k)*l)]  ])

        MO = np.array([[1.,L], [0,1.]])
        #print 'Ms = ', [MF, MO, MD, MD, MO, MF]
        M1 = MF.dot(MO).dot(MD).dot(MD).dot(MO).dot(MF)
        #print 'Mprod = ', M1

        betaMAX = M1[0,1]/np.sqrt(1-(M1[0,0])**2)

        M2 = MD.dot(MO).dot(MF).dot(MF).dot(MO).dot(MD)

        betaMIN = M2[0,1]/np.sqrt(1-(M2[0,0])**2)

        #print 'betaMAX = ', betaMAX
        #print 'betaMIN = ', betaMIN

        twiss_x = np.array([   [betaMIN, 0],
                            [0, 1/betaMIN]   ])
        twiss_y = np.array([   [betaMAX, 0],
                            [0, 1/betaMAX]   ])
        #print 'input twiss = ', twiss_x

        #print 'output twiss = ', M1.dot(twiss_x).dot(M1.transpose())



        return twiss_x, twiss_y

    def calcmatch_FODO_matrixmodel(self, beamEnergyMeV = 3370., undQuadGrad = 37.5):
        # note that for genesis simulation (where the undulator magnets are 30 cm), undQuadGrad = gradient integral / length = (3 T) / (0.3 m) = 10 T/m
        # for the actual accelerator lattice, the undQuadGrad = gradient integral / length = (3 T) / (0.08 m) = 37.5 T/m

        a = 3.e8/(beamEnergyMeV*1.e6) # electron charge divided by relativistic momentum, SI units m/(V*s)
        k = a*undQuadGrad
        l = 0.08 #quad length
        L = 3.79 #drift length
        MF = np.array([  [np.cos(0.5*np.sqrt(k)*l),1/np.sqrt(k)*np.sin(0.5*np.sqrt(k)*l)],
                            [-np.sqrt(k)*np.sin(0.5*np.sqrt(k)*l),np.cos(0.5*np.sqrt(k)*l)]  ])
        MD = np.array([  [np.cosh(0.5*np.sqrt(k)*l),1/np.sqrt(k)*np.sinh(0.5*np.sqrt(k)*l)],
                            [np.sqrt(k)*np.sinh(0.5*np.sqrt(k)*l),np.cosh(0.5*np.sqrt(k)*l)]  ])

        MO = np.array([[1.,L], [0,1.]])
        #print 'Ms = ', [MF, MO, MD, MD, MO, MF]
        M1 = MF.dot(MO).dot(MD).dot(MD).dot(MO).dot(MF)
        #print 'Mprod = ', M1

        betaMAX = M1[0,1]/np.sqrt(1-(M1[0,0])**2)

        M2 = MD.dot(MO).dot(MF).dot(MF).dot(MO).dot(MD)

        betaMIN = M2[0,1]/np.sqrt(1-(M2[0,0])**2)

        #print 'betaMAX = ', betaMAX
        #print 'betaMIN = ', betaMIN

        twiss_x = np.array([   [betaMIN, 0],
                            [0, 1/betaMIN]   ])
        twiss_y = np.array([   [betaMAX, 0],
                            [0, 1/betaMAX]   ])
        #print 'input twiss = ', twiss_x

        #print 'output twiss = ', M1.dot(twiss_x).dot(M1.transpose())



        return twiss_x, twiss_y

    def get_model(self):
        #data = pd.read_csv('fullmachine_quads_rmats_2018-05-13-090800-XLEAP-3880MeV', delimiter='\t')
        #data = pd.read_csv('fullmachine_rmats_2018-05-13-090800-XLEAP-3880MeV', delimiter='\t')
        #data = pd.read_csv('fullmachine_rmats_2018-05-15-142400-XLEAP-3430MeV', delimiter='\t')
        #data = pd.read_csv('/home/physics/kennedy1/genesis_corrplotting/vcurrent/fullmachine_rmats', delimiter='\t')
        try:
            data = pd.read_csv('../../fullmachine_rmats', delimiter='\t')
        except:
            data = pd.read_csv('./matrixmodel/fullmachine_rmats', delimiter='\t')

        elements = []
        keys = data.keys()[0]
        keys = keys.strip()
        keys = re.split(r'\s{2,}', keys)
        for i in range(len(data)):
            elem = data.iloc[i][0]
            elem = elem.strip()
            elem = re.split(r'\s{2,}', elem)
            try: #because some elems do not have a pv name as elem[2], elem[2][:8] may return 'out of bounds'. If so, the elem is not a quad anyway, so just continue to the next elem.
                if (elem[2][:8] == 'QUAD:LTU' or elem[2][:8] == 'QUAD:UND'): #only get the LTU and UND quads
                    elements = elements +[elem]
                else: continue
            except: continue


        myarray = np.array(elements)

        df = pd.DataFrame(myarray, columns=keys)

        for ii in range(len(df)):
            if (df['EPICS_CHANNEL_ACCESS_NAME'][ii] == 'QUAD:LTU1:620' and df['POSITION_INDEX'][ii] == 'BEGIN'):
                cut_start = ii
        df1 = df[cut_start:] #cut the dataframe down to only the quads after and including LTU1:620
        df1 = df1.reset_index(drop=True) #reindex so quad LTU1:620 is at index 0 of dataframe df1

        #df1 now has 3 entries (one each for beginning, middle, end) for every quad between and including quads LTU1:620 - UND1:3380
        #len(df1) should therefore be divisible by 3, and the result len(df1)/3 should be the number of full quads in this section
        num_quads = len(df1)/3
        quads_pos = []
        quads_len = []
        keys2 = []
        for j in range(num_quads):
            i = 3*j #i will now be the index of the beginning of the jth quad, i+2 will be the index of the end of the jth quad
            quad_start_pos = float(df1.iloc[i]['Z_POSITION'])
            quad_end_pos = float(df1.iloc[i+2]['Z_POSITION'])

            quads_pos = quads_pos + [quad_start_pos]
            quads_len = quads_len + [quad_end_pos-quad_start_pos]
            key = df1.iloc[i]['EPICS_CHANNEL_ACCESS_NAME']
            keys2 = keys2 + [key]

        myarray2 = [quads_pos, quads_len]
        myarray2 = np.array(myarray2)
        df2 = pd.DataFrame(myarray2, index=['Z_POSITION','FULL_LENGTH'], columns=keys2)

        ##df2 should now be able to be queried. Example:
        #z_pos = df2['QUAD:LTU1:620'].loc['Z_POSITION']
        #full_length = df2['QUAD:LTU1:620'].loc['FULL_LENGTH']
        #print (     'QUAD:LTU1:620 -- \n' +
        #            'Z_POSITION = ' + str(z_pos) + '\n' +
        #            'FULL_LENGTH = ' + str(full_length)    )


        #process df2 into m_quads and u_quads, formatted for FODO.py~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        m_quads = []; m_quads_names = []
        u_quads = []; u_quads_names = []
        keys = df2.keys()
        for i in range(len(keys)-1):
            length = df2[keys[i]].loc['FULL_LENGTH']
            if(self.disable_epics):
                grad = 0.
                #print 'WARNING: epics is disabled so could not epics.caget(', keys[i], ':BACT)'
            else:
                grad = epics.caget(keys[i]+':BACT')#*0.1/length #########################Check this
            drift = df2[keys[i+1]].loc['Z_POSITION'] - (df2[keys[i]].loc['Z_POSITION'] + length)
            this_quad = [grad, length, drift]
            if keys[i][:8] == 'QUAD:LTU':
                m_quads = m_quads + [this_quad]
                m_quads_names += [keys[i]]
            elif keys[i][:8] == 'QUAD:UND':
                u_quads = u_quads + [this_quad]
                u_quads_names += [keys[i]]
        length = df2[keys[-1]].loc['FULL_LENGTH']
        if(self.disable_epics):
            grad = 0.
            #print 'WARNING: epics is disabled so could not epics.caget(', keys[-1], ':BACT)'
        else:
            grad = epics.caget(keys[-1]+':BACT')#*0.1/length #############################Check this (same as above)
        drift = 0  ######################################## What should this value be???? zero, for now...
        last_uquad = [grad, length, drift]
        u_quads = u_quads + [last_uquad]
        # m_quads and u_quads are now almost in correct format for FODO.py:       [   [grad1, length1, drift1],
        #                                                                      [grad2, length2, drift2],
        #                                                                       etc...                   ] ~~~~~~~~~~

        #print 'm_quads_names = '    , m_quads_names
        #print 'u_quads_names = ', u_quads_names

        #break first uquad into halves and put first half onto the end of mquads (necessary for FODO.py to calc matching twiss, since they are matched in the first half
        test_uquads = np.copy(u_quads[0:2])
        for tt in range(15):
            test_uquads = np.append(test_uquads, u_quads[0:2], axis=0)
        test_uquads[0,0:2] = test_uquads[0,0:2]/2.
        first_halfuquad = np.copy(test_uquads[0:1])
        first_halfuquad[0,2] = 0
        test_mquads = np.copy(m_quads)
        test_mquads = np.append(test_mquads, first_halfuquad, axis=0)

        #return test_mquads, test_uquads
        return test_mquads, test_uquads, m_quads_names, u_quads_names

    def MQ(self, k,l):
        #calculates the transport matrix for a quad with focusing parameter k, and length l (meters)
        if k>0: #focusing
            M = np.array([  [np.cos(np.sqrt(k)*l),1/np.sqrt(k)*np.sin(np.sqrt(k)*l)],
                            [-np.sqrt(k)*np.sin(np.sqrt(k)*l),np.cos(np.sqrt(k)*l)]  ])
        elif k<0: #defocusing
            k = abs(k)
            M = np.array([  [np.cosh(np.sqrt(k)*l),1/np.sqrt(k)*np.sinh(np.sqrt(k)*l)],
                                [np.sqrt(k)*np.sinh(np.sqrt(k)*l),np.cosh(np.sqrt(k)*l)]  ])
        elif k == 0: #drift
            M = np.array([  [1., l],
                            [0, 1.]   ])
        return M

    def MO(self, l):
        #calculates the transport matrix for a drift of length l (meters)
        M = np.array([[1.,l], [0,1.]])
        return M

    def calc_Ms(self, z, Quads, dim):
        s, c, remainder = self.find_segment(z, Quads)
        #s is the segment index in which position z is found,
        #c is a string that denotes whether this segment is a quad or a drift space,
        #remainder is the distance into the final segment that the z position specified extends.

        quads = np.copy(Quads)
        for i in range(len(quads)-1): #these 3 lines convert the pvs (field integrals in kG) to the gradients (in T/m)
            quads[i,0] = quads[i,0]*0.1/quads[i,1]
        quads[-1,0] = quads[-1,0]*0.1/(2.*quads[-1,1]) # the last matching quad is the first half of the first undulator quad. To get the field gradient from the field integral, you need to divide by the length of the quad. The true length of this quad is twice the length that is stored in the array (because the length has been cut in half).

        if dim == 'x':
            sign = 1
        if dim == 'y':  #need to reverse sign of quad focusing parameters (k) if dealing in the y-dimension
            sign = -1

        Ms = [] # storage

        # process all the full elements up to the last one
        for t in range(s):
            Ms = Ms + [ self.MQ(sign*self.a*quads[t,0], quads[t,1]), self.MO(quads[t,2]) ] # self.a is electron charge divided by relativistic momentum

        # process the last element (which may have a partial length)
        if c == 'quad':
            Ms = Ms + [ self.MQ(sign*self.a*quads[s,0], remainder) ]
        if c == 'drift':
            Ms = Ms + [ self.MQ(sign*self.a*quads[s,0], quads[s,1]), self.MO(remainder) ]

        #print 'matching quad and drift space transport matrices for dim = ', dim, ': ', Ms
        return Ms

    def Mprod(self, Ms): #Ms is list of transfer matrices, ORDERED "CHRONOLOGICALLY", each matrix is 2x2 np array
        M = Ms[0]
        for ii in range(1,len(Ms)):
            M = Ms[ii].dot(M)
        return M

    def find_segment(self, z, quads): #returns index s of quad segment that contains the z position. the second value of the tuple is 'quad' if the z value is in the quad at index s, or 'drift' if the z value is in the drift space at index s. remainder is how far into the final quad or drift the z value goes.
        quad_lengths = quads[:,1]
        drifts = quads[:,2]
        for s in range(len(quads)):
            if z <= ( sum(quad_lengths[0:s+1]) + sum(drifts[0:s]) ):
                remainder = z - ( sum(quad_lengths[0:s]) + sum(drifts[0:s]) )
                return s, 'quad', remainder
            elif z <= ( sum(quad_lengths[0:s+1]) + sum(drifts[0:s+1]) ):
                remainder = z - ( sum(quad_lengths[0:s+1]) + sum(drifts[0:s]) )
                return s, 'drift', remainder
            else: continue
        return 'z out of bounds'

    def twiss(self, z, twiss_x0, twiss_y0, quads):
        #quads is nx3 array, quads[n][0] is focusing parameter (k) of nth quad, quads[n][1] is length (l) of nth quad, quads[n][2] is drift length that immediately follows the nth quad.
        Ms = self.calc_Ms(z, quads, dim = 'x')
        #print 'Ms = ', Ms
        M = self.Mprod(Ms)
        #print 'Mprod = ', M
        twiss_x = M.dot(twiss_x0).dot(M.transpose())

        Ms = self.calc_Ms(z, quads, dim = 'y')
        M = self.Mprod(Ms)
        twiss_y = M.dot(twiss_y0).dot(M.transpose())

        return twiss_x, twiss_y

    def configure_serpent(self, s, input_twiss_x = None, input_twiss_y = None, optimize_detuning = True, detuning_zstop = None, plotQ = False):
        if type(input_twiss_x) == type(None):
            input_twiss_x = self.matched_twiss_x
        if type(input_twiss_y) == type(None):
            input_twiss_y = self.matched_twiss_y

        s.input_params['emitx'] = self.emitx
        s.input_params['emity'] = self.emity
        s.input_params['gamma0'] = self.gamma_rel
        s.input_params['curpeak'] = self.beamCurrent
        s.input_params['delgam'] = self.delgam
        if type(self.und_Ks) is not type(None):
            s.und_Ks = self.und_Ks
            s.quad_grads = [-10.*(-1)**i for i in range(len(s.und_Ks)+1)]
            s.subtract_off_K_slope(plotQ=plotQ)
        s.input_params['rxbeam'] = np.sqrt(input_twiss_x[0,0] * self.emitx / self.gamma_rel)
        s.input_params['rybeam'] = np.sqrt(input_twiss_y[0,0] * self.emity / self.gamma_rel)
        s.input_params['alphax'] = input_twiss_x[0,1]
        s.input_params['alphay'] = input_twiss_y[0,1]
        s.input_params['xkx'] = self.xkx ##################################### !!!!!!!!!!!!!!!!!!! should be 0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!###################################################
        s.input_params['xky'] = self.xky ##################################### !!!!!!!!!!!!!!!!!!! should probably be 1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!###################################################
        s.input_params['npart'] = self.npart

        #calculate shotnoise and 3d gainlength/'zrayl':

        #reslambda = lambda_fel #previous reslambda def
        lambda_undulator = s.input_params['xlamd'] #xlamd is 0.03
        # calculate with the resonance condition (ignoring emittance) # should also add angular spread prob.
        reslambda = lambda_undulator / 2. / s.input_params['gamma0']**2 * (1. + 0.5 * self.und_Ks[0]**2)
        s.input_params['xlamds'] = reslambda #new reslambda def

        #resK = resonantK(gamma_beam, lambda_fel, lambda_undulator) #previous resK fcn call
        resK = self.und_Ks[0]
        #rho1 = rhoFEL(gamma_beam, current_beam, 1., resK, lambda_undulator) #previous rhoFEL fcn call
        rho1 = self.rhoFEL(self.gamma_rel, self.beamCurrent, 1., resK, lambda_undulator)
        #rho_all = fastRhoFEL(sizes, rho1) #previous rho_all fcn call
        sigmax, sigmay = s.input_params['rxbeam'], s.input_params['rybeam']
        beam_size = np.sqrt(sigmax*sigmay)
        rho_all0 = self.fastRhoFEL(beam_size, rho1)
        shotnoise = self.shot_noise(rho_all0, reslambda, self.beamCurrent, self.gamma_rel)
        Lg1D = self.gainLength1D(rho_all0, lambda_undulator)
        fillfactor = 3.3/3.9 # fraction of undulator segments with undulator magnets
        Lg1D /= fillfactor # make the gain length longer (no gain where no undulator magnets)
        # Ming Xie perturbative fitting formula
        eta_d = Lg1D/(2*beam_size**2)*reslambda/(2.*np.pi)
        sigma_g = self.delgam #this was the default value we were using in Ming Xie previously. We were never calculating it for specific configs. Should we be? !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        eta_g = (2.*np.pi/lambda_undulator)*2.*(sigma_g/self.gamma_rel)*Lg1D
        emittance = np.sqrt(s.input_params['emitx'] * s.input_params['emity'])
        #eta_e = Lg1D/(beam_size**2/emittance)*(4*np.pi*emittance/reslambda) #old wrongness. see email from Joe (/panos) july24 2018.
        #eta_e = 4.*np.pi*Lg1D/reslambda*(s.input_params['emitx']/s.input_params['gamma0'])*(1./input_twiss_x[0,0]) # IS THIS RIGHT, THO????????
        gamma_geomean_xy = np.sqrt( ((1.+input_twiss_x[0,1]**2.)/input_twiss_x[0,0]) * ((1.+input_twiss_y[0,1]**2.)/input_twiss_y[0,0]) )
        eta_e = 4.*np.pi*Lg1D/reslambda*(s.input_params['emitx']/s.input_params['gamma0'])*gamma_geomean_xy # IS THIS RIGHT, THO????????
        a = np.array( [0.45, 0.57, 0.55, 1.6, 3., 2., 0.35, 2.9, 2.4, 51., 0.95, 3., 5.4, 0.7, 1.9, 1140., 2.2, 2.9, 3.2] )

        LambdaMX = a[0]*(eta_d**a[1]) + a[2]*(eta_e**a[3]) + a[4]*(eta_g**a[5]) + a[6]*(eta_e**a[7])*(eta_g**a[8]) + a[9]*(eta_d**a[10])*(eta_g**a[11]) + a[12]*(eta_d**a[13])*(eta_e**a[14]) + a[15]*(eta_d**a[16])*(eta_e**a[17])*(eta_g**a[18])

        gain_length = Lg1D * (1. + LambdaMX) # should adjust for 3D effects

        convo_factor = np.sqrt(s.input_params['xlamds'] * gain_length / 4. / np.pi)
        rad_waist_squared = 2. * beam_size * convo_factor
        s.input_params['zrayl'] = np.pi * rad_waist_squared / s.input_params['xlamds']
        s.input_params['prad0'] = shotnoise
        print 'shotnoise = ', shotnoise
        print 'gain length = ', gain_length
        #print 'Lg1d = ', Lg1D
        #print 'LambdaMX = ', LambdaMX
        if optimize_detuning:
            if type(detuning_zstop) == type(None):
                print 'detuning_zstop = ', 3*gain_length
                s.optimize_detuning(zstop = 3*gain_length, plotQ=plotQ)
            else:
                print 'detuning_zstop = ', detuning_zstop
                s.optimize_detuning(zstop = detuning_zstop, plotQ=plotQ)



    def shot_noise(self, rho, lambda_fel, beam_current, gamma_beam):
        N = (beam_current*lambda_fel/(3e8*1.602e-19))
        beam_power = gamma_beam*.511e6*beam_current
        return 6*np.sqrt(np.pi)*rho**2*beam_power/(N*np.sqrt(np.log(N/rho)))

    def resonantK(self, gamma_beam, lambda_fel, lambda_undulator = 0.03):
        return np.sqrt(2.* lambda_fel * gamma_beam**2. / lambda_undulator - 1.)

    def resonantlambda(self, gamma_beam, Kund = 2.48, lambda_undulator = 0.03):
        return lambda_undulator * (1. + Kund**2.) / (2.* gamma_beam**2.)


    def rhoFEL(self, gamma_beam, current_beam, sigmax_beam = 1., K_undulator = 2.48, lambda_undulator = 0.03):
        K_undulator_squared = K_undulator ** 2.
        JJarg = 0.5 * K_undulator_squared / (1. + K_undulator_squared)
        JJ = special.j0(JJarg) - special.j1(JJarg)
        return 0.5 / gamma_beam * ((current_beam / 17045.) * (lambda_undulator * K_undulator * JJ / 2. / np.pi / sigmax_beam)**2.) ** 0.3333333333333333

    def fastRhoFEL(self, sigmax_beam, rhoFEL_eval_with_sigmax_beam_equal1):
        # as it says, evaluate rhoFEL once with sigmax_beam=1., then use that as argument of this fcn for fast access
        return rhoFEL_eval_with_sigmax_beam_equal1 * sigmax_beam ** -0.66666666666666667

    def gainLength1D(self, rho, lambda_undulator = 0.03):
        return 0.045944074618482676 * lambda_undulator / rho # prefactor is 1./(4.*np.pi*np.sqrt(3))

    def Pbeam(self, gamma_beam, current_beam):
        return 0.511e6 * gamma_beam * current_beam


    # one argument for parallelization

    def beam_size_undulator_only(self, input_twiss_x, input_twiss_y, uquads):

        # beam size stuff
        zund_start = 0
        quads = np.copy(uquads)

        zend = sum(quads[:-1,1]) + sum(quads[:-1,2]) ###################WHY DOES THE LAST UND QUAD NOT FOCUS PROPERLY?????????????????????????????
        stepsize = 0.1
        zs = np.arange(0, zend, stepsize)
        zs_und = np.arange(zund_start, zend, stepsize)

        beta_xs = []
        beta_ys = []
        for z in zs:
            twiss_x, twiss_y = self.twiss(z, input_twiss_x, input_twiss_y, quads)
            beta_xs += [twiss_x[0,0]]
            beta_ys += [twiss_y[0,0]]

        sigma_xs = np.sqrt(self.emitx_u*np.array(beta_xs))
        sigma_ys = np.sqrt(self.emity_u*np.array(beta_ys))
        sizes = np.sqrt(sigma_xs*sigma_ys)
        avg_size = np.mean( sizes )

        return avg_size, zs, sizes, sigma_xs, sigma_ys, beta_xs, beta_ys

    def beam_size_undulator_only_onearg(self, onearg, uquads):
        input_twiss_x, input_twiss_y = onearg
        # beam size stuff
        zund_start = 0
        quads = np.copy(uquads)

        zend = sum(quads[:-1,1]) + sum(quads[:-1,2]) ###################WHY DOES THE LAST UND QUAD NOT FOCUS PROPERLY?????????????????????????????
        stepsize = 0.1
        zs = np.arange(0, zend, stepsize)
        zs_und = np.arange(zund_start, zend, stepsize)

        beta_xs = []
        beta_ys = []
        for z in zs:
            twiss_x, twiss_y = self.twiss(z, input_twiss_x, input_twiss_y, quads)
            beta_xs += [twiss_x[0,0]]
            beta_ys += [twiss_y[0,0]]

        sigma_xs = np.sqrt(self.emitx_u*np.array(beta_xs))
        sigma_ys = np.sqrt(self.emity_u*np.array(beta_ys))
        sizes = np.sqrt(sigma_xs*sigma_ys)
        avg_size = np.mean( sizes )

        return avg_size, zs, sizes, sigma_xs, sigma_ys, beta_xs, beta_ys

    def beam_size_undulator_only_onearg_fast(self, onearg, uquads):
        input_twiss_x, input_twiss_y = onearg
        # beam size stuff

        quads = np.copy(uquads)

        zend = sum(quads[:-1,1]) + sum(quads[:-1,2]) ###################WHY DOES THE LAST UND QUAD NOT FOCUS PROPERLY?????????????????????????????
        zs = [0]
        for quad in quads:
            zs += [zs[-1]+quad[1]]
            zs += [zs[-1]+quad[2]]

        beta_xs = []
        beta_ys = []
        for z in zs:
            twiss_x, twiss_y = self.twiss(z, input_twiss_x, input_twiss_y, quads)
            beta_xs += [twiss_x[0,0]]
            beta_ys += [twiss_y[0,0]]

        sigma_xs = np.sqrt(self.emitx_u*np.array(beta_xs))
        sigma_ys = np.sqrt(self.emity_u*np.array(beta_ys))
        sizes = np.sqrt(sigma_xs*sigma_ys)
        avg_size = np.mean( sizes )

        return avg_size, zs, sizes, sigma_xs, sigma_ys, beta_xs, beta_ys

    def beam_size_faster_FEL_onearg(self, onearg_tuple, current_beam = 1., sigma_g = 2.6, zstop = 1e6, lambda_fel = None, lambda_undulator = 0.03):

        input_twiss_x, input_twiss_y, mquads, uquads, gamma_beam = onearg_tuple

        # beam size stuff

        if type(mquads) == type(None):
            zund_start = 0
            quads = np.copy(uquads)
        else:
            quads = np.append(mquads,uquads, axis=0)
            zund_start = sum(mquads[:,1]) + sum(mquads[:,2])
        zend = sum(quads[:,1]) + sum(quads[:,2])
        stepsize = 0.1
        zs = np.arange(0, zend, stepsize)
        nstepmax = np.sum(zs < zstop)
        zs_und = np.arange(zund_start, zend, stepsize)

        beta_xs = []
        beta_ys = []
        for z in zs:
            twiss_x, twiss_y = twiss(z, input_twiss_x, input_twiss_y, quads)
            beta_xs += [twiss_x[0,0]]
            beta_ys += [twiss_y[0,0]]

        sigma_xs = np.sqrt(emitx_u*np.array(beta_xs))
        sigma_ys = np.sqrt(emity_u*np.array(beta_ys))
        sizes = np.sqrt(sigma_xs*sigma_ys)
        sizes_all = np.copy(sizes) # everywhere

        sizes = sizes[(len(zs)-len(zs_und)):] # only in und
        avg_size = np.mean( sizes )
        zs_relund = zs_und - zund_start # origin at und start
        sizes = sizes[ zs_relund < zstop ] #keep only sizes within the zstop condition

        # Ming Xie stuff
        if type(lambda_fel) == type(None):
            resK = 2.48
            reslambda = resonantlambda(gamma_beam, resK, lambda_undulator)
        else:
            resK = resonantK(gamma_beam, lambda_fel, lambda_undulator)
            reslambda = lambda_fel
        # end if/else
        rho1 = rhoFEL(gamma_beam, current_beam, 1., resK, lambda_undulator)
        rho_all = fastRhoFEL(sizes, rho1)
        Lg1D = gainLength1D(rho_all, lambda_undulator)
        fillfactor = 3.3/3.9 # fraction of undulator segments with undulator magnets
        fillfactor *= 30./32. # fraction of undulators in the undulator line
        Lg1D /= fillfactor # make the gain length longer (no gain where no undulator magnets)

        # Ming Xie perturbative fitting formula
        eta_d = Lg1D/(2*sizes**2)*reslambda/(2*np.pi)
        eta_g = (2*np.pi/lambda_undulator)*2*(sigma_g/gamma_beam)*Lg1D
        eta_e = Lg1D/(sizes**2/emittance)*(4*np.pi*emittance/reslambda)
        a = np.array( [0.45, 0.57, 0.55, 1.6, 3., 2., 0.35, 2.9, 2.4, 51., 0.95, 3., 5.4, 0.7, 1.9, 1140., 2.2, 2.9, 3.2] )

        LambdaMX = a[0]*(eta_d**a[1]) + a[2]*(eta_e**a[3]) + a[4]*(eta_g**a[5]) + a[6]*(eta_e**a[7])*(eta_g**a[8]) + a[9]*(eta_d**a[10])*(eta_g**a[11]) + a[12]*(eta_d**a[13])*(eta_e**a[14]) + a[15]*(eta_d**a[16])*(eta_e**a[17])*(eta_g**a[18])

        Lg = Lg1D * (1. + LambdaMX) # should adjust for 3D effects
        #P_shotnoise = 1000. # should change this to real estimate
        P_shotnoise = shot_noise(rho_all[0], reslambda, current_beam, gamma_beam)
        P_shotnoise = 2e-4
        #print 'shot noise = ' , P_shotnoise
        P_beam = Pbeam(gamma_beam, current_beam)
        #print ' P_beam = ', P_beam

        #cut = zs_und - zund_start < 25.
        #print 'np.sum(cut) = ', np.sum(cut)
        #print 'len(Lg) = ', len(Lg)
        #Lg = Lg[cut]
        #print 'len(Lg) = ', len(Lg)

        # increment power and find max power
        expargmax = np.log(rho_all * P_beam / P_shotnoise)
        exparglist = stepsize / Lg; exparg = exparglist[0] # 0th order approx to integral of the argument (could linearly interpolate for next order)

        ## find first saturation point
        #exparg = exparglist.cumsum()
        #upto1stsat = exparg <= expargmax
        #after1stsat = exparg > expargmax

        ## figure out saturation power here
        #lastexpargmax = expargmax[upto1stsat][-1] # for max power at that point
        #lastdexparg = exparglist[upto1stsat][-1] # for increment at that ponit
        #exparg = np.min([exparg[upto1stsat][-1] + lastdexparg, lastexpargmax]) # increment or cap at maxpower if exceeded

        # now iteratively increment power
        #exparglist = exparglist[after1stsat]
        #expargmax = expargmax[after1stsat]
        #print 'len(Lg) = ', len(Lg)
        #print 'len(exparglist) = ', len(exparglist)
        #print 'len(zs) = ', len(zs)
        exparg_cum = []
        zsat = len(exparglist)*stepsize
        satQ = False
        for i in range(len(exparglist)):
            if exparg > expargmax[i]: #add modulo condition for vacant undulators here??
                if satQ == False: # only call the first time we saturate
                    #print 'saturated: i = ', i, '\t z = ', i*stepsize, '\t P = ', P_shotnoise * np.exp(exparg)
                    zsat = i*stepsize
                    satQ = True
                pass # saturated
                #break
            else:
                exparg = np.min([exparg + exparglist[i], expargmax[i]]) # saturate
            exparg_cum += [exparg] # increment if not saturated
        #print 'len(exparg_cum) = ', len(exparg_cum)
        power_est = P_shotnoise * np.exp(exparg)
        #logPower_alongz = np.log(P_shotnoise * np.exp(exparg_cum))
        logPower_alongz = (P_shotnoise * np.exp(exparg_cum))

        #print 'len(logPower_alongz) = ', len(logPower_alongz)

        #print 'fel params: ', np.mean(rho_all), np.mean(Lg), power_est, resK, reslambda
        print 'FEL evaluated'
        return avg_size, sizes_all, sigma_xs, sigma_ys, zs, zs_und, power_est, Lg, logPower_alongz, zsat
        #return avg_size, sizes, sigma_xs, sigma_ys, zs, zs_und, np.mean(Lg), Lg
        #return avg_size, sizes, sigma_xs, sigma_ys, zs, zs_und, np.mean(rho_all), Lg

    def fit_beamsize_corrplot(self, corrplot, quadval_grid):
        xdata = []
        ydata = []
        for row in quadval_grid:
            for elem in row:
                xdata += [elem] #reformat the quadvals from the grid shape to a 1 dimensional list of ordered pairs of quad vals [quadx, quady]
        for row in corrplot:
            for elem in row:
                ydata += [elem] #reformat the beamsize data from the grid shape to a 1 dimensional list
        ydata = ydata - np.min(ydata) #shift all y values down, so that the MIN y value is zero.
        ydata = -1.*ydata #invert ydata so that the results are concave down. the MAX y-value should now be zero.
        ydata = np.exp(ydata) #exponentiate the ydata. the ydata should now resemble a (correlated) gaussian with amplitude=1.
        guess_amp = np.max(ydata)
        print 'xdata = ', xdata
        print 'ydata = ', ydata
        print np.argmax(ydata)
        guess_mux = xdata[np.argmax(ydata)][0]
        guess_muy = xdata[np.argmax(ydata)][1]
        guess_Sigma11 = 1.
        guess_Sigma12 = 0.
        guess_Sigma22 = 1.
        fit_amp, fit_mux, fit_muy, fit_Sigma11, fit_Sigma12, fit_Sigma22 = fit(self.fitfcn, xdata=xdata, ydata=ydata, p0 = [guess_amp, guess_mux, guess_muy, guess_Sigma11, guess_Sigma12, guess_Sigma22] )[0] #fit a correlated gaussian to the corrplot data
        fit_Sigma = np.array([ [fit_Sigma11, fit_Sigma12],
                                [fit_Sigma12, fit_Sigma22]  ])
        return fit_amp, fit_mux, fit_muy, fit_Sigma11, fit_Sigma12, fit_Sigma22

    #def plot_fitfcn(self, quadval_grid, fit_amp, fit_mux, fit_muy, fit_Sigma11, fit_Sigma12, fit_Sigma22):
        #plotvals = np.zeros(np.shape(quadval_grid))
        #plotvals = np.zeros(len(quadval_grid), len(quadval_grid[0]))
        #for i, row in enumerate(quadval_grid):
            #for j, elem in enumerate(row):
                #plotvals[i,j] = fitfcn(elem, fit_amp, fit_mux, fit_muy, fit_Sigma11, fit_Sigma12, fit_Sigma22)
        ##assign plot range to axes
        #top = rangey[1]
        #bottom = rangey[0]
        #left = rangex[0]
        #right = rangex[1]
        #extent = [left, right, bottom, top] #left, right, bottom, top
        #plt.imshow(corrplot, cmap='hot', interpolation='nearest', extent=extent)
        #plt.title('Gaussian Beam Size FIT')
        #plt.xlabel('QUAD:LTU1:620 (kG)')
        #plt.ylabel('QUAD:LTU1:640 (kG)')
        #cbar = plt.colorbar()
        #cbar.set_label('Meters')
        #plt.show()
        #plt.close()

    def fitfcn(self, x, amp, mu_x, mu_y, Sigma11, Sigma12, Sigma22):
        x = np.array(x)
        mu = np.array([mu_x, mu_y])
        Sigma = np.array([  [Sigma11, Sigma12],
                            [Sigma12, Sigma22]      ])
        return amp * np.exp(-0.5*( (x-mu).dot(Sigma).dot(x-mu) ) )
