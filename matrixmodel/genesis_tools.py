# -*- coding: iso-8859-1 -*-

""" 
 Joe Duris / jduris@slac.stanford.edu / 2018-07-31
 
 forbidden_fruit - Genesis 1.3 v2 interface for Python
 Grants (dubious?) knowledge of the FEL.
 Manages Genesis simulations
 
 serpent - controller of the forbidden_fruit
 Named after the manipulating serpent in the book of Genesis.
 Manages forbidden_fruit to execute and clean Genesis sims.
 Also allows optimizing the detuning for time independent sims.
 
 TODO: parallelize serpent routines
 TODO: calculate gradients and hessians
 TODO: read in/out particle distributions and radiation files
 
"""

import os, errno, random, string, subprocess, copy
import numpy as np

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def randomword(length):
   letters = string.ascii_letters + string.digits
   return ''.join(random.choice(letters) for i in range(length))

class serpent():
    """ This class allows us to control Genesis runs."""
    
    def __init__(self):
        # list of sims
        self.forbidden_fruits = []
        
        #self.genesis_bin = 'genesis' # genesis command
        #self.genesis_bin = '~/jduris/bin/genesis' # genesis command works from OPIs
        #self.genesis_bin = '~jduris/bin/genesis_single' # genesis command works from AFS
        self.genesis_bin = 'genesis_single' # genesis command works from AFS
        
        self.sim_path_base = '/dev/shm'
        self.init_dir = os.getcwd() # save initial directory to return to once cleaned
        
        # input lattice
        # quads are gradients in Tesla/meter (use a negative gradient to defocus)
        #self.quad_half_length_xlamd_units = 5 # multiply by 2*xlamd for full quad length
        self.quad_grads = [12.84,-12.64,12.84,-12.64,12.84,-12.64,12.84,-12.64,12.84,-12.64,12.84,-12.64] # 6 FODO
        # Ks are the list of peak undulator strengths (NOT RMS) since epics gives us peak
        self.und_Ks = [np.sqrt(2.) * K for K in [2.473180,2.473180,2.473180,2.473180,2.473180,2.473180,2.473180,2.473180,2.473180,2.473180,2.473180,2.473180]]
        
        # input params for the sims
        # param descriptions here http://genesis.web.psi.ch/download/documentation/genesis_manual.pdf
        ff = forbidden_fruit()
        self.input_params = copy.deepcopy(ff.input_params); del ff
        #self.input_params = None # if we want to use defaults..
        
    def subtract_off_K_slope(self,flat_up_to_und=13,plotQ=True):
        
        # backup Ks
        origKs = copy.deepcopy(self.und_Ks)
        
        # fit a line to the first flat_up_to_und undulators
        fitKs = origKs[origKs != 0] #remove the zeros before linear fit
        
        xs = np.array(range(flat_up_to_und))
        ys = np.array([fitKs[i] for i in xs])
        
        xm = xs.mean() # average position
        ym = ys.mean() # average K
        
        m = np.sum((xs-xm) * (ys-ym)) / np.sum((xs-xm) * (xs-xm)) # slope
        b = ym - m * xm # intercept
        
        # subtract off linear slope
        for i in range(len(self.und_Ks)):
            if (self.und_Ks[i] != 0): 
                self.und_Ks[i] -= m * i # subtract off slope only for the nonzero undulators (thus keeping the zero undulators at zero)
        #print 'Flattened Ks = ', self.und_Ks
        print 'Linear component of undulator taper has been flattened.'
        if plotQ:
            from matplotlib import pyplot as plt
            plt.plot(origKs, label='orig')
            plt.hold('on')
            plt.plot(self.und_Ks, label='flattened')
            plt.legend()
            plt.xlabel('z / 3.9 m')
            plt.ylabel('K_peak')
            plt.savefig('taper.png')
            plt.hold('off')
            plt.close()
        
    # stub - fcn to calculate the twiss given a lattice
    def matched_twiss(self):
        pass
    
    def input_twiss(self):
        
        betax = self.input_params['rxbeam']**2 * self.input_params['gamma0'] / self.input_params['emitx']
        betay = self.input_params['rybeam']**2 * self.input_params['gamma0'] / self.input_params['emity']
        alphax = self.input_params['alphax']
        alphay = self.input_params['alphay']
        
        return {'betax':betax, 'betay':betay, 'alphax':alphax, 'alphay':alphay} 
    
    def run_genesis_for_twiss(self, betax=None, betay=None, alphax=None, alphay=None, shotnoise_power=None, gain_length=None, plotQ=True, plotName=None, hostname=None):
        # note: if hostname is not None, forbidden_fruit uses ssh to run remotely on that hostname
        
        ff = forbidden_fruit(self.sim_path_base)
        ff.genesis_bin = self.genesis_bin
        ff.hostname = hostname
        
        ff.input_params = copy.deepcopy(self.input_params)
        ff.quad_grads = copy.deepcopy(self.quad_grads) # quads are gradients in Tesla/meter (use a negative gradient to defocus)
        ff.und_Ks = copy.deepcopy(self.und_Ks) # Ks are the list of peak undulator strengths (NOT RMS) since epics gives us peak
        
        #########################################################################
        ## WARNING: this should be changed:
        #ff.input_params['zstop'] = 30.6
        #########################################################################
                
        gamma0 = ff.input_params['gamma0'] # beam energy
        emitx = ff.input_params['emitx'] # normalized emit
        emity = ff.input_params['emity']
        
        # transverse size stuff
        
        if type(betax) is not type(None):
            ff.input_params['rxbeam'] = np.sqrt(betax * emitx / gamma0)
        else:
            betax = ff.input_params['rxbeam'] ** 2. * gamma0 / emitx
        
        if type(betay) is not type(None):
            ff.input_params['rybeam'] = np.sqrt(betay * emity / gamma0)
        else:
            betay = ff.input_params['rybeam'] ** 2. * gamma0 / emity
        
        if type(alphax) is not type(None):
            ff.input_params['alphax'] = alphax
        else:
            alphax = ff.input_params['alphax']
            
        if type(alphay) is not type(None):
            ff.input_params['alphay'] = alphay
        else:
            alphay = ff.input_params['alphay']
        
        # shot noise stuff - note that the factor of 2 below is from: waist = np.sqrt(2) * sigma 
        
        beam_size = np.sqrt(ff.input_params['rxbeam'] * ff.input_params['rybeam'])
        
        if type(gain_length) is not type(None): # convolve the beam size with the gain (longer gain spreads => more diffraction => larger mode)
            convo_factor = np.sqrt(ff.input_params['xlamds'] * gain_length / 4. / np.pi)
            rad_waist_squared = 2. * beam_size * convo_factor
        else:
            rad_waist_squared = 2. * beam_size * beam_size # if we don't know the gain length, then assume radiation size = beam size
        
        ff.input_params['zrayl'] = np.pi * rad_waist_squared / ff.input_params['xlamds'] 
        
        if type(shotnoise_power) is not type(None):
            ff.input_params['prad0'] = shotnoise_power
        
        # run genesis with these parameters
        
        ffout = ff.run_genesis_and_read_output() # return tuple of lists (zs, powers)
        
        rxbeam_vs_z = ff.read_output(7) # (zs, x_sizes)
        rybeam_vs_z = ff.read_output(8) # (zs, y_sizes)

        radphase_vs_z = ff.read_output(3) # (zs, radiation_phase)
        bunching_vs_z = ff.read_output(6) # (zs, bunching_factors)
        bunphase_vs_z = ff.read_output(-1) # (zs, bunch_phases)

        # column order:    power         increment     p_mid         phi_mid       r_size        energy        bunching      xrms          yrms          error         <x>           <y>           e-spread      far_field    bunch_phase
        
        ffout = (ffout[0], ffout[1], rxbeam_vs_z[1], rybeam_vs_z[1]) # (zs, ps, xrms, yrms)
        
        if plotQ:
        
            # figure out plot name
            if type(plotName) is not type(None):
                plot_path = self.init_dir + '/' + plotName
            else:
                #plot_path = self.init_dir + '/' + ff.sim_id
                plot_path = self.init_dir + '/' + 'twiss_' + str(round(betax,1)) + '_' + str(round(betay,1)) + '_' + str(round(alphax,1)) + '_' + str(round(alphay,1)) 
            
            from matplotlib import pyplot as plt

            plt.plot(ffout[0], np.array(bunching_vs_z[1]))
            plt.xlabel('z (m)')
            plt.ylabel('bunching factor')
            plt.savefig(plot_path + '_Bvsz.png')
            plt.close()

            plt.plot(ffout[0], np.array(radphase_vs_z[1]))
            plt.xlabel('z (m)')
            plt.ylabel('radiation phase')
            plt.savefig(plot_path + '_radphasevsz.png')
            plt.close()

            plt.plot(ffout[0], np.array(bunphase_vs_z[1]))
            plt.xlabel('z (m)')
            plt.ylabel('bunch phase')
            plt.savefig(plot_path + '_bunphasevsz.png')
            plt.close()

            plt.plot(ffout[0], 1.e-9 * np.array(ffout[1]))
            plt.xlabel('z (m)')
            plt.ylabel('power (GW)')
            plt.savefig(plot_path + '_Pvsz.png')
            plt.close()
            
            plt.plot(ffout[0], np.log(ffout[1]))
            plt.xlabel('z (m)')
            plt.ylabel('log power (log(W))')
            plt.savefig(plot_path + '_logPvsz.png')
            plt.close()
            
            plt.plot(rxbeam_vs_z[0], 1.e6 * rxbeam_vs_z[1], label='x')
            plt.hold('on')
            plt.plot(rybeam_vs_z[0], 1.e6 * rybeam_vs_z[1], label='y')
            plt.plot(rybeam_vs_z[0], 1.e6 * np.sqrt(np.array(rxbeam_vs_z[1]) * np.array(rybeam_vs_z[1])), label='sqrt(x*y)')
            plt.legend()
            plt.xlabel('z (m)')
            plt.ylabel('beam size (um)')
            plt.savefig(plot_path + '_sizevsz.png')
            plt.hold('off')
            plt.close()
        
        return ffout
        
        #self.forbidden_fruits += [ff]
        
    # run genesis parallel
    def run_genesis_for_twiss_parallel(self, betaxs=None, betays=None, alphaxs=None, alphays=None, shotnoise_powers=None, gain_lengths=None, plotQ=True, hosts=None):
        
        # require all the lists be the same length
        inputs = [betaxs, betays, alphaxs, alphays, shotnoise_powers, gain_lengths]
        inputlens = []
        for l in inputs:
            try:
                inputlens += [len(l)]
            except:
                inputlens += [0] # None types and numbers don't have length    
        try:
            hostslen = len(hosts)
        except:
            hostslen = 0
            
        maxlen = max(inputlens)
        for ll in inputlens:
            if not (ll == 0 or ll == 1 or ll == maxlen):
                print 'serpent.run_genesis_for_twiss_parallel - ERROR: inputs need to be the same length (1-entry lists)'
                return
        
        # build of args lists
        args = []
        for i in range(maxlen):
            arg = []
            for j,l in enumerate(inputs):
                if inputlens[j] == 0:
                    arg += [l]
                elif inputlens[j] == 1:
                    arg += [l[0]]
                else:
                    arg += [l[i]]
            arg += [plotQ] # plotQ
            arg += [None] # plotName
            #if hostslen:
                #arg += [hosts[i % hostslen]] # loop through the list of hosts
            #else:
                #arg += [hosts]
            args += [arg]
        
        # parallel map
        from parallelstuff import parallelmap2
        #res = parallelmap2(self.run_genesis_for_twiss, args, hosts)
        res = parallelmap2(self.run_genesis_for_twiss, args, None)
        
        return res # list of run_genesis_for_twiss outputs
        
    def acr_hostlist(self):
        hostnames = ['lcls-srv0'+str(i+1) for i in range(1)] # can access the lcls-srv0* cpus
        cpucounts = [2 for i in range(len(hostnames))] # each of them have 8 cores
        
        # perhaps we're running on an opi, so recruit those too
        import subprocess, multiprocessing
        current_hostname = subprocess.check_output(['hostname'])
        current_hostname = current_hostname[:-1] # shave off the new line character
        current_cores = int(multiprocessing.cpu_count()) # how many cores on this host?
        
        if np.sum([current_hostname == hn for hn in hostnames]) == 0: # current_hostname is not in the list of hostnames
            hostnames += [current_hostname]
            cpucounts += [current_cores]
            
        hosts = []
        for i in range(len(hostnames)):
            for j in range(cpucounts[i]):
                hosts += [hostnames[i]]
        
        self.hosts = hosts # save it in case we need to reuse it
        
        return hosts
        
    # when running time independent sims (itdp=0), need to optimize detuning
    def optimize_detuning(self, relative_range = 0.03, nsteps = 21, zstop = 4., plotQ=True):
        print 'OPTIMIZING DETUNING'
        # calculate with the resonance condition (ignoring emittance) # should also add angular spread prob.
        xlamds_guess = self.input_params['xlamd'] / 2. / self.input_params['gamma0']**2 * (1. + 0.5 * self.und_Ks[0]**2)
        
        #xlamds_list = xlamds_guess * (1. + relative_range*np.linspace(-0.5,1.5,nsteps)) # list of xlamds to try
        xlamds_list = xlamds_guess * (1. + relative_range*np.linspace(-1,1,nsteps)) # list of xlamds to try
        maxps = []

        iterator = 0
        for xlamds in xlamds_list:
        
            ff = forbidden_fruit(self.sim_path_base)
            ff.genesis_bin = self.genesis_bin
            
            ff.input_params = copy.deepcopy(self.input_params)
            ff.quad_grads = copy.deepcopy(self.quad_grads) # quads are gradients in Tesla/meter (use a negative gradient to defocus)
            ff.und_Ks = copy.deepcopy(self.und_Ks) # Ks are the list of peak undulator strengths (NOT RMS) since epics gives us peak
            
            ff.input_params['xlamds'] = xlamds
            if type(zstop) is not type(None):
                ff.input_params['zstop'] = zstop # ~1 undulator FODO just for quick detuning optimization
            
            (zs, ps) = ff.run_genesis_and_read_output()
            
            maxps += [ps[-1]]
            iterator +=1
            print 'Detuning scan ', iterator, '/', nsteps,' complete.'

            del ff
            
        # interpolate to find maximum
        
        maxps = np.array(maxps) # convert to numpy array
        
        #print (xlamds_list,maxps) # maybe try to make a plot of the scan result (print to a file; not to a window)
        #from matplotlib import pyplot as plt
        
        from scipy import interpolate
        
        interp = interpolate.interp1d(xlamds_list, maxps, kind='cubic')
        
        xlamds_list_finer = xlamds_guess * (1. + relative_range*np.linspace(-1,1,nsteps*100+1))
        maxps_finer = np.array([interp(x) for x in xlamds_list_finer])
        
        xlamds_best = xlamds_list_finer[maxps_finer == max(maxps_finer)][0]
        p_best = interp(xlamds_best)
        
        xlamds0 = self.input_params['xlamds'] # save old xlamds
        p0 = interp(xlamds0)
        
        self.input_params['xlamds'] = xlamds_best # automatically overwrite xlamds with the optimum
        print 'FINISHED OPTIMIZING DETUNING'
        print 'Guessed resonant wavelength of ', xlamds_guess, ' m. Changed xlamds from ', xlamds0, ' m to ', xlamds_best, ' m'
        
        if plotQ:
            from matplotlib import pyplot as plt
            plt.plot(1.e9 * xlamds_list_finer, maxps_finer)
            plt.hold('on')
            plt.plot(1.e9 * np.array(xlamds0), np.array(p0), '.', label='orig')
            plt.plot(1.e9 * np.array(xlamds_best), np.array(p_best), '.', label='best')
            plt.legend()
            plt.xlabel('xlamds (nm)')
            plt.ylabel('max power (W)')
            plt.savefig(self.init_dir + '/' + 'detuning_scan.png')
            plt.hold('off')
            plt.close()
        
        return xlamds_best
        
    # stub 
    def hessian(self):
        pass
        
    # gradient 
    def gradient(self):
        pass
        
    # needed for parallelization
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance atributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        
        # Remove unpicklable entries (these would need to be recreated
        # in the __setstate__ function. 
        # Example: del state['file'] # since the file handle isn't pickleable
        
        return state
        
    # needed for parallelization
    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)
        
        # Should also manually recreate unpicklable members.
        # Example: file = load(self.filename)

class forbidden_fruit():
    """ This class allows us to write inputs, run genesis, return data, and clean up genesis junk."""
    
    def __del__(self):
        self.clean() # clean the crap before deleting
        
    # needed for parallelization
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance atributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        
        # Remove unpicklable entries (these would need to be recreated
        # in the __setstate__ function. 
        # Example: del state['file'] # since the file handle isn't pickleable
        
        return state
        
    # needed for parallelization
    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)
        
        # Should also manually recreate unpicklable members.
        # Example: file = load(self.filename)
    
    def __init__(self, sim_path_base = '/dev/shm'):
        self.class_name = 'forbidden_fruit'
        
        #self.genesis_bin = 'genesis' # genesis command
        #self.genesis_bin = '~/jduris/bin/genesis' # genesis command works from OPIs
        #self.genesis_bin = '~jduris/bin/genesis_single' # genesis command works from AFS
        self.genesis_bin = 'genesis_single' # genesis command works from AFS and now from OPIs
        self.hostname = None # string to declare hostname to run the job on
        self.niceness = 10 # default niceness to run with
        
        self.init_dir = os.getcwd() # save initial directory to return to once cleaned
        
        # make simulation directory
        self.sim_id = 'genesis_run_' + randomword(10)
        self.sim_path = sim_path_base + '/' + self.sim_id + '/'
        mkdir_p(self.sim_path)
        os.chdir(self.sim_path)
        
        # some file paths (more in self.input_params just below)
        self.sim_input_file = 'genesis.in'
        self.sim_log_file = 'genesis.log'
        
        # input lattice
        # quads are gradients in Tesla/meter (use a negative gradient to defocus)
        self.quad_grads = [12.84,-12.64,12.84,-12.64,12.84,-12.64,12.84,-12.64,12.84,-12.64,12.84,-12.64] # 6 FODO
        # Ks are the list of peak undulator strengths (NOT RMS) since epics gives us peak
        self.und_Ks = [np.sqrt(2.) * K for K in [2.473180,2.473180,2.473180,2.473180,2.473180,2.473180,2.473180,2.473180,2.473180,2.473180,2.473180,2.473180]]
        
        # input params
        # param descriptions here http://genesis.web.psi.ch/download/documentation/genesis_manual.pdf
        self.input_params = {'aw0'   :  2.473180,
                            'xkx'   :  0.000000E+00,
                            'xky'   :  1.000000E+00,
                            'wcoefz':  [7.500000E-01,   0.000000E+00,   1.000000E+00],
                            'xlamd' :  3.000000E-02,
                            'fbess0':  0.000000E+00,
                            'delaw' :  0.000000E+00,
                            'iertyp':    0,
                            'iwityp':    0,
                            'awd'   :  2.473180,
                            'awx'   :  0.000000E+00,
                            'awy'   :  0.000000E+00,
                            'iseed' :   10,
                            'npart' :   2048,
                            'gamma0':  6.586752E+03,
                            'delgam':  2.600000E+00,
                            'rxbeam':  2.846500E-05,
                            'rybeam':  1.233100E-05,
                            'alphax':  0,
                            'alphay': -0,
                            'emitx' :  4.000000E-07,
                            'emity' :  4.000000E-07,
                            'xbeam' :  0.000000E+00,
                            'ybeam' :  0.000000E+00,
                            'pxbeam':  0.000000E+00,
                            'pybeam':  0.000000E+00,
                            'conditx' :  0.000000E+00,
                            'condity' :  0.000000E+00,
                            'bunch' :  0.000000E+00,
                            'bunchphase' :  0.000000E+00,
                            'emod' :  0.000000E+00,
                            'emodphase' :  0.000000E+00,
                            'xlamds':  2.472300E-09,
                            'prad0' :  2.000000E-04,
                            'pradh0':  0.000000E+00,
                            'zrayl' :  3.000000E+01,
                            'zwaist':  0.000000E+00,
                            'ncar'  :  251,
                            'dgrid' :  7.500000E-04,
                            'lbc'   :    0,
                            'rmax0' :  1.100000E+01,
                            'nscr'  :    1,
                            'nscz'  :    0,
                            'nptr'  :   40,
                            'nwig'  :  112,
                            'zsep'  :  1.000000E+00,
                            'delz'  :  1.000000E+00,
                            'nsec'  :    1,
                            'iorb'  :    0,
                            'zstop' :  3.195000E+11, # note: this is huge
                            'magin' :    1,
                            'magout':    0,
                            #'quadf' :  1.667000E+01,
                            #'quadd' : -1.667000E+01,
                            'quadf' :  0,
                            'quadd' :  0,
                            'fl'    :  8.000000E+00,
                            'dl'    :  8.000000E+00,
                            'drl'   :  1.120000E+02,
                            'f1st'  :  0.000000E+00,
                            'qfdx'  :  0.000000E+00,
                            'qfdy'  :  0.000000E+00,
                            'solen' :  0.000000E+00,
                            'sl'    :  0.000000E+00,
                            'ildgam':    9,
                            'ildpsi':    1,
                            'ildx'  :    2,
                            'ildy'  :    3,
                            'ildpx' :    5,
                            'ildpy' :    7,
                            'itgaus':    1,
                            'nbins' :    8,
                            'igamgaus' :    1,
                            'inverfc' :    1,
                            'lout'  : [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
                            'iphsty':    1,
                            'ishsty':    1,
                            'ippart':    0,
                            'ispart':    2,
                            'ipradi':    0,
                            'isradi':    0,
                            'idump' :    0,
                            'iotail':    1,
                            'nharm' :    1,
                            'iallharm' :    1,
                            'iharmsc' :    0,
                            'curpeak':  4.500000E+03,
                            'curlen':  0.000000E+00,
                            'ntail' :    0,
                            'nslice': 4129,
                            'shotnoise':  1.000000E+00,
                            'isntyp':    1,
                            'iall'  :    1,
                            'itdp'  :    0,
                            'ipseed':    1,
                            'iscan' :    0,
                            'nscan' :    3,
                            'svar'  :  1.000000E-02,
                            'isravg':    0,
                            'isrsig':    1,
                            'cuttail': -1.000000E+00,
                            'eloss' :  0.000000E+00,
                            'version':  1.000000E-01,
                            'ndcut' :  150,
                            'idmpfld':    0,
                            'idmppar':    0,
                            'ilog'  :    0,
                            'ffspec':    1,
                            'convharm':    1,
                            'ibfield':  0.000000E+00,
                            'imagl':    0.000000E+00,
                            'idril':    0.000000E+00,
                            'alignradf':    0,
                            'offsetradf':    0,
                            'multconv':    0,
                            'igamref':  0.000000E+00,
                            'rmax0sc':  0.000000E+00,
                            'iscrkup':    0,
                            'trama':    0,
                            'itram11':  1.000000E+00,
                            'itram12':  0.000000E+00,
                            'itram13':  0.000000E+00,
                            'itram14':  0.000000E+00,
                            'itram15':  0.000000E+00,
                            'itram16':  0.000000E+00,
                            'itram21':  0.000000E+00,
                            'itram22':  1.000000E+00,
                            'itram23':  0.000000E+00,
                            'itram24':  0.000000E+00,
                            'itram25':  0.000000E+00,
                            'itram26':  0.000000E+00,
                            'itram31':  0.000000E+00,
                            'itram32':  0.000000E+00,
                            'itram33':  1.000000E+00,
                            'itram34':  0.000000E+00,
                            'itram35':  0.000000E+00,
                            'itram36':  0.000000E+00,
                            'itram41':  0.000000E+00,
                            'itram42':  0.000000E+00,
                            'itram43':  0.000000E+00,
                            'itram44':  1.000000E+00,
                            'itram45':  0.000000E+00,
                            'itram46':  0.000000E+00,
                            'itram51':  0.000000E+00,
                            'itram52':  0.000000E+00,
                            'itram53':  0.000000E+00,
                            'itram54':  0.000000E+00,
                            'itram55':  1.000000E+00,
                            'itram56':  0.000000E+00,
                            'itram61':  0.000000E+00,
                            'itram62':  0.000000E+00,
                            'itram63':  0.000000E+00,
                            'itram64':  0.000000E+00,
                            'itram65':  0.000000E+00,
                            'itram66':  1.000000E+00,
                            'outputfile' : 'genesis.out',
                            'maginfile' : 'genesis.lat',
                            'distfile': None,
                            'filetype':'ORIGINAL'}
    
    def input_twiss(self):
        
        betax = self.input_params['rxbeam']**2 * self.input_params['gamma0'] / self.input_params['emitx']
        betay = self.input_params['rybeam']**2 * self.input_params['gamma0'] / self.input_params['emity']
        alphax = self.input_params['alphax']
        alphay = self.input_params['alphay']
        
        return {'betax':betax, 'betay':betay, 'alphax':alphax, 'alphay':alphay} 
    
    def clean(self):
        cmd = 'rm -rf ' + self.sim_path # command to clean crap
        if type(self.hostname) is not type(None): # to run remotely
            cmd = 'ssh ' + self.hostname + ' ' + cmd
        os.system(cmd)
        os.chdir(self.init_dir)
        
    def write_input(self):
        
        f = open(self.sim_path + self.sim_input_file, "w")
        
        f.write("$newrun\n")
        
        import numbers # so many numbers, so list time
        
        # parse
        for key, value in self.input_params.iteritems():
            #if type(value) == type(1) or type(value) == type(1.): # numbers
            if isinstance(value,numbers.Number): # numbers
                f.write(key + ' = ' + str(value) + '\n')
            elif type(value) == type([]): # lists
                liststr = ''
                for item in value:
                    liststr += str(item) + ' '
                f.write(key + ' = ' + liststr + '\n')
            elif type(value) == type('a'): # strings
                f.write(key + ' = ' + "'" + value + "'" + '\n') # genesis input may need apostrophes
            else:
                #print 'couldn\'t determine data type so skipped: key, value = ', key, value
                pass
        
        f.write("$end\n")
        
        f.close()

    # genesis does drifts in a weird way with AD elements
    # takes undulator rms K (Genesis calls this AW) and returns AD
    def calculate_AD(self, AW):

        Ndrift = 20 # number of drift lengths (in units of xlamds)
        K0 = AW # wiggler rms K
        maxslip = np.floor(Ndrift/(1.+1./K0**2) - np.ceil(Ndrift/(1.+K0**2)))
        
        Kd = np.sqrt(K0**2 - (1.+K0**2)/Ndrift * np.ceil(Ndrift/(1.+K0**2) + maxslip)) # calculate the effective K needed to minimize the R56 without adding phase drift

        return Kd
    
    # write the magnetic lattice file for Genesis 1.3 v2
    def write_lattice(self):
        # quads are gradients in Tesla/meter (use a negative gradient to defocus)
        # Ks are the list of peak undulator strengths (NOT RMS) since epics gives us peak
        
        #self.quad_grads, self.und_Ks
        
        quads = self.quad_grads
        Ks = self.und_Ks / np.sqrt(2.) # change peak to rms
        
        nquad = len(quads)
        nund = len(Ks)
        nund = min(nquad,nund)
        
        f = open(self.sim_path + self.input_params['maginfile'], "w")
        
        f.write("? VERSION = 1.0" + '\n')
        f.write("? UNITLENGTH = " + str(self.input_params['xlamd']) + '\n')
        f.write('\n')
        f.write("QF " + str(quads[0]) + " 5 0" + '\n') # half of first quad
        f.write('\n')

        dynamicAD = True  # calculate the Genesis drift param dynamically?
        zeroKdriftQ = True # replace K=0 undulators with a drift?
        zeroKthresold = 0.01 # just in case there's noise in actual PV signal
        defaultK = 2.48 # we shouldn't hit this but just in case
        
        # parse
        for i in range(nund):
            if dynamicAD:
                lastK = defaultK # just in case
                lastKs = Ks[:i+1] # grab last Ks
                lastKs = lastKs[lastKs > zeroKthresold] # cut zeros
                if len(lastKs):
                    lastK = lastKs[-1]
                else: # looks like no nonzero Ks met yet...
                    lastKs = Ks[i+1:] # grab last Ks
                    lastKs = lastKs[lastKs > zeroKthresold] # cut zeros
                    if len(lastKs):
                        lastK = lastKs[0]
                AD = self.calculate_AD(lastK) # this tries to do a better job
            else:
                AD = 0.29 # static AD is close
                
            if zeroKdriftQ and Ks[i] < zeroKthresold:
                f.write("AD " + str(AD) + " 130 0" + '\n')
            else:
                f.write("AW " + str(Ks[i]) + " 110 20" + '\n')
                f.write("AD " + str(AD) + " 20 110" + '\n')
            
            try:
                f.write("QF " + str(quads[i+1]) + " 10 120" + '\n\n')
            except:
                #if i >= nund-1:  # this will never be true
                print self.class_name + '.write_lattice - WARNING: ran out of quads for lattice...'
                break
        
        f.close()
        
    def run_genesis_and_read_output(self, column=0, waitQ=True):
        
        self.run_genesis()
        
        return self.read_output(column)
        
    def run_genesis(self, waitQ=True):
        
        self.write_input()
        
        self.write_lattice()

        cmd = 'nice -n ' + str(self.niceness) + ' ' + self.genesis_bin + ' ' + self.sim_input_file + ' > genesis.log' # command
        if type(self.hostname) is not type(None): # to run remotely
            cmd = 'ssh ' + self.hostname + ' ' + cmd
        os.system(cmd)

    # based on https://raw.githubusercontent.com/archman/felscripts/master/calpulse/calpulse.py
    def read_output(self, column=0, stat=np.max):
        # column => columns of genesis.out; column 0 is power
        # column order:    power         increment     p_mid         phi_mid       r_size        energy        bunching      xrms          yrms          error         <x>           <y>           e-spread      far_field    bunch_phase

        #filename1 = self.outputfilename # TDP output filename defined by external parameter
        filename1 = self.input_params['outputfile']
        #slicetoshow = int(sys.argv[2]) # slice number to show as picture
        #zrecordtoshow = int(sys.argv[2])# z-record num
        idx = column

        #open files
        f1 = open(self.sim_path + filename1, 'r')

        #extract z, au, QF [BEGIN]
        while not f1.readline().strip().startswith('z[m]'):pass
        zaq   = []
        line  = f1.readline().strip()
        count = 0
        while line:
            zaq.append(line)
            line = f1.readline().strip()
            count += 1
        #print "%d lines have been read!" %count
        #count: total z-record number
        #extraxt z, au, QF [END]


        #find where to extract power ...
        slicenum = 0 # count read slice num
        data=[]
        while True:
            while not f1.readline().strip().startswith('power'):pass
            data.append([])
            slicenum += 1
            line = f1.readline().strip()
            while line: 
        #        data[slicenum-1].append(["%2.6E" %float(x) for x in line])
                data[slicenum-1].append(line)
                line = f1.readline().strip()
        #    print 'Read slice %05d' %slicenum
            if not f1.readline():break

        f1.close()
        #data extraction end, close opened file

        #print sys.getsizeof(zaq)
        #raw_input()

        #cmd1 = "/bin/grep -m1 sepe " + filename1 + " | awk '{print $1}'"
        cmd2 = "/bin/grep xlamd "    + filename1 + " | /bin/grep -v xlamds | awk -F'=' '{print $NF}' | sed 's/[D,d]/e/g'"
        cmd3 = "/bin/grep delz "     + filename1 + " | awk -F'=' '{print $NF}' | sed 's/[D,d]/e/g'"
        cmd4 = "/bin/grep zsep "     + filename1 + " | awk -F'=' '{print $NF}' | sed 's/[D,d]/e/g'"
        cmd5 = "/bin/grep iphsty "     + filename1 + " | awk -F'=' '{print $NF}' | sed 's/[D,d]/e/g'"
        cmd6 = "/bin/grep ishsty "     + filename1 + " | awk -F'=' '{print $NF}' | sed 's/[D,d]/e/g'"
        cmd7 = "/bin/grep xlamds "     + filename1 + " | awk -F'=' '{print $NF}' | sed 's/[D,d]/e/g'"

        try:
            import subprocess
            #dels  = float(subprocess.check_output(cmd1, shell = True))
            xlamd = float(subprocess.check_output(cmd2, shell = True))
            delz  = float(subprocess.check_output(cmd3, shell = True))
            zsep = float(subprocess.check_output(cmd4, shell = True))
            iphsty = float(subprocess.check_output(cmd5, shell = True))
            ishsty = float(subprocess.check_output(cmd6, shell = True))
            xlamds = float(subprocess.check_output(cmd7, shell = True))
        except AttributeError:
            import os
            #dels  = float(os.popen4(cmd1)[1].read())
            xlamd = float(os.popen4(cmd2)[1].read())
            delz  = float(os.popen4(cmd3)[1].read())
            zsep = float(os.popen4(cmd4)[1].read())
            iphsty = float(os.popen4(cmd5)[1].read())
            ishsty = float(os.popen4(cmd6)[1].read())
            xlamds = float(os.popen4(cmd7)[1].read())

        c0 = 299792458.0

        dz = xlamd * delz * iphsty
        ds = xlamds * zsep * ishsty
        dt = ds / c0

        import numpy as np
        x  =  np.arange(count)
        s  =  np.arange(slicenum)
        z  =  np.array([float(zaq[i].split()[0]) for i in x])
        #p1 =  [data[slicetoshow][i].split()[0] for i in x]
        ##ps =  [data[i][zrecordtoshow].split()[0] for i in s]
        ##plot(s,ps,'r-')
        ##plot(z,p1,'r-')

        j=0
        pe=[]
        pmax = 0
        #idx = int(sys.argv[4]) # moved up
        """
        idx = 0  # fundamental power
        idx = 15 # 3rd harmonic power
        idx = 23 # 5th harmonic power
        """
        ssnaps = [] # list of list for a constant s (beam coordinate)
        #while j < count:
        for j in range(count): # loop over s slices (beam coords)
                psi =  [data[i][j].split()[idx] for i in s]
                ssnap = [float(x) for x in psi]
                ssnaps += [ssnap]
                
                #ptmp = max(ssnap) # max seen at this s-coord
                #if ptmp > pmax:
                        #pmax = ptmp
                #pe.append(sum([float(x) for x in psi])*dt)
        #maxpe = max(pe)
        #psmax = [data[i][pe.index(maxpe)].split()[0] for i in s]
        #print "Pulse Energy: ", maxpe*1e9, "nJ @ z= ", pe.index(maxpe)*dz
        #print "Max Power: ",pmax, "W"

        #print 'count = ', count
        #print 'np.shape(ssnaps) = ', np.shape(ssnaps)
        #print 'np.shape(data) = ', np.shape(data)
        #print 'np.shape(x) = ', np.shape(x)
        #print 'np.shape(s) = ', np.shape(s)
        #print 'np.shape(z) = ', np.shape(z)
        #print 'z = ', z

        #print 'np.shape(np.mean(ssnaps,axis=1)) = ', np.shape(np.mean(ssnaps,axis=1))
        #print 'np.mean(ssnaps,axis=1) = ', np.mean(ssnaps,axis=1)
        #print 'stat(ssnaps,axis=1) = ', stat(ssnaps,axis=1)
        
        stat_vs_z = stat(ssnaps,axis=1)
        
        self.output = (z, stat_vs_z)
        
        return self.output
        