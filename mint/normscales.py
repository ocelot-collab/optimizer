# -*- coding: iso-8859-1 -*-
import numpy as np

"""
TODO: note these methods should be brought into each respective machine interface rathyer than living here
"""

#_________________________________________________________________________________________________________________________________________________
def normscales(mi, devices, default_length_scale=1., correlationsQ=False, verboseQ=True):
    """
    fixme
    """
    print("******************** NORMSCALES - NORMSCALES")
    # stuff for multinormal simulation interface
    if mi.name == 'MultinormalInterface':
        return normscales_MultinormalInterface(mi)
    elif mi.name == 'LCLSMachineInterface':
        return normscales_LCLSMachineInterface(mi, devices, default_length_scale, correlationsQ=False, verboseQ=verboseQ)  # np.ones(len(pvs))
    elif mi.name == 'SPEARMachineInterface':
        return normscales_SPEARMachineInterface(mi, devices, default_length_scale, verboseQ=verboseQ)  # np.ones(len(pvs))
    else:
        return np.array([None] * np.size(devices)), None, None, None, None


#_________________________________________________________________________________________________________________________________________________
def normscales_MultinormalInterface(mi):
    # hyperparams for multinormal simulation interface
    
    #noise_param = 2.*np.log((self.mi.bgNoise + self.mi.sigAmp * self.mi.sigNoiseScaleFactor) * (self.mi.noiseScaleFactor+1.e-15) / np.sqrt(self.mi.numSamples))
    noise = (mi.bgNoise + mi.sigAmp * mi.sigNoiseScaleFactor) * (mi.noiseScaleFactor+1.e-15)
    
    length_scales = np.array(mi.sigmas)             # device length scales
    amp_variance = mi.sigAmp                           # signal amplitude
    single_noise_variance = noise                      # noise variance of 1 point
    mean_noise_variance = noise / mi.points   # noise variance of mean of n points
    precision_matrix = mi.invcovarmat               # correlations and length scales are combined in a precision matrix, which is the inverse of the covariance matrix
    
    return length_scales, amp_variance, single_noise_variance, mean_noise_variance, precision_matrix


#_________________________________________________________________________________________________________________________________________________
def normscales_SPEARMachineInterface(mi, devices, default_length_scale=1., verboseQ=True):
    
    import pickle
    
    #print('WARNING: mint.normscales.normscales_SPEARMachineInterface - method not properly implemented yet!')
    
    length_params_file = 'parameters/spear_hyperparams.pkl'
    
    try:
        #with open('test.pkl', 'wb') as f: pickle.dump({'a':1,'b':2},f, 0) # the zero stores as text so that it's sorta editable manually
        with open(length_params_file, 'rb') as f: hyper_dict = pickle.load(f)
        length_scales = np.array([hyper_dict[dev.eid] for dev in devices])
    except:
        if verboseQ: print('WARNING: mint.normscapes - Could not load length scales from file ', length_params_file)
        length_scales = np.ones(len(devices))
        if verboseQ: print('WARNING: mint.normscapes - setting length scales to ', length_scales)
        
    try:
        amp_variance = hyper_dict['amp']
    except:
        amp_variance = 1.
        if verboseQ: print('WARNING: mint.normscapes - setting amplitude scale to ', amp_variance)
        
    try:
        single_noise_variance = hyper_dict['noise']
    except:
        single_noise_variance = 0.01
        if verboseQ: print('WARNING: mint.normscapes - setting single sample noise variance to ', single_noise_variance)
        
    try:
        mean_noise_variance = hyper_dict['noise'] / mi.points
    except:
        mean_noise_variance = single_noise_variance
        if verboseQ: print('WARNING: mint.normscapes - setting average noise variance (variance of the mean) to ', mean_noise_variance)
        
    precision_matrix = None
    
    print( length_scales, amp_variance, single_noise_variance, mean_noise_variance, precision_matrix)
    return length_scales, amp_variance, single_noise_variance, mean_noise_variance, precision_matrix


#_________________________________________________________________________________________________________________________________________________
def normscales_LCLSMachineInterface(mi, devices, default_length_scale=1., correlationsQ=False, verboseQ=True):
    
    import pandas as pd
    
    #____________________________
    # Grab device length scales
    
    #Load in a npy file containing hyperparameters binned for every 1 GeV of beam energy
    #get current L3 beam energy
    
    energy = mi.get_energy()

    pvs = [dev.eid for dev in devices]
    vals = [dev.get_value() for dev in devices]

    key = str(int(round(energy))).encode('utf-8')
    filename = 'parameters/hype3.npy'
    if verboseQ: print("Loading raw data for ",key," GeV from", filename)
    f = np.load(str(filename), encoding='bytes'); filedata0 = f[0][key]; names0 = filedata0.keys()
    if verboseQ: 
        print('\n\n\n\\', f[0])
        print(f[0][key])
        print('\n\n\n\\')
    if verboseQ: print(energy, names0)
    filedata = filedata0

    # scrapes
    prior_params_file = 'parameters/fit_params_2018-01_to_2018-01.pkl'
    prior_params_file_older = 'parameters/fit_params_2017-05_to_2018-01.pkl'
    if verboseQ: 
        print('Building hyper params from data in file ', prior_params_file)
        print('Next, filling in gaps with ', prior_params_file_older)
        print('Next, filling in gaps with ', filename)
        print('Finally, filling in gaps with estimate from starting point and limits')
    filedata_recent = pd.read_pickle(prior_params_file) # recent fits
    filedata_older = pd.read_pickle(prior_params_file_older) # fill in sparsely scanned quads with more data from larger time range
    names_recent = filedata_recent.T.keys()
    names_older = filedata_older.T.keys()
    # pvs = [pv.replace(":","_") for pv in pvs]

    # store results
    length_scales = [] # PV length scales

    # calculate the length scales
    for i, pv in enumerate(pvs):
        # note: we pull data from most recent runs, but to fill in the gaps, we can use data from a larger time window
        #       it seems like the best configs change with time so we prefer recent data

        pv_ = pv.replace(":","_")

        # pv is in the data scrapes
        if pv_ in names_recent or pv_ in names_older:

            # use recent data unless too sparse (less than 10 points)
            if pv_ in names_recent and filedata_recent.get_value(pv_, 'number of points fitted')>10:
                if verboseQ: print('Length scales: ' + pv + ' RECENT DATA LOOKS GOOD')
                filedata = filedata_recent
            # fill in for sparse data with data from a larger time range
            elif pv_ in names_older:
                if verboseQ: print('Length scales: '+pv+' DATA TOO SPARSE <= 10 ################################################')
                filedata = filedata_older
            else:
                if verboseQ: print('Length scales: '+pv+' DATA TOO SPARSE <= 10 ################################################')
                filedata = filedata_recent

            # calculate hyper width
            width_m = filedata.get_value(pv_, 'width slope')
            width_b = filedata.get_value(pv_, 'width intercept')
            # width_res_m = filedata.get_value(pv, 'width residual slope')
            # width_res_b = filedata.get_value(pv, 'width residual intercept')
            pvwidth = (width_m * energy + width_b) # prior widths
            length_scales += [pvwidth]
            if verboseQ: print("calculated length scale from scrapes:", pv, pvwidth)

        # data is not in the scrapes so check if in the
        elif pv in names0:
            if verboseQ: print('WARNING: Using length scale from ', filename)
            ave = float(filedata[pv][0])
            std = float(filedata[pv][1])
            pvwidth = std / 2.0 + 0.01
            length_scales += [pvwidth]
            if verboseQ: print("calculated length scale from operator list:", pv, ave, std)

        # default to estimate from limits
        else:
            try:
                if verboseQ: print('WARNING: for now, default length scale is calculated in some weird legacy way. Should calculate from range and starting value.')
                ave = float(vals[i])
                std = np.sqrt(abs(ave))
                pvwidth = std / 2.0 + 0.01
                length_scales.append(pvwidth)
                if verboseQ:
                    print("calculated hyper param from Mitch's function:", pv, ave, std, hyp)
                    print('calculated from values: ', float(vals[i]))
            except:
                if verboseQ: print('WARNING: Defaulting to ',default_length_scale,' for now... (should estimate from starting value and limits in the future)')
                length_scales += [default_length_scale]
                
    length_scales = np.array(length_scales)
    
    
    #____________________________
    # Grab correlations
        
    # if we'd like to try to use correlations from the LCLS matrix model...  SHOULD BE PUSHED INTO mint.normscales
    if correlationsQ:
        
        try: # import the model and calculate the corrlation matrix
            try:
                from matrixmodel.beamconfig import Beamconfig
                import matrixmodel.correlation_tools as corrtools
            except:
                print('ERROR - mint.mint: could not import Beamconfig from matrixmodel.beamconfig')
                raise
            
            try:
                # read result from file
                hess_mat_filepath = 'matrixmodel/configs/current_config/hessian_scaled.npy'
                hessian_scaled = np.load(hess_mat_filepath)
                if np.shape(hessian_scaled) != (len(length_scales),len(length_scales)):
                    breakshit
            except:
                print('ERROR - mint.mint: There was an error importing a correlation matrix from ',hess_mat_filepath)
            
                try:
                    # matrixmodel calculates beam size averaged over the undulator line (~50 ms per point)
                    bc = Beamconfig(config_type='current')
                    pv_base_names = [pv[:-6] for pv in dev_ids]  # chop off :BCTRL
                    hessian_scaled = bc.curvature_matrix(pv_base_names)[0]

                    # limit correlations to something managable
                    offdiag_function = [corrtools.truncate_offdiag, corrtools.scale_offdiag][-1]  # truncate or scale correlations?
                    hessian_scaled = corrtools.limit_maxeigenlength(hessian_scaled, eigenlengthmax=3,
                                                        offdiag_function=corrtools.offdiag_function)
                except:
                    print('ERROR - mint.mint: There was an error importing a correlation matrix from the matrix model.')
                    raise
        except:
            print('WARNING - mint.mint: Proceeding without correlations.')

        # build covariance matrix from correlation matrix and length scales
        diaglens = np.diag(length_scales)
        invdiaglens = np.linalg.inv(diaglens)
        precision_matrix = np.dot(invdiaglens, np.dot(hessian_scaled, invdiaglens))
        
    else:
        precision_matrix = None
    
    #____________________________
    # Figure out a decent amplitude and noise scale
    try:
        obj_func = mi.target.get_value()[:2] # get the current mean and std of the chosen detector
        if verboseQ: 
            print(('mi.points =', mi.points))
            print(('obj_func = ',obj_func))
        try:
            # SLACTarget.get_value() returns tuple with elements stat, stdev, ...
            ave = obj_func[0]
            std = obj_func[1]
        except:
            if verboseQ: print("Detector is not a waveform, Using scalar for hyperparameter calc")
            ave = obj_func
            # Hard code in the std when obj func is a scalar
            # Not a great way to deal with this, should probably be fixed
            std = 0.1
            
        if verboseQ: print(('INFO: amp = ', ave))

        # print('WARNING: overriding amplitude and variance hyper params')
        # ave = 1.
        # std = 0.1

        # calculating the amplitude parameter
        # start with 3 times what we see currently (stand to gain 
        ave *= 3.
        if verboseQ: print(('INFO: amp = ', ave))
        # next, take larger of this and 2x the most we've seen in the past
        try:
            ave = np.max([ave, 2*np.max(peakFELs)])
            ave = np.max([ave, 1.5*np.max(peakFELs)])
            #ave = np.max(peakFELs)
            if verboseQ: print(('INFO: prior peakFELs = ', peakFELs))
            if verboseQ: print(('INFO: amp = ', ave))
        except:
            ave = 7. # most mJ we've ever seen
            if verboseQ: print(('WARNING: using ', ave, ' mJ (most weve ever seen) for amp'))
        # next as a catch all, make the ave at least as large as 2 mJ
        ave = np.max([ave, 2.])
        if verboseQ: print(('INFO: amp = ', ave))
        # finally, we've never seen more than 6 mJ so keep the amp parameter less than 10 mJ
        ave = np.min([ave, 10.])
        if verboseQ: print(('INFO: amp = ', ave))

        # inflate a bit to account for shot noise near peak?
        std = 1.5 * std
        
        print('WARNNGL: mint.normascales.py - PLEASE FIX ME!!!')
        amp_variance = ave                               # signal amplitude
        single_noise_variance = std**2                      # noise variance of 1 point
        mean_noise_variance = std**2 / mi.points   # noise variance of mean of n points
        
    except:
        if verboseQ: print('WARNING: Could not grab objective since it wasn\'t passed properly to the machine interface as mi.target')
        amp_variance = 1.                                # signal amplitude
        single_noise_variance = 0.1**2                      # noise vsarince of 1 point
        mean_noise_variance = 0.01**2                       # noise variance of mean of n points
        

    return length_scales, amp_variance, single_noise_variance, mean_noise_variance, precision_matrix

