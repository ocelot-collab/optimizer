# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def normscales(mi, pvs, default_length_scale=1., verboseQ=False):
    """
    Method to load in the hyperparameters, or length scales, from a .npy file.
    Sorts data, ordering parameters with this objects pv list.
    Formats data into tuple format that the GP model object can accept.
    ( [device_1, ..., device_N ], coefficent, noise)
    Args:
            mi: Machine interface (calls mi.get_enery and mi.name)
            pvs: list of pv names
            default_length_scale: value to return for devices not in the database
    Returns:
            List of length scales for each device in order of pvs
    """
    print("******************** NORMSCALES - NORMSCALES")
    # stuff for multinormal simulation interface
    if mi.name == 'MultinormalInterface':
        return mi.sigmas
    if mi.name != 'LCLSMachineInterface':
        return None  # np.ones(len(pvs))
    
    #Load in a npy file containing hyperparameters binned for every 1 GeV of beam energy
    #get current L3 beam energy
    
    energy = mi.get_energy()

    vals = [mi.get_value(dev_id) for dev_id in pvs]

    key = str(int(round(energy)))
    filename = 'parameters/hype3.npy'
    if verboseQ: print("Loading raw data for ",key," GeV from", filename)
    f = np.load(str(filename)); filedata0 = f[0][key]; names0 = filedata0.keys()
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
    lens = [] # PV length scales

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
            lens += [pvwidth]
            if verboseQ: print("calculated length scale from scrapes:", pv, pvwidth)

        # data is not in the scrapes so check if in the
        elif pv in names0:
            if verboseQ: print('WARNING: Using length scale from ', filename)
            ave = float(filedata[pv][0])
            std = float(filedata[pv][1])
            pvwidth = std / 2.0 + 0.01
            lens += [pvwidth]
            print("calculated length scale from operator list:", pv, ave, std)

        # default to estimate from limits
        else:
            try:
                if verboseQ: print('WARNING: for now, default length scale is calculated in some weird legacy way. Should calculate from range and starting value.')
                ave = float(vals[i])
                std = np.sqrt(abs(ave))
                pvwidth = std / 2.0 + 0.01
                lens.append(pvwidth)
                if verboseQ:
                    print("calculated hyper param from Mitch's function:", pv, ave, std, hyp)
                    print('calculated from values: ', float(vals[i]))
            except:
                if verboseQ: print('WARNING: Defaulting to ',default_length_scale,' for now... (should estimate from starting value and limits in the future)')
                lens += [default_length_scale]

    return lens

