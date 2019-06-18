# -*- coding: iso-8859-1 -*-
"""
Python data logger script. Dumps a dictionary of data to the current folder.

Based on a copy of lcls-srv0:/usr/local/lcls/tools/python/toolbox/matlog.py (dated Feb 16  2017)
but made for local saves
"""
from scipy.io import savemat, loadmat
from datetime import datetime as dt
import time
import os
import errno

# Fix Python 2.x.
try:
    UNICODE_EXISTS = bool(type(unicode))
except NameError:
    unicode = str


# scipy.savemat doesn't play well with unicode in Windows so let's make it ascii
def byteify(input):
    if isinstance(input, dict):
        return {byteify(key): byteify(value)
                for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input


def removeUnicodeKeys(input_dict):
    return dict([(byteify(e[0]), e[1]) for e in input_dict.items()])


def getPath():
    """
    Gets matlab data directory from the current date

    Returns (str): Directory string
    """

    # choose directory
    # try: # if running under a profile, save to profile directory
    # username = os.environ['PHYSICS_USER']
    # basepath = '/home/physics/' + username + '/OcelotLogs/'
    # except: # otherwise, save to current directory
    # basepath = os.environ['PWD']

    ## make directory if it doesn't exist
    # try:
    # os.makedirs(basepath)
    # except:
    # pass

    try:
        homepath = os.environ['HOME']
        basepath = homepath + '/ocelot/simscans/'

        year = str(dt.fromtimestamp(time.time()).strftime('%Y'))
        month = str(dt.fromtimestamp(time.time()).strftime('%m'))
        day = str(dt.fromtimestamp(time.time()).strftime('%d'))
        basepath = str(basepath + year + '/' + year + '-' + month + '/' + year + '-' + month + '-' + day + '/')

        os.makedirs(basepath)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            print("GetPath resulted into an exception: ", exc)
            # basepath = os.environ['PWD'] + '/' # doesn't work on windows
            # basepath = os.getcwd() + '/'
            basepath = os.path.curdir + '/'

    return basepath

    ## from original matlog.py
    # base_dir = "/u1/lcls/matlab/data/"
    # year = str(dt.fromtimestamp(time.time()).strftime('%Y'))
    # month = str(dt.fromtimestamp(time.time()).strftime('%m'))
    # day = str(dt.fromtimestamp(time.time()).strftime('%d'))
    # out = str(base_dir+year+'/'+year+'-'+month+'/'+year+'-'+month+'-'+day+'/')
    # return out


def getFileTs():
    """
    Makes a timestamp formated like other matlab gui data files

    Returns (str): Time string
    """

    return str(dt.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H%M%S'))


# function to save the data
# Arguments:
#
# data:    Dictionary of data to save
# guiname: String for the files name, usually formated as FILENAME-PV
# path:    Defaults to the current physics data folder, Overwrite for another directory
# All other keyword arguments are forwarded to scipy.io.savemat.
def save(guiname, data, path='default', **kws):
    """
    Save input python dict as a .mat file to a directory:

    Args:
            guiname: String for the files name, usually formated as FILENAME-PV
            data:    Dictionary of data to save
            path:    Defaults to the current physics data folder, Overwrite 'default' with directory string for another directory

    Returns (str): Directory string of the saved .mat file
    """

    # make file name
    name = guiname + '-' + getFileTs()

    # get path to stick file
    if path == 'default':
        path = getPath()
        # test path to see if it exists
        if not os.path.exists(path):
            print("Trying to create new path")
            os.makedirs(path)
    else:
        path = path
        name = guiname

    print("Simlog Write Path: ", path)
    # save path with filename
    fout = os.path.join(path, name)

    # add ins to data
    ts = time.time()
    data['ts'] = ts
    data['ts_str'] = str(dt.fromtimestamp(ts).strftime('%Y-%m-%d---%H-%M-%S'))

    # nested dict to data structure is the same as other matlab GUIs
    # data = {'data':data}
    data = {'data': removeUnicodeKeys(data)}  # need to remove unicode keys since scipy.savemat doesn't like them

    #try:
    print("Will call savemat with: ", fout, data, kws)
    savemat(fout, data, **kws)
    print('INFO: simlog - Saved scan data to ' + fout + '.mat')
    #except Exception as ex:
    #    print('ERROR: simlog - Looks like there was an error saving ' + fout + '.mat\nException was: \n', ex)
    return fout


def load(path):
    return loadmat(path)


if __name__ == "__main__":
    """ Will print out the days matlab data directory to command line if this file is executed """
    path = getPath()
    print(path)
