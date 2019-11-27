"""
Main Ocelot optimization file
Contains the setup for using the scipy.optimize package run simplex and other algorothms
Modified for use at LCLS from Ilya's version

The file was modified and were introduced new Objects and methods.
S. Tomin, 2017

"""
from __future__ import print_function, absolute_import
import time
import scipy
import numpy as np
from mint.opt_objects import *

from threading import Thread


class Minimizer(object):
    def __init__(self):
        self.mi = None
        self.max_iter = 100
        self.maximize = False    # if True - maximize objective function
        self.devices = None      # list of devices
        self.target = None       # Target class
        self.opt_ctrl = None     # OptControl class
        self.x_init = None       # array of initial actuators values

    def minimize(self, error_func, x):
        pass

    def normalize(self, x):
        """
        Normalize actuator values.
        The method is used when the function "minimize" cannot be controlled
        but transformation of actuator values is needed, e.g. the standard simplex method in Scipy.

        :param x: array of actuator values
        :return: x_normalize
        """
        return x

    def unnormalize(self, xnorm, norm_coef, scaling_coef):
        """
        Un normalize parameters back to physical values.
        The method is used when the function "minimize" cannot be controlled
        but transformation of actuator values is needed, e.g. the standard simplex method in Scipy.

        :param xnorm:
        :param norm_coef: scaling coefficient
        :param scaling_coef: scaling coefficient
        :return:
        """
        return xnorm

    def preprocess(self):
        """
        The method is called before starting optimization

        :return:
        """
        pass


class MachineStatus:
    def __init__(self):
        self.alarm_device = None
        self.alarm_value = None
        self.alarm_min = -1
        self.alarm_max = 1

    def is_ok(self):
        if self.alarm_device is None:
            return True
        self.alarm_value = self.alarm_device.get_value()
        print('alarm value %f' % self.alarm_value)
        print("ALARM: ", self.alarm_value, self.alarm_min, self.alarm_max)
        if self.alarm_min <= self.alarm_value <= self.alarm_max:
                return True
        return False
        

class OptControl:
    """
    Optimization control

    :param m_status: MachineStatus (Device class), indicator of the machine state (beam on/off)
    :param timeot: 0.1, timeout between machine status (m_status) readings
    :param alarm_timeout: timeout between Machine status is again OK and optimization continuation

    """
    def __init__(self):
        self.penalty = []
        self.dev_sets = []
        self.devices = []
        self.nsteps = 0
        self.m_status = MachineStatus()
        self.pause = False
        self.kill = False
        self.is_ok = True
        self.timeout = 0.1
        self.alarm_timeout = 0.

    def wait(self):
        """
        check if the machine is OK. If it is not the infinite loop is launched with checking of the machine state

        :return:
        """
        print(self.m_status.alarm_device)
        if self.m_status.is_ok():
            return 1
        else:
            while 1:
                if self.m_status.is_ok():
                    self.is_ok = True
                    time.sleep(self.alarm_timeout)
                    return 1
                self.is_ok = False
                if self.kill==True:
                    return 1
                time.sleep(self.timeout)
                print(".",)

    def stop(self):
        self.kill = True

    def start(self):
        self.kill = False

    def back_nsteps(self, n):
        if n <= self.nsteps:
            n = -1 - n
        else:
            print("OptControl: back_nsteps n > nsteps. return last step")
            n = -1
        return self.dev_sets[-n]

    def save_step(self, pen, x):
        self.penalty.append(pen)
        self.dev_sets.append(x)
        self.nsteps = len(self.penalty)

    def best_step(self):
        #if len(self.penalty)== 0:
        #    print("No ")
        print("BEST ", self.penalty)
        x = self.dev_sets[np.argmin(self.penalty)]
        return x

    def clean(self):
        self.penalty = []
        self.dev_sets = []
        self.nsteps = 0


class Optimizer(Thread):
    def __init__(self):
        super(Optimizer, self).__init__()
        self.debug = False
        self.minimizer = Minimizer()
        self.logging = False
        # self.kill = False #intructed by tmc to terminate thread of this class
        self.log_file = "log.txt"
        self.devices = []
        self.target = None
        self.timeout = 0
        self.opt_ctrl = OptControl()
        self.seq = []
        self.set_best_solution = True
        #self.normalization = False
        self.norm_coef = 0.05
        self.maximization = True
        self.scaling_coef = 1.0

    def eval(self, seq=None, logging=False, log_file=None):
        """
        Run the sequence of tuning events
        """
        if seq is not None:
            self.seq = seq
        for s in self.seq:
            s.apply()
            s.finalize()

    def exceed_limits(self, x):
        for i in range(len(x)):
            if self.devices[i].check_limits(x[i]):
                return True
        return False

    def get_values(self):
        print(time.time())
        for i in range(len(self.devices)):
            print('reading ', self.devices[i].id)
            self.devices[i].get_value(save=True)

    def set_values(self, x):
        for i in range(len(self.devices)):
            #print('setting', self.devices[i].id, '->', x[i])
            self.devices[i].set_value(x[i])

    def set_triggers(self):
        for i in range(len(self.devices)):
            #print('triggering ', self.devices[i].id)
            self.devices[i].trigger()

    def do_wait(self):
        for i in range(len(self.devices)):
            print('waiting ', self.devices[i].id)
            self.devices[i].wait()

    def error_func(self, x):

        x = self.minimizer.unnormalize(x, self.norm_coef, self.scaling_coef)

        if self.opt_ctrl.kill:
            #self.minimizer.kill = self.opt_ctrl.kill
            print('Killed from external process')
            # NEW CODE - to kill if run from outside thread
            return self.target.pen_max

        self.opt_ctrl.wait()

        # check limits
        if self.exceed_limits(x):
            return self.target.pen_max
        # set values
        self.set_values(x)
        self.set_triggers()
        self.do_wait()
        self.opt_ctrl.wait()
        self.get_values()
        if self.opt_ctrl.m_status.alarm_device!=None:
            while 1:
                self.opt_ctrl.m_status.alarm_value=self.opt_ctrl.m_status.alarm_device.get_value()
                if self.opt_ctrl.m_status.alarm_min<=self.opt_ctrl.m_status.alarm_value<=self.opt_ctrl.m_status.alarm_max:
                    break
                if self.opt_ctrl.kill==True:
                    break
                print('alarm ...')
                time.sleep(0.5)
        
        print('sleeping ' + str(self.timeout))
        time.sleep(self.timeout)

        coef = -1
        if self.maximization:
            coef = 1

        pen = coef*self.target.get_penalty()
        print('penalty:', pen)
        if self.debug:
            print('penalty:', pen)

        self.opt_ctrl.save_step(pen, x)
        return pen

    def max_target_func(self, target, devices, params={}):
        """
        Direct target function optimization with simplex/GP, using Devices as a multiknob
        """
        [dev.clean() for dev in devices]
        target.clean()
        self.target = target
        #print(self.target)
        self.devices = devices
        # testing
        self.minimizer.devices = devices
        self.minimizer.maximize = self.maximization
        self.minimizer.target = target
        self.minimizer.opt_ctrl = self.opt_ctrl

        self.target.devices = self.devices
        dev_ids = [dev.eid for dev in self.devices]
        if self.debug: print('starting multiknob optimization, devices = ', dev_ids)

        target_ref = self.target.get_penalty()

        x = [dev.get_value(save=True) for dev in self.devices]
        x_init = x

        self.minimizer.x_init = x_init

        self.minimizer.preprocess()

        x = self.minimizer.normalize(x)
        res = self.minimizer.minimize(self.error_func, x)
        print("result", res)

        # set best solution
        if self.set_best_solution:
            print("SET the best solution", x)
            x = self.opt_ctrl.best_step()
            if self.exceed_limits(x):
                return self.target.pen_max
            self.set_values(x)

        target_new = self.target.get_penalty()

        print ('step ended changing sase from/to', target_ref, target_new)


    def run(self):
        self.opt_ctrl.start()
        self.eval(self.seq)
        print("FINISHED")
        #self.opt_ctrl.stop()
        return 0


class Action:
    def __init__(self, func, args=None, id=None):
        self.func = func
        self.args = args

        self.id = id

    def apply(self):
        print ('applying...', self.id)
        self.func(*self.args)

    def finalize(self):
        """
        the method is called after method self.apply() is completed.

        :return:
        """
        pass

