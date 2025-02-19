"""
Main Ocelot optimization file
Contains the setup for using the scipy.optimize package run simplex and other algorothms
Modified for use at LCLS from Ilya's version

The file was modified and new Objects and methods were introduced.
S. Tomin, 2017

"""
from __future__ import print_function, absolute_import
import time
import scipy
import numpy as np
from mint.opt_objects import *

from threading import Thread
import logging

logger = logging.getLogger(__name__)

class Minimizer(object):
    def __init__(self):
        self.mi = None
        self.max_iter = 100
        self.maximize = False    # if True - maximize objective function
        self.devices = None      # list of devices
        self.target = None       # Target class
        self.opt_ctrl = None     # OptControl class
        self.x_init = None       # array of initial actuators values
        self.norm_coef = 0.05    # normalization coefficient, step_size = delta * norm_coef (relative step, see GUI)
        self.scaling_coef = 1    # scaling coefficient, step_size *= scaling_coef (Scaling Coefficient, see GUI)

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

    def unnormalize(self, xnorm):
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
        self.alarm_min = -1
        self.alarm_max = 1

    def is_ok(self):
        if self.alarm_device is None:
            return True
        alarm_value = self.alarm_device.get_value()
        if self.alarm_min <= alarm_value <= self.alarm_max:
            return True
        logger.info(" ALARM: Machine is DOWN. alarm value: " + str(alarm_value) + ". min/max = " + str(self.alarm_min) + "/" + str(self.alarm_max))
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
        self.target = None

    def wait(self):
        """
        check if the machine is OK. If it is not the infinite loop is launched with checking of the machine state

        :return:
        """
        if self.m_status.is_ok():
            return 1
        else:
            while 1:
                if self.kill:
                    return 1
                if self.m_status.is_ok():
                    self.is_ok = True
                    time.sleep(self.alarm_timeout)
                    return 1
                time.sleep(self.timeout)
                self.is_ok = False
                print(".",)

    def stop(self):
        self.kill = True
        if self.target is not None:
            self.target.interval = 0.0

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
        x = self.dev_sets[np.argmin(self.penalty)]
        return x

    def clean(self):
        self.penalty = []
        self.dev_sets = []
        self.nsteps = 0


class MetaDevice(object):
    """
    The class is an intermediary between the Devices and the Optimizer.
    The class allows to work with real devices (e.g. Quads) through virtual devices, e.g. beta functions.

    NOTE: The MetaDevice is only to be used via scripted optimization and it is not valid for UI optimizations.
          In a scripted optimization, the child class can have methods to map abstract values into machine parameters.
    """

    def __init__(self):
        self.devices = []

    def preprocess(self):
        """
        Method is called once in  Optmizer.max_target_func()
        :return:
        """
        pass

    def get_values(self):
        """
        Method gets values from devices. In Child class of MetaDevice it can be redefined.
        Example: converting physical parameters (PhysDevices) e.g. Quads strength, Undulator gaps, etc. into
        abstract parameters "x" (Virtual Devices which Optimizer class controls) e.g. beta mismatch, tapering coefficients etc.

        :return: values
        """
        x = []
        for i in range(len(self.devices)):
            val = self.devices[i].get_value(save=True)
            logger.debug('reading: {} --> {}'.format(self.devices[i].id, val))
            x.append(val)
        return x

    def set_values(self, x):
        """
        Method sets values to devices. In Child class of MetaDevice it can be redefined.
        Example: converting abstract parameters "x" e.g. beta mismatch, tapering coefficients etc. into physical params
        e.g. Quads strength, Undulator gaps, etc.

        :param x:
        :return:
        """
        for i in range(len(self.devices)):
            logger.debug('set: {} <-- {}'.format(self.devices[i].id, x[i]))
            self.devices[i].set_value(x[i])

    def set_triggers(self):
        for i in range(len(self.devices)):
            self.devices[i].trigger()

    def do_wait(self):
        for i in range(len(self.devices)):
            self.devices[i].wait()

    def set(self, x):
        """
        Method sets new values to the Devices. The method is used in the Optimizer.

        :param x: list of values
        :return:
        """
        self.set_values(x)
        self.set_triggers()
        self.do_wait()

    def get(self):
        """
        Method gets values from devices. The method is used in the Optimizer

        :return: list of values
        """
        x = self.get_values()
        return x

    def exceed_limits(self, x):
        """
        The method checks if x is out of range of any device. The method is used in the Optimizer

        :param x: list of values
        :return: False if x is in the ranges, True if x is out
        """
        for i in range(len(x)):
            if self.devices[i].check_limits(x[i]):
                return True
        return False

    def clean(self):
        """
        method cleans devices. The method is used in Optimizer

        :return:
        """
        for dev in self.devices:
            dev.clean()


class Optimizer(Thread):
    def __init__(self):
        super(Optimizer, self).__init__()
        self.minimizer = Minimizer()
        # self.kill = False #intructed by tmc to terminate thread of this class
        self.devices = []
        self.target = None
        self.timeout = 0
        self.opt_ctrl = OptControl()
        self.meta_dev = MetaDevice()
        self.seq = []
        self.set_best_solution = True
        self.maximization = True

    def eval(self, seq=None, logging=False, log_file=None):
        """
        Run the sequence of tuning events
        """
        if seq is not None:
            self.seq = seq
        for i, s in enumerate(self.seq):
            logger.info(" Optimizer: action #{} is started".format(i))
            s.apply(func=self.max_target_func)
            s.finalize()

    def error_func(self, x):

        x = self.minimizer.unnormalize(x)

        if self.opt_ctrl.kill:
            #self.minimizer.kill = self.opt_ctrl.kill
            logger.info('Killed from external process')
            # NEW CODE - to kill if run from outside thread
            return self.target.pen_max

        self.opt_ctrl.wait()

        # check limits
        if self.meta_dev.exceed_limits(x):
            return self.target.pen_max

        self.meta_dev.set(x)
        self.meta_dev.get()

        logger.info('sleeping ' + str(self.timeout))
        time.sleep(self.timeout)

        coef = -1
        if self.maximization:
            coef = 1

        pen = coef*self.target.get_penalty()
        logger.debug('penalty: ' + str(pen))

        self.opt_ctrl.save_step(pen, x)
        return pen

    def max_target_func(self, target, devices, params={}):
        """
        Direct target function optimization with simplex/GP, using Devices as a multiknob
        """
        self.meta_dev.devices = devices
        self.meta_dev.preprocess()

        self.meta_dev.clean()
        target.clean()
        self.target = target
        self.devices = devices

        # devices are needed for calculating initial step size
        self.minimizer.devices = devices
        self.minimizer.maximize = self.maximization

        # target in minizer is needed for calculating normscales
        self.minimizer.target = target
        self.opt_ctrl.target = target
        self.minimizer.opt_ctrl = self.opt_ctrl

        # devices are needed for "devmode"
        self.target.devices = self.devices

        target_ref = self.target.get_penalty()

        x = self.meta_dev.get()
        x_init = x

        self.minimizer.x_init = x_init

        self.minimizer.preprocess()

        x = self.minimizer.normalize(x)
        res = self.minimizer.minimize(self.error_func, x)

        # set best solution
        if self.set_best_solution:
            logger.info("SET the best solution: " + str(x))
            x = self.opt_ctrl.best_step()
            if self.meta_dev.exceed_limits(x):
                return self.target.pen_max
            self.meta_dev.set(x)

        target_new = self.target.get_penalty()

        logger.info('step ended changing target from/to = {}/{}.'.format(target_ref, target_new))

    def run(self):
        self.opt_ctrl.start()
        self.eval(self.seq)
        logger.info("FINISHED")
        #self.opt_ctrl.stop()
        return 0


class Action:
    def __init__(self, func=None, args=None, id=None):
        self.func = func
        self.args = args

        self.id = id

    def apply(self, func=None):
        logger.info('Action applying. action id: ' + str(self.id))
        if func is None:
            self.func(*self.args)
        else:
            func(*self.args)

    def finalize(self):
        """
        the method is called after method self.apply() is completed.

        :return:
        """
        pass
