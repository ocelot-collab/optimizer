"""
Objective Function

S.Tomin, 2017
"""
from __future__ import absolute_import, print_function

from mint.opt_objects import Target
import numpy as np
import time


class PETRATarget(Target):
    """
    Objective function

    :param mi: Machine interface
    :param pen_max: 100, maximum penalty
    :param niter: 0, calls number get_penalty()
    :param penalties: [], appending penalty
    :param times: [], appending the time evolution of get_penalty()
    :param nreadings: 1, number of objective function readings
    :param interval: 0 (secunds), interval between readings
    """
    def __init__(self, mi=None, eid="x57**2 + y57**2 + x59**2 + y59"):
        super(PETRATarget, self).__init__(eid=eid)
        self.mi = mi
        self.debug = False
        self.kill = False
        self.pen_max = 100
        self.clean()
        self.nreadings = 1
        self.interval = 0.0

    def get_alarm(self):
        """
        Method to get alarm level (e.g. BLM value).

        alarm level must be normalized: 0 is min, 1 is max

        :return: alarm level
        """
        return 0

    def read_bpms(self, bpms, nreadings):
        orbits = np.zeros((nreadings, len(bpms)))
        for i in range(nreadings):
            for j, bpm in enumerate(bpms):
                orbits[i, j] = self.mi.get_value(bpm)
            time.sleep(0.1)
        return np.mean(orbits, axis=0)

    def get_value(self):
        """
        Method to get signal of target function (e.g. SASE signal).

        :return: value
        XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.AMPL
        """
        bpms = ["XFEL.DIAG/BPM/BPMA.59.I1/X.ALL",
        "XFEL.DIAG/BPM/BPMA.72.I1/X.ALL",
        "XFEL.DIAG/BPM/BPMA.75.I1/X.ALL",
        "XFEL.DIAG/BPM/BPMA.77.I1/X.ALL",
        "XFEL.DIAG/BPM/BPMA.80.I1/X.ALL",
        "XFEL.DIAG/BPM/BPMA.82.I1/X.ALL",
        "XFEL.DIAG/BPM/BPMA.85.I1/X.ALL",
        "XFEL.DIAG/BPM/BPMA.87.I1/X.ALL",
        "XFEL.DIAG/BPM/BPMA.90.I1/X.ALL",
        "XFEL.DIAG/BPM/BPMA.92.I1/X.ALL",
        "XFEL.DIAG/BPM/BPMF.95.I1/X.ALL",
        "XFEL.DIAG/BPM/BPMC.134.L1/X.ALL", 
        "XFEL.DIAG/BPM/BPMA.117.I1/X.ALL",
        "XFEL.DIAG/BPM/BPMC.158.L1/X.ALL",
        "XFEL.DIAG/BPM/BPMA.179.B1/X.ALL"]
        Vinit = self.mi.get_value("XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.AMPL")

        orbit1 = self.read_bpms(bpms=bpms, nreadings=7)
        
        time.sleep(0.1)
        self.mi.set_value("XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.AMPL", Vinit - 2)
        time.sleep(0.9)
        
        orbit2 = self.read_bpms(bpms=bpms, nreadings=7)
        
        self.mi.set_value("XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.AMPL", Vinit)
        time.sleep(0.9)
        
        target = -np.sqrt(np.sum((orbit2 - orbit1)**2))
        return target
        #return -np.sqrt(a ** 2 + b ** 2 + c**2)

    def get_value_test(self):
        """
        For testing

        :return:
        """
        values = np.array([dev.get_value() for dev in self.devices])
        value = 2*np.sum(np.exp(-np.power((values - np.ones_like(values)), 2) / 5.))
        value = value * (1. + (np.random.rand(1)[0] - 0.5) * 0.001)
        return value 


    def get_penalty(self):
        """
        Method to calculate the penalty on the basis of the value and alarm level.

        penalty = -get_value() + alarm()


        :return: penalty
        """
        sase = 0.
        for i in range(self.nreadings):
            sase += self.get_value()
            time.sleep(self.interval)
        sase = sase/self.nreadings
        print("SASE", sase)
        alarm = self.get_alarm()
        if self.debug: print('alarm:', alarm)
        if self.debug: print('sase:', sase)
        pen = 0.0
        if alarm > 1.0:
            return self.pen_max
        if alarm > 0.7:
            return alarm * self.pen_max / 2.
        pen += alarm
        pen -= sase
        if self.debug: print('penalty:', pen)
        self.niter += 1
        # print("niter = ", self.niter)
        self.penalties.append(pen)
        self.times.append(time.time())
        self.values.append(sase)
        self.alarms.append(alarm)
        return pen

    def get_spectrum(self):
        return [0, 0]

    def get_stat_params(self):
        # spetrum = self.get_spectrum()
        # ave = np.mean(spetrum[(2599 - 5 * 120):-1])
        # std = np.std(spetrum[(2599 - 5 * 120):-1])
        ave = self.get_value()
        std = 0.1
        return ave, std

    def get_energy(self):
        return 3

    def clean(self):
        self.niter = 0
        self.penalties = []
        self.times = []
        self.alarms = []
        self.values = []
