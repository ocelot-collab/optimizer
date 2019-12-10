"""
Objective Function

S.Tomin, 2017
"""

from mint.opt_objects import Target
import numpy as np
import time


class XFELTarget(Target):
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
    def __init__(self, mi=None, eid="orbit"):
        super(XFELTarget, self).__init__(eid=eid)
        self.mi = mi

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
        bpms = [
        "XFEL.DIAG/BPM/BPME.2252.SA2/X.ALL",
        "XFEL.DIAG/BPM/BPME.2258.SA2/X.ALL",
        "XFEL.DIAG/BPM/BPME.2264.SA2/X.ALL",
        
        ]

        orbit1 = self.read_bpms(bpms=bpms, nreadings=7)
        
        orbit2 = np.zeros(len(bpms))  # just [0, 0, 0, ... ] 
        

        target = np.sqrt(np.sum((orbit2 - orbit1)**2))
        return target