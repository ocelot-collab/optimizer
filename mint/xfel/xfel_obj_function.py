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
    :param dp: Device property
    :param pen_max: 100, maximum penalty
    :param niter: 0, calls number get_penalty()
    :param penalties: [], appending penalty
    :param times: [], appending the time evolution of get_penalty()
    :param nreadings: 1, number of objective function readings
    :param interval: 0 (secunds), interval between readings
    """
    def __init__(self, mi=None, dp=None, eid="x57**2 + y57**2 + x59**2 + y59"):
        super(XFELTarget, self).__init__(eid=eid)

    def get_value(self):
        """
        Method to get signal of target function (e.g. SASE signal).

        :return: value
        """
        x57 = self.mi.get_value("XFEL.DIAG/ORBIT/BPMA.57.I1/X.SA1")
        y57 = self.mi.get_value("XFEL.DIAG/ORBIT/BPMA.57.I1/Y.SA1")
        x59 = self.mi.get_value("XFEL.DIAG/ORBIT/BPMA.59.I1/X.SA1")
        y59 = self.mi.get_value("XFEL.DIAG/ORBIT/BPMA.59.I1/Y.SA1")
        return -np.sqrt(x57 ** 2 + y57 ** 2 + x59 ** 2 + y59 ** 2)
