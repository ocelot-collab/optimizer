# -*- coding: iso-8859-1 -*-
import time
import numpy as np
from ..opt_objects import Device



class SPEARDevice(Device):
    def __init__(self, eid=None, mi=None):
        super(SPEARDevice, self).__init__(eid=eid)
        self.mi = mi
        self.value_percent = 25.0
        self.range_percent = 2.0

    def get_delta(self):
        """
        Calculate and return the travel range for this device.

        :return: (float) Travel Range
        """
        ll, hl = self.get_limits()
        val = self.get_value()

        # Method 1: % of Range
        m1 = np.abs((hl-ll)*self.range_percent/100.0)

        # Method 2: % of Current Value
        m2 = np.abs(val*self.value_percent/100.0)

        # Method 3: Mean(M1, M2)
        m3 = (m1+m2)/2.0

        if m1 != 0.0 and m2 != 0.0:
            return m3
        if m1 == 0:
            return m2
        else:
            return m1


class SPEARMCORDevice(SPEARDevice):
    def __init__(self, eid=None, mi=None):
        super(SPEARMCORDevice, self).__init__(eid=eid, mi=mi)
        #self._can_edit_limits = True
        if eid.endswith(':Curr1') or eid.endswith(':CurrSetpt'):
            prefix = eid[:eid.rfind(':')+1]
        else:
            prefix = eid+':'
        self.pv_set = '{}{}'.format(prefix, 'CurrSetpt')
        self.pv_read = '{}{}'.format(prefix, 'Curr1')


    def get_delta(self):
        """
        Calculate and return the travel range for this device.

        :return: (float) Travel Range
        """
        return 30 

    def set_value(self, val):
        self.target = val
        self.mi.set_value(self.pv_set, [val])
    
    def get_value(self, save=False):
        if self.mi.read_only:
            val = self.target
            if val is None:
                val = self.mi.get_value(self.pv_read)
        else:
            val = self.mi.get_value(self.pv_read)
        if save:
            self.values.append(val)
            self.times.append(time.time())

        return val 

