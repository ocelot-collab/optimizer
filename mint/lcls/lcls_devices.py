import time
import numpy as np
from ..opt_objects import Device


class LCLSDevice(Device):
    def __init__(self, eid=None):
        super(LCLSDevice, self).__init__(eid=eid)
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


class LCLSQuad(LCLSDevice):
    def __init__(self, eid=None):
        super(LCLSQuad, self).__init__(eid=eid)
        self._can_edit_limits = False
        if eid.endswith(':BACT') or eid.endswith(":BCTRL"):
            prefix = eid[:eid.rfind(':')+1]
        else:
            prefix = eid+":"
        self.pv_set = "{}{}".format(prefix, "BCTRL")
        self.pv_read = "{}{}".format(prefix, "BACT")
        self.pv_low = "{}{}".format(prefix, "BCTRL.DRVL")
        self.pv_high = "{}{}".format(prefix, "BCTRL.DRVH")

    def set_value(self, val):
        self.target = val
        self.mi.set_value(self.eid, val)

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

    def get_limits(self):
        self.low_limit = self.mi.get_value(self.pv_low)
        self.high_limit = self.mi.get_value(self.pv_high)
        return [self.low_limit, self.high_limit]

    def set_low_limit(self, val):
        # We will not allow limits to be changed from the interface
        return

    def set_high_limit(self, val):
        # We will not allow limits to be changed from the interface
        return
