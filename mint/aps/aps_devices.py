import time
import numpy as np
from ..opt_objects import Device
import re


class APSDevice(Device):
    def __init__(self, eid=None, mi=None):
        super(APSDevice, self).__init__(eid=eid)
        self.mi = mi
        self.value_percent = 25.0
        self.range_percent = 2.0
        self.tol=0.01
        self.mi.bounds=0.2 #maximum variable step size

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

    def update_limits_from_pv(self):
        ll = self.mi.get_value(self.pv_low)
        hl = self.mi.get_value(self.pv_high)
        self.default_limits = [ll, hl]
        self.low_limit = self.default_limits[0]
        self.high_limit = self.default_limits[1]
        print("Limits for {} are: {}".format(self.eid, self.default_limits))

    def get_limits(self):
        return self.low_limit, self.high_limit

    def set_low_limit(self, val):
        if not hasattr(self, 'pv_low'):
            super(APSDevice, self).set_low_limit(val)
            return

        self.update_limits_from_pv()
        if val >= self.high_limit-0.0001:
            return
        if val >= self.default_limits[0]:
            self.low_limit = val
        else:
            self.low_limit = self.default_limits[0]

    def set_high_limit(self, val):
        if not hasattr(self, 'pv_high'):
            super(APSDevice, self).set_high_limit(val)
            return

        self.update_limits_from_pv()
        if val <= self.low_limit+0.0001:
            return
        if val <= self.default_limits[1]:
            self.high_limit = val
        else:
            self.high_limit = self.default_limits[1]


class APSQuad(APSDevice):
    def __init__(self, eid=None, mi=None):
        super(APSQuad, self).__init__(eid=eid, mi=mi)
        self._can_edit_limits = True
        if eid.endswith('CurrentAO') or eid.endswith(":BCTRL"):
            prefix = eid[:eid.rfind(':')+1]
        else:
            prefix = eid+":"
        self.timeout=30
        self.mi = mi
        self.tol=0.01
        self.mi.bounds = 0.1 #maximim variable step size
        self.stepSizeLimit=0.1
        #self.pv_set = "{}{}".format(prefix, "CurrentAO")
        self.pv_set = self.eid
        #L3:SM:SC1:VL:PS:setCurrentAO L3:SM:SC1:VL:PS:setCurrentAO read back are special (different from others)
        if re.search("L3:SM:SC1", eid):
            self.pv_read = eid.replace("setCurrentAO","measCurrAI")
        elif re.search("setCurrentAO", eid):
            self.pv_read = eid.replace("setCurrentAO","measCurrentAI")
        else:
            self.pv_read = eid.replace(":CurrentAO",":CurrentAI")
        self.pv_low = eid + ".DRVL"
        self.pv_high = eid + ".DRVH"
        #self.pv_read = "{}{}".format(prefix, "CurrentAI")
        #self.pv_low = "{}{}".format(prefix, "CurrentAO.DRVL")
        #self.pv_high = "{}{}".format(prefix, "CurrentAO.DRVH")
        print(self.pv_read)
        print("Let's get the limits....")
        self.update_limits_from_pv()

    def set_value(self, val):
        corrector_pvs = np.array(['L1:SC3:HZ:CurrentAO','L1:SC3:VL:CurrentAO','L1:SC4:HZ:CurrentAO','L1:SC4:VL:CurrentAO'])
        corrector_check = np.any(self.eid == corrector_pvs)
        if corrector_check:
            print('WARNING: disabling caput for device', self.eid)
        else:
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

       
