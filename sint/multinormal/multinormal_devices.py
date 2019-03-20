from mint.opt_objects import Device


class MultinormalDevice(Device):
    def __init__(self, eid=None):
        super(MultinormalDevice, self).__init__(eid=eid)
        self.value_percent = 25.0
        self.range_percent = 2.0
        self.default_limits = [-5, 5]
        self.low_limit = -5
        self.high_limit = 5

    def get_delta(self):
        """
        Calculate and return the travel range for this device.

        :return: (float) Travel Range
        """
        ll, hl = self.get_limits()
        val = self.get_value()

        # Method 1: % of Range
        m1 = (hl-ll)*self.range_percent/100.0

        # Method 2: % of Current Value
        m2 = val*self.value_percent/100.0

        # Method 3: Mean(M1, M2)
        m3 = (m1+m2)/2.0

        if m1 != 0.0 and m2 != 0.0:
            return m3
        if m1 == 0:
            return m2
        else:
            return m1

    def get_limits(self):
        return self.low_limit, self.high_limit

    def set_low_limit(self, val):
        if val >= self.high_limit-0.0001:
            return
        if val >= self.default_limits[0]:
            self.low_limit = val
        else:
            self.low_limit = self.default_limits[0]

    def set_high_limit(self, val):
        if val <= self.low_limit+0.0001:
            return
        if val <= self.default_limits[1]:
            self.high_limit = val
        else:
            self.high_limit = self.default_limits[1]
