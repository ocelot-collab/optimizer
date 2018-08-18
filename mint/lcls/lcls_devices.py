from ..opt_objects import Device


class LCLSQuad(Device):
    def __init__(self, eid=None):
        super(LCLSQuad, self).__init__(eid=eid)
        prefix = eid[:eid.rfind(':')+1]
        self.pv_set = eid
        self.pv_read = "{}{}".format(prefix, "BACT")
        self.pv_low = "{}{}".format(prefix, "BCTRL.DRVL")
        self.pv_high = "{}{}".format(prefix, "BCTRL.DRVH")

    def get_value(self):
        val = self.mi.get_value(self.pv_read)
        print("Get Value for: {} is {}.".format(self.pv_read, val))
        return val

    def get_limits(self):
        low = self.mi.get_value(self.pv_low)
        high = self.mi.get_value(self.pv_high)
        print("Get Value for: {} is {} + {}.".format(self.pv_read, low, high))
        return [low, high]
