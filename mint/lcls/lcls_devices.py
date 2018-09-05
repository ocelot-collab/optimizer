import time

from ..opt_objects import Device


class LCLSQuad(Device):
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