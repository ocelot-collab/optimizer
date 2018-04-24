import time
import numpy as np

from opt_objects import Target
import stats.stats as stats


class SLACTarget(Target):
    def __init__(self, mi=None, eid=None):
        """
        :param mi: Machine interface
        :param eid: ID
        """
        super(SLACTarget, self).__init__(eid=eid)
        self.mi = mi
        self.std_dev = []
        self.charge = []
        self.current = []

    def get_penalty(self):
        sase, std, charge, current = self.get_value()
        alarm = self.get_alarm()
        pen = 0.0
        if alarm > 1.0:
            return self.pen_max
        if alarm > 0.7:
            return alarm * 50.0
        pen += alarm
        pen -= sase
        self.penalties.append(pen)
        self.times.append(time.time())
        self.values.append(sase)
        self.std_dev.append(std)
        self.alarms.append(alarm)
        self.charge.append(charge)
        self.current.append(current)
        self.niter += 1
        return pen

    def get_value(self):
        """
        Returns data for the ojective function (sase) from the selected detector PV.

        At lcls the repetition is  120Hz and the readout buf size is 2800.
        The last 120 entries correspond to pulse energies over past 1 second.

        Returns:
                Float of SASE or other detecor measurment
        """
        TARGET_PV = 'OPT:Waveform'
        points = 120
        datain = self.mi.get_value(TARGET_PV)

        if self.stats is None:
            self.stats = stats.StatNone
        
        try:
            data = datain[-int(points):]
            data_size = points

            statistic = self.stats.compute(data)
            sigma = np.std(data)

        except:
            data = datain
            data_size = 1
            statistic = data
            sigma = -1

        charge, current = 0.0, 1.0  # self.mi.get_charge_current()

        print(self.stats.display_name, ' of ', data_size, ' points is ', statistic, ' and standard deviation is ', sigma)

        return statistic, sigma, charge, current

    def clean(self):
        Target.clean(self)
        self.std_dev = []
        self.charge = []
        self.current = []

target_class = SLACTarget
