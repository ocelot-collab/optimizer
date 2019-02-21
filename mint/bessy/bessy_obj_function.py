import time
import numpy as np

from mint.opt_objects import Target
import stats.stats as stats


class BESSYTarget(Target):
    def __init__(self, mi=None, eid='GDET:FEE1:241:ENRCHSTBR'):
        """
        :param mi: Machine interface
        :param eid: ID
        """
        super(BESSYTarget, self).__init__(eid=eid)

        self.mi = mi
        self.kill = False
        self.objective_acquisition = None
        self.objective_mean = None
        self.objective_stdev = None

        self.objective_acquisitions = []  # all the points
        self.objective_means = []
        self.std_dev = []
        self.charge = []
        self.current = []
        self.losses = []
        self.points = None

    def get_penalty(self):
        sase = self.get_value()
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
        self.values.append(sase)  # statistic
        self.objective_acquisitions.append(self.objective_acquisition)  # array of points
        self.objective_means.append(self.objective_mean)
        self.std_dev.append(std)
        self.alarms.append(alarm)
        self.charge.append(charge)
        self.current.append(current)
        self.losses.append(losses)
        self.niter += 1
        return pen

    def get_value(self):
        """
        Returns data for the ojective function (sase) from the selected detector PV.

        At lcls the repetition is  120Hz and the readout buf size is 2800.
        The last 120 entries correspond to pulse energies over past 1 second.

        Returns:
                Float of SASE or other detecor measurement
        """
        datain = self.mi.get_value(self.eid)
        if self.points is None:
            self.points = 120
        print("Get Value of : ", self.points, " points.")

        if self.stats is None:
            self.stats = stats.StatNone

        try:
            data = datain[-int(self.points):]
            self.objective_acquisition = data
            self.objective_mean = np.mean(self.objective_acquisition)
            self.objective_stdev = np.std(self.objective_acquisition)
            self.statistic = self.stats.compute(data)
        except:  # if average fails use the scalar input
            print("Detector is not a waveform PV, using scalar value")
            self.objective_acquisition = datain
            self.objective_mean = datain
            self.objective_stdev = -1
            self.statistic = datain

       
        return datain

    def clean(self):
        Target.clean(self)
        self.objective_acquisitions = []  # all the points
        self.objective_means = []
        self.std_dev = []
        self.charge = []
        self.current = []
        self.losses = []
