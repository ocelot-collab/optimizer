import time
import numpy as np
import os

from mint.opt_objects import Target
import stats.stats as stats


class LINACTarget(Target):
    def __init__(self, mi=None, eid='L3:CM1:measCurrentCM'):
        """
        :param mi: Machine interface
        :param eid: ID
        """
        super(LINACTarget, self).__init__(eid=eid)

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
        self.device_values=[]

    def get_penalty(self):
        sase, std, charge, current, losses = self.get_value()
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
        #(x_new,y_new)=self.mi.getState()
        #print(x_new)
        #print(y_new)
        return pen

    def get_value(self):
        """
        Returns data for the ojective function (sase) from the selected detector PV.

        At aps the repetition is  120Hz and the readout buf size is 2800.
        The last 120 entries correspond to pulse energies over past 1 second.

        Returns:
                Float of SASE or other detecor measurement
        """
        #datain = []
        #for i in range(self.nreadings):
        #    datain.extend(self.mi.get_value(self.eid))
        #    time.sleep(self.interval)
        if self.points is None:
            self.points = 120
        print("Get Value of : ", self.points, " points.")

        #try:
        #    rate = self.mi.get_beamrate()
        #    nap_time = self.points/(rate*1.0)
        #except Exception as ex:
        #    nap_time = 1
        #    print("Something went wrong with the beam rate calculation. Let's sleep 1 second.")
        #    print("Exception was: ", ex)
        nap_time = 0.5
        #time.sleep(nap_time)
        
        #wait for objective value to stablize
        timeout=10
        stoptime=time.time()+timeout
        #while (time.time()<stoptime):
        #    data0=[]
        #    for i in range(5):
        #        data0.append(self.mi.get_value(self.eid))
        #        time.sleep(nap_time)
        #    obj_error=np.std(data0)
        #    if obj_error<0.005:
        #       break
        #run system command to check the steering controllaw convergence
        os.system('/home/helios/SHANG/oag/apps/src/tcltkapp/oagapp/measL3Charge')
        data0=[]
        for i in range(self.points):
            data0.append(self.mi.get_value(self.eid))
            time.sleep(nap_time)
        datain = np.mean(data0)
        #datain = self.mi.get_value(self.eid)

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
        print(time.time())
        print(self.stats.display_name, ' of ', len(self.objective_acquisitions), ' points is ', self.statistic,
              ' and standard deviation is ', self.objective_stdev)

        charge, current = self.mi.get_charge_current()
        #read devices and losses pv values
        
        losses = self.mi.get_losses()
        return self.statistic, self.objective_stdev, charge, current, losses

    def clean(self):
        Target.clean(self)
        self.objective_acquisitions = []  # all the points
        self.objective_means = []
        self.std_dev = []
        self.charge = []
        self.current = []
        self.losses = []
        self.device_values=[]
