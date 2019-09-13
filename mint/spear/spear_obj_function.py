# -*- coding: iso-8859-1 -*-
import time
import numpy as np

from mint.opt_objects import Target
import stats.stats as stats


class SLACTarget(Target):
    def __init__(self, mi=None, eid='SPEAR:BeamCurrAvg'):
        """
        :param mi: Machine interface
        :param eid: ID
        """
        super(SLACTarget, self).__init__(eid=eid)

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

        #choose objective function calculation: 0-SPEARBO, 1-Xiaobiao
        self.obj_choice = 0

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
        return pen

    def get_obj(self, duration=1):
	
        while True:
            #print('I am calculating a time-invariant current loss rate for SPEAR using {}...'.format(self.eid))
            curr1 = self.mi.get_value(self.eid, with_time=True)
            #print('First current point: {} at time {}'.format( curr1[0], curr1[1]))
            time.sleep(duration)
            curr2 = self.mi.get_value(self.eid, with_time=True)
            #print('Second current point: {} at time {}'.format( curr2[0], curr2[1]))
            if curr1[0]!=curr2[0]:
                break
        
        diff, avg =(curr2[0]-curr1[0]), 0.5*(curr1[0]+curr2[0])
        
        if duration==1:
                return -diff*500**2/(duration*avg**2)
        else:
            return -diff*500/(duration*avg)

    def get_value(self):
        """
        Returns data for the ojective function (sase) from the selected detector PV.

        At lcls the repetition is  120Hz and the readout buf size is 2800.
        The last 120 entries correspond to pulse energies over past 1 second.

        Returns:
                Float of SASE or other detecor measurement
        """
        #datain = []
        #for i in range(self.nreadings):
        #    datain.extend(self.mi.get_value(self.eid))
        #    time.sleep(self.interval)
        if self.points is None:
            self.points = 1
        print("Get Value of : ", self.points, " points.")

        try:
            rate = self.mi.get_beamrate()
            nap_time = self.points/(rate*1.0)
        except Exception as ex:
            nap_time = 1
            print("Something went wrong with the beam rate calculation. Let's sleep 1 second.")
            print("Exception was: ", ex)

        time.sleep(nap_time)

        #meas = [self.get_obj_DELETE() for i in range(10)]
        #nums = np.array([m[0] for m in meas])
        #dens = np.array([m[1] for m in meas])

        #print('*********************ATTENTION! ', np.std(nums/dens))
        #print('*********************ATTENTION! ', np.std(nums/np.round(1+0*dens)), ' desired times')
        #print('*********************ATTENTION! ', np.std(nums/np.round(dens)), ' rounding on')
        #print('nums = ', nums)
        #print('np.round(dens) = ', np.round(dens))

        if self.obj_choice==0:
        	datain = self.get_obj()
        else:
            datain=self.get_obj(duration=6)
            print('This here is our new objective: ', datain)

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

        charge, current = self.mi.get_charge_current()
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
