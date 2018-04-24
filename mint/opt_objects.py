"""
Objective function and devices
S.Tomin, 2017
"""

import numpy as np
import time


class MachineInterface(object):
    def __init__(self):
        self.debug = False

    def get_value(self, channel):
        """
        Getter function for a given Machine.

        :param channel: (str) String of the devices name used
        :return: Data from the read on the Control System, variable data type depending on channel
        """
        raise NotImplementedError

    def set_value(self, channel, val):
        """
        Method to set value to a channel

        :param channel: (str) String of the devices name used
        :param val: value
        :return: None
        """
        raise NotImplementedError

    def send_to_logbook(self, *args, **kwargs):
        """
        Send information to the electronic logbook.

        :param args:
            Values sent to the method without keywork
        :param kwargs:
            Dictionary with key value pairs representing all the metadata
            that is available for the entry.
        :return: bool
            True when the entry was successfully generated, False otherwise.
        """
        pass

    def device_factory(self, pv):
        """
        Create a device for the given PV using the proper Device Class.

        :param pv: (str) The process variable for which to create the device.
        :return: (Device) The device instance for the given PV.
        """
        return Device(eid=pv)

    def get_preset_settings(self):
        """
        Return the preset settings to be assembled as Push Buttons at the user interface for quick load of settings.

        :return: (dict) Dictionary with Key being the group name and as value an array of dictionaries following the
        format:
            {"display": "Text of the PushButton", "filename": "my_file.json"}
        """
        return dict()

class Device(object):
    def __init__(self, eid=None):
        self.eid = eid
        self.id = eid
        self.values = []
        self.times = []
        self.simplex_step = 0
        self.mi = None
        self.tol = 0.001
        self.timeout = 5  # seconds
        self.target = None
        self.low_limit = 0.
        self.high_limit = 0.

    def set_value(self, val):
        self.values.append(val)
        self.times.append(time.time())
        self.target = val
        self.mi.set_value(self.eid, val)

    def set_low_limit(self, val):
        self.low_limit = val

    def set_high_limit(self, val):
        self.high_limit = val

    def get_value(self):
        val = self.mi.get_value(self.eid)
        return val

    def trigger(self):
        pass

    def wait(self):
        if self.target is None:
            return

        start_time = time.time()
        while start_time + self.timeout <= time.time():
            if np.abs(self.get_value()-self.target) < self.tol:
                return
            time.sleep(0.05)

    def state(self):
        """
        Check if device is readable

        :return: state, True if readable and False if not
        """
        state = True
        try:
            self.get_value()
        except:
            state = False
        return state

    def clean(self):
        self.values = []
        self.times = []

    def check_limits(self, value):
        limits = self.get_limits()
        # Disable Limits when both are 0.
        if limits[0] == limits[1] and limits[0] == 0.:
            return False
        if value < limits[0] or value > limits[1]:
            print('limits exceeded for ', self.id, " - ", value, limits[0], value, limits[1])
            return True
        return False

    def get_limits(self):
        return [self.low_limit, self.high_limit]


# for testing
class TestDevice(Device):
    def __init__(self, eid=None):
        super(TestDevice, self).__init__(eid=eid)
        self.test_value = 0.
        self.values = []
        self.times = []
        self.nsets = 0
        self.mi = None

    def get_value(self):
        return self.test_value

    def set_value(self, value):
        self.values.append(value)
        self.nsets += 1
        self.times.append(time.time())
        self.test_value = value


class Target(object):
    def __init__(self, eid=None):
        """

        :param eid: ID
        """
        self.eid = eid
        self.id = eid
        self.pen_max = 100

        self.penalties = []
        self.values = []
        self.alarms = []
        self.times = []
        self.nreadings = 1
        self.interval = 0.0
        self.stats = None

    def get_value(self):
        return 0

    def get_penalty(self):
        """
        Method to calculate the penalty on the basis of the value and alarm level.

        penalty = -get_value() + alarm()


        :return: penalty
        """
        sase = self.get_value()
        for i in range(self.nreadings):
            sase += self.get_value()
            time.sleep(self.interval)
        sase = sase/self.nreadings
        print("SASE", sase)
        alarm = self.get_alarm()
        pen = 0.0
        if alarm >= 0.95:
            alarm = self.pen_max
        if alarm > 0.7:
            alarm = self.pen_max / 2.
        pen += alarm
        pen -= sase
        self.niter += 1
        # print("niter = ", self.niter)
        self.penalties.append(pen)
        self.times.append(time.time())
        self.values.append(sase)
        self.alarms.append(alarm)
        return pen

    def get_alarm(self):
        return 0

    def clean(self):
        self.niter = 0
        self.penalties = []
        self.times = []
        self.alarms = []
        self.values = []


class Target_test(Target):
    def __init__(self, mi=None, eid=None):
        super(Target_test, self).__init__(eid=eid)
        """
        :param mi: Machine interface
        :param eid: ID
        """
        self.mi = mi
        self.debug = False
        self.kill = False
        self.pen_max = 100
        self.niter = 0
        self.penalties = []
        self.times = []
        self.alarms = []
        self.values = []

    def get_penalty(self):
        sase = self.get_value()
        alarm = self.get_alarm()

        if self.debug: print('alarm:', alarm)
        if self.debug: print('sase:', sase)
        pen = 0.0
        if alarm > 1.0:
            return self.pen_max
        if alarm > 0.7:
            return alarm * 50.0
        pen += alarm
        pen -= sase
        if self.debug: print('penalty:', pen)
        self.niter += 1
        print("niter = ", self.niter)
        self.penalties.append(pen)
        self.times.append(time.time())
        self.values.append(sase)
        self.alarms.append(alarm)
        return pen

    def get_value(self):
        values = np.array([dev.get_value() for dev in self.devices])
        return np.sum(np.exp(-np.power((values - np.ones_like(values)), 2) / 5.))

    def get_spectrum(self):
        return [0, 0]

    def get_stat_params(self):
        #spetrum = self.get_spectrum()
        #ave = np.mean(spetrum[(2599 - 5 * 120):-1])
        #std = np.std(spetrum[(2599 - 5 * 120):-1])
        ave = self.get_value()
        std = 0.1
        return ave, std

    def get_alarm(self):
        return 0

    def get_energy(self):
        return 3


