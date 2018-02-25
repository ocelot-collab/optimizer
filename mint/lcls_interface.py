"""
Machine interface file for the LCLS to ocelot optimizer


"""
from __future__ import absolute_import, print_function
try:
    import epics
except ImportError:
    # Ignore the error since maybe no one is trying to use it... we will raise on the ctor.
    pass

import sys
from mint.opt_objects import MachineInterface


class LCLSMachineInterface(MachineInterface):
    def __init__(self):
        super(LCLSMachineInterface, self).__init__()

        if 'epics' not in sys.modules:
            raise Exception('No module named epics. LCLSMachineInterface will not work. Try simulation mode instead.')

        # Interface Name
        self.name = 'LCLSMachineInterface'
        self.logbook = 'lclslogbook'  # TODO: Check the proper name with Joe Duris
        self.pvs = dict()

    def get_value(self, device_name):
        """
        Getter function for lcls.

        :param device_name: (str) PV name used in caput
        :return: (object) Data from the PV. Variable data type depending on PV type
        """
        pv = self.pvs.get(device_name, None)
        if pv is None:
            self.pvs[device_name] = epics.PV(device_name)
            return None
        else:
            if not pv.connected:
                return None
            else:
                return pv.get()

    def set_value(self, device_name, val):
        """
        Setter function for lcls.

        :param device_name: (str) PV name used in caput
        :param val: (object) Value to write to device. Variable data type depending on PV type
        """
        pv = self.pvs.get(device_name, None)
        if pv is None:
            self.pvs[device_name] = epics.PV(device_name)
            return None
        else:
            if not pv.connected:
                return None
            else:
                return pv.put(val)

    def send_to_logbook(self, *args, **kwargs):
        """
        Send information to the electronic logbook.

        :param args: (list) Values sent to the method without keywork
        :param kwargs: (dict) Dictionary with key value pairs representing all the metadata hat is available for the entry.
        :return: (bool) True when the entry was successfully generated, False otherwise.
        """
        print("Called send to Logbook with: \nArgs: {}\nand\nKwargs: {}".format(args, kwargs))
