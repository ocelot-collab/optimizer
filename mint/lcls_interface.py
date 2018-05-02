"""
Machine interface file for the LCLS to ocelot optimizer


"""
from __future__ import absolute_import, print_function
try:
    import epics
    epics.ca.DEFAULT_CONNECTION_TIMEOUT = 0.1
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
            return self.pvs[device_name].get()
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

    def get_charge(self):
        """
        Returns the charge.

        :return: (float)
        """
        charge = self.get_value('SIOC:SYS0:ML00:CALC252')
        return charge

    def get_charge_current(self):
        """
        Returns the current charge and current tuple.

        :return: (tuple) Charge, Current
        """
        charge = self.get_charge()
        current = self.get_value('BLEN:LI24:886:BIMAX')
        return charge, current

    def send_to_logbook(self, *args, **kwargs):
        """
        Send information to the electronic logbook.

        :param args: (list) Values sent to the method without keywork
        :param kwargs: (dict) Dictionary with key value pairs representing all the metadata hat is available for the entry.
        :return: (bool) True when the entry was successfully generated, False otherwise.
        """
        print("Called send to Logbook with: \nArgs: {}\nand\nKwargs: {}".format(args, kwargs))

    def get_obj_function_module(self):
        from mint import lcls_obj_function
        return lcls_obj_function

    def get_preset_settings(self):
        """
        Return the preset settings to be assembled as Push Buttons at the user interface for quick load of settings.

        :return: (dict) Dictionary with Key being the group name and as value an array of dictionaries following the
        format:
            {"display": "Text of the PushButton", "filename": "my_file.json"}
        """
        # presets = {
        #     "QUADS Optimization": [
        #         {"display": "1. Launch QUADS", "filename": "sase1_1.json"},
        #     ]
        # }
        presets = dict()
        return presets

    def get_quick_add_devices(self):
        """
        Return a dictionary with:
        {
        "QUADS1" : ["...", "..."],
        "QUADS2": ["...", "..."]
        }

        That is converted into a combobox which allow users to easily populate the devices list

        :return: dict
        """
        devs = {
            "IN20 M. Quads": ["QUAD:IN20:361:BCTRL", "QUAD:IN20:371:BCTRL", "QUAD:IN20:425:BCTRL",
                              "QUAD:IN20:441:BCTRL", "QUAD:IN20:511:BCTRL", "QUAD:IN20:525:BCTRL"],
            "LI21 M. Quads": ["QUAD:LI21:201:BCTRL", "QUAD:LI21:211:BCTRL", "QUAD:LI21:271:BCTRL",
                              "QUAD:LI21:278:BCTRL"],
            "LI26 201-501": ["QUAD:LI26:201:BCTRL", "QUAD:LI26:301:BCTRL", "QUAD:LI26:401:BCTRL",
                             "QUAD:LI26:501:BCTRL"],
            "LI26 601-901": ["QUAD:LI26:601:BCTRL", "QUAD:LI26:701:BCTRL", "QUAD:LI26:801:BCTRL",
                             "QUAD:LI26:901:BCTRL"],
            "LTU M. Quads": ["QUAD:LTU1:620:BCTRL", "QUAD:LTU1:640:BCTRL", "QUAD:LTU1:660:BCTRL",
                             "QUAD:LTU1:680:BCTRL"],
            "Dispersion Quads": ["QUAD:LI21:221:BCTRL", "QUAD:LI21:251:BCTRL", "QUAD:LI24:740:BCTRL",
                                 "QUAD:LI24:860:BCTRL", "QUAD:LTU1:440:BCTRL", "QUAD:LTU1:460:BCTRL"],
            "CQ01/SQ01/Sol.": ["SOLN:IN20:121:BCTRL", "QUAD:IN20:121:BCTRL", "QUAD:IN20:122:BCTRL"],
            "DMD PVs": ["DMD:IN20:1:DELAY_1", "DMD:IN20:1:DELAY_2", "DMD:IN20:1:WIDTH_2", "SIOC:SYS0:ML03:AO956"]

        }
        return devs