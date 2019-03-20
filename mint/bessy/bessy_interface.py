"""
Machine interface file for the LCLS to ocelot optimizer


"""
from __future__ import absolute_import, print_function
import os
import sys
from collections import OrderedDict
import numpy as np
import pandas as pd

from re import sub
from xml.etree import ElementTree
from shutil import copy
from datetime import datetime

from PyQt5.QtWidgets import QWidget

try:
    import Image
except:
    try:
        from Pillow import Image
    except:
        try:
            from PIL import Image
        except:
            print('No Module named Image')

try:
    import epics
    epics.ca.DEFAULT_CONNECTION_TIMEOUT = 0.1
except ImportError:
    # Ignore the error since maybe no one is trying to use it... we will raise on the ctor.
    pass

from mint.opt_objects import MachineInterface
#from mint.lcls.lcls_devices import LCLSQuad, LCLSDevice


class BESSYMachineInterface(MachineInterface):
    name = 'BESSYMachineInterface'

    def __init__(self, args=None):
        super(BESSYMachineInterface, self).__init__(args)
        self._save_at_exit = True
        self._use_num_points = True
 
        path2root = os.path.abspath(os.path.join(__file__ , "../../../.."))
        self.config_dir = os.path.join(path2root, "config_optim")
        if 'epics' not in sys.modules:
            raise Exception('No module named epics. LCLSMachineInterface will not work. Try simulation mode instead.')

        self.data = dict()
        self.pvs = dict()


    #@staticmethod
    #def get_params_folder():
    #    """
    #    Returns the path to parameters/lcls folder in this tree.
    #
    #    :return: (str)
    #    """
    #    this_dir = os.path.dirname(os.path.realpath(__file__))
    #    return os.path.realpath(os.path.join(this_dir, '..', '..', 'parameters', 'lcls'))

    #def device_factory(self, pv):
    #    if pv.startswith("QUAD:"):
    #        return LCLSQuad(pv)
    #    d = LCLSDevice(eid=pv)
    #    return d

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

    def get_energy(self):
        """
        Returns the energy.

        :return: (float)
        """
        return 1.7# self.get_value("BEND:DMP1:400:BDES")

    def get_charge(self):
        """
        Returns the charge.

        :return: (float)
        """
        charge = self.get_value('SIOC:SYS0:ML00:CALC252')
        return 0.5# charge

    def get_charge_current(self):
        """
        Returns the current charge and current tuple.

        :return: (tuple) Charge, Current
        """
        charge = self.get_charge()
        current = self.get_value('BLEN:LI24:886:BIMAX')
        return charge, current

    def get_losses(self):
        losses = [self.get_value(pv) for pv in self.losspvs]
        return losses

    def logbook(self, gui):
        pass

    def screenShot(self, gui, filename, filetype):
        """
        Takes a screenshot of the whole gui window, saves png and ps images to file
        """
        pass

    def get_obj_function_module(self):
        from mint.bessy import bessy_obj_function
        return bessy_obj_function

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
        devs = OrderedDict([
            ("TEMP", ["WFGENC2S7G:setVolt"]),
            ("LI21 M. Quads", ["QUAD:LI21:201:BCTRL", "QUAD:LI21:211:BCTRL", "QUAD:LI21:271:BCTRL",
                               "QUAD:LI21:278:BCTRL"]),
            ("LI26 201-501", ["QUAD:LI26:201:BCTRL", "QUAD:LI26:301:BCTRL", "QUAD:LI26:401:BCTRL",
                              "QUAD:LI26:501:BCTRL"]),
            ("LI26 601-901", ["QUAD:LI26:601:BCTRL", "QUAD:LI26:701:BCTRL", "QUAD:LI26:801:BCTRL",
                              "QUAD:LI26:901:BCTRL"]),
            ("LTU M. Quads", ["QUAD:LTU1:620:BCTRL", "QUAD:LTU1:640:BCTRL", "QUAD:LTU1:660:BCTRL",
                              "QUAD:LTU1:680:BCTRL"]),
            ("Dispersion Quads", ["QUAD:LI21:221:BCTRL", "QUAD:LI21:251:BCTRL", "QUAD:LI24:740:BCTRL",
                                  "QUAD:LI24:860:BCTRL", "QUAD:LTU1:440:BCTRL", "QUAD:LTU1:460:BCTRL"]),
            ("CQ01/SQ01/Sol.", ["SOLN:IN20:121:BCTRL", "QUAD:IN20:121:BCTRL", "QUAD:IN20:122:BCTRL"]),
            ("DMD PVs", ["DMD:IN20:1:DELAY_1", "DMD:IN20:1:DELAY_2", "DMD:IN20:1:WIDTH_2", "SIOC:SYS0:ML03:AO956"])
        ])
        return devs

    def get_plot_attrs_(self):
        """
        Returns a list of tuples in which the first element is the attributes to be fetched from Target class
        to present at the Plot 1 and the second element is the label to be used at legend.

        :return: (list) Attributes from the Target class to be used in the plot.
        """
        return [("values"), ("objective_means", "mean")]

    def write_data(self, method_name, objective_func, devices=[], maximization=False, max_iter=0):
        """
        Save optimization parameters to the Database

        :param method_name: (str) The used method name.
        :param objective_func: (Target) The Target class object.
        :param devices: (list) The list of devices on this run.
        :param maximization: (bool) Whether or not the data collection was a maximization. Default is False.
        :param max_iter: (int) Maximum number of Iterations. Default is 0.

        :return: status (bool), error_msg (str)
        """
        pass
