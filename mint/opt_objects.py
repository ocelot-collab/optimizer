"""
Objective function and devices
S.Tomin, 2017
"""
from __future__ import absolute_import, print_function
import os
import numpy as np
import time
from datetime import datetime
import json

from PyQt5.QtWidgets import QWidget


class MachineInterface(object):
    def __init__(self, args):
        self.debug = False
        self._save_at_exit = True
        self._use_num_points = False
        path2optimizer = os.path.abspath(os.path.join(__file__ , "../.."))
        self.config_dir = os.path.join(path2optimizer, "parameters")
        self.path2jsondir = os.path.join(os.path.abspath(os.path.join(__file__ , "../../..")), "data")

    def save_at_exit(self):
        """
        Determine whether or not to save to file the screen options when closing
        the software.
        :return: (bool)
        """
        return self._save_at_exit

    @staticmethod
    def add_args(subparser):
        """
        Method that will add the Machine interface specific arguments to the
        command line argument parser.

        :param subparser: (ArgumentParser)
        :return:
        """
        return

    def use_num_points(self):
        """
        Determine whether or not a "Number of Points" to acquire must be added at the interface and passed on to the
        Target function.
        This is useful at machines in which you have many points during one acquisition and you want to perform some
        statistics on the data.
        :return: (bool)
        """
        return self._use_num_points

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

    def customize_ui(self, gui):
        """
        Method invoked to modify the UI and apply customizations pertinent to the
        Machine Interface

        :param gui: (MainWindow) The application Main Window
        :return: None
        """
        pass

    def logbook(self, gui):
        """
        Method invoked when the Logbook button is pressed at the Main Screen.

        :param gui: (MainWindow) The application Main Window
        :return: None
        """
        filename = "screenshot"
        filetype = "png"
        self.screenShot(gui, filename, filetype)
        table = gui.Form.scan_params

        # curr_time = datetime.now()
        # timeString = curr_time.strftime("%Y-%m-%dT%H:%M:%S")
        text = ""

        if not gui.cb_use_predef.checkState():
            if str(gui.le_a.text()) != "" and gui.is_le_addr_ok(gui.le_b):
                text += "obj func: A   : " + str(gui.le_a.text()).split("/")[-2] + "/" + str(gui.le_a.text()).split("/")[-1] + "\n"
            if str(gui.le_b.text()) != "" and gui.is_le_addr_ok(gui.le_b):
                text += "obj func: B   : " + str(gui.le_b.text()).split("/")[-2] + "/" + \
                        str(gui.le_b.text()).split("/")[-1] + "\n"
            if str(gui.le_c.text()) != "" and gui.is_le_addr_ok(gui.le_c):
                text += "obj func: C   : " + str(gui.le_c.text()).split("/")[-2] + "/" + \
                        str(gui.le_c.text()).split("/")[-1] + "\n"
            if str(gui.le_d.text()) != "" and gui.is_le_addr_ok(gui.le_d):
                text += "obj func: D   : " + str(gui.le_d.text()).split("/")[-2] + "/" + \
                        str(gui.le_d.text()).split("/")[-1] + "\n"
            if str(gui.le_e.text()) != "" and gui.is_le_addr_ok(gui.le_e):
                text += "obj func: E   : " + str(gui.le_e.text()).split("/")[-2] + "/" + \
                        str(gui.le_e.text()).split("/")[-1] + "\n"
            text += "obj func: expr: " + str(gui.le_obf.text()) + "\n"
        else:
            try:
                text += "obj func: A   : predefined  " + gui.Form.objective_func.eid + "\n"
            except:
                pass
        if table is not None:
            for i, dev in enumerate(table["devs"]):
                # print(dev.split("/"))
                text += "dev           : " + dev.split("/")[-2] + "/" + dev.split("/")[-1] + "   " + str(
                    table["currents"][i][0]) + " --> " + str(
                    table["currents"][i][1]) + "\n"

            text += "iterations    : " + str(table["iter"]) + "\n"
            text += "delay         : " + str(gui.Form.total_delay) + "\n"
            text += "START-->STOP  : " + str(table["sase"][0]) + " --> " + str(table["sase"][1]) + "\n"
            text += "Method        : " + str(table["method"]) + "\n"
        screenshot_data = None
        try:
            with open(gui.Form.optimizer_path + filename + "." + filetype, 'rb') as screenshot:
                screenshot_data = screenshot.read()
        except IOError as ioe:
            print("Could not find screenshot to read. Exception was: ", ioe)
        if gui.Form is not None and gui.Form.mi is not None:
            res = self.send_to_logbook(author="", title="OCELOT Optimization", severity="INFO", text=text,
                                               image=screenshot_data)

        if not res:
            gui.Form.error_box("error during eLogBook sending")

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

    def screenShot(self, gui, filename, filetype="png"):
        """
        Takes a screenshot of the whole gui window, saves png and ps images to file
        :param filename: (str) Directory string of where to save the file
        :param filetype: (str) String of the filetype to save
        :return:
        """

        s = str(filename) + "." + str(filetype)
        p = QWidget.grab(gui.Form)
        p.save(s, 'png')
        p = p.scaled(465, 400)
        # save again a small image to use for the logbook thumbnail
        p.save(str(s[:-4]) + "_sm.png", 'png')

    def device_factory(self, pv):
        """
        Create a device for the given PV using the proper Device Class.

        :param pv: (str) The process variable for which to create the device.
        :return: (Device) The device instance for the given PV.
        """
        return Device(eid=pv)

    def get_plot_attrs(self):
        """
        Returns a list of attributes to be fetched from Target class to present at the Plot 1.

        :return: (list) Attributes from the Target class to be used in the plot.
        """
        return [("values", "values")]

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

        if objective_func is None:
            return False, "Objective Function required to save data."


        dump2json = {}

        for dev in devices:
            dump2json[dev.eid] = dev.values

        dump2json["method"] = method_name
        dump2json["dev_times"] = devices[0].times
        dump2json["obj_values"] = objective_func.values
        dump2json["obj_times"] = objective_func.times
        dump2json["maximization"] = maximization
        dump2json["std"] = objective_func.std_dev
        dump2json["nreadings"] = objective_func.nreadings
        dump2json["function"] = objective_func.eid
        if not os.path.exists(self.path2jsondir):
            os.makedirs(self.path2jsondir)

        filename = os.path.join(self.path2jsondir, datetime.now().strftime("%Y-%m-%d %H-%M-%S") + ".json")
        try:
            with open(filename, 'w+') as f:
                json.dump(dump2json, f)
        except:
            print("ERROR. Could not write data.")
        return True, ""


    def get_preset_settings(self):
        """
        Return the preset settings to be assembled as Push Buttons at the user interface for quick load of settings.

        :return: (dict) Dictionary with Key being the group name and as value an array of dictionaries following the
        format:
            {"display": "Text of the PushButton", "filename": "my_file.json"}
        """
        return dict()

    def get_obj_function_module(self):
        """
        Return the module with the proper Target class.

        :return: module
        """
        from mint import opt_objects
        return opt_objects

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
        self._can_edit_limits = True
        self.istep = None               # initial step

    def set_value(self, val):
        self.values.append(val)
        self.times.append(time.time())
        self.target = val
        self.mi.set_value(self.eid, val)

    def set_low_limit(self, val):
        self.low_limit = val

    def set_high_limit(self, val):
        self.high_limit = val

    def get_value(self, save=False):
        val = self.mi.get_value(self.eid)
        return val

    def trigger(self):
        pass

    def wait(self):
        if self.target is None:
            return

        start_time = time.time()
        while  time.time() <= start_time + self.timeout:
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
        """
        return True if value is out of limits, otherwise False

        :param value:
        :return:
        """
        limits = self.get_limits()
        # Disable Limits when both are 0.
        if np.abs(limits[0]) < 1e-15 and np.abs(limits[1]) < 1e-15:
            return False
        if value < limits[0] or value > limits[1]:
            print('limits exceeded for ', self.id, " - ", value, limits[0], value, limits[1])
            return True
        return False

    def get_limits(self):
        return [self.low_limit, self.high_limit]

    def get_delta(self):
        """
        Calculate and return the travel range for this device.

        :return: (float) Travel Range
        """
        ll, hl = self.get_limits()
        return hl-ll


# for testing
class TestDevice(Device):
    def __init__(self, eid=None):
        super(TestDevice, self).__init__(eid=eid)
        self.test_value = 0.
        self.values = []
        self.times = []
        self.nsets = 0
        self.mi = None

    def get_value(self, save=False):
        return self.test_value

    def set_value(self, value):
        self.values.append(value)
        self.nsets += 1
        self.times.append(time.time())
        self.test_value = value


class Target(object):
    """
    The class calculates of the penalty of the optimized function.
    Example:
    --------
    goal is SASE maximization:
    penalty = - sase_value + alarm_value
    penalty goes down -> SASE goes up.

    """
    def __init__(self, eid=None):
        """

        :param eid: ID
        """
        self.eid = eid
        self.id = eid
        self.pen_max = 100

        self.clean()
        self.nreadings = 1
        self.interval = 0.0
        self.stats = None
        self.points = None
        self.mi = None
        self.clean_ref_data()
        self.devices = []

    def collect_ref_data(self):
        try:
            ref_sase = self.mi.get_ref_sase_signal()
        except:
            print("ERROR: could not read ref_sase")
            ref_sase = None
        self.ref_sase.append(ref_sase)

    def clean_ref_data(self):
        self.ref_sase = []

    def get_value(self):
        return 0

    def get_penalty(self):
        """
        Method to calculate the penalty on the basis of the value and alarm level.
        penalty = -get_value() + alarm()

        :return: penalty
        """
        data = []
        for i in range(self.nreadings ):
            data.append(self.get_value())
            time.sleep(self.interval)
        sase = np.mean(data)
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
        self.objective_acquisitions.append(np.array(data))
        self.std_dev.append(np.std(data))
        self.penalties.append(pen)
        self.times.append(time.time())
        self.values.append(sase)
        self.alarms.append(alarm)
        self.collect_ref_data()
        return pen

    def get_alarm(self):
        return 0

    def clean(self):
        self.niter = 0
        self.penalties = []
        self.times = []
        self.alarms = []
        self.values = []
        self.objective_acquisitions = []
        self.std_dev = []


class Target_test(Target):
    def __init__(self, mi=None, eid=None):
        super(Target_test, self).__init__(eid=eid)
        """
        :param mi: Machine interface
        :param eid: ID
        """
        self.mi = mi

    def get_value(self):
        values = np.array([dev.get_value() for dev in self.devices])
        return np.sum(np.exp(-np.power((values - np.ones_like(values)), 2) / 5.))



