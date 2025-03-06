from __future__ import absolute_import, print_function
import os
import random

try:
    import epics
except:
    pass # Show message on Constructor if we try to use it.

from mint.opt_objects import MachineInterface
from PyQt5.QtWidgets import QWidget


class SwissFELInterface(MachineInterface):
    name = 'SwissFELInterface'

    def __init__(self, args=None):
        super(SwissFELInterface, self).__init__(args=args)
        self.config_dir = os.path.join(self.config_dir, "swissfel")
        self.read_only = False
        self.pvs = dict()
        path2root = os.path.abspath(os.path.join(__file__ , "../../../.."))
        self.config_dir = os.path.join(path2root, "config_optim")

    def get_value(self, device_name):
        pv = self.pvs.get(device_name, None)
        if pv is None:
            self.pvs[device_name] = epics.get_pv(device_name)
            return self.pvs[device_name].get()
        else:
            if not pv.connected:
                return None
            else:
                return pv.get()

    def set_value(self, device_name, val):
        pv = self.pvs.get(device_name, None)
        if pv is None:
            self.pvs[device_name] = epics.get_pv(device_name)
            return None
        else:
            if not pv.connected:
                return None
            else:
                return pv.put(val)


    def get_charge(self):
        return self.get_value("SINEG01-DBPM340:Q1")

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

    def logbook(self, gui):
        objective_func = gui.Form.objective_func
        objective_func_pv = objective_func.eid
        message = ""
        if len(objective_func.values) > 0:
            message = "Gain (" + str(objective_func_pv) + "): " + str(round(objective_func.values[0], 4)) + " -> " + str(
            round(objective_func.values[-1], 4))
        message += "\nIterations: " + str(objective_func.niter) + "\n"
        message += "Trim delay: " + str(gui.sb_tdelay.value())  +"\n"
        message += "Points Requested: " + str(objective_func.points) + "\n"
        message += "Normalization Amp Coeff: " + str(gui.sb_scaling_coef.value()) + "\n"
        message += "Type of optimization: " + str(gui.cb_select_alg.currentText()) + "\n"
        elog = "SwissFEL commissioning data"
        title = "title"
        category = "Info"
        application = "Ocelot"
        self.screenShot(gui, "screenshot", filetype="png")

        attachments = ["screenshot.png"]
        encoding = 1
        cmd =           'G_CS_ELOG_add -l "' + elog          + '" '
        cmd =     cmd + '-a "Title='         + title         + '" '
        cmd =     cmd + '-a "Category='      + category      + '" '
        cmd =     cmd + '-a "Application='   + application   + '" '
        for attachment in attachments:
            cmd = cmd + '-f "'               + attachment    + '" '
        cmd =     cmd + '-n '                + str(encoding)
        cmd =     cmd + ' "'                 + message       + '"'
        import subprocess
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()
        if (err is not None) and err!="":
            raise Exception(err)
            success = False
        else:
            success = True
        return success

    def get_preset_settings(self):
        presets = {
                "Aramis 1": [
                        {"display": "1. matching quads", "filename": "Aramis_matching.json"},
                        {"display": "2. phase shifters", "filename": "Aramis_phase_shifters.json"}],
                "Aramis 2": [
                        {"display": "3. undulator Ks", "filename": "Aramis_K.json"},
                        {"display": "4. ", "filename": "Aramis_matching.json"}],
                "Athos 1": [
                        {"display": "1. matching quads", "filename": "Athos_matching.json"},
                        {"display": "2. phase shifters", "filename": "Athos_phase_shifters.json"}],
                "Athos 2": [
                        {"display": "3. undulator Ks", "filename": "Athos_K.json"},
                        {"display": "4. ", "filename": "Athos_matching.json"}]
                }
        return presets
