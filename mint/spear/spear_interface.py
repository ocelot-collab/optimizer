# -*- coding: iso-8859-1 -*-
"""
Machine interface file for the LCLS to ocelot optimizer


"""
from __future__ import absolute_import, print_function
import os
import sys
from collections import OrderedDict
import numpy as np

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

# Fix Python 2.x.
try:
    UNICODE_EXISTS = bool(type(unicode))
except NameError:
    unicode = str

from mint.opt_objects import MachineInterface
from mint.spear.spear_devices import SPEARMCORDevice, SPEARDevice


def no_op(*args, **kwargs):
    print("Write operation disabled. Running in Read Only Mode")


class SPEARMachineInterface(MachineInterface):
    name = 'SPEARMachineInterface'

    def __init__(self, args=None):
        super(SPEARMachineInterface, self).__init__(args)
        self.config_dir = os.path.join(self.config_dir,
                                       "spear")  # <ocelot>/parameters/lcls
        self._save_at_exit = False
        self._use_num_points = True
        self.read_only = False

        if 'epics' not in sys.modules:
            raise Exception('No module named epics. SPEARMachineInterface will not work. Try simulation mode instead.')

        if args.get('read_only', False):
            print("****************************************************************")
            print("************************* WARNING ******************************")
            print("****************************************************************")
            print("          Runnning SPEAR Interface in Read Only Mode.")
            print("          Runnning SPEAR Interface in Read Only Mode.")
            print("          Runnning SPEAR Interface in Read Only Mode.")
            print("          Runnning SPEAR Interface in Read Only Mode.")
            print("          Runnning SPEAR Interface in Read Only Mode.")
            print("          Runnning SPEAR Interface in Read Only Mode.")
            print("          Runnning SPEAR Interface in Read Only Mode.")
            print("          Runnning SPEAR Interface in Read Only Mode.")
            print("          Runnning SPEAR Interface in Read Only Mode.")
            print("          Runnning SPEAR Interface in Read Only Mode.")
            print("          Runnning SPEAR Interface in Read Only Mode.")
            print("****************************************************************")
            epics.caput = no_op
            epics.ca.put = no_op
            self.read_only = True

        self.data = dict()
        self.pvs = dict()

 #        # grab loss pvs # TODO: Fix this filename...
 #        self.losses_filename = os.path.join(self.get_params_folder(), 'lion.pvs')
 #        try:
 #            self.losspvs = pd.read_csv(self.losses_filename, header=None)  # ionization chamber values
 #            self.losspvs = [pv[0] for pv in np.array(self.losspvs)]
 #            print(self.name, ' - INFO: Loaded ', len(self.losspvs), ' loss PVs from ', self.losses_filename)
 #        except:
 #            self.losspvs = []
 #            print(self.name, ' - WARNING: Could not read ', self.losses_filename)
 #        self.get_energy()
 #        self.get_charge_current()
 #        self.get_beamrate()
 #        self.get_losses()
        

    @staticmethod
    def add_args(parser):
        """
        Method that will add the Machine interface specific arguments to the
        command line argument parser.

        :param parser: (ArgumentParser)
        :return:
        """
        if not parser:
            return

        parser.add_argument('--read-only', default=False, required=False, action='store_true',
                            help="Disable write operations to the process variables.")

    def customize_ui(self, gui):
        gui.ui.sb_max_pen.setVisible(False)
        gui.ui.label_26.setVisible(False)        

    @staticmethod
    def get_params_folder():
        """
        Returns the path to parameters/lcls folder in this tree.

        :return: (str)
        """
        this_dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.realpath(os.path.join(this_dir, '..', '..', 'parameters', 'spear'))

    def device_factory(self, pv):
        if any([x in pv for x in (':CurrSetpt', ':Curr1')]):
            return SPEARMCORDevice(pv, mi=self)
        d = SPEARDevice(eid=pv, mi=self)
        return d
        

    def get_value(self, device_name, with_time=False):
        """
        Getter function for lcls.

        :param device_name: (str) PV name used in caput
        :return: (object) Data from the PV. Variable data type depending on PV type
        """
        pv = self.pvs.get(device_name, None)
        if pv is None:
            self.pvs[device_name] = epics.get_pv(device_name)
            if with_time==True:
               return (self.pvs[device_name].value, self.pvs[device_name].timestamp)
            else: 
               return self.pvs[device_name].value
        else:
            if not pv.connected:
                #print('{} is not connected'.format(device_name))
                return None
            else:
                if with_time==True:
                    return pv.value, pv.timestamp
                else:
                    return pv.value

#    def get_limits(self, device_name):
#        """
#        Getter function for lcls device limits.
#
#        :param device_name: (str) PV name used in caput
#        :return: (object) Data from the PV. Variable data type depending on PV type
#        """
#        pv = self.pvs.get(device_name, None)
#        if pv is None:
#            self.pvs[device_name] = epics.get_pv(device_name, form='ctrl')
#            #print('{} is at least being gotten from epics'.format(device_name))
#            return (self.pvs[device_name].lower_ctrl_limit, self.pvs[device_name].upper_ctrl_limit)
#        else:
#            if not pv.connected:
#                #print('{} is not connected'.format(device_name))
#                return None
#            else:
#                return (self.pvs[device_name].lower_ctrl_limit, self.pvs[device_name].upper_ctrl_limit)

    def set_value(self, device_name, val):
        
        """
        Setter function for SPEAR.
        performs ControlState and set point sequence to change skew quad setting

        :param device_name: (str) PV name used in caput
        :param val: (object) Value to write to device. Variable data type depending on PV type
        """
        pv = self.pvs.get(device_name, None)
        if pv is None:
            self.pvs[device_name] = epics.get_pv(device_name)
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
        return 1.

    def get_charge(self):
        """
        Returns the charge.

        :return: (float)
        """
        charge = 0.
        return charge

    def get_charge_current(self):
        """
        Returns the current charge and current tuple.

        :return: (tuple) Charge, Current
        """
        charge = self.get_charge()
        current = self.get_value('SPEAR:BeamCurrAvg')
        return charge, current

    def get_beamrate(self):
        return 1

    def get_losses(self):
        losses = [self.get_value('SPEAR:BeamLossRate')]
        return losses

    def logbook(self, gui):
        # Put an extra string into the logbook function
        objective_func = gui.Form.objective_func
        objective_func_pv = objective_func.eid

        log_text = ""
        if len(objective_func.values) > 0:
            log_text = "Gain (" + str(objective_func_pv) + "): " + str(round(objective_func.values[0], 4)) + " -> " + str(
            round(objective_func.values[-1], 4))
        log_text += "\nIterations: "+str(objective_func.niter)+"\n"
        log_text += "Trim delay: "+str(gui.sb_tdelay.value())+"\n"
        log_text += "Points Requested: "+str(objective_func.points)+"\n"
        log_text += "Normalization Amp Coeff: "+str(gui.sb_scaling_coef.value())+"\n"
        log_text += "Type of optimization: "+str(gui.cb_select_alg.currentText())+"\n"
                    
        try:
            log_text += "Data location: "+self.last_filename
            if(self.last_filename[-4:] != '.mat'):
                log_text += ".mat"
            log_text += "\n"
        except:
            pass
        try:
            log_text += "Log location: "+self.logpath+"\n"
        except:
            pass
                            
        curr_time = datetime.now()
        timeString = curr_time.strftime("%Y-%m-%dT%H:%M:%S")
        log_entry = ElementTree.Element(None)
        severity = ElementTree.SubElement(log_entry, 'severity')
        location = ElementTree.SubElement(log_entry, 'location')
        keywords = ElementTree.SubElement(log_entry, 'keywords')
        time = ElementTree.SubElement(log_entry, 'time')
        isodate = ElementTree.SubElement(log_entry, 'isodate')
        log_user = ElementTree.SubElement(log_entry, 'author')
        category = ElementTree.SubElement(log_entry, 'category')
        title = ElementTree.SubElement(log_entry, 'title')
        metainfo = ElementTree.SubElement(log_entry, 'metainfo')
        imageFile = ElementTree.SubElement(log_entry, 'link')
        imageFile.text = timeString + '-00.ps'
        thumbnail = ElementTree.SubElement(log_entry, 'file')
        thumbnail.text = timeString + "-00.png"
        text = ElementTree.SubElement(log_entry, 'text')
        log_entry.attrib['type'] = "LOGENTRY"
        category.text = "USERLOG"
        location.text = "not set"
        severity.text = "NONE"
        keywords.text = "none"
        time.text = curr_time.strftime("%H:%M:%S")
        isodate.text = curr_time.strftime("%Y-%m-%d")
        metainfo.text = timeString + "-00.xml"
        fileName = "/tmp/" + metainfo.text
        fileName = fileName.rstrip(".xml")
        log_user.text = " "
        title.text = unicode("Ocelot Interface")
        text.text = log_text
        if text.text == "":
            text.text = " "  # If field is truly empty, ElementTree leaves off tag entirely which causes logbook parser to fail
        xmlFile = open(fileName + '.xml', "w")
        rawString = ElementTree.tostring(log_entry, 'utf-8')
        parsedString = sub(r'(?=<[^/].*>)', '\n', rawString)
        xmlString = parsedString[1:]
        xmlFile.write(xmlString)
        xmlFile.write("\n")  # Close with newline so cron job parses correctly
        xmlFile.close()
        self.screenShot(gui, fileName, 'png')
        try:
            try:
                path = "/u1/lcls/physics/logbook/data/"
                copy(fileName + '.ps', path)
                copy(fileName + '.png', path)
                copy(fileName + '.xml', path)
            except:
                path = "~/ocelot/logbook/data/"
                copy(fileName + '.ps', path)
                copy(fileName + '.png', path)
                copy(fileName + '.xml', path)
        except:
            print("Logbook could not copy files {} to {}".format(fileName, path))

    def screenShot(self, gui, filename, filetype):
        """
        Takes a screenshot of the whole gui window, saves png and ps images to file
        """
        s = str(filename)+"."+str(filetype)
        p = QWidget.grab(gui.Form)
        p.save(s, 'png')
        im = Image.open(s)
        im.save(s[:-4]+".ps")
        p = p.scaled(465,400)
        p.save(str(s), 'png')

    def get_obj_function_module(self):
        from mint.spear import spear_obj_function
        return spear_obj_function

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
            ("SPEAR Skew Quads", ["01G-QSS4:Curr1", "02G-QSS3:Curr1", "05G-QSS3:Curr1", "07G-QSS2:Curr1",
                                    "08G-QSS2:Curr1", "09G-QSS1:Curr1", "10G-QSS4:Curr1",
                                    "11G-QSS3:Curr1", "12G-QSS3:Curr1", "14G-QSS2:Curr1",
                                    "16G-QSS2:Curr1", "17G-QSS2:Curr1", "18G-QSS1:Curr1"])
        ])
        return devs

    def get_plot_attrs(self):
        """
        Returns a list of tuples in which the first element is the attributes to be fetched from Target class
        to present at the Plot 1 and the second element is the label to be used at legend.

        :return: (list) Attributes from the Target class to be used in the plot.
        """
        return [("values", "statistics"), ("objective_means", "mean")]

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
        try:
            import mint.spear.simlog as simlog
            import matlog
        except ImportError as ex:
            print(
                "Error importing matlog, reverting to simlog. The error was: ",
                ex)
            import mint.spear.simlog as matlog

        def byteify(input):
            if isinstance(input, dict):
                return {byteify(key): byteify(value)
                        for key, value in input.iteritems()}
            elif isinstance(input, list):
                return [byteify(element) for element in input]
            elif isinstance(input, unicode):
                return input.encode('utf-8')
            else:
                return input

        def removeUnicodeKeys(input_dict):  # implemented line 91
            return dict([(byteify(e[0]), e[1]) for e in input_dict.items()])

        print(self.name + " - Write Data: ", method_name)
        try:  # if GP is used, the model is saved via saveModel first
            self.data
        except:
            self.data = {}  # dict of all devices deing scanned

        objective_func_pv = objective_func.eid

        self.data[objective_func_pv] = []  # detector data array
        self.data['DetectorAll'] = []  # detector acquisition array
        self.data['DetectorStat'] = []  # detector mean array
        self.data['DetectorStd'] = []  # detector std array
        self.data['timestamps'] = []  # timestamp array
        self.data['charge'] = []
        self.data['current'] = []
        self.data['stat_name'] = []
        # end try/except
        self.data['pv_list'] = [dev.eid for dev in devices]  # device names
        for dev in devices:
            # these 3 lines are to preserve backwards compatibility
            self.data[dev.eid] = []
            vals = len(dev.values)
            self.data[dev.eid].append(dev.values)
            # these 3 lines add redundancy but fix a problem with the MAT file format
            self.data['PV_'+dev.eid.replace('-','_')] = []
            self.data['PV_'+dev.eid.replace('-','_')].append(dev.values)
        if vals < len(objective_func.values):  # first point is duplicated for some reason so dropping
            objective_func.values = objective_func.values[1:]
            objective_func.objective_means = objective_func.objective_means[1:]
            objective_func.objective_acquisitions = objective_func.objective_acquisitions[1:]
            objective_func.times = objective_func.times[1:]
            objective_func.std_dev = objective_func.std_dev[1:]
            objective_func.charge = objective_func.charge[1:]
            objective_func.current = objective_func.current[1:]
            try:
                objective_func.losses = objective_func.losses[1:]
            except:
                pass
            objective_func.niter -= 1
        self.data[objective_func_pv].append(objective_func.objective_means)  # this is mean for compat
        self.data['DetectorAll'].append(objective_func.objective_acquisitions)
        self.data['DetectorStat'].append(objective_func.values)
        self.data['DetectorStd'].append(objective_func.std_dev)
        self.data['timestamps'].append(objective_func.times)
        self.data['charge'].append(objective_func.charge)
        self.data['current'].append(objective_func.current)
        self.data['stat_name'].append(objective_func.stats.display_name)
 #        for ipv in range(len(self.losspvs)):
 #            self.data[self.losspvs[ipv]] = [a[ipv] for a in objective_func.losses]

        self.detValStart = self.data[objective_func_pv][0]
        self.detValStop = self.data[objective_func_pv][-1]

        # replace with matlab friendly strings
        for key in self.data:
            key2 = key.replace(":", "_")
            self.data[key2] = self.data.pop(key)

        # extra into to add into the save file
        self.data["MachineInterface"] = self.name
        try:
            self.data["epicsname"] = epics.name  # returns fakeepics if caput has been disabled
        except:
            pass
        #self.data["BEND_DMP1_400_BDES"] = self.get_value("BEND:DMP1:400:BDES")
        self.data["Energy"] = self.get_energy()
        self.data["ScanAlgorithm"] = str(method_name)  # string of the algorithm name
        self.data["ObjFuncPv"] = str(objective_func_pv)  # string identifing obj func pv
        self.data['DetectorMean'] = str(
            objective_func_pv.replace(":", "_"))  # reminder to look at self.data[objective_func_pv]
        # TODO: Ask Joe if this is really needed...
        #self.data["NormAmpCoeff"] = norm_amp_coeff
        self.data["niter"] = objective_func.niter

        # save data
        if self.read_only:
            self.last_filename = simlog.save("OcelotScan", removeUnicodeKeys(self.data), path='default')
        else:
            self.last_filename = matlog.save("OcelotScan", removeUnicodeKeys(self.data), path='default')  # self.save_path)

        print('Saved scan data to ', self.last_filename)

        # clear for next run
        self.data = dict()

        return True, ""
