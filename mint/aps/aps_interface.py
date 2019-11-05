"""
Machine interface file for the APS to ocelot optimizer


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
import sdds
import mint.aps.simlog as simlog

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
from mint.aps.aps_devices import APSQuad, APSDevice


def no_op(*args, **kwargs):
    print("Write operation disabled. Running in Read Only Mode")


class APSMachineInterface(MachineInterface):
    name = 'APSMachineInterface'

    def __init__(self, args=None):
        super(APSMachineInterface, self).__init__(args)
        self.config_dir = os.path.join(self.config_dir,
                                       "aps")  # <ocelot>/parameters/aps
        self._save_at_exit = False
        self._use_num_points = True
        self.read_only = False

        if 'epics' not in sys.modules:
            raise Exception('No module named epics. APSMachineInterface will not work. Try simulation mode instead.')

        if args.get('read_only', False):
            print("****************************************************************")
            print("************************* WARNING ******************************")
            print("****************************************************************")
            print("          Runnning APS Interface in Read Only Mode.")
            print("          Runnning APS Interface in Read Only Mode.")
            print("          Runnning APS Interface in Read Only Mode.")
            print("          Runnning APS Interface in Read Only Mode.")
            print("          Runnning APS Interface in Read Only Mode.")
            print("          Runnning APS Interface in Read Only Mode.")
            print("          Runnning APS Interface in Read Only Mode.")
            print("          Runnning APS Interface in Read Only Mode.")
            print("          Runnning APS Interface in Read Only Mode.")
            print("          Runnning APS Interface in Read Only Mode.")
            print("          Runnning APS Interface in Read Only Mode.")
            print("****************************************************************")
            epics.caput = no_op
            epics.ca.put = no_op
            self.read_only = True

        self.data = dict()
        self.pvs = dict()

        # grab loss pvs # TODO: Fix this filename...
        self.losses_filename = os.path.join(self.get_params_folder(), 'lion.pvs')
        try:
            self.losspvs = pd.read_csv(self.losses_filename, header=None)  # ionization chamber values
            self.losspvs = [pv[0] for pv in np.array(self.losspvs)]
            print(self.name, ' - INFO: Loaded ', len(self.losspvs), ' loss PVs from ', self.losses_filename)
        except:
            self.losspvs = []
            print(self.name, ' - WARNING: Could not read ', self.losses_filename)
        self.get_energy()
        self.get_charge_current()
        self.get_beamrate()
        self.get_losses()
        

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
        Returns the path to parameters/aps folder in this tree.

        :return: (str)
        """
        this_dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.realpath(os.path.join(this_dir, '..', '..', 'parameters', 'aps'))

    def device_factory(self, pv):
        if pv.endswith("CurrentAO"):
            return APSQuad(pv, mi=self)
        d = APSDevice(eid=pv, mi=self)
        return d

    def get_value(self, device_name):
        """
        Getter function for aps

        :param device_name: (str) PV name used in caput
        :return: (object) Data from the PV. Variable data type depending on PV type
        """
        #should use readback here ?
        #readback =
        device_name.replace("CurrentAO","CurrentAI")
        pv = self.pvs.get(device_name, None)
        #print(device_name)
        
       # print(pv)
        if pv is None:
            self.pvs[device_name] = epics.get_pv(device_name)
            return self.pvs[device_name].get()
        else:
            if not pv.connected:
                return None
            else:
                return pv.get()

    def set_value(self, device_name, val):
        """
        Setter function for aps.

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
                #temporarily suspend for testing sdds logger
                return pv.put(val)
                #return

    def get_energy(self):
        """
        Returns the energy.

        :return: (float)
        """
        """kicker voltage changes the L3 charge quite a lot, we do optimization at fixed kicker voltage"""
        return self.get_value("L1:RG2:KIK:PFNVoltageAI")

    def get_charge(self):
        """
        Returns the charge.

        :return: (float)
        L3 charge is our objective function, we need maximize it
        """
        charge = self.get_value('L3:CM1:measCurrentCM')
        return charge

    def get_charge_current(self):
        """
        Returns the current charge and current tuple.

        :return: (tuple) Charge, Current
        L1:RG2:CM1:measCurrentCM  this current should be 0.13 to 0.15 to make sure there is enough linac beam
        """
        charge = self.get_charge()
        current = self.get_value('L1:RG2:CM1:measCurrentCM')
        return charge, current

    def get_beamrate(self):
        """this is linac RF rate"""
        rate = self.get_value('LI:TM:rfTrigCountCC.VAL')
        return rate

    def get_losses(self):
        losses = [self.get_value(pv) for pv in self.losspvs]
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
            path = "/home/helios/oagData/linac/ocelotOptLog/data/"
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
        from mint.aps import aps_obj_function
        return aps_obj_function

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
            ("L3 Charge Opt NoSteering ", ["L1:RG2:QM1:CurrentAO", "L1:RG2:QM2:CurrentAO","L1:RG2:QM3:CurrentAO","L1:RG2:QM4:CurrentAO",
                                "L1:QM3:CurrentAO", "L1:QM4:CurrentAO", "L1:RG2:SC1:VL:CurrentAO", "L1:RG2:SC2:HZ:CurrentAO",
                                "L1:RG2:SC2:VL:CurrentAO", "L1:RG2:SC3:HZ:CurrentAO", "L1:RG2:SC3:VL:CurrentAO", "L1:QM5:CurrentAO"]),
            ("L3 Charge Opt ", ["L1:RG2:QM1:CurrentAO", "L1:RG2:QM2:CurrentAO","L1:RG2:QM3:CurrentAO","L1:RG2:QM4:CurrentAO",
                                "L1:QM3:CurrentAO", "L1:QM4:CurrentAO", "L1:RG2:SC1:VL:CurrentAO", "L1:RG2:SC2:HZ:CurrentAO",
                                "L1:RG2:SC2:VL:CurrentAO", "L1:RG2:SC3:HZ:CurrentAO", "L1:RG2:SC3:VL:CurrentAO", "L1:SC3:HZ:CurrentAO",
                                "L1:SC3:VL:CurrentAO", "L1:QM5:CurrentAO", "L1:SC4:HZ:CurrentAO", "L1:SC4:VL:CurrentAO"]),
            ("ITS test", ["LTS:H1:CurrentAO", "LTS:V1:CurrentAO", "LTS:H2:CurrentAO"]),
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

    def get_plot_attrs(self):
        """
        Returns a list of tuples in which the first element is the attributes to be fetched from Target class
        to present at the Plot 1 and the second element is the label to be used at legend.

        :return: (list) Attributes from the Target class to be used in the plot.
        """
        return [("values", "statistics"), ("objective_means", "mean")]

    def write_data_old(self, method_name, objective_func, devices=[], maximization=False, max_iter=0):
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
            import mint.aps.simlog as simlog
            import matlog
        except ImportError as ex:
            print(
                "Error importing matlog, reverting to simlog. The error was: ",
                ex)
            import mint.aps.simlog as matlog

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
            self.data[dev.eid] = []
        for dev in devices:
            vals = len(dev.values)
            self.data[dev.eid].append(dev.values)
        
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
        for ipv in range(len(self.losspvs)):
            self.data[self.losspvs[ipv]] = [a[ipv] for a in objective_func.losses]

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
        self.data["BEND_DMP1_400_BDES"] = self.get_value("BEND:DMP1:400:BDES")
        self.data["Energy"] = self.get_energy()
        self.data["ScanAlgorithm"] = str(method_name)  # string of the algorithm name
        self.data["ObjFuncPv"] = str(objective_func_pv)  # string identifing obj func pv
        self.data['DetectorMean'] = str(
            objective_func_pv.replace(":", "_"))  # reminder to look at self.data[objective_func_pv]
        # TODO: Ask Joe if this is really needed...
        #self.data["NormAmpCoeff"] = norm_amp_coeff
        self.data["niter"] = objective_func.niter

        # save data
        self.last_filename = simlog.save("OcelotScan", removeUnicodeKeys(self.data), path='default')
        
        print('Saved scan data to ', self.last_filename)

        # clear for next run
        self.data = dict()

        return True, ""
    #write data in sdds format
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
        """
        H. Shang, found the first point of the devices are reduandant, should be removed
        for GP optimizer, the first 3 points and the last point of the objective should be removed.
        for other optimizers, the first and last point of the objective should be removed.
        """
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
        print(objective_func_pv)
        
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
            self.data[dev.eid] = []
        for dev in devices:
            vals = len(dev.values)
            #removed the frist point of device, which is redundant to the second point
            self.data[dev.eid].append(dev.values[1:])
        self.data['devtime']=dev.times[1:]
        
        print(objective_func.times)
        objvals = len(objective_func.values)
        
        start=0
        end=objvals
        if objvals-vals==1:
            #for other optimizers, the first and last objective should be removed (frist point is redundant)
            #the objective fucntion has 1 more values than the variables
            start=1
            end=-1
            objective_func.niter -=2
        if objvals-vals==3:
            #for gp optimizaers, the objective has 3 more values than the variables
            #the first 3 and the last of objective should be remove (first point is redundant)
            start=3
            end=-1
            objective_func.niter -=4
        
        objective_func.values = objective_func.values[start:end]
        objective_func.objective_means = objective_func.objective_means[start:end]
        objective_func.objective_acquisitions = objective_func.objective_acquisitions[start:end]
        objective_func.times = objective_func.times[start:end]
        objective_func.std_dev = objective_func.std_dev[start:end]
        objective_func.charge = objective_func.charge[start:end]
        objective_func.current = objective_func.current[start:end]
        try:
            objective_func.losses = objective_func.losses[start:end]
        except:
            pass
        
        self.data[objective_func_pv].append(objective_func.objective_means)  # this is mean for compat
        print(self.data[objective_func_pv])
        self.data['DetectorAll'].append(objective_func.objective_acquisitions)
        self.data['DetectorStat'].append(objective_func.values)
        self.data['DetectorStd'].append(objective_func.std_dev)
        self.data['timestamps'].append(objective_func.times)
        self.data['charge'].append(objective_func.charge)
        self.data['current'].append(objective_func.current)
        self.data['stat_name'].append(objective_func.stats.display_name)
        for ipv in range(len(self.losspvs)):
            self.data[self.losspvs[ipv]] = [a[ipv] for a in objective_func.losses]

        self.detValStart = self.data[objective_func_pv][0]
        self.detValStop = self.data[objective_func_pv][-1]
      
        # replace with matlab friendly strings
       # for key in self.data:
       #     key2 = key.replace(":", "_")
       #     self.data[key2] = self.data.pop(key)

        # extra into to add into the save file
        self.data["MachineInterface"] = self.name
        try:
            self.data["epicsname"] = epics.name  # returns fakeepics if caput has been disabled
        except:
            pass
        
        self.data["niter"] = objective_func.niter
        #self.data["BEND_DMP1_400_BDES"] = self.get_value("BEND:DMP1:400:BDES")
        #self.data["Energy"] = self.get_energy()
        self.data["ScanAlgorithm"] = str(method_name)  # string of the algorithm name
        self.data["ObjFuncPv"] = str(objective_func_pv)  # string identifing obj func pv
        self.data['DetectorMean'] = str(
            objective_func_pv.replace(":", "_"))  # reminder to look at self.data[objective_func_pv]
        # TODO: Ask Joe if this is really needed...
        #self.data["NormAmpCoeff"] = norm_amp_coeff
      
        path = simlog.getPath();
        filename = 'OcelotScan-' + method_name + '-' + simlog.getFileTs()+'.sdds'
        fout = os.path.join(path,filename)
    
        sddsData = sdds.SDDS(0)
        sddsData.mode = sddsData.SDDS_ASCII
        objective_func_pv = objective_func.eid
        sddsData.defineSimpleParameter("Objective",sddsData.SDDS_STRING)
        sddsData.defineSimpleParameter("ScanAlgorithm",sddsData.SDDS_STRING)
        sddsData.defineSimpleParameter("Number_Of_Iterations", sddsData.SDDS_LONG)
        sddsData.defineSimpleParameter("DataPoints",sddsData.SDDS_LONG)
        sddsData.defineSimpleParameter("HyperParFile",sddsData.SDDS_STRING)
        sddsData.defineSimpleColumn("Time",sddsData.SDDS_DOUBLE)
        sddsData.defineSimpleColumn("DeviceTime",sddsData.SDDS_DOUBLE)
        sddsData.defineSimpleColumn(objective_func_pv, sddsData.SDDS_DOUBLE)
        print(fout)
        for device in self.data['pv_list']:
           # print(device)
            sddsData.defineSimpleColumn(device,sddsData.SDDS_DOUBLE)
        for ipv in range(len(self.losspvs)):
            sddsData.defineSimpleColumn(self.losspvs[ipv], sddsData.SDDS_DOUBLE)
            self.data[self.losspvs[ipv]] = [a[ipv] for a in objective_func.losses]

        sddsData.setParameterValue("Objective", str(objective_func_pv), 1)
        sddsData.setParameterValue("ScanAlgorithm", self.data["ScanAlgorithm"], 1)
        sddsData.setParameterValue("Number_Of_Iterations",int(self.data["niter"]), 1)
        sddsData.setParameterValue("DataPoints",objective_func.points,1)
        parFile='/usr/local/oag/3rdParty/OcelotOptimizer-dev/parameters/anl_hyperparams.pkl'
        sddsData.setParameterValue('HyperParFile',os.readlink(parFile),1)
        print('timestamp from objective function')
        print(self.data['timestamps'][0])
        for pv in self.data['pv_list']:
            print(pv)
            print(self.data[pv][0])
            rows=len(self.data[pv][0])
            sddsData.setColumnValueList(pv, self.data[pv][0],1)
        rows1 = len(self.data['timestamps'][0])
        #print(self.data['devtime'])
        #print((self.data['timestamps'][0])
        sddsData.setColumnValueList("DeviceTime",self.data['devtime'],1)
        sddsData.setColumnValueList("Time",self.data['timestamps'][0], 1)
        sddsData.setColumnValueList(str(objective_func_pv), self.data[objective_func_pv][0], 1)
      #  print(self.data[objective_func_pv])
       # sddsData.setColumnValueList(str(objective_func_pv), self.data[objective_func_pv][0], 1)
        print('objective values from objective function')
        print(self.data[objective_func_pv][0])
        rows1 = len(self.data[objective_func_pv][0])
        #for pv in self.data['pv_list']:
        #    print(self.data[pv][0])
        #    sddsData.setColumnValueList(pv, self.data[pv][0],1)
        for ipv in range(len(self.losspvs)):
           # print(self.data[self.losspvs[ipv]])
           sddsData.setColumnValueList(self.losspvs[ipv], self.data[self.losspvs[ipv]],1)
        print('save sdds')
        try:
            sddsData.save(fout)
            self.last_filename = fout
            print('Saved scan data to ', self.last_filename)
        except:
            print('Error saving sdds file')

        # clear for next run
        self.data = dict()

        return True, ""
