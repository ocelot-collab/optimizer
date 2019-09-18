"""
XFEL machine interface
S.Tomin, 2017
"""
from __future__ import absolute_import, print_function

try:
    # in server "doocsdev12" set environment
    #  $ export PYTHONPATH=/home/ttflinac/user/python-2.7/Debian/
    import pydoocs
except:
    pass # Show message on Constructor if we try to use it.

import os
import sys
import numpy as np
import subprocess
import base64
from mint.opt_objects import MachineInterface, Device, TestDevice
from collections import OrderedDict
from datetime import datetime
import json

class AlarmDevice(Device):
    """
    Devices for getting information about Machine status
    """
    def __init__(self, eid=None):
        super(AlarmDevice, self).__init__(eid=eid)


class FLASHMachineInterface(MachineInterface):
    """
    Machine Interface for European XFEL
    """
    name = 'XFELMachineInterface'

    def __init__(self, args=None):
        super(FLASHMachineInterface, self).__init__(args)
        if 'pydoocs' not in sys.modules:
            print('error importing doocs library')
        self.logbook_name = "ttflog"

        path2root = os.path.abspath(os.path.join(__file__ , "../../../.."))
        self.config_dir = os.path.join(path2root, "config_optim")

    def get_value(self, channel):
        """
        Getter function for XFEL.

        :param channel: (str) String of the devices name used in doocs
        :return: Data from pydoocs.read(), variable data type depending on channel
        """
        val = pydoocs.read(channel)
        return val["data"]

    def set_value(self, channel, val):
        """
        Method to set value to a channel

        :param channel: (str) String of the devices name used in doocs
        :param val: value
        :return: None
        """
        pydoocs.write(channel, float(val))
        return


    def get_charge(self):
        try:
            fl1 = self.get_value("FLASH.DIAG/TOROID/3GUN/CHARGE.FLASH1")
        except:
            fl1 = None
        try:
            fl2 = self.get_value("FLASH.DIAG/TOROID/3GUN/CHARGE.FLASH2")
        except:
            fl2 = None
        return [fl1, fl2]

    def get_sases(self):
        try:
            fl1 = self.get_value("TTF2.FEL/BKR.FLASH.STATE/BKR.FLASH.STATE/SLOW.INTENSITY")
        except:
            fl1 = None
        try:
            fl2 = self.get_value("FLASH.FEL/XGM.PHOTONFLUX/FL2.TUNNEL/PHOTONFLUX.UJ")
        except:
            fl2 = None
        return [fl1, fl2]

    def get_beam_energy(self):
        try:
            fl1 = self.get_value("FLASH.RF/LLRF.ENERGYGAIN.ML/M2.ACC67/ENERGYGAIN_TOTAL.1")
        except:
            fl1 = None
        try:
            fl2 = self.get_value("FLASH.RF/LLRF.ENERGYGAIN.ML/M2.ACC67/ENERGYGAIN_TOTAL.2")
        except:
            fl2 = None

        return [fl1, fl2]

    def get_wavelength(self):
        try:
            fl1 = self.get_value("TTF2.DAQ/ENERGY.DOGLEG/LAMBDA_MEAN/VAL")
        except:
            fl1 = None
        try:
            fl2 = self.get_value("TTF2.FEEDBACK/FL2.WAVELENGTHCONTROL/FLASH2/WAVELENGTH.USED")
        except:
            fl2 = None

        return [fl1, fl2]

    def get_ref_sase_signal(self):
        try:
            fl1 = self.get_value("TTF2.FEL/BKR.FLASH.STATE/BKR.FLASH.STATE/SLOW.INTENSITY")
        except:
            fl1 = None
        try:
            fl2 = self.get_value("FLASH.FEL/XGM.PHOTONFLUX/FL2.TUNNEL/PHOTONFLUX.UJ")
        except:
            fl2 = None
        #try:
        #    sa3 = self.get_value("XFEL.FEL/XGM.PHOTONFLUX/XGM.3130.T10/WAVELENGTH")
        #except:
        #    sa3 = None
        return [fl1, fl2]

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
        dump2json["obj_times"] = objective_func.times
        dump2json["maximization"] = maximization
        dump2json["nreadings"] = [objective_func.nreadings]
        dump2json["function"] = objective_func.eid
        dump2json["beam_energy"] = self.get_beam_energy()
        dump2json["wavelength"] = self.get_wavelength()
        dump2json["obj_values"] = np.array(objective_func.values).tolist()
        dump2json["std"] = np.array(objective_func.std_dev).tolist()
        try:
            dump2json["ref_sase"] = [objective_func.ref_sase[0], objective_func.ref_sase[-1]]
        except Exception as e:
            print("ERROR. Read ref sase: " + str(e))
            dump2json["ref_sase"] = [None]


        try:
            dump2json["charge"] = [self.get_charge()]
        except Exception as e:
            print("ERROR. Read charge: " + str(e))
            dump2json["charge"] = [None]

        if not os.path.exists(self.path2jsondir):
            os.makedirs(self.path2jsondir)

        filename = os.path.join(self.path2jsondir, datetime.now().strftime("%Y-%m-%d %H-%M-%S") + ".json")
        try:
            with open(filename, 'w') as f:
                json.dump(dump2json, f)
        except Exception as e:
            print("ERROR. Could not write data: " + str(e))
        return True, ""


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
        author = kwargs.get('author', '')
        title = kwargs.get('title', '')
        severity = kwargs.get('severity', '')
        text = kwargs.get('text', '')
        image = kwargs.get('image', None)
        elog = self.logbook_name

        # The DOOCS elog expects an XML string in a particular format. This string
        # is beeing generated in the following as an initial list of strings.
        succeded = True  # indicator for a completely successful job
        # list beginning
        elogXMLStringList = ['<?xml version="1.0" encoding="ISO-8859-1"?>', '<entry>']

        # author information
        elogXMLStringList.append('<author>')
        elogXMLStringList.append(author)
        elogXMLStringList.append('</author>')
        # title information
        elogXMLStringList.append('<title>')
        elogXMLStringList.append(title)
        elogXMLStringList.append('</title>')
        # severity information
        elogXMLStringList.append('<severity>')
        elogXMLStringList.append(severity)
        elogXMLStringList.append('</severity>')
        # text information
        elogXMLStringList.append('<text>')
        elogXMLStringList.append(text)
        elogXMLStringList.append('</text>')
        # image information
        if image:
            try:
                encodedImage = base64.b64encode(image)
                elogXMLStringList.append('<image>')
                elogXMLStringList.append(encodedImage.decode())
                elogXMLStringList.append('</image>')
            except:  # make elog entry anyway, but return error (succeded = False)
                succeded = False
        # list end
        elogXMLStringList.append('</entry>')
        # join list to the final string
        elogXMLString = '\n'.join(elogXMLStringList)
        # open printer process
        try:
            lpr = subprocess.Popen(['/usr/bin/lp', '-o', 'raw', '-d', elog],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            # send printer job
            lpr.communicate(elogXMLString.encode('utf-8'))
        except:
            succeded = False
        return succeded

    def get_obj_function_module(self):
        from mint.xfel import xfel_obj_function
        return xfel_obj_function

    def get_preset_settings(self):
        """
        Return the preset settings to be assembled as Push Buttons at the user interface for quick load of settings.

        :return: (dict) Dictionary with Key being the group name and as value an array of dictionaries following the
        format:
            {"display": "Text of the PushButton", "filename": "my_file.json"}
        """
        presets = {
            "SASE1 opt 1": [
                {"display": "1. Launch orbit", "filename": "sase1_1.json"},
                {"display": "2. Match Quads", "filename": "sase1_2.json"},
                 {"display": "3. SASE1 CAX CAY", "filename": "SASE1_CAX_CAY.json"}],
        "SASE1 opt 2": [
                  {"display": "4. SASE1 CAX CBX", "filename": "SASE1_CAX_CBX.json"},
                {"display": "5. SASE1 phase-shifters", "filename": "SASE1_phase_shifter.json"},
                {"display": "6. SASE1 inj elements", "filename": "SASE1_tuning_with_injector_elements.json"},
            ],
            
            "SASE2 Opt": [
                 {"display": "1. Match Quads", "filename": "SASE2_matching_quads.json"},
                  {"display": "2. AirCoils", "filename": "SASE2_CAX_CBX_CAY_CBY.json"},
                  {"display": "3. Phase-shifters", "filename": "SASE2_BPS.json"},
            ],
            
            "Dispersion Minimization": [
                {"display": "1. I1 Horizontal", "filename": "disp_1.json"},
                {"display": "2. I1 Vertical", "filename": "disp_2.json"},
            ]
        }
        
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
            ("Launch SASE1", ["XFEL.MAGNETS/MAGNET.ML/CFX.2162.T2/CURRENT.SP",
                               "XFEL.MAGNETS/MAGNET.ML/CFX.2219.T2/CURRENT.SP",
                               "XFEL.MAGNETS/MAGNET.ML/CFY.2177.T2/CURRENT.SP",
                               "XFEL.MAGNETS/MAGNET.ML/CFY.2207.T2/CURRENT.SP"]),

            ("Match Quads SASE1", ["XFEL.MAGNETS/MAGNET.ML/CFX.2162.T2/CURRENT.SP",
                                "XFEL.MAGNETS/MAGNET.ML/CFX.2219.T2/CURRENT.SP",
                                "XFEL.MAGNETS/MAGNET.ML/CFY.2177.T2/CURRENT.SP",
                                "XFEL.MAGNETS/MAGNET.ML/CFY.2207.T2/CURRENT.SP"]),
            ("I1 Hor. Disp.", ["XFEL.MAGNETS/MAGNET.ML/CBB.62.I1D/KICK_MRAD.SP",
                                "XFEL.MAGNETS/MAGNET.ML/CIX.90.I1/KICK_MRAD.SP",
                                "XFEL.MAGNETS/MAGNET.ML/CIX.95.I1/KICK_MRAD.SP",
                                "XFEL.MAGNETS/MAGNET.ML/CIX.65.I1/KICK_MRAD.SP",
                                "XFEL.MAGNETS/MAGNET.ML/CIX.51.I1/KICK_MRAD.SP",
                                "XFEL.MAGNETS/MAGNET.ML/CIX.102.I1/KICK_MRAD.SP",
                                "XFEL.MAGNETS/MAGNET.ML/CX.39.I1/KICK_MRAD.SP",
                                "XFEL.MAGNETS/MAGNET.ML/BL.50I.I1/KICK_DEG.SP",
                                "XFEL.MAGNETS/MAGNET.ML/BL.50II.I1/KICK_DEG.SP"]),
            ("I1 Ver. Disp.", ["XFEL.MAGNETS/MAGNET.ML/CIY.92.I1/KICK_MRAD.SP",
                                "XFEL.MAGNETS/MAGNET.ML/CIY.72.I1/KICK_MRAD.SP",
                                "XFEL.MAGNETS/MAGNET.ML/CY.39.I1/KICK_MRAD.SP"])
        ])
        return None
# test interface

