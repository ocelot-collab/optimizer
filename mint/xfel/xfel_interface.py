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
from mint.opt_objects import MachineInterface, Device
from collections import OrderedDict

class AlarmDevice(Device):
    """
    Devices for getting information about Machine status
    """
    def __init__(self, eid=None):
        super(AlarmDevice, self).__init__(eid=eid)


class XFELMachineInterface(MachineInterface):
    """
    Machine Interface for European XFEL
    """
    name = 'XFELMachineInterface'

    def __init__(self, args=None):
        super(XFELMachineInterface, self).__init__(args)
        if 'pydoocs' not in sys.modules:
            print('error importing doocs library')
        self.logbook_name = "xfellog"

        path2root = os.path.abspath(os.path.join(__file__ , "../../../.."))
        self.config_dir = os.path.join(path2root, "config_optim_new")
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
        return self.get_value("XFEL.DIAG/CHARGE.ML/TORA.25.I1/CHARGE.SA1")

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
            "SASE Optimization": [
                {"display": "1. Launch orbit SASE1", "filename": "sase1_1.json"},
                {"display": "2. Match Quads SASE1", "filename": "sase1_2.json"},
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
        return devs
# test interface


class TestMachineInterface(XFELMachineInterface):
    """
    Machine interface for testing
    """
    name = 'TestMachineInterface'

    def __init__(self, args):
        super(TestMachineInterface, self).__init__(args)
        self.data = 1.
        pass

    def get_alarms(self):
        return np.random.rand(4)#0.0, 0.0, 0.0, 0.0]

    def get_value(self, device_name):
        """
        Testing getter function for XFEL.

        :param channel: (str) String of the devices name used in doocs
        :return: Data from pydoocs.read(), variable data type depending on channel
        """
        #if "QUAD" in device_name:
        #    return 0
        return np.random.rand(1)[0]-0.5 #self.data

    def set_value(self, device_name, val):
        """
        Testing Method to set value to a channel

        :param channel: (str) String of the devices name used in doocs
        :param val: value
        :return: None
        """
        #print("set:", device_name,  "-->", val)
        self.data += np.sqrt(val**2)
        return 0.0

    def get_bpms_xy(self, bpms):
        """
        Testing method for getting bmps data

        :param bpms: list of string. BPMs names
        :return: X, Y - two arrays in [m]
        """
        X = np.zeros(len(bpms))
        Y = np.zeros(len(bpms))
        return X, Y

    def get_charge(self):
        return 0

    @staticmethod
    def send_to_logbook(*args, **kwargs):
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
        elog = kwargs.get('elog', '')
        image = kwargs.get('image', None)

        # TODO: @sergey.tomin Figure out what to do for logbook at the TestMachineInterface
        print('Send to Logbook not implemented for TestMachineInterface.')
        return True

    def get_obj_function_module(self):
        from mint.xfel import xfel_obj_function
        return xfel_obj_function


