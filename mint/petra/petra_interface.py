"""
PETRA III machine interface (tine)
"""
from __future__ import absolute_import, print_function


try:
    import PyTine as pt
except ImportError:
    # Ignore the error since maybe no one is trying to use it... we will raise on the ctor.
    pass


import sys
import numpy as np
import subprocess
import base64
import os
try:
    from mint.opt_objects import MachineInterface, Device
except:
    from opt_objects import MachineInterface, Device

# /LINAC2/Bunche_L2/IMA-L23/BunchParticlesE9.SCH/
class AlarmDevice(Device):
    """
    Devices for getting information about Machine status
    """
    def __init__(self, eid=None):
        super(AlarmDevice, self).__init__(eid=eid)


class PETRAMachineInterface(MachineInterface):
    """
    main class
    """
    name = 'PETRAMachineInterface'
    def __init__(self, args=None):
        print('initializing PETRA interface...')
        super(PETRAMachineInterface, self).__init__(args)
        if 'PyTine' not in sys.modules:
            print('error importing tine library')
        self.logbook_name = "petralog"
        print("PETRA INTERFCE")
        path2root = os.path.abspath(os.path.join(__file__ , "../../../.."))
        self.config_dir = os.path.join(path2root, "config_optim_new")

    def parse_read_value(self,ch, kwd, val):
        if kwd.find('attSolidState') >=0:
            return val[0]
        if ch.find('RFModulator') >=0:
            return val[0][0]
        return val

    def parse_write_value(self,ch, kwd, val):
        if kwd.find('attSolidState') >=0:
            return [val]
        if ch.find('RFModulator') >=0:
            return [(val,0)]
        return val


    def get_value(self, channel):
        """
        Getter function for XFEL.

        :param channel: (str) String of the devices name used in doocs
        :return: Data from pydoocs.read(), variable data type depending on channel
        """
        kwd = channel.split('/')[-1]
        idx = len(kwd)+1
        ch = channel[0:-idx]
        val = pt.get(ch, kwd)
        return self.parse_read_value(ch,kwd,val["data"])

    def get_obj_function_module(self):
        from mint.petra import petra_obj_function
        return petra_obj_function
    

    def set_value(self, channel, val):
        """
        Method to set value to a channel

        :param channel: (str) String of the devices name used in doocs
        :param val: value
        :return: None
        """
        #print("SETTING")
        kwd = channel.split('/')[-1]
        idx = len(kwd)+1
        ch = channel[0:-idx]
        #val = pt.get(ch, kwd)
        print(ch,kwd,val)
        pt.set(ch, kwd, self.parse_write_value(ch, kwd, val) )
        return



    def get_charge(self):
        return sum(self.get_value('/PETRA/ARCHIVER/keyword/CurBunch'))


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

