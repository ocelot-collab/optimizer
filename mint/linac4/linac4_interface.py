# -*- coding: utf-8 -*-

"""
Linac4 machine interface
S.Tomin, I.Agapov 2019
"""
from __future__ import absolute_import, print_function
import sys
import os
import sys
import numpy as np
import subprocess
import base64
from mint.opt_objects import MachineInterface, Device, TestDevice
from collections import OrderedDict
from datetime import datetime
import json
import time
import zmq
import pickle


class Linac4MachineInterface(MachineInterface):
#class Linac4MachineInterface():
    """
    Machine Interface for Linac4
    need ZMQ server to communicate with
    """
    name = 'Linac4MachineInterface'

    def __init__(self, args=None):
        super(Linac4MachineInterface, self).__init__(args)

        self.logbook_name = ""

        path2root = os.path.abspath(os.path.join(__file__ , "../../../.."))
        self.config_dir = os.path.join(path2root, "config_optim")
        self.timestamp = time.time()

        self.connect2server()
    
    def connect2server(self):
        self.context = zmq.Context()
        print("Conncting to server...")
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5556")
        
    def get_value(self, channel):
        """
        Getter function for XFEL.

        :param channel: (str) String of the devices name used in doocs
        :return: Data from pydoocs.read(), variable data type depending on channel
        """

        obj = {"cmd": "get", "name": channel}
        b = pickle.dumps(obj)
        self.socket.send(b)
        b = self.socket.recv()
        obj2 = pickle.loads(b)
        return obj2["val"]

    def set_value(self, channel, val):
        """
        Method to set value to a channel

        :param channel: (str) String of the devices name used in doocs
        :param val: value
        :return: None
        """
        obj = {"cmd": "set", "name": channel, "val": val}
        self.socket.send(pickle.dumps(obj))
        b = self.socket.recv()
        obj2 = pickle.loads(b)
        return
    
    def get_obj_function_module(self):
        from mint.linac4 import linac4_obj_function
        return linac4_obj_function


# test interface
if __name__ == '__main__':
    print('started...')
    mi = Linac4MachineInterface()  
    print('the end...')
    

