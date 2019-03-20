from __future__ import absolute_import, print_function

import os
import random

from mint.opt_objects import MachineInterface


class DemoInterface(MachineInterface):
    name = 'DemoInterface'

    def __init__(self, args=None):
        super(DemoInterface, self).__init__(args=args)

        # self.config_dir is path to a directory where a default config will be saved (the tool state)
        # self.config_dir = "<optimizer>/parameters/" is default path in the parent class MachineInterface
        self.config_dir = os.path.join(self.config_dir, "demo")  # <optimizer>/parameters/demo

        # self.path2jsondir is a path to a folder where optimization histories will be saved in json format
        # by default self.path2jsondir = <data> on the same level that <optimizer>
        # the folder will be created automatically

        # flag from LCLSInterface which not allow Optimizer to write to control system
        self.read_only = False

    def get_value(self, channel):
        print("Called get_value for channel: {}.".format(channel))
        return random.random()

    def set_value(self, channel, val):
        print("Called set_value for channel: {}, with value: {}".format(channel, val))
