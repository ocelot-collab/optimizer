from __future__ import absolute_import, print_function

import os
import random

from mint.opt_objects import MachineInterface


class DemoInterface(MachineInterface):
    name = 'DemoInterface'

    def __init__(self, args=None):
        super(DemoInterface, self).__init__(args=args)
        self.config_dir = os.path.join(self.config_dir,
                                       "demo")  # <ocelot>/parameters/demo
        self.read_only = False

    def get_value(self, channel):
        print("Called get_value for channel: {}.".format(channel))
        return random.random()

    def set_value(self, channel, val):
        print("Called set_value for channel: {}, with value: {}".format(channel, val))