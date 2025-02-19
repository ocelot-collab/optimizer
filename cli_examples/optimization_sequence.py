#from optimizer import *
import sys
sys.path.append("/home/xfeloper/user/tomins/ocelot_new/optimizer")
from mint.opt_objects import *
from mint.xfel.xfel_interface import *
from mint.mint import *
from op_methods.simplex import *
import matplotlib.pyplot as plt
import numpy as np

import logging
logging.basicConfig(level=logging.DEBUG)

# init machine interface
mi = TestMachineInterface(args=None)

# timeout between device setting and signal reading
timeout = 0.1

# type of optimization
maximization = True

# max iteration
max_iter = 35


# redefine method get_value for Target.
"""
# it is a real example which will work with TestMachineInterface and XFELMachineInterface
def get_value():
    val = mi.get_value("XFEL.FEL/XGM.PREPROCESSING/XGM.2595.T6.CH0/RESULT.TD")
    return val

target = Target()
target.mi = mi
target.nreadings = 35 # number of readings
target.interval = 0.01 # in [s] between readings
target.get_value = get_value

target2 = Target()
target.mi = mi
target2.nreadings = 35
target2.interval = 0.01 # in sec between readings
target2.get_value = get_value
"""
# to be able to run "real" optimization we can use simple test target class
class Target_test(Target):
    def __init__(self, mi=None, eid=None):
        super(Target_test, self).__init__(eid=eid)
        self.mi = mi

    def get_value(self):
        values = np.array([dev.get_value() for dev in self.devices])
        return np.sum(np.exp(-np.power((values - np.ones_like(values)), 2) / 5.))

target = Target_test()
target.mi = mi
target.nreadings = 35 # number of readings
target.interval = 0.01 # in [s] between readings

target2 = Target_test()
target.mi = mi
target2.nreadings = 35
target2.interval = 0.01 # in sec between readings


# create devices for ACTION 1
pvs = [
"XFEL.FEL/UNDULATOR.SASE2/CBY.CELL16.SA2/FIELD.OFFSET", 
"XFEL.FEL/UNDULATOR.SASE2/CAX.CELL17.SA2/FIELD.OFFSET", 
"XFEL.FEL/UNDULATOR.SASE2/CBX.CELL17.SA2/FIELD.OFFSET", 
"XFEL.FEL/UNDULATOR.SASE2/CAY.CELL17.SA2/FIELD.OFFSET", ]

devices1 = []
ivalues = []
dev_steps = []
for pv in pvs:
    d = mi.device_factory(pv=pv)
    d.mi = mi
    d.get_limits = lambda : [-1, 1]
    d.istep = 0.2                   # initial step
    devices1.append(d)

# if Action's argument "func = None" optimization uses function Optimizer.max_target_func()
a1 = Action(func=None, args=[target, devices1])
# for logging
a1.finalize = lambda : mi.write_data(method_name="simplex", objective_func=target, devices=devices1,
                                     maximization=maximization, max_iter=max_iter)


# create devices for ACTION 2
pvs = [
"XFEL.FEL/UNDULATOR.SASE2/CBY.CELL19.SA2/FIELD.OFFSET", 
"XFEL.FEL/UNDULATOR.SASE2/CAX.CELL20.SA2/FIELD.OFFSET", 
"XFEL.FEL/UNDULATOR.SASE2/CBX.CELL20.SA2/FIELD.OFFSET", 
"XFEL.FEL/UNDULATOR.SASE2/CAY.CELL20.SA2/FIELD.OFFSET" ]

devices2 = []
for pv in pvs:
    d = mi.device_factory(pv=pv)
    d.mi = mi
    d.get_limits = lambda : [-1, 1]
    d.istep = 0.2                   # initial step
    devices2.append(d)


a2 = Action(func=None, args=[target2, devices2])
a2.finalize = lambda : mi.write_data("simplex", target2, devices2, maximization, max_iter)

# sequence of optimizations
seq = [a1, a2]

# set Machine protection
alarm_dev = mi.device_factory(pv="XFEL.DIAG/TOROID/TORA.1865.TL/CHARGE.ALL")
alarm_dev.mi = mi

m_status = MachineStatus()
m_status.alarm_device = alarm_dev
m_status.alarm_max = 1
m_status.alarm_min = -1


# init Minimizer
minimizer = Simplex()
minimizer.max_iter = max_iter

# init Optimizer
opt = Optimizer()
opt.maximization = maximization
opt.timeout = timeout

opt.opt_ctrl = OptControl()
opt.opt_ctrl.m_status = m_status
opt.opt_ctrl.timeout = 5 # if machine runs again after interruption wait 5 sec before continue optimization
opt.minimizer = minimizer

# run optimizations
opt.eval(seq)

#%%
# plotting
t = np.array(target.times) - target.times[0]
v = target.values
v_std = target.std_dev

t2 = np.array(target2.times) - target2.times[0] + (target.times[-1] - target.times[0])
v2 = target2.values
v_std2 = target2.std_dev

plt.subplot(311)
plt.plot(t, v, lw=2, label="obj func")
plt.plot(t2, v2, lw=2,  label="obj func")
plt.legend()

plt.subplot(312)
plt.plot(t, v_std, lw=2, label="obj func std")
plt.plot(t2, v_std2, lw=2, label="obj func std")
plt.legend()

plt.subplot(313)
for dev in devices1:
    t = np.array(dev.times) - dev.times[0]
    v = dev.values
    plt.plot(t, v, label=dev.id)

for dev in devices2:
    t = np.array(dev.times) - dev.times[0] + devices1[0].times[-1] - devices1[0].times[0]
    v = dev.values
    plt.plot(t, v, label=dev.id)
plt.legend()
plt.show()
#mi.write_data("simplex", target, devices, opt.maximization, minimizer.max_iter)