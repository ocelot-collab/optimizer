from __future__ import absolute_import, print_function
import os
import numpy as np
from collections import OrderedDict

from mint.opt_objects import MachineInterface
from sint.multinormal.multinormal_devices import MultinormalDevice
from sint.multinormal.multinormal_ui import MultinormalDisplay

# Fix Python 2.x.
try:
    UNICODE_EXISTS = bool(type(unicode))
except NameError:
    unicode = str

class MultinormalInterface(MachineInterface):
    name = "MultinormalInterface"

    def __init__(self, args=None):
        super(MultinormalInterface, self).__init__(args)
        self.config_dir = os.path.join(self.config_dir,
                                       "multinormal")  # <ocelot>/parameters/lcls
        self.points = 1
        self._use_num_points = True

        self.ebeam_energy = 7.  # GeV

        self.losspvs = []

        self.simmode = 1  # 0: multinormal distribution
        # 1: correlation plot fit

        self.ndims = args.get('ndims', 10)
        params = self.setup_params(self.ndims)

        self.y = -1
        self.mean = 0
        self.stdev = 0
        self.stdev_nsample = 0

        self.sigAmp = 1.
        self.bgNoise = args.get('bgNoise', 0.064)  # something typical
        self.sigNoiseScaleFactor = args.get('sigNoiseScaleFactor', 0.109)  # seems like something typical is amp_noise / sqrt(amp_signal) ~= 0.193/np.sqrt(3.113) = 0.109
        self.noiseScaleFactor = args.get('noiseScaleFactor', 1.)  # easy to use this as a noise toggle

        self.numBatchSamples = 1.
        self.SNRgoal = 0  # specify SNR goal; if 0, then numSamples is unchanged
        self.maxNumSamples = 30. * 120.  # a limit on how long people would want to wait if changing numSamples to achieve the SNRgoal

        self.last_numSamples = self.points
        self.last_SNR = self.SNRgoal

        # making this its own function in case we want to call again later
        self.store_moments(params[0], params[1], params[2])

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

        parser.add_argument('--ndims', default=10, type=int, required=False,
                            help="Number of dimensions for the correlation matrix.")
        parser.add_argument('--bgNoise', default=0.064, type=float,
                            required=False,
                            help="Background noise.")
        parser.add_argument('--sigNoiseScaleFactor', default=0.109, type=float,
                            required=False,
                            help="Signal to noise scale factor.")
        parser.add_argument('--noiseScaleFactor', default=1.0, type=float,
                            required=False,
                            help='Noise scale factor. Easy to use this as a noise toggle')

    def setup_params(self, ndims):
        # these set the statistical properties of the
        offset_nsigma = 2.  # scales the magnitude of the distance between start and goal so that the distance has a zscore of nsigma
        offsets = np.random.randn(
            ndims)
        offsets = np.round(offsets * offset_nsigma / np.linalg.norm(offsets),
                           2)  # 1.*np.ones(self.sim_ndim) # peak location is an array
        projected_widths = np.ones(
            ndims)  # widths of the marginalized distributions
        correlation_matrix = np.diag(
            np.ones(ndims))  # correlations between coordinates

        return offsets, projected_widths, correlation_matrix

    def customize_ui(self, gui):
        """
        Method invoked to modify the UI and apply customizations pertinent to the
        Machine Interface

        :param gui: (MainWindow) The application Main Window
        :return: None
        """
        gui.hyper_file = "devmode"
        gui.ui.pb_hyper_file.setText(gui.hyper_file)

        # Seed File
        gui.ui.lineEdit_4.setText("parameters/simSeed.mat")
        self.display_tab = MultinormalDisplay(parent=gui, mi=self)
        tab_widget = gui.ui.tabWidget
        tab_widget.addTab(self.display_tab, "Simulation Mode")

        # Clears the devices list
        gui.ui.widget.pvs = []
        gui.ui.widget.devices = []
        gui.ui.widget.ui.tableWidget.setRowCount(0)

        # Add the default devices to the UI
        pvs = ["sim_device_{}".format(i + 1) for i in range(self.ndims)]
        for pv in pvs:
            gui.ui.widget.addPv(pvs, force_active=True)

    def get_obj_function_module(self):
        from sint.multinormal import multinormal_obj_function
        return multinormal_obj_function

    def device_factory(self, pv):
        d = MultinormalDevice(eid=pv)
        return d

    def get_plot_attrs(self):
        """
        Returns a list of tuples in which the first element is the attributes to be fetched from Target class
        to present at the Plot 1 and the second element is the label to be used at legend.

        :return: (list) Attributes from the Target class to be used in the plot.
        """
        return [("values", "statistics"), ("objective_means", "mean")]

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
            ("All", ["sim_device_{}".format(i+1) for i in range(self.ndims)])
        ])
        return devs

    def get_charge_current(self):
        charge = np.nan
        current = np.nan
        return charge, current

    def get_losses(self):
        losses = []
        return losses

    def get_value(self, variable_names):
        val = self.get1(variable_names)
        return val

    def set_value(self, variable_names, values):
        self.set1(variable_names, values)

    def get_energy(self):
        return self.ebeam_energy

    # simple access fcn
    def get1(self, pvname):
        index = self.pvdict[pvname]
        if index == len(self.pvs) - 1:
            self.f(self.x)
            if hasattr(self.y, '__iter__'):
                return self.y[0]
            else:
                return self.y
        else:
            return self.x[-1, index]

    # simple access fcn
    def set1(self, pvname, value):
        index = self.pvdict[pvname]
        if index == len(self.pvs) - 1:
            self.y = value
        else:
            self.x[-1, index] = value

    def store_moments(self, offsets, projected_widths, correlation_matrix):
        # check sizes
        if offsets.size != projected_widths.size or offsets.size != np.sqrt(
                correlation_matrix.size):
            print("MultinormalInterface - ERROR: Dimensions of input parameters are inconsistant.")

        # store inputs
        self.offsets = offsets  # list of peak location coords
        self.sigmas = np.abs(
            projected_widths)  # list of peak widths projected to each axis
        self.corrmat = correlation_matrix  # correlation matrix of peaks
        self.covarmat = np.dot(np.diag(self.sigmas), np.dot(self.corrmat,
                                                            np.diag(
                                                                self.sigmas)))  # matrix of covariances
        self.invcovarmat = np.linalg.inv(
            self.covarmat)  # inverse of covariance matrix (computed once and stored)

        # seems like in fint, he wants to store the last random number generated by the last fcn call so let's store some random number
        self.x = np.array(np.zeros(self.offsets.size), ndmin=2)

        # reference for goal
        self.pvs_optimum_value = np.array([self.offsets, 1.])
        self.detector_optimum_value = self.pvs_optimum_value[-1]

        # name these PVs
        self.pvs = np.array(["sim_device_" + str(i) for i in
                             np.array(range(self.offsets.size)) + 1])
        self.detector = "sim_objective"
        self.pvs = np.append(self.pvs, self.detector)
        self.pvdict = dict()  # for simple lookup
        for i in range(len(self.pvs)):
            self.pvdict[self.pvs[i]] = i  # objective fcn is last here

    def fmean(self, x_new):  # to calculate ground truth
        self.x = x_new
        self.dx = self.x - self.offsets

        self.mean = self.sigAmp * np.exp(
            -0.5 * np.dot(self.dx, np.dot(self.invcovarmat, self.dx.T)))

        return self.mean

    def f(self, x_new):
        # let this method update means and stdevs

        self.x = x_new
        self.dx = self.x - self.offsets

        # set result mean (perturb by noise below)
        self.mean = abs(self.sigAmp * np.exp(
            -0.5 * np.dot(self.dx, np.dot(self.invcovarmat, self.dx.T))))

        # set resulting 1 sample noise (nsample noise below)
        self.stdev = np.abs(self.bgNoise) + np.abs(
            self.sigNoiseScaleFactor) * np.sqrt(self.mean)
        self.stdev = abs(self.noiseScaleFactor) * self.stdev

        # figure out the number of samples needed to achieve the SNRgoal
        if (self.SNRgoal > 0 and self.noiseScaleFactor > 0):
            # analytic way
            self.points = min(
                [(self.SNRgoal * self.stdev / self.mean) ** 2.,
                 self.maxNumSamples])
            self.points = max(
                [np.ceil(self.points / self.numBatchSamples),
                 1.]) * self.numBatchSamples
            self.points = self.points  # for compatibility with machine interface api

        # perturb mean by nsample noise
        # self.y = np.array(
        #     [self.mean + np.random.normal(0., self.stdev, self.mean.shape)
        #      for i in range(int(self.points))])
        self.y = np.random.normal(self.mean[0][0], self.stdev[0][0], self.points)

        return np.array(self.y, ndmin=2)

    def SNR(self):
        return self.mean / self.stdev_nsample

    # =======================================================#
    # ------------------- Data Saving --------------------- #
    # =======================================================#
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
        import mint.lcls.simlog as matlog

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
        self.data["BEND_DMP1_400_BDES"] = self.get_energy()
        self.data["Energy"] = self.get_energy()
        self.data["ScanAlgorithm"] = str(method_name)  # string of the algorithm name
        self.data["ObjFuncPv"] = str(objective_func_pv)  # string identifing obj func pv
        self.data['DetectorMean'] = str(
            objective_func_pv.replace(":", "_"))  # reminder to look at self.data[objective_func_pv]
        # TODO: Ask Joe if this is really needed...
        #self.data["NormAmpCoeff"] = norm_amp_coeff
        self.data["niter"] = objective_func.niter

        # save data
        self.last_filename = matlog.save("OcelotScan", removeUnicodeKeys(self.data), path='default')  # self.save_path)

        print('Saved scan data to ', self.last_filename)

        # clear for next run
        self.data = dict()

        return True, ""
