#!/opt/anaconda4/bin/python
"""
This is deep modification of SLAC version of the Ocelot GUI for the European XFEL facility.
Sergey Tomin, 2017.
Ocelot GUI, interface for running and testing accelerator optimization methods
This file primarily contains the code for the UI and GUI
The scanner classes are contained in an external file, scannerThreads.py
The resetpanel widget is also contained in a separate module, resetpanel
Tyler Cope, 2016
"""
from __future__ import absolute_import, print_function
import sys
import os
import argparse
import sklearn
import functools
import inspect

sklearn_version = sklearn.__version__

path = os.path.realpath(__file__)
#indx = path.find("ocelot/optimizer")
print("PATH", os.path.realpath(__file__))
#sys.path.append(path[:indx])

# for pyqtgraph import
#sys.path.append(path[:indx]+"ocelot")

from PyQt5.QtWidgets import (QApplication, QFrame, QGroupBox, QLabel, QComboBox, QPushButton, QSpacerItem,
                             QVBoxLayout, QDesktopWidget)
import platform
import pyqtgraph as pg
if sys.version_info[0] == 2:
    from imp import reload
else:
    from importlib import reload


from gui_main import *

from mint.opt_objects import *
from mint import mint
from mint import opt_objects as obj

from mint.xfel_interface import *
from mint.lcls.lcls_interface import *
from stats import stats


class OcelotInterfaceWindow(QFrame):
    """ Main class for the GUI application """
    def __init__(self):
        """
        Initialize the GUI and QT UI aspects of the application.
        Initialize the scan parameters.
        Connect start and logbook buttons on the scan panel.
        Initialize the plotting.
        Make the timer object that updates GUI on clock cycle during a scan.
        """
        # PATHS
        self.plot1_curves = dict()
        self.optimizer_args = None
        self.parse_arguments()
        self.dev_mode = self.optimizer_args.devmode

        if self.dev_mode:
            self.mi = TestMachineInterface()
        else:
            class_name = self.optimizer_args.mi
            if class_name not in globals():
                print("Could not find Machine Interface with name: {}. Loading XFELMachineInterface instead.".format(class_name))
                self.mi = XFELMachineInterface()
            else:
                self.mi = globals()[class_name]()

        self.config_dir = self.mi.config_dir
        self.path2preset = os.path.join(self.config_dir, "standard")
        self.set_file = os.path.join(self.config_dir, "default.json")  # ./parameters/default.json"
        self.obj_save_path = os.path.join(self.config_dir, "obj_funcs")

        # initialize
        QFrame.__init__(self)

        self.ui = MainWindow(self)

        self.name_simplex = "Nelder-Mead Simplex"
        self.name_gauss = "Gaussian Process"
        self.name_gauss_sklearn = "Gaussian Process sklearn"
        self.name_custom = "Custom Minimizer"
        self.name_simplex_norm = "Simplex Norm."
        self.name_es = "Extremum Seeking"
        # self.name4 = "Conjugate Gradient"
        # self.name5 = "Powell's Method"
        # switch of GP and custom Mininimizer
        self.ui.cb_select_alg.addItem(self.name_simplex)
        #self.ui.cb_select_alg.addItem(self.name_gauss)
        self.ui.cb_select_alg.addItem(self.name_custom)
        self.ui.cb_select_alg.addItem(self.name_simplex_norm)
        #self.ui.cb_select_alg.addItem(self.name_es)
        if sklearn_version >= "0.18":
            self.ui.cb_select_alg.addItem(self.name_gauss_sklearn)


        #self.ui.pb_help.clicked.connect(lambda: os.system("firefox file://"+self.optimizer_path+"docs/build/html/index.html"))
        self.ui.pb_help.clicked.connect(self.ui.open_help)

        self.assemble_preset_box()
        self.assemble_quick_add_box()

        if not self.mi.use_num_points():
            self.ui.label_datapoints.setVisible(False)
            self.ui.sb_datapoints.setVisible(False)

        self.total_delay = self.ui.sb_tdelay.value()

        self.objective_func_pv = "test_obj"

        self.addPlots()

        # database

        self.scan_params = None
        self.hyper_file = "../parameters/hyperparameters.npy"

        self.ui.pb_start_scan.clicked.connect(self.start_scan)
        self.ui.pb_edit_obj_func.clicked.connect(self.run_editor)
        self.ui.cb_use_predef.stateChanged.connect(self.set_obj_fun)

        # fill in statistics methods
        self.ui.cb_statistics.clear()
        for st in stats.all_stats:
            self.ui.cb_statistics.addItem(st.display_name, st)

        self.ui.cb_statistics.currentIndexChanged.connect(self.statistics_select)
        self.ui.cb_statistics.setCurrentIndex(0)

        self.ui.restore_state(self.set_file)

        obj_func_file = self.mi.get_obj_function_module().__file__
        self.path_to_obj_func = obj_func_file if not obj_func_file.endswith("pyc") else obj_func_file[:-1]

        self.set_obj_fun()
        self.m_status = mint.MachineStatus()
        self.set_m_status()

        self.opt_control = mint.OptControl()
        self.opt_control.alarm_timeout = self.ui.sb_alarm_timeout.value()
        self.opt_control.m_status = self.m_status

        #timer for plots, starts when scan starts
        self.multiPvTimer = QtCore.QTimer()
        self.multiPvTimer.timeout.connect(self.getPlotData)


    def parse_arguments(self):
        parser = argparse.ArgumentParser(description="Ocelot Optimizer")
        parser.add_argument('--devmode', action='store_true',
                            help='Enable development mode.', default=False)
        parser.add_argument('--mi', help="Which Machine Interface to use. Defaults to XFELMachineInterface.", default="XFELMachineInterface")
        self.optimizer_args = parser.parse_args()

    def statistics_select(self, value):
        if self.objective_func is not None:
            self.objective_func.stats = self.ui.cb_statistics.currentData()

    def scan_method_select(self):
        """
        Sets scanner method from options panel combo box selection.
        This method executes from the runScan() method, when the UI "Start Scan" button is pressed.
        :return: Selected scanner object
                 These objects are contrained in the scannerThreads.py file
        """
        current_method = self.ui.cb_select_alg.currentText()

        #GP Method
        if current_method == self.name_gauss:
            minimizer = mint.GaussProcess()

        elif current_method == self.name_gauss_sklearn:
            minimizer = mint.GaussProcessSKLearn()
            minimizer.seed_iter = self.ui.sb_seed_iter.value()
        # Custom Minimizer
        elif current_method == self.name_custom:
            minimizer = mint.CustomMinimizer()

        elif current_method == self.name_simplex_norm:
            minimizer = mint.Simplex()
        elif current_method == self.name_es:
            minimizer = mint.ESMin()
        #simplex Method
        else:
            minimizer = mint.Simplex()

        self.method_name = minimizer.__class__.__name__

        return minimizer

    def closeEvent(self, event):
        self.ui.save_state(self.set_file)
        if self.ui.pb_start_scan.text() == "Stop optimization":
            self.opt.opt_ctrl.stop()
            self.m_status.is_ok = lambda: True
            del(self.opt)
            self.ui.pb_start_scan.setStyleSheet("color: rgb(85, 255, 127);")
            self.ui.pb_start_scan.setText("Start optimization")
        QFrame.closeEvent(self, event)

    def start_scan(self):
        """
        Method to start/stop the Optimizer.
        """
        self.scanStartTime = time.time()


        if self.ui.pb_start_scan.text() == "Stop optimization":
            # stop the optimization
            self.opt.opt_ctrl.stop()

            self.m_status.is_ok = lambda: True
            # Save the optimization parameters to the database
            try:
                ret, msg = self.save2db()
                if not ret:
                    self.error_box(message=msg)
            except Exception as ex:
                print("ERROR start_scan: can not save to db. Exception was: ", ex)
            del(self.opt)
            # Setting the button
            self.ui.pb_start_scan.setStyleSheet("color: rgb(85, 255, 127);")
            self.ui.pb_start_scan.setText("Start optimization")
            return 0

        self.pvs = self.ui.widget.getPvsFromCbState()
        self.devices = self.ui.widget.get_devices(self.pvs)

        if len(self.devices) == 0:
            self.error_box(message="Check Devices")
            return 0
        for dev in self.devices:
            val = dev.get_value()
            if dev.check_limits(val):
                self.error_box(message="Check the Limits")
                return 0
        self.setUpMultiPlot(self.devices)
        self.multiPvTimer.start(100)

        # set the Objective function from GUI or from file mint.obj_function.py (reloading)
        self.set_obj_fun()

        self.objective_func_pv = self.objective_func.eid

        if self.mi.use_num_points():
            self.objective_func.points = self.ui.sb_datapoints.value()

        self.updatePlotLabels()
        # Set minimizer - the optimization method (Simplex, GP, ...)
        minimizer = self.scan_method_select()

        # configure the Minimizer
        if minimizer.__class__ in [mint.GaussProcess, mint.GaussProcessSKLearn]:
            minimizer.seed_iter = self.ui.sb_seed_iter.value()
            minimizer.seed_timeout = self.ui.sb_tdelay.value()
            minimizer.hyper_file = self.hyper_file
            minimizer.norm_coef = self.ui.sb_isim_rel_step.value()/ 100.

        elif minimizer.__class__ == mint.Simplex:
            if self.ui.cb_use_isim.checkState():
                minimizer.dev_steps = []

                for dev in self.devices:
                    if dev.simplex_step == 0:
                        lims = dev.get_limits()
                        rel_step = self.ui.sb_isim_rel_step.value()
                        d_lims = lims[1] - lims[0]
                        # set the same step as for pure Simplex if delta lims is zeros
                        if np.abs(d_lims) < 2e-5:
                            val0 = dev.get_value()
                            if np.abs(val0) < 1e-8:
                                step = 0.00025
                            else:
                                step = val0*0.05
                            minimizer.dev_steps.append(step)
                        else:
                            minimizer.dev_steps.append(d_lims * rel_step / 100.)
            else:
                minimizer.dev_steps = None

        elif minimizer.__class__ == mint.CustomMinimizer:
            minimizer.dev_steps = []

            for dev in self.devices:
                if dev.simplex_step == 0:
                    lims = dev.get_limits()
                    rel_step = self.ui.sb_isim_rel_step.value()
                    print(dev.id, rel_step)
                    minimizer.dev_steps.append((lims[1] - lims[0]) * rel_step / 100.)
            print("MINImizer steps", minimizer.dev_steps)

        self.max_iter = self.ui.sb_num_iter.value()
        minimizer.max_iter = self.max_iter

        # Optimizer initialization
        self.opt = mint.Optimizer()

        # solving minimization or maximization problem
        self.opt.maximization = self.ui.rb_maximize.isChecked()

        if self.ui.cb_select_alg.currentText() == self.name_simplex_norm:
            self.opt.normalization = True
            self.opt.norm_coef = self.ui.sb_isim_rel_step.value()*0.01
            print("OPT", self.opt.norm_coef)
        # Option - set best solution after optimization or not
        self.opt.set_best_solution = self.ui.cb_set_best_sol.checkState()

        self.set_m_status()
        self.opt_control.m_status = self.m_status
        self.opt_control.clean()
        self.opt.opt_ctrl = self.opt_control
        self.opt.timeout = self.total_delay

        self.opt.minimizer = minimizer

        seq = [mint.Action(func=self.opt.max_target_func, args=[self.objective_func, self.devices])]
        self.opt.seq = seq

        #self.opt.eval(seq)
        self.opt.start()

        # Setting the button
        self.ui.pb_start_scan.setText("Stop optimization")
        self.ui.pb_start_scan.setStyleSheet("color: red")

    def scan_finished(self):
        try:
            if self.ui.pb_start_scan.text() == "Stop optimization" and not (self.opt.isAlive()):
                self.ui.pb_start_scan.setStyleSheet("color: rgb(85, 255, 127);")
                self.ui.pb_start_scan.setText("Start optimization")
                ret, msg = self.save2db()
                if not ret:
                    self.error_box(message=msg)
                print("scan_finished: OK")
        except Exception as ex:
            print("scan_finished: ERROR. Exception was: ", ex)

    def save2db(self):
        # first try to gather minimizer data
        try:
            self.opt.minimizer.saveModel()  # need to save GP model first
        except:
            pass

        if self.mi is not None:
            method_name = self.method_name
            obj_func = self.objective_func
            devices = self.devices
            maximization = self.ui.rb_maximize.isChecked()
            max_iter = self.max_iter
            return self.mi.write_data(method_name, obj_func, devices, maximization, max_iter)
        else:
            return False, "Machine Interface is not defined."

    def create_devices(self, pvs):
        """
        Method to create devices using only channels (PVs)

        :param pvs: str, device address/channel/PV
        :return: list of the devices [mint.opt_objects.Device(eid=pv[0]), mint.opt_objects.Device(eid=pv[1]), ... ]
        """
        # TODO: add new method for creation of devices
        devices = []
        for pv in pvs:
            if self.dev_mode:
                dev = obj.TestDevice(eid=pv)
            else:
                dev = self.mi.device_factory(pv=pv)
            dev.mi = self.mi
            devices.append(dev)
        return devices

    def indicate_machine_state(self):
        """
        Method to indicate of the machine status. Red frames around graphics means that machine status is not OK.
        :return:
        """
        if not self.opt_control.is_ok:
            if self.ui.widget_3.styleSheet() == "background-color:red;":
                return
            self.ui.widget_2.setStyleSheet("background-color:red;")
            self.ui.widget_3.setStyleSheet("background-color:red;")

        else:
            if self.ui.widget_3.styleSheet() == "background-color:323232;":
                return
            self.ui.widget_2.setStyleSheet("background-color:323232;")
            self.ui.widget_3.setStyleSheet("background-color:323232;")

    def set_obj_fun(self):
        """
        Method to set objective function from the GUI (channels A,B,C) or reload module obj_function.py

        :return: None
        """
        try:
            obj_function_module = self.mi.get_obj_function_module()
            reload(obj_function_module)
            self.ui.pb_edit_obj_func.setStyleSheet("background: #4e4e4e")
        except Exception as ex:
            self.ui.pb_edit_obj_func.setStyleSheet("background: red")
            self.ui.cb_use_predef.setCheckState(False)
            print("ERROR set objective function. Exception was: ", ex)

        self.ui.use_predef_fun()

        if self.ui.cb_use_predef.checkState():
            print("RELOAD Module Objective Function")
            obj_function_module = self.mi.get_obj_function_module()
            if 'target_class' in dir(obj_function_module):
                tclass = obj_function_module.target_class
            else:
                tclass = [obj for name, obj in inspect.getmembers(obj_function_module) if
                            inspect.isclass(obj) and issubclass(obj, Target) and obj != Target][0]

            print("Target Class: ", tclass)
            self.objective_func = tclass(mi=self.mi)
            self.objective_func.devices = []
            self.objective_func.stats = self.ui.cb_statistics.currentData()
        else:
            # disable button "Edit Objective Function"
            # self.ui.pb_edit_obj_func.setEnabled(False)
            line_edits = [self.ui.le_a, self.ui.le_b, self.ui.le_c, self.ui.le_d, self.ui.le_e]

            a_str = str(self.ui.le_a.text())
            state_a = self.ui.is_le_addr_ok(self.ui.le_a)

            b_str = str(self.ui.le_b.text())
            state_b = self.ui.is_le_addr_ok(self.ui.le_b)

            c_str = str(self.ui.le_c.text())
            state_c = self.ui.is_le_addr_ok(self.ui.le_c)

            d_str = str(self.ui.le_d.text())
            state_d = self.ui.is_le_addr_ok(self.ui.le_d)

            e_str = str(self.ui.le_e.text())
            state_e = self.ui.is_le_addr_ok(self.ui.le_e)

            func = str(self.ui.le_obf.text())

            def get_value_exp():
                A = 0.
                B = 0.
                C = 0.
                D = 0.
                E = 0.
                if state_a:
                    A = self.mi.get_value(a_str)
                if state_b:
                    B = self.mi.get_value(b_str)
                if state_c:
                    C = self.mi.get_value(c_str)
                if state_d:
                    D = self.mi.get_value(d_str)
                if state_e:
                    E = self.mi.get_value(e_str)
                return eval(func)

            self.objective_func = Target(eid=a_str)
            self.objective_func.devices = []
            self.objective_func.get_value = get_value_exp

        # set maximum penalty
        self.objective_func.pen_max = self.ui.sb_max_pen.value()
        # set number of the readings
        self.objective_func.nreadings = self.ui.sb_nreadings.value()
        # set interval between readings
        self.objective_func.interval = self.ui.sb_ddelay.value()
        if self.dev_mode:
            def get_value_dev_mode():
                values = np.array([dev.get_value() for dev in self.devices])
                return np.sum(np.exp(-np.power((values - np.ones_like(values)), 2) / 5.))

            self.objective_func.get_value = get_value_dev_mode

    def set_m_status(self):
        """
        Method to set the MachineStatus method self.is_ok using GUI Alarm channel and limits
        :return: None
        """
        alarm_dev = str(self.ui.le_alarm.text()).replace(" ", "")
        print("set_m_status: alarm_dev", alarm_dev)
        if alarm_dev == "":
            return

        state = self.ui.is_le_addr_ok(self.ui.le_alarm)
        print("alarm device", self.ui.le_alarm, state)

        a_dev = AlarmDevice(alarm_dev)
        a_dev.mi = self.mi
        print(a_dev)
        if not state:
            def is_ok():
                print("ALARM switched off")
                return True
        else:
            def is_ok():
                #alarm_dev = str(self.ui.le_alarm.text())
                alarm_min = self.ui.sb_alarm_min.value()
                alarm_max = self.ui.sb_alarm_max.value()
                #alarm_value = self.mi.get_value(alarm_dev)

                alarm_value = a_dev.get_value()

                print("ALARM: ", alarm_value, alarm_min, alarm_max)
                if alarm_min <= alarm_value <= alarm_max:
                    return True
                return False

        self.m_status.is_ok = is_ok


    def getPlotData(self):
        """
        Collects data and updates plot on every GUI clock cycle.
        """
        #get times, penalties obj func data from the machine interface
        if len(self.objective_func.times) == 0:
            return

        x = np.array(self.objective_func.times) - self.objective_func.times[0]

        for plot_item, _ in self.mi.get_plot_attrs():
            pg_plot_curve = self.plot1_curves[plot_item]
            try:
                y_data = getattr(self.objective_func, plot_item, None)
                y_data = np.array(y_data)
                if y_data is None:
                    continue
                if y_data.size != x.size:
                    return
                pg_plot_curve.setData(x=x, y=y_data)
            except Exception as ex:
                print("No data to plot for: ", plot_item, ". Exception was: ", ex)

        #plot data for all devices being scanned
        for dev in self.devices:
            if len(dev.times) == 0:
                return
            y = np.array(dev.values) - self.multiPlotStarts[dev.eid]
            x = np.array(dev.times) - np.array(dev.times)[0]
            line = self.multilines[dev.eid]
            line.setData(x=x, y=y)

    def updatePlotLabels(self):
        self.plot1.plotItem.setLabels(**{'left': str(self.objective_func_pv), 'bottom': "Time (seconds)"})

    def addPlots(self):
        """
        Initializes the GUIs plots and labels on startup.
        """
        #self.objective_func_pv = "test_obj"
        #setup plot 1 for obj func monitor
        self.plot1 = pg.PlotWidget(title="Objective Function Monitor", labels={'left': str(self.objective_func_pv), 'bottom':"Time (seconds)"})
        self.plot1.showGrid(1, 1, 1)
        self.plot1.getAxis('left').enableAutoSIPrefix(enable=False) # stop the auto unit scaling on y axes
        layout = QtGui.QGridLayout()
        self.ui.widget_2.setLayout(layout)
        layout.addWidget(self.plot1, 0, 0)

        self.plot1_curves = dict()
        self.leg1 = customLegend(offset=(75, 20))
        self.leg1.setParentItem(self.plot1.graphicsItem())
        for plot_item, item_label in self.mi.get_plot_attrs():
            # create the obj func line object
            color = self.randColor()
            pen = pg.mkPen(color, width=3)
            self.plot1_curves[plot_item] = pg.PlotCurveItem(x=[], y=[], pen=pen, antialias=True, name=plot_item)
            self.plot1.addItem(self.plot1_curves[plot_item])
            self.leg1.addItem(self.plot1_curves[plot_item], item_label, color=str(color.name()))

        #setup plot 2 for device monitor
        self.plot2 = pg.PlotWidget(title="Device Monitor", labels={'left': "Device (Current - Start)", 'bottom': "Time (seconds)"})
        self.plot2.showGrid(1, 1, 1)
        self.plot2.getAxis('left').enableAutoSIPrefix(enable=False) # stop the auto unit scaling on y axes
        layout = QtGui.QGridLayout()
        self.ui.widget_3.setLayout(layout)
        layout.addWidget(self.plot2, 0, 0)

        #legend for plot 2
        self.leg2 = customLegend(offset=(75, 20))
        self.leg2.setParentItem(self.plot2.graphicsItem())

    def setUpMultiPlot(self, devices):
        """
        Reset plots when a new scan is started.
        """
        self.plot2.clear()
        self.multilines      = {}
        self.multiPvData     = {}
        self.multiPlotStarts = {}
        x = []
        y = []
        self.leg2.scene().removeItem(self.leg2)
        self.leg2 = customLegend(offset=(50, 10))
        self.leg2.setParentItem(self.plot2.graphicsItem())

        default_colors = [QtGui.QColor(255, 51, 51), QtGui.QColor(51, 255, 51), QtGui.QColor(255, 255, 51),QtGui.QColor(178, 102, 255)]
        for i, dev in enumerate(devices):

            #set the first 4 devices to have the same default colors
            if i < 4:
                color = default_colors[i]
            else:
                color = self.randColor()

            pen=pg.mkPen(color, width=2)
            self.multilines[dev.eid] = pg.PlotCurveItem(x, y, pen=pen, antialias=True, name=str(dev.eid))
            self.multiPvData[dev.eid] = []
            self.multiPlotStarts[dev.eid] = dev.get_value()
            self.plot2.addItem(self.multilines[dev.eid])
            self.leg2.addItem(self.multilines[dev.eid], dev.eid, color=str(color.name()))

    def randColor(self):
        """
        Generate random line color for each device plotted.
        :return: QColor object of a random color
        """
        hi = 255
        lo = 128
        c1 = np.random.randint(lo,hi)
        c2 = np.random.randint(lo,hi)
        c3 = np.random.randint(lo,hi)
        return QtGui.QColor(c1,c2,c3)

    def run_editor(self):
        """
        Run the editor for edition of the objective function in obj_function.py
        :return:
        """
        if platform.system() == 'Darwin':
            #subprocess.call(['open', '-a', 'TextEdit', self.path_to_obj_func])
            subprocess.call(['open', self.path_to_obj_func])
        elif platform.system() == 'Windows':
            subprocess.call(['C:\\Windows\\System32\\notepad.exe', self.path_to_obj_func])
        elif platform.system() == 'Linux':
            subprocess.call(['gedit', self.path_to_obj_func])
        else:
            print("Unknown platform")
            return
        self.set_obj_fun()

    def error_box(self, message):
        QtGui.QMessageBox.about(self, "Error box", message)
        #QtGui.QMessageBox.critical(self, "Error box", message)

    def assemble_quick_add_box(self):
        devs = self.mi.get_quick_add_devices()
        if devs is None or len(devs) == 0:
            return

        resetpanel_box = self.ui.widget
        layout_quick_add = resetpanel_box.ui.layout_quick_add

        def add_to_list():
            pvs = cb_quick_list.currentData()
            resetpanel_box.addPv(pvs, force_active=True)

        def clear_list():
            resetpanel_box.pvs = []
            resetpanel_box.devices = []
            resetpanel_box.ui.tableWidget.setRowCount(0)

        pb_clear_dev = QPushButton(resetpanel_box)
        pb_clear_dev.setText("Clear Devices")
        pb_clear_dev.setMaximumWidth(100)
        pb_clear_dev.clicked.connect(clear_list)
        pb_add_dev = QPushButton(resetpanel_box)
        pb_add_dev.setText("Add Devices")
        pb_add_dev.clicked.connect(add_to_list)
        pb_add_dev.setMaximumWidth(100)
        lb_from_list = QLabel()
        lb_from_list.setText("From List: ")
        lb_from_list.setMaximumWidth(75)
        cb_quick_list = QComboBox()
        cb_quick_list.addItem("", [])
        cb_quick_list.setMinimumWidth(200)
        for display, itms in devs.items():
            cb_quick_list.addItem(display, itms)

        layout_quick_add.addWidget(pb_clear_dev)
        layout_quick_add.addWidget(pb_add_dev)
        layout_quick_add.addWidget(lb_from_list)
        layout_quick_add.addWidget(cb_quick_list)

    def assemble_preset_box(self):
        print("Assembling Preset Box")
        presets = self.mi.get_preset_settings()
        layout = self.ui.preset_layout
        for display, methods in presets.items():
            gb = QGroupBox(self)
            gb.setTitle(display)
            inner_layout = QVBoxLayout(gb)
            for m in methods:
                btn = QPushButton(gb)
                btn.setText(m["display"])
                inner_layout.addWidget(btn)
                part = functools.partial(self.ui.load_settings,os.path.join(self.path2preset, m["filename"]))
                btn.clicked.connect(part)
            vert_spacer = QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
            inner_layout.addItem(vert_spacer)
            layout.addWidget(gb)

class customLegend(pg.LegendItem):
    """
    STUFF FOR PG CUSTOM LEGEND (subclassed from pyqtgraph).
    Class responsible for drawing a single item in a LegendItem (sans label).
    This may be subclassed to draw custom graphics in a Legend.
    """
    def __init__(self, size=None, offset=None):
        pg.LegendItem.__init__(self, size, offset)

    def addItem(self, item, name, color="CCFF00"):

        label = pg.LabelItem(name, color=color, size="10pt", bold=True)
        sample = None
        row = self.layout.rowCount()
        self.items.append((sample, label))
        self.layout.addItem(sample, row, 0)
        self.layout.addItem(label, row, 1)
        self.layout.setSpacing(0)


def main():

    """
    Funciton to start up the main program.
    Slecting a PV parameter set:
    If launched from the command line will take an argument with the filename of a parameter file.
    If no argv[1] is provided, the default list in ./parameters/lclsparams is used.
    Development mode:
    If devmode == False - GUI defaults to normal parameter list, defaults to nelder mead simplex
    if devmode == True  - GUI uses 4 development matlab PVs and loaded settings in the method "devmode()"
    """

    #try to get a pv list file name from commandline arg
    #this goes into initializing the reset panel PVs that show up in the GUI
    #try:
    #    pvs = sys.argv[1]   # arg filename of params
    #except:
    #    pvs = 'parameters/lcls_short.txt'#default filename

    #make pyqt threadsafe
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_X11InitThreads)

    #create the application
    app = QApplication(sys.argv)
    # setting the path variable for icon
    path = os.path.join(os.path.dirname(sys.modules[__name__].__file__), 'ocelot.png')
    app.setWindowIcon(QtGui.QIcon(path))
    window = OcelotInterfaceWindow()

    frame_gm = window.frameGeometry()
    center_point = QDesktopWidget().availableGeometry().center()
    frame_gm.moveCenter(center_point)
    window.move(frame_gm.topLeft())
    # setting the path variable for icon
    #path = os.path.join(os.path.dirname(sys.modules[__name__].__file__), 'ocelot.png')
    #window.setWindowIcon(QtGui.QIcon(path))

    #Build the PV list from dev PVs or selected source
    #window.ui.widget.getPvList(pvs)
    # set checkbot status
    #window.ui.widget.uncheckBoxes()

    timer = pg.QtCore.QTimer()
    timer.timeout.connect(window.scan_finished)
    timer.start(300)

    indicator = QtCore.QTimer()
    indicator.timeout.connect(window.indicate_machine_state)
    indicator.start(10)

    #show app

    #window.setWindowIcon(QtGui.QIcon('ocelot.png'))
    window.show()

    #Build documentaiton if source files have changed
    # TODO: make more universal
    #os.system("cd ./docs && xterm -T 'Ocelot Doc Builder' -e 'bash checkDocBuild.sh' &")
    #exit script
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
