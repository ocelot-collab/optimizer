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
try:
   import sklearn
   sklearn_version = sklearn.__version__
except:
   sklearn_version = None
   
import functools
import inspect
import parameters

print('GENERIC OPTIM PATH: ',  os.path.abspath(os.path.join(__file__ ,"../")) + os.sep)


path = os.path.realpath(__file__)
#indx = path.find("ocelot/optimizer")
print("PATH", os.path.realpath(__file__))
#sys.path.append(path[:indx])

# for pyqtgraph import
#sys.path.append(path[:indx]+"ocelot")

from PyQt5.QtWidgets import (QApplication, QFrame, QGroupBox, QLabel, QComboBox,
                             QPushButton, QSpacerItem, QVBoxLayout, QDesktopWidget,
                             QFormLayout, QLineEdit)
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

from mint.xfel.xfel_interface import *
from mint.lcls.lcls_interface import *
from mint.aps.aps_interface import *
from mint.bessy.bessy_interface import *
from mint.demo.demo_interface import *
from mint.petra.petra_interface import *
from sint.multinormal.multinormal_interface import *
from op_methods.simplex import *
from op_methods.gp_slac import *
from op_methods.es import *
from op_methods.custom_minimizer import *
from op_methods.powell import *
from op_methods.gp_sklearn import *


from stats import stats

AVAILABLE_MACHINE_INTERFACES = [XFELMachineInterface, LCLSMachineInterface, APSMachineInterface,
                                TestMachineInterface, BESSYMachineInterface, MultinormalInterface, PETRAMachineInterface,
                                DemoInterface]


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
        self.plots_dict = dict()
        self.optimizer_args = None
        self.parse_arguments()
        self.dev_mode = self.optimizer_args.devmode

        args = vars(self.optimizer_args)
        if self.dev_mode:
            self.mi = TestMachineInterface(args)
        else:
            class_name = self.optimizer_args.mi
            if class_name not in globals():
                print("Could not find Machine Interface with name: {}. Loading XFELMachineInterface instead.".format(class_name))
                self.mi = XFELMachineInterface(args)
            else:
                self.mi = globals()[class_name](args)
        self.optimizer_path = os.path.abspath(os.path.join(__file__ ,"../")) + os.sep
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
        self.name_powell = "Powell"
        self.name_gauss_gpy = "GP GPy"
        # self.name4 = "Conjugate Gradient"
        # self.name5 = "Powell's Method"
        # switch of GP and custom Mininimizer
        self.ui.cb_select_alg.addItem(self.name_simplex)
        self.ui.cb_select_alg.addItem(self.name_gauss)
        # self.ui.cb_select_alg.addItem(self.name_custom)
        self.ui.cb_select_alg.addItem(self.name_simplex_norm)
        self.ui.cb_select_alg.addItem(self.name_es)
        self.ui.cb_select_alg.addItem(self.name_powell)
        #self.ui.cb_select_alg.addItem(self.name_gauss_gpy)
        # if sklearn_version >= "0.18":
        #     self.ui.cb_select_alg.addItem(self.name_gauss_sklearn)

        #self.ui.pb_help.clicked.connect(lambda: os.system("firefox file://"+self.optimizer_path+"docs/build/html/index.html"))
        self.ui.pb_help.clicked.connect(self.ui.open_help)

        self.assemble_preset_box()
        self.assemble_quick_add_box()

        if self.mi.use_num_points():
            # Get rid of Interval Between Readings
            self.ui.label_6.setVisible(False)
            self.ui.sb_ddelay.setVisible(False)

            # Get rid of number of readings
            self.ui.label_30.setVisible(False)
            self.ui.sb_nreadings.setVisible(False)

            # Get rid of Cycle Period
            self.ui.label_7.setVisible(False)
        else:
            self.ui.label_datapoints.setVisible(False)
            self.ui.sb_datapoints.setVisible(False)

        self.total_delay = self.ui.sb_tdelay.value()

        self.objective_func_pv = "test_obj"

        self.setup_plots()

        # database

        self.scan_params = None
        self.hyper_file = parameters.path_to_hyps

        self.set_obj_fun()

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


        self.m_status = mint.MachineStatus()
        self.set_m_status()

        self.opt_control = mint.OptControl()
        self.opt_control.alarm_timeout = self.ui.sb_alarm_timeout.value()
        self.opt_control.m_status = self.m_status

        #timer for plots, starts when scan starts
        self.update_plot_timer = QtCore.QTimer()
        self.update_plot_timer.timeout.connect(self.update_plots)
        self.update_plots()

        self.ui.browser_data_slider.valueChanged.connect(self.browser_slider_changed)
        self.ui.browser_restore_btn.clicked.connect(self.browser_restore_clicked)
        self.mi.customize_ui(self)

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description="Ocelot Optimizer",
                                         add_help=False)
        parser.set_defaults(mi='XFELMachineInterface')
        parser.add_argument('--devmode', action='store_true',
                            help='Enable development mode.', default=False)

        parser_mi = argparse.ArgumentParser()

        mis = [mi.__class__.__name__ for mi in AVAILABLE_MACHINE_INTERFACES]
        subparser = parser_mi.add_subparsers(title='Machine Interface Options', dest="mi")
        for mi in AVAILABLE_MACHINE_INTERFACES:
            mi_parser = subparser.add_parser(mi.__name__, help='{} arguments'.format(mi.__name__))
            mi.add_args(mi_parser)
        self.optimizer_args, others = parser.parse_known_args()

        if len(others) != 0:
            self.optimizer_args = parser_mi.parse_args(others, namespace=self.optimizer_args)

    def setup_plots(self):
        self.setup_objhist_plot()
        self.setup_obj_plot('plot_obj')
        self.setup_obj_plot('plot_obj_browser')
        self.setup_devices_plot('plot_dev')
        self.setup_devices_plot('plot_dev_browser')

        layout = QVBoxLayout()
        self.ui.widget_2.setLayout(layout)
        layout.addWidget(self.plots_dict['plot_obj']['plot'])

        layout = QVBoxLayout()
        self.ui.browser_obj_plot_panel.setLayout(layout)
        layout.addWidget(self.plots_dict['plot_obj_browser']['plot'])

        layout = QVBoxLayout()
        self.ui.widget_3.setLayout(layout)
        layout.addWidget(self.plots_dict['plot_dev']['plot'])

        layout = QVBoxLayout()
        self.ui.browser_dev_plot_panel.setLayout(layout)
        layout.addWidget(self.plots_dict['plot_dev_browser']['plot'])

        layout = QVBoxLayout()
        self.ui.browser_objhist_plot_panel.setLayout(layout)
        layout.addWidget(self.plots_dict['plot_objhist_browser']['plot'])

        headers = ["Variable", "Value"]
        table = self.ui.browser_data_table
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)  # No user edits on talbe
        table.horizontalHeader().setResizeMode(QtGui.QHeaderView.Stretch)
        table.setRowCount(0)

        self.setup_region('plot_obj_browser')
        self.setup_region('plot_dev_browser')

    def setup_region(self, name):
        plot = self.plots_dict[name]['plot']
        self.plots_dict[name]['region'] = pg.LinearRegionItem((0,0))
        region = self.plots_dict[name]['region']
        region.setMovable(False)
        # region.setZValue(-10)
        plot.addItem(region)

    def setup_objhist_plot(self):
        name = 'plot_objhist_browser'
        self.plots_dict[name] = dict()
        plot_curves = dict()

        plot = pg.PlotWidget(title="Objective Function Histogram Monitor", labels={'left': 'Count', 'bottom':'Data'})
        plot.showGrid(1, 1, 1)
        plot.getAxis('left').enableAutoSIPrefix(enable=False) # stop the auto unit scaling on y axes

        color = self.randColor()
        pen = pg.mkPen(color, width=3)
        plot_curves['histogram'] = pg.PlotCurveItem(x=[0,1], y=[0], pen=pen, antialias=True, name='histogram', stepMode = True, fillLevel = 0, brush = (0, 0, 255, 150))
        plot.addItem(plot_curves['histogram'])

        self.plots_dict[name]['plot'] = plot
        self.plots_dict[name]['legend'] = None
        self.plots_dict[name]['curves'] = plot_curves

    def setup_obj_plot(self, name):
        self.plots_dict[name] = dict()
        plot_curves = dict()

        plot = pg.PlotWidget(title="Objective Function Monitor", labels={'left': str(self.objective_func_pv), 'bottom':"Time (seconds)"})
        plot.showGrid(1, 1, 1)
        plot.getAxis('left').enableAutoSIPrefix(enable=False) # stop the auto unit scaling on y axes

        legend = customLegend(offset=(75, 20))
        legend.setParentItem(plot.graphicsItem())

        default_colors = [QtGui.QColor(0, 255, 255),
                          QtGui.QColor(108, 237, 125),
                          QtGui.QColor(255, 255, 51),
                          QtGui.QColor(178, 102, 255)]

        idx = 0
        for plot_item, item_label in self.mi.get_plot_attrs():
            # set the first 4 to have the same default colors
            if idx < 4:
                color = default_colors[idx]
            else:
                color = self.randColor()

            # create the obj func line object
            # color = self.randColor()
            pen = pg.mkPen(color, width=5)
            plot_curves[plot_item] = pg.PlotCurveItem(x=[], y=[], pen=pen, antialias=True, name=plot_item)
            plot.addItem(plot_curves[plot_item])
            legend.addItem(plot_curves[plot_item], item_label, color=str(color.name()))
            idx += 1

        self.plots_dict[name]['plot'] = plot
        self.plots_dict[name]['legend'] = legend
        self.plots_dict[name]['curves'] = plot_curves

    def setup_devices_plot(self, name):
        self.plots_dict[name] = dict()

        #setup plot 2 for device monitor
        plot = pg.PlotWidget(title="Device Monitor", labels={'left': "Device (Current - Start)", 'bottom': "Time (seconds)"})
        plot.showGrid(1, 1, 1)
        plot.getAxis('left').enableAutoSIPrefix(enable=False) # stop the auto unit scaling on y axes

        #legend for plot 2
        legend = customLegend(offset=(75, 20))
        legend.setParentItem(plot.graphicsItem())

        self.plots_dict[name]['plot'] = plot
        self.plots_dict[name]['legend'] = legend
        self.plots_dict[name]['curves'] = dict()

    def browser_restore_clicked(self):
        confirm_msg = "Are you sure you want to restore the selected values?"
        reply = QtGui.QMessageBox.question(self, 'Message',
                                           confirm_msg, QtGui.QMessageBox.Yes,
                                           QtGui.QMessageBox.No)

        if reply != QtGui.QMessageBox.Yes:
            return

        index = self.ui.browser_data_slider.value()
        print("***** Restoring Devices to value at index: ", index)
        for dev in self.devices:
            try:
                val = dev.values[index]
                print("Restore: {} to value: {}".format(dev.eid, val))
                dev.set_value(val)
            except IndexError:
                print("Restore: {} failed. Index Error.".format(dev.eid))

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
            scaling_coef = self.ui.sb_scaling_coef.value()
            #minimizer = GaussProcess()
            minimizer = GaussProcess(searchBoundScaleFactor=scaling_coef)
            minimizer.seedScanBool = self.ui.cb_use_live_seed.isChecked()

        elif current_method == self.name_gauss_sklearn:
            minimizer = GaussProcessSKLearn()
            minimizer.seed_iter = self.ui.sb_seed_iter.value()

        elif current_method == self.name_gauss_gpy:
            minimizer = GPgpy()
            minimizer.seed_iter = self.ui.sb_seed_iter.value()

        # Custom Minimizer
        elif current_method == self.name_custom:
            minimizer = CustomMinimizer()

        elif current_method == self.name_simplex_norm:
            minimizer = SimplexNorm()
        elif current_method == self.name_es:
            minimizer = ESMin()
        elif current_method == self.name_powell:
            minimizer = Powell()
        #simplex Method
        else:
            minimizer = Simplex()

        self.method_name = minimizer.__class__.__name__

        return minimizer

    def closeEvent(self, event):
        if self.mi.save_at_exit():
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
            self.opt.join()
            self.m_status.is_ok = lambda: True

            # Save the optimization parameters to the database
            #try:
            ret, msg = self.save2db()
            if not ret:
               self.error_box(message=msg)
            #except Exception as ex:
            #    print("ERROR start_scan: can not save to db. Exception was: ", ex)
            #    traceback.print_exc()
            del self.opt
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
        self.update_devices_plot(self.devices)
        self.update_plot_timer.start(100)
        # set the Objective function from GUI or from file mint.obj_function.py
        # (reloading)
        self.set_obj_fun(update_objfunc_text=False)
        # if self.ui.le_obf.text():
        #     self.objective_func.eid = self.ui.le_obf.text()
        self.objective_func_pv = self.objective_func.eid

        if self.mi.use_num_points():
            self.objective_func.points = self.ui.sb_datapoints.value()

        self.update_plot_obj_labels()
        # Set minimizer - the optimization method (Simplex, GP, ...)
        minimizer = self.scan_method_select()

        # configure the Minimizer
        minimizer.mi = self.mi
        if minimizer.__class__ in [GaussProcess, GaussProcessSKLearn]:
            minimizer.seed_iter = self.ui.sb_seed_iter.value()
            minimizer.seed_timeout = self.ui.sb_tdelay.value()
            minimizer.hyper_file = self.hyper_file
            minimizer.norm_coef = self.ui.sb_isim_rel_step.value() / 100.

            if self.ui.cb_use_isim.checkState():
                if self.ui.cb_use_isim.checkState():

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
                                    step = val0 * 0.05
                                dev.istep = step
                            else:
                                dev.istep = d_lims * rel_step / 100.

        elif minimizer.__class__ in [Simplex, Powell]:

            if self.ui.cb_use_isim.checkState():

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
                            dev.istep = step
                        else:
                            dev.istep = d_lims * rel_step / 100.

        elif minimizer.__class__ in [ESMin]:


            bounds = []
            for dev in self.devices:
                bounds.append(dev.get_limits())
            minimizer.bounds = bounds

            minimizer.norm_coef = self.ui.sb_isim_rel_step.value() / 100.

        elif minimizer.__class__ == CustomMinimizer:
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

        self.opt.scaling_coef = self.ui.sb_scaling_coef.value()
        print("Using Scaling Coeficient of: ", self.opt.scaling_coef)

        # solving minimization or maximization problem
        self.opt.maximization = self.ui.rb_maximize.isChecked()

        if self.ui.cb_select_alg.currentText() in [self.name_simplex_norm]:
            self.opt.normalization = True
            self.opt.norm_coef = self.ui.sb_isim_rel_step.value()*0.01
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

    def set_obj_fun(self, update_objfunc_text=True):
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
            print("RELOAD Module Objective Function", self.mi.get_obj_function_module())
            obj_function_module = self.mi.get_obj_function_module()
            if 'target_class' in dir(obj_function_module):
                tclass = obj_function_module.target_class
            else:
                tclass = [obj for name, obj in inspect.getmembers(obj_function_module) if
                            inspect.isclass(obj) and issubclass(obj, Target) and obj != Target][0]

            print("Target Class: ", tclass)
            self.objective_func = tclass(mi=self.mi)
            if update_objfunc_text:
                self.ui.le_obf.setText(self.objective_func.eid)
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
            func_id = func
            for v, v_str in zip(["A", "B", "C", "D", "E"], [a_str, b_str, c_str, d_str, e_str]):
                if v in func:
                    func_id += " @ " + v + " = " + v_str
            self.objective_func = Target(eid=func_id)
            self.objective_func.devices = []
            self.objective_func.get_value = get_value_exp
            self.objective_func.mi = self.mi

        # set maximum penalty
        self.objective_func.pen_max = self.ui.sb_max_pen.value()
        # set number of the readings
        self.objective_func.nreadings = self.ui.sb_nreadings.value()
        # set interval between readings
        self.objective_func.interval = self.ui.sb_ddelay.value()
        if self.dev_mode:
            def get_value_dev_mode():
                values = np.array([dev.get_value() for dev in self.devices])
                return np.sum(np.exp(-np.power((values - np.ones_like(values)), 2) / 5.)) #+ np.random.rand()*0.1

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
            self.m_status.alarm_device = None
        else:

            self.m_status.alarm_device = a_dev
            self.m_status.alarm_min = self.ui.sb_alarm_min.value()
            self.m_status.alarm_max = self.ui.sb_alarm_max.value()

    def update_plots(self):
        """
        Collects data and updates plot on every GUI clock cycle.
        """
        #get times, penalties obj func data from the machine interface
        if len(self.objective_func.times) == 0:
            self.ui.browser_data_slider.setEnabled(False)
            return

        self.ui.browser_data_slider.setEnabled(True)


        scan_running = self.ui.pb_start_scan.text() == "Stop optimization"
        self.ui.browser_restore_btn.setEnabled(not scan_running)

        if self.ui.browser_data_table.rowCount() == 0:
            self.browser_data_changed(0)

        x = np.array(self.objective_func.times) - self.objective_func.times[0]
        n_dev_sets = len(self.devices[0].times)
        self.ui.browser_data_slider.setMaximum(n_dev_sets-1)

        for plot_item, _ in self.mi.get_plot_attrs():
            for plot_name in ['plot_obj', 'plot_obj_browser']:
                line = self.plots_dict[plot_name]['curves'][plot_item]
                try:
                    y_data = getattr(self.objective_func, plot_item, None)
                    y_data = np.array(y_data)
                    if y_data is None:
                        continue
                    if y_data.size != x.size:
                        return
                    line.setData(x=x, y=y_data)
                except Exception as ex:
                    print("No data to plot for: ", plot_item, ". Exception was: ", ex)

        #plot data for all devices being scanned
        for dev in self.devices:
            for plot_name in ['plot_dev', 'plot_dev_browser']:
                if len(dev.times) == 0:
                    return
                y = np.array(dev.values) - self.multiPlotStarts[dev.eid]
                x = np.array(dev.times) - np.array(dev.times)[0]
                line = self.plots_dict[plot_name]['curves'][dev.eid]
                line.setData(x=x, y=y)

    def browser_slider_changed(self, index):
        self.browser_data_changed(index)

    def browser_data_changed(self, index, region=False):
        x = np.array(self.objective_func.times) - self.objective_func.times[0]
        if not region:
            for plot_name in ['plot_dev_browser', 'plot_obj_browser']:
                region = self.plots_dict[plot_name]['region']
                index_val = x[index]

                if len(x) > 1 and plot_name == 'plot_obj_browser':
                    region.setBounds([x[index+1], x[index+1]])
                else:
                    region.setBounds([index_val, index_val])

        histogram_data_key = 'values'
        if hasattr(self.objective_func, 'objective_acquisitions'):
            histogram_data_key = 'objective_acquisitions'

        try:
            val = getattr(self.objective_func, histogram_data_key)[index+1]
            hist, bins = np.histogram(val, bins='auto')
            line = self.plots_dict['plot_objhist_browser']['curves']['histogram']
            line.setData(x=bins, y=hist)
        except Exception as ex:
            print("No data to plot histogram. Exception was: ", ex)

        table = self.ui.browser_data_table
        table.setRowCount(len(self.devices) + len(self.mi.get_plot_attrs()))
        table_data = []

        for plot_item, display_name in self.mi.get_plot_attrs():
            try:
                y_data = getattr(self.objective_func, plot_item, None)
                y_data = np.array(y_data)
                if y_data is None:
                    continue
                table_data.append((display_name, y_data[index+1]))
            except Exception as ex:
                print("No data to plot for: ", plot_item, ". Exception was: ",
                      ex)

        for dev in self.devices:
            if index > len(dev.values):
                continue
            y = np.array(dev.values)
            if len(y)> 0:
                table_data.append((dev.eid, y[index]))

        for row, data in enumerate(table_data):
            label, value = data
            table.setItem(row, 0, QtGui.QTableWidgetItem(str(label)))
            table.setItem(row, 1, QtGui.QTableWidgetItem(str(value)))

    def update_plot_obj_labels(self):
        for plot_name in ['plot_obj', 'plot_obj_browser']:
            plot = self.plots_dict[plot_name]['plot']
            plot.plotItem.setLabels(**{'left': str(self.objective_func_pv),
                                       'bottom': "Time (seconds)"})

    def update_devices_plot(self, devices):
        """
        Reset plots when a new scan is started.
        """
        self.multiPlotStarts = {}

        for idx, plot_name in enumerate(['plot_dev', 'plot_dev_browser']):
            plot = self.plots_dict[plot_name]['plot']
            plot.clear()
            x = []
            y = []
            leg = self.plots_dict[plot_name]['legend']
            leg.scene().removeItem(leg)
            leg = customLegend(offset=(50, 10))
            leg.setParentItem(plot.graphicsItem())
            self.plots_dict[plot_name]['legend'] = leg

            default_colors = [QtGui.QColor(255, 51, 51), QtGui.QColor(51, 255, 51), QtGui.QColor(255, 255, 51),QtGui.QColor(178, 102, 255)]
            for i, dev in enumerate(devices):

                #set the first 4 devices to have the same default colors
                if i < 4:
                    color = default_colors[i]
                else:
                    color = self.randColor()

                pen=pg.mkPen(color, width=3)
                item = pg.PlotCurveItem(x, y, pen=pen, antialias=True, name=str(dev.eid))
                if idx == 0:
                    self.multiPlotStarts[dev.eid] = dev.get_value()
                plot.addItem(item)
                leg.addItem(item, dev.eid, color=str(color.name()))
                self.plots_dict[plot_name]['curves'][dev.eid] = item

        self.setup_region('plot_dev_browser')

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

        def add_from_txt(le):
            pv = le.text()
            if pv:
                resetpanel_box.addPv(pv, force_active=True)

        def clear_list():
            resetpanel_box.pvs = []
            resetpanel_box.devices = []
            resetpanel_box.ui.tableWidget.setRowCount(0)

        layout_buttons = QVBoxLayout()


        pb_clear_dev = QPushButton(resetpanel_box)
        pb_clear_dev.setText("Clear Devices")
        pb_clear_dev.setMaximumWidth(100)
        pb_clear_dev.clicked.connect(clear_list)

        pb_add_dev = QPushButton(resetpanel_box)
        pb_add_dev.setText("Add Devices")
        pb_add_dev.setStyleSheet("color: orange")
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

        if len(devs.items()) >= 1:
            cb_quick_list.setCurrentIndex(1)

        lb_manually = QLabel()
        lb_manually.setText("Or Manually Enter: ")
        lb_manually.setMaximumWidth(75)
        le_manually = QLineEdit()
        le_manually.setMinimumWidth(200)
        le_manually.returnPressed.connect(functools.partial(add_from_txt, le_manually))

        layout_buttons.addWidget(pb_add_dev)
        layout_buttons.addWidget(pb_clear_dev)

        frm_layout = QFormLayout()
        frm_layout.addRow(lb_from_list, cb_quick_list)
        frm_layout.addRow(lb_manually, le_manually)

        layout_quick_add.addLayout(layout_buttons)
        layout_quick_add.addLayout(frm_layout)

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
    indicator.start(300)

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
