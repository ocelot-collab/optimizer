"""
Most of GUI logic is placed here.
S.Tomin, 2017
"""

from UIOcelotInterface_gen import *
import os
import json
import scipy
from PyQt5.QtGui import QCursor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget
import subprocess
import base64
import numpy as np
import sys
import webbrowser
from shutil import copy


class MainWindow(Ui_Form):
    def __init__(self, Form):
        Ui_Form.__init__(self)
        self.setupUi(Form)
        self.Form = Form
        # load in the dark theme style sheet
        self.loadStyleSheet()
        self.widget.set_parent(Form)
        self.pb_save_as.clicked.connect(self.save_state_as)
        self.pb_load.clicked.connect(self.load_state_from)
        self.pb_rewrite.clicked.connect(self.rewrite_default)
        self.cb_use_isim.stateChanged.connect(self.change_state_scipy_setup)
        self.pb_hyper_file.clicked.connect(self.get_hyper_file)
        self.pb_logbook.clicked.connect(self.logbook)

        self.le_a.editingFinished.connect(self.check_address)
        self.le_b.editingFinished.connect(self.check_address)
        self.le_c.editingFinished.connect(self.check_address)
        self.le_d.editingFinished.connect(self.check_address)
        self.le_e.editingFinished.connect(self.check_address)
        self.le_alarm.editingFinished.connect(self.check_address)

        self.sb_tdelay.valueChanged.connect(self.set_cycle)
        self.sb_ddelay.valueChanged.connect(self.set_cycle)
        self.sb_nreadings.valueChanged.connect(self.set_cycle)
        self.cb_select_alg.currentIndexChanged.connect(self.change_state_scipy_setup)
        self.read_alarm = QtCore.QTimer()
        self.read_alarm.timeout.connect(self.alarm_value)
        self.read_alarm.start(1000)

    def alarm_value(self):
        """
        reading alarm value
        :return:
        """
        if self.le_alarm.hasFocus():
            return
        dev = str(self.le_alarm.text())
        try:
            value = self.Form.mi.get_value(dev)
            self.label_alarm.setText(str(np.round(value, 2)))
        except:
            self.label_alarm.setText(str("None"))

    def set_cycle(self):
        """
        Select time for objective method data collection time.
        Scanner will wait this long to collect new data.
        """
        self.trim_delay = self.sb_tdelay.value()
        data_delay = self.sb_ddelay.value()*self.sb_nreadings.value()
        self.label_7.setText("Cycle Period = " + str(np.around(self.trim_delay + data_delay, 3)))
        self.Form.total_delay = self.trim_delay

    def check_address(self):
        self.is_le_addr_ok(self.le_a)
        self.is_le_addr_ok(self.le_b)
        self.is_le_addr_ok(self.le_c)
        self.is_le_addr_ok(self.le_d)
        self.is_le_addr_ok(self.le_e)
        self.is_le_addr_ok(self.le_alarm)

    def is_le_addr_ok(self, line_edit):
        if not line_edit.isEnabled():
            return False
        dev = str(line_edit.text())
        state = True
        try:
            val = self.Form.mi.get_value(dev)
            if val is None:
                state = False
        except:
            state = False

        if state:
            line_edit.setStyleSheet("color: rgb(85, 255, 0);")
        else:
            line_edit.setStyleSheet("color: red")
        line_edit.clearFocus()
        return state

    def save_state(self, filename):
        # pvs = self.ui.widget.pvs
        table = self.widget.get_state()

        table["use_predef"] = self.cb_use_predef.checkState()
        table["statistics"] = self.cb_statistics.currentIndex()
        table["data_points"] = self.sb_datapoints.value()

        max_pen = self.sb_max_pen.value()
        timeout = self.sb_tdelay.value()


        max_iter = self.sb_num_iter.value()
        # objective function
        fun_a = str(self.le_a.text())
        fun_b = str(self.le_b.text())
        fun_c = str(self.le_c.text())
        obj_fun = str(self.le_obf.text())
        # alarm
        alarm_dev = str(self.le_alarm.text())
        alarm_min = self.sb_alarm_min.value()
        alarm_max = self.sb_alarm_max.value()

        table["max_pen"] = max_pen
        table["timeout"] = timeout
        table["nreadings"] = self.sb_nreadings.value()
        table["interval"] = self.sb_ddelay.value()

        table["max_iter"] = max_iter
        table["fun_a"] = fun_a
        table["fun_b"] = fun_b
        table["fun_c"] = fun_c
        table["fun_d"] = str(self.le_d.text())
        table["fun_e"] = str(self.le_e.text())
        table["obj_fun"] = obj_fun

        table["alarm_dev"] = alarm_dev
        table["alarm_min"] = alarm_min
        table["alarm_max"] = alarm_max
        table["alarm_timeout"] = self.sb_alarm_timeout.value()

        table["seed_iter"] = self.sb_seed_iter.value()
        table["use_live_seed"] = self.cb_use_live_seed.checkState()

        table["isim_rel_step"] = self.sb_isim_rel_step.value()
        table["use_isim"] = self.cb_use_isim.checkState()

        table["hyper_file"] = self.Form.hyper_file

        table["set_best_sol"] = self.cb_set_best_sol.checkState()

        table["algorithm"] = str(self.cb_select_alg.currentText())
        table["maximization"] = self.rb_maximize.isChecked()

        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
            # create folder "obj_funcs" incide new folder
            obj_funcs_name = os.path.dirname(filename) + os.sep + "obj_funcs"
            os.makedirs(obj_funcs_name)

        with open(filename, 'w') as f:
            json.dump(table, f)
        # pickle.dump(table, filename)
        print("SAVE State")

    def restore_state(self, filename):
        try:
            with open(filename, 'r') as f:
                # data_new = pickle.load(f)
                table = json.load(f)
        except Exception as ex:
            print("Restore State failed for file: {}. Exception was: {}".format(filename, ex))
            return


        # Build the PV list from dev PVs or selected source
        pvs = table["id"]
        self.widget.set_machine_interface(self.Form.mi)
        self.widget.getPvList(pvs)
        # set checkbot status
        self.widget.uncheckBoxes()
        self.widget.set_state(table)

        try:

            max_pen = table["max_pen"]
            timeout = table["timeout"]
            max_iter = table["max_iter"]
            fun_a = table["fun_a"]
            fun_b = table["fun_b"]
            fun_c = table["fun_c"]
            obj_fun = table["obj_fun"]

            if "use_predef" in table.keys(): self.cb_use_predef.setCheckState(table["use_predef"])
            if "statistics" in table.keys(): self.cb_statistics.setCurrentIndex(table["statistics"])
            if "data_points" in table.keys(): self.sb_datapoints.setValue(table["data_points"])
            self.sb_max_pen.setValue(max_pen)
            self.sb_tdelay.setValue(timeout)
            self.sb_nreadings.setValue(table["nreadings"])
            self.sb_ddelay.setValue(table["interval"])

            self.sb_num_iter.setValue(max_iter)
            self.le_a.setText(fun_a)
            self.le_b.setText(fun_b)
            self.le_c.setText(fun_c)
            self.le_d.setText(table["fun_d"])
            self.le_e.setText(table["fun_e"])
            self.le_obf.setText(obj_fun)

            self.le_alarm.setText(table["alarm_dev"])
            self.sb_alarm_min.setValue(table["alarm_min"])
            self.sb_alarm_max.setValue(table["alarm_max"])
            self.sb_alarm_timeout.setValue(table["alarm_timeout"])

            self.sb_seed_iter.setValue(table["seed_iter"])
            self.cb_use_live_seed.setCheckState(table["use_live_seed"])

            self.sb_isim_rel_step.setValue(table["isim_rel_step"])
            self.cb_use_isim.setCheckState(table["use_isim"])
            self.change_state_scipy_setup()

            self.Form.hyper_file = table["hyper_file"]
            self.pb_hyper_file.setText(self.Form.hyper_file)

            self.cb_set_best_sol.setCheckState(table["set_best_sol"])
            if "maximization" in table.keys():
                #if table["maximization"] == True:
                self.rb_maximize.setChecked(table["maximization"])
                self.rb_minimize.setChecked(not table["maximization"])
            if "algorithm" in table.keys():
                index = self.cb_select_alg.findText(table["algorithm"], QtCore.Qt.MatchFixedString)

                if index >= 0:
                    self.cb_select_alg.setCurrentIndex(index)
            print("RESTORE STATE: OK")
        except:
            print("RESTORE STATE: ERROR")

    def save_state_as(self):
        filename = QtGui.QFileDialog.getSaveFileName(self.Form, 'Save State',
                                                     self.Form.config_dir, "txt (*.json)", None,
                                                     QtGui.QFileDialog.DontUseNativeDialog)[0]
        if filename:
            name = filename.split("/")[-1]
            parts = name.split(".")
            body_name = parts[0]

            if len(parts)<2 or parts[1] !="json":
                part = filename.split(".")[0]
                filename = part + ".json"

            copy(self.Form.path_to_obj_func, self.Form.obj_save_path + os.sep + body_name +".py")
            #self.Form.set_file = filename
            self.save_state(filename)

    def load_state_from(self):
        filename = QtGui.QFileDialog.getOpenFileName(self.Form, 'Load State',
                                                     self.Form.config_dir, "txt (*.json)", None,
                                                     QtGui.QFileDialog.DontUseNativeDialog)[0]
        if filename:
            self.load_settings(filename)

    def load_settings(self, filename):
        print("Load Settings with: ", filename)
        path, file = os.path.split(filename)
        (body_name, extension) = file.split(".")
        copy(os.path.join(self.Form.obj_save_path,  body_name + ".py"), self.Form.path_to_obj_func)
        self.restore_state(filename)

    def get_hyper_file(self):
        #filename = QtGui.QFileDialog.getOpenFileName(self.Form, 'Load Hyper Parameters', filter="txt (*.npy *.)")
        filename, _ = QtGui.QFileDialog.getOpenFileName(self.Form, 'Load Hyper Parameters',
                                                     self.Form.optimizer_path + "parameters", "txt (*.npy)"
                                                     )
        if filename:
            self.Form.hyper_file = str(filename)
            self.pb_hyper_file.setText(self.Form.hyper_file)
            # print(filename)

    def rewrite_default(self):
        #self.Form.set_file = "default.json"
        self.save_state(self.Form.set_file)

    def logbook(self):
        """
        Method to send Optimization parameters + screenshot to eLogboob
        :return:
        """
        self.Form.mi.logbook(self)

    def loadStyleSheet(self):
        """
        Sets the dark GUI theme from a css file.
        :return:
        """
        try:
            self.cssfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "style.css")
            with open(self.cssfile, "r") as f:
                self.Form.setStyleSheet(f.read())
        except IOError:
            print ('No style sheet found!')

    def change_state_scipy_setup(self):
        """
        Method to enable/disable "Scipy Scanner Setup". If scipy version < "0.18" then QGroup will be disable.
        :return:
        """
        #print("SCIPY", str(self.cb_select_alg.currentText()))
        if scipy.__version__ < "0.18" and str(self.cb_select_alg.currentText()) == self.Form.name_simplex:
            #self.cb_use_isim.setCheckState(False)
            self.g_box_isim.setEnabled(False)
            self.g_box_isim.setTitle("Initial Simplex does not work: scipy version: " + scipy.__version__)
            self.g_box_isim.setStyleSheet('QGroupBox  {color: red;}')
        elif scipy.__version__ >= "0.18" and str(self.cb_select_alg.currentText()) == self.Form.name_simplex:
            #print(str(self.cb_select_alg.currentText()))
            self.g_box_isim.setEnabled(True)
            self.cb_use_isim.setEnabled(True)
            self.g_box_isim.setTitle("Simplex/Scipy Scanner Setup")
            self.g_box_isim.setStyleSheet('QGroupBox  {color: white;}')

        if self.cb_use_isim.checkState():
            self.label_23.setEnabled(True)
            self.sb_isim_rel_step.setEnabled(True)
        else:
            self.label_23.setEnabled(False)
            self.sb_isim_rel_step.setEnabled(False)

        if str(self.cb_select_alg.currentText()) == self.Form.name_custom:
            self.g_box_isim.setEnabled(True)
            self.label_23.setEnabled(True)
            self.sb_isim_rel_step.setEnabled(True)
            self.g_box_isim.setTitle("Custom Minimizer Scanner Setup")
            self.g_box_isim.setStyleSheet('QGroupBox  {color: white;}')
            #self.cb_use_isim.setCheckState(True)
            self.cb_use_isim.setEnabled(False)
            self.sb_isim_rel_step.setValue(5)

        if str(self.cb_select_alg.currentText()) in [self.Form.name_simplex_norm, self.Form.name_gauss_sklearn]:
            self.g_box_isim.setEnabled(True)
            self.label_23.setEnabled(True)
            self.sb_isim_rel_step.setEnabled(True)
            self.g_box_isim.setTitle("Simplex With Normalization")
            self.g_box_isim.setStyleSheet('QGroupBox  {color: white;}')
            #self.cb_use_isim.setCheckState(True)
            self.cb_use_isim.setEnabled(False)
            self.sb_isim_rel_step.setValue(5)

        if str(self.cb_select_alg.currentText()) in [self.Form.name_gauss, self.Form.name_gauss_sklearn]:
            self.groupBox_2.setEnabled(True)
            for w in self.groupBox_2.findChildren(QWidget):
                w.setEnabled(True)
        else:
            self.groupBox_2.setEnabled(False)
            for w in self.groupBox_2.findChildren(QWidget):
                w.setEnabled(False)

        if str(self.cb_select_alg.currentText()) in [self.Form.name_es]:
            self.g_box_isim.setEnabled(True)
            self.label_23.setEnabled(True)
            self.sb_isim_rel_step.setEnabled(True)
            self.g_box_isim.setTitle("Extremum Seeking")
            self.g_box_isim.setStyleSheet('QGroupBox  {color: white;}')
            #self.cb_use_isim.setCheckState(True)
            self.cb_use_isim.setEnabled(False)
            self.sb_isim_rel_step.setValue(5)

    def use_predef_fun(self):
        if self.cb_use_predef.checkState():
            self.le_a.setEnabled(False)
            self.le_b.setEnabled(False)
            self.le_c.setEnabled(False)
            self.le_d.setEnabled(False)
            self.le_e.setEnabled(False)
            # self.le_obf.setEnabled(False)

            self.label_16.setEnabled(False)
            self.label_19.setEnabled(False)
            self.label_20.setEnabled(False)
            # self.label_21.setEnabled(False)
            self.label_28.setEnabled(False)
            self.label_29.setEnabled(False)

            self.cb_statistics.setEnabled(True)
            self.pb_edit_obj_func.setEnabled(True)
            self.pb_edit_obj_func.setCursor(QCursor(Qt.PointingHandCursor))
        else:
            self.le_a.setEnabled(True)
            self.le_b.setEnabled(True)
            self.le_c.setEnabled(True)
            self.le_d.setEnabled(True)
            self.le_e.setEnabled(True)
            # self.le_obf.setEnabled(True)

            self.label_16.setEnabled(True)
            self.label_19.setEnabled(True)
            self.label_20.setEnabled(True)
            # self.label_21.setEnabled(True)
            self.label_28.setEnabled(True)
            self.label_29.setEnabled(True)

            self.cb_statistics.setEnabled(False)
            self.pb_edit_obj_func.setEnabled(False)
            self.pb_edit_obj_func.setCursor(QCursor(Qt.ForbiddenCursor))

    def open_help(self):
        """
        method to open the Help in the webbrowser
        :return: None
        """

        if sys.platform == 'win32':
            url = self.Form.optimizer_path+"docs\\_build\\html\\index.html"
            #os.startfile(url)
            webbrowser.open(url)
        elif sys.platform == 'darwin':
            url = "file://"+self.Form.optimizer_path+"docs/_build/html/index.html"
            webbrowser.open(url)
            #subprocess.Popen(['open', url])
        else:
            url = "file://" + self.Form.optimizer_path + "docs/_build/html/index.html"
            try:
                subprocess.Popen(['xdg-open', url])
            except OSError:
                print('Please open a browser on: ' + url)

