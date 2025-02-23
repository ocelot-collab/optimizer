"""
PYQT interface for running OCELOT simplex optimization.
Created as a QT widget for use in other applications as well.
Tyler Cope, 2016
The class was modified and was introduced new methods.
S. Tomin, 2017
"""

from __future__ import absolute_import, print_function
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QFrame
from PyQt5 import QtWidgets
from PyQt5 import QtGui, QtCore, uic
from mint.opt_objects import *

from resetpanel.UIresetpanel import Ui_Form
import os
import logging

logger = logging.getLogger(__name__)

sys.path.append("..")


class ResetpanelWindow(QFrame):
    """
    Main GUI class for the resetpanel.
    """

    def __init__(self, parent=None):

        # initialize
        QFrame.__init__(self)
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # blank data
        self.pvs = []
        self.devices = []
        self.startValues = {}
        self.pv_objects = {}

        # button connections
        self.ui.updateReference.clicked.connect(self.updateReference)
        self.ui.resetAll.clicked.connect(self.launchPopupAll)

        # fast timer start
        self.trackTimer = QtCore.QTimer()
        self.trackTimer.timeout.connect(self.updateCurrentValues)
        self.trackTimer.start(500)  # refresh every 100 ms

        # dark theme
        self.loadStyleSheet()

    def loadStyleSheet(self):
        """ Load in the dark theme style sheet. """
        try:
            self.cssfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "style.css")
            with open(self.cssfile, "r") as f:
                style = f.read()
                self.setStyleSheet(style)
                QApplication.instance().setStyleSheet(style)
        except IOError:
            logger.warning('No style sheet found!')

    def getStartValues(self):
        """ Initializes start values for the PV list. """
        for dev in self.table_devices:
            dev.get_start_val()

    def updateReference(self):
        """Updates reference values for all PVs on button click."""
        self.ui.updateReference.setText("Getting vals...")
        self.getStartValues()
        for dev in self.table_devices:
            dev.set_saved_val(dev.start_value)
        self.ui.updateReference.setText("Update Reference")

    def updateCurrentValues(self):
        """
        Method to update the table on every clock cycle.
        Loops through the pv list and gets new data, then updates the Current Value column.
        Hard coded to turn Current Value column red at 0.1% differenct from Ref Value.
        It would be better to update the table on a callback, but PyEpics crashes with cb funcitons.
        """
        percent = 0.001
        self.currentValues = {}
        for row, td in enumerate(self.table_devices):
            dev = td.device
            try:
                value = dev.get_value()
            except:
                # print("ERROR getting value. Device:", dev.eid)
                value = None

            if "prev_lim_status" not in dev.__dict__:
                dev.prev_lim_status = not dev.check_limits(value)

            if td.start_value is None and value is not None:
                td.saved_value = value
                logger.info(" updateCurrentValues: startValues[{}}] = {}}".format(dev.eid, value))

            if td.start_value is None or value is None:
                item = self.ui.tableWidget.item(row, 6)

                if item is None:
                    continue

                item.setFlags(QtCore.Qt.NoItemFlags)
                for col in [0, 5]:
                    self.ui.tableWidget.item(row, col).setBackground(QtGui.QColor(255, 0, 0))  # red

                if td.start_value is None:
                    self.ui.tableWidget.setItem(row, 1, QtWidgets.QTableWidgetItem(str("None")))
                    self.ui.tableWidget.item(row, 1).setBackground(QtGui.QColor(255, 0, 0))  # red
                else:
                    if value is not None:
                        str_val = str(np.around(value, 4))
                    else:
                        str_val = str(None)
                    self.ui.tableWidget.setItem(row, 1, QtWidgets.QTableWidgetItem(str_val))
                    self.ui.tableWidget.item(row, 1).setBackground(QtGui.QColor(89, 89, 89))  # grey

                if value is None:
                    self.ui.tableWidget.setItem(row, 2, QtWidgets.QTableWidgetItem(str("None")))
                    self.ui.tableWidget.item(row, 2).setBackground(QtGui.QColor(255, 0, 0))  # red
                else:
                    self.ui.tableWidget.setItem(row, 2,
                                                QtWidgets.QTableWidgetItem(str(np.around(self.currentValues[dev.eid], 4))))
                    self.ui.tableWidget.item(row, 2).setBackground(QtGui.QColor(89, 89, 89))  # grey

                continue
            # if value out of the limits
            if dev.prev_lim_status is not dev.check_limits(value):

                if dev.check_limits(value):
                    for col in [3, 4]:
                        spin_box = self.ui.tableWidget.cellWidget(row, col)
                        spin_box.setStyleSheet("color: yellow; font-size: 16px; background-color:red;")

                else:
                    for col in [3, 4]:
                        spin_box = self.ui.tableWidget.cellWidget(row, col)
                        if col == 3:
                            spin_box.setStyleSheet("color: rgb(153,204,255); font-size: 16px; background-color:#595959;")
                        if col == 4:
                            spin_box.setStyleSheet("color: rgb(255,0,255); font-size: 16px; background-color:#595959;")
            dev.prev_lim_status = dev.check_limits(value)

            lim_low, lim_high = dev.get_limits()
            
            # stop update min spinbox if it has focus
            if not (self.ui.tableWidget.cellWidget(row, 3).hasFocus() or self.ui.tableWidget.cellWidget(row, 5).hasFocus()):
                spin_box = self.ui.tableWidget.cellWidget(row, 3)
                td.set_low_lim_from_abs(lim_low)
                spin_box.setEnabled(dev._can_edit_limits)

            # stop update max spinbox if it has focus
            if not (self.ui.tableWidget.cellWidget(row, 4).hasFocus() or self.ui.tableWidget.cellWidget(row, 5).hasFocus()):
                spin_box = self.ui.tableWidget.cellWidget(row, 4)
                td.set_high_lim_from_abs(lim_high)
                spin_box.setEnabled(dev._can_edit_limits)

            pv = dev.eid

            self.currentValues[pv] = value  # dev.get_value()
            if self.ui.tableWidget.item(row, 2) is None:
                self.ui.tableWidget.setItem(row, 2, QtWidgets.QTableWidgetItem(str(np.around(self.currentValues[pv], 4))))
            else:
                self.ui.tableWidget.item(row, 2).setText(str(np.around(self.currentValues[pv], 4)))

            tol = abs(td.start_value * percent)
            diff = abs(abs(td.start_value) - abs(self.currentValues[pv]))
            if diff > tol:
                self.ui.tableWidget.item(row, 2).setForeground(QtGui.QColor(255, 101, 101))  # red
            else:
                self.ui.tableWidget.item(row, 2).setForeground(QtGui.QColor(255, 255, 255))  # white

            self.table_devices[row].set_flags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)

            for col in [0, 1, 2, 6]:
                if row % 2 == 0:
                    self.ui.tableWidget.item(row, col).setBackground(QtGui.QColor(89, 89, 89))
                else:
                    self.ui.tableWidget.item(row, col).setBackground(QtGui.QColor(100, 100, 100))

        QApplication.processEvents()


    def resetAll(self):
        """Set all PVs back to their reference values."""
        for dev in self.devices:
            val = self.startValues[dev.eid]
            logger.info(" resetAll: {} <-- {}".format(dev.eid, val))
            dev.set_value(val)  # epics.caput(pv,val)

    def launchPopupAll(self):
        """Launches the ARE YOU SURE popup window for pv reset."""
        self.ui_check = uic.loadUi("UIareyousure.ui")
        self.ui_check.exit.clicked.connect(self.ui_check.close)
        self.ui_check.reset.clicked.connect(self.resetAll)
        self.ui_check.reset.clicked.connect(self.ui_check.close)
        frame_gm = self.ui_check.frameGeometry()
        center_point = QtWidgets.QDesktopWidget().availableGeometry().center()
        frame_gm.moveCenter(center_point)
        self.ui_check.move(frame_gm.topLeft())
        self.ui_check.show()


def main():
    """
    Main functino to open a resetpanel GUI.
    If passed a file name, will try and load PV list from that file.
    Otherwise defaults to a file in the base directory with pre-loaded common tuned PVs.
    """
    try:  # try to get a pv list file name from commandline arg
        pvs = sys.argv[1]
    except:
        pvs = "./lclsparams"

    app = QApplication(sys.argv)
    window = ResetpanelWindow()
    window.setWindowIcon(QtGui.QIcon('/usr/local/lcls/tools/python/toolbox/py_logo.png'))
    window.getPvList(pvs)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

