"""
PYQT interface for running OCELOT simplex optimization.
Created as a QT widget for use in other applications as well.
Tyler Cope, 2016
The class was modified and was introduced new methods.
S. Tomin, 2017
"""

from __future__ import absolute_import, print_function
import sys

from PyQt5.QtWidgets import QApplication, QFrame
from PyQt5 import QtGui, QtCore, uic
from mint.opt_objects import *

from resetpanel.UIresetpanel import Ui_Form
import os

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
            print('No style sheet found!')

    def getStartValues(self):
        """ Initializes start values for the PV list. """
        for dev in self.devices:
            try:
                self.startValues[dev.eid] = dev.get_value()
            except Exception as ex:
                self.startValues[dev.eid] = None
                print("Get Start Value: ", dev.eid, " not working. Exception was: ", ex)
                # print(self.startValues[dev.eid])
                # self.pv_objects[pv].add_callback(callback=self.PvGetCallBack)

    def updateReference(self):
        """Updates reference values for all PVs on button click."""
        self.ui.updateReference.setText("Getting vals...")
        self.getStartValues()
        for row in range(len(self.pvs)):
            pv = self.pvs[row]
            self.ui.tableWidget.setItem(row, 1, QtGui.QTableWidgetItem(str(np.around(self.startValues[pv], 4))))
        self.ui.updateReference.setText("Update Reference")

    def initTable(self):
        """ Initialize the UI table object """
        headers = ["PVs", "Reference Value", "Current Value"]
        self.ui.tableWidget.setColumnCount(len(headers))
        self.ui.tableWidget.setHorizontalHeaderLabels(headers)
        self.ui.tableWidget.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)  # No user edits on talbe
        self.ui.tableWidget.horizontalHeader().setResizeMode(QtGui.QHeaderView.Stretch)
        for row in range(len(self.pvs)):

            self.ui.tableWidget.setRowCount(row + 1)
            pv = self.pvs[row]
            # put PV in the table
            self.ui.tableWidget.setItem(row, 0, QtGui.QTableWidgetItem(str(pv)))
            #self.ui.tableWidget.item(row, 0).setTextColor(QtGui.QColor(0, 255, 255))
            self.ui.tableWidget.item(row, 0).setForeground(QtGui.QColor(0, 255, 255))
            tip = "/".join(str(pv).split("/")[-2:])
            self.ui.tableWidget.item(row, 0).setToolTip(tip)
            # self.ui.tableWidget.item(row, 0).setFont(font)
            # put start val in
            s_val = self.startValues[pv]
            if s_val != None:
                s_val = np.around(s_val, 4)
            self.ui.tableWidget.setItem(row, 1, QtGui.QTableWidgetItem(str(s_val)))

    def updateCurrentValues(self):
        """
        Method to update the table on every clock cycle.
        Loops through the pv list and gets new data, then updates the Current Value column.
        Hard coded to turn Current Value column red at 0.1% differenct from Ref Value.
        It would be better to update the table on a callback, but PyEpics crashes with cb funcitons.
        """
        percent = 0.001
        self.currentValues = {}
        for row, dev in enumerate(self.devices):
            try:
                value = dev.get_value()
            except:
                # print("ERROR getting value. Device:", dev.eid)
                value = None

            if self.startValues[dev.eid] is None and value is not None:
                self.startValues[dev.eid] = value

            if self.startValues[dev.eid] is None or value is None:
                item = self.ui.tableWidget.item(row, 5)

                if item is None:
                    continue

                item.setFlags(QtCore.Qt.NoItemFlags)
                for col in [0, 5]:
                    self.ui.tableWidget.item(row, col).setBackground(QtGui.QColor(255, 0, 0))  # red

                if self.startValues[dev.eid] is None:
                    self.ui.tableWidget.setItem(row, 1, QtGui.QTableWidgetItem(str("None")))
                    self.ui.tableWidget.item(row, 1).setBackground(QtGui.QColor(255, 0, 0))  # red
                else:
                    self.ui.tableWidget.setItem(row, 1, QtGui.QTableWidgetItem(str(np.around(value, 4))))
                    self.ui.tableWidget.item(row, 1).setBackground(QtGui.QColor(89, 89, 89))  # grey

                if value is None:
                    self.ui.tableWidget.setItem(row, 2, QtGui.QTableWidgetItem(str("None")))
                    self.ui.tableWidget.item(row, 2).setBackground(QtGui.QColor(255, 0, 0))  # red
                else:
                    self.ui.tableWidget.setItem(row, 2,
                                                QtGui.QTableWidgetItem(str(np.around(self.currentValues[dev.eid], 4))))
                    self.ui.tableWidget.item(row, 2).setBackground(QtGui.QColor(89, 89, 89))  # grey

                continue
            # if value out of the limits
            
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

            lim_low, lim_high = dev.get_limits()
            
            # stop update min spinbox if it has focus
            if not self.ui.tableWidget.cellWidget(row, 3).hasFocus():
                spin_box = self.ui.tableWidget.cellWidget(row, 3)
                spin_box.setValue(lim_low)
                spin_box.setEnabled(dev._can_edit_limits)

            # stop update max spinbox if it has focus
            if not self.ui.tableWidget.cellWidget(row, 4).hasFocus():
                spin_box = self.ui.tableWidget.cellWidget(row, 4)
                spin_box.setValue(lim_high)
                spin_box.setEnabled(dev._can_edit_limits)

            pv = dev.eid

            self.currentValues[pv] = value  # dev.get_value()
            if self.ui.tableWidget.item(row, 2) is None:
                self.ui.tableWidget.setItem(row, 2, QtGui.QTableWidgetItem(str(np.around(self.currentValues[pv], 4))))
            else:
                self.ui.tableWidget.item(row, 2).setText(str(np.around(self.currentValues[pv], 4)))
            # print(self.currentValues[pv])
            #print(self.ui.tableWidget.item(row, 2) is None)
            tol = abs(self.startValues[pv] * percent)
            diff = abs(abs(self.startValues[pv]) - abs(self.currentValues[pv]))
            if diff > tol:
                self.ui.tableWidget.item(row, 2).setForeground(QtGui.QColor(255, 101, 101))  # red
            else:
                self.ui.tableWidget.item(row, 2).setForeground(QtGui.QColor(255, 255, 255))  # white

            self.ui.tableWidget.item(row, 5).setFlags(
                QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)

            for col in [0, 1, 2, 5]:
                # self.ui.tableWidget.item(row, col).setFlags(
                #    QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                self.ui.tableWidget.item(row, col).setBackground(QtGui.QColor(89, 89, 89))
            #print("check rest = ", time.time() - start)
            
        QApplication.processEvents()


    def resetAll(self):
        """Set all PVs back to their reference values."""
        for dev in self.devices:
            val = self.startValues[dev.eid]
            dev.set_value(val)  # epics.caput(pv,val)

    def launchPopupAll(self):
        """Launches the ARE YOU SURE popup window for pv reset."""
        self.ui_check = uic.loadUi("UIareyousure.ui")
        self.ui_check.exit.clicked.connect(self.ui_check.close)
        self.ui_check.reset.clicked.connect(self.resetAll)
        self.ui_check.reset.clicked.connect(self.ui_check.close)
        frame_gm = self.ui_check.frameGeometry()
        center_point = QtGui.QDesktopWidget().availableGeometry().center()
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

