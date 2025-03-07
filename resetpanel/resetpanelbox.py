#!/usr/local/lcls/package/python/current/bin/python
"""
Subclass of ResetpanelWindow, used to make a version with check box selction

Tyler Cope, 2016

The class was modified and was introduced new methods.

S. Tomin, 2017
"""

from __future__ import absolute_import, print_function
import os, sys
import functools
import numpy as np
import traceback

from resetpanel.resetpanel import ResetpanelWindow
from PyQt5.QtWidgets import QApplication, QPushButton, QTableWidget, QInputDialog
from PyQt5 import QtWidgets
from PyQt5 import QtGui, QtCore, uic
from PyQt5.QtGui import QClipboard
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent, QSound
from PyQt5.QtCore import QUrl
import logging

logger = logging.getLogger(__name__)

class customTW(QTableWidget):
    """
    Subclass the tablewidget to add a custom PV on middle click.
    """
    def __init__(self, parent):
        """Init for the table widget, nothing special here"""
        QTableWidget.__init__(self, parent)
        #self.setMinimumWidth(800)
        self.parent = parent
        self.setAlternatingRowColors(True)

    def mouseReleaseEvent(self, evt):
        """
        Grabs string from the clipboard if middle button click.

        Tries to then add the PV to the parent GUI pv lists.
        Calls parent.addPv() method.

        Args:
                evt (QEvent): Event object passed in from QT
        """
        button = evt.button()
        if (button == 4) and (self.parent.enableMiddleClick):
            pv = QtWidgets.QApplication.clipboard().text()
            self.parent.addPv(pv)
        else:
            QTableWidget.mouseReleaseEvent(self, evt)

    def mouseReleaseEvent_mclick(self, evt):
        """
        Grabs string from the clipboard if middle button click.

        Tries to then add the PV to the parent GUI pv lists.
        Calls parent.addPv() method.

        Args:
                evt (QEvent): Event object passed in from QT
        """
        button = evt.button()
        if (button == 4) and (self.parent.enableMiddleClick):
            pv = QtWidgets.QApplication.clipboard().text(mode=QClipboard.Selection)
            self.parent.addPv(pv)
        else:
            QTableWidget.mouseReleaseEvent(self, evt)

    def contextMenuEvent(self, event):
        self.menu = QtWidgets.QMenu(self)
        if self.selectionModel().selection().indexes():
            rows = []
            for i in self.selectionModel().selection().indexes():
                row, column = i.row(), i.column()
                rows.append(row)
            #self.menu = QtGui.QMenu(self)
            deleteAction = QtWidgets.QAction('Delete', self)

            deleteAction.triggered.connect(lambda: self.deleteSlot(rows))

            self.menu.addAction(deleteAction)

            #editAction = QtGui.QAction('Edit', self)
            #self.menu.addAction(editAction)
            # add other required actions
            #self.menu.popup(QtGui.QCursor.pos())

        addChannel = QtWidgets.QAction('Add Channel', self)
        addChannel.triggered.connect(self.addRow)
        self.menu.addAction(addChannel)
        self.menu.popup(QtGui.QCursor.pos())

    def deleteSlot(self, rows):
        print ("delete rows called", rows)
        for row in rows[::-1]:
            self.parent.ui.tableWidget.removeRow(row)

        table = self.parent.get_state()
        pvs = table["id"]
        self.parent.getPvList(pvs)
        self.parent.uncheckBoxes()
        self.parent.set_state(table)

    def addRow(self):
        dlg = QInputDialog(self)
        dlg.setInputMode(QInputDialog.TextInput)
        dlg.setLabelText("Channel:")
        dlg.resize(500, 100)
        ok = dlg.exec_()
        pv = dlg.textValue()
        #pv, ok = QInputDialog.getText(self, "Input", "Enter channel")
        if ok:
            print(pv)
            self.parent.addPv(pv)


class DeviceTableClass:
    def __init__(self, row, table, device):
        self.row = row
        self.table = table
        self.device = device
        self.pv = device.eid
        self._start_value = self.get_start_value()  # Internal storage for start_value

    @property
    def start_value(self):
        return self._start_value

    @start_value.setter
    def start_value(self, new_value):
        self._start_value = new_value

    def add_to_table(self):
        eng = QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates)
        row = self.row
        self.table.setRowCount(row + 1)

        # Column 0: PV (Device Name)
        self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(self.pv)))
        self.table.item(row, 0).setForeground(QtGui.QColor(0, 255, 255))

        # Column 1: Saved Value (Initial)
        s_val = self.start_value
        if s_val is not None:
            s_val = np.around(s_val, 4)
        self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(s_val)))

        # Column 2: Current Value (You may update this elsewhere)

        # Columns 3 and 4: Spinboxes for Min and Max
        for i in range(2):
            spin_box = QtWidgets.QDoubleSpinBox()
            spin_box.setMaximumWidth(85)
            spin_box.setLocale(eng)
            spin_box.setDecimals(3)
            spin_box.setMaximum(999999)
            spin_box.setMinimum(-999999)
            spin_box.setSingleStep(0.1)
            spin_box.setAccelerated(True)

            if i == 0:  # Min limit
                spin_box.setStyleSheet("color: rgb(153,204,255); font-size: 16px; background-color:#595959;")
                spin_box.valueChanged.connect(functools.partial(self.spinbox_changed, high_limit=False))
            else:  # Max limit
                spin_box.setStyleSheet("color: rgb(255,0,255); font-size: 16px; background-color:#595959;")
                spin_box.valueChanged.connect(functools.partial(self.spinbox_changed, high_limit=True))

            self.table.setCellWidget(row, 3 + i, spin_box)

        # Column 5: Active Checkbox
        checkBoxItem = QtWidgets.QTableWidgetItem()
        checkBoxItem.setCheckState(QtCore.Qt.Checked)
        checkBoxItem.setFlags(checkBoxItem.flags())
        self.table.setItem(row, 5, checkBoxItem)

    def spinbox_changed(self, val, high_limit):
        """
        Callback for when a spinbox is changed in the table.
        """
        if high_limit:
            self.device.set_high_limit(val)
        else:
            self.device.set_low_limit(val)

    def get_start_value(self):
        try:
            val = self.device.get_value()
            start_value = val
            logger.info(" getStartValues: startValues[{}] <-- {}".format(self.device.eid, val))
        except Exception as ex:
            start_value = None
            logger.warning("Get Start Value: " + str(self.device.eid) + " not working. Exception was: " + str(ex))
        return start_value

    def set_box(self, checked=True):
        """
        Method to check or uncheck box.
        :param checked: (bool)
        """
        if checked:
            val = QtCore.Qt.Checked
        else:
            val = QtCore.Qt.Unchecked

        item = self.table.item(self.row, 5)
        item.setCheckState(val)

    def set_saved_val(self, val):
        self.table.setItem(self.row, 1, QtWidgets.QTableWidgetItem(str(val)))

    def get_name(self):
        return str(self.table.item(self.row, 0).text())

    def get_lims(self):
        return [self.table.cellWidget(self.row, 3).value(), self.table.cellWidget(self.row, 4).value()]

    def set_lims(self, lims):
        lo, hi = lims
        self.set_low_lim(lo)
        self.set_high_lim(hi)

    def set_low_lim(self, low_limit):
        """
        set spin box value from device which uses only absolute limits
        """
        min_spinbox = self.table.cellWidget(self.row, 3)
        min_spinbox.setValue(low_limit)

    def set_high_lim(self, high_limit):
        """
        set spin box value from device which uses only absolute limits
        """
        max_spinbox = self.table.cellWidget(self.row, 4)
        max_spinbox.setValue(high_limit)

    def get_check_state(self):
        return self.table.item(self.row, 5).checkState()

    def set_check_state(self, state):
        self.table.item(self.row, 5).setCheckState(state)

    def get_flags(self):
        self.table.item(self.row, 5).flags()

    def set_flags(self, flags):
        self.table.item(self.row, 5).setFlags(flags)


class ResetpanelBoxWindow(ResetpanelWindow):
    """
    The main GUI class to add in checkboxes, subclassed from the ResetPanelWindow class.
    """
    def __init__(self,  parent=None):
        """
        Init method, adds in some new UI changes.

        Adds two buttons, to check and uncheck selected rows.
        Add in subclassed table to enable middle click PV add.
        """

        #initialize

        ResetpanelWindow.__init__(self)


        self.ui.check.clicked.connect(lambda: self.getRows(2))
        self.ui.uncheck.clicked.connect(lambda: self.getRows(0))
        self.mi = None
        self.table_devices = []

        #make button text bigger
        #self.check.setStyleSheet('font-size: 18pt; font-family: Courier;')

        #enable middle click method
        self.enableMiddleClick = True

        #make the custom table for middle click
        self.ui.tableWidget.setParent(None) #remove old table
        self.ui.tableWidget = customTW(self) # make new widget
        self.ui.gridLayout.addWidget(self.ui.tableWidget, 0, 0)
        self.ui.pb_set_group_lims.clicked.connect(self.set_limits_to_selected_rows)
        self.ui.sb_ref_value.setEnabled(self.ui.cb_ref_value.isChecked())
        self.ui.cb_ref_value.stateChanged.connect(self.toggle_ref_value)


    def toggle_ref_value(self, state):
        self.ui.sb_ref_value.setEnabled(state == 2)


    def set_limits_to_selected_rows(self):
        """
        Method to set the UI checkbox state from slected rows.

        Loops though the rows and gets the selected state from the 'Active" column.
        If highlighted, check box is set the the 'state' input arg.

        Args:
                state (bool): Bool of whether the boxes should be checked or unchecked.
        """
        rows=[]
        for idx in self.ui.tableWidget.selectedIndexes():
            rows.append(idx.row())
            row = idx.row()
            delta = self.ui.sb_delta.value()

            if self.ui.cb_ref_value.isChecked():
                lo = self.ui.sb_ref_value.value() - delta
                hi = self.ui.sb_ref_value.value() + delta
            else:
                lo = self.table_devices[row].start_value - delta
                hi = self.table_devices[row].start_value + delta
            self.table_devices[row].set_lims(lims=[lo, hi])

    def mouseReleaseEvent(self, evt):
        """
        Get PV coppied from the system clipboard on button release

        Button 4 is the middle click button.
        """
        button = evt.button()
        if (button == 4) and (self.enableMiddleClick):
            pv = QtWidgets.QApplication.clipboard().text(mode=QClipboard.Selection)
            self.addPv(pv)

    def set_parent(self, parent):
        self.parent = parent

    def addPv(self, pv, force_active=False):
        """
        Add another PV to the GUI on middle click.

        :param pv: (str or list): String name of the PV to add or list of strings to add
        :param force_active: (bool): Whether or not to force the checkbox to be checked. Default is False.
        """
        if sys.version_info[0] >= 3 and isinstance(pv, str):
            pvs = [pv]
        elif sys.version_info[0] < 3 and isinstance(pv, unicode):
            pvs = [pv]
        else:
            pvs = pv

        for pv in pvs:
            pv = str(pv)
            if pv in self.pvs:
                print ("PV already in list: ", pv)
                continue
            try:
                dev = self.parent.create_devices(pvs=[pv])[0]
            except Exception as e:
                print ("Failed to create device {}. Error was: ".format(pv), e)
                traceback.print_exc()
                continue

            state = dev.state()
            print("state=", state)
            if state:
                self.pvs.append(pv)
                self.devices.append(dev)

        table = self.get_state()
        self.table_devices = self.init_table()
        if not force_active:
            self.uncheckBoxes()
        self.set_state(table, force_active=force_active)

    def get_devices(self, pvs):
        d_pvs = [dev.eid for dev in self.devices]
        inxs = [d_pvs.index(pv) for pv in pvs]
        return [self.devices[inx] for inx in inxs]

    def set_machine_interface(self, mi):
        self.mi = mi
        self.getPvList(self.pvs)

    def getPvList(self, pvs_in=None):
        """
        Redefine method to add in checkboxes when getting the PV list.

        Copied from resetpanel.py
        """
        print ("LOADING:")
        if pvs_in is None:
            return

        if type(pvs_in) != list:
            self.pvs = []
            for line in open(pvs_in):
                l = line.rstrip('\n')
                if l[0] == '#':
                    continue
                self.pvs.append(str(l))
        else:
            self.pvs = pvs_in

        self.devices = self.parent.create_devices(self.pvs)
        self.table_devices = self.init_table()

    def get_state(self):
        devs = {"id":[], "lims": [], "checked": [], "mode": []}
        for row in range(self.ui.tableWidget.rowCount()):
            devs["id"].append(self.table_devices[row].get_name())
            devs["lims"].append(self.table_devices[row].get_lims())
            devs["checked"].append(self.table_devices[row].get_check_state())
            devs["mode"].append("Absolute")
        return devs

    def get_limits(self, pv):
        for row in range(self.ui.tableWidget.rowCount()):
            if pv == self.table_devices[row].get_name():
                lims = self.table_devices[row].get_lims()
                return lims
        return None

    def set_state(self, table, force_active=False):
        print(table)
        for row in range(self.ui.tableWidget.rowCount()):
            pv = self.table_devices[row].get_name()
            if force_active:
                self.table_devices[row].set_check_state(QtCore.Qt.Checked)

            if pv in table["id"]:
                indx = table["id"].index(pv)

                self.table_devices[row].set_lims(table["lims"][indx])
                if self.table_devices[row].get_flags() == QtCore.Qt.NoItemFlags:
                   continue

                state = table.get("checked", None)
                if state is not None:
                    self.table_devices[row].set_check_state(state[indx])

    def getRows(self, state):
        """
        Method to set the UI checkbox state from slected rows.

        Loops though the rows and gets the selected state from the 'Active" column.
        If highlighted, check box is set the the 'state' input arg.

        Args:
                state (bool): Bool of whether the boxes should be checked or unchecked.
        """
        rows=[]
        for idx in self.ui.tableWidget.selectedIndexes():
            rows.append(idx.row())
            row = idx.row()
            if self.table_devices[row].get_flags() == QtCore.Qt.NoItemFlags:
                print("item disabled")
                continue
            self.table_devices[row].set_check_state(state)

    def init_table(self):
        table_devices = []
        headers = ["PVs", "Saved Val.", "Current Val.", "Min", "Max", "Active"]
        self.ui.tableWidget.setColumnCount(len(headers))
        self.ui.tableWidget.setHorizontalHeaderLabels(headers)
        header = self.ui.tableWidget.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QtWidgets.QHeaderView.Fixed)
        for row in range(len(self.devices)):
            pv = self.devices[row].eid
            table_dev = DeviceTableClass(row=row, table=self.ui.tableWidget, device=self.devices[row])
            table_dev.add_to_table()
            table_devices.append(table_dev)
        return table_devices

    def setBoxes(self, checked=True):
        """
        Method to check or uncheck all the boxes.
        :param checked: (bool)
        """
        for td in self.table_devices:
            td.set_box(checked)

    def uncheckBoxes(self):
        """ Method to unchecked all active boxes """
        self.setBoxes(False)

    def checkBoxes(self):
        """ Method to unchecked all active boxes """
        self.setBoxes()

    def resetAll(self):
        """
        Resets all PVs with a box checked.

        Rewrote this function to only change selected rows, not all rows.
        """
        for row, td in enumerate(self.table_devices):
            val = td.start_value
            state = td.get_check_state() #self.ui.tableWidget.item(row, 5).checkState()
            if state == 2:
                logger.info(" resetAll: {} <-- {}".format(td.device.eid, val))
                td.device.set_value(val)
                #epics.caput(pv, val)

    def updateReference(self):
        """
        Update reference values for selected rows.

        Rewrote this function to only update selected rows, not all.
        """
        self.ui.updateReference.setText("Getting vals...")
        for row, td in enumerate(self.table_devices):
            pv = self.pvs[row]
            state = self.table_devices[row].get_check_state() # self.ui.tableWidget.item(row, 5).checkState()
            if state == 2:
                val = td.device.get_value()
                if val is not None:
                    td.start_value = val
                    logger.info(" updateReference: startValues[{}] <-- {}".format(td.device.eid, val))
                    td.set_saved_val(val=np.around(val, 4))
        self.ui.updateReference.setText("Update Reference")

    def getPvsFromCbState(self):
        """
        Gets list of all pvs that have checked boxes.

        Returns:
                List of PV strings
        """
        pvs = []
        for row in range(len(self.pvs)):
            state = self.table_devices[row].get_check_state()# ui.tableWidget.item(row, 5).checkState()
            if state == 2:
                pvs.append(self.table_devices[row].get_name())
        return pvs

    #switch string from to SELECTED
    def launchPopupAll(self):
        """
        Launches the ARE YOU SURE popup for pv reset value function.

        Rewrote to change the warning string to say "checkbox selected" instead of "all" to avoid confusion with number
        of devices being reset.
        """
        path = os.path.dirname(os.path.realpath(__file__))
        self.ui_check = uic.loadUi(os.path.join(path, "UIareyousure.ui"))
        self.ui_check.exit.clicked.connect(self.ui_check.close)
        self.ui_check.reset.clicked.connect(self.resetAll)
        self.ui_check.reset.clicked.connect(self.ui_check.close)
        self.ui_check.label.setText("Are you sure you want to implement \nchanges to checkbox selected PVs?")
        frame_gm = self.ui_check.frameGeometry()
        center_point = QtWidgets.QDesktopWidget().availableGeometry().center()
        frame_gm.moveCenter(center_point)
        self.ui_check.move(frame_gm.topLeft())
        self.ui_check.show()


def main():
    """
    Start up the main program if launch from comamnd line.
    """
    import sys
    try:
        pvs = sys.argv[1]
    except:
        pvs = "./lclsparams"

    app = QApplication(sys.argv)
    window = ResetpanelBoxWindow()
    window.setWindowIcon(QtGui.QIcon('/usr/local/lcls/tools/python/toolbox/py_logo.png'))
    window.getPvList(pvs)
    window.uncheckBoxes()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
