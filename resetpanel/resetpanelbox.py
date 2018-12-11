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

from resetpanel.resetpanel import ResetpanelWindow
from PyQt5.QtWidgets import QApplication, QPushButton, QTableWidget, QInputDialog
from PyQt5 import QtGui, QtCore, uic
from PyQt5.QtGui import QClipboard


class customTW(QTableWidget):
    """
    Subclass the tablewidget to add a custom PV on middle click.
    """
    def __init__(self, parent):
        """Init for the table widget, nothing special here"""
        QTableWidget.__init__(self, parent)
        self.parent = parent

    def mouseReleaseEvent(self, evt):
        """
        Grabs string from the clipboard if middle button click.

        Tries to then add the PV to the parent GUI pv lists.
        Calls parent.addPv() method.

        Args:
                evt (QEvent): Event object passed in from QT
        """
        button = evt.button()
        #print("TW Release event: ", button, self.parent.enableMiddleClick)
        if (button == 4) and (self.parent.enableMiddleClick):

            pv = QtGui.QApplication.clipboard().text()
            self.parent.addPv(pv)
            #print(QtGui.QApplication.clipboard().text())
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
            pv = QtGui.QApplication.clipboard().text(mode=QClipboard.Selection)
            self.parent.addPv(pv)
        else:
            QTableWidget.mouseReleaseEvent(self, evt)

    def contextMenuEvent(self, event):
        self.menu = QtGui.QMenu(self)
        if self.selectionModel().selection().indexes():
            rows = []
            for i in self.selectionModel().selection().indexes():
                row, column = i.row(), i.column()
                rows.append(row)
            #self.menu = QtGui.QMenu(self)
            deleteAction = QtGui.QAction('Delete', self)

            deleteAction.triggered.connect(lambda: self.deleteSlot(rows))

            self.menu.addAction(deleteAction)

            #editAction = QtGui.QAction('Edit', self)
            #self.menu.addAction(editAction)
            # add other required actions
            #self.menu.popup(QtGui.QCursor.pos())

        addChannel = QtGui.QAction('Add Channel', self)
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
        dlg.resize(500,100)
        ok = dlg.exec_()
        pv = dlg.textValue()
        #pv, ok = QInputDialog.getText(self, "Input", "Enter channel")
        if ok:
            print(pv)
            self.parent.addPv(pv)

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

        self.check = QPushButton(self)
        self.check.setText('Check')
        self.uncheck = QPushButton(self)
        self.uncheck.setText('Uncheck')
        self.ui.horizontalLayout.addWidget(self.check)
        self.ui.horizontalLayout.addWidget(self.uncheck)
        self.check.clicked.connect(lambda: self.getRows(2))
        self.uncheck.clicked.connect(lambda: self.getRows(0))
        self.mi = None

        #make button text bigger
        #self.check.setStyleSheet('font-size: 18pt; font-family: Courier;')

        #enable middle click method
        self.enableMiddleClick = True

        #make the custom table for middle click
        self.ui.tableWidget.setParent(None) #remove old table
        self.ui.tableWidget = customTW(self) # make new widget
        self.ui.gridLayout.addWidget(self.ui.tableWidget,0,0)
        #self.ui.tableWidget.itemClicked.connect(self.con)


    def mouseReleaseEvent(self, evt):
        """
        Get PV coppied from the system clipboard on button release

        Button 4 is the middle click button.
        """
        button = evt.button()
        #print("Release event: ", button, self.parent.enableMiddleClick)
        if (button == 4) and (self.enableMiddleClick):
            pv = QtGui.QApplication.clipboard().text(mode=QClipboard.Selection)
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
            except:
                print ("bad string")
                continue

            state = dev.state()
            print("state=", state)
            if state:
                self.pvs.append(pv)
                self.devices.append(dev)
                self.getStartValues()

        table = self.get_state()
        self.initTable()
        self.addCheckBoxes()
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
        if pvs_in == None:
            #print ('Exiting', pvs_in)
            return

        if type(pvs_in) != list:
            self.pvs = []
            #print(pvs_in)
            for line in open(pvs_in):
                l = line.rstrip('\n')
                if l[0] == '#':
                    continue
                self.pvs.append(str(l))
        else:
            self.pvs = pvs_in

        #print ("PVS LOADED", self.pvs)
        self.devices = self.parent.create_devices(self.pvs)
        self.getStartValues()
        self.initTable()
        self.addCheckBoxes()

    def get_state(self):
        devs = {"id":[], "lims": [], "checked": []}
        for row in range(self.ui.tableWidget.rowCount()):
            name = str(self.ui.tableWidget.item(row, 0).text())
            devs["id"].append(name)
            devs["lims"].append([self.ui.tableWidget.cellWidget(row, 3).value(), self.ui.tableWidget.cellWidget(row, 4).value()])
            devs["checked"].append(self.ui.tableWidget.item(row, 5).checkState())
        return devs

    def get_limits(self, pv):
        for row in range(self.ui.tableWidget.rowCount()):
            if pv == str(self.ui.tableWidget.item(row, 0).text()):
                lims = [self.ui.tableWidget.cellWidget(row, 3).value(), self.ui.tableWidget.cellWidget(row, 4).value()]
                return lims
        return None

    def set_state(self, table, force_active=False):
        for row in range(self.ui.tableWidget.rowCount()):
            pv = str(self.ui.tableWidget.item(row, 0).text())
            item = self.ui.tableWidget.item(row, 5)

            if force_active:
                item.setCheckState(QtCore.Qt.Checked)

            if pv in table["id"]:
                indx = table["id"].index(pv)
                self.ui.tableWidget.cellWidget(row, 3).setValue(table["lims"][indx][0])
                self.ui.tableWidget.cellWidget(row, 4).setValue(table["lims"][indx][1])
                if item.flags() == QtCore.Qt.NoItemFlags:
                   continue

                state = table.get("checked", None)
                if state is not None:
                    item.setCheckState(state[indx])

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
            #item = self.ui.tableWidget.cellWidget(idx.row(), 5)
            item = self.ui.tableWidget.item(idx.row(), 5)
            if item.flags() == QtCore.Qt.NoItemFlags:
                print("item disabled")
                continue
            item.setCheckState(state)
            #print item.text()
        #print rows

    def addCheckBoxes(self):
        """
        Creates additional column in UI table for check box.

        Must be called again if the user adds another PV with middle click function.
        """
        headers = ["PVs", "Saved Val.", "Current Val.", "Min", "Max", "Active"]
        self.ui.tableWidget.setColumnCount(len(headers))
        self.ui.tableWidget.setHorizontalHeaderLabels(headers)
        header = self.ui.tableWidget.horizontalHeader()
        header.setResizeMode(0, QtGui.QHeaderView.Stretch)
        header.setResizeMode(1, QtGui.QHeaderView.ResizeToContents)
        header.setResizeMode(2, QtGui.QHeaderView.ResizeToContents)
        header.setResizeMode(3, QtGui.QHeaderView.ResizeToContents)
        header.setResizeMode(4, QtGui.QHeaderView.ResizeToContents)
        header.setResizeMode(5, QtGui.QHeaderView.Fixed)
        for row in range(len(self.pvs)):
            eng = QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates)
            for i in range(2):
                spin_box = QtGui.QDoubleSpinBox()
                spin_box.setMaximumWidth(85)
                if i == 0:
                    spin_box.setStyleSheet("color: rgb(153,204,255); font-size: 12px; background-color:#595959;")
                else:
                    spin_box.setStyleSheet("color: rgb(255,0,255); font-size: 12px; background-color:#595959;")
                spin_box.setLocale(eng)
                spin_box.setDecimals(3)
                spin_box.setMaximum(999999)
                spin_box.setMinimum(-999999)
                spin_box.setSingleStep(0.1)
                spin_box.setAccelerated(True)
                if i == 0:  # Running for low limit spin box
                    spin_box.valueChanged.connect(functools.partial(self.spinbox_changed, high_limit=False, row=row))
                else: # Running for the high limit spin box
                    spin_box.valueChanged.connect(functools.partial(self.spinbox_changed, high_limit=True, row=row))
                self.ui.tableWidget.setCellWidget(row, 3+i, spin_box)
                self.ui.tableWidget.resizeColumnsToContents()

            #spin_box
            checkBoxItem = QtGui.QTableWidgetItem()
            checkBoxItem.setCheckState(QtCore.Qt.Checked)
            flags = checkBoxItem.flags()
            checkBoxItem.setFlags(flags)
            self.ui.tableWidget.setItem(row, 5, checkBoxItem)

    def spinbox_changed(self, val, high_limit, row):
        """
        Callback for when a spinbox is changed on the table so we can hook it up with the device limits.

        :param val: (float) The new limit value
        :param high_limit: (bool) Wether or not this is the high limit value
        :param row: (int) The row number
        """
        device = self.devices[row]
        if high_limit:
            device.set_high_limit(val)
        else:
            device.set_low_limit(val)

    def setBoxes(self, checked=True):
        """
        Method to check or uncheck all the boxes.
        :param checked: (bool)
        """
        if checked:
            val = QtCore.Qt.Checked
        else:
            val = QtCore.Qt.Unchecked

        for row in range(len(self.pvs)):
            item = self.ui.tableWidget.item(row, 5)
            item.setCheckState(val)

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
        for row, dev in enumerate(self.devices):
            val = self.startValues[dev.eid]
            state = self.ui.tableWidget.item(row, 5).checkState()
            if state == 2:
                dev.set_value(val)
                #epics.caput(pv, val)

    def updateReference(self):
        """
        Update reference values for selected rows.

        Rewrote this function to only update selected rows, not all.
        """
        self.ui.updateReference.setText("Getting vals...")
        for row, dev in enumerate(self.devices):
            pv = self.pvs[row]
            state = self.ui.tableWidget.item(row, 5).checkState()
            if state == 2:
                val = dev.get_value()
                if val is not None:
                    self.startValues[pv] = val
                    self.ui.tableWidget.setItem(row, 1, QtGui.QTableWidgetItem(str(np.around(self.startValues[pv], 4))))
        self.ui.updateReference.setText("Update Reference")

    def getPvsFromCbState(self):
        """
        Gets list of all pvs that have checked boxes.

        Returns:
                List of PV strings
        """
        pvs = []
        for row in range(len(self.pvs)):
            state = self.ui.tableWidget.item(row, 5).checkState()
            if state == 2:
                pvs.append(str(self.ui.tableWidget.item(row, 0).text()))
        return pvs

    #switch string from to SELECTED
    def launchPopupAll(self):
        """
        Launches the ARE YOU SURE popup for pv reset value function.

        Rewrote to change the warning string to say "checkbox selected" instead of "all" to avoid confusion with number
        of devices being reset.
        """
        #self.ui_check = uic.loadUi("/home/physics/tcope/python/tools/resetpanel/UIareyousure.ui")
        path = os.path.dirname(os.path.realpath(__file__))
        self.ui_check = uic.loadUi(os.path.join(path, "UIareyousure.ui"))
        self.ui_check.exit.clicked.connect(self.ui_check.close)
        self.ui_check.reset.clicked.connect(self.resetAll)
        self.ui_check.reset.clicked.connect(self.ui_check.close)
        self.ui_check.label.setText("Are you sure you want to implement \nchanges to checkbox selected PVs?")
        frame_gm = self.ui_check.frameGeometry()
        center_point = QtGui.QDesktopWidget().availableGeometry().center()
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
