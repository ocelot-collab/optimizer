# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UIresetpanel.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(517, 718)
        Form.setMinimumSize(QtCore.QSize(400, 0))
        Form.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        Form.setStyleSheet("background-color: white")
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.updateReference = QtWidgets.QPushButton(Form)
        font = QtGui.QFont()
        font.setFamily("DejaVu Sans")
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.updateReference.setFont(font)
        self.updateReference.setStyleSheet("color: orange")
        self.updateReference.setObjectName("updateReference")
        self.horizontalLayout.addWidget(self.updateReference)
        self.resetAll = QtWidgets.QPushButton(Form)
        font = QtGui.QFont()
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.resetAll.setFont(font)
        self.resetAll.setStyleSheet("color: red")
        self.resetAll.setObjectName("resetAll")
        self.horizontalLayout.addWidget(self.resetAll)
        self.check = QtWidgets.QPushButton(Form)
        self.check.setObjectName("check")
        self.horizontalLayout.addWidget(self.check)
        self.uncheck = QtWidgets.QPushButton(Form)
        self.uncheck.setObjectName("uncheck")
        self.horizontalLayout.addWidget(self.uncheck)
        self.gridLayout.addLayout(self.horizontalLayout, 4, 0, 1, 1)
        self.tableWidget = QtWidgets.QTableWidget(Form)
        self.tableWidget.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.gridLayout.addWidget(self.tableWidget, 1, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.line = QtWidgets.QFrame(self.groupBox)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout_2.addWidget(self.line, 1, 3, 1, 1)
        self.cb_ref_value = QtWidgets.QCheckBox(self.groupBox)
        self.cb_ref_value.setObjectName("cb_ref_value")
        self.gridLayout_2.addWidget(self.cb_ref_value, 1, 2, 1, 1)
        self.sb_ref_value = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.sb_ref_value.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.sb_ref_value.setDecimals(5)
        self.sb_ref_value.setMinimum(-10000.0)
        self.sb_ref_value.setMaximum(10000.0)
        self.sb_ref_value.setSingleStep(0.1)
        self.sb_ref_value.setObjectName("sb_ref_value")
        self.gridLayout_2.addWidget(self.sb_ref_value, 1, 1, 1, 1)
        self.pb_set_group_lims = QtWidgets.QPushButton(self.groupBox)
        self.pb_set_group_lims.setObjectName("pb_set_group_lims")
        self.gridLayout_2.addWidget(self.pb_set_group_lims, 1, 7, 1, 1)
        self.sb_delta = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.sb_delta.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.sb_delta.setDecimals(5)
        self.sb_delta.setMaximum(10000.0)
        self.sb_delta.setSingleStep(0.1)
        self.sb_delta.setObjectName("sb_delta")
        self.gridLayout_2.addWidget(self.sb_delta, 1, 5, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 1, 4, 1, 1)
        self.horizontalLayout_2.addWidget(self.groupBox)
        self.gridLayout.addLayout(self.horizontalLayout_2, 3, 0, 1, 1)
        self.layout_quick_add = QtWidgets.QHBoxLayout()
        self.layout_quick_add.setObjectName("layout_quick_add")
        self.gridLayout.addLayout(self.layout_quick_add, 5, 0, 1, 1)
        self.label = QtWidgets.QLabel(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setItalic(True)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 2, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Reset Panel"))
        self.updateReference.setText(_translate("Form", "Update reference "))
        self.resetAll.setText(_translate("Form", "Reset All"))
        self.check.setText(_translate("Form", "Check"))
        self.uncheck.setText(_translate("Form", "Uncheck"))
        self.groupBox.setTitle(_translate("Form", "Limit Group Control"))
        self.cb_ref_value.setText(_translate("Form", "Ref. Val"))
        self.pb_set_group_lims.setText(_translate("Form", "Set"))
        self.label_2.setText(_translate("Form", "Delta"))
        self.label.setText(_translate("Form", "Middle click a PV then the table to add your favorite device!"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
