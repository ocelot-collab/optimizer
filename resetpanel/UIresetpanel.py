# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UIresetpanel.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(500, 850)
        Form.setMinimumSize(QtCore.QSize(400, 0))
        Form.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        Form.setStyleSheet("background-color: white")
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.tableWidget = QtWidgets.QTableWidget(Form)
        self.tableWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.gridLayout.addWidget(self.tableWidget, 1, 0, 1, 1)
        self.layout_quick_add = QtWidgets.QHBoxLayout()
        self.layout_quick_add.setObjectName("layout_quick_add")
        self.gridLayout.addLayout(self.layout_quick_add, 5, 0, 1, 1)
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
        self.gridLayout.addLayout(self.horizontalLayout, 4, 0, 1, 1)
        self.label = QtWidgets.QLabel(Form)
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
        self.label.setText(_translate("Form", "Middle click a PV then the table to add your favorite device!"))

