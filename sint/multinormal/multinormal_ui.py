import numpy as np
from PyQt5.QtWidgets import (QWidget, QLabel, QLineEdit, QTableWidget,
                             QPushButton, QHBoxLayout, QVBoxLayout, QFormLayout,
                             QGridLayout, QTableWidgetItem)


class MultinormalDisplay(QWidget):
    def __init__(self, parent, mi=None):
        super(MultinormalDisplay, self).__init__(parent=parent)
        self.setup_ui()
        self.mi = mi
        self.fill_from_mi()

    def setup_ui(self):
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        lbl_title = QLabel(parent=self)
        lbl_title.setText("Multivariate Normal Distribution with Correlations")
        main_layout.addWidget(lbl_title)

        layout_configs = QHBoxLayout()
        main_layout.addLayout(layout_configs)

        layout_frm_features = QFormLayout()
        layout_configs.addLayout(layout_frm_features)
        lbl_num_features = QLabel()
        lbl_num_features.setText("Number of Features")
        self.txt_num_features = QLineEdit()
        self.txt_num_features.setEnabled(False)
        layout_frm_features.addRow(lbl_num_features, self.txt_num_features)

        layout_frm_amplitude = QFormLayout()
        layout_configs.addLayout(layout_frm_amplitude)
        lbl_obj_amplitude = QLabel()
        lbl_obj_amplitude.setText("Objective Amplitude")
        self.txt_obj_amplitude = QLineEdit()
        self.txt_obj_amplitude.returnPressed.connect(self.update_objective_peak)
        layout_frm_amplitude.addRow(lbl_obj_amplitude, self.txt_obj_amplitude)

        layout_tables = QGridLayout()
        main_layout.addLayout(layout_tables)

        lbl_means = QLabel()
        lbl_means.setText("Centroid and Proj. Widths")
        layout_tables.addWidget(lbl_means, 0, 0)
        self.tbl_centroids = QTableWidget()
        self.tbl_centroids.setMaximumWidth(230)
        self.tbl_centroids.setColumnCount(2)
        self.tbl_centroids.setHorizontalHeaderLabels(['Centroid', 'Width'])
        layout_tables.addWidget(self.tbl_centroids, 1, 0)

        layout_corr_control = QHBoxLayout()
        layout_tables.addLayout(layout_corr_control, 0, 1)
        lbl_corr = QLabel()
        lbl_corr.setText("Correlation Matrix")
        layout_corr_control.addWidget(lbl_corr)
        self.btn_clear_corr = QPushButton()
        self.btn_clear_corr.setText("Clear Correlations")
        layout_corr_control.addWidget(self.btn_clear_corr)

        self.tbl_corr = QTableWidget()
        layout_tables.addWidget(self.tbl_corr, 1, 1)

    def fill_from_mi(self):
        # Lets fill in the UI with the values from the Machine Interface
        # Add the Number of Features
        self.txt_num_features.setText(str(self.mi.ndims))

        # Add the Objective Amplitude
        self.txt_obj_amplitude.setText(str(self.mi.sigAmp))

        # setup correlation matrix table
        self.tbl_corr.setRowCount(len(self.mi.corrmat))
        self.tbl_corr.setColumnCount(len(self.mi.corrmat[0]))
        for i, row in enumerate(self.mi.corrmat):
            for j, val in enumerate(row):
                self.tbl_corr.setItem(i, j, QTableWidgetItem(str(val)))

        self.tbl_corr.resizeColumnsToContents()
        self.tbl_corr.resizeRowsToContents()

        # connect actions to slots
        self.tbl_corr.itemChanged.connect(self.symmetrize)
        self.btn_clear_corr.released.connect(self.clear_correlations)

        # setup widths and centroids table
        self.tbl_centroids.setRowCount(len(self.mi.offsets))
        self.tbl_centroids.setColumnCount(2)
        for i, val in enumerate(self.mi.offsets):
            self.tbl_centroids.setItem(i, 0,
                                       QTableWidgetItem(str(val)))
        for i, val in enumerate(self.mi.sigmas):
            self.tbl_centroids.setItem(i, 1,
                                       QTableWidgetItem(str(val)))

        # resize cells to fit data
        self.tbl_centroids.resizeColumnsToContents()
        self.tbl_centroids.resizeRowsToContents()

        self.tbl_centroids.itemChanged.connect(self.update_sim_moments)

    def update_objective_peak(self):
        try:
            val = float(self.txt_obj_amplitude.text())
        except:
            val = 1.  # default offset

        # update sim
        self.mi.sigAmp = val
        print("Multinormal simulation objective set to ", self.mi.sigAmp)

    def symmetrize(self):
        self.tbl_corr.itemChanged.disconnect(self.symmetrize)

        for item in self.tbl_corr.selectedIndexes():
            # get item value
            itemVal = self.tbl_corr.item(item.row(),
                                                       item.column()).text()
            try:
                itemVal = float(self.tbl_corr.item(item.row(),
                                                                 item.column()).text())
            except:
                itemVal = 0.
            if abs(itemVal) > 0.99: itemVal = 0.99 * np.sign(itemVal)
            if item.row() == item.column(): itemVal = 1.

            # update table
            self.tbl_corr.setItem(item.row(), item.column(),
                                                QTableWidgetItem(str(itemVal)))
            self.tbl_corr.setItem(item.column(), item.row(),
                                                QTableWidgetItem(str(itemVal)))

            # update simulation correlation matrix
            self.mi.corrmat[item.row(), item.column()] = itemVal
            self.mi.corrmat[item.column(), item.row()] = itemVal

        # update simulation correlation matrix
        self.mi.store_moments(self.mi.offsets, self.mi.sigmas, self.mi.corrmat)

        # resize stuff to fit
        self.tbl_corr.resizeColumnsToContents()
        self.tbl_corr.itemChanged.connect(self.symmetrize)

    def clear_correlations(self):
        """
        Clear off-diagonal elements in the correlation matrix
        """
        self.tbl_corr.itemChanged.disconnect(self.symmetrize)
        # unit matrix
        self.mi.corrmat = np.diag(np.ones(self.sim_ndim))

        # update table
        for i, row in enumerate(self.mi.corrmat):
            for j, val in enumerate(row):
                self.tbl_corr.setItem(i, j, QTableWidgetItem(str(val)))

        self.tbl_corr.itemChanged.connect(self.symmetrize)

    def update_sim_moments(self):
        """
        Resize tableWidget container to fit simulation correlation table
        """
        tableWidget = self.tbl_centroids
        tableWidget.itemChanged.disconnect(self.update_sim_moments)

        # selectedIndexes()
        for item in tableWidget.selectedIndexes():
            # print "selectedIndexes", item.row(), item.column()

            # get item value
            itemVal = tableWidget.item(item.row(), item.column()).text()
            try: itemVal = float(tableWidget.item(item.row(), item.column()).text())
            except:
                if item.column()==0: itemVal = 0. # default offset
                else: itemVal = 1. # default width
            if item.column()==1: # special cases for width values
                if itemVal == 0: itemVal = 1.
                if itemVal < 0: itemVal = abs(itemVal)

            # update table
            tableWidget.setItem(item.row(), item.column(), QTableWidgetItem(str(itemVal)))

            # update simulation moments
            if item.column()==0:
                self.mi.offsets[item.row()] = itemVal
            if item.column()==1:
                self.mi.sigmas[item.row()] = itemVal

        # update simulation correlation matrix
        self.mi.store_moments(self.mi.offsets, self.mi.sigmas, self.mi.corrmat)

        tableWidget.itemChanged.connect(self.update_sim_moments)