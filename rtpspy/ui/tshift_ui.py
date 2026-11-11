# -*- coding: utf-8 -*-
"""
GUI panel of RtpTshift

@author: mmisaki@laureateinstitute.org
"""

# %% import ===================================================================
from PyQt6 import QtWidgets

from .module_ui import RtpModuleUI


# %% RtpTshiftUI ==============================================================
class RtpTshiftUI(RtpModuleUI):
    """GUI panel of RtpTshift."""

    _widget_setters = {
        "method": ("ui_method_cmbBx", "setCurrentText"),
        "ref_time": ("ui_refTime_dSpBx", "setValue"),
        "TR": ("ui_TR_dSpBx", "setValue"),
        "slice_timing": ("ui_SlTiming_lnEd", "setText", str),
        "slice_dim": ("ui_sliceDim_cmbBx", "setCurrentIndex"),
    }

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def make_ui(self):
        o = self.obj
        o.ui_objs = []
        ui_rows = [self.enabled_row()]

        # method
        var_lb = QtWidgets.QLabel("Interpolation :")
        o.ui_method_cmbBx = QtWidgets.QComboBox()
        o.ui_method_cmbBx.addItems(['linear', 'cubic'])
        o.ui_method_cmbBx.setCurrentText(o.method)
        o.ui_method_cmbBx.currentIndexChanged.connect(
                lambda idx:
                o.set_param('method',
                            o.ui_method_cmbBx.currentText(),
                            o.ui_method_cmbBx.setCurrentText))
        ui_rows.append((var_lb, o.ui_method_cmbBx))
        o.ui_objs.extend([var_lb, o.ui_method_cmbBx])

        # ref_time
        var_lb = QtWidgets.QLabel("Reference time :")
        o.ui_refTime_dSpBx = QtWidgets.QDoubleSpinBox()
        o.ui_refTime_dSpBx.setMinimum(0.000)
        o.ui_refTime_dSpBx.setSingleStep(0.001)
        o.ui_refTime_dSpBx.setDecimals(3)
        o.ui_refTime_dSpBx.setSuffix(" seconds")
        o.ui_refTime_dSpBx.setValue(o.ref_time)
        o.ui_refTime_dSpBx.valueChanged.connect(
                lambda x: o.set_param('ref_time', x,
                                      o.ui_refTime_dSpBx.setValue))
        ui_rows.append((var_lb, o.ui_refTime_dSpBx))
        o.ui_objs.extend([var_lb, o.ui_refTime_dSpBx])

        # Load from sample button
        o.ui_setfrmSample_btn = QtWidgets.QPushButton(
                'Read TR and slice timing from an image...')
        o.ui_setfrmSample_btn.clicked.connect(
                lambda: o.set_param('set_from_sample'))
        ui_rows.append((o.ui_setfrmSample_btn,))
        o.ui_objs.append(o.ui_setfrmSample_btn)

        # TR
        var_lb = QtWidgets.QLabel("TR :")
        o.ui_TR_dSpBx = QtWidgets.QDoubleSpinBox()
        o.ui_TR_dSpBx.setMinimum(0.000)
        o.ui_TR_dSpBx.setSingleStep(0.001)
        o.ui_TR_dSpBx.setDecimals(3)
        o.ui_TR_dSpBx.setSuffix(" seconds")
        o.ui_TR_dSpBx.setValue(o.TR)
        o.ui_TR_dSpBx.valueChanged.connect(
                lambda x: o.set_param('TR', x, o.ui_TR_dSpBx.setValue))
        ui_rows.append((var_lb, o.ui_TR_dSpBx))
        o.ui_objs.extend([var_lb, o.ui_TR_dSpBx])

        # slice_timing
        var_lb = QtWidgets.QLabel("Slice timing :")
        o.ui_SlTiming_lnEd = QtWidgets.QLineEdit()
        o.ui_SlTiming_lnEd.setToolTip(
            "Acquisition time (s) of each slice: [1st, 2nd, ...]")
        o.ui_SlTiming_lnEd.setText("{}".format(o.slice_timing))
        o.ui_SlTiming_lnEd.editingFinished.connect(
                lambda: o.set_param('slice_timing',
                                    o.ui_SlTiming_lnEd.text(),
                                    o.ui_SlTiming_lnEd.setText))
        ui_rows.append((var_lb, o.ui_SlTiming_lnEd))
        o.ui_objs.extend([var_lb, o.ui_SlTiming_lnEd])

        # slice_dim
        var_lb = QtWidgets.QLabel("Slice orientation :")
        o.ui_sliceDim_cmbBx = QtWidgets.QComboBox()
        o.ui_sliceDim_cmbBx.addItems(['x (Sagittal)', 'y (Coronal)',
                                      'z (Axial)'])
        o.ui_sliceDim_cmbBx.setCurrentIndex(o.slice_dim)
        o.ui_sliceDim_cmbBx.currentIndexChanged.connect(
                lambda idx:
                o.set_param('slice_dim',
                            o.ui_sliceDim_cmbBx.currentIndex(),
                            o.ui_sliceDim_cmbBx.setCurrentIndex))
        ui_rows.append((var_lb, o.ui_sliceDim_cmbBx))
        o.ui_objs.extend([var_lb, o.ui_sliceDim_cmbBx])

        # --- Checkbox row ----------------------------------------------------
        ui_rows.append(self.save_proc_row())

        return ui_rows
