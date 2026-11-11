# -*- coding: utf-8 -*-
"""
GUI panel of RtpSmooth

@author: mmisaki@laureateinstitute.org
"""

# %% import ===================================================================
from PyQt6 import QtWidgets

from .module_ui import (RtpModuleUI, value_combo, MASK_ITEMS, mask_text,
                        ZERO_OUT_MASK_TEXT)


# %% RtpSmoothUI ==============================================================
class RtpSmoothUI(RtpModuleUI):
    """GUI panel of RtpSmooth."""

    _widget_setters = {
        "blur_fwhm": ("ui_FWHM_dSpBx", "setValue"),
        "mask_file": ("ui_mask_lnEd", "setText", mask_text),
        # Mask source (0: external file)
        "mask_source": ("ui_mask_cmbBx", "setCurrentIndex"),
    }

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def make_ui(self):
        o = self.obj
        o.ui_objs = []
        ui_rows = [self.enabled_row()]

        # blur_fwhm
        var_lb = QtWidgets.QLabel("Gaussian FWHM :")
        o.ui_FWHM_dSpBx = QtWidgets.QDoubleSpinBox()
        o.ui_FWHM_dSpBx.setMinimum(0.0)
        o.ui_FWHM_dSpBx.setSingleStep(1.0)
        o.ui_FWHM_dSpBx.setDecimals(2)
        o.ui_FWHM_dSpBx.setSuffix(" mm")
        o.ui_FWHM_dSpBx.setValue(o.blur_fwhm)
        o.ui_FWHM_dSpBx.valueChanged.connect(
                lambda x: o.set_param('blur_fwhm', x,
                                      o.ui_FWHM_dSpBx.setValue))
        ui_rows.append((var_lb, o.ui_FWHM_dSpBx))
        o.ui_objs.extend([var_lb, o.ui_FWHM_dSpBx])

        # mask_file
        var_lb = QtWidgets.QLabel("Mask :")
        o.ui_mask_cmbBx = value_combo(MASK_ITEMS)
        o.ui_mask_cmbBx.activated.connect(
                lambda idx:
                o.set_param('mask_file',
                            o.ui_mask_cmbBx.currentData(),
                            o.ui_mask_cmbBx.setCurrentIndex))
        ui_rows.append((var_lb, o.ui_mask_cmbBx))

        o.ui_mask_lnEd = QtWidgets.QLineEdit()
        o.ui_mask_lnEd.setReadOnly(True)
        o.ui_mask_lnEd.setStyleSheet(
            'border: 0px none;')
        ui_rows.append((None, o.ui_mask_lnEd))

        o.ui_objs.extend([var_lb, o.ui_mask_cmbBx, o.ui_mask_lnEd])

        if type(o.mask_file) is int and o.mask_file == 0:
            o.ui_mask_cmbBx.setCurrentIndex(1)
            o.ui_mask_lnEd.setText(ZERO_OUT_MASK_TEXT)
        else:
            o.ui_mask_cmbBx.setCurrentIndex(0)
            o.ui_mask_lnEd.setText(str(o.mask_file))

        # --- Checkbox row ----------------------------------------------------
        ui_rows.append(self.save_proc_row())

        return ui_rows
