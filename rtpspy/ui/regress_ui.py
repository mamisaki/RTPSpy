# -*- coding: utf-8 -*-
"""
GUI panel of RtpRegress

@author: mmisaki@laureateinstitute.org
"""

# %% import ===================================================================
from pathlib import Path

import numpy as np
from PyQt6 import QtWidgets

from ..rtp_regress import RtpRegress, MOT_REG_OPTS, PHYS_REG_OPTS
from .module_ui import (RtpModuleUI, value_combo, MASK_ITEMS, mask_text,
                        ZERO_OUT_MASK_TEXT)
from .widgets import MatplotlibWindow


# %% RtpRegressUI =============================================================
class RtpRegressUI(RtpModuleUI):
    """GUI panel of RtpRegress."""

    _widget_setters = {
        "mask_file": ("ui_mask_lnEd", "setText", mask_text),
        # Mask source (0: external file)
        "mask_source": ("ui_mask_cmbBx", "setCurrentIndex"),
        "wait_num": ("ui_waitNum_lb", "setText",
                     lambda n: f"{n} volumes"),
        # Wait mode (0: number of regressors, 1: set value)
        "wait_num_mode": ("ui_waitNum_cmbBx", "setCurrentIndex"),
        "max_scan_length": ("ui_maxLen_spBx", "setValue"),
        "max_poly_order": ("ui_maxPoly_lb", "setText",
                           RtpRegress._poly_order_text),
        # Polynomial order mode (0: auto, 1: set)
        "max_poly_mode": ("ui_maxPoly_cmbBx", "setCurrentIndex"),
        "mot_reg": ("ui_motReg_cmbBx", "setCurrentIndex",
                    MOT_REG_OPTS.index),
        "phys_reg": ("ui_physReg_cmbBx", "setCurrentIndex",
                     PHYS_REG_OPTS.index),
        "censor_FD": ("ui_censorFD_spBx", "setValue"),
        "ridge_c": ("ui_ridgeC_spBx", "setValue"),
        "reg_retro_proc": ("ui_retroProc_chb", "setChecked"),
    }

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def update_widget(self, attr, val, **kwargs):
        o = self.obj
        if attr in ('GS_reg', 'WM_reg', 'Vent_reg'):
            widget = self._widget(f"ui_{attr}_chb")
            if widget is not None:
                widget.setChecked(val)

        elif attr in ('GS_mask', 'WM_mask', 'Vent_mask'):
            widget = self._widget(f"ui_{attr}_lnEd")
            if widget is not None:
                widget.setText(str(val))

        elif attr == 'desMtx_loaded':
            # The design matrix is loaded (val=True) or unset
            if hasattr(o, 'ui_showDesMtx_btn'):
                o.ui_showDesMtx_btn.setEnabled(val)
            if hasattr(o, 'ui_loadDesMtx_btn'):
                o.ui_loadDesMtx_btn.setText('Unset' if val else 'Set')

        else:
            super().update_widget(attr, val, **kwargs)
            if attr == 'enabled' and o.desMtx_read is None and \
                    hasattr(o, 'ui_showDesMtx_btn'):
                o.ui_showDesMtx_btn.setEnabled(False)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def widget_value(self, key):
        o = self.obj
        if key == 'wait_num_mode':
            # Text of the wait mode selection
            if hasattr(o, 'ui_waitNum_cmbBx'):
                return o.ui_waitNum_cmbBx.currentData()

        elif key == 'desMtx_btn_unset':
            # The design matrix button is 'Unset' (a matrix is loaded)
            return hasattr(o, 'ui_loadDesMtx_btn') and \
                'Unset' in o.ui_loadDesMtx_btn.text()

        return None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def select_mask(self, mask_param, start_dir):
        """Ask the user to select the mask file of mask_param ('GS_mask',
        'WM_mask', or 'Vent_mask') in a file dialog opened at start_dir.
        Returns -1 if canceled (see RtpRegress.set_param)."""
        widget = self._widget(f"ui_{mask_param}_lnEd")
        if widget is None:
            return None
        return self.obj.set_param(mask_param, start_dir, widget.setText)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def show_design_matrix(self):
        """Show the design matrix in a plot window."""
        o = self.obj
        if o.desMtx_read is None:
            return

        self.plt_win = MatplotlibWindow()
        self.plt_win.setWindowTitle('Design matrix')
        ax = self.plt_win.canvas.figure.subplots(1, 1)
        ax.matshow(o.desMtx_read, cmap='gray')
        ax.set_aspect('auto')

        self.plt_win.show()
        self.plt_win.canvas.draw()
        self.plt_win.canvas.start_event_loop(0.005)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def make_ui(self):
        """
        Make the widgets. They call set_param with their setter as
        reset_fn.
        """
        o = self.obj
        o.ui_objs = []
        ui_rows = [self.enabled_row()]

        # mask_file
        var_lb = QtWidgets.QLabel("Mask :")
        o.ui_mask_cmbBx = value_combo(MASK_ITEMS)
        o.ui_mask_cmbBx.activated.connect(
            lambda idx:
            o.set_param('mask_file',
                        o.ui_mask_cmbBx.currentData(),
                        o.ui_mask_cmbBx.setCurrentIndex))

        o.ui_mask_lnEd = QtWidgets.QLineEdit()
        o.ui_mask_lnEd.setReadOnly(True)
        o.ui_mask_lnEd.setStyleSheet('border: 0px none;')
        o.ui_objs.extend([var_lb, o.ui_mask_cmbBx,
                          o.ui_mask_lnEd])

        if o.mask_file == 0:
            o.ui_mask_cmbBx.setCurrentIndex(1)
            o.ui_mask_lnEd.setText(ZERO_OUT_MASK_TEXT)
        else:
            o.ui_mask_cmbBx.setCurrentIndex(0)
            o.ui_mask_lnEd.setText(str(o.mask_file))

        mask_hLayout = QtWidgets.QHBoxLayout()
        mask_hLayout.addWidget(o.ui_mask_cmbBx)
        mask_hLayout.addWidget(o.ui_mask_lnEd)
        ui_rows.append((var_lb, mask_hLayout))

        # wait_num
        var_lb = QtWidgets.QLabel("Start after :")
        var_lb.setToolTip("REGRESS starts after receiving this number of"
                          " volumes")
        o.ui_waitNum_cmbBx = value_combo(
            [('Regressors + 1', 'number of regressors'),
             ('Set...', 'set value')])
        o.ui_waitNum_cmbBx.activated.connect(
            lambda idx:
            o.set_param('wait_num',
                        o.ui_waitNum_cmbBx.currentData(),
                        o.ui_waitNum_cmbBx.setCurrentIndex))

        o.ui_waitNum_lb = QtWidgets.QLabel()
        o.set_wait_num()
        o.ui_waitNum_lb.setText(f'{o.wait_num} volumes')
        o.ui_objs.extend([var_lb, o.ui_waitNum_cmbBx,
                          o.ui_waitNum_lb])

        wait_num_hLayout = QtWidgets.QHBoxLayout()
        wait_num_hLayout.addWidget(o.ui_waitNum_cmbBx)
        wait_num_hLayout.addWidget(o.ui_waitNum_lb)
        ui_rows.append((var_lb, wait_num_hLayout))

        # max_scan_length
        var_lb = QtWidgets.QLabel("Max scan length :")
        o.ui_maxLen_spBx = QtWidgets.QSpinBox()
        o.ui_maxLen_spBx.setSuffix(" volumes")
        o.ui_maxLen_spBx.setMinimum(1)
        o.ui_maxLen_spBx.setMaximum(9999)
        o.ui_maxLen_spBx.setValue(o.max_scan_length)
        o.ui_maxLen_spBx.editingFinished.connect(
            lambda: o.set_param('max_scan_length',
                                o.ui_maxLen_spBx.value(),
                                o.ui_maxLen_spBx.setValue))
        ui_rows.append((var_lb, o.ui_maxLen_spBx))
        o.ui_objs.extend([var_lb, o.ui_maxLen_spBx])

        # max_poly_order
        var_lb = QtWidgets.QLabel("Max polynomial order :")
        var_lb.setToolTip("Polynomial regressors for slow drifts")
        o.ui_maxPoly_cmbBx = value_combo([('Auto', 'auto'), ('Set...', 'set')])
        o.ui_maxPoly_cmbBx.activated.connect(
            lambda idx:
            o.set_param('max_poly_order',
                        o.ui_maxPoly_cmbBx.currentData(),
                        o.ui_maxPoly_cmbBx.setCurrentIndex))

        o.ui_maxPoly_lb = QtWidgets.QLabel()
        o.ui_objs.extend([var_lb, o.ui_maxPoly_cmbBx,
                          o.ui_maxPoly_lb])
        if np.isinf(o.max_poly_order):
            o.ui_maxPoly_cmbBx.setCurrentIndex(0)
        else:
            o.ui_maxPoly_cmbBx.setCurrentIndex(1)
        o.ui_maxPoly_lb.setText(
            o._poly_order_text(o.max_poly_order))

        maxPoly_hLayout = QtWidgets.QHBoxLayout()
        maxPoly_hLayout.addWidget(o.ui_maxPoly_cmbBx)
        maxPoly_hLayout.addWidget(o.ui_maxPoly_lb)
        ui_rows.append((var_lb, maxPoly_hLayout))

        # mot_reg (the item order follows MOT_REG_OPTS)
        var_lb = QtWidgets.QLabel("Motion :")
        o.ui_motReg_cmbBx = value_combo(
            [('None', 'None'),
             ('6 parameters', '6 motions (yaw, pitch, roll, dS, dL, dP)'),
             ('6 parameters + derivatives',
              '12 motions (6 motions and their temporal derivatives)'),
             ('6 derivatives', '6 motion derivatives')])
        o.ui_motReg_cmbBx.setToolTip(
            "Motion parameters: yaw, pitch, roll, dS, dL, dP")
        o.ui_motReg_cmbBx.setCurrentIndex(
            MOT_REG_OPTS.index(o.mot_reg))
        o.ui_motReg_cmbBx.currentIndexChanged.connect(
            lambda idx:
            o.set_param('mot_reg',
                        o.ui_motReg_cmbBx.currentData(),
                        o.ui_motReg_cmbBx.setCurrentIndex))
        ui_rows.append((var_lb, o.ui_motReg_cmbBx))
        o.ui_objs.extend([var_lb, o.ui_motReg_cmbBx])

        # ROI mean signal regressors
        ui_rows.append(self._roi_reg_row('GS', "Global signal :"))
        ui_rows.append(self._roi_reg_row('WM', "WM signal :"))
        ui_rows.append(self._roi_reg_row('Vent', "Ventricle signal :"))

        # phys_reg (the item order follows PHYS_REG_OPTS; RVT is not
        # supported)
        var_lb = QtWidgets.QLabel("Physio (RICOR) :")
        o.ui_physReg_cmbBx = value_combo(
            [('None', 'None'),
             ('4 resp + 4 card', '8 RICOR (4 Resp and 4 Card)')])
        o.ui_physReg_cmbBx.setCurrentIndex(
            PHYS_REG_OPTS.index(o.phys_reg))
        o.ui_physReg_cmbBx.currentIndexChanged.connect(
            lambda idx:
            o.set_param('phys_reg',
                        o.ui_physReg_cmbBx.currentData(),
                        o.ui_physReg_cmbBx.setCurrentIndex))
        ui_rows.append((var_lb, o.ui_physReg_cmbBx))
        o.ui_objs.extend([var_lb, o.ui_physReg_cmbBx])

        # desMtx
        var_lb = QtWidgets.QLabel("Design matrix :")

        desMtx_hBLayout = QtWidgets.QHBoxLayout()
        o.ui_loadDesMtx_btn = QtWidgets.QPushButton(
            'Set' if o.desMtx_read is None else 'Unset')
        o.ui_loadDesMtx_btn.clicked.connect(
            lambda: o.set_param('desMtx_f', 'set'))
        desMtx_hBLayout.addWidget(o.ui_loadDesMtx_btn)

        o.ui_showDesMtx_btn = QtWidgets.QPushButton()
        o.ui_showDesMtx_btn.clicked.connect(
            lambda: o.set_param('showDesMtx'))
        desMtx_hBLayout.addWidget(o.ui_showDesMtx_btn)

        o.ui_objs.extend([var_lb, o.ui_loadDesMtx_btn,
                          o.ui_showDesMtx_btn])
        ui_rows.append((var_lb, desMtx_hBLayout))
        o.ui_showDesMtx_btn.setText('Show')
        o.ui_showDesMtx_btn.setEnabled(o.desMtx_read is not None)

        # censor_FD
        var_lb = QtWidgets.QLabel("Censor FD above :")
        o.ui_censorFD_spBx = QtWidgets.QDoubleSpinBox()
        o.ui_censorFD_spBx.setMinimum(0.0)
        o.ui_censorFD_spBx.setMaximum(10.0)
        o.ui_censorFD_spBx.setDecimals(2)
        o.ui_censorFD_spBx.setSingleStep(0.1)
        o.ui_censorFD_spBx.setSuffix(" mm")
        o.ui_censorFD_spBx.setSpecialValueText("Off")
        o.ui_censorFD_spBx.setValue(o.censor_FD)
        o.ui_censorFD_spBx.setToolTip(
            "Exclude volumes with framewise displacement (Power et al., 2012)"
            " larger than this value from the GLM fits.")
        o.ui_censorFD_spBx.editingFinished.connect(
            lambda: o.set_param('censor_FD',
                                o.ui_censorFD_spBx.value(),
                                o.ui_censorFD_spBx.setValue))
        ui_rows.append((var_lb, o.ui_censorFD_spBx))
        o.ui_objs.extend([var_lb, o.ui_censorFD_spBx])

        # ridge_c
        var_lb = QtWidgets.QLabel("Early-fit ridge :")
        o.ui_ridgeC_spBx = QtWidgets.QDoubleSpinBox()
        o.ui_ridgeC_spBx.setMinimum(0.0)
        o.ui_ridgeC_spBx.setMaximum(10.0)
        o.ui_ridgeC_spBx.setDecimals(2)
        o.ui_ridgeC_spBx.setSingleStep(0.1)
        o.ui_ridgeC_spBx.setSpecialValueText("Off")
        o.ui_ridgeC_spBx.setValue(o.ridge_c)
        o.ui_ridgeC_spBx.setToolTip(
            "Ridge penalty of the nuisance regressors, ridge_c * (p/n)^2,"
            " which reduces overfitting while the number of volumes n is"
            " close to the number of regressors p. 0: least squares.")
        o.ui_ridgeC_spBx.editingFinished.connect(
            lambda: o.set_param('ridge_c',
                                o.ui_ridgeC_spBx.value(),
                                o.ui_ridgeC_spBx.setValue))
        ui_rows.append((var_lb, o.ui_ridgeC_spBx))
        o.ui_objs.extend([var_lb, o.ui_ridgeC_spBx])

        # --- Checkbox row ----------------------------------------------------
        # Retrospective process
        o.ui_retroProc_chb = QtWidgets.QCheckBox(
            "Process the volumes before the start")
        o.ui_retroProc_chb.setToolTip(
            "Process the volumes received before REGRESS started"
            " retrospectively at the first fit")
        o.ui_retroProc_chb.setChecked(o.reg_retro_proc)
        o.ui_retroProc_chb.stateChanged.connect(
            lambda state: setattr(o, 'reg_retro_proc', state > 0))
        o.ui_objs.append(o.ui_retroProc_chb)

        # In the field column (a long check box in the label column widens
        # all the labels)
        ui_rows.append((None, o.ui_retroProc_chb))

        # Save
        ui_rows.append(self.save_proc_row())

        return ui_rows

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _roi_reg_row(self, reg, label):
        """
        Make the UI row for the ROI mean signal regressor reg ('GS', 'WM', or
        'Vent'): a check box for <reg>_reg, and the <reg>_mask file with a
        button to select it. The widgets are set to obj.ui_<reg>_reg_chb,
        obj.ui_<reg>_mask_lnEd, and obj.ui_<reg>mask_btn.
        """
        o = self.obj
        reg_chb = QtWidgets.QCheckBox(label)
        reg_chb.setChecked(getattr(o, f"{reg}_reg"))
        reg_chb.stateChanged.connect(
            lambda state: o.set_param(f"{reg}_reg", state > 0))
        setattr(o, f"ui_{reg}_reg_chb", reg_chb)

        mask_hBLayout = QtWidgets.QHBoxLayout()
        mask_lnEd = QtWidgets.QLineEdit()
        mask_lnEd.setText(str(getattr(o, f"{reg}_mask")))
        mask_lnEd.setReadOnly(True)
        mask_lnEd.setStyleSheet('border: 0px none;')
        mask_hBLayout.addWidget(mask_lnEd)
        setattr(o, f"ui_{reg}_mask_lnEd", mask_lnEd)

        mask_btn = QtWidgets.QPushButton('Set')
        mask_btn.clicked.connect(
            lambda: o.set_param(f"{reg}_mask",
                                Path(mask_lnEd.text()).parent,
                                mask_lnEd.setText))
        mask_hBLayout.addWidget(mask_btn)
        setattr(o, f"ui_{reg}mask_btn", mask_btn)

        o.ui_objs.extend([reg_chb, mask_lnEd, mask_btn])
        return (reg_chb, mask_hBLayout)
