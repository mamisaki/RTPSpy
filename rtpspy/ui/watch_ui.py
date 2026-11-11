# -*- coding: utf-8 -*-
"""
GUI panel of RtpWatch

@author: mmisaki@laureateinstitute.org
"""

# %% import ===================================================================
from pathlib import Path

from PyQt6 import QtWidgets

from .module_ui import RtpModuleUI


# %% RtpWatchUI ===============================================================
class RtpWatchUI(RtpModuleUI):
    """GUI panel of RtpWatch."""

    _widget_setters = {
        "watch_dir": ("ui_wdir_lnEd", "setText", str),
        "file_type": ("ui_watchFType_cmbBx", "setCurrentText"),
        "watch_file_pattern": ("ui_watchPat_lnEd", "setText", str),
        "read_timeout": ("ui_dcmreadTimeout_dSpBx", "setValue"),
        "polling_observer": ("ui_pollingObserver_chb", "setChecked"),
        "polling_timeout": ("ui_pollingTimeout_dSpBx", "setValue"),
        "clean_ready": ("ui_cleanReady_chb", "setChecked"),
    }

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def make_ui(self):
        o = self.obj
        o.ui_objs = []
        ui_rows = [self.enabled_row()]

        # watch_dir
        var_lb = QtWidgets.QLabel("Watch directory :")
        o.ui_wdir_lnEd = QtWidgets.QLineEdit()
        o.ui_wdir_lnEd.setReadOnly(True)
        o.ui_wdir_lnEd.setStyleSheet("border: 0px none;")
        if o.watch_dir is not None and Path(o.watch_dir).is_dir():
            o.ui_wdir_lnEd.setText(str(o.watch_dir))
        ui_rows.append((var_lb, o.ui_wdir_lnEd))
        o.ui_objs.extend([var_lb, o.ui_wdir_lnEd])

        # file_type
        var_lb = QtWidgets.QLabel("File type :")
        o.ui_watchFType_cmbBx = QtWidgets.QComboBox()
        o.ui_watchFType_cmbBx.addItems(o._supported_file_types)
        o.ui_watchFType_cmbBx.setCurrentText(o.file_type)
        o.ui_watchFType_cmbBx.currentIndexChanged.connect(
            lambda idx: o.set_param(
                "file_type",
                o.ui_watchFType_cmbBx.currentText(),
                o.ui_watchFType_cmbBx.setCurrentIndex,
            )
        )
        ui_rows.append((var_lb, o.ui_watchFType_cmbBx))
        o.ui_objs.extend([var_lb, o.ui_watchFType_cmbBx])

        # watch_file_pattern
        var_lb = QtWidgets.QLabel("File name pattern :")
        o.ui_watchPat_lnEd = QtWidgets.QLineEdit()
        o.ui_watchPat_lnEd.setText(str(o.watch_file_pattern))
        o.ui_watchPat_lnEd.editingFinished.connect(
            lambda: o.set_param(
                "watch_file_pattern",
                o.ui_watchPat_lnEd.text(),
                o.ui_watchPat_lnEd.setText,
            )
        )
        ui_rows.append((var_lb, o.ui_watchPat_lnEd))
        o.ui_objs.extend([var_lb, o.ui_watchPat_lnEd])

        # read_timeout
        var_lb = QtWidgets.QLabel("Read timeout :")
        o.ui_dcmreadTimeout_dSpBx = QtWidgets.QDoubleSpinBox()
        o.ui_dcmreadTimeout_dSpBx.setMinimum(0.0)
        o.ui_dcmreadTimeout_dSpBx.setSingleStep(0.1)
        o.ui_dcmreadTimeout_dSpBx.setDecimals(3)
        o.ui_dcmreadTimeout_dSpBx.setSuffix(" seconds")
        o.ui_dcmreadTimeout_dSpBx.setValue(o.read_timeout)
        o.ui_dcmreadTimeout_dSpBx.valueChanged.connect(
            lambda x: o.set_param(
                "read_timeout", x, o.ui_dcmreadTimeout_dSpBx.setValue
            )
        )
        ui_rows.append((var_lb, o.ui_dcmreadTimeout_dSpBx))
        o.ui_objs.extend([var_lb, o.ui_dcmreadTimeout_dSpBx])

        # polling_observer check
        o.ui_pollingObserver_chb = QtWidgets.QCheckBox(
            "Poll the directory (for network drives)")
        o.ui_pollingObserver_chb.setToolTip(
            "Check the directory at intervals (watchdog PollingObserver)"
            " instead of file system events, which network drives may"
            " not send")
        o.ui_pollingObserver_chb.setChecked(o.polling_observer)
        o.ui_pollingObserver_chb.stateChanged.connect(
            lambda: o.set_param(
                "polling_observer",
                o.ui_pollingObserver_chb.isChecked(),
                o.ui_pollingObserver_chb.setChecked,
            )
        )
        ui_rows.append((None, o.ui_pollingObserver_chb))
        o.ui_objs.extend([o.ui_pollingObserver_chb])

        # polling_timeout
        var_lb = QtWidgets.QLabel("Polling interval :")
        o.ui_pollingTimeout_dSpBx = QtWidgets.QDoubleSpinBox()
        o.ui_pollingTimeout_dSpBx.setMinimum(0.0)
        o.ui_pollingTimeout_dSpBx.setSingleStep(0.1)
        o.ui_pollingTimeout_dSpBx.setDecimals(3)
        o.ui_pollingTimeout_dSpBx.setSuffix(" seconds")
        o.ui_pollingTimeout_dSpBx.setValue(o.polling_timeout)
        o.ui_pollingTimeout_dSpBx.valueChanged.connect(
            lambda x: o.set_param(
                "polling_timeout", x, o.ui_pollingTimeout_dSpBx.setValue
            )
        )
        ui_rows.append((var_lb, o.ui_pollingTimeout_dSpBx))
        o.ui_objs.extend([var_lb, o.ui_pollingTimeout_dSpBx])

        # clean_ready
        # The checkbox is made but not shown in the panel.
        o.ui_cleanReady_chb = QtWidgets.QCheckBox(
            "Clean watch dir at ready"
        )
        o.ui_cleanReady_chb.setChecked(o.clean_ready)
        o.ui_cleanReady_chb.stateChanged.connect(
            lambda state: setattr(o, "clean_ready", state > 0)
        )
        o.ui_objs.append(o.ui_cleanReady_chb)

        # --- Checkbox row ----------------------------------------------------
        ui_rows.append(self.save_proc_row())

        return ui_rows
