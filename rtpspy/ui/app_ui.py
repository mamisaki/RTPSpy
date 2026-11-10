# -*- coding: utf-8 -*-
"""
GUI panel of RtpApp: the application settings, the session control buttons,
the session monitor, and the ROI signal plot

@author: mmisaki@laureateinstitute.org
"""

# %% import ===================================================================
from pathlib import Path
import os
import sys
import time
import shlex
import subprocess
from functools import partial

import numpy as np
from PyQt5 import QtWidgets, QtCore
import matplotlib.pyplot as plt

from ..rtp_app import _FILE_PARAM_LABELS, _send_scan_start
from .module_ui import RtpModuleUI
from .widgets import MatplotlibWindow

# Set button color in the UI
_SET_BTN_STYLE = "background-color: rgb(151,217,235);"


# %% RtpAppUI =================================================================
class RtpAppUI(RtpModuleUI):
    """GUI panel of RtpApp (and of an application class derived from it)."""

    _widget_setters = {
        "extApp_cmd": ("ui_extApp_cmd_lnEd", "setText"),
        "ROI_resample": ("ui_ROI_resample_cmbBx", "setCurrentText"),
        "no_FastSeg": ("ui_no_FastSeg_chb", "setChecked"),
    }

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, obj):
        super().__init__(obj)

        # Timer to check the running status (see check_run_status)
        self.chk_run_timer = QtCore.QTimer()
        self.chk_run_timer.setSingleShot(True)
        self.chk_run_timer.timeout.connect(self.check_run_status)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def update_widget(self, attr, val, **kwargs):
        o = self.obj
        if attr == "enable_RTP":
            if hasattr(o, "ui_enableRTP_chb"):
                if val == 0:
                    o.ui_enableRTP_chb.setCheckState(0)
                else:
                    o.ui_enableRTP_chb.setCheckState(2)

        elif attr == "extApp_addr":
            # Show the address without calling set_param again
            if hasattr(o, "ui_extApp_addr_lnEd"):
                o.ui_extApp_addr_lnEd.blockSignals(True)
                o.ui_extApp_addr_lnEd.setText("{}:{}".format(*val))
                o.ui_extApp_addr_lnEd.blockSignals(False)

        elif attr in _FILE_PARAM_LABELS:
            widget = self._widget(f"ui_{attr}_lnEd")
            if widget is not None:
                widget.setText(str(val))

        else:
            super().update_widget(attr, val, **kwargs)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_create_masks_enabled(self, enabled):
        """Enable or disable the 'Create masks' button."""
        if hasattr(self.obj, "ui_CreateMasks_btn"):
            self.obj.ui_CreateMasks_btn.setEnabled(enabled)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def create_masks(self):
        """'Create masks' button: run make_masks with the progress dialog.
        Shift-click: overwrite. Ctrl-click: ask_cmd."""
        o = self.obj
        shift = QtCore.Qt.ShiftModifier
        ctrl = QtCore.Qt.ControlModifier
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        overwrite = modifiers in (shift, ctrl | shift)
        ask_cmd = modifiers in (ctrl, ctrl | shift)
        o.make_masks(progress_dlg=True, no_FastSeg=o.no_FastSeg,
                     overwrite=overwrite, ask_cmd=ask_cmd)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run_dcm2nii(self):
        """Convert DICOM to NIfTI using dcm2niix."""
        o = self.obj
        # Select DICOM directory
        watch_dir = o.rtp_objs["WATCH"].watch_dir
        dcm_dir = QtWidgets.QFileDialog.getExistingDirectory(
            o.main_win, "Select DICOM file directory", str(watch_dir)
        )

        if dcm_dir == "":
            return

        out_dir = Path(o.work_dir).absolute()
        cmd = "dcm2niix -f sub-%n_ses-%t_ser-%s_desc-%d"
        cmd += f" -i n -z o -w 0 -o {out_dir} {dcm_dir}"
        try:
            # Progress dialog without buttons
            msgBox = QtWidgets.QMessageBox(o.main_win)
            msgBox.setWindowTitle("dcm2niix")
            msgBox.setText("Converting DICOM to NIfTI ...")
            msgBox.setStandardButtons(QtWidgets.QMessageBox.NoButton)
            msgBox.show()
            time.sleep(0.1)
            msgBox.repaint()
            QtWidgets.QApplication.processEvents()

            proc = subprocess.Popen(
                shlex.split(cmd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            while proc.poll() is None:
                QtWidgets.QApplication.processEvents()
                time.sleep(0.5)

            stdout, stderr = proc.communicate()
            ostr = "\n".join([stdout.decode(), stderr.decode()])

            # Copy DICOMs to the working directory
            dst_dir = out_dir / "dicom"
            cmd = f"rsync -auvz {dcm_dir}/ {dst_dir}/"
            subprocess.check_call(shlex.split(cmd), stdout=subprocess.DEVNULL)

            msgBox.accept()
            QtWidgets.QMessageBox.information(
                o.main_win, "dcm2niix", ostr, QtWidgets.QMessageBox.Ok
            )

        except Exception as e:
            try:
                msgBox.accept()
            except Exception:
                pass
            errmsg = f"Error during dcm2niix: {e}"
            o._logger.error(errmsg)
            o.err_popup(errmsg)

    # --- Session control -----------------------------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def on_RTP_setup_done(self):
        """Enable the Ready button and show the parameter list."""
        o = self.obj
        o.ui_ready_btn.setEnabled(True)
        o.ui_ready_btn.setText("Ready")
        o.ui_quit_btn.setEnabled(True)

        # Show the parameter list tab
        o.main_win.show_options_list()
        o.main_win.options_tab.setCurrentIndex(2)
        QtWidgets.QApplication.processEvents()
        time.sleep(0.1)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_enabled_all(self, enabled):
        """Enable or disable the widgets of all the modules and
        applications in the GUI."""
        o = self.obj
        objs = list(o.main_win.rtp_objs.values())
        objs += list(o.main_win.rtp_apps.values())
        for pobj in objs:
            if not hasattr(pobj, "ui_objs"):
                continue

            for ui in pobj.ui_objs:
                ui.setEnabled(enabled)

            if hasattr(pobj, "ui_enabled_rdb"):
                pobj.ui_enabled_rdb.setEnabled(enabled)

        if hasattr(o.main_win, "chbRecSignal"):
            o.main_win.chbRecSignal.setEnabled(enabled)
        if hasattr(o.main_win, "chbUseGPU"):
            o.main_win.chbUseGPU.setEnabled(enabled)
        o.main_win.btnSetWorkDir.setEnabled(enabled)
        o.main_win.btnSetWatchDir.setEnabled(enabled)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def show_run_plots(self):
        """Show the motion and physio plots of the enabled modules."""
        o = self.obj
        if o.rtp_objs["VOLREG"].enabled:
            o.main_win.chbShowMotion.setCheckState(2)

        if (
            o.rtp_objs["REGRESS"].enabled
            and o.rtp_objs["REGRESS"].phys_reg != "None"
        ):
            o.main_win.chbShowPhysio.setCheckState(2)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def start_run_check(self):
        """Start the running-status check every second."""
        self.chk_run_timer.start(1000)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def stop_run_check(self):
        self.chk_run_timer.stop()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def start_wait_onset(self):
        """Wait for the scan onset in a thread, and set the buttons for the
        waiting state."""
        o = self.obj
        self.th_wait_onset = QtCore.QThread()
        self.wait_onset = WAIT_ONSET(
            o, o.rtp_objs["TTLPHYSIO"], o.extApp_sock
        )
        self.wait_onset.moveToThread(self.th_wait_onset)
        self.th_wait_onset.started.connect(self.wait_onset.run)
        self.wait_onset.finished.connect(self.th_wait_onset.quit)
        self.th_wait_onset.start()

        o.ui_ready_btn.setText("Waiting for scan start ...")
        o.ui_ready_btn.setEnabled(False)
        o.ui_manStart_btn.setEnabled(True)
        o.ui_quit_btn.setEnabled(True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def abort_wait_onset(self):
        """Abort the WAIT_ONSET thread if it is running."""
        if (
            hasattr(self, "th_wait_onset")
            and self.th_wait_onset.isRunning()
        ):
            self.wait_onset.abort = True
            if not self.th_wait_onset.wait(1):
                self.wait_onset.finished.emit()
                self.th_wait_onset.wait()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def on_end_run_start(self):
        """Disable the session buttons at the start of end_run."""
        o = self.obj
        o.ui_ready_btn.setText("Ready")
        o.ui_ready_btn.setEnabled(False)
        o.ui_quit_btn.setEnabled(False)
        o.ui_manStart_btn.setEnabled(False)
        o.main_win.repaint()
        QtWidgets.QApplication.instance().processEvents()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def on_end_run_done(self):
        """Enable the widgets at the end of end_run."""
        self.obj.ui_setEnabled(True)
        self.obj.main_win.options_tab.setCurrentIndex(0)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def check_run_status(self):
        """Check the running status every second and end the session if
        requested by the external application or no new image arrives."""
        o = self.obj
        if o.extApp_sock is not None:
            # Check a message from the external application
            msg = o.recv_extApp(timeout=0.001)
            if msg is not None and "END_SESSION" in msg.decode():
                if o.ui_quit_btn.isEnabled():
                    o._logger.info("Recv END_SESSION. End session.")
                    if not o._isRunning_end_run_proc:
                        o.end_run()
                else:
                    o.ui_ready_btn.setEnabled(False)
                return

        # Check the time since the last image in WATCH
        if (
            o.enable_RTP
            and not np.isnan(o.max_watch_wait)
            and len(o.rtp_objs["WATCH"]._proc_time)
        ):
            delay = time.time() - o.rtp_objs["WATCH"]._proc_time[-1]
            if delay > o.max_watch_wait:
                o._logger.info(
                    f"No new file was seen for {delay:.3f} s. End session."
                )
                if not o._isRunning_end_run_proc:
                    o.end_run()
                return

        # Update the button text when the scan has started
        if (
            hasattr(o, "ui_ready_btn")
            and "waiting" in o.ui_ready_btn.text().lower()
        ):
            if o._scanning:
                o.ui_ready_btn.setText("Session is running")
                o.ui_manStart_btn.setEnabled(False)

        # Schedule the next check
        self.chk_run_timer.start(1000)

    # --- ROI signal plot -----------------------------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def open_ROISig_plot(self, num_ROIs=1, roi_labels=[]):
        """Open the ROI signal plot."""
        if hasattr(self, "pltROISig") and self.pltROISig.isRunning():
            return

        self.pltROISig = PlotROISignal(
            self.obj,
            num_ROIs=num_ROIs,
            roi_labels=roi_labels,
            main_win=self.obj.main_win,
        )
        self.pltROISig.run()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def close_ROISig_plot(self):
        """Close the ROI signal plot."""
        if hasattr(self, "pltROISig"):
            pltROISig = self.pltROISig
            del self.pltROISig
            pltROISig.end_thread()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def reset_ROISig_plot(self):
        """Clear the ROI signal plot."""
        if hasattr(self, "pltROISig"):
            self.pltROISig.reset_plot()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def show_ROIsig_chk(self, state):
        """Checkbox to show or hide the ROI signal plot."""
        if state == 2:
            self.open_ROISig_plot(
                num_ROIs=self.obj.num_ROIs, roi_labels=self.obj.roi_labels
            )
        else:
            self.close_ROISig_plot()

    # --- Panel ---------------------------------------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _add_file_row(
        self, layout, row, label, attr, set_dir, show_value=True,
        highlight=False,
    ):
        """Add a row of [label, file path, Set, Unset] widgets for a file
        parameter to a grid layout.

        Parameters
        ----------
        layout : QGridLayout
            Layout to add the widgets.
        row : int
            Row index in the layout.
        label : str
            Label text.
        attr : str
            File parameter name.
        set_dir : function
            Returns the value given to set_param when Set is clicked (the
            start directory of the file dialog).
        show_value : bool, optional
            Show the current value in the path field.
        highlight : bool, optional
            Color the Set button.

        Returns
        -------
        (label, path field, Set button, Unset button) widgets
        """
        o = self.obj
        lb = QtWidgets.QLabel(label)
        layout.addWidget(lb, row, 0)

        lnEd = QtWidgets.QLineEdit()
        if show_value:
            lnEd.setText(str(getattr(o, attr)))
        lnEd.setReadOnly(True)
        lnEd.setStyleSheet("border: 0px none;")
        layout.addWidget(lnEd, row, 1)

        set_btn = QtWidgets.QPushButton("Set")
        set_btn.clicked.connect(
            lambda: o.set_param(attr, set_dir(), lnEd.setText)
        )
        if highlight:
            set_btn.setStyleSheet(_SET_BTN_STYLE)
        layout.addWidget(set_btn, row, 2)

        del_btn = QtWidgets.QPushButton("Unset")
        del_btn.clicked.connect(
            lambda: o.delete_file(attr, keepfile=True)
        )
        layout.addWidget(del_btn, row, 3)

        return lb, lnEd, set_btn, del_btn

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def make_ui(self):
        """Make the UI widgets.

        Returns
        -------
        ui_rows : list of tuples
            Widget rows to be placed by RtpGUI.
        """
        o = self.obj
        ui_rows = []
        o.ui_objs = []  # Widgets disabled while running

        # enabled
        o.ui_enableRTP_chb = QtWidgets.QCheckBox("Enable RTP")
        o.ui_enableRTP_chb.setChecked(o.enable_RTP)
        o.ui_enableRTP_chb.stateChanged.connect(
            lambda x: o.set_param(
                "enable_RTP", x, o.ui_enableRTP_chb.setCheckState
            )
        )
        ui_rows.append((o.ui_enableRTP_chb, None))
        o.ui_objs.append(o.ui_enableRTP_chb)

        # tab
        o.ui_top_tabs = QtWidgets.QTabWidget()
        ui_rows.append((o.ui_top_tabs,))

        # Values given to set_param by the Set buttons of the file rows
        # (the start directory of the file dialog)
        def default_dir():
            return ""

        def work_dir():
            return o.work_dir

        # region: Preprocessing tab -------------------------------------------
        o.ui_preprocessingTab = QtWidgets.QWidget()
        o.ui_top_tabs.addTab(o.ui_preprocessingTab, "Preprocessing")
        o.ui_preprocessing_fLayout = QtWidgets.QFormLayout(
            o.ui_preprocessingTab
        )

        # region: dcm2nii --
        o.ui_dcm2nii_btn = QtWidgets.QPushButton("dcm2niix")
        o.ui_dcm2nii_btn.clicked.connect(lambda: self.run_dcm2nii())
        o.ui_preprocessing_fLayout.addRow(o.ui_dcm2nii_btn)
        o.ui_objs.append(o.ui_dcm2nii_btn)
        # endregion

        # region: Reference images group ---
        o.ui_RefImg_grpBx = QtWidgets.QGroupBox("Reference images")
        RefImg_gLayout = QtWidgets.QGridLayout(o.ui_RefImg_grpBx)
        o.ui_preprocessing_fLayout.addRow(o.ui_RefImg_grpBx)
        o.ui_objs.append(o.ui_RefImg_grpBx)

        # -- Anatomy orig image --
        (
            _, o.ui_anat_orig_lnEd, o.ui_anat_orig_btn,
            o.ui_anat_orig_del_btn
        ) = self._add_file_row(
            RefImg_gLayout, 0, "Anatomy image :", "anat_orig", default_dir,
            show_value=False, highlight=True,
        )
        if o.anat_orig and Path(o.anat_orig).exists():
            o.ui_anat_orig_lnEd.setText(str(o.anat_orig))

        # -- function orig image --
        (
            _, o.ui_func_orig_lnEd, o.ui_func_orig_btn,
            o.ui_func_orig_del_btn
        ) = self._add_file_row(
            RefImg_gLayout, 1, "Base function image : ", "func_orig",
            default_dir, show_value=False, highlight=True,
        )

        # -- parameter reference image --
        (
            _, o.ui_func_param_ref_lnEd, o.ui_param_ref_btn,
            o.ui_param_ref_del_btn
        ) = self._add_file_row(
            RefImg_gLayout, 2, "fMRI parameter reference : ",
            "func_param_ref", default_dir, show_value=False, highlight=True,
        )
        # endregion

        # region: Fieldmap PEPOLAR images group --
        o.ui_Fmap_grpBx = QtWidgets.QGroupBox("Fieldmap PEPOLAR images")
        Fmap_gLayout = QtWidgets.QGridLayout(o.ui_Fmap_grpBx)
        o.ui_preprocessing_fLayout.addRow(o.ui_Fmap_grpBx)
        o.ui_objs.append(o.ui_Fmap_grpBx)

        # -- Fieldmap with the same phase encoding --
        (
            _, o.ui_fieldmap_posi_lnEd, o.ui_fieldmap_posi_btn,
            o.ui_fieldmap_posi_del_btn
        ) = self._add_file_row(
            Fmap_gLayout, 0, "Same PE direction :", "fieldmap_posi",
            default_dir, show_value=False, highlight=True,
        )

        # -- Fieldmap with the opposite phase encoding --
        (
            _, o.ui_fieldmap_nega_lnEd, o.ui_fieldmap_nega_btn,
            o.ui_fieldmap_nega_del_btn
        ) = self._add_file_row(
            Fmap_gLayout, 1, "Opposite PE direction :", "fieldmap_nega",
            default_dir, show_value=False, highlight=True,
        )
        # endregion

        # region: CreateMasks button --
        ui_CreateMasks_hLayout = QtWidgets.QHBoxLayout()
        o.ui_preprocessing_fLayout.addRow(ui_CreateMasks_hLayout)

        o.ui_CreateMasks_btn = QtWidgets.QPushButton(
            "Create masks (+shift=overwrite)"
        )
        o.ui_CreateMasks_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        o.ui_CreateMasks_btn.clicked.connect(
            lambda x: self.create_masks()
        )
        o.ui_CreateMasks_btn.setStyleSheet(_SET_BTN_STYLE)
        ui_CreateMasks_hLayout.addWidget(o.ui_CreateMasks_btn)
        o.ui_objs.append(o.ui_CreateMasks_btn)

        # no_FastSeg checkbox
        o.ui_no_FastSeg_chb = QtWidgets.QCheckBox("No FastSeg")
        o.ui_no_FastSeg_chb.setChecked(o.no_FastSeg)
        o.ui_no_FastSeg_chb.stateChanged.connect(
            lambda x: o.set_param(
                "no_FastSeg", x, o.ui_no_FastSeg_chb.setCheckState
            )
        )
        ui_CreateMasks_hLayout.addWidget(o.ui_no_FastSeg_chb)
        o.ui_objs.append(o.ui_no_FastSeg_chb)
        # endregion

        # region: Check masks group ---
        o.ui_ChkMask_grpBx = QtWidgets.QGroupBox(
            "Display the masks in AFNI"
        )
        ChkMask_gLayout = QtWidgets.QGridLayout(o.ui_ChkMask_grpBx)
        o.ui_preprocessing_fLayout.addRow(o.ui_ChkMask_grpBx)
        o.ui_objs.append(o.ui_ChkMask_grpBx)

        o.ui_openAFNI_btn = QtWidgets.QPushButton("Open AFNI")
        o.ui_openAFNI_btn.clicked.connect(lambda: o.check_onAFNI())
        ChkMask_gLayout.addWidget(o.ui_openAFNI_btn, 0, 0)

        o.ui_chkFuncAnat_btn = QtWidgets.QPushButton("func-anat align")
        o.ui_chkFuncAnat_btn.clicked.connect(
            lambda: o.check_onAFNI("anat", "func")
        )
        ChkMask_gLayout.addWidget(o.ui_chkFuncAnat_btn, 0, 1)

        o.ui_chkROIFunc_btn = QtWidgets.QPushButton("ROI on function")
        o.ui_chkROIFunc_btn.clicked.connect(
            lambda: o.check_onAFNI("func", "roi")
        )
        ChkMask_gLayout.addWidget(o.ui_chkROIFunc_btn, 0, 2)

        o.ui_chkROIAnat_btn = QtWidgets.QPushButton("ROI on anatomy")
        o.ui_chkROIAnat_btn.clicked.connect(
            lambda: o.check_onAFNI("anat", "roi")
        )
        ChkMask_gLayout.addWidget(o.ui_chkROIAnat_btn, 0, 3)

        o.ui_chkWMFunc_btn = QtWidgets.QPushButton("WM on anatomy")
        o.ui_chkWMFunc_btn.clicked.connect(
            lambda: o.check_onAFNI("anat", "wm")
        )
        ChkMask_gLayout.addWidget(o.ui_chkWMFunc_btn, 1, 0)

        o.ui_chkVentFunc_btn = QtWidgets.QPushButton("Vent on anatomy")
        o.ui_chkVentFunc_btn.clicked.connect(
            lambda: o.check_onAFNI("anat", "vent")
        )
        ChkMask_gLayout.addWidget(o.ui_chkVentFunc_btn, 1, 1)

        o.ui_chkGSRmask_btn = QtWidgets.QPushButton("GSR mask")
        o.ui_chkGSRmask_btn.clicked.connect(
            lambda: o.check_onAFNI("func", "GSRmask")
        )
        ChkMask_gLayout.addWidget(o.ui_chkGSRmask_btn, 1, 2)

        o.ui_chkRTPmask_btn = QtWidgets.QPushButton("RTP mask")
        o.ui_chkRTPmask_btn.clicked.connect(
            lambda: o.check_onAFNI("func", "RTPmask")
        )
        ChkMask_gLayout.addWidget(o.ui_chkRTPmask_btn, 1, 3)
        # endregion

        # endregion: Preprocessing tab ----------------------------------------

        # --- Template tab ----------------------------------------------------
        o.ui_templateTab = QtWidgets.QWidget()
        o.ui_top_tabs.addTab(o.ui_templateTab, "Template")
        template_fLayout = QtWidgets.QFormLayout(o.ui_templateTab)

        o.ui_Template_grpBx = QtWidgets.QGroupBox("Template images")
        Template_gLayout = QtWidgets.QGridLayout(o.ui_Template_grpBx)
        template_fLayout.addWidget(o.ui_Template_grpBx)
        o.ui_objs.append(o.ui_Template_grpBx)

        # The file dialog for a template file starts from the directory of
        # the current file.
        # -- Template image --
        (
            _, o.ui_template_lnEd, o.ui_template_btn,
            o.ui_template_del_btn
        ) = self._add_file_row(
            Template_gLayout, 0, "Template brain :", "template",
            lambda: os.path.dirname(o.ui_template_lnEd.text()),
        )

        # -- ROI on template --
        (
            o.ui_ROI_template_lb, o.ui_ROI_template_lnEd,
            o.ui_ROI_template_btn, o.ui_ROI_template_del_btn
        ) = self._add_file_row(
            Template_gLayout, 1, "ROI on template :", "ROI_template",
            lambda: os.path.dirname(o.ui_ROI_template_lnEd.text()),
        )

        # ROI resampling mode
        var_ROI_resample_lb = QtWidgets.QLabel("ROI resampling :")
        Template_gLayout.addWidget(var_ROI_resample_lb, 2, 0)

        o.ui_ROI_resample_cmbBx = QtWidgets.QComboBox()
        o.ui_ROI_resample_cmbBx.addItems(o.ROI_resample_opts)
        o.ui_ROI_resample_cmbBx.setCurrentText(o.ROI_resample)
        o.ui_ROI_resample_cmbBx.currentIndexChanged.connect(
            lambda idx: o.set_param(
                "ROI_resample",
                o.ui_ROI_resample_cmbBx.currentText(),
                o.ui_ROI_resample_cmbBx.setCurrentIndex,
            )
        )
        Template_gLayout.addWidget(o.ui_ROI_resample_cmbBx, 2, 1)

        # -- WM on template --
        (
            o.ui_WM_template_lb, o.ui_WM_template_lnEd,
            o.ui_WM_template_btn, o.ui_WM_template_del_btn
        ) = self._add_file_row(
            Template_gLayout, 3, "White matter on template :", "WM_template",
            lambda: os.path.dirname(o.ui_WM_template_lnEd.text()),
        )

        # -- Ventricle on template --
        (
            o.ui_Vent_template_lb, o.ui_Vent_template_lnEd,
            o.ui_Vent_template_btn, o.ui_Vent_template_del_btn
        ) = self._add_file_row(
            Template_gLayout, 4, "Ventricle on template :", "Vent_template",
            lambda: os.path.dirname(o.ui_Vent_template_lnEd.text()),
        )

        # --- Processed image tab ---------------------------------------------
        o.ui_procImgTab = QtWidgets.QWidget()
        o.ui_top_tabs.addTab(o.ui_procImgTab, "Processed images")
        procImg_gLayout = QtWidgets.QGridLayout(o.ui_procImgTab)

        # -- Distortion correction warp --
        (
            _, o.ui_sdc_invwarp_lnEd, o.ui_sdc_invwarp_btn,
            o.ui_sdc_invwarp_del_btn
        ) = self._add_file_row(
            procImg_gLayout, 0, "Distortion warp :", "sdc_invwarp", work_dir
        )

        # -- Distortion-corrected functional image --
        (
            _, o.ui_func_orig_sdc_lnEd, o.ui_func_orig_sdc_btn,
            o.ui_func_orig_sdc_del_btn
        ) = self._add_file_row(
            procImg_gLayout, 1, "Corrected function :", "func_orig_sdc",
            work_dir,
        )

        # -- Aligned anatomy --
        alAnat_row = self._add_file_row(
            procImg_gLayout, 2, "Aligned anatomy :", "alAnat", work_dir
        )
        (
            _, o.ui_alAnat_lnEd, o.ui_alAnat_btn, o.ui_alAnat_del_btn
        ) = alAnat_row
        o.ui_objs.extend(alAnat_row)

        # -- Warped images --
        o.ui_wrpImg_grpBx = QtWidgets.QGroupBox("Warped images")
        wrpImg_gLayout = QtWidgets.QGridLayout(o.ui_wrpImg_grpBx)
        procImg_gLayout.addWidget(o.ui_wrpImg_grpBx, 3, 0, 1, 4)
        o.ui_objs.append(o.ui_wrpImg_grpBx)

        # -- WM in the original space --
        (
            _, o.ui_WM_orig_lnEd, o.ui_WM_orig_btn,
            o.ui_WM_orig_del_btn
        ) = self._add_file_row(
            wrpImg_gLayout, 0, "WM mask in original :", "WM_orig", work_dir
        )

        # -- Vent in the original space --
        (
            _, o.ui_Vent_orig_lnEd, o.ui_Vent_orig_btn,
            o.ui_Vent_orig_del_btn
        ) = self._add_file_row(
            wrpImg_gLayout, 1, "Vent mask in original :", "Vent_orig",
            work_dir,
        )

        # -- ROI in the original space --
        (
            o.ui_ROI_orig_lb, o.ui_ROI_orig_lnEd, o.ui_ROI_orig_btn,
            o.ui_ROI_orig_del_btn
        ) = self._add_file_row(
            wrpImg_gLayout, 2, "ROI mask in original :", "ROI_orig", work_dir
        )

        # -- RTP mask --
        RTP_mask_row = self._add_file_row(
            procImg_gLayout, 4, "RTP mask :", "RTP_mask", work_dir
        )
        (
            _, o.ui_RTP_mask_lnEd, o.ui_RTP_mask_btn,
            o.ui_RTP_mask_del_btn
        ) = RTP_mask_row
        o.ui_objs.extend(RTP_mask_row)

        # -- GSR mask --
        GSR_mask_row = self._add_file_row(
            procImg_gLayout, 5, "GSR mask :", "GSR_mask", work_dir
        )
        (
            _, o.ui_GSR_mask_lnEd, o.ui_GSR_mask_btn,
            o.ui_GSR_mask_del_btn
        ) = GSR_mask_row
        o.ui_objs.extend(GSR_mask_row)

        # -- Delete all processed images --
        o.ui_delAllProcImgs_btn = QtWidgets.QPushButton("Delete All")
        o.ui_delAllProcImgs_btn.setStyleSheet(
            "background-color: rgb(255,0,0);"
        )
        o.ui_delAllProcImgs_btn.clicked.connect(
            lambda: o.delete_file("AllProc")
        )
        procImg_gLayout.addWidget(o.ui_delAllProcImgs_btn, 6, 3)
        o.ui_objs.append(o.ui_delAllProcImgs_btn)

        if o.run_extApp:
            # --- External App tab ------------------------------------------
            o.ui_extAppTab = QtWidgets.QWidget()
            o.ui_top_tabs.addTab(o.ui_extAppTab, "Ext App")
            o.ui_extApp_fLayout = QtWidgets.QFormLayout(o.ui_extAppTab)

            # Command line
            var_lb = QtWidgets.QLabel("App command:")
            o.ui_extApp_cmd_lnEd = QtWidgets.QLineEdit()
            o.ui_extApp_cmd_lnEd.setText(str(o.extApp_cmd))
            o.ui_extApp_cmd_lnEd.editingFinished.connect(
                lambda: o.set_param(
                    "extApp_cmd", o.ui_extApp_cmd_lnEd.text()
                )
            )
            o.ui_extApp_fLayout.addRow(var_lb, o.ui_extApp_cmd_lnEd)
            o.ui_objs.append(o.ui_extApp_cmd_lnEd)

            # Boot App
            o.ui_extApp_run_btn = QtWidgets.QPushButton("Run App")
            o.ui_extApp_run_btn.clicked.connect(
                lambda x: o.boot_extApp()
            )
            o.ui_extApp_fLayout.addRow(o.ui_extApp_run_btn, None)
            o.ui_objs.append(o.ui_extApp_run_btn)

            # Address
            var_lb = QtWidgets.QLabel("App server address\n(host:port):")
            o.ui_extApp_addr_lnEd = QtWidgets.QLineEdit()
            o.ui_extApp_addr_lnEd.editingFinished.connect(
                lambda: o.set_param(
                    "extApp_addr",
                    o.ui_extApp_addr_lnEd.text(),
                    o.ui_extApp_addr_lnEd.setText,
                )
            )
            if o.extApp_addr is not None:
                addr_str = "{}:{}".format(*o.extApp_addr)
                o.ui_extApp_addr_lnEd.setText(addr_str)
            o.ui_extApp_fLayout.addRow(var_lb, o.ui_extApp_addr_lnEd)
            o.ui_objs.append(o.ui_extApp_addr_lnEd)

            # Check alive
            o.ui_extApp_isAlive_btn = QtWidgets.QPushButton(
                "Check connection"
            )
            o.ui_extApp_isAlive_btn.clicked.connect(
                lambda x: o.set_param(
                    "extApp_isAlive", o.ui_extApp_addr_lnEd.text(),
                    o.ui_extApp_isAlive_lb.setText
                )
            )
            o.ui_extApp_isAlive_lb = QtWidgets.QLabel("")
            o.ui_extApp_fLayout.addRow(
                o.ui_extApp_isAlive_btn, o.ui_extApp_isAlive_lb
            )
            o.ui_objs.append(o.ui_extApp_isAlive_btn)

        else:
            # --- Output tab ------------------------------------------------
            o.ui_extAppTab = QtWidgets.QWidget()
            o.ui_top_tabs.addTab(o.ui_extAppTab, "Output")
            o.ui_extApp_gLayout = QtWidgets.QGridLayout(o.ui_extAppTab)

            # Real-time signal output file
            var_lb = QtWidgets.QLabel("Real-time output file :")
            o.ui_extApp_gLayout.addWidget(var_lb, 0, 0)

            o.ui_sigSaveFile_lnEd = QtWidgets.QLineEdit()
            o.ui_extApp_gLayout.addWidget(o.ui_sigSaveFile_lnEd, 0, 1)
            o.ui_sigSaveFile_lnEd.setText(str(o.sig_save_file))
            o.ui_sigSaveFile_lnEd.editingFinished.connect(
                lambda: o.set_param(
                    "sig_save_file",
                    o.ui_sigSaveFile_lnEd.text(),
                    o.ui_sigSaveFile_lnEd.setText,
                )
            )

            o.ui_sigSaveFile_btn = QtWidgets.QPushButton("Set")
            o.ui_sigSaveFile_btn.clicked.connect(
                lambda: o.set_param(
                    "sig_save_file", "", o.ui_sigSaveFile_lnEd.setText
                )
            )
            o.ui_sigSaveFile_btn.setStyleSheet(_SET_BTN_STYLE)
            o.ui_extApp_gLayout.addWidget(o.ui_sigSaveFile_btn, 0, 2)

        # --- RTP setup button ------------------------------------------------
        o.ui_setRTP_btn = QtWidgets.QPushButton("RTP setup")
        o.ui_setRTP_btn.setStyleSheet(
            "background-color: rgb(151,217,235);height: 20px;"
        )

        o.ui_setRTP_btn.clicked.connect(o.RTP_setup)
        ui_rows.append((o.ui_setRTP_btn,))
        o.ui_objs.append(o.ui_setRTP_btn)

        # --- Show Extracted Signal checkbox ----------------------------------
        o.ui_showROISig_cbx = QtWidgets.QCheckBox("Show Extracted Signal")
        o.ui_showROISig_cbx.setCheckState(0)
        o.ui_showROISig_cbx.stateChanged.connect(
            lambda x: self.show_ROIsig_chk(x)
        )
        # RtpGUI.layout_ui places this checkbox.

        # --- Ready and Quit buttons ------------------------------------------
        readyQuitWdt = QtWidgets.QWidget()
        o.main_win.hBoxExpCtrls.addWidget(readyQuitWdt)
        readyQuit_gLayout = QtWidgets.QGridLayout(readyQuitWdt)

        # -- Ready button --
        o.ui_ready_btn = QtWidgets.QPushButton("Ready")
        o.ui_ready_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        o.ui_ready_btn.clicked.connect(o.ready_to_run)
        o.ui_ready_btn.setEnabled(False)
        o.ui_ready_btn.setStyleSheet(
            "background-color: rgb(255,201,32);height: 20px;"
        )
        readyQuit_gLayout.addWidget(o.ui_ready_btn, 0, 0, 1, 4)

        # -- Manual start button --
        o.ui_manStart_btn = QtWidgets.QPushButton("Manual start")
        o.ui_manStart_btn.setStyleSheet("background-color: rgb(94,63,153);")
        o.ui_manStart_btn.clicked.connect(o.manual_start)
        o.ui_manStart_btn.setEnabled(False)
        readyQuit_gLayout.addWidget(o.ui_manStart_btn, 0, 4, 1, 1)

        # -- Quit button --
        o.ui_quit_btn = QtWidgets.QPushButton("Cancel")
        o.ui_quit_btn.setStyleSheet("background-color: rgb(255,0,0);")
        o.ui_quit_btn.clicked.connect(partial(o.end_run, True))
        o.ui_quit_btn.setEnabled(False)
        readyQuit_gLayout.addWidget(o.ui_quit_btn, 0, 5, 1, 1)

        return ui_rows


# %% PlotROISignal ============================================================
class PlotROISignal(QtCore.QObject):
    """Window to plot the ROI signal in real time."""

    finished = QtCore.pyqtSignal()

    def __init__(self, root, num_ROIs=1, roi_labels=[], main_win=None):
        super().__init__()

        self.root = root
        self.main_win = main_win
        self.abort = False

        self.plt_win = MatplotlibWindow()
        self.plt_win.setWindowTitle("ROI signal")

        # Place the window to the right of the main window
        if main_win is not None:
            main_geom = main_win.geometry()
            x = main_geom.x() + main_geom.width() + 10
            y = main_geom.y() + 735
        else:
            x, y = (0, 0)
        win_height = 80 + 70 * num_ROIs
        self.plt_win.setGeometry(x, y, 500, win_height)

        self.roi_labels = roi_labels
        self._axes = self.plt_win.canvas.figure.subplots(num_ROIs, 1)
        if num_ROIs == 1:
            self._axes = [self._axes]

        self.plt_win.canvas.figure.subplots_adjust(
            left=0.15, bottom=0.18, right=0.95, top=0.96, hspace=0.35
        )
        self._color_cycle = plt.get_cmap("tab10")

        self.reset_plot()

        self.plt_win.show()
        self.plt_win.canvas.draw()

        # Qt widgets must be updated in the GUI thread. A timer in the
        # GUI thread polls the ROI signal and updates the plot.
        self._plt_xi = []
        self._running = False
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._update_plot)

    # ---------------------------------------------------------------------
    def reset_plot(self):
        self._ln = []
        for ii, ax in enumerate(self._axes):
            ax.cla()
            if len(self.roi_labels) > ii:
                ax.set_ylabel(self.roi_labels[ii])
            ax.set_xlim(0, 10)
            self._ln.append(ax.plot(0, 0, color=self._color_cycle(ii + 1)))
        self._axes[-1].set_xlabel("TR")

    # ---------------------------------------------------------------------
    def run(self):
        """Start updating the plot."""
        self._plt_xi = self.root._plt_xi.copy()
        self._running = True
        self._timer.start()

    # ---------------------------------------------------------------------
    def isRunning(self):
        return self._running

    # ---------------------------------------------------------------------
    def _update_plot(self):
        main_closed = (
            self.main_win is not None and not self.main_win.isVisible()
        )
        if not self.plt_win.isVisible() or self.abort or main_closed:
            self.end_thread()
            return

        if len(self.root._plt_xi) == len(self._plt_xi):
            return

        try:
            # Plot signal
            plt_xi = self.root._plt_xi.copy()
            self._plt_xi = plt_xi
            plt_roi_sig = self.root._roi_sig
            for ii, ax in enumerate(self._axes):
                ll = min(len(plt_xi), len(plt_roi_sig[ii]))
                if ll == 0:
                    continue

                y = plt_roi_sig[ii][:ll]
                self._ln[ii][0].set_data(plt_xi[:ll], y)

                # Adjust the y range, ignoring outliers beyond 4 SD
                ax.relim()
                if np.sum(~np.isnan(y)) > 10:
                    yl_orig = np.array(ax.get_ylim())
                    yl = yl_orig.copy()
                    sd = np.nanstd(y)
                    mu = np.nanmean(y)
                    if np.nanmin(y) < yl[0]:
                        yl[0] = max(np.nanmin(y), mu - 4 * sd)

                    if np.nanmax(y) > yl[1]:
                        yl[1] = min(np.nanmax(y), mu + 4 * sd)

                    if np.any(yl != yl_orig):
                        ax.set_ylim(yl)

                ax.autoscale_view()

                # Extend the x range in steps of 10
                xl = ax.get_xlim()
                if (plt_xi[-1] // 10 + 1) * 10 > xl[1]:
                    ax.set_xlim([0, (plt_xi[-1] // 10 + 1) * 10])
            self.plt_win.canvas.draw()

        except IndexError:
            pass

        except Exception as e:
            self.root._logger.error(str(e))
            sys.stdout.flush()

    # ---------------------------------------------------------------------
    def end_thread(self):
        if not self._running:
            return
        self._running = False
        self._timer.stop()

        if self.plt_win.isVisible():
            self.plt_win.close()

        self.finished.emit()

        if hasattr(self.root, "ui_showROISig_cbx"):
            self.root.ui_showROISig_cbx.setCheckState(0)


# %% WAIT_ONSET ===============================================================
class WAIT_ONSET(QtCore.QObject):
    """Wait for the scan onset and notify the external application.

    Run in a QThread. The onset time is read from onsetObj
    (RtpTTLPhysio).
    """

    finished = QtCore.pyqtSignal()

    def __init__(self, parent, onsetObj=None, extApp_sock=None):
        super().__init__()
        self.parent = parent
        self.onsetObj = onsetObj
        self.extApp_sock = extApp_sock
        self.abort = False

    def run(self):
        onset_time = 0
        while not self.abort:
            if self.onsetObj is not None and self.onsetObj.available:
                onset_time = self.onsetObj.scan_onset
                if onset_time != 0:
                    break
            time.sleep(1 / 60)

        if self.abort:
            self.finished.emit()
            return

        self.parent._scanning = True
        self.parent._wait_start = False
        self.parent.scan_onset = onset_time
        if self.onsetObj is not None and self.onsetObj.available:
            self.onsetObj.start_scan()

        if self.extApp_sock is not None:
            _send_scan_start(self.extApp_sock, onset_time)

        self.finished.emit()
