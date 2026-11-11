# -*- coding: utf-8 -*-
"""
GUI panel of RtpVolreg, with the motion plot window

@author: mmisaki@laureateinstitute.org
"""

# %% import ===================================================================
import sys
import traceback
import logging

import numpy as np
from PyQt6 import QtWidgets, QtCore

from ..rtp_volreg import RtpVolreg, framewise_displacement
from .module_ui import RtpModuleUI, value_combo
from .widgets import MatplotlibWindow
from . import window_place


# %% RtpVolregUI ==============================================================
class RtpVolregUI(RtpModuleUI):
    """GUI panel of RtpVolreg."""

    _widget_setters = {
        "ignore_init": ("ui_ignorInit_spBx", "setValue"),
        "regmode": ("ui_regmode_cmbBx", "setCurrentIndex",
                    RtpVolreg.reg_regmodes.index),
        "max_scan_length": ("ui_maxLen_spBx", "setValue"),
        # Reference volume source (0: external file) and description
        "ref_vol_source": ("ui_baseVol_cmbBx", "setCurrentIndex"),
        "ref_vol_text": ("ui_baseVol_lnEd", "setText", str),
    }

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def make_ui(self):
        o = self.obj
        o.ui_objs = []
        ui_rows = [self.enabled_row()]

        # ignore_init
        var_lb = QtWidgets.QLabel("Ignore initial volumes :")
        o.ui_ignorInit_spBx = QtWidgets.QSpinBox()
        o.ui_ignorInit_spBx.setValue(o.ignore_init)
        o.ui_ignorInit_spBx.setMinimum(0)
        o.ui_ignorInit_spBx.valueChanged.connect(
                lambda x: o.set_param('ignore_init', x,
                                      o.ui_ignorInit_spBx.setValue))
        ui_rows.append((var_lb, o.ui_ignorInit_spBx))
        o.ui_objs.extend([var_lb, o.ui_ignorInit_spBx])

        # ref_vol
        var_lb = QtWidgets.QLabel("Base volume :")
        o.ui_baseVol_cmbBx = value_combo(
            [('File...', 'external file'),
             ('Volume of this run...', 'index of internal run')])
        o.ui_baseVol_cmbBx.activated.connect(
                lambda idx:
                o.set_param('ref_vol',
                            o.ui_baseVol_cmbBx.currentData(),
                            o.ui_baseVol_cmbBx.setCurrentIndex))
        ui_rows.append((var_lb, o.ui_baseVol_cmbBx))

        o.ui_baseVol_lnEd = QtWidgets.QLineEdit()
        o.ui_baseVol_lnEd.setReadOnly(True)
        o.ui_baseVol_lnEd.setStyleSheet(
            'border: 0px none;')
        ui_rows.append((None, o.ui_baseVol_lnEd))

        o.ui_objs.extend([var_lb, o.ui_baseVol_cmbBx,
                          o.ui_baseVol_lnEd])

        if o.ref_fname:
            o.ui_baseVol_cmbBx.setCurrentIndex(0)
            o.ui_baseVol_lnEd.setText(str(o.ref_fname))
        elif not isinstance(o.ref_vol, int):
            o.ui_baseVol_cmbBx.setCurrentIndex(1)
            o.ui_baseVol_lnEd.setText("Set from this run")
        else:
            o.ui_baseVol_cmbBx.setCurrentIndex(1)
            o.ui_baseVol_lnEd.setText(
                    f"Volume {o.ref_vol} of this run")

        # regmode
        var_lb = QtWidgets.QLabel("Resampling :")
        o.ui_regmode_cmbBx = QtWidgets.QComboBox()
        o.ui_regmode_cmbBx.addItems(
            [m.capitalize() for m in RtpVolreg.reg_regmodes])
        o.ui_regmode_cmbBx.setCurrentIndex(
            RtpVolreg.reg_regmodes.index(o.regmode))
        o.ui_regmode_cmbBx.currentIndexChanged.connect(
                lambda idx:
                o.set_param('regmode', RtpVolreg.reg_regmodes[idx],
                            o.ui_regmode_cmbBx.setCurrentIndex))
        ui_rows.append((var_lb, o.ui_regmode_cmbBx))
        o.ui_objs.extend([var_lb, o.ui_regmode_cmbBx])

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

        # --- Checkbox row ----------------------------------------------------
        ui_rows.append(self.save_proc_row())

        return ui_rows

    # --- Motion plot ---------------------------------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def open_motion_plot(self):
        if hasattr(self, 'pltMotion') and self.pltMotion.isRunning():
            return

        self.pltMotion = PlotMotion(self.obj, main_win=self.obj.main_win)
        self.pltMotion.run()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def close_motion_plot(self):
        if hasattr(self, 'pltMotion'):
            pltMotion = self.pltMotion
            del self.pltMotion
            pltMotion.end_thread()

        main_win = self.obj.main_win
        if main_win is not None:
            if main_win.chbShowMotion.isChecked():
                main_win.chbShowMotion.setChecked(False)


# FD (mm) at and above which a volume is marked red in the motion plot
FD_WARN_MM = 0.5


# %% PlotMotion ===============================================================
class PlotMotion(QtCore.QObject):
    """Plot the motion parameters and framewise displacement (FD)."""
    finished = QtCore.pyqtSignal()

    def __init__(self, root, main_win=None):
        super().__init__()

        self._logger = logging.getLogger(self.__class__.__name__)
        self.root = root
        self.main_win = main_win
        self.abort = False

        # Initialize the figure
        self.plt_win = MatplotlibWindow()
        self.plt_win.setWindowTitle('Motion')

        # Next to the main window (window_place.py)
        window_place.set_geometry(self.plt_win, main_win, "motion")

        # Set the axes
        self.mot_labels = ['roll (deg.)', 'pitch (deg.)', 'yaw (deg.)',
                           'dS (mm)', 'dL (mm)', 'dP (mm)', 'FD (mm)']
        self._axes = self.plt_win.canvas.figure.subplots(7, 1)
        # Margins fitted to the tick and axis labels (they are not cut in a
        # narrow window)
        self.plt_win.canvas.figure.set_layout_engine(
            "constrained", w_pad=0.04, h_pad=0.03)
        self._ln = []
        for ii, ax in enumerate(self._axes):
            ax.set_ylabel(self.mot_labels[ii])
            ax.set_xlim(0, 10)
            self._ln.append(ax.plot(0, 0))
            if ii < 6:
                ax.set_ylim(-0.1, 0.1)
            else:
                # FD warning threshold
                ax.axhline(FD_WARN_MM, c='r', ls='--', lw=0.5)
                ax.set_ylim(0, FD_WARN_MM + 0.1)
        ax.set_xlabel('TR')

        self.plt_win.show()
        self.plt_win.canvas.draw()

        # Qt widgets must be updated in the GUI thread. A timer in the
        # GUI thread polls the motion parameters and updates the plot.
        self._fd_alarm_bands = []
        self._plt_xi = []
        self._running = False
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._update_plot)

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
            # No new volume
            return

        try:
            plt_xi = np.array(self.root._plt_xi.copy())
            self._plt_xi = list(plt_xi)
            plt_motion = self.root._plt_motion
            # Framewise displacement (Power et al., 2012; the same as the FD
            # of REGRESS): rotations converted to mm on a 50 mm sphere
            n_mot = min(len(m) for m in plt_motion)
            plt_FD = framewise_displacement(
                np.array([m[:n_mot] for m in plt_motion]).T)

            for ii, ax in enumerate(self._axes):
                yl = ax.get_ylim()
                if ii < len(plt_motion):
                    ll = min(len(plt_xi), len(plt_motion[ii]))
                    if ll == 0:
                        continue
                    self._ln[ii][0].set_data(plt_xi[:ll],
                                             plt_motion[ii][:ll])
                    ymin = np.nanmin(plt_motion[ii]) - \
                        (yl[1] - yl[0]) * 0.1
                    ymax = np.nanmax(plt_motion[ii]) + \
                        (yl[1] - yl[0]) * 0.1
                else:
                    ll = min(len(plt_xi), len(plt_FD))
                    if ll == 0:
                        continue
                    self._ln[ii][0].set_data(plt_xi[:ll],
                                             plt_FD[:ll])
                    ymin = np.nanmin(plt_FD) - (yl[1] - yl[0]) * 0.1
                    ymax = np.nanmax(plt_FD) + (yl[1] - yl[0]) * 0.1

                # Expand ylim in steps of 0.05
                if ymin < yl[0] or ymax > yl[1]:
                    ymin = min(ymin, yl[0])
                    ymin = np.floor(ymin/0.05)*0.05
                    ymax = max(ymax, yl[1])
                    ymax = np.ceil(ymax/0.05)*0.05
                    ax.set_ylim([ymin, ymax])

                ax.relim()
                ax.autoscale_view()

                xl = ax.get_xlim()
                if (plt_xi[-1]//10 + 1)*10 > xl[1]:
                    ax.set_xlim([0, (plt_xi[-1]//10 + 1)*10])

            # Mark volumes with a large FD
            for band in self._fd_alarm_bands:
                band.remove()

            self._fd_alarm_bands = []
            fd_mark_idx = np.argwhere(
                plt_FD[:len(plt_xi)] >= FD_WARN_MM).ravel()

            ax_FD = self._axes[-1]
            for x in plt_xi[fd_mark_idx]:
                band = ax_FD.axvspan(x-1, x, ec='k', fc='red',
                                     alpha=0.3, linewidth=0.1)
                self._fd_alarm_bands.append(band)

            self.plt_win.canvas.draw()

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errmsg = ''.join(
                traceback.format_exception(exc_type, exc_obj, exc_tb))
            self._logger.error(str(e) + '\n' + errmsg)
            sys.stderr.write(errmsg)

    # ---------------------------------------------------------------------
    def end_thread(self):
        """Stop updating and close the plot (called in the GUI thread)."""
        if not self._running:
            return
        self._running = False
        self._timer.stop()

        if self.plt_win.isVisible():
            self.plt_win.close()

        self.finished.emit()

        if self.main_win is not None:
            if hasattr(self.main_win, 'chbShowMotion'):
                self.main_win.chbShowMotion.setChecked(False)
