# -*- coding: utf-8 -*-
"""
GUI panel of RtpVolreg, with the motion plot window

@author: mmisaki@laureateinstitute.org
"""

# %% import ===================================================================
import sys
import time
import traceback
import logging

import numpy as np
from PyQt5 import QtWidgets, QtCore

from ..rtp_volreg import RtpVolreg
from .module_ui import RtpModuleUI
from .widgets import MatplotlibWindow


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
        o.ui_baseVol_cmbBx = QtWidgets.QComboBox()
        o.ui_baseVol_cmbBx.addItems(['external file',
                                     'index of internal run'])
        o.ui_baseVol_cmbBx.activated.connect(
                lambda idx:
                o.set_param('ref_vol',
                            o.ui_baseVol_cmbBx.currentText(),
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
            o.ui_baseVol_lnEd.setText("Set from the internal run")
        else:
            o.ui_baseVol_cmbBx.setCurrentIndex(1)
            o.ui_baseVol_lnEd.setText(
                    f"Internal run volume index {o.ref_vol}")

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
        var_lb = QtWidgets.QLabel("Maximum scan length :")
        o.ui_maxLen_spBx = QtWidgets.QSpinBox()
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
        if hasattr(self, 'thPltMotion') and self.thPltMotion.isRunning():
            return

        self.thPltMotion = QtCore.QThread()
        self.pltMotion = PlotMotion(self.obj, main_win=self.obj.main_win)
        self.pltMotion.moveToThread(self.thPltMotion)
        self.thPltMotion.started.connect(self.pltMotion.run)
        self.pltMotion.finished.connect(self.thPltMotion.quit)
        self.thPltMotion.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def close_motion_plot(self):
        if hasattr(self, 'thPltMotion') and self.thPltMotion.isRunning():
            self.pltMotion.abort = True
            if not self.thPltMotion.wait(1):
                self.pltMotion.finished.emit()
                self.thPltMotion.wait()

            del self.thPltMotion

        if hasattr(self, 'pltMotion'):
            del self.pltMotion

        main_win = self.obj.main_win
        if main_win is not None:
            if main_win.chbShowMotion.checkState() != 0:
                main_win.chbShowMotion.setCheckState(0)


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

        # Set the position
        if main_win is not None:
            main_geom = main_win.geometry()
            x = main_geom.x() + main_geom.width() + 10
            y = main_geom.y() + 400
        else:
            x, y = (0, 0)
        self.plt_win.setGeometry(x, y, 500, 500)

        # Set the axes
        self.mot_labels = ['roll (deg.)', 'pitch (deg.)', 'yaw (deg.)',
                           'dS (mm)', 'dL (mm)', 'dP (mm)', 'FD']
        self._axes = self.plt_win.canvas.figure.subplots(7, 1)
        self.plt_win.canvas.figure.subplots_adjust(
                left=0.15, bottom=0.08, right=0.95, top=0.97, hspace=0.35)
        self._ln = []
        for ii, ax in enumerate(self._axes):
            ax.set_ylabel(self.mot_labels[ii])
            ax.set_xlim(0, 10)
            self._ln.append(ax.plot(0, 0))
            if ii < 6:
                ax.set_ylim(-0.1, 0.1)
            else:
                # FD alarm thresholds
                ax.axhline(0.2, c='k', ls=':', lw=0.5)
                ax.axhline(0.3, c='r', ls='--', lw=0.5)
                ax.set_ylim(0, 0.5)
        ax.set_xlabel('TR')

        self.plt_win.show()
        self.plt_win.canvas.draw()

    # ---------------------------------------------------------------------
    def run(self):
        self._fd_alarm_bands = []
        plt_xi = self.root._plt_xi.copy()
        while self.plt_win.isVisible() and not self.abort:
            if self.main_win is not None and not self.main_win.isVisible():
                break

            if len(self.root._plt_xi) == len(plt_xi):
                # No new volume
                time.sleep(0.1)
                continue

            try:
                plt_xi = np.array(self.root._plt_xi.copy())
                plt_motion = self.root._plt_motion
                # Framewise displacement: sum of the absolute
                # differences of the six parameters
                plt_FD = np.zeros_like(plt_motion[0])
                if len(plt_motion[0]) > 1:
                    plt_FD[1:] = np.sum(
                        np.abs(np.diff(np.array(plt_motion), axis=1)),
                        axis=0)

                for ii, ax in enumerate(self._axes):
                    yl = ax.get_ylim()
                    if ii < len(plt_motion):
                        ll = min(len(plt_xi), len(plt_motion[ii]))
                        if ll == 0:
                            continue
                        self._ln[ii][0].set_data(plt_xi[:ll],
                                                 plt_motion[ii][:ll])
                        ymin = np.nanmin(plt_motion[ii]) - \
                            np.diff(yl) * 0.1
                        ymax = np.nanmax(plt_motion[ii]) + \
                            np.diff(yl) * 0.1
                    else:
                        ll = min(len(plt_xi), len(plt_FD))
                        if ll == 0:
                            continue
                        self._ln[ii][0].set_data(plt_xi[:ll],
                                                 plt_FD[:ll])
                        ymin = np.nanmin(plt_FD) - np.diff(yl) * 0.1
                        ymax = np.nanmax(plt_FD) + np.diff(yl) * 0.1

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
                fd_mark_02_idx = np.argwhere(
                    (plt_FD >= 0.2) & (plt_FD < 0.3)).ravel()
                fd_mark_03_idx = np.argwhere(plt_FD >= 0.3).ravel()

                ax_FD = self._axes[-1]
                for x in plt_xi[fd_mark_02_idx]:
                    band = ax_FD.axvspan(x-1, x, ec='k', fc='yellow',
                                         alpha=0.3, linewidth=0.1)
                    self._fd_alarm_bands.append(band)

                for x in plt_xi[fd_mark_03_idx]:
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
                continue

        self.end_thread()

    # ---------------------------------------------------------------------
    def end_thread(self):
        if self.plt_win.isVisible():
            self.plt_win.close()

        self.finished.emit()

        if self.main_win is not None:
            if hasattr(self.main_win, 'chbShowMotion'):
                self.main_win.chbShowMotion.setCheckState(0)
