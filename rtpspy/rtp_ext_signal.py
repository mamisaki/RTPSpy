#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mmisaki@laureateinstitute.org
"""

# %% import ===================================================================
from multiprocessing import shared_memory
import time

import numpy as np
from PyQt5 import QtWidgets, QtCore
import matplotlib as mpl

try:
    from .rtp_common import RTP
    from .rtp_physio_gpio import call_rt_physio
except Exception:
    from rtpspy.rtp_common import RTP
    from rtpspy.rtp_physio_gpio import call_rt_physio

mpl.rcParams['font.size'] = 8


# %% RtpExtSignal class ======================================================
class RtpExtSignal(RTP):
    """
    RETORICOR timeseries creation
    Receive TTL, cardiogram, and respiration signals from rtp_physio
    process via shared memory.
    """
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, rtp_physio_address=('localhost', 63212)):
        super().__init__()  # call __init__() in RTP class

        self.rtp_physio_address = rtp_physio_address

        # --- recording status ---
        self._wait_start = False
        self._scanning = False
        self.scan_onset = 0.0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ getter method +++
    @property
    def scanning(self):
        return self.is_scan_on()

    @property
    def physio_available(self):
        try:
            return call_rt_physio(self.rtp_physio_address, 'ping')
        except Exception:
            return False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def reset(self):
        if self._verb:
            msg = "Reset scan status."
            self.logmsg(msg)

        self._wait_start = False
        self._scanning = False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def wait_scan_onset(self):
        if not self._scanning and not self._wait_start and \
                self.physio_available:
            call_rt_physio(self.rtp_physio_address, 'WAIT_TTL_ON')
            self._wait_start = True

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def abort_waiting(self):
        if self._wait_start and self.physio_available:
            call_rt_physio(self.rtp_physio_address, 'CANCEL_WAIT_TTL')
            self._wait_start = False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def is_scan_on(self):
        if not self._scanning and self._wait_start and self.physio_available:
            # Get scan onset
            shm = shared_memory.SharedMemory(name='scan_onset')
            onset = np.ndarray((1,), dtype=np.dtype(float), buffer=shm.buf)[0]
            shm.close()

            if onset > 0:
                self.scan_onset = onset
                self._scanning = True
                self._wait_start = False

        return self._scanning

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def manual_start(self):
        if not self._wait_start:
            return

        self.scan_onset = time.time()
        if self.verb:
            self.logmsg("Manual start")
        self._scanning = True

        self.abort_waiting()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end_reset(self):
        self._wait_start = False
        self._scanning = False
        self.scan_onset = 0.0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_param(self, attr, val=None, reset_fn=None, echo=False):
        """
        When reset_fn is None, set_param is considered to be called from
        load_parameters function.
        """

        # -- check value --
        if attr == 'sig_port' and reset_fn is None:
            if self.sig_port == val:
                return

            idx = self.ui_sigPort_cmbBx.findText(val, QtCore.Qt.MatchContains)
            if idx == -1:
                return

            if hasattr(self, 'ui_sigPort_cmbBx'):
                self.ui_sigPort_cmbBx.setCurrentIndex(idx)

            if val is not None:
                self._init_onsig_port(val)

            return

        elif attr == 'sample_freq':
            if reset_fn is None and hasattr(self, 'ui_sampFreq_dSpBx'):
                self.ui_sampFreq_dSpBx.setValue(val)

        elif attr == 'plot_len_sec' and reset_fn is None:
            if hasattr(self, 'ui_pltLen_dSpBx'):
                self.ui_pltLen_dSpBx.setValue(val)

        elif attr == 'buf_len_sec' and reset_fn is None:
            if hasattr(self, 'ui_bufLen_dSpBx'):
                self.ui_bufLen_dSpBx.setValue(val)

        elif attr == 'verb':
            if hasattr(self, 'ui_verb_chb'):
                self.ui_verb_chb.setChecked(val)

        elif reset_fn is None:
            # Ignore an unrecognized parameter
            if not hasattr(self, attr):
                self.errmsg(f"{attr} is unrecognized parameter.", no_pop=True)
                return

        # -- Set value --
        setattr(self, attr, val)
        if echo and self.verb:
            print("{}.".format(self.__class__.__name__) + attr, '=',
                  getattr(self, attr))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ui_set_param(self):

        ui_rows = []
        self.ui_objs = []

        # # sig_port combobox
        # var_lb = QtWidgets.QLabel("USB serial port to receive signals :")
        # self.ui_sigPort_cmbBx = QtWidgets.QComboBox()
        # devlist = sorted([f"{lab} : ({dev})"
        #                   for lab, dev in self._dict_onsig_port.items()])
        # self.ui_sigPort_cmbBx.addItems(devlist)
        # if len(devlist) and self.sig_port is not None:
        #     try:
        #         selIdx = list(self._dict_onsig_port).index(self.sig_port)
        #         self.ui_sigPort_cmbBx.setCurrentIndex(selIdx)
        #     except ValueError:
        #         pass

        # self.ui_sigPort_cmbBx.activated.connect(
        #         lambda idx: self.set_param(
        #                 'sig_port', self.ui_sigPort_cmbBx.currentText(),
        #                 self.ui_sigPort_cmbBx.setCurrentIndex))
        # ui_rows.append((var_lb, self.ui_sigPort_cmbBx))
        # self.ui_objs.extend([var_lb, self.ui_sigPort_cmbBx])

        # # update port list button
        # self.ui_serPortUpdate_btn = QtWidgets.QPushButton('Update port list')
        # self.ui_serPortUpdate_btn.clicked.connect(self._update_port_list)
        # ui_rows.append((self.ui_serPortUpdate_btn,))
        # self.ui_objs.extend([var_lb, self.ui_serPortUpdate_btn])

        # # sample_freq
        # var_lb = QtWidgets.QLabel("Sampling frequency :")
        # self.ui_sampFreq_dSpBx = QtWidgets.QDoubleSpinBox()
        # self.ui_sampFreq_dSpBx.setMinimum(1.0)
        # self.ui_sampFreq_dSpBx.setMaximum(1000.0)
        # self.ui_sampFreq_dSpBx.setSingleStep(10.0)
        # self.ui_sampFreq_dSpBx.setDecimals(2)
        # self.ui_sampFreq_dSpBx.setSuffix(" Hz")
        # self.ui_sampFreq_dSpBx.setValue(self.sample_freq)
        # self.ui_sampFreq_dSpBx.valueChanged.connect(
        #         lambda x: self.set_param('sample_freq', x,
        #                                  self.ui_sampFreq_dSpBx.setValue))
        # ui_rows.append((var_lb, self.ui_sampFreq_dSpBx))
        # self.ui_objs.extend([var_lb, self.ui_sampFreq_dSpBx])

        # # plot_len_sec
        # var_lb = QtWidgets.QLabel("Signal plot length :")
        # self.ui_pltLen_dSpBx = QtWidgets.QDoubleSpinBox()
        # self.ui_pltLen_dSpBx.setMinimum(1)
        # self.ui_pltLen_dSpBx.setSingleStep(1)
        # self.ui_pltLen_dSpBx.setDecimals(1)
        # self.ui_pltLen_dSpBx.setSuffix(" seconds")
        # self.ui_pltLen_dSpBx.setValue(self.plot_len_sec)
        # self.ui_pltLen_dSpBx.valueChanged.connect(
        #         lambda x: self.set_param('plot_len_sec', x,
        #                                  self.ui_pltLen_dSpBx.setValue))
        # ui_rows.append((var_lb, self.ui_pltLen_dSpBx))
        # self.ui_objs.extend([var_lb, self.ui_pltLen_dSpBx])

        # # buf_len_sec
        # var_lb = QtWidgets.QLabel("Recording buffer size :")
        # self.ui_bufLen_dSpBx = QtWidgets.QDoubleSpinBox()
        # self.ui_bufLen_dSpBx.setMinimum(5)
        # self.ui_bufLen_dSpBx.setMaximum(36000)
        # self.ui_bufLen_dSpBx.setSingleStep(5)
        # self.ui_bufLen_dSpBx.setDecimals(0)
        # self.ui_bufLen_dSpBx.setSuffix(" seconds")
        # self.ui_bufLen_dSpBx.setValue(self.buf_len_sec)
        # self.ui_bufLen_dSpBx.valueChanged.connect(
        #         lambda x: self.set_param('buf_len_sec', x,
        #                                  self.ui_bufLen_dSpBx.setValue))
        # ui_rows.append((var_lb, self.ui_bufLen_dSpBx))
        # self.ui_objs.extend([var_lb, self.ui_bufLen_dSpBx])

        # manual start button
        self.ui_manualStart_btn = QtWidgets.QPushButton()
        self.ui_manualStart_btn.setText('Manual start')
        self.ui_manualStart_btn.setStyleSheet("background-color: rgb(255,0,0)")
        self.ui_manualStart_btn.clicked.connect(self.manual_start)
        ui_rows.append((None, self.ui_manualStart_btn))

        # --- Checkbox row ----------------------------------------------------
        # verb
        self.ui_verb_chb = QtWidgets.QCheckBox("Verbose logging")
        self.ui_verb_chb.setChecked(self.verb)
        self.ui_verb_chb.stateChanged.connect(
                lambda state: setattr(self, 'verb', state > 0))
        self.ui_objs.append(self.ui_verb_chb)

        chb_hLayout = QtWidgets.QHBoxLayout()
        chb_hLayout.addStretch()
        chb_hLayout.addWidget(self.ui_verb_chb)
        ui_rows.append((None, chb_hLayout))

        return ui_rows

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_params(self):
        all_opts = super().get_params()
        excld_opts = ('scan_onset', 'ignore_init')
        sel_opts = {}
        for k, v in all_opts.items():
            if k in excld_opts or k[0] == '_':
                continue
            sel_opts[k] = v

        return sel_opts

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        self.abort_waiting()


# %%
if __name__ == '__main__':
    rtp_ext_sig = RtpExtSignal()

    while not rtp_ext_sig.physio_available:
        pass

    rtp_ext_sig.wait_scan_onset()

    while not rtp_ext_sig.is_scan_on():
        pass

    print(time.ctime(rtp_ext_sig.scan_onset))
