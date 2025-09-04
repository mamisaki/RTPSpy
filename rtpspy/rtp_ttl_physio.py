#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

# %% import ===================================================================
from pathlib import Path
import time
import traceback
import subprocess
import shlex
import warnings

import numpy as np
from scipy.interpolate import interp1d

try:
    # Load modules from the same directory
    from .rtp_common import RTP
    from .rpc_socket_server import RPCSocketCom
    from .rtp_retrots import RtpRetroTS
    from .rt_physio import SharedMemoryRingBuffer

except Exception:
    # For DEBUG environment
    from rtpspy.rtp_common import RTP
    from rtpspy.rpc_socket_server import RPCSocketCom
    from rtpspy.rtp_retrots import RtpRetroTS
    from rtpspy.rt_physio import SharedMemoryRingBuffer


# %% RtpTTLPhysio =============================================================
class RtpTTLPhysio(RTP):
    """
    Interface for rt_physio process
    """
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(
        self,
        physio_log_file=None,
        rt_physio_address_name=["localhost", None, "RtTTLPhysioSocketServer"],
        config_path=Path.home() / ".RTPSpy" / "rtmri_config.json",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._retrots = RtpRetroTS()
        self.config_path = config_path
        self._rt_physio_proc = None
        self.TTLPhysioCom = None
        self.sample_freq = None
        self._scan_onset = None
        self._rbuf = {}

        self.physio_log_file = physio_log_file
        if self.physio_log_file is None:
            self.physio_log_file = Path(self.work_dir) / "rt_physio.log"
        self.rt_physio_address_name = rt_physio_address_name
        self.config_path = config_path
        self.init_timeout = 2

        self.init_rt_physio_access()

        self.scan_onset = 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def init_rt_physio_access(self):
        try:
            # Boot the rt_physio as an independent process
            rtpspy_path = Path(__file__).resolve().parent
            physio = rtpspy_path / 'rt_physio.py'
            cmd = (
                f"python {physio}"
                f" --log_file {self.physio_log_file.resolve()}"
                f" --rpc_socket_name {self.rt_physio_address_name[2]}"
                f" --config_path {self.config_path}"
            )

            self._rt_physio_proc = subprocess.Popen(shlex.split(cmd))
            # Check if the process is running
            if self._rt_physio_proc.poll() is not None:
                self._logger.error(
                    "rt_physio process terminated unexpectedly."
                )
                self._rt_physio_proc = None
                self.TTLPhysioCom = None
                return

            # Create the RPC communication channel
            self.TTLPhysioCom = RPCSocketCom(
                self.rt_physio_address_name, self.config_path
            )

            st = time.time()
            while (
                not self.TTLPhysioCom.rpc_ping() and
                (time.time() - st) < self.init_timeout
            ):
                time.sleep(0.1)

            if self.TTLPhysioCom is not None:
                self.init_rbuf_access()

        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            self._logger.error(f"Failed to connect to rt_physio: {errstr}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def init_rbuf_access(self):
        if self.TTLPhysioCom is None:
            return

        if not self.TTLPhysioCom.rpc_ping():
            self._logger.warning("rt_physio RPC server does not respond.")
            return

        # Get access to scan onset
        scan_onset = self.TTLPhysioCom.call_rt_proc(
            "GET_SCAN_ONSET", get_return=True
        )
        if scan_onset is None:
            self._logger.warning(
                "Failed to get scan onset from rt_physio RPC server."
            )

        self._scan_onset = SharedMemoryRingBuffer(**scan_onset)

        # Get access to ring buffers
        physio_rbufs = self.TTLPhysioCom.call_rt_proc(
            "GET_RBUF", get_return=True
        )
        if physio_rbufs is None:
            self._logger.warning(
                "Failed to get ring buffers from rt_physio RPC server."
            )
            return

        for lab, rb in physio_rbufs.items():
            self._rbuf[lab] = SharedMemoryRingBuffer(**rb)

        # Get sampling frequency
        physio_params = self.TTLPhysioCom.call_rt_proc(
            ("GET_PARAMS", ("sample_freq",)), pkl=True, get_return=True
        )
        self.sample_freq = physio_params.get("sample_freq", None)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # getter, setter
    @property
    def scan_onset(self):
        if self._scan_onset is None:
            self._logger.warning("Scan onset is not initialized.")
            return 0
        else:
            return self._scan_onset.get()[0]

    @property
    def available(self):
        if self.TTLPhysioCom is None:
            self.init_rbuf_access()

        return self.TTLPhysioCom is not None

    @scan_onset.setter
    def scan_onset(self, onset):
        if self._scan_onset is not None:
            self._scan_onset.append(onset)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def quit(self):
        if self.available:
            self.TTLPhysioCom.call_rt_proc("QUIT")
            self.TTLPhysioCom = None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def show(self):
        if self.available:
            self.TTLPhysioCom.call_rt_proc("SHOW")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def hide(self):
        if self.available:
            self.TTLPhysioCom.call_rt_proc("HIDE")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def move(self, geometry):
        if self.available:
            self.TTLPhysioCom.call_rt_proc(
                ("SET_GEOMETRY", geometry),
                pkl=True
            )

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def standby_scan(self):
        self.scan_onset = 0
        if self.available:
            params = {"scan_onset": 0, "wait_ttl_on": True}
            self.TTLPhysioCom.call_rt_proc(
                ("SET_PARAMS", params), pkl=True
            )

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def release_standby_scan(self):
        if self.available:
            self.TTLPhysioCom.call_rt_proc("CANCEL_WAIT_TTL")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def start_scan(self):
        if self.available:
            self.TTLPhysioCom.call_rt_proc("START_SCAN")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end_scan(self):
        if self.available:
            self.TTLPhysioCom.call_rt_proc("END_SCAN")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save_physio_data(self, onset=None, series_duration=None,
                         fname_fmt=None):
        if self.available:
            args = (
                "SAVE_PHYSIO_DATA",
                onset,
                series_duration,
                fname_fmt,
            )
            self.TTLPhysioCom.call_rt_proc(args, pkl=True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def dump(self):
        if not self.available or len(self._rbuf) == 0:
            return None

        try:
            ttl_onsets = self._rbuf["ttl_onsets"].get().copy()
            ttl_offsets = self._rbuf["ttl_offsets"].get().copy()
            tstamp = self._rbuf["tstamp"].get().copy()
            card = self._rbuf["card"].get().copy()
            resp = self._rbuf["resp"].get().copy()

            # Remove nan
            ttl_onsets = ttl_onsets[~np.isnan(ttl_onsets)]
            ttl_offsets = ttl_offsets[~np.isnan(ttl_offsets)]

            tmask = (
                ~np.isnan(tstamp)
                & ~np.isnan(card)
                & ~np.isnan(resp)
                & (tstamp > 0)
            )
            card = card[tmask]
            resp = resp[tmask]
            tstamp = tstamp[tmask]
            if len(tstamp) == 0:
                return None

            # Sort by time stamp
            sidx = np.argsort(tstamp)
            card = card[sidx]
            resp = resp[sidx]
            tstamp = tstamp[sidx]
            ttl_onsets = np.sort(ttl_onsets)
            ttl_offsets = np.sort(ttl_offsets)

            data = {
                "ttl_onsets": ttl_onsets,
                "ttl_offsets": ttl_offsets,
                "card": card,
                "resp": resp,
                "tstamp": tstamp,
            }

            return data

        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            self._logger.error(f"Error dumping ring buffer: {errstr}")
            return None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_retrots(self, tr, n_vol=np.inf, tshift=0, timeout=1.5):
        if not self.available:
            return None

        onset = self.scan_onset
        if onset == 0:
            return None

        data = self.dump()
        if data is None:
            return None

        tstamp = data["tstamp"] - onset

        if np.isinf(n_vol):
            n_vol = int(np.nanmax(tstamp) // tr)
        else:
            timeout_st = time.time()
            while (
                int(np.nanmax(tstamp) // tr) < n_vol
                and time.time() - timeout_st < timeout
            ):
                # Wait until Nvol samples
                time.sleep(0.001)
                data = self.dump()
                tstamp = data["tstamp"] - onset

            if int(np.nanmax(tstamp) // tr) < n_vol:
                # ERROR: timeout
                self._logger.error(
                    "Not received enough data to make RETROICOR regressors"
                    f" for {timeout} s."
                )
                return None

        dataMask = (tstamp >= -tr) & ~np.isnan(tstamp)
        dataMask &= ~np.isnan(data["resp"])
        dataMask &= ~np.isnan(data["card"])
        resp = data["resp"][dataMask]
        card = data["card"][dataMask]
        tstamp = tstamp[dataMask]

        # Resample
        physFS = self.sample_freq
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            res_t = np.arange(0, n_vol * tr + 1.0, 1.0 / physFS)
            resp_res_f = interp1d(tstamp, resp, bounds_error=False)
            Resp = resp_res_f(res_t)
            Resp = Resp[~np.isnan(Resp)]

            card_res_f = interp1d(tstamp, card, bounds_error=False)
            Card = card_res_f(res_t)
            Card = Card[~np.isnan(Card)]

        retroTSReg = self._retrots.RetroTs(
            Resp, Card, tr, physFS, tshift, n_vol)

        return retroTSReg

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        self.quit()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_param(self, attr, val, reset_fn=None, echo=False):
        self._logger.debug(f"set_param: {attr} = {val}")

        pass

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # def ui_set_param(self):
    #     """
    #     When reset_fn is None, set_param is considered to be called from
    #     load_parameters function.
    #     """
    #     return
    #     ui_rows = []
    #     self.ui_objs = []

        # # enabled
        # self.ui_enabled_rdb = QtWidgets.QRadioButton("Enable")
        # self.ui_enabled_rdb.setChecked(self.enabled)
        # self.ui_enabled_rdb.toggled.connect(
        #         lambda checked: self.set_param('enabled', checked,
        #                                        self.ui_enabled_rdb.setChecked))
        # ui_rows.append((self.ui_enabled_rdb, None))

        # # mask_file
        # var_lb = QtWidgets.QLabel("Mask :")
        # self.ui_mask_cmbBx = QtWidgets.QComboBox()
        # self.ui_mask_cmbBx.addItems(['external file',
        #                              'initial volume of internal run'])
        # self.ui_mask_cmbBx.activated.connect(
        #         lambda idx:
        #         self.set_param('mask_file',
        #                        self.ui_mask_cmbBx.currentText(),
        #                        self.ui_mask_cmbBx.setCurrentIndex))

        # self.ui_mask_lnEd = QtWidgets.QLineEdit()
        # self.ui_mask_lnEd.setReadOnly(True)
        # self.ui_mask_lnEd.setStyleSheet(
        #     'border: 0px none;')
        # self.ui_objs.extend([var_lb, self.ui_mask_cmbBx,
        #                      self.ui_mask_lnEd])

        # if self.mask_file == 0:
        #     self.ui_mask_cmbBx.setCurrentIndex(1)
        #     self.ui_mask_lnEd.setText('zero-out initial received volume')
        # else:
        #     self.ui_mask_cmbBx.setCurrentIndex(0)
        #     self.ui_mask_lnEd.setText(str(self.mask_file))

        # mask_hLayout = QtWidgets.QHBoxLayout()
        # mask_hLayout.addWidget(self.ui_mask_cmbBx)
        # mask_hLayout.addWidget(self.ui_mask_lnEd)
        # ui_rows.append((var_lb, mask_hLayout))

        # # wait_num
        # var_lb = QtWidgets.QLabel("Wait REGRESS until (volumes) :")
        # self.ui_waitNum_cmbBx = QtWidgets.QComboBox()
        # self.ui_waitNum_cmbBx.addItems(['number of regressors', 'set value'])
        # self.ui_waitNum_cmbBx.activated.connect(
        #         lambda idx:
        #         self.set_param('wait_num',
        #                        self.ui_waitNum_cmbBx.currentText(),
        #                        self.ui_waitNum_cmbBx.setCurrentIndex))

        # self.ui_waitNum_lb = QtWidgets.QLabel()
        # regNum = self.get_reg_num()
        # self.ui_waitNum_lb.setText(
        #         f'Wait REGRESS until receiving {regNum} volumes')
        # self.ui_objs.extend([var_lb, self.ui_waitNum_cmbBx,
        #                      self.ui_waitNum_lb])

        # wait_num_hLayout = QtWidgets.QHBoxLayout()
        # wait_num_hLayout.addWidget(self.ui_waitNum_cmbBx)
        # wait_num_hLayout.addWidget(self.ui_waitNum_lb)
        # ui_rows.append((var_lb, wait_num_hLayout))

        # # max_scan_length
        # var_lb = QtWidgets.QLabel("Maximum scan length :")
        # self.ui_maxLen_spBx = QtWidgets.QSpinBox()
        # self.ui_maxLen_spBx.setMinimum(1)
        # self.ui_maxLen_spBx.setMaximum(9999)
        # self.ui_maxLen_spBx.setValue(self.max_scan_length)
        # self.ui_maxLen_spBx.editingFinished.connect(
        #         lambda: self.set_param('max_scan_length',
        #                                self.ui_maxLen_spBx.value(),
        #                                self.ui_maxLen_spBx.setValue))
        # ui_rows.append((var_lb, self.ui_maxLen_spBx))
        # self.ui_objs.extend([var_lb, self.ui_maxLen_spBx])

        # # max_poly_order
        # var_lb = QtWidgets.QLabel("Maximum polynomial order :\n"
        #                           "regressors for slow fluctuation")
        # self.ui_maxPoly_cmbBx = QtWidgets.QComboBox()
        # self.ui_maxPoly_cmbBx.addItems(['auto', 'set'])
        # self.ui_maxPoly_cmbBx.activated.connect(
        #         lambda idx:
        #         self.set_param('max_poly_order',
        #                        self.ui_maxPoly_cmbBx.currentText(),
        #                        self.ui_maxPoly_cmbBx.setCurrentIndex))

        # self.ui_maxPoly_lb = QtWidgets.QLabel()
        # self.ui_objs.extend([var_lb, self.ui_maxPoly_cmbBx,
        #                      self.ui_maxPoly_lb])
        # if np.isinf(self.max_poly_order):
        #     self.ui_maxPoly_cmbBx.setCurrentIndex(0)
        #     self.ui_maxPoly_lb.setText('Increase polynomial order ' +
        #                                'with the scan length')
        # else:
        #     self.ui_maxPoly_cmbBx.setCurrentIndex(1)
        #     self.ui_maxPoly_lb.setText('Increase polynomial order ' +
        #                                'with the scan length' +
        #                                f' up to {self.max_poly_order}')

        # maxPoly_hLayout = QtWidgets.QHBoxLayout()
        # maxPoly_hLayout.addWidget(self.ui_maxPoly_cmbBx)
        # maxPoly_hLayout.addWidget(self.ui_maxPoly_lb)
        # ui_rows.append((var_lb, maxPoly_hLayout))

        # # mot_reg
        # var_lb = QtWidgets.QLabel("Motion regressor :")
        # self.ui_motReg_cmbBx = QtWidgets.QComboBox()
        # self.ui_motReg_cmbBx.addItems(
        #         ['None', '6 motions (yaw, pitch, roll, dS, dL, dP)',
        #          '12 motions (6 motions and their temporal derivatives)',
        #          '6 motion derivatives'])
        # ci = {'None': 0, 'mot6': 1, 'mot12': 2, 'dmot6': 3}[self.mot_reg]
        # self.ui_motReg_cmbBx.setCurrentIndex(ci)
        # self.ui_motReg_cmbBx.currentIndexChanged.connect(
        #         lambda idx:
        #         self.set_param('mot_reg',
        #                        self.ui_motReg_cmbBx.currentText(),
        #                        self.ui_motReg_cmbBx.setCurrentIndex))
        # ui_rows.append((var_lb, self.ui_motReg_cmbBx))
        # self.ui_objs.extend([var_lb, self.ui_motReg_cmbBx])

        # # GS ROI regressor
        # self.ui_GS_reg_chb = QtWidgets.QCheckBox("Regress global signal :")
        # self.ui_GS_reg_chb.setChecked(self.GS_reg)
        # self.ui_GS_reg_chb.stateChanged.connect(
        #         lambda state: self.set_param('GS_reg', state > 0))

        # GSmask_hBLayout = QtWidgets.QHBoxLayout()
        # self.ui_GS_mask_lnEd = QtWidgets.QLineEdit()
        # self.ui_GS_mask_lnEd.setText(str(self.GS_mask))
        # self.ui_GS_mask_lnEd.setReadOnly(True)
        # self.ui_GS_mask_lnEd.setStyleSheet(
        #     'border: 0px none;')
        # GSmask_hBLayout.addWidget(self.ui_GS_mask_lnEd)

        # self.ui_GSmask_btn = QtWidgets.QPushButton('Set')
        # self.ui_GSmask_btn.clicked.connect(
        #         lambda: self.set_param(
        #                 'GS_mask',
        #                 Path(self.ui_GS_mask_lnEd.text()).parent,
        #                 self.ui_GS_mask_lnEd.setText))
        # GSmask_hBLayout.addWidget(self.ui_GSmask_btn)

        # self.ui_objs.extend([self.ui_GS_reg_chb, self.ui_GS_mask_lnEd,
        #                      self.ui_GSmask_btn])
        # ui_rows.append((self.ui_GS_reg_chb, GSmask_hBLayout))

        # # WM ROI regressor
        # self.ui_WM_reg_chb = QtWidgets.QCheckBox("Regress WM signal :")
        # self.ui_WM_reg_chb.setChecked(self.WM_reg)
        # self.ui_WM_reg_chb.stateChanged.connect(
        #         lambda state: self.set_param('WM_reg', state > 0))

        # WMmask_hBLayout = QtWidgets.QHBoxLayout()
        # self.ui_WM_mask_lnEd = QtWidgets.QLineEdit()
        # self.ui_WM_mask_lnEd.setText(str(self.WM_mask))
        # self.ui_WM_mask_lnEd.setReadOnly(True)
        # self.ui_WM_mask_lnEd.setStyleSheet(
        #     'border: 0px none;')
        # WMmask_hBLayout.addWidget(self.ui_WM_mask_lnEd)

        # self.ui_WMmask_btn = QtWidgets.QPushButton('Set')
        # self.ui_WMmask_btn.clicked.connect(
        #         lambda: self.set_param(
        #                 'WM_mask',
        #                 Path(self.ui_WM_mask_lnEd.text()).parent,
        #                 self.ui_WM_mask_lnEd.setText))
        # WMmask_hBLayout.addWidget(self.ui_WMmask_btn)

        # self.ui_objs.extend([self.ui_WM_reg_chb, self.ui_WM_mask_lnEd,
        #                      self.ui_WMmask_btn])
        # ui_rows.append((self.ui_WM_reg_chb, WMmask_hBLayout))

        # # Vent ROI regressor
        # self.ui_Vent_reg_chb = QtWidgets.QCheckBox("Regress Vent signal :")
        # self.ui_Vent_reg_chb.setChecked(self.Vent_reg)
        # self.ui_Vent_reg_chb.stateChanged.connect(
        #         lambda state: self.set_param('Vent_reg', state > 0))

        # Ventmask_hBLayout = QtWidgets.QHBoxLayout()

        # self.ui_Vent_mask_lnEd = QtWidgets.QLineEdit()
        # self.ui_Vent_mask_lnEd.setText(str(self.Vent_mask))
        # self.ui_Vent_mask_lnEd.setReadOnly(True)
        # self.ui_Vent_mask_lnEd.setStyleSheet(
        #     'border: 0px none;')
        # Ventmask_hBLayout.addWidget(self.ui_Vent_mask_lnEd)

        # self.ui_Ventmask_btn = QtWidgets.QPushButton('Set')
        # self.ui_Ventmask_btn.clicked.connect(
        #         lambda: self.set_param(
        #                 'Vent_mask',
        #                 Path(self.ui_Vent_mask_lnEd.text()).parent,
        #                 self.ui_Vent_mask_lnEd.setText))
        # Ventmask_hBLayout.addWidget(self.ui_Ventmask_btn)

        # self.ui_objs.extend([self.ui_Vent_reg_chb, self.ui_Vent_mask_lnEd,
        #                      self.ui_Ventmask_btn])
        # ui_rows.append((self.ui_Vent_reg_chb, Ventmask_hBLayout))

        # # phys_reg
        # var_lb = QtWidgets.QLabel("RICOR regressor :")
        # self.ui_physReg_cmbBx = QtWidgets.QComboBox()
        # self.ui_physReg_cmbBx.addItems(
        #         ['None', '8 RICOR (4 Resp and 4 Card)']
        #         )
        # ci = {'None': 0, 'RICOR8': 1, 'RVT5': 2,
        #       'RVT+RICOR13': 3}[self.phys_reg]
        # self.ui_physReg_cmbBx.setCurrentIndex(ci)
        # self.ui_physReg_cmbBx.currentIndexChanged.connect(
        #         lambda idx:
        #         self.set_param('phys_reg',
        #                        self.ui_physReg_cmbBx.currentText(),
        #                        self.ui_physReg_cmbBx.setCurrentIndex))
        # ui_rows.append((var_lb, self.ui_physReg_cmbBx))
        # self.ui_objs.extend([var_lb, self.ui_physReg_cmbBx])

        # # desMtx
        # var_lb = QtWidgets.QLabel("Design matrix :")

        # desMtx_hBLayout = QtWidgets.QHBoxLayout()
        # self.ui_loadDesMtx_btn = QtWidgets.QPushButton('Set')
        # self.ui_loadDesMtx_btn.clicked.connect(
        #         lambda: self.set_param('desMtx_f', 'set'))
        # desMtx_hBLayout.addWidget(self.ui_loadDesMtx_btn)

        # self.ui_showDesMtx_btn = QtWidgets.QPushButton()
        # self.ui_showDesMtx_btn.clicked.connect(
        #         lambda: self.set_param('showDesMtx'))
        # desMtx_hBLayout.addWidget(self.ui_showDesMtx_btn)

        # self.ui_objs.extend([var_lb, self.ui_loadDesMtx_btn,
        #                      self.ui_showDesMtx_btn])
        # ui_rows.append((var_lb, desMtx_hBLayout))
        # self.ui_showDesMtx_btn.setText('Show desing matrix')
        # if self.desMtx_read is None:
        #     self.ui_showDesMtx_btn.setEnabled(False)
        # else:
        #     self.ui_showDesMtx_btn.setEnabled(True)

        # # --- Checkbox row ------------------------------------------------
        # # Restrocpective process
        # self.ui_retroProc_chb = QtWidgets.QCheckBox("Retrospective process")
        # self.ui_retroProc_chb.setChecked(self.reg_retro_proc)
        # self.ui_retroProc_chb.stateChanged.connect(
        #         lambda state: setattr(self, 'reg_retro_proc', state > 0))
        # self.ui_objs.append(self.ui_retroProc_chb)

        # # Save
        # self.ui_saveProc_chb = QtWidgets.QCheckBox("Save processed image")
        # self.ui_saveProc_chb.setChecked(self.save_proc)
        # self.ui_saveProc_chb.stateChanged.connect(
        #         lambda state: setattr(self, 'save_proc', state > 0))
        # self.ui_objs.append(self.ui_saveProc_chb)

        # chb_hLayout = QtWidgets.QHBoxLayout()
        # chb_hLayout.addStretch()
        # chb_hLayout.addWidget(self.ui_saveProc_chb)
        # ui_rows.append((self.ui_retroProc_chb, chb_hLayout))

    #     return ui_rows

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # def get_params(self):
        all_opts = super().get_params()
        incld_opts = ("device", "sample_freq", "buf_len_sec", "save_ttl")
        sel_opts = {}
        for k, v in all_opts.items():
            if k not in incld_opts:
                continue
            if isinstance(v, Path):
                v = str(v)
            sel_opts[k] = v

        return sel_opts
