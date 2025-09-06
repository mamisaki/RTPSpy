#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LA-NF application with RtpApp

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
from datetime import timedelta
import time
import sys

import numpy as np
import nibabel as nib
from PyQt5 import QtWidgets, QtCore

from rtpspy.rtp_app import RtpApp


# %% LANF class =============================================================
class LANF(RtpApp):
    """
    Left amygdala (LA) ROI signal neurofeedback.
    The signal is sent to an external neurofedback presentation application.
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, default_rtp_params=None, **kwargs):

        super(LANF, self).__init__(**kwargs)

        # Task parameters
        self.NF_target_levels = {'Practice': 0.2, 'NF1': 0.4, 'NF2': 0.8,
                                 'NF3': 1.0}
        self.sham = False

        # Task (block) durations (seconds)
        self.initDur = 96
        self.NFBlockDur = 40
        self.CountBlockDur = 40
        self.RestBlockDur = 40
        self.NrBlockRep = 4
        self.restDur = 480
        self.update_total_dur()

        # Set default parameters
        self.default_rtp_params = default_rtp_params
        self.set_default_params()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_default_params(self):
        if self.default_rtp_params is not None:
            # Set default parameters
            for proc, params in self.default_rtp_params.items():
                if proc in self.rtp_objs:
                    for attr, val in params.items():
                        self.rtp_objs[proc].set_param(attr, val)

            if 'APP' in self.default_rtp_params:
                for attr, val in self.default_rtp_params['APP'].items():
                    self.set_param(attr, val)

        if hasattr(self, 'ui_showROISig_cbx'):
            # Disable ROI signal plot
            self.ui_showROISig_cbx.setCheckState(0)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def setup_extApp(self, session=None, no_RTP=None):
        """
        Automatic setup of the external appllication and calling RTP_setup()
        """

        # --- Check extApp is alive -------------------------------------------
        if not self.isAlive_extApp():
            # Boot external application
            self.boot_extApp(timeout=30)
            if not self.isAlive_extApp():
                # Failed to boot App
                return -1

        # --- Initialize extApp -----------------------------------------------
        # Make the extApp state 'cmd_loop'
        if not self.send_extApp('GET_STATE;'.encode('utf-8')):
            # Failed in sending the command
            return -1

        state = self.recv_extApp(timeout=5)
        if state is None:
            self.errmsg('No response to GET_STATE from the external app.')
            return -1

        if state.decode() != 'cmd_loop':
            # Send END to exit to the 'cmd_loop' state
            if not self.send_extApp('END;'.encode('utf-8')):
                # Failed in sending the command
                return -1

            ret = self.recv_extApp(timeout=5)
            if ret is None:
                self.errmsg('No response to QUIT from the external app.')
                return -1

        # Send session log directory
        if not self.send_extApp(
                f'SET_LOGDIR {self.work_dir/"log"};'.encode('utf-8')):
            return -1

        # --- Send session preparation command to extApp ----------------------
        if session == 'Rest':
            prep_cmd = 'PREP_Rest;'
            task_param = {'session': session, 'total_duration': self.restDur}
        else:
            prep_cmd = 'PREP_Task;'
            timings = [self.initDur, self.NFBlockDur, self.CountBlockDur,
                       self.RestBlockDur, self.NrBlockRep]
            task_param = {'session': session, 'TR': self.rtp_objs['TSHIFT'].TR,
                          'timings': timings}
            if session in ('Baseline', 'Transfer'):
                task_param['nofb'] = True
            else:
                task_param['nofb'] = False
                task_param['target_level'] = self.NF_target_levels[session]

        if not self.send_extApp(prep_cmd.encode('utf-8')):
            # Failed in sending the command
            return -1

        if not self.send_extApp(task_param, pkl=True):
            # Failed in sending the command
            return -1

        # Check response from extApp
        recv = self.recv_extApp(timeout=3)
        if recv is None:
            self.errmsg('No response from the external app.')
            return -1
        elif recv.decode() != 'PREPED;':
            self.errmsg(f'External app returned {recv.decode()}')
            return -1

        self.running_session = session

        # --- RTP setup -------------------------------------------------------
        if no_RTP is None:
            if 'Rest' in session or 'Think' in session or 'View' in session:
                no_RTP = True
            else:
                no_RTP = False

        ret = self.RTP_setup(rtp_params={'enable_RTP': not no_RTP})
        if ret == -1:
            return

        # --- Disable extApp ui ---
        for uiobj in ('ui_extApp_cmd_lnEd', 'ui_extApp_run_btn',
                      'ui_extApp_addr_lnEd'):
            if hasattr(self, uiobj):
                getattr(self, uiobj).setEnabled(False)

        # --- Disable ui_taskTiming_grpBx and ui_sessSetup_grpBx ---
        if hasattr(self, 'ui_taskTiming_grpBx'):
            self.ui_taskTiming_grpBx.setEnabled(False)

        if hasattr(self, 'ui_sessSetup_grpBx'):
            self.ui_sessSetup_grpBx.setEnabled(False)

        return 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ready_to_run(self):
        super().ready_to_run()

        # Send READY message to extApp
        if self.send_extApp('READY;'.encode()):
            recv = self.recv_extApp(timeout=3)
            if recv is not None:
                if self._verb:
                    self.logmsg(f"Recv {recv.decode()}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def do_proc(self, fmri_img, vol_idx=None, pre_proc_time=None):
        """
        Extract left amygdala mean signal and send it to the external
        application.
        """

        try:
            # Increment the number of received volume
            self._vol_num += 1  # 1- base number of volumes recieved by this
            if vol_idx is None:
                vol_idx = self._vol_num - 1  # 0-base index

            if vol_idx < self.ignore_init:
                # Skip ignore_init volumes
                return

            if self._proc_start_idx < 0:
                self._proc_start_idx = vol_idx

            dataV = fmri_img.get_fdata()

            # --- Initialize --------------------------------------------------
            if Path(self.ROI_orig).is_file() and self.ROI_mask is None:
                # Load ROI mask
                self.ROI_mask = np.asanarry(nib.load(self.ROI_orig).dataobj)

            # --- Run the procress --------------------------------------------
            if not self.sham:
                roi_id = 1
            else:
                roi_id = 2

            # Get mean signal in the ROI
            roimask = (self.ROI_mask == roi_id) & (np.abs(dataV) > 0.0)
            mean_sig = np.nanmean(dataV[roimask])

            if self.extApp_sock is None:
                # Error: Socket is not opened.
                self.errmsg('No socket to an external app.', no_pop=True)

            else:
                # Send data to the external application via socket,
                # self.extApp_sock
                try:
                    scan_onset = self.rtp_objs['EXTSIG'].scan_onset
                    val_str = f"{time.time()-scan_onset:.4f},"
                    val_str += f"{vol_idx},{mean_sig:.6f}"
                    msg = f"NF {val_str};"

                    self.send_extApp(msg.encode(), no_err_pop=True)
                    if self._verb:
                        self.logmsg(f"Sent '{msg}' to an external app")

                except Exception as e:
                    self.errmsg(str(e), no_pop=True)

            # --- Post procress -----------------------------------------------
            # Record process time
            tstamp = time.time()
            self._proc_time.append(tstamp)
            if pre_proc_time is not None:
                proc_delay = self._proc_time[-1] - pre_proc_time
                if self.save_delay:
                    self.proc_delay.append(proc_delay)

            # log message
            if self._verb:
                if fmri_img.get_filename():
                    fname = Path(fmri_img.get_filename()).name
                else:
                    fname = "unknown.nii.gz"
                msg = f"#{vol_idx+1};ROI signal extraction;{fname}"
                msg += f";tstamp={tstamp}"
                if pre_proc_time is not None:
                    msg += f";took {proc_delay:.4f}s"
                self.logmsg(msg)

            # Update signal plot
            self._plt_xi.append(vol_idx+1)
            self._roi_sig[0].append(mean_sig)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errmsg = '{}, {}:{}'.format(
                    exc_type, exc_tb.tb_frame.f_code.co_filename,
                    exc_tb.tb_lineno)
            self.errmsg(str(e) + '\n' + errmsg, no_pop=True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end_proc(self):
        # --- Enable ui_taskTiming_grpBx and ui_sessSetup_grpBx ---
        if hasattr(self, 'ui_taskTiming_grpBx'):
            self.ui_taskTiming_grpBx.setEnabled(True)

        if hasattr(self, 'ui_sessSetup_grpBx'):
            self.ui_sessSetup_grpBx.setEnabled(True)

    # -------------------------------------------------------------------------
    # Custom utility mrthods
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def update_total_dur(self):
        existAttr = [hasattr(self, attr) for attr in
                     ('initDur', 'NFBlockDur', 'CountBlockDur', 'RestBlockDur',
                      'NrBlockRep')]
        if not np.all(existAttr):
            self.totalTaskDur = np.nan
            return

        self.totalTaskDur = self.initDur + self.NrBlockRep * \
            (self.NFBlockDur + self.CountBlockDur + self.RestBlockDur)
        durStr = f"{str(timedelta(seconds=self.totalTaskDur))}"

        if hasattr(self, 'ui_total_task_dur'):
            self.ui_total_task_dur.setText(durStr)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def show_ui_shamChb(self, event):
        if self.ui_shamFb_chb.isVisible():
            self.ui_shamFb_chb.setVisible(False)
        else:
            modifiers = QtWidgets.QApplication.keyboardModifiers()
            if modifiers == (QtCore.Qt.ControlModifier |
                             QtCore.Qt.ShiftModifier):
                self.ui_shamFb_chb.setVisible(True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_param(self, attr, val=None, reset_fn=None, echo=False):
        """
        When reset_fn is None, set_param is considered called from
        load_parameters function.
        """

        super().set_param(attr, val, reset_fn, echo)

        self._logger.debug(f"set_param: {attr} = {val}")

        if attr == 'ROI_orig':
            if Path(self.ROI_orig).is_file():
                self.ui_NFROI_lnEd.setText(str(self.ROI_orig))
            else:
                self.ui_NFROI_lnEd.setText("N/A")

        elif attr in ('initDur', 'NFBlockDur', 'CountBlockDur', 'RestBlockDur',
                      'NrBlockRep', 'restDur', 'NF_target_levels'):
            setattr(self, attr, val)
            self.update_total_dur()
            if attr == 'initDur':
                # Set REGRESS wait_num
                self.rtp_objs['REGRESS'].set_wait_num((self.initDur-6)//2-5)

            if hasattr(self, f'ui_{attr}_spBx'):
                getattr(self, f'ui_{attr}_spBx').setValue(val)

        else:
            # Ignore others: Those should be handled insuper().set_param.
            return

        setattr(self, attr, val)
        if echo:
            print("{}.".format(self.__class__.__name__) + attr, '=',
                  getattr(self, attr))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ui_set_param(self):
        ui_rows = super().ui_set_param()

        # --- Clear/Reset buttons ---------------------------------------------
        cleare_reset_hLayout = QtWidgets.QHBoxLayout()
        cleare_reset_hLayout.addStretch()

        self.ui_restParam_btn = QtWidgets.QPushButton(
            'Reset to default parameters')
        self.ui_restParam_btn.clicked.connect(self.set_default_params)
        cleare_reset_hLayout.addWidget(self.ui_restParam_btn)
        self.ui_objs.append(self.ui_restParam_btn)

        # Put at the top right
        ui_rows[0] = (self.ui_enableRTP_chb, cleare_reset_hLayout)

        # --- Task operation tab ----------------------------------------------
        self.ui_taskTab = QtWidgets.QWidget()
        self.ui_top_tabs.insertTab(0, self.ui_taskTab, 'Task')
        self.ui_top_tabs.setCurrentWidget(self.ui_taskTab)
        op_fLayout = QtWidgets.QFormLayout(self.ui_taskTab)

        # --- Neurofeedback ROI group ---
        ui_ROIdef_grpBx = QtWidgets.QGroupBox("Neurofeedback ROI")
        op_fLayout.addWidget(ui_ROIdef_grpBx)
        ROIdef_gLayout = QtWidgets.QGridLayout(ui_ROIdef_grpBx)
        self.ui_objs.append(ui_ROIdef_grpBx)

        row_i = 0

        # ROI make button
        self.ui_makeROI_btn = QtWidgets.QPushButton('Go to parameter setting')
        self.ui_makeROI_btn.clicked.connect(
                lambda:
                self.ui_top_tabs.setCurrentWidget(self.ui_preprocessingTab))
        self.ui_makeROI_btn.setStyleSheet(
            "background-color: rgb(151,217,235);")
        ROIdef_gLayout.addWidget(self.ui_makeROI_btn, row_i, 0)

        # ROI file
        ui_NFROIh_lb = QtWidgets.QLabel("NF ROI mask file :")
        self.ui_NFROI_lnEd = QtWidgets.QLineEdit()
        self.ui_NFROI_lnEd.setReadOnly(True)
        self.ui_NFROI_lnEd.setStyleSheet(
            'border: 0px none;')
        if Path(self.ROI_orig).is_file():
            self.ui_NFROI_lnEd.setText(str(self.ROI_orig))

        ROIdef_gLayout.addWidget(ui_NFROIh_lb, row_i, 1)
        ROIdef_gLayout.addWidget(self.ui_NFROI_lnEd, row_i, 2, 1, 4)

        # Add 'Go to Task' button in the 'Preprocessing' tab
        self.ui_gotoTask_btn = QtWidgets.QPushButton('Go to Task')
        self.ui_gotoTask_btn.clicked.connect(
                lambda: self.ui_top_tabs.setCurrentWidget(self.ui_taskTab))
        self.ui_gotoTask_btn.setStyleSheet(
            "background-color: rgb(151,217,235);")
        self.ui_preprocessing_fLayout.addRow(self.ui_gotoTask_btn)
        self.ui_objs.append(self.ui_gotoTask_btn)

        # --- Task timing group ---
        self.ui_taskTiming_grpBx = QtWidgets.QGroupBox("Task timings")
        op_fLayout.addWidget(self.ui_taskTiming_grpBx)
        taskTiming_gLayout = QtWidgets.QGridLayout(self.ui_taskTiming_grpBx)
        self.ui_objs.append(self.ui_taskTiming_grpBx)
        row_i = 0

        # --- Label row ---
        var_lb = QtWidgets.QLabel("Burn-in")
        var_lb.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignBottom)
        taskTiming_gLayout.addWidget(var_lb, row_i, 0)

        var_lb = QtWidgets.QLabel("+ ( NF block")
        var_lb.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignBottom)
        taskTiming_gLayout.addWidget(var_lb, row_i, 1)

        var_lb = QtWidgets.QLabel("+ Count block")
        var_lb.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignBottom)
        taskTiming_gLayout.addWidget(var_lb, row_i, 2)

        var_lb = QtWidgets.QLabel("+ Rest block )")
        var_lb.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignBottom)
        taskTiming_gLayout.addWidget(var_lb, row_i, 3)

        var_lb = QtWidgets.QLabel("x Repeat blocks")
        var_lb.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignBottom)
        taskTiming_gLayout.addWidget(var_lb, row_i, 4)

        row_i += 1

        # --- Initial idling duration ---
        self.ui_initDur_spBx = QtWidgets.QSpinBox()
        self.ui_initDur_spBx.setMinimum(0)
        self.ui_initDur_spBx.setMaximum(1000)
        self.ui_initDur_spBx.setSuffix(' seconds')
        self.ui_initDur_spBx.setValue(self.initDur)
        self.ui_initDur_spBx.valueChanged.connect(
                lambda x: self.set_param('initDur', x,
                                         self.ui_initDur_spBx.setValue))
        taskTiming_gLayout.addWidget(self.ui_initDur_spBx, row_i, 0)

        # --- NF block duration ---
        self.ui_NFBlockDur_spBx = QtWidgets.QSpinBox()
        self.ui_NFBlockDur_spBx.setMinimum(0)
        self.ui_NFBlockDur_spBx.setMaximum(1000)
        self.ui_NFBlockDur_spBx.setSuffix(' seconds')
        self.ui_NFBlockDur_spBx.setValue(self.NFBlockDur)
        self.ui_NFBlockDur_spBx.valueChanged.connect(
                lambda x: self.set_param('NFBlockDur', x,
                                         self.ui_NFBlockDur_spBx.setValue))
        taskTiming_gLayout.addWidget(self.ui_NFBlockDur_spBx, row_i, 1)

        # --- Count block duration ---
        self.ui_CountBlockDur_spBx = QtWidgets.QSpinBox()
        self.ui_CountBlockDur_spBx.setMinimum(0)
        self.ui_CountBlockDur_spBx.setMaximum(1000)
        self.ui_CountBlockDur_spBx.setSuffix(' seconds')
        self.ui_CountBlockDur_spBx.setValue(self.CountBlockDur)
        self.ui_CountBlockDur_spBx.valueChanged.connect(
                lambda x: self.set_param('CountBlockDur', x,
                                         self.ui_CountBlockDur_spBx.setValue))
        taskTiming_gLayout.addWidget(self.ui_CountBlockDur_spBx, row_i, 2)

        # --- Rest block duration ---
        self.ui_RestBlockDur_spBx = QtWidgets.QSpinBox()
        self.ui_RestBlockDur_spBx.setMinimum(0)
        self.ui_RestBlockDur_spBx.setMaximum(1000)
        self.ui_RestBlockDur_spBx.setSuffix(' seconds')
        self.ui_RestBlockDur_spBx.setValue(self.RestBlockDur)
        self.ui_RestBlockDur_spBx.valueChanged.connect(
                lambda x: self.set_param('RestBlockDur', x,
                                         self.ui_RestBlockDur_spBx.setValue))
        taskTiming_gLayout.addWidget(self.ui_RestBlockDur_spBx, row_i, 3)

        # --- Repeat blocks ---
        self.ui_NrBlockRep_spBx = QtWidgets.QSpinBox()
        self.ui_NrBlockRep_spBx.setMinimum(0)
        self.ui_NrBlockRep_spBx.setMaximum(100)
        self.ui_NrBlockRep_spBx.setPrefix('x ')
        self.ui_NrBlockRep_spBx.setValue(self.NrBlockRep)
        self.ui_NrBlockRep_spBx.valueChanged.connect(
                lambda x: self.set_param('NrBlockRep', x,
                                         self.ui_NrBlockRep_spBx.setValue))
        taskTiming_gLayout.addWidget(self.ui_NrBlockRep_spBx, row_i, 4)
        row_i += 1

        # Disable timing edits
        '''
        self.ui_initDur_spBx.setReadOnly(True)
        self.ui_NFBlockDur_spBx.setReadOnly(True)
        self.ui_CountBlockDur_spBx.setReadOnly(True)
        self.ui_RestBlockDur_spBx.setReadOnly(True)
        self.ui_NrBlockRep_spBx.setReadOnly(True)
        '''

        # --- Total duration ---
        var_lb = QtWidgets.QLabel(" Total task duration :")
        var_lb.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        taskTiming_gLayout.addWidget(var_lb, row_i, 0)

        durStr = f"{str(timedelta(seconds=self.totalTaskDur))}"
        self.ui_total_task_dur = QtWidgets.QLabel(durStr)
        self.ui_total_task_dur.setAlignment(QtCore.Qt.AlignLeft |
                                            QtCore.Qt.AlignVCenter)
        taskTiming_gLayout.addWidget(self.ui_total_task_dur, row_i, 1)

        row_i += 1

        # --- Session setup group ---
        self.ui_sessSetup_grpBx = QtWidgets.QGroupBox("Session setup")
        op_fLayout.addWidget(self.ui_sessSetup_grpBx)
        taskRun_vLayout = QtWidgets.QVBoxLayout(self.ui_sessSetup_grpBx)
        self.ui_objs.append(self.ui_sessSetup_grpBx)

        # --- Rest ---
        rest_row_hLayout = QtWidgets.QHBoxLayout()
        taskRun_vLayout.addLayout(rest_row_hLayout)
        self.ui_RestRun_btn = QtWidgets.QPushButton('Rest')
        self.ui_RestRun_btn.setStyleSheet(
            "background-color: rgb(151,217,235); min-width: 90px;")
        self.ui_RestRun_btn.clicked.connect(
            lambda: self.setup_extApp(session='Rest', no_RTP=True))
        rest_row_hLayout.addWidget(self.ui_RestRun_btn)

        # Rest duration
        var_lb = QtWidgets.QLabel("Rest duration :")
        var_lb.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        rest_row_hLayout.addWidget(var_lb)

        self.ui_restDur_spBx = QtWidgets.QSpinBox()
        self.ui_restDur_spBx.setMinimum(0)
        self.ui_restDur_spBx.setMaximum(10000)
        self.ui_restDur_spBx.setSuffix(' seconds')
        self.ui_restDur_spBx.setValue(self.restDur)
        self.ui_restDur_spBx.valueChanged.connect(
            lambda x: self.set_param('restDur', x,
                                     self.ui_restDur_spBx.setValue))
        rest_row_hLayout.addWidget(self.ui_restDur_spBx)
        rest_row_hLayout.addStretch()

        # --- NF ---
        task_row_hLayout = QtWidgets.QHBoxLayout()
        taskRun_vLayout.addLayout(task_row_hLayout)

        # Baseline
        self.ui_PracticeRun_btn = QtWidgets.QPushButton('Baseline')
        self.ui_PracticeRun_btn.setStyleSheet(
            "background-color: rgb(151,217,235);")
        self.ui_PracticeRun_btn.clicked.connect(
            lambda: self.setup_extApp(session='Baseline', no_RTP=True))
        task_row_hLayout.addWidget(self.ui_PracticeRun_btn)

        # Paractice
        self.ui_PracticeRun_btn = QtWidgets.QPushButton('Practice')
        self.ui_PracticeRun_btn.setStyleSheet(
            "background-color: rgb(151,217,235);")
        self.ui_PracticeRun_btn.clicked.connect(
            lambda: self.setup_extApp(session='Practice', no_RTP=False))
        task_row_hLayout.addWidget(self.ui_PracticeRun_btn)

        # NF1
        self.ui_PracticeRun_btn = QtWidgets.QPushButton('NF1')
        self.ui_PracticeRun_btn.setStyleSheet(
            "background-color: rgb(151,217,235);")
        self.ui_PracticeRun_btn.clicked.connect(
            lambda: self.setup_extApp(session='NF1', no_RTP=False))
        task_row_hLayout.addWidget(self.ui_PracticeRun_btn)

        # NF2
        self.ui_PracticeRun_btn = QtWidgets.QPushButton('NF2')
        self.ui_PracticeRun_btn.setStyleSheet(
            "background-color: rgb(151,217,235);")
        self.ui_PracticeRun_btn.clicked.connect(
            lambda: self.setup_extApp(session='NF2', no_RTP=False))
        task_row_hLayout.addWidget(self.ui_PracticeRun_btn)

        # NF3
        self.ui_PracticeRun_btn = QtWidgets.QPushButton('NF3')
        self.ui_PracticeRun_btn.setStyleSheet(
            "background-color: rgb(151,217,235);")
        self.ui_PracticeRun_btn.clicked.connect(
            lambda: self.setup_extApp(session='NF3', no_RTP=False))
        task_row_hLayout.addWidget(self.ui_PracticeRun_btn)

        # Transfer
        self.ui_PracticeRun_btn = QtWidgets.QPushButton('Transfer')
        self.ui_PracticeRun_btn.setStyleSheet(
            "background-color: rgb(151,217,235);")
        self.ui_PracticeRun_btn.clicked.connect(
            lambda: self.setup_extApp(session='Transfer', no_RTP=True))
        task_row_hLayout.addWidget(self.ui_PracticeRun_btn)

        # --- Sham tab ----------------------------------------------------
        shamTab = QtWidgets.QWidget()
        self.ui_top_tabs.addTab(shamTab, 'Sham')
        sham_hLayout = QtWidgets.QHBoxLayout(shamTab)

        # -- label to access hidden options --
        var_lb = QtWidgets.QLabel("Sham?")
        var_lb.mouseReleaseEvent = self.show_ui_shamChb
        sham_hLayout.addWidget(var_lb)

        # -- Sham check --
        self.ui_shamFb_chb = QtWidgets.QCheckBox("Sham feedback")
        self.ui_shamFb_chb.setChecked(self.sham*2)
        self.ui_shamFb_chb.stateChanged.connect(
                lambda state: self.set_param('sham', state > 0))
        self.ui_shamFb_chb.setVisible(False)
        sham_hLayout.addWidget(self.ui_shamFb_chb)
        self.ui_objs.append(self.ui_shamFb_chb)

        sham_hLayout.addStretch()

        # --- Setup button ----------------------------------------------------
        self.ui_setRTP_btn.setVisible(False)

        return ui_rows

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_params(self):
        opts = super().get_params()

        excld_opts = ('default_rtp_params',)

        for k in excld_opts:
            if k in opts:
                del opts[k]

        return opts
