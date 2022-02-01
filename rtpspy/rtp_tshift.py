#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTP online slice-timeing correction

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import sys
import time
import copy
import traceback
import warnings

import pydicom
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from nibabel.nicom.dicomreaders import mosaic_to_nii

import numpy as np
import nibabel as nib
from six import string_types
from PyQt5 import QtWidgets

try:
    from .rtp_common import RTP
except Exception:
    from rtpspy.rtp_common import RTP


# %% class RTP_TSHIFT =========================================================
class RTP_TSHIFT(RTP):
    """ Online slice-timing correction """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, ignore_init=3, method='cubic', ref_time=0, TR=2.0,
                 slice_timing=[], slice_dim=2, **kwargs):
        """
        TR and slice_timing can be set from a sample fMRI data file using the
        'slice_timing_from_sample' method

        Parameters
        ----------
        ignore_init : int, optional
            Number of ignoring volumes before staring a process.
            The default is 3.
        method : str ['linear'|'cubic'], optional
            Temporal interpolation method for the correction.
            The default is 'cubic'.
        ref_time : float, optional
            Reference slice time. The default is 0.
        TR : float, optional
            Ccan interval (second). The default is 2.0.
        slice_timing : array like, optional
            Timing of each slices. The default is [].
        slice_dim : int, optional
            silce dimension
                0: x axis (sagital slice)
                1: y axis (coronal slice)
                2: z axis (axial slice)
            The default is 2.

        """
        super(RTP_TSHIFT, self).__init__(**kwargs)

        # Set instance parameters
        self.ignore_init = ignore_init
        self.method = method
        self.ref_time = ref_time

        self.TR = TR
        self.slice_timing = slice_timing
        self.slice_dim = slice_dim

        # Init parameters
        self.prep_weight = False  # Flag if precalculation is done.
        self.pre_data = []  # previous data volumes

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ready_proc(self):
        if self.slice_timing is None or self.TR is None:
            errmsg = "Slice timing is not set. "
            self.errmsg(errmsg)
            self._proc_ready = False
            return

        self._proc_ready = True  # ready in any case
        if self.next_proc:
            self._proc_ready &= self.next_proc.ready_proc()

        return self._proc_ready

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def do_proc(self, fmri_img, vol_idx=None, pre_proc_time=None, **kwargs):
        try:
            # Increment the number of received volume: vol_num is 0-based
            self.vol_num += 1
            if vol_idx is None:
                vol_idx = self.vol_num

            if vol_idx < self.ignore_init:
                # Skip ignore_init volumes
                return

            if self.proc_start_idx < 0:
                self.proc_start_idx = vol_idx

            # --- Initialize --------------------------------------------------
            if fmri_img.shape[self.slice_dim] != len(self.slice_timing):
                self.errmsg('Slice timing array mismathces to data.',
                            no_pop=True)
                return

            # Set the interpolation weights.
            if not self.prep_weight:
                self._pre_culc_weight(fmri_img.shape[:3])

            dataV = fmri_img.get_fdata()
            if dataV.ndim > 3:
                dataV = np.squeeze(dataV)

            # If there is no previous data, return.
            if len(self.pre_data) == 0:
                self.pre_fmri_img = copy.deepcopy(fmri_img)
                self.pre_data.append(dataV)
                return

            # --- Run the procress --------------------------------------------
            # Retrospective correction
            if hasattr(self, 'pre_fmri_img'):
                #
                if self.method == 'linear':
                    retro_shft_dataV = \
                        self.r1wm1 * self.pre_data[-1] + self.r1w0 * dataV

                elif self.method == 'cubic':
                    if len(self.pre_data) == 1:
                        retro_shft_dataV = \
                            self.r1wm2 * self.pre_data[-1] + \
                            self.r1wm1 * self.pre_data[-1] + \
                            self.r1w0 * dataV + \
                            self.r1wp1 * dataV
                    elif len(self.pre_data) == 2:
                        retro_shft_dataV = \
                            self.r1wm2 * self.pre_data[-2] + \
                            self.r1wm1 * self.pre_data[-1] + \
                            self.r1w0 * dataV + \
                            self.r1wp1 * dataV

                self.pre_fmri_img.uncache()
                self.pre_fmri_img._dataobj = retro_shft_dataV
                self.pre_fmri_img.set_data_dtype = retro_shft_dataV.dtype

                # log message
                if self._verb:
                    fname = Path(self.pre_fmri_img.get_filename()).name
                    fname = fname.replace('.nii.gz', '')
                    msg = f"#{vol_idx-1}, "
                    msg += "Retrospective slice-timing correction"
                    msg += f" is done for {fname}."
                    self.logmsg(msg)

                # Set save_name
                self.pre_fmri_img.set_filename(
                    'ts.' + Path(self.pre_fmri_img.get_filename()).name)

                # Save processed image
                if self.save_proc:
                    self.keep_processed_image(self.pre_fmri_img,
                                              save_temp=self.online_saving,
                                              vol_num=self.vol_num-1)

                if self.next_proc:
                    # Keep the current processed data
                    self.proc_data = np.asanyarray(self.pre_fmri_img.dataobj)
                    # Run the next process
                    self.next_proc.do_proc(self.pre_fmri_img, vol_idx-1,
                                           time.time())

                if self.method == 'cubic':
                    # Two previous volumes are required for cubic intepolation.
                    if len(self.pre_data) < 2:
                        self.pre_data.append(dataV)
                        self.pre_fmri_img = copy.deepcopy(fmri_img)
                        return

                del self.pre_fmri_img

            # Slice timing correction for the current data
            if self.method == 'linear':
                shift_dataV = self.wm1 * self.pre_data[-1] + self.w0 * dataV
            elif self.method == 'cubic':
                shift_dataV = \
                    self.wm2 * self.pre_data[-2] + \
                    self.wm1 * self.pre_data[-1] + \
                    self.w0 * dataV + \
                    self.wp1 * dataV

            # set corrected data in fmri_img
            fmri_img.uncache()
            fmri_img._dataobj = shift_dataV
            fmri_img.set_data_dtype = shift_dataV.dtype

            # update pre_data list
            self.pre_data.append(dataV)
            self.pre_data.pop(0)

            # --- Post procress -----------------------------------------------
            # Record process time
            self.proc_time.append(time.time())
            if pre_proc_time is not None:
                proc_delay = self.proc_time[-1] - pre_proc_time
                if self.save_delay:
                    self.proc_delay.append(proc_delay)

            # log message
            if self._verb:
                f = Path(fmri_img.get_filename()).name
                msg = f'#{vol_idx}, Slice-timing correction is done for {f}'
                if pre_proc_time is not None:
                    msg += f' (took {proc_delay:.4f}s)'
                msg += '.'
                self.logmsg(msg)

            # Set filename
            fmri_img.set_filename('ts.' + Path(fmri_img.get_filename()).name)

            if self.next_proc:
                # Keep the current processed data
                self.proc_data = np.asanyarray(fmri_img.dataobj)
                save_name = fmri_img.get_filename()

                # Run the next process
                self.next_proc.do_proc(fmri_img, vol_idx=vol_idx,
                                       pre_proc_time=self.proc_time[-1])

            # Save processed image
            if self.save_proc:
                if self.next_proc is not None:
                    # Recover the processed data in this module
                    fmri_img.uncache()
                    fmri_img._dataobj = self.proc_data
                    fmri_img.set_data_dtype = self.proc_data.dtype
                    fmri_img.set_filename(save_name)

                self.keep_processed_image(fmri_img,
                                          save_temp=self.online_saving)

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errmsg = f'{exc_type}, {exc_tb.tb_frame.f_code.co_filename}' + \
                     f':{exc_tb.tb_lineno}'
            self.errmsg(errmsg, no_pop=True)
            traceback.print_exc(file=self._err_out)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end_reset(self):
        """ End process and reset process parameters. """

        if self.verb:
            self.logmsg(f"Reset {self.__class__.__name__} module.")

        self.pre_data = []

        return super(RTP_TSHIFT, self).end_reset()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def slice_timing_from_sample(self, fmri_img):
        """
        Set slice timing from a sample mri_data

        Parameters
        ----------
        fmri_img: string_types or nibabel image object
            If fmri_img is string_types, it should be a BRIK filename of sample
            data.
        """

        if isinstance(fmri_img, string_types) or isinstance(fmri_img, Path):
            try:
                img = nib.load(fmri_img)
            except nib.filebasedimages.ImageFileError:
                img = mosaic_to_nii(pydicom.read_file(fmri_img))

            vol_shape = img.shape[:3]
            header = img.header
            fname = fmri_img
        else:
            vol_shape = fmri_img.shape[:3]
            fname = fmri_img.get_filename()
            header = fmri_img.header

        if hasattr(header, 'get_slice_times'):
            try:
                self.slice_timing = header.get_slice_times()
            except nib.spatialimages.HeaderDataError:
                self.errmsg(f'{fname} has no slice timing info.')
                return
        elif hasattr(header, 'info') and 'TAXIS_FLOATS' in header.info:
            self.slice_timing = header.info['TAXIS_OFFSETS']
        else:
            self.errmsg(f'{fname} has no slice timing info.')
            return

        if self._verb:
            msg = f'Slice timing = {self.slice_timing}.'
            self.logmsg(msg)

        # set interpolation weight
        self._pre_culc_weight(vol_shape)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _pre_culc_weight(self, vol_shape):
        """
        Pre-calculation of weights for temporal interpolation

        Set variables
        -------------
        Interpolation weights
        self.wm1: weigth for minus 1 TR data
        self.w0: weigth for current TR data

        Only for cubic interpolation
        self.wm2: weigth for minus 2 TR data
        self.wp1: weigth for plus 1 TR data
                  used by cubic interpolation assuming the next data is the
                  same as current data

        Retrospective extrapolation weights
        self.r1wm1: 1-TR retrospective weigth for minus 1 TR data
        self.r1w0: 1-TR retrospective weigth for current TR data

        Only for cubic interpolation
        self.r1wm2: 1-TR retrospective weigth for minus 2 TR data
        self.r1wp1: 1-TR retrospective weigth for plus 1 TR data

        self.r2wm1: 2-TR retrospective weigth for minus 1 TR data
        self.r2w0: 2-TR retrospective weigth for current TR data
        self.r2wm2: 2-TR retrospective weigth for minus 2 TR data
        self.r2wp1: 2-TR retrospective weigth for plus 1 TR data
        """

        if self.slice_timing is None or self.TR is None:
            self.errmsg("slice timing is not set.")
            return

        if self.method not in ['linear', 'cubic']:
            self.errmsg("{} is not supported.".format(self.method))
            return

        # Set reference time
        ref_time = self.ref_time

        # slice timing shift from ref_time (relative to TR)
        shf = [(slt - ref_time)/self.TR for slt in self.slice_timing]

        # Initialize masked weight
        self.wm1 = np.ones(vol_shape, dtype=np.float32)  # weight for t-1
        self.w0 = np.ones(vol_shape, dtype=np.float32)  # weight for t
        # retrospective weight
        self.r1wm1 = np.ones(vol_shape, dtype=np.float32)  # weight for t-1
        self.r1w0 = np.ones(vol_shape, dtype=np.float32)  # weight for t

        if self.method == 'cubic':
            self.wm2 = np.ones(vol_shape, dtype=np.float32)  # weight for t-2
            self.wp1 = np.ones(vol_shape, dtype=np.float32)  # weight for t+1
            # retrospective weight
            self.r1wm2 = np.ones(vol_shape, dtype=np.float32)  # weight for t-1
            self.r1wp1 = np.ones(vol_shape, dtype=np.float32)  # weight for t

        # Set weight
        for sli in range(len(self.slice_timing)):
            if self.method == 'linear':
                wm1 = shf[sli]  # <=> x1
                w0 = 1.0 - wm1  # <=> -x0
                if self.slice_dim == 0:
                    self.wm1[sli, :, :] *= wm1
                    self.w0[sli, :, :] *= w0
                elif self.slice_dim == 1:
                    self.wm1[:, sli, :] *= wm1
                    self.w0[:, sli, :] *= w0
                elif self.slice_dim == 2:
                    self.wm1[:, :, sli] *= wm1
                    self.w0[:, :, sli] *= w0

                # Retrospective weight
                r1wm1 = shf[sli] + 1.0  # <=> x1 - (-1)
                r1w0 = -shf[sli]  # <=> -X1
                if self.slice_dim == 0:
                    self.r1wm1[sli, :, :] *= r1wm1
                    self.r1w0[sli, :, :] *= r1w0
                elif self.slice_dim == 1:
                    self.r1wm1[:, sli, :] *= r1wm1
                    self.r1w0[:, sli, :] *= r1w0
                elif self.slice_dim == 2:
                    self.r1wm1[:, :, sli] *= r1wm1
                    self.r1w0[:, :, sli] *= r1w0

            elif self.method == 'cubic':
                aa = 1.0 - shf[sli]
                wm2 = aa * (1.0-aa) * (aa-2.0) * 0.1666667
                wm1 = (aa+1.0) * (aa-1.0) * (aa-2.0) * 0.5
                w0 = aa * (aa+1.0) * (2.0-aa) * 0.5
                wp1 = aa * (aa+1.0) * (aa-1.0) * 0.1666667
                if self.slice_dim == 0:
                    self.wm2[sli, :, :] *= wm2
                    self.wm1[sli, :, :] *= wm1
                    self.w0[sli, :, :] *= w0
                    self.wp1[sli, :, :] *= wp1
                elif self.slice_dim == 1:
                    self.wm2[:, sli, :] *= wm2
                    self.wm1[:, sli, :] *= wm1
                    self.w0[:, sli, :] *= w0
                    self.wp1[:, sli, :] *= wp1
                elif self.slice_dim == 2:
                    self.wm2[:, :, sli] *= wm2
                    self.wm1[:, :, sli] *= wm1
                    self.w0[:, :, sli] *= w0
                    self.wp1[:, :, sli] *= wp1

                # 1-TR Retrospective weight
                aa = 1.0 - shf[sli] - 1.0
                r1wm2 = aa * (1.0-aa) * (aa-2.0) * 0.1666667
                r1wm1 = (aa+1.0) * (aa-1.0) * (aa-2.0) * 0.5
                r1w0 = aa * (aa+1.0) * (2.0-aa) * 0.5
                r1wp1 = aa * (aa+1.0) * (aa-1.0) * 0.1666667
                if self.slice_dim == 0:
                    self.r1wm2[sli, :, :] *= r1wm2
                    self.r1wm1[sli, :, :] *= r1wm1
                    self.r1w0[sli, :, :] *= r1w0
                    self.r1wp1[sli, :, :] *= r1wp1
                elif self.slice_dim == 1:
                    self.r1wm2[:, sli, :] *= r1wm2
                    self.r1wm1[:, sli, :] *= r1wm1
                    self.r1w0[:, sli, :] *= r1w0
                    self.r1wp1[:, sli, :] *= r1wp1
                elif self.slice_dim == 2:
                    self.r1wm2[:, :, sli] *= r1wm2
                    self.r1wm1[:, :, sli] *= r1wm1
                    self.r1w0[:, :, sli] *= r1w0
                    self.r1wp1[:, :, sli] *= r1wp1

        self.prep_weight = True

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_slice_timing_uis(self):
        if hasattr(self, 'ui_TR_dSpBx'):
            self.ui_TR_dSpBx.setValue(self.TR)

        if hasattr(self, 'ui_SlTiming_lnEd'):
            self.ui_SlTiming_lnEd.setText("{}".format(self.slice_timing))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_param(self, attr, val=None, reset_fn=None, echo=False):
        """
        When reset_fn is None, set_param is considered to be called from
        load_parameters function
        """

        # -- check value --
        if attr == 'enabled':
            if hasattr(self, 'ui_enabled_rdb'):
                self.ui_enabled_rdb.setChecked(val)

            if hasattr(self, 'ui_objs'):
                for ui in self.ui_objs:
                    ui.setEnabled(val)

        elif attr == 'work_dir':
            if val is None or not Path(val).is_dir():
                return

            val = Path(val)
            setattr(self, attr, val)

            if self.main_win is not None:
                self.main_win.set_workDir(val)

        elif attr == 'ignore_init' and reset_fn is None:
            if hasattr(self, 'ui_ignorInit_spBx'):
                self.ui_ignorInit_spBx.setValue(val)

        elif attr == 'method' and reset_fn is None:
            if hasattr(self, 'ui_method_cmbBx'):
                self.ui_method_cmbBx.setCurrentText(val)

        elif attr == 'ref_time' and reset_fn is None:
            if hasattr(self, 'ui_refTime_dSpBx'):
                self.ui_refTime_dSpBx.setValue(val)

        elif attr == 'slice_timing_from_sample':
            if val is None:
                fname = self.select_file_dlg(
                        'TSHIFT: Selct slice timing sample',
                        self.work_dir, "*.BRIK* *.nii*")
                if fname[0] == '':
                    return -1

                val = fname[0]

            self.slice_timing_from_sample(val)
            self.set_slice_timing_uis()
            return 0

        elif attr == 'TR' and reset_fn is None:
            if hasattr(self, 'ui_TR_dSpBx'):
                self.ui_TR_dSpBx.setValue(val)

        elif attr == 'slice_timing':
            if type(val) == str:
                try:
                    val = eval(val)
                except Exception:
                    if reset_fn:
                        reset_fn(str(getattr(self, attr)))
                    return
            else:
                if hasattr(self, 'ui_SlTiming_lnEd'):
                    self.ui_SlTiming_lnEd.setText(str(val))

        elif attr == 'slice_dim' and reset_fn is None:
            if hasattr(self, 'ui_sliceDim_cmbBx'):
                self.ui_sliceDim_cmbBx.setCurrentIndex(val)

        elif attr == 'save_proc':
            if hasattr(self, 'ui_saveProc_chb'):
                self.ui_saveProc_chb.setChecked(val)

        elif attr == '_verb':
            if hasattr(self, 'ui_verb_chb'):
                self.ui_verb_chb.setChecked(val)

        elif reset_fn is None:
            # Ignore an unrecognized parameter
            if not hasattr(self, attr):
                self.errmsg(f"{attr} is unrecognized parameter.", no_pop=True)
                return

        # -- Set value --
        setattr(self, attr, val)
        if echo and self._verb:
            print("{}.".format(self.__class__.__name__) + attr, '=',
                  getattr(self, attr))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ui_set_param(self):

        ui_rows = []
        self.ui_objs = []

        # enabled
        self.ui_enabled_rdb = QtWidgets.QRadioButton("Enable")
        self.ui_enabled_rdb.setChecked(self.enabled)
        self.ui_enabled_rdb.toggled.connect(
                lambda checked:
                self.set_param('enabled', checked,
                               self.ui_enabled_rdb.setChecked))
        ui_rows.append((self.ui_enabled_rdb, None))

        # ignore_init
        var_lb = QtWidgets.QLabel("Ignore initial volumes :")
        self.ui_ignorInit_spBx = QtWidgets.QSpinBox()
        self.ui_ignorInit_spBx.setValue(self.ignore_init)
        self.ui_ignorInit_spBx.setMinimum(0)
        self.ui_ignorInit_spBx.valueChanged.connect(
                lambda x: self.set_param('ignore_init', x,
                                         self.ui_ignorInit_spBx.setValue))
        ui_rows.append((var_lb, self.ui_ignorInit_spBx))
        self.ui_objs.extend([var_lb, self.ui_ignorInit_spBx])

        # method
        var_lb = QtWidgets.QLabel("Temporal interpolation method :")
        self.ui_method_cmbBx = QtWidgets.QComboBox()
        self.ui_method_cmbBx.addItems(['linear', 'cubic'])
        self.ui_method_cmbBx.setCurrentText(self.method)
        self.ui_method_cmbBx.currentIndexChanged.connect(
                lambda idx:
                self.set_param('method',
                               self.ui_method_cmbBx.currentText(),
                               self.ui_method_cmbBx.setCurrentText))
        ui_rows.append((var_lb, self.ui_method_cmbBx))
        self.ui_objs.extend([var_lb, self.ui_method_cmbBx])

        # ref_time
        var_lb = QtWidgets.QLabel("Reference time :")
        self.ui_refTime_dSpBx = QtWidgets.QDoubleSpinBox()
        self.ui_refTime_dSpBx.setMinimum(0.000)
        self.ui_refTime_dSpBx.setSingleStep(0.001)
        self.ui_refTime_dSpBx.setDecimals(3)
        self.ui_refTime_dSpBx.setSuffix(" seconds")
        self.ui_refTime_dSpBx.setValue(self.ref_time)
        self.ui_refTime_dSpBx.valueChanged.connect(
                lambda x: self.set_param('ref_time', x,
                                         self.ui_refTime_dSpBx.setValue))
        ui_rows.append((var_lb, self.ui_refTime_dSpBx))
        self.ui_objs.extend([var_lb, self.ui_refTime_dSpBx])

        # Load from sample button
        self.ui_setfrmSample_btn = QtWidgets.QPushButton(
                'Get slice timing parameters from a sample file')
        self.ui_setfrmSample_btn.clicked.connect(
                lambda: self.set_param('slice_timing_from_sample'))
        ui_rows.append((self.ui_setfrmSample_btn,))
        self.ui_objs.append(self.ui_setfrmSample_btn)

        # TR
        var_lb = QtWidgets.QLabel("TR :")
        self.ui_TR_dSpBx = QtWidgets.QDoubleSpinBox()
        self.ui_TR_dSpBx.setMinimum(0.000)
        self.ui_TR_dSpBx.setSingleStep(0.001)
        self.ui_TR_dSpBx.setDecimals(3)
        self.ui_TR_dSpBx.setSuffix(" seconds")
        self.ui_TR_dSpBx.setValue(self.TR)
        self.ui_TR_dSpBx.valueChanged.connect(
                lambda x: self.set_param('TR', x, self.ui_TR_dSpBx.setValue))
        ui_rows.append((var_lb, self.ui_TR_dSpBx))
        self.ui_objs.extend([var_lb, self.ui_TR_dSpBx])

        # slice_timing
        var_lb = QtWidgets.QLabel("Slice timings (sec.) :\n[1st, 2nd, ...]")
        self.ui_SlTiming_lnEd = QtWidgets.QLineEdit()
        self.ui_SlTiming_lnEd.setText("{}".format(self.slice_timing))
        self.ui_SlTiming_lnEd.editingFinished.connect(
                lambda: self.set_param('slice_timing',
                                       self.ui_SlTiming_lnEd.text(),
                                       self.ui_SlTiming_lnEd.setText))
        ui_rows.append((var_lb, self.ui_SlTiming_lnEd))
        self.ui_objs.extend([var_lb, self.ui_SlTiming_lnEd])

        # slice_dim
        var_lb = QtWidgets.QLabel("Slice orientation :")
        self.ui_sliceDim_cmbBx = QtWidgets.QComboBox()
        self.ui_sliceDim_cmbBx.addItems(['x (Sagital)', 'y (Coronal)',
                                         'z (Axial)'])
        self.ui_sliceDim_cmbBx.setCurrentIndex(self.slice_dim)
        self.ui_sliceDim_cmbBx.currentIndexChanged.connect(
                lambda idx:
                self.set_param('slice_dim',
                               self.ui_sliceDim_cmbBx.currentIndex(),
                               self.ui_sliceDim_cmbBx.setCurrentIndex))
        ui_rows.append((var_lb, self.ui_sliceDim_cmbBx))
        self.ui_objs.extend([var_lb, self.ui_sliceDim_cmbBx])

        # --- Checkbox row ----------------------------------------------------
        # Save
        self.ui_saveProc_chb = QtWidgets.QCheckBox("Save processed image")
        self.ui_saveProc_chb.setChecked(self.save_proc)
        self.ui_saveProc_chb.stateChanged.connect(
                lambda state: setattr(self, 'save_proc', state > 0))
        self.ui_objs.append(self.ui_saveProc_chb)

        # verb
        self.ui_verb_chb = QtWidgets.QCheckBox("Verbose logging")
        self.ui_verb_chb.setChecked(self.verb)
        self.ui_verb_chb.stateChanged.connect(
                lambda state: setattr(self, 'verb', state > 0))
        self.ui_objs.append(self.ui_verb_chb)

        chb_hLayout = QtWidgets.QHBoxLayout()
        chb_hLayout.addStretch()
        chb_hLayout.addWidget(self.ui_saveProc_chb)
        chb_hLayout.addWidget(self.ui_verb_chb)
        ui_rows.append((None, chb_hLayout))

        return ui_rows

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_params(self):
        all_opts = super().get_params()
        excld_opts = ('work_dir', 'pre_data',
                      'pre_mri_data', 'prep_weight', 'wm2', 'wm1', 'w0', 'wp1',
                      'r1wm2', 'r1wm1', 'r1w0', 'r1wp1')
        sel_opts = {}
        for k, v in all_opts.items():
            if k in excld_opts:
                continue
            if isinstance(v, Path):
                v = str(v)
            sel_opts[k] = v

        return sel_opts


# %% __main__ (test) ==========================================================
if __name__ == '__main__':
    # --- Test ---
    # test data directory
    test_dir = Path(__file__).absolute().parent.parent / 'test'

    # Load test data
    testdata_f = test_dir / 'func_epi.nii.gz'
    assert testdata_f.is_file()
    img = nib.load(testdata_f)
    img_data = np.asanyarray(img.dataobj)
    N_vols = img.shape[-1]

    work_dir = test_dir / 'work'
    if not work_dir.is_dir():
        work_dir.mkdir()

    # Create RTP_TSHIFT instance
    rtp_tshift = RTP_TSHIFT()
    rtp_tshift.method = 'cubic'
    rtp_tshift.ignore_init = 3
    rtp_tshift.verb = True
    rtp_tshift.ref_time = 0

    # Set slice timing from a sample data
    rtp_tshift.slice_timing_from_sample(testdata_f)

    # Set parameters for debug
    rtp_tshift.work_dir = work_dir
    rtp_tshift.save_proc = True
    rtp_tshift.save_delay = True
    rtp_tshift.verb = True

    # Run tshift
    rtp_tshift.end_reset()

    N_vols = img.shape[-1]
    for ii in range(N_vols):
        save_filename = f"test_nr_{ii:04d}.nii.gz"
        fmri_img = nib.Nifti1Image(img_data[:, :, :, ii], affine=img.affine)
        fmri_img.set_filename(save_filename)
        st = time.time()
        rtp_tshift.do_proc(fmri_img, ii, st)

    proc_delay = rtp_tshift.proc_delay
    rtp_tshift.end_reset()

    # --- Load processed data  ---
    img_ts = nib.load(rtp_tshift.saved_filename)
    img_data_ts = img_ts.get_fdata()

    # --- Plot ---
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(proc_delay[1:], bins='auto')
    np.median(proc_delay[1:])

    plot_range = 30  # TRs
    sli = 1
    xi = np.arange(plot_range)
    plt.figure()
    xraw = (np.array(xi)+rtp_tshift.ignore_init) * rtp_tshift.TR + \
        rtp_tshift.slice_timing[sli]
    plt.plot(xraw, img_data[64, 64, sli, rtp_tshift.ignore_init+xi],
             linestyle='-', marker='o', label='raw')
    xshift = (np.array(xi)+rtp_tshift.ignore_init) * rtp_tshift.TR + \
        rtp_tshift.ref_time
    plt.plot(xshift, img_data_ts[64, 64, sli, xi],
             linestyle='--', marker='x', label='shift')
    plt.legend()
    plt.title(f"Shift -{rtp_tshift.slice_timing[sli]:.3f}s")
    plt.xlabel('Time (s)')
