#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTP spatial smoothing

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import re
import time
import ctypes
import sys
import traceback

import numpy as np
import nibabel as nib
from six import string_types

from PyQt5 import QtWidgets

try:
    from .rtp_common import RTP
except Exception:
    from rtpspy.rtp_common import RTP

if sys.platform == 'linux':
    lib_name = 'librtp.so'
elif sys.platform == 'darwin':
    lib_name = 'librtp.dylib'

try:
    librtp_path = str(Path(__file__).absolute().parent / lib_name)
except Exception:
    librtp_path = f'./{lib_name}'


# %% RtpSmooth class =========================================================
class RtpSmooth(RTP):
    """
    RTP spatial smoothing
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, blur_fwhm=6.0, mask_file=0, **kwargs):
        """
        Parameters
        ----------
        blur_fwhm : float, optional
            FWHM of Gaussina smoothing kernel. The default is 6.0.
        mask_file : int (0) or str, optional
            0: mask is made with the initial recevied volume (zero-out).
            str: mask filename.
            The default is 0.

        """
        super(RtpSmooth, self).__init__(**kwargs)

        # Set instance parameters
        self.blur_fwhm = blur_fwhm
        self.mask_file = mask_file

        if isinstance(self.mask_file, string_types) or \
                isinstance(self.mask_file, Path):
            # Set byte mask
            self.set_mask(self.mask_file)
        else:
            self.mask_byte = None

        # Initialize C library function call
        self.setup_libfuncs()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ready_proc(self):
        self._proc_ready = True

        if self.next_proc:
            self._proc_ready &= self.next_proc.ready_proc()

        return self._proc_ready

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def do_proc(self, fmri_img, vol_idx=None, pre_proc_time=None, **kwargs):
        try:
            # Increment the number of received volume: vol_num is 0-based
            self._vol_num += 1  # 1- base number of volumes recieved by this
            if vol_idx is None:
                vol_idx = self._vol_num - 1  # 0-base index

            if vol_idx < self.ignore_init:
                # Skip ignore_init volumes
                return

            if self._proc_start_idx < 0:
                self._proc_start_idx = vol_idx

            # --- Initialize --------------------------------------------------
            # Unless the mask is set, set it by a received volume
            if self.mask_byte is None or not hasattr(self, 'maskV'):
                msg = f"Mask is set by a received volume, index {vol_idx}"
                self._logger.info(msg)

                self.set_mask(fmri_img.get_fdata())

            # --- Run the procress --------------------------------------------
            # Perform smoothing
            smoothed_img_data = self.smooth(fmri_img)

            fmri_img.uncache()
            fmri_img._dataobj = smoothed_img_data
            fmri_img.set_data_dtype = smoothed_img_data.dtype

            # --- Post procress -----------------------------------------------
            # Record process time
            tstamp = time.time()
            self._proc_time.append(tstamp)
            if pre_proc_time is not None:
                proc_delay = self._proc_time[-1] - pre_proc_time
                if self.save_delay:
                    self.proc_delay.append(proc_delay)

            # log message
            f = Path(fmri_img.get_filename()).name
            msg = f"#{vol_idx+1};Smoothing;{f}"
            msg += f";tstamp={tstamp}"
            if pre_proc_time is not None:
                msg += f";took {proc_delay:.4f}s"
            self._logger.info(msg)

            # Set save_name
            fmri_img.set_filename('sm.' + Path(fmri_img.get_filename()).name)

            if self.next_proc:
                # Keep the current processed data
                self.proc_data = np.asanyarray(fmri_img.dataobj)
                save_name = fmri_img.get_filename()

                # Run the next process
                self.next_proc.do_proc(fmri_img, vol_idx=vol_idx,
                                       pre_proc_time=self._proc_time[-1])

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
            self._logger.error(errmsg)
            traceback.print_exc(file=self._err_out)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end_reset(self):
        """ End process and reset process parameters. """

        self._logger.info(f"Reset {self.__class__.__name__} module.")

        if type(self.mask_file) is int and self.mask_file == 0:
            self.mask_byte = None

        return super(RtpSmooth, self).end_reset()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def setup_libfuncs(self):
        """ Setup library function access """

        librtp = ctypes.cdll.LoadLibrary(librtp_path)

        # -- Define rtp_smooth --
        self.rtp_smooth = librtp.rtp_smooth
        self.rtp_smooth.argtypes = \
            [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
             ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_void_p,
             ctypes.c_float]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_mask(self, maskdata, sub_i=0, method='zero_out'):
        if isinstance(maskdata, string_types) or isinstance(maskdata, Path):
            ma = re.search(r"\'*\[(\d+)\]\'*$", str(maskdata))
            if ma:
                sub_i = int(ma.groups()[0])
                maskdata = re.sub(r"\'*\[(\d+)\]\'*$", '', str(maskdata))

            if not Path(maskdata).is_file():
                errmsg = f"Not found mask file: {maskdata}"
                self._logger.error(errmsg)
                self.err_popup(errmsg)
                self.mask_file = 0
                return

            self.mask_file = str(maskdata)

            maskdata = np.squeeze(nib.load(maskdata).get_fdata())
            if maskdata.ndim > 3:
                maskdata = maskdata[:, :, :, sub_i]

            msg = f"Mask = {self.mask_file}"
            if ma:
                msg += f"[{sub_i}]"
            self._logger.info(msg)

        if method == 'zero_out':
            self.maskV = maskdata != 0

        # Set byte mask
        self.mask_byte = self.maskV.astype('u2')
        self.mask_byte_p = self.mask_byte.ctypes.data_as(
                ctypes.POINTER(ctypes.c_ushort))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def smooth(self, fmri_img):
        """
        Run spatial smoothing with rtp_smooth function in librtp.so

        C function definition
        ---------------------
        int rtp_smooth(float *im, int nx, int ny, int nz, float dx, float dy,
                       float dz, unsigned short *mask, float fwhm)

        Parameters
        ----------
        fmri_img : nibabel image object
            Input image.

        Returns
        -------
        fim_arr : array
            smoothed image data.

        """
        # image dimension
        nx, ny, nz = fmri_img.shape[:3]
        if hasattr(fmri_img.header, 'info'):
            # BRIK
            dx, dy, dz = np.abs(fmri_img.header.info['DELTA'])
        elif hasattr(fmri_img.header, 'get_zooms'):
            # NIfTI
            dx, dy, dz = fmri_img.header.get_zooms()[:3]
        else:
            errmsg = "No voxel size information in fmri_img header"
            self._logger.error(errmsg)
            self.err_popup(errmsg)

        # Copy function image data and get pointer
        fim_arr = fmri_img.get_fdata().astype(np.float32)
        if fim_arr.ndim > 3:
            fim_arr = np.squeeze(fim_arr)
        fim_arr = np.moveaxis(np.moveaxis(fim_arr, 0, -1), 0, 1).copy()
        fim_p = fim_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self.rtp_smooth(fim_p, nx, ny, nz, dx, dy, dz, self.mask_byte_p,
                        self.blur_fwhm)
        fim_arr = np.moveaxis(np.moveaxis(fim_arr, 0, -1), 0, 1)

        return fim_arr

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_param(self, attr, val=None, reset_fn=None, echo=False):
        """
        When reset_fn is None, set_param is considered to be called from
        load_parameters function.
        """
        self._logger.debug(f"set_param: {attr} = {val}")

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

        if attr == 'blur_fwhm' and reset_fn is None:
            if hasattr(self, 'ui_FWHM_dSpBx'):
                self.ui_FWHM_dSpBx.setValue(val)

        elif attr == 'mask_file':
            if isinstance(val, Path):
                val = str(val)
                self.set_mask(val)
                if self.mask_file != 0 and hasattr(self, 'ui_mask_cmbBx'):
                    self.ui_mask_cmbBx.setCurrentText('external file')
                    self.ui_mask_lnEd.setText(str(val))

            elif type(val) is int and val == 0:
                if hasattr(self, 'ui_mask_lnEd'):
                    self.ui_mask_lnEd.setText(
                        'zero-out initial received volume')
                self.mask_byte = None

            elif 'internal' in val:
                val = 0
                if hasattr(self, 'ui_mask_lb'):
                    self.ui_mask_lnEd.setText(
                        'zero-out initial received volume')
                self.mask_byte = None

            elif 'external' in val:
                fname = self.select_file_dlg('SMOOTH: Select mask volume',
                                             self.work_dir, "*.BRIK* *.nii*")
                if fname[0] == '':
                    if reset_fn:
                        reset_fn(1)
                    return -1

                mask_img = nib.load(fname[0])
                mask_fname = fname[0]
                if len(mask_img.shape) > 3 and mask_img.shape[3] > 1:
                    num, okflag = QtWidgets.QInputDialog.getInt(
                            None, "Select sub volume",
                            "sub-volume index (0 is first)", 0, 0,
                            mask_img.shape[3])
                    if fname[0] == '':
                        if reset_fn:
                            reset_fn(1)
                        return -1

                    mask_fname += f"[{num}]"
                else:
                    num = 0

                self.set_mask(fname[0], num)
                if hasattr(self, 'ui_mask_lnEd'):
                    self.ui_mask_lnEd.setText(str(mask_fname))
                val = mask_fname

            elif type(val) is str:
                ma = re.search(r"\[(\d+)\]", val)
                if ma:
                    num = int(ma.groups()[0])
                    fname = re.sub(r"\[(\d+)\]", '', val)
                else:
                    fname = val
                    num = 0

                if not Path(fname).is_file():
                    return

                if reset_fn is None and hasattr(self, 'ui_mask_cmbBx'):
                    # set 'external'
                    self.ui_mask_cmbBx.setCurrentIndex(0)

                self.set_mask(fname, num)
                if hasattr(self, 'ui_mask_lnEd'):
                    self.ui_mask_lnEd.setText(str(val))

        elif attr == 'save_proc':
            if hasattr(self, 'ui_saveProc_chb'):
                self.ui_saveProc_chb.setChecked(val)

        elif reset_fn is None:
            # Ignore an unrecognized parameter
            if not hasattr(self, attr):
                errmsg = f"{attr} is unrecognized parameter."
                self._logger.error(errmsg)
                self.err_popup(errmsg)
                return

        # -- Set value --
        setattr(self, attr, val)
        if echo:
            print(f"{self.__class__.__name__}." + attr, '=',
                  getattr(self, attr))

        return 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ui_set_param(self):

        ui_rows = []
        self.ui_objs = []

        # enabled
        self.ui_enabled_rdb = QtWidgets.QRadioButton("Enable")
        self.ui_enabled_rdb.setChecked(self.enabled)
        self.ui_enabled_rdb.toggled.connect(
                lambda checked: self.set_param('enabled', checked,
                                               self.ui_enabled_rdb.setChecked))
        ui_rows.append((self.ui_enabled_rdb, None))

        # blur_fwhm
        var_lb = QtWidgets.QLabel("Gaussian FWHM :")
        self.ui_FWHM_dSpBx = QtWidgets.QDoubleSpinBox()
        self.ui_FWHM_dSpBx.setMinimum(0.0)
        self.ui_FWHM_dSpBx.setSingleStep(1.0)
        self.ui_FWHM_dSpBx.setDecimals(2)
        self.ui_FWHM_dSpBx.setSuffix(" mm")
        self.ui_FWHM_dSpBx.setValue(self.blur_fwhm)
        self.ui_FWHM_dSpBx.valueChanged.connect(
                lambda x: self.set_param('blur_fwhm', x,
                                         self.ui_FWHM_dSpBx.setValue))
        ui_rows.append((var_lb, self.ui_FWHM_dSpBx))
        self.ui_objs.extend([var_lb, self.ui_FWHM_dSpBx])

        # mask_file
        var_lb = QtWidgets.QLabel("Mask :")
        self.ui_mask_cmbBx = QtWidgets.QComboBox()
        self.ui_mask_cmbBx.addItems(['external file',
                                     'initial volume of internal run'])
        self.ui_mask_cmbBx.activated.connect(
                lambda idx:
                self.set_param('mask_file',
                               self.ui_mask_cmbBx.currentText(),
                               self.ui_mask_cmbBx.setCurrentIndex))
        ui_rows.append((var_lb, self.ui_mask_cmbBx))

        self.ui_mask_lnEd = QtWidgets.QLineEdit()
        self.ui_mask_lnEd.setReadOnly(True)
        self.ui_mask_lnEd.setStyleSheet(
            'border: 0px none;')
        ui_rows.append((None, self.ui_mask_lnEd))

        self.ui_objs.extend([var_lb, self.ui_mask_cmbBx, self.ui_mask_lnEd])

        if type(self.mask_file) is int and self.mask_file == 0:
            self.ui_mask_cmbBx.setCurrentIndex(1)
            self.ui_mask_lnEd.setText('zero-out initial received volume')
        else:
            self.ui_mask_cmbBx.setCurrentIndex(0)
            self.ui_mask_lnEd.setText(str(self.mask_file))

        # --- Checkbox row ----------------------------------------------------
        # Save
        self.ui_saveProc_chb = QtWidgets.QCheckBox("Save processed image")
        self.ui_saveProc_chb.setChecked(self.save_proc)
        self.ui_saveProc_chb.stateChanged.connect(
                lambda state: setattr(self, 'save_proc', state > 0))
        self.ui_objs.append(self.ui_saveProc_chb)

        chb_hLayout = QtWidgets.QHBoxLayout()
        chb_hLayout.addStretch()
        chb_hLayout.addWidget(self.ui_saveProc_chb)
        ui_rows.append((None, chb_hLayout))

        return ui_rows

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_params(self):
        all_opts = super().get_params()
        excld_opts = ('work_dir', 'mask_byte', 'maskV')
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
    test_dir = Path(__file__).absolute().parent.parent / 'tests'

    # Load test data
    testdata_f = test_dir / 'func_epi.nii.gz'
    assert testdata_f.is_file()
    img = nib.load(testdata_f)
    img_data = np.asanyarray(img.dataobj)
    N_vols = img.shape[-1]

    work_dir = test_dir / 'work'
    if not work_dir.is_dir():
        work_dir.mkdir()

    # Create RtpTshift and RtpVolreg  instance
    from rtpspy.rtp_tshift import RtpTshift
    from rtpspy.rtp_volreg import RtpVolreg

    rtp_tshift = RtpTshift()
    rtp_tshift.method = 'cubic'
    rtp_tshift.ignore_init = 3
    rtp_tshift.ref_time = 0

    rtp_volreg = RtpVolreg(regmode='cubic')

    rtp_smooth = RtpSmooth(blur_fwhm=6)

    # Set slice timing from a sample data
    rtp_tshift.slice_timing_from_sample(testdata_f)

    # Set reference volume
    refname = str(testdata_f) + '[0]'
    rtp_volreg.set_ref_vol(refname)

    # Set mask
    rtp_smooth.set_param('mask_file', 0)

    # Set parameters for debug
    rtp_tshift.work_dir = work_dir
    rtp_tshift.save_proc = True
    rtp_volreg.work_dir = work_dir
    rtp_volreg.save_proc = True
    rtp_smooth.work_dir = work_dir
    rtp_smooth.save_proc = True
    rtp_smooth.save_delay = True

    # Chain tshift -> volreg -> smooth
    rtp_tshift.next_proc = rtp_volreg
    rtp_volreg.next_proc = rtp_smooth
    rtp_tshift.end_reset()

    # Run
    N_vols = img.shape[-1]
    for ii in range(N_vols):
        save_filename = f"test_nr_{ii:04d}.nii.gz"
        fmri_img = nib.Nifti1Image(img_data[:, :, :, ii], affine=img.affine)
        fmri_img.set_filename(save_filename)
        st = time.time()
        rtp_tshift.do_proc(fmri_img, ii, st)  # run rtp_tshift -> rtp_volreg

    proc_delay = rtp_smooth.proc_delay
    rtp_tshift.end_reset()

    # --- Plot ---
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(proc_delay[1:], bins='auto')
    np.median(proc_delay[1:])
