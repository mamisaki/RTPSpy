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

import numpy as np
import nibabel as nib

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
    Real-time spatial smoothing within a mask. The AFNI-based rtp_smooth
    function in librtp.so is called via ctypes.
    """

    # Text shown when the mask is made from the first received volume
    _ZERO_OUT_MASK_TEXT = 'zero-out initial received volume'

    # Log label and output filename prefix (see RTP.do_proc)
    _proc_label = 'Smoothing'
    _out_prefix = 'sm.'

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, blur_fwhm=6.0, mask_file=0, **kwargs):
        """
        Parameters
        ----------
        blur_fwhm : float, optional
            FWHM (mm) of the Gaussian smoothing kernel. The default is 6.0.
        mask_file : int (0), str, or Path, optional
            0: The mask is made from the first received volume (nonzero
            voxels).
            str or Path: Mask filename.
            The default is 0.
        """
        super(RtpSmooth, self).__init__(**kwargs)

        # Set instance parameters
        self.blur_fwhm = blur_fwhm
        self.mask_file = mask_file

        if isinstance(self.mask_file, (str, Path)):
            self.set_mask(self.mask_file)
        else:
            self.mask_byte = None

        self.setup_libfuncs()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ready_proc(self):
        self._proc_ready = True

        if self.next_proc:
            self._proc_ready &= self.next_proc.ready_proc()

        return self._proc_ready

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _process(self, fmri_img, vol_idx, pre_proc_time):
        """Smooth the volume (see RTP.do_proc)."""
        # --- Initialize ------------------------------------------------------
        # If no mask is set, make it from this volume.
        if self.mask_byte is None or not hasattr(self, 'maskV'):
            self._logger.info(
                f"Mask is set from the received volume {vol_idx}")
            self.set_mask(np.asarray(fmri_img.dataobj))

        # --- Run the process -------------------------------------------------
        return fmri_img.with_data(self.smooth(fmri_img))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end_reset(self):
        """ End process and reset process parameters. """

        self._logger.info(f"Reset {self.__class__.__name__} module.")

        if type(self.mask_file) is int and self.mask_file == 0:
            # Make the mask again from the first volume of the next scan
            self.mask_byte = None

        return super(RtpSmooth, self).end_reset()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def setup_libfuncs(self):
        """ Set up the C library function. """

        librtp = ctypes.cdll.LoadLibrary(librtp_path)

        # -- rtp_smooth --
        self.rtp_smooth = librtp.rtp_smooth
        self.rtp_smooth.argtypes = \
            [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
             ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_void_p,
             ctypes.c_float]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_mask(self, maskdata, sub_i=0, method='zero_out'):
        """
        Set the smoothing mask.

        Parameters
        ----------
        maskdata : str, Path, or array
            Mask filename (an AFNI-style volume index, e.g., 'mask.nii[1]',
            overrides sub_i) or mask image data.
        sub_i : int, optional
            Volume index in a 4D mask file. The default is 0.
        method : str, optional
            'zero_out': nonzero voxels are in the mask. The default is
            'zero_out'.
        """
        if isinstance(maskdata, (str, Path)):
            ma = re.search(r"\'*\[(\d+)\]\'*$", str(maskdata))
            if ma:
                sub_i = int(ma.groups()[0])
                maskdata = re.sub(r"\'*\[(\d+)\]\'*$", '', str(maskdata))

            if not Path(maskdata).is_file():
                errmsg = f"Mask file not found: {maskdata}"
                self._logger.error(errmsg)
                self.err_popup(errmsg)
                # Fall back to the zero-out mask from the first volume
                self.mask_file = 0
                self.mask_byte = None
                return

            self.mask_file = str(maskdata)

            maskdata = np.squeeze(np.asarray(nib.load(maskdata).dataobj))
            if maskdata.ndim > 3:
                maskdata = maskdata[:, :, :, sub_i]

            msg = f"Mask = {self.mask_file}"
            if ma:
                msg += f"[{sub_i}]"
            self._logger.info(msg)

        if method == 'zero_out':
            self.maskV = np.squeeze(maskdata) != 0

        # Make the byte mask in the memory order of the image passed to the C
        # function (x fastest)
        self.mask_byte = np.ascontiguousarray(
            self.maskV.astype('u2').transpose(2, 1, 0))
        self.mask_byte_p = self.mask_byte.ctypes.data_as(
                ctypes.POINTER(ctypes.c_ushort))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def smooth(self, fmri_img):
        """
        Run spatial smoothing with the rtp_smooth function in librtp.so.

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
            Smoothed image data.
        """
        # Image dimensions and voxel size
        nx, ny, nz = fmri_img.shape[:3]
        if hasattr(fmri_img.header, 'info'):
            # BRIK
            dx, dy, dz = np.abs(fmri_img.header.info['DELTA'])
        elif hasattr(fmri_img.header, 'get_zooms'):
            # NIfTI
            dx, dy, dz = fmri_img.header.get_zooms()[:3]
        else:
            errmsg = "No voxel size information in the fmri_img header"
            self._logger.error(errmsg)
            self.err_popup(errmsg)

        if self.maskV.shape != (nx, ny, nz):
            # The C function reads nx*ny*nz mask values
            raise ValueError(
                f"Mask shape {self.maskV.shape} does not match the image"
                f" {(nx, ny, nz)}")

        # Copy the functional image data and get its pointer
        fim_arr = np.asarray(fmri_img.dataobj, dtype=np.float32)
        if fim_arr.ndim > 3:
            fim_arr = np.squeeze(fim_arr)
        # x,y,z -> z,y,x (C order with x fastest)
        fim_arr = fim_arr.transpose(2, 1, 0).copy()
        fim_p = fim_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        ret = self.rtp_smooth(fim_p, nx, ny, nz, dx, dy, dz,
                              self.mask_byte_p, self.blur_fwhm)
        if ret is not None and ret < 0:
            raise RuntimeError("rtp_smooth failed (check blur_fwhm and the"
                               " voxel size)")
        fim_arr = fim_arr.transpose(2, 1, 0)  # z,y,x -> x,y,z

        return fim_arr

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_param(self, attr, val=None, reset_fn=None, echo=False):
        """
        Set a parameter and update the GUI.

        When reset_fn is None, set_param is assumed to be called from the
        load_parameters function.
        """
        self._logger.debug(f"set_param: {attr} = {val}")

        # -- Check the value --
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

        elif attr == 'blur_fwhm' and reset_fn is None:
            if hasattr(self, 'ui_FWHM_dSpBx'):
                self.ui_FWHM_dSpBx.setValue(val)

        elif attr == 'mask_file':
            if isinstance(val, Path):
                val = str(val)
                self.set_mask(val)
                if self.mask_file == 0:
                    # set_mask failed and has reset mask_file to 0
                    return
                if hasattr(self, 'ui_mask_cmbBx'):
                    self.ui_mask_cmbBx.setCurrentText('external file')
                    self.ui_mask_lnEd.setText(str(val))

            elif type(val) is int and val == 0:
                if hasattr(self, 'ui_mask_lnEd'):
                    self.ui_mask_lnEd.setText(self._ZERO_OUT_MASK_TEXT)
                self.mask_byte = None

            elif val in ('initial volume of internal run', 'internal') \
                    or val.startswith('zero-out'):
                val = 0
                if hasattr(self, 'ui_mask_lnEd'):
                    self.ui_mask_lnEd.setText(self._ZERO_OUT_MASK_TEXT)
                self.mask_byte = None

            elif val == 'external file':
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
                            None, "SMOOTH: Select sub volume",
                            "sub-volume index (0 is first)", 0, 0,
                            mask_img.shape[3] - 1)
                    if not okflag:
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
                    # 'external file'
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
                errmsg = f"{attr} is not a recognized parameter."
                self._logger.error(errmsg)
                self.err_popup(errmsg)
                return

        # -- Set the value --
        setattr(self, attr, val)
        if echo:
            print(f"{self.__class__.__name__}." + attr, '=',
                  getattr(self, attr))

        return 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ui_set_param(self):

        ui_rows = []
        self.ui_objs = []

        # Enabled
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
            self.ui_mask_lnEd.setText(self._ZERO_OUT_MASK_TEXT)
        else:
            self.ui_mask_cmbBx.setCurrentIndex(0)
            self.ui_mask_lnEd.setText(str(self.mask_file))

        # --- Checkbox row ----------------------------------------------------
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
    test_dir = Path(__file__).absolute().parent.parent / 'tests'

    # Load the test data
    testdata_f = test_dir / 'func_epi.nii.gz'
    assert testdata_f.is_file()
    img = nib.load(testdata_f)
    img_data = np.asanyarray(img.dataobj)

    work_dir = test_dir / 'work'
    if not work_dir.is_dir():
        work_dir.mkdir()

    # Create RtpTshift, RtpVolreg, and RtpSmooth instances
    from rtpspy.rtp_tshift import RtpTshift
    from rtpspy.rtp_volreg import RtpVolreg

    rtp_tshift = RtpTshift()
    rtp_tshift.method = 'cubic'
    rtp_tshift.ignore_init = 3
    rtp_tshift.ref_time = 0

    rtp_volreg = RtpVolreg(regmode='cubic')

    rtp_smooth = RtpSmooth(blur_fwhm=6)

    # Set the slice timing from sample data
    rtp_tshift.set_from_sample(testdata_f)

    # Set the reference volume
    refname = str(testdata_f) + '[0]'
    rtp_volreg.set_ref_vol(refname)

    # Make the mask from the first received volume
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
        rtp_tshift.do_proc(fmri_img, ii, st)  # tshift -> volreg -> smooth

    proc_delay = rtp_smooth.proc_delay
    rtp_tshift.end_reset()

    # --- Plot ---
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(proc_delay[1:], bins='auto')
    print(f"Median processing delay: {np.median(proc_delay[1:]):.4f} s")
    plt.show()
