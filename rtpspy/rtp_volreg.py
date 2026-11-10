#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
"""
RTP volume registration for motion correction

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import sys
import time
import re
import ctypes

import numpy as np
import nibabel as nib

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


# %% RtpVolreg ===============================================================
class RtpVolreg(RTP):
    """
    Real-time online volume registration for motion correction.
    The AFNI functions in librtp.so (built from the AFNI libmri source) are
    called via ctypes.
    """

    # Resampling method IDs of the C library
    regmode_dict = {'nn': 0, 'linear': 1, 'cubic': 2, 'fourier': 3,
                    'quintic': 4, 'heptic': 5, 'tsshift': 6}
    # Resampling methods usable for registration. 'nn' cannot be used: the
    # gradient images are all zero and the least-squares fit fails in the C
    # library (the process aborts).
    reg_regmodes = ['linear', 'cubic', 'fourier', 'quintic', 'heptic']

    # Log label and output filename prefix (see RTP.do_proc)
    _proc_label = 'Volume registration'
    _out_prefix = 'vr.'

    # GUI panel (rtpspy.ui.volreg_ui)
    _ui_class = "volreg_ui:RtpVolregUI"

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, ref_vol=0, regmode='heptic', max_scan_length=None,
                 **kwargs):
        """
        Parameters
        ----------
        ref_vol : int, str, or Path, optional
            Reference volume.
            int: Index of the volume in the current scan.
            str or Path: Reference volume filename. A sub-volume can be
            selected with an AFNI-style index, e.g., 'func.nii.gz[3]'.
            The default is 0.
        regmode : str, optional
            Image resampling method.
            ['linear'|'cubic'|'fourier'|'quintic'|'heptic']
            The default is 'heptic'. See
            https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dvolreg.html
            for details.
        max_scan_length : int, optional
            Maximum scan length, used to pre-allocate the motion parameter
            array. The default is DEFAULT_MAX_SCAN_LENGTH.
        """
        super(RtpVolreg, self).__init__(max_scan_length=max_scan_length,
                                        **kwargs)

        # Set instance parameters
        self.regmode = self._check_regmode(regmode)
        self.ref_fname = ''  # External reference file ('' for internal)

        # Motion parameters of all volumes (AFNI 3dvolreg output order)
        self._motion = np.zeros([self.max_scan_length, 6], dtype=np.float32)
        # Motion parameters of the last aligned volume in the order of the C
        # function (roll, pitch, yaw, dx, dy, dz), used as the initial values
        # for the next volume.
        self._last_motpar = None
        self.setup_libfuncs()
        self.set_ref_vol(ref_vol)

        # Alignment parameters (AFNI plug_realtime default values)
        self.max_iter = 9
        self.dxy_thresh = 0.05  # pixels
        self.phi_thresh = 0.07  # degree

        # Data for the motion plot
        self._plt_xi = []
        self._plt_motion = [[] for _ in range(6)]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ready_proc(self):
        self._proc_ready = self.ref_vol is not None
        if not self._proc_ready:
            errmsg = "Reference volume or volume index has not been set."
            self._logger.error(errmsg)
            self.err_popup(errmsg)

        # Reset running variables
        self._motion = np.zeros([self.max_scan_length, 6], dtype=np.float32)
        self._last_motpar = None

        # Reset plot values
        self._plt_xi[:] = []
        for plt_mot in self._plt_motion:
            plt_mot[:] = []

        if self.next_proc:
            self._proc_ready &= self.next_proc.ready_proc()

        return self._proc_ready

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _process(self, fmri_img, vol_idx, pre_proc_time):
        """Align the volume to the reference (see RTP.do_proc)."""
        # --- Initialize ------------------------------------------------------
        # If self.ref_vol is an int (index of the reference volume in the
        # current scan), wait for that volume and set it as the reference.
        if type(self.ref_vol) is int:
            if vol_idx < self.ref_vol:
                # No reference yet. The volume cannot be aligned.
                self._logger.warning(
                    f"Volume {vol_idx} is before the reference volume "
                    f"{self.ref_vol} and is not processed.")
                return None

            ref_vi = self.ref_vol
            self.ref_vol = nib.Nifti1Image(
                np.asarray(fmri_img.dataobj, dtype=np.float32).reshape(
                    fmri_img.shape[:3]), fmri_img.affine)
            self.align_setup()
            self._logger.info(
                f"Alignment reference is set to volume {ref_vi}"
                " of the current scan.")
            # The reference volume itself is aligned (motion ~ 0) and
            # passed to the next process.

        if vol_idx >= self._motion.shape[0]:
            self._logger.warning(
                "The scan is longer than max_scan_length "
                f"({self.max_scan_length}). The motion array is extended.")
            self._motion = np.concatenate(
                [self._motion,
                 np.zeros([max(100, vol_idx + 1 - self._motion.shape[0]),
                           6], dtype=np.float32)])

        # --- Run the process -------------------------------------------------
        aligned_data, motpar = self.align_one(fmri_img, vol_idx)
        out_img = fmri_img.with_data(aligned_data)

        # (roll, pitch, yaw, dx, dy, dz) -> AFNI 3dvolreg output order
        self._motion[vol_idx, :] = motpar[[0, 1, 2, 5, 3, 4]]

        # Update the motion plot
        self._plt_xi.append(vol_idx+1)
        for ii in range(6):
            self._plt_motion[ii].append(self._motion[vol_idx, ii])

        return out_img

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end_reset(self):
        """ End process and reset process parameters. """

        self._logger.info(f"Reset {self.__class__.__name__} module.")
        return super(RtpVolreg, self).end_reset()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _check_regmode(self, regmode):
        """Return a resampling method name usable for registration."""
        if isinstance(regmode, (int, np.integer)):
            # Legacy index of regmode_dict
            names = {v: k for k, v in RtpVolreg.regmode_dict.items()}
            regmode = names.get(int(regmode), regmode)
        if regmode not in RtpVolreg.reg_regmodes:
            self._logger.error(
                f"Resampling '{regmode}' cannot be used for registration."
                " 'heptic' is used.")
            regmode = 'heptic'
        return regmode

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def setup_libfuncs(self):
        """ Set up the C library functions. """

        librtp = ctypes.cdll.LoadLibrary(librtp_path)

        # -- rtp_align_setup --
        self.rtp_align_setup = librtp.rtp_align_setup
        self.rtp_align_setup.argtypes = \
            [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
             ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int,
             ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
             ctypes.c_void_p, ctypes.c_void_p]

        # -- rtp_align_one --
        self.rtp_align_one = librtp.rtp_align_one
        self.rtp_align_one.argtypes = \
            [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
             ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_void_p,
             ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
             ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
             ctypes.c_void_p]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_ref_vol(self, ref_vol, ref_vi=0):
        """
        Set the alignment reference.

        Parameters
        ----------
        ref_vol : int, str, or Path
            int: Index of the reference volume in the current scan.
            str or Path: Reference image filename. An AFNI-style volume
            index (e.g., 'func.nii.gz[3]') overrides ref_vi.
        ref_vi : int, optional
            Volume index in a 4D reference file. The default is 0.
        """
        if isinstance(ref_vol, (str, Path)):
            ref_vol0 = ref_vol

            # Get the volume index
            ma = re.search(r"\'*\[(\d+)\]\'*$", str(ref_vol))
            if ma:
                ref_vi = int(ma.groups()[0])
                ref_vol = re.sub(r"\'*\[(\d+)\]\'*$", '', str(ref_vol))

            ref_vol = nib.load(ref_vol)
            if len(ref_vol.shape) == 4:
                vol = np.asarray(ref_vol.dataobj[:, :, :, ref_vi],
                                 dtype=np.float32)
                ref_vol = nib.Nifti1Image(vol, ref_vol.affine)
            assert len(ref_vol.shape) == 3

            self.ref_vol = ref_vol
            ref_fname = re.sub(r"\'*\[.*$", '', str(ref_vol0))
            self.ref_fname = f"{ref_fname}[{ref_vi}]"
            self.align_setup()

            msg = f"Alignment reference = {ref_vol0}"
            if ma is None:
                msg += f"[{ref_vi}]"
            self._logger.info(msg)

        else:
            # ref_vol is a volume index. The reference is taken from that
            # volume of the ongoing scan in do_proc, and then kept for the
            # lifetime of the instance.
            self.ref_vol = int(ref_vol)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def align_setup(self):
        """
        Set up the reference image array and the Cholesky-decomposed matrix
        for the least-squares fit.

        The reference image array includes the base image, three rotation
        gradient images, and three shift gradient images.

        Calls rtp_align_setup in librtp.so, a modified version of
        mri_3dalign_setup in mri_3dalign.c of the AFNI source.

        Set variables
        -------------
        self.fitim: numpy array of float32
            Seven reference volumes.
        self.chol_fitim: numpy array of float64
            Cholesky decomposition of the weighted covariance matrix of the
            reference images.

        C function definition
        ---------------------
        int rtp_align_setup(float *base_im, int nx, int ny, int nz,
                            float dx, float dy, float dz,
                            int ax1, int ax2, int ax3, int regmode,
                            int nref, float *ref_ims, double *chol_fitim)
        """

        # Image dimensions and voxel size
        self.nx, self.ny, self.nz = self.ref_vol.shape
        if hasattr(self.ref_vol.header, 'info'):
            self.dx, self.dy, self.dz = \
                np.abs(self.ref_vol.header.info['DELTA'])
        elif hasattr(self.ref_vol.header, 'get_zooms'):
            self.dx, self.dy, self.dz = self.ref_vol.header.get_zooms()[:3]
        else:
            errmsg = "No voxel size information in the ref_vol header"
            self._logger.error(errmsg)

        # Rotation axes
        self.ax1 = 2  # z-axis, roll
        self.ax2 = 0  # x-axis, pitch
        self.ax3 = 1  # y-axis, yaw

        # Copy the base image data and get its pointer
        base_im_arr = np.asarray(self.ref_vol.dataobj, dtype=np.float32)
        # x,y,z -> z,y,x (C order with x fastest)
        base_im_arr = base_im_arr.reshape(
            self.ref_vol.shape[:3]).transpose(2, 1, 0).copy()
        base_im_p = base_im_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # -- Prepare the output arrays --
        # Reference image array
        nxyz = self.nx * self.ny * self.nz
        nref = 7
        ref_img_arr = np.ndarray(nref * nxyz, dtype=np.float32)
        ref_ims_p = ref_img_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Cholesky-decomposed matrix for the least-squares fit
        chol_fitim_arr = np.ndarray((nref, nref), dtype=np.float64)
        chol_fitim_p = chol_fitim_arr.ctypes.data_as(
                ctypes.POINTER(ctypes.c_double))

        regmode_id = RtpVolreg.regmode_dict[self.regmode]
        self.rtp_align_setup(base_im_p, self.nx, self.ny, self.nz, self.dx,
                             self.dy, self.dz, self.ax1, self.ax2, self.ax3,
                             regmode_id, nref, ref_ims_p, chol_fitim_p)

        self.fitim = ref_img_arr
        self.chol_fitim = chol_fitim_arr
        self._last_motpar = None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def align_one(self, fmri_img, vol_idx):
        """
        Align a single-volume fmri_img to the reference image.

        Calls rtp_align_one in librtp.so, a modified version of
        mri_3dalign_one in mri_3dalign.c of the AFNI source.

        C function definition
        ---------------------
        int rtp_align_one(float *fim, int nx, int ny, int nz, float dx,
                          float dy, float dz, float *fitim, double *chol_fitim,
                          int nref, int ax1, int ax2, int ax3,
                          float *init_motpar, int regmode, float *tim,
                          float *motpar)

        Parameters
        ----------
        fmri_img : nibabel image object
            Single-volume fMRI image to be aligned.
        vol_idx : int
            Volume index (not used).

        Returns
        -------
        tim : 3D array of float32
            Aligned volume data.
        motpar : array
            Six motion parameters: roll, pitch, yaw, dx, dy, dz.
        """

        # Copy the functional image data and get its pointer
        fim_arr = np.asarray(fmri_img.dataobj, dtype=np.float32)
        if fim_arr.ndim > 3:
            fim_arr = np.squeeze(fim_arr)
        if fim_arr.shape != (self.nx, self.ny, self.nz):
            # The C function reads nx*ny*nz values; a different shape would
            # read out of bounds.
            raise ValueError(
                f"Volume shape {fim_arr.shape} does not match the reference"
                f" {(self.nx, self.ny, self.nz)}")
        # x,y,z -> z,y,x (C order with x fastest)
        fim_arr = fim_arr.transpose(2, 1, 0).copy()
        fim_p = fim_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Pointers to the reference arrays
        fitim_p = self.fitim.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        chol_fitim_p = self.chol_fitim.ctypes.data_as(
                ctypes.POINTER(ctypes.c_double))

        # Initial motion parameters: those of the last aligned volume, in
        # the order of the C function (roll, pitch, yaw, dx, dy, dz).
        # (self._motion has the AFNI 3dvolreg output order.)
        if self._last_motpar is not None:
            init_motpar = self._last_motpar.copy()
        else:
            init_motpar = np.zeros(6, dtype=np.float32)

        init_motpar_p = init_motpar.ctypes.data_as(
                ctypes.POINTER(ctypes.c_float))

        # Prepare the output arrays. motpar[0] is the intensity scale factor
        # and motpar[1:] are the six motion parameters.
        nxyz = self.nx * self.ny * self.nz
        tim_arr = np.ndarray(nxyz, dtype=np.float32)
        tim_p = tim_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        motpar = np.ndarray(7, dtype=np.float32)
        motpar_p = motpar.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        regmode_id = RtpVolreg.regmode_dict[self.regmode]
        self.rtp_align_one(fim_p, self.nx, self.ny, self.nz, self.dx, self.dy,
                           self.dz, fitim_p, chol_fitim_p, 7, self.ax1,
                           self.ax2, self.ax3, init_motpar_p, regmode_id,
                           tim_p, motpar_p)

        # z,y,x -> x,y,z
        tim = np.reshape(tim_arr, (self.nz, self.ny, self.nx))
        tim = tim.transpose(2, 1, 0)
        self._last_motpar = motpar[1:].astype(np.float32)

        return tim, motpar[1:]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def open_motion_plot(self):
        """Open the motion plot window (only in the GUI)."""
        self._ui_call('open_motion_plot')

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def close_motion_plot(self):
        """Close the motion plot window (only in the GUI)."""
        self._ui_call('close_motion_plot')

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
            self._ui_update(attr, val)

        elif attr == 'work_dir':
            if val is None or not Path(val).is_dir():
                return

            val = Path(val)
            setattr(self, attr, val)

            if self.main_win is not None:
                self.main_win.set_workDir(val)

        elif attr == 'ignore_init' and reset_fn is None:
            self._ui_update(attr, val)

        elif attr == 'ref_vol':
            if isinstance(val, Path):
                val = str(val)

            if type(val) is int:
                self.set_ref_vol(val)
                self._ui_update('ref_vol_text',
                                f"Internal run volume index {val}")

            elif isinstance(val, nib.filebasedimages.SerializableImage):
                setattr(self, attr, val)
                self._ui_update('ref_vol_source', 0)

            elif val == 'index of internal run':
                num, okflag = self._notifier().get_int(
                        "Internal base volume index",
                        "volume index (0 is first)")
                if not okflag:
                    return

                self.set_ref_vol(num)
                self._ui_update('ref_vol_text',
                                f"Internal run volume index {num}")
                self.ref_fname = ''

            elif val == 'external file':
                fname = self.select_file_dlg('VOLREG: Select base volume',
                                             self.work_dir, "*.BRIK* *.nii*")
                if fname[0] == '':
                    if reset_fn:
                        reset_fn(1)
                    return -1

                ref_img = nib.load(fname[0])
                ref_fname = fname[0]
                if len(ref_img.shape) > 3 and ref_img.shape[3] > 1:
                    num, okflag = self._notifier().get_int(
                            "VOLREG: Select sub volume",
                            "sub-volume index (0 is first)", 0, 0,
                            ref_img.shape[3] - 1)
                    if not okflag:
                        if reset_fn:
                            reset_fn(1)
                        return

                    ref_fname += f"[{num}]"
                else:
                    num = 0

                # set_ref_vol also sets self.ref_fname
                self.set_ref_vol(fname[0], num)
                self._ui_update('ref_vol_text', ref_fname)

            elif type(val) is str:
                fname, num = self._split_vol_index(val)
                if not Path(fname).is_file():
                    return

                if reset_fn is None:
                    self._ui_update('ref_vol_source', 0)

                # set_ref_vol also sets self.ref_fname
                self.set_ref_vol(fname, num)
                self._ui_update('ref_vol_text', self.ref_fname)

            return

        elif attr == 'ref_fname':
            if len(val):
                val, num = self._split_vol_index(val)
                if not Path(val).is_file():
                    return

                # set_ref_vol sets self.ref_fname with the volume index
                self.set_ref_vol(val, num)
                self._ui_update('ref_vol_text', self.ref_fname)
                return

        elif attr == 'regmode':
            val = self._check_regmode(val)
            if reset_fn is None:
                self._ui_update(attr, val)

        elif attr == 'max_scan_length':
            val = int(val)
            if val > self._motion.shape[0]:
                # Extend the motion array, keeping the values
                self._motion = np.concatenate(
                    [self._motion,
                     np.zeros([val - self._motion.shape[0], 6],
                              dtype=np.float32)])

            if reset_fn is None:
                self._ui_update(attr, val)

        elif attr == 'save_proc':
            self._ui_update(attr, val)

        elif reset_fn is None:
            # Ignore an unrecognized parameter
            if not hasattr(self, attr):
                errmsg = f"{attr} is not a recognized parameter."
                self._logger.error(errmsg)
                return

        # -- Set the value --
        setattr(self, attr, val)
        if echo:
            print(f"{self.__class__.__name__}." + attr, '=',
                  getattr(self, attr))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def _split_vol_index(fname):
        """Split 'file[N]' into ('file', N). N is 0 without an index."""
        ma = re.search(r"\[(\d+)\]", fname)
        if ma:
            return re.sub(r"\[(\d+)\]", '', fname), int(ma.groups()[0])
        return fname, 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_params(self):
        all_opts = super().get_params()
        # The reference and its derived values are restored from ref_fname.
        excld_opts = ('work_dir', 'chol_fitim', 'ref_vol', 'nx', 'ny', 'nz',
                      'fitim', 'dx', 'dy', 'dz', 'ax1', 'ax2', 'ax3')
        sel_opts = {}
        for k, v in all_opts.items():
            if k in excld_opts or k[0] == '_':
                continue
            if isinstance(v, Path):
                v = str(v)
            sel_opts[k] = v

        return sel_opts

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        self.close_motion_plot()


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

    # Create RtpTshift and RtpVolreg instances
    from rtpspy.rtp_tshift import RtpTshift

    rtp_tshift = RtpTshift()
    rtp_tshift.method = 'cubic'
    rtp_tshift.ignore_init = 3
    rtp_tshift.ref_time = 0

    rtp_volreg = RtpVolreg(regmode='cubic')

    # Set the slice timing from sample data
    rtp_tshift.set_from_sample(testdata_f)

    # Set the reference volume
    refname = str(testdata_f) + '[0]'
    rtp_volreg.set_ref_vol(refname)

    # Set parameters for debug
    rtp_tshift.work_dir = work_dir
    rtp_tshift.save_proc = True
    rtp_volreg.work_dir = work_dir
    rtp_volreg.save_proc = True
    rtp_volreg.save_delay = True

    # Chain tshift -> volreg
    rtp_tshift.next_proc = rtp_volreg
    rtp_tshift.end_reset()

    # Run
    N_vols = img.shape[-1]
    for ii in range(N_vols):
        save_filename = f"test_nr_{ii:04d}.nii.gz"
        fmri_img = nib.Nifti1Image(img_data[:, :, :, ii], affine=img.affine)
        fmri_img.set_filename(save_filename)
        st = time.time()
        rtp_tshift.do_proc(fmri_img, ii, st)  # run rtp_tshift -> rtp_volreg

    # rtp_volreg tends to estimate smaller volume-to-volume differences than
    # 3dvolreg, because rtp_volreg starts the estimation from the previous
    # motion parameters for faster convergence, while 3dvolreg starts from 0.

    proc_delay = rtp_volreg.proc_delay
    motion = rtp_volreg._motion[rtp_volreg._proc_start_idx:N_vols, :]
    rtp_tshift.end_reset()

    mot_f = rtp_volreg.saved_filename.name.replace('.nii.gz', '')
    mot_f = rtp_volreg.saved_filename.parent / ('motion_' + mot_f + '.1D')
    np.savetxt(mot_f, motion)

    # --- Plot ---
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(proc_delay[1:], bins='auto')
    print(f"Median processing delay: {np.median(proc_delay[1:]):.4f} s")
    plt.show()
