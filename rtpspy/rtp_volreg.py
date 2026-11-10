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
import traceback
import logging

import numpy as np
import nibabel as nib

from PyQt5 import QtWidgets, QtCore
import matplotlib as mpl

try:
    from .rtp_common import RTP, MatplotlibWindow
except Exception:
    from rtpspy.rtp_common import RTP, MatplotlibWindow

mpl.rcParams['font.size'] = 8

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
    def do_proc(self, fmri_img, vol_idx=None, pre_proc_time=None, **kwargs):
        try:
            self._vol_num += 1  # 1-based number of received volumes
            if vol_idx is None:
                vol_idx = self._vol_num - 1  # 0-based index

            if vol_idx < self.ignore_init:
                return

            if self._proc_start_idx < 0:
                self._proc_start_idx = vol_idx

            # --- Initialize --------------------------------------------------
            # If self.ref_vol is an int (index of the reference volume in the
            # current scan), wait for that volume and set it as the
            # reference.
            if type(self.ref_vol) is int:
                if vol_idx < self.ref_vol:
                    # No reference yet. The volume cannot be aligned.
                    self._logger.warning(
                        f"Volume {vol_idx} is before the reference volume "
                        f"{self.ref_vol} and is not processed.")
                    return

                ref_vi = self.ref_vol
                self.ref_vol = nib.Nifti1Image(
                    np.asarray(fmri_img.dataobj, dtype=np.float32).reshape(
                        fmri_img.shape[:3]), fmri_img.affine)
                self.align_setup()
                self._logger.info(
                    f"Alignment reference is set to volume {ref_vi}"
                    " of the current scan.")
                # The reference volume itself is aligned (motion ~ 0) and
                # passed to the next process below.

            if vol_idx >= self._motion.shape[0]:
                self._logger.warning(
                    "The scan is longer than max_scan_length "
                    f"({self.max_scan_length}). The motion array is extended.")
                self._motion = np.concatenate(
                    [self._motion,
                     np.zeros([max(100, vol_idx + 1 - self._motion.shape[0]),
                               6], dtype=np.float32)])

            # --- Run the process ---------------------------------------------
            aligned_data, motpar = self.align_one(fmri_img, vol_idx)

            fmri_img.uncache()
            fmri_img._dataobj = aligned_data
            fmri_img.set_data_dtype(aligned_data.dtype)

            # (roll, pitch, yaw, dx, dy, dz) -> AFNI 3dvolreg output order
            self._motion[vol_idx, :] = motpar[[0, 1, 2, 5, 3, 4]]

            # --- Post-process ------------------------------------------------
            # Record the processing time and log
            tstamp = time.time()
            self._proc_time.append(tstamp)

            if fmri_img.get_filename():
                fname = Path(fmri_img.get_filename()).name
            else:
                fname = "unknown.nii.gz"
            msg = f"#{vol_idx+1};Volume registration;{fname};tstamp={tstamp}"
            if pre_proc_time is not None:
                proc_delay = tstamp - pre_proc_time
                if self.save_delay:
                    self.proc_delay.append(proc_delay)
                msg += f";took {proc_delay:.4f}s"
            self._logger.info(msg)

            fmri_img.set_filename('vr.' + fname)

            if self.next_proc is not None:
                # Keep this module's output, as the next process replaces
                # the data in fmri_img.
                self.proc_data = np.asanyarray(fmri_img.dataobj)
                save_name = fmri_img.get_filename()

                self.next_proc.do_proc(fmri_img, vol_idx=vol_idx,
                                       pre_proc_time=tstamp)

            # Update the motion plot
            self._plt_xi.append(vol_idx+1)
            for ii in range(6):
                self._plt_motion[ii].append(self._motion[vol_idx, ii])

            if self.save_proc:
                if self.next_proc is not None:
                    # Restore this module's output
                    fmri_img.uncache()
                    fmri_img._dataobj = self.proc_data
                    fmri_img.set_data_dtype(self.proc_data.dtype)
                    fmri_img.set_filename(save_name)

                self.keep_processed_image(fmri_img, vol_num=vol_idx,
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

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def open_motion_plot(self):
        if hasattr(self, 'thPltMotion') and self.thPltMotion.isRunning():
            return

        self.thPltMotion = QtCore.QThread()
        self.pltMotion = RtpVolreg.PlotMotion(self, main_win=self.main_win)
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

        if self.main_win is not None:
            if self.main_win.chbShowMotion.checkState() != 0:
                self.main_win.chbShowMotion.setCheckState(0)

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

        elif attr == 'ignore_init' and reset_fn is None:
            if hasattr(self, 'ui_ignorInit_spBx'):
                self.ui_ignorInit_spBx.setValue(val)

        elif attr == 'ref_vol':
            if isinstance(val, Path):
                val = str(val)

            if type(val) is int:
                self.set_ref_vol(val)
                if hasattr(self, 'ui_baseVol_lnEd'):
                    self.ui_baseVol_lnEd.setText(
                        f"Internal run volume index {val}")

            elif isinstance(val, nib.filebasedimages.SerializableImage):
                setattr(self, attr, val)
                if hasattr(self, 'ui_baseVol_cmbBx'):
                    self.ui_baseVol_cmbBx.setCurrentIndex(0)

            elif val == 'index of internal run':
                num, okflag = QtWidgets.QInputDialog.getInt(
                        None, "Internal base volume index",
                        "volume index (0 is first)")
                if not okflag:
                    return

                self.set_ref_vol(num)
                if hasattr(self, 'ui_baseVol_lnEd'):
                    self.ui_baseVol_lnEd.setText(
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
                    num, okflag = QtWidgets.QInputDialog.getInt(
                            None, "VOLREG: Select sub volume",
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
                if hasattr(self, 'ui_baseVol_lnEd'):
                    self.ui_baseVol_lnEd.setText(str(ref_fname))

            elif type(val) is str:
                fname, num = self._split_vol_index(val)
                if not Path(fname).is_file():
                    return

                if reset_fn is None and hasattr(self, 'ui_baseVol_cmbBx'):
                    self.ui_baseVol_cmbBx.setCurrentIndex(0)

                # set_ref_vol also sets self.ref_fname
                self.set_ref_vol(fname, num)
                if hasattr(self, 'ui_baseVol_lnEd'):
                    self.ui_baseVol_lnEd.setText(str(self.ref_fname))

            return

        elif attr == 'ref_fname':
            if len(val):
                val, num = self._split_vol_index(val)
                if not Path(val).is_file():
                    return

                # set_ref_vol sets self.ref_fname with the volume index
                self.set_ref_vol(val, num)
                if hasattr(self, 'ui_baseVol_lnEd'):
                    self.ui_baseVol_lnEd.setText(str(self.ref_fname))
                return

        elif attr == 'regmode':
            val = self._check_regmode(val)
            if reset_fn is None and hasattr(self, 'ui_regmode_cmbBx'):
                self.ui_regmode_cmbBx.setCurrentIndex(
                    RtpVolreg.reg_regmodes.index(val))

        elif attr == 'max_scan_length':
            val = int(val)
            if val > self._motion.shape[0]:
                # Extend the motion array, keeping the values
                self._motion = np.concatenate(
                    [self._motion,
                     np.zeros([val - self._motion.shape[0], 6],
                              dtype=np.float32)])

            if reset_fn is None:
                if hasattr(self, 'ui_maxLen_spBx'):
                    self.ui_maxLen_spBx.setValue(val)

        elif attr == 'save_proc':
            if hasattr(self, 'ui_saveProc_chb'):
                self.ui_saveProc_chb.setChecked(val)

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

        # ref_vol
        var_lb = QtWidgets.QLabel("Base volume :")
        self.ui_baseVol_cmbBx = QtWidgets.QComboBox()
        self.ui_baseVol_cmbBx.addItems(['external file',
                                        'index of internal run'])
        self.ui_baseVol_cmbBx.activated.connect(
                lambda idx:
                self.set_param('ref_vol',
                               self.ui_baseVol_cmbBx.currentText(),
                               self.ui_baseVol_cmbBx.setCurrentIndex))
        ui_rows.append((var_lb, self.ui_baseVol_cmbBx))

        self.ui_baseVol_lnEd = QtWidgets.QLineEdit()
        self.ui_baseVol_lnEd.setReadOnly(True)
        self.ui_baseVol_lnEd.setStyleSheet(
            'border: 0px none;')
        ui_rows.append((None, self.ui_baseVol_lnEd))

        self.ui_objs.extend([var_lb, self.ui_baseVol_cmbBx,
                             self.ui_baseVol_lnEd])

        if self.ref_fname:
            self.ui_baseVol_cmbBx.setCurrentIndex(0)
            self.ui_baseVol_lnEd.setText(str(self.ref_fname))
        elif not isinstance(self.ref_vol, int):
            self.ui_baseVol_cmbBx.setCurrentIndex(1)
            self.ui_baseVol_lnEd.setText("Set from the internal run")
        else:
            self.ui_baseVol_cmbBx.setCurrentIndex(1)
            self.ui_baseVol_lnEd.setText(
                    f"Internal run volume index {self.ref_vol}")

        # regmode
        var_lb = QtWidgets.QLabel("Resampling :")
        self.ui_regmode_cmbBx = QtWidgets.QComboBox()
        self.ui_regmode_cmbBx.addItems(
            [m.capitalize() for m in RtpVolreg.reg_regmodes])
        self.ui_regmode_cmbBx.setCurrentIndex(
            RtpVolreg.reg_regmodes.index(self.regmode))
        self.ui_regmode_cmbBx.currentIndexChanged.connect(
                lambda idx:
                self.set_param('regmode', RtpVolreg.reg_regmodes[idx],
                               self.ui_regmode_cmbBx.setCurrentIndex))
        ui_rows.append((var_lb, self.ui_regmode_cmbBx))
        self.ui_objs.extend([var_lb, self.ui_regmode_cmbBx])

        # max_scan_length
        var_lb = QtWidgets.QLabel("Maximum scan length :")
        self.ui_maxLen_spBx = QtWidgets.QSpinBox()
        self.ui_maxLen_spBx.setMinimum(1)
        self.ui_maxLen_spBx.setMaximum(9999)
        self.ui_maxLen_spBx.setValue(self.max_scan_length)
        self.ui_maxLen_spBx.editingFinished.connect(
                lambda: self.set_param('max_scan_length',
                                       self.ui_maxLen_spBx.value(),
                                       self.ui_maxLen_spBx.setValue))
        ui_rows.append((var_lb, self.ui_maxLen_spBx))
        self.ui_objs.extend([var_lb, self.ui_maxLen_spBx])

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
