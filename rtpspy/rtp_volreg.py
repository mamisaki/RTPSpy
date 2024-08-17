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
from six import string_types

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
    AFNI functions in librtp.so (compiled from libmri.so) is called.
    """

    regmode_dict = {'nn': 0, 'linear': 1, 'cubic': 2, 'fourier': 3,
                    'quintic': 4, 'heptic': 5, 'tsshift': 6}

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, ref_vol=0, regmode='heptic', max_scan_length=1000,
                 **kwargs):
        """
        Parameters
        ----------
        ref_vol : int or str, optional
            Reference volume filename or index the current scan.
            int: Index of volume in the current scan.
            str (or Path): reference volume filename.
            The default is 0.
        regmode : str, optional
            Image resampling method.
            ['nn'|'linear'|'cubic'| 'fourier'|'quintic'|heptic'|'tsshift']
            The default is 'heptic'. See
            https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dvolreg.html
            for details.
        max_scan_length : int, optional
            Maximum scan length. This is used for pre-allocating memory space
            for motion matrix. The default is 1000.
        """
        super(RtpVolreg, self).__init__(**kwargs)

        # Set instance parameters
        self.regmode = regmode
        self.max_scan_length = max_scan_length

        # Initialize parameters and C library function call
        # motion parameter
        self._motion = np.zeros([self.max_scan_length, 6], dtype=np.float32)
        self.set_ref_vol(ref_vol)
        self.setup_libfuncs()

        # alignment parameters (Set from AFNI plug_realtime default values)
        self.max_iter = 9
        self.dxy_thresh = 0.05  # pixels
        self.phi_thresh = 0.07  # degree

        # --- initialize for motion plot ---
        self._plt_xi = []
        self._plt_motion = []
        for ii in range(6):
            self._plt_motion.append(list([]))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ready_proc(self):
        self._proc_ready = self.ref_vol is not None
        if not self._proc_ready:
            errmsg = "Refence volume or volume index has not been set."
            self._logger.error(errmsg)
            self.err_popup(errmsg)

        # Reset running variables
        self._motion = np.zeros([self.max_scan_length, 6], dtype=np.float32)

        # Reset plot values
        self._plt_xi[:] = []
        for ii in range(6):
            self._plt_motion[ii][:] = []

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
            # if self.ref_vol is int (index of reference volume in the current
            # scan), wait for the reference index and set the reference.
            if type(self.ref_vol) is int:
                if vol_idx < self.ref_vol:
                    return

                elif vol_idx >= self.ref_vol:
                    ref_vi = self.ref_vol
                    self.ref_vol = fmri_img
                    self.align_setup()
                    msg = f"Alignment reference is set to volume {ref_vi}"
                    msg += " of current sequence."
                    self._logger.info(msg)
                    return

            # --- Run the procress --------------------------------------------
            # Perform volume alignment
            reg_dataV, mot = self.align_one(fmri_img, vol_idx)

            # Set aligned data in fmri_img and motions in self._motion.
            fmri_img.uncache()
            fmri_img._dataobj = reg_dataV
            fmri_img.set_data_dtype = reg_dataV.dtype

            self._motion[vol_idx, :] = mot[[0, 1, 2, 5, 3, 4]]

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
            msg = f"#{vol_idx+1};Volume registration;{f};tstamp={tstamp}"
            if pre_proc_time is not None:
                msg += f";took {proc_delay:.4f}s"
            self._logger.info(msg)

            # Set save_name
            fmri_img.set_filename('vr.' + Path(fmri_img.get_filename()).name)

            if self.next_proc:
                # Keep the current processed data
                self.proc_data = np.asanyarray(fmri_img.dataobj)
                save_name = fmri_img.get_filename()

                # Run the next process
                self.next_proc.do_proc(fmri_img, vol_idx=vol_idx,
                                       pre_proc_time=self._proc_time[-1])

            # Update motion plot
            self._plt_xi.append(vol_idx+1)
            for ii in range(6):
                self._plt_motion[ii].append(self._motion[vol_idx, ii])

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
        return super(RtpVolreg, self).end_reset()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def setup_libfuncs(self):
        """ Setup library function access """

        librtp = ctypes.cdll.LoadLibrary(librtp_path)

        # -- Define rtp_align_setup --
        self.rtp_align_setup = librtp.rtp_align_setup
        self.rtp_align_setup.argtypes = \
            [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
             ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int,
             ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
             ctypes.c_void_p, ctypes.c_void_p]

        # -- Define rtp_align_one --
        self.rtp_align_one = librtp.rtp_align_one
        self.rtp_align_one.argtypes = \
            [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
             ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_void_p,
             ctypes.c_void_p, ctypes.c_int, ctypes.c_int,  ctypes.c_int,
             ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
             ctypes.c_void_p]

        # -- Define THD_rota_vol --
        self.THD_rota_vol = librtp.THD_rota_vol
        self.THD_rota_vol.argtypes = \
            [ctypes.c_int, ctypes.c_int, ctypes.c_int,
             ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_void_p,
             ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_float,
             ctypes.c_int, ctypes.c_float,
             ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_ref_vol(self, ref_vol, ref_vi=0):
        if isinstance(ref_vol, string_types) or isinstance(ref_vol, Path):
            # refname is a filename
            ref_vol0 = ref_vol

            # Get volume index
            ma = re.search(r"\'*\[(\d+)\]\'*$", str(ref_vol))
            if ma:
                ref_vi = int(ma.groups()[0])
                ref_vol = re.sub(r"\'*\[(\d+)\]\'*$", '', str(ref_vol))

            # Load volume
            ref_vol = nib.load(ref_vol)
            if len(ref_vol.shape) == 4:
                vol = ref_vol.get_fdata()[:, :, :, ref_vi]
                ref_vol = nib.Nifti1Image(vol, ref_vol.affine)
            assert len(ref_vol.shape) == 3

            self.ref_vol = ref_vol
            self.align_setup()  # Prepare alignment volume

            msg = f"Alignment reference = {ref_vol0}"
            if ma is None:
                msg += f"[{ref_vi}]"
            self._logger.info(msg)

        else:
            # ref_vol is a number. Get reference from ref_vol-th volume in
            # ongoing scan. Once this reference is set, this will be kept
            # during the lifetime of the instance.
            self.ref_vol = int(ref_vol)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def align_setup(self):
        """
        Set up refernce image array and Choleski decomposed matrix for lsqfit

        Reference image array includes original image, three rotation gradient
        images and three shift gradient images

        Calls rtp_align_setup in librtp.so, which is a modified version of
        mri_3dalign_setup in mri_3dalign.c of afni source.

        Setup variables
        ---------------
        self.fitim: numpy array of float32
            seven reference volumes
        self.chol_fitim: numpy array of float64
            Choleski decomposition of weighted covariance matrix between
            refrence images.

        C function definition
        ---------------------
        int rtp_align_setup(float *base_im, int nx, int ny, int nz,
                            float dx, float dy, float dz,
                            int ax1, int ax2, int ax3, int regmode,
                            int nref, float *ref_ims, double *chol_fitim)
        """

        # -- Set parameters --
        # image dimension
        self.nx, self.ny, self.nz = self.ref_vol.shape
        if hasattr(self.ref_vol.header, 'info'):
            self.dx, self.dy, self.dz = \
                np.abs(self.ref_vol.header.info['DELTA'])
        elif hasattr(self.ref_vol.header, 'get_zooms'):
            self.dx, self.dy, self.dz = self.ref_vol.header.get_zooms()[:3]
        else:
            errmsg = "No voxel size information in ref_vol header"
            self._logger.error(errmsg)

        # rotate orientation
        self.ax1 = 2  # z-axis, roll
        self.ax2 = 0  # x-axis, pitch
        self.ax3 = 1  # y-axis, yow

        # Copy base image data and get pointer
        base_im_arr = self.ref_vol.get_fdata().astype(np.float32)
        # x,y,z -> y,z,x -> z,y,x
        base_im_arr = np.moveaxis(np.moveaxis(base_im_arr, 0, -1), 0, 1).copy()
        base_im_p = base_im_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # -- Prepare return arrays --
        # Reference image array
        nxyz = self.nx * self.ny * self.nz
        nref = 7
        ref_img_arr = np.ndarray(nref * nxyz, dtype=np.float32)
        ref_ims_p = ref_img_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Choleski decomposed matrix for lsqfit
        chol_fitim_arr = np.ndarray((nref, nref), dtype=np.float64)
        chol_fitim_p = chol_fitim_arr.ctypes.data_as(
                ctypes.POINTER(ctypes.c_double))

        # -- run func --
        regmode_id = RtpVolreg.regmode_dict[self.regmode]
        self.rtp_align_setup(base_im_p, self.nx, self.ny, self.nz, self.dx,
                             self.dy, self.dz, self.ax1, self.ax2, self.ax3,
                             regmode_id, nref, ref_ims_p, chol_fitim_p)

        self.fitim = ref_img_arr
        self.chol_fitim = chol_fitim_arr

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def align_one(self, fmri_img, vol_idx):
        """
        Align one volume [0] of fmri_img to the reference image.

        Calls rtp_align_one in librtp.so, a modified version of
        mri_3dalign_one in mri_3dalign.c from afni source.

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
            fMRI single volumne image to be aligned.

        Returns
        -------
        tim : 3D array of float32
            Aligned volume data.
        motpar : array
            Six motion parameters: roll, pitch, yaw, dx, dy, dz.


        """

        # Copy function image data and get pointer
        fim_arr = fmri_img.get_fdata().astype(np.float32)
        if fim_arr.ndim > 3:
            fim_arr = np.squeeze(fim_arr)
        fim_arr = np.moveaxis(np.moveaxis(fim_arr, 0, -1), 0, 1).copy()
        fim_p = fim_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Get fitim and chol_fitim data pointer
        fitim_p = self.fitim.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        chol_fitim_p = self.chol_fitim.ctypes.data_as(
                ctypes.POINTER(ctypes.c_double))

        # Initial motion parameter
        if not np.all(self._motion[vol_idx-1, :] == 0):
            init_motpar = self._motion[vol_idx-1, :]
        else:
            init_motpar = np.zeros(6, dtype=np.float32)

        init_motpar_p = init_motpar.ctypes.data_as(
                ctypes.POINTER(ctypes.c_float))

        # Prepare return arrays
        nxyz = self.nx * self.ny * self.nz
        tim_arr = np.ndarray(nxyz, dtype=np.float32)
        tim_p = tim_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        motpar = np.ndarray(7, dtype=np.float32)
        motpar_p = motpar.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # -- run func --
        regmode_id = RtpVolreg.regmode_dict[self.regmode]
        self.rtp_align_one(fim_p, self.nx, self.ny, self.nz, self.dx, self.dy,
                           self.dz, fitim_p, chol_fitim_p, 7, self.ax1,
                           self.ax2, self.ax3, init_motpar_p, regmode_id,
                           tim_p, motpar_p)

        tim = np.reshape(tim_arr, (self.nz, self.ny, self.nx))
        tim = np.moveaxis(np.moveaxis(tim, 0, -1), 0, 1)

        return tim, motpar[1:]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    class PlotMotion(QtCore.QObject):
        finished = QtCore.pyqtSignal()

        def __init__(self, root, main_win=None):
            super().__init__()

            self._logger = logging.getLogger(self.__class__.__name__)
            self.root = root
            self.main_win = main_win
            self.abort = False

            # Initialize figure
            plt_winname = 'Motion'
            self.plt_win = MatplotlibWindow()
            self.plt_win.setWindowTitle(plt_winname)

            # set position
            if main_win is not None:
                main_geom = main_win.geometry()
                x = main_geom.x() + main_geom.width() + 10
                y = main_geom.y() + 400
            else:
                x, y = (0, 0)
            self.plt_win.setGeometry(x, y, 500, 500)

            # Set axis
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
                    ax.axhline(0.2, c='k', ls=':', lw=0.5)
                    ax.axhline(0.3, c='r', ls='--', lw=0.5)
                    ax.set_ylim(0, 0.5)
            ax.set_xlabel('TR')

            # show window
            self.plt_win.show()
            self.plt_win.canvas.draw()

        # ---------------------------------------------------------------------
        def run(self):
            self._mot_alart_band = []
            plt_xi = self.root._plt_xi.copy()
            while self.plt_win.isVisible() and not self.abort:
                if self.main_win is not None and not self.main_win.isVisible():
                    break

                if len(self.root._plt_xi) == len(plt_xi):
                    time.sleep(0.1)
                    continue

                try:
                    # Plot motion
                    plt_xi = np.array(self.root._plt_xi.copy())
                    plt_motion = self.root._plt_motion
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

                        # Rescale ylim
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

                    # Motion alarm band
                    for b in self._mot_alart_band:
                        b.remove()

                    self._mot_alart_band = []
                    fd_mark_02_idx = np.argwhere(
                        (plt_FD >= 0.2) & (plt_FD < 0.3)).ravel()
                    fd_mark_03_idx = np.argwhere(plt_FD >= 0.3).ravel()

                    ax_FD = self._axes[-1]
                    for x in plt_xi[fd_mark_02_idx]:
                        band = ax_FD.axvspan(x-1, x, ec='k', fc='yellow',
                                             alpha=0.3, linewidth=0.1)
                        self._mot_alart_band.append(band)

                    for x in plt_xi[fd_mark_03_idx]:
                        band = ax_FD.axvspan(x-1, x, ec='k', fc='red',
                                             alpha=0.3, linewidth=0.1)
                        self._mot_alart_band.append(band)

                    self.plt_win.canvas.draw()

                # except IndexError:
                #     continue

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

            elif 'internal' in val:
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
                fname = self.select_file_dlg('VOLREG: Selct base volume',
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
                            ref_img.shape[3])
                    if fname[0] == '':
                        if reset_fn:
                            reset_fn(1)
                        return

                    ref_fname += f"[{num}]"
                else:
                    num = 0

                self.ref_fname = ref_fname
                self.set_ref_vol(fname[0], num)
                if hasattr(self, 'ui_baseVol_lnEd'):
                    self.ui_baseVol_lnEd.setText(str(ref_fname))

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

                if reset_fn is None and hasattr(self, 'ui_baseVol_cmbBx'):
                    self.ui_baseVol_cmbBx.setCurrentIndex(0)

                self.ref_fname = val
                self.set_ref_vol(fname, num)
                if hasattr(self, 'ui_baseVol_lnEd'):
                    self.ui_baseVol_lnEd.setText(str(self.ref_fname))

            return

        elif attr == 'ref_fname':
            if len(val):
                ma = re.search(r"\[(\d+)\]", val)
                if ma:
                    num = int(ma.groups()[0])
                    val = re.sub(r"\[(\d+)\]", '', val)
                else:
                    num = 0

                if not Path(val).is_file():
                    return

                if hasattr(self, 'ui_baseVol_lnEd'):
                    self.ui_baseVol_lnEd.setText(str(val))
                self.set_ref_vol(val, num)

        elif attr == 'regmode' and reset_fn is None:
            if hasattr(self, 'ui_regmode_cmbBx'):
                if type(val) is int:
                    if val not in list(RtpVolreg.regmode_dict.values()):
                        return

                    regmode_id = val
                    val = list(RtpVolreg.regmode_dict.keys())[
                        list(RtpVolreg.regmode_dict.values()).index(val)]
                else:
                    regmode_id = RtpVolreg.regmode_dict[val]
                self.ui_regmode_cmbBx.setCurrentIndex(regmode_id)

        elif attr == 'max_scan_length':
            if self.desMtx_read is not None and \
                    self.desMtx_read.shape[0] > val:
                val = self.desMtx_read.shape[0]
                if reset_fn:
                    reset_fn(val)

            # Update self.desMtx0
            self.setup_regressor_template(self.desMtx_read,
                                          max_scan_length=val,
                                          col_names_read=self.col_names_read)
            if val < self.wait_num:
                val = self.wait_num + 1

            if reset_fn is None:
                if hasattr(self, 'ui_maxLen_spBx'):
                    self.ui_maxLen_spBx.setValue(val)

        elif attr == 'save_proc':
            if hasattr(self, 'ui_saveProc_chb'):
                self.ui_saveProc_chb.setChecked(val)

        elif reset_fn is None:
            # Ignore an unrecognized parameter
            if not hasattr(self, attr):
                errmsg = f"{attr} is unrecognized parameter."
                self._logger.error(errmsg)
                return

        # -- Set value --
        setattr(self, attr, val)
        if echo:
            print(f"{self.__class__.__name__}." + attr, '=',
                  getattr(self, attr))

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
            'background: white; border: 0px none;')
        ui_rows.append((None, self.ui_baseVol_lnEd))

        self.ui_objs.extend([var_lb, self.ui_baseVol_cmbBx,
                             self.ui_baseVol_lnEd])

        if isinstance(self.ref_vol, string_types):
            self.ui_baseVol_cmbBx.setCurrentIndex(0)
            self.ui_baseVol_lnEd.setText(str(self.ref_vol))
        else:
            self.ui_baseVol_cmbBx.setCurrentIndex(1)
            self.ui_baseVol_lnEd.setText(
                    f"Internal run volume index {self.ref_vol}")

        # regmode
        var_lb = QtWidgets.QLabel("Resampling :")
        self.ui_regmode_cmbBx = QtWidgets.QComboBox()
        self.ui_regmode_cmbBx.addItems(['Nearest Neighbor', 'Linear', 'Cubic',
                                        'Fourier', 'Quintic', 'Heptic'])
        regmode_id = RtpVolreg.regmode_dict[self.regmode]
        self.ui_regmode_cmbBx.setCurrentIndex(regmode_id)
        self.ui_regmode_cmbBx.currentIndexChanged.connect(
                lambda idx:
                self.set_param('regmode',
                               self.ui_regmode_cmbBx.currentIndex(),
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

    rtp_tshift = RtpTshift()
    rtp_tshift.method = 'cubic'
    rtp_tshift.ignore_init = 3
    rtp_tshift.ref_time = 0

    rtp_volreg = RtpVolreg(regmode='cubic')

    # Set slice timing from a sample data
    rtp_tshift.slice_timing_from_sample(testdata_f)

    # Set reference volume
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

    """
    rtp_volreg tends to estimate temporal difference smaller than 3dvolreg
    because rtp_volreg starts estimation from the previous motion values for
    faster covergence, while 3dvolreg starts from 0.
    """

    proc_delay = rtp_volreg.proc_delay
    motion = rtp_volreg._motion[rtp_volreg._proc_start_idx:, :]
    rtp_tshift.end_reset()

    mot_f = rtp_volreg.saved_filename.name.replace('.nii.gz', '')
    mot_f = rtp_volreg.saved_filename.parent / ('motion_' + mot_f + '.1D')
    np.savetxt(mot_f, motion)

    # --- Plot ---
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(proc_delay[1:], bins='auto')
    np.median(proc_delay[1:])
