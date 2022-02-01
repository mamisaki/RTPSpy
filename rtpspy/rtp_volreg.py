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


# %% RTP_VOLREG ===============================================================
class RTP_VOLREG(RTP):
    """
    Real-time online volume registration for motion correction.
    AFNI functions in librtp.so (compiled from libmri.so) is called.
    """

    regmode_dict = {'nn': 0, 'linear': 1, 'cubic': 2, 'fourier': 3,
                    'quintic': 4, 'heptic': 5, 'tsshift': 6}

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, ref_vol=0, regmode='heptic', **kwargs):
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
        """
        super(RTP_VOLREG, self).__init__(**kwargs)

        # Set instance parameters
        self.regmode = regmode

        # Initialize parameters and C library function call
        self.motion = np.ndarray([0, 6], dtype=np.float32)  # motion parameter
        self.set_ref_vol(ref_vol)
        self.setup_libfuncs()

        # alignment parameters (Set from AFNI plug_realtime default values)
        self.max_iter = 9
        self.dxy_thresh = 0.05  # pixels
        self.phi_thresh = 0.07  # degree

        # --- initialize for motion plot ---
        self.plt_xi = []
        self.plt_motion = []
        for ii in range(6):
            self.plt_motion.append(list([]))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ready_proc(self):
        self._proc_ready = self.ref_vol is not None
        if not self._proc_ready:
            errmsg = "Refence volume or volume index has not been set."
            self.errmsg(errmsg)

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
            # Fill motion vectors of missing volumes with zero.
            while self.motion.shape[0] < vol_idx:
                self.motion = np.concatenate(
                        [self.motion, np.zeros((1, 6), dtype=np.float32)],
                        axis=0)

            # if self.ref_vol is int (index of reference volume in the current
            # scan), wait for the reference index and set the reference.
            if type(self.ref_vol) is int:
                if vol_idx < self.ref_vol:
                    # Append zero vector
                    mot = np.zeros((1, 6), dtype=np.float32)
                    self.motion = np.concatenate([self.motion, mot], axis=0)
                    return

                elif vol_idx >= self.ref_vol:
                    ref_vi = self.ref_vol
                    self.ref_vol = fmri_img
                    self.align_setup()
                    # Append zero vector
                    mot = np.zeros((1, 6), dtype=np.float32)
                    self.motion = np.concatenate([self.motion, mot], axis=0)
                    if self._verb:
                        msg = f"Alignment reference is set to volume {ref_vi}"
                        msg += " of current sequence."
                        self.logmsg(msg)
                    return

            # --- Run the procress --------------------------------------------
            # Perform volume alignment
            reg_dataV, mot = self.align_one(fmri_img)

            # Set aligned data in fmri_img and motions in self.motion.
            fmri_img.uncache()
            fmri_img._dataobj = reg_dataV
            fmri_img.set_data_dtype = reg_dataV.dtype

            mot = mot[np.newaxis, [0, 1, 2, 5, 3, 4]]
            self.motion = np.concatenate([self.motion, mot], axis=0)

            # Update motion plot
            self.plt_xi.append(vol_idx)
            for ii in range(6):
                self.plt_motion[ii].append(self.motion[-1][ii])

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
                msg = f'#{vol_idx}, Volume registration is done for {f}'
                if pre_proc_time is not None:
                    msg += f' (took {proc_delay:.4f}s)'
                msg += '.'
                self.logmsg(msg)

            # Set save_name
            fmri_img.set_filename('vr.' + Path(fmri_img.get_filename()).name)

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

        # Reset running variables
        self.motion = np.ndarray([0, 6], dtype=np.float32)

        # Reset plot values
        self.plt_xi[:] = []
        for ii in range(6):
            self.plt_motion[ii][:] = []

        return super(RTP_VOLREG, self).end_reset()

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

            if self._verb:
                msg = f"Alignment reference = {ref_vol0}"
                if ma is None:
                    msg += f"[{ref_vi}]"
                self.logmsg(msg)

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
            self.errmsg("No voxel size information in ref_vol header")

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
        regmode_id = RTP_VOLREG.regmode_dict[self.regmode]
        self.rtp_align_setup(base_im_p, self.nx, self.ny, self.nz, self.dx,
                             self.dy, self.dz, self.ax1, self.ax2, self.ax3,
                             regmode_id, nref, ref_ims_p, chol_fitim_p)

        self.fitim = ref_img_arr
        self.chol_fitim = chol_fitim_arr

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def align_one(self, fmri_img):
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
        if len(self.motion) and not np.any(np.isnan(self.motion[-1])):
            init_motpar = self.motion[-1]
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
        regmode_id = RTP_VOLREG.regmode_dict[self.regmode]
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
                y = main_geom.y() + 230
            else:
                x, y = (0, 0)
            self.plt_win.setGeometry(x, y, 500, 500)

            # Set axis
            self.mot_labels = ['roll (deg.)', 'pitch (deg.)', 'yaw (deg.)',
                               'dS (mm)', 'dL (mm)', 'dP (mm)']
            self._axes = self.plt_win.canvas.figure.subplots(6, 1)
            self.plt_win.canvas.figure.subplots_adjust(
                    left=0.15, bottom=0.08, right=0.95, top=0.97, hspace=0.35)
            self._ln = []
            for ii, ax in enumerate(self._axes):
                ax.set_ylabel(self.mot_labels[ii])
                ax.set_xlim(0, 10)
                self._ln.append(ax.plot(0, 0))

            ax.set_xlabel('TR')

            # show window
            self.plt_win.show()

            self.plt_win.canvas.draw()
            self.plt_win.canvas.start_event_loop(0.005)

        # ---------------------------------------------------------------------
        def run(self):
            plt_xi = self.root.plt_xi.copy()
            while self.plt_win.isVisible() and not self.abort:
                if self.main_win is not None and not self.main_win.isVisible():
                    break

                if len(self.root.plt_xi) == len(plt_xi):
                    time.sleep(0.1)
                    continue

                try:
                    # Plot motion
                    plt_xi = self.root.plt_xi.copy()
                    plt_motion = self.root.plt_motion
                    for ii, ax in enumerate(self._axes):
                        ll = min(len(plt_xi), len(plt_motion[ii]))
                        if ll == 0:
                            continue

                        self._ln[ii][0].set_data(plt_xi[:ll],
                                                 plt_motion[ii][:ll])
                        ax.relim()
                        ax.autoscale_view()

                        xl = ax.get_xlim()
                        if (plt_xi[-1]//10 + 1)*10 > xl[1]:
                            ax.set_xlim([0, (plt_xi[-1]//10 + 1)*10])

                    self.plt_win.canvas.draw()
                    self.plt_win.canvas.start_event_loop(0.01)

                except IndexError:
                    continue

                except Exception as e:
                    self.root.errmsg(e)
                    sys.stdout.flush()
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
        self.pltMotion = RTP_VOLREG.PlotMotion(self, main_win=self.main_win)
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

        elif attr == 'ref_vol':
            if isinstance(val, Path):
                val = str(val)

            if type(val) == int:
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

            elif type(val) == str:
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
                if type(val) == int:
                    if val not in list(RTP_VOLREG.regmode_dict.values()):
                        return

                    regmode_id = val
                    val = list(RTP_VOLREG.regmode_dict.keys())[
                        list(RTP_VOLREG.regmode_dict.values()).index(val)]
                else:
                    regmode_id = RTP_VOLREG.regmode_dict[val]
                self.ui_regmode_cmbBx.setCurrentIndex(regmode_id)

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
        regmode_id = RTP_VOLREG.regmode_dict[self.regmode]
        self.ui_regmode_cmbBx.setCurrentIndex(regmode_id)
        self.ui_regmode_cmbBx.currentIndexChanged.connect(
                lambda idx:
                self.set_param('regmode',
                               self.ui_regmode_cmbBx.currentIndex(),
                               self.ui_regmode_cmbBx.setCurrentIndex))
        ui_rows.append((var_lb, self.ui_regmode_cmbBx))
        self.ui_objs.extend([var_lb, self.ui_regmode_cmbBx])

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
        excld_opts = ('work_dir', 'motion', 'plt_xi',
                      'plt_motion', 'chol_fitim', 'ref_vol', 'nx', 'ny', 'nz',
                      'fitim', 'dx', 'dy', 'dz', 'ax1', 'ax2', 'ax3')
        sel_opts = {}
        for k, v in all_opts.items():
            if k in excld_opts:
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

    # Create RTP_TSHIFT and RTP_VOLREG  instance
    from rtpspy.rtp_tshift import RTP_TSHIFT

    rtp_tshift = RTP_TSHIFT()
    rtp_tshift.method = 'cubic'
    rtp_tshift.ignore_init = 3
    rtp_tshift.ref_time = 0

    rtp_volreg = RTP_VOLREG(regmode='cubic')

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
    motion = rtp_volreg.motion[rtp_volreg.proc_start_idx:, :]
    rtp_tshift.end_reset()

    mot_f = rtp_volreg.saved_filename.name.replace('.nii.gz', '')
    mot_f = rtp_volreg.saved_filename.parent / ('motion_' + mot_f + '.1D')
    np.savetxt(mot_f, motion)

    # --- Plot ---
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(proc_delay[1:], bins='auto')
    np.median(proc_delay[1:])
