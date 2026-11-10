#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTP regression: online GLM denoising of fMRI volumes

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import sys
import time
import re
import traceback
import logging
import inspect

import numpy as np
import nibabel as nib
import psutil
from PyQt5 import QtWidgets
import pandas as pd
import torch

try:
    from .rtp_common import RTP, MatplotlibWindow
except Exception:
    from rtpspy.rtp_common import RTP, MatplotlibWindow

gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()

# Regressor names
MOT_NAMES = ('roll', 'pitch', 'yaw', 'dS', 'dL', 'dP')
DMOT_NAMES = ('dtroll', 'dtpitch', 'dtyaw', 'dtdS', 'dtdL', 'dtdP')
PHYS_REG_NAMES = {
    'RVT5': ['RVT0', 'RVT1', 'RVT2', 'RVT3', 'RVT4'],
    'RICOR8': ['Resp0', 'Resp1', 'Resp2', 'Resp3',
               'Card0', 'Card1', 'Card2', 'Card3'],
    'RVT+RICOR13': ['RVT0', 'RVT1', 'RVT2', 'RVT3', 'RVT4',
                    'Resp0', 'Resp1', 'Resp2', 'Resp3',
                    'Card0', 'Card1', 'Card2', 'Card3']}
ROI_REGS = ('GS', 'WM', 'Vent')  # Mean signal regressors in ROI masks
# Regressors standardized in each fit
NORM_REGS = MOT_NAMES + DMOT_NAMES + ROI_REGS

# UI option lists. The index of an option is its combo box index.
MOT_REG_OPTS = ('None', 'mot6', 'mot12', 'dmot6')
PHYS_REG_OPTS = ('None', 'RICOR8', 'RVT5', 'RVT+RICOR13')
MASK0_TEXT = 'zero-out initial received volume'

# Columns of the online QC table (see RtpRegress.get_qc_table)
QC_COLS = ('vol_idx', 'FD', 'DVARS', 'tSNR', 'censored')
# Radius (mm) of the sphere used to convert rotations to displacements in FD
# (Power et al., 2012)
FD_HEAD_RADIUS = 50.0


# %% lstsq_SVDsolver ==========================================================
def lstsq_SVDsolver(A, B, rcond=None):
    """
    Solve a linear system AX = B in the least-squares sense (minimize
    ||AX - B||^2) with SVD.

    Parameters
    ----------
    A : 2D tensor (n x m)
        n: number of samples (time points), m: number of variables
        (regressors). The number of rows (n) must be larger than the number
        of columns (m).
    B : 2D tensor (n x k)
        n: number of samples, k: number of dependent variables (e.g.,
        voxels).
    rcond : float, optional
        Cut-off ratio for small singular values. For the purpose of rank
        determination, singular values are treated as zero if they are smaller
        than rcond times the largest singular value.
        The default is the float32 machine precision times min(A.shape).

    Returns
    -------
    X: 2D tensor (m x k)
        Minimum-norm least-squares solution. Singular values below the
        cut-off (including an all-zero A) give zero contribution.
    """

    if rcond is None:
        rcond = np.finfo(np.float32).eps * min(A.shape)

    # Thin SVD for A: A = U diag(S) Vh
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)

    # Pseudo-inverse of the singular values; values <= S.max() * rcond are
    # treated as zero (no Python loop, no GPU-host sync).
    sthresh = S.max() * rcond
    Sinv = torch.where(S > sthresh, 1.0 / S, torch.zeros_like(S))

    # X = V diag(Sinv) U^T B
    X = Vh.T @ (Sinv[:, None] * (U.T @ B))

    return X


# %% RtpRegress class ========================================================
class RtpRegress(RTP):
    """
    RTP online regression analysis.
    A cumulative GLM is used: the GLM is fitted again at each volume with all
    the available volumes (or the latest ones when the scan is longer than
    the buffers; see max_scan_length).

    Online QC
    ---------
    At each received volume (after ignore_init), the following measures are
    computed and kept in self.qc (arrays indexed by the 0-based volume
    index; see get_qc_table). The latest values are in self.qc_last and are
    passed to next_proc.do_proc as the keyword argument
    qc={'FD', 'DVARS', 'tSNR', 'censored'} if next_proc.do_proc accepts it
    (a 'qc' parameter or **kwargs).

    FD : Framewise displacement (Power et al., 2012), mm.
        FD_t = sum(|d translation_t|) + 50 * sum(|d rotation_t|),
        where d is the difference from the previous received volume, the
        translations (dS, dL, dP) are in mm, and the rotations (roll, pitch,
        yaw; degrees in volreg._motion) are converted to radians (50 mm is
        the head radius). 0 for the first volume. NaN if volreg is not set
        or not enabled.
    DVARS : Root mean square over the mask voxels of the volume-to-volume
        difference of the REGRESS input data, as percent of the mean
        in-mask signal of the current volume.
        DVARS_t = 100 * sqrt(mean_v((y_t,v - y_t-1,v)^2)) / mean_v(y_t,v),
        where v runs over the processing mask (maskV). NaN for the first
        volume (and if the mean signal is ~0).
    tSNR : Median over the mask voxels of the voxel-wise running temporal
        SNR (mean / standard deviation with ddof=1) of the REGRESS input
        data, from the first received volume to the current one (all
        volumes, including censored ones). The running mean and variance
        are updated with Welford's algorithm on the device. Voxels with zero
        variance are excluded. With an even number of voxels, the lower of
        the two middle values is used (torch.nanmedian). NaN until 3
        volumes are received.
    censored : True if FD > censor_FD (censor_FD > 0).

    The QC table is saved as a CSV file at end_reset when save_proc is True
    (next to the saved processed image; see end_reset).

    Motion censoring
    ----------------
    When censor_FD > 0, a volume with FD > censor_FD is excluded from all
    the GLM fits while it is in the window: its row of the design matrix
    used for fitting is zeroed (a zero row does not change the
    least-squares solution; the Y buffer is unchanged), and it is not used
    for the standardization of the motion/ROI regressors nor for the
    signal scaling mean (Y_mean). The censored volume itself is still
    output: its residual is computed with its own regressors and the betas
    fitted to the other volumes. The first fit waits until the number of
    uncensored volumes in the window exceeds the number of regressors.
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, max_poly_order=np.inf, TR=2.0, mot_reg='None',
                 volreg=None, GS_reg=False, GS_mask=None, WM_reg=False,
                 WM_mask=None, Vent_reg=False, Vent_mask=None,
                 mask_src_proc=None, phys_reg='None', rtp_physio=None,
                 tshift=0.0, desMtx=None, wait_num=0, mask_file=0,
                 max_scan_length=800, onGPU=gpu_available, reg_retro_proc=True,
                 censor_FD=0.0, **kwargs):
        """
        Parameters
        ----------
        max_poly_order : int or np.inf, optional
            Maximum order of the polynomial regressors. The polynomial order
            increases automatically with the scan length. The default is
            np.inf.
        TR : float, optional
            fMRI TR (seconds). This is used to set the polynomial order;
            order = 1 + int(TR * number_of_volumes / 150).
            The default is 2.0.
        mot_reg : str, ['None'|'mot6'|'mot12'|'dmot6'], optional
            Motion regressor type.
            'None': no motion regressor
            'mot6': six motion parameters; yaw, pitch, roll, dx, dy, dz
            'mot12': mot6 plus their temporal derivatives
            'dmot6': temporal derivatives of the six motion parameters
            The default is 'None'.
        volreg : RtpVolreg object, optional
            RtpVolreg instance to read the motion parameters from. The default
            is None.
        GS_reg : bool, optional
            Use the global signal regressor. The default is False.
        GS_mask : str or Path, optional
            Mask file for the global signal. The default is None.
        WM_reg : bool, optional
            Use the mean white matter signal regressor. The default is False.
        WM_mask : str or Path, optional
            Mask file for the white matter. The default is None.
        Vent_reg : bool, optional
            Use the mean ventricle signal regressor. The default is False.
        Vent_mask : str or Path, optional
            Mask file for the ventricles. The default is None.
        mask_src_proc : RTP object, optional
            RTP module whose output (proc_data) is used to calculate the ROI
            mean signal regressors (GS, WM, Vent). These regressors should be
            made from unsmoothed data, so when the input to RtpRegress is
            smoothed, set this to a module before smoothing (e.g., RtpVolreg).
            Required when GS_reg, WM_reg, or Vent_reg is True. The default is
            None.
        phys_reg : str, ['None'|'RICOR8'], optional
            Physiological signal regressor type.
            'None': no physio regressor
            'RICOR8': four respiration and four cardiac RETROICOR regressors
            'RVT5' (five RVT regressors) and 'RVT+RICOR13' (RVT5 + RICOR8)
            are not supported by the real-time RetroTs. RVT is not recommended
            for RTP anyway (see Misaki and Bodurka, 2021).
            The default is 'None'.
        rtp_physio : RtpTTLPhysio object, optional
            Object to get the RETROICOR regressors from (get_retrots). The
            default is None.
        tshift : float, optional
            Slice timing offset (seconds) for making the RETROICOR regressors.
            The default is 0.0.
        desMtx : 2D array, optional
            Additional design matrix other than the motion, physio, and
            polynomial regressors. The rows must include the initial ignored
            volumes, which are not used in the regression. The default is
            None.
        wait_num : int, optional
            Minimum number of volumes (excluding the ignored volumes) to
            receive before starting the regression. If it is smaller than the
            number of regressors + 1, the regression waits for that number of
            volumes instead. The default is 0.
        mask_file : 0 or str (Path), optional
            Mask of the processing region. 0 means a mask of the non-zero
            voxels in the first received volume. The default is 0.
        max_scan_length : int, optional
            Maximum scan length (volumes). The X and Y buffers are allocated
            for this length at the scan start (no reallocation during a scan).
            If the scan is longer, the buffers are used as ring buffers and
            the GLM is fitted with the latest max_scan_length volumes. If
            desMtx is given, max_scan_length is at least the length of desMtx.
            The default is 800.
        onGPU : bool, optional
            Run the regression on a GPU when available. The default is
            torch.cuda.is_available() or torch.backends.mps.is_available().
        reg_retro_proc : bool, optional
            Process the volumes received before the regression started
            retrospectively at the first fit. The default is True.
        censor_FD : float, optional
            FD threshold (mm) for motion censoring. A volume with FD >
            censor_FD is excluded from the GLM fits (see the class
            docstring). 0 turns censoring off. The default is 0.0.

        """
        super(RtpRegress, self).__init__(**kwargs)

        # --- Set parameters ---
        # Polynomial regressors
        self.max_poly_order = max_poly_order
        self.TR = TR
        # Motion regressors
        self.mot_reg = mot_reg
        self.volreg = volreg
        # ROI mean signal regressors
        self.GS_reg = GS_reg
        self.GS_mask = GS_mask
        self.WM_reg = WM_reg
        self.WM_mask = WM_mask
        self.Vent_reg = Vent_reg
        self.Vent_mask = Vent_mask
        self.mask_src_proc = mask_src_proc
        # Physiological signal regressors
        self.phys_reg = phys_reg
        self.rtp_physio = rtp_physio
        self.tshift = tshift
        # Additional design matrix
        self.desMtx_read = desMtx
        # Operational parameters
        self.mask_file = mask_file
        self.max_scan_length = max_scan_length
        self.onGPU = onGPU
        self.reg_retro_proc = reg_retro_proc
        self.wait_num = wait_num
        self.censor_FD = censor_FD

        # --- Initialize working data ---
        # Masks
        self.maskV = None  # Processing region
        self.GS_maskdata = None  # ROI masks within maskV
        self.WM_maskdata = None
        self.Vent_maskdata = None
        # Regressors
        self.reg_names = []
        self.col_names_read = []  # Column names of desMtx_read
        # Regressor template (without polynomials), including the ignored
        # volumes. Rows: max_scan_length, or the length of desMtx_read.
        self.desMtx0 = None
        self.desMtx = None  # X buffer (tensor)
        self.mot0 = None  # Previous motion parameters for the derivatives
        # Data
        self.YMtx = None  # Y buffer (tensor)
        self.Y_mean = None  # Mean signal for scaling the data
        self._ring_warned = False  # Warned that the buffers are ring buffers
        # Memory budget (bytes) for the X and Y buffers. None: half of the
        # free GPU (or system) memory.
        self.max_buffer_bytes = None
        # Censor flag of each buffer row (set when the row is written)
        self._censor_row = None
        self._censor_warned = False  # Warned that few volumes are left

        # Online QC
        self.qc = None  # {'FD', 'DVARS', 'tSNR', 'censored', 'received'}
        self.qc_last = None  # QC of the latest volume
        self._qc_init_state()
        # (next_proc, whether its do_proc accepts the qc keyword)
        self._qc_kw_cache = (None, False)

        # --- Set the number of volumes to wait ---
        self.set_wait_num()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @property
    def onGPU(self):
        return self.device != 'cpu'

    @onGPU.setter
    def onGPU(self, use_gpu):
        if use_gpu:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self._error("GPU is not available.")
                self.device = 'cpu'
        else:
            self.device = 'cpu'

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _error(self, errmsg):
        """Log an error message and show it in a popup (GUI only)."""
        self._logger.error(errmsg)
        self.err_popup(errmsg)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _to_device(self, arr):
        """Convert a numpy array to a float32 tensor on self.device."""
        return torch.from_numpy(arr.astype(np.float32)).to(self.device)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ready_proc(self):
        self._proc_ready = True

        errmsgs = []
        if self.TR is None:
            errmsgs.append('TR is not set.')

        if self.mot_reg != 'None' and self.volreg is None:
            errmsgs.append('RtpVolreg object is not set.')

        if self.phys_reg not in ('None', 'RICOR8'):
            # RetroTs makes only the 8 RETROICOR regressors.
            errmsgs.append(
                f"Physio regressor '{self.phys_reg}' is not supported."
                " Use 'RICOR8' or 'None'.")

        if self.phys_reg != 'None' and self.rtp_physio is None:
            errmsgs.append('Physio recorder object is not set.')

        if (self.GS_reg or self.WM_reg or self.Vent_reg) and \
                self.mask_src_proc is None:
            errmsgs.append(
                'mask_src_proc must be set for GS/WM/Vent regressors.')

        if self.desMtx_read is None and not self.max_scan_length:
            errmsgs.append(
                'Either the design matrix or max_scan_length must be set.')

        for errmsg in errmsgs:
            self._error(errmsg)
            self._proc_ready = False

        if self.censor_FD > 0 and not self._volreg_enabled():
            self._logger.warning(
                "Motion censoring is set (censor_FD > 0) but VOLREG is not"
                " available. FD cannot be calculated and no volume is"
                " censored.")

        if self.next_proc:
            self._proc_ready &= self.next_proc.ready_proc()

        if self._proc_ready:
            self.setup_regressor_template(self.desMtx_read,
                                          self.max_scan_length,
                                          self.col_names_read)
            if isinstance(self.mask_file, (str, Path)):
                self.set_mask(self.mask_file)

            self.set_wait_num()

        return self._proc_ready

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def do_proc(self, fmri_img, vol_idx=None, pre_proc_time=None, **kwargs):
        """
        Regress out the nuisance signals from a volume and pass the residual
        to the next process.

        The X (self.desMtx) and Y (self.YMtx) buffers have a fixed number of
        rows (buf_len). The data of volume vol_idx is stored in row
        vol_idx % buf_len, so the buffers work as ring buffers when the scan
        is longer than buf_len. Each GLM fit uses the latest
        min(number of processed volumes, buf_len) volumes.

        Parameters
        ----------
        fmri_img : nibabel image object
            Volume to process. Its data are replaced with the residual.
        vol_idx : int, optional
            0-based index of the volume from the scan start. If None,
            the number of received volumes - 1 is used.
        pre_proc_time : float, optional
            Time stamp when the previous module finished.

        The online QC of the volume (see the class docstring) is computed
        here and passed to next_proc.do_proc as the keyword argument qc.
        """

        try:
            self._vol_num += 1  # Number of volumes received in this module
            if vol_idx is None:
                vol_idx = self._vol_num - 1

            if vol_idx < self.ignore_init:
                return

            if self._proc_start_idx < 0:
                self._proc_start_idx = vol_idx

            dataV = fmri_img.get_fdata()
            if dataV.ndim > 3:
                dataV = np.squeeze(dataV)

            # --- Initialize --------------------------------------------------
            if self.maskV is None:
                # Make the processing mask from the first received volume
                self.set_mask(dataV)
                self._logger.info(
                    f"Mask is set with the volume index {vol_idx}.")

            # Read the ROI masks. If a mask is not available, the regressor
            # option is reset to False (only once).
            for reg in ROI_REGS:
                if getattr(self, f"{reg}_reg") and \
                        getattr(self, f"{reg}_maskdata") is None:
                    self.read_roi_mask(reg, dataV)

            if self.desMtx is None:
                self._init_X_buffer()

            if self.YMtx is None:
                self._init_Y_buffer()

            if self._qc_M is None:
                self._init_qc()

            # --- Store the volume in the buffers -----------------------------
            buf_len = self.YMtx.shape[0]
            row = vol_idx % buf_len  # Row of this volume in the buffers
            reg0_num = self.desMtx0.shape[1]  # Number of template regressors
            if vol_idx >= buf_len:
                if not self._ring_warned:
                    self._logger.warning(
                        "The scan is longer than max_scan_length "
                        f"({buf_len}). The GLM is fitted with the latest "
                        f"{buf_len} volumes.")
                    self._ring_warned = True
                # The row is reused: reset its template part (e.g., the
                # design matrix read from a file) for this volume.
                if vol_idx < self.desMtx0.shape[0]:
                    self.desMtx[row, :reg0_num] = torch.from_numpy(
                        self.desMtx0[vol_idx]).to(self.device)
                else:
                    self.desMtx[row, :reg0_num] = 0.0

            # Y. Once Y_mean is set (at the first fit), the data are scaled
            # to percent of the mean.
            ydata = self._to_device(dataV[self.maskV])

            # Online QC of this volume (ydata is not modified)
            qc = self._update_qc(vol_idx, ydata)
            if self._censor_row is None or len(self._censor_row) != buf_len:
                self._censor_row = np.zeros(buf_len, dtype=bool)
            self._censor_row[row] = qc['censored']  # Reset for a reused row
            if qc['censored']:
                self._logger.info(
                    f"#{vol_idx+1};Censored;FD={qc['FD']:.4f}"
                    f" > {self.censor_FD}")

            if self.Y_mean is not None:
                ydata = torch.where(self.Y_mean_mask,
                                    ydata / self.Y_mean * 100,
                                    torch.zeros_like(ydata))
                ydata.clamp_(max=200)

            self.YMtx[row, :] = ydata

            # Motion regressors (self.motcols is contiguous)
            if self.mot_reg != 'None':
                mot = self.volreg._motion[vol_idx, :]
                if self.mot_reg in ('mot12', 'dmot6'):
                    if self._vol_num > 1 and self.mot0 is not None:
                        dmot = mot - self.mot0
                    else:
                        dmot = np.zeros(6, dtype=np.float32)
                    self.mot0 = mot

                if self.mot_reg in ('mot6', 'mot12'):
                    col0 = self.motcols[0]
                    self.desMtx[row, col0:col0+6] = self._to_device(mot)
                    if self.mot_reg == 'mot12':
                        col6 = self.motcols[6]
                        self.desMtx[row, col6:col6+6] = \
                            self._to_device(dmot)
                elif self.mot_reg == 'dmot6':
                    col0 = self.motcols[0]
                    self.desMtx[row, col0:col0+6] = self._to_device(dmot)

            # ROI mean signal regressors from the mask_src_proc output
            if self.GS_reg or self.WM_reg or self.Vent_reg:
                src_data = self.mask_src_proc.proc_data[self.maskV]
                for reg in ROI_REGS:
                    if getattr(self, f"{reg}_reg"):
                        roi_mask = getattr(self, f"{reg}_maskdata")
                        self.desMtx[row, getattr(self, f"{reg}_col")] = \
                            float(src_data[roi_mask].mean())

            # --- Wait until enough volumes are received ----------------------
            # Number of volumes received after the ignored ones
            n_proc = vol_idx - self._proc_start_idx + 1
            if n_proc <= self.wait_num:
                tstamp, proc_delay = self._record_proc_time(pre_proc_time)
                wait_idx = self._proc_start_idx + self.wait_num
                self._log_proc(
                    f"#{vol_idx+1};Regression;"
                    f"Wait until volume #{wait_idx+1};"
                    f"{self._img_name(fmri_img)}", tstamp, proc_delay)
                return

            # --- Volumes used in the GLM -------------------------------------
            # The latest n_win volumes (all volumes until the buffer is
            # full), in time order, and their rows in the buffers.
            n_win = min(n_proc, buf_len)
            win_vols = np.arange(vol_idx - n_win + 1, vol_idx + 1)
            win_rows_np = win_vols % buf_len
            win_rows = torch.from_numpy(win_rows_np).to(self.device)
            # The rows are contiguous unless the buffer has wrapped.
            contiguous = win_rows_np[-1] - win_rows_np[0] == n_win - 1

            # --- Censored volumes in the window ------------------------------
            first_fit = self.Y_mean is None
            n_cens = 0
            if self.censor_FD > 0:
                win_cens = self._censor_row[win_rows_np]
                n_cens = int(win_cens.sum())

            if n_cens > 0:
                # Uncensored volumes must be more than the regressors
                n_reg = reg0_num + \
                    min(1 + int(n_win*self.TR/150), self.max_poly_order) + 1
                n_keep = n_win - n_cens
                if n_keep <= n_reg:
                    if first_fit:
                        tstamp, proc_delay = \
                            self._record_proc_time(pre_proc_time)
                        self._log_proc(
                            f"#{vol_idx+1};Regression;"
                            f"Wait for more uncensored volumes ({n_keep} <="
                            f" {n_reg} regressors);"
                            f"{self._img_name(fmri_img)}", tstamp,
                            proc_delay)
                        return
                    elif not self._censor_warned:
                        self._logger.warning(
                            f"Only {n_keep} uncensored volumes for {n_reg}"
                            " regressors in the GLM window. The fit is"
                            " underdetermined (minimum-norm solution).")
                        self._censor_warned = True
                else:
                    self._censor_warned = False

                # Rows (in the window order) used in the fit
                keep_t = torch.from_numpy(~win_cens).to(self.device)

            # --- Update the RETROICOR regressors -----------------------------
            retro_ok = True
            if self.phys_reg != 'None' and self.rtp_physio is not None:
                retro_ok = self._update_retroicor(vol_idx, win_vols,
                                                  win_rows)

            # --- Fit the GLM -------------------------------------------------
            if first_fit:
                # Set Y_mean and scale the data in the window in place
                Ywin = self.YMtx[win_rows]
                if n_cens > 0:
                    self.Y_mean = Ywin[keep_t].mean(axis=0)
                else:
                    self.Y_mean = Ywin.mean(axis=0)
                self.Y_mean_mask = self.Y_mean.abs() > 1e-6

                Ywin[:, self.Y_mean_mask] = \
                    Ywin[:, self.Y_mean_mask] / \
                    self.Y_mean[self.Y_mean_mask] * 100
                Ywin.clamp_(max=200)
                Ywin[:, ~self.Y_mean_mask] = 0.0
                self.YMtx[win_rows] = Ywin
                del Ywin

            # Regressors of the window in time order, with the polynomials
            # for the window length
            polyreg = self.poly_reg(n_win, self.TR)
            polyreg_num = polyreg.shape[1]
            Xp = self.desMtx[win_rows, :reg0_num+polyreg_num].clone()
            Xp[:, reg0_num:reg0_num+polyreg_num] = \
                torch.from_numpy(polyreg).to(self.device)

            # Standardize the motion and ROI signal regressors
            norm_cols = [ii for ii, reg_name in enumerate(self.reg_names)
                         if reg_name in NORM_REGS]
            if len(norm_cols):
                reg = Xp[:, norm_cols]
                if n_cens > 0:
                    reg_mean = reg[keep_t].mean(axis=0)
                    reg_std = reg[keep_t].std(axis=0)
                else:
                    reg_mean = reg.mean(axis=0)
                    reg_std = reg.std(axis=0)
                # A constant regressor (std ~ 0) is zeroed out; the
                # least-squares solver ignores a zero column.
                valid = reg_std > 1e-6 * reg_mean.abs().clamp(min=1.0)
                reg = torch.where(valid, (reg - reg_mean) / reg_std,
                                  torch.zeros_like(reg))
                Xp[:, norm_cols] = reg

            if not retro_ok:
                # Exclude the physio regressors from this fit
                Xp[:, self.retrocols] = 0.0

            # Regressors for the fit: the rows of the censored volumes are
            # zeroed, which removes them from the least-squares fit (a zero
            # row adds a constant |y|^2 to the cost; with the SVD solver,
            # the corresponding rows of U are zero). Xp itself (used for the
            # residuals) is kept.
            if n_cens > 0:
                Xp_fit = Xp * keep_t[:, None].to(Xp.dtype)
            else:
                Xp_fit = Xp

            # Y of the window. Columns out of Y_mean_mask are zero, so their
            # residuals are 0.
            # The least-squares solution does not depend on the row order.
            # When the buffer is full, the whole buffer is used as it is
            # (no copy of Y), with the regressors placed at the same rows.
            if n_win == buf_len:
                Yp = self.YMtx
                Xfit = torch.empty_like(Xp)
                Xfit[win_rows] = Xp_fit
            elif contiguous:
                Yp = self.YMtx[win_rows_np[0]:win_rows_np[-1]+1]
                Xfit = Xp_fit
            else:
                Yp = self.YMtx[win_rows]
                Xfit = Xp_fit

            # Beta minimizing ||Y - XB||^2
            Beta = lstsq_SVDsolver(Xfit, Yp)

            if self.reg_retro_proc and first_fit:
                # Residuals of all the volumes in the window. The volumes
                # before the current one are processed retrospectively.
                Resids = (self.YMtx[win_rows] - torch.matmul(Xp, Beta)
                          ).cpu().numpy()
                self._retro_proc(fmri_img, vol_idx, Resids[:-1])
                Resid = Resids[-1, :]
                del Resids
            else:
                # Residual of the current volume only (the last row of Xp)
                Resid = (self.YMtx[row] - torch.matmul(Xp[-1, :], Beta)
                         ).cpu().numpy()

            del Beta

            # Replace the data of fmri_img with the residual
            vol_data = np.zeros_like(self.maskV, dtype=np.float32)
            vol_data[self.maskV] = Resid
            fmri_img.uncache()
            fmri_img._dataobj = vol_data
            fmri_img.set_data_dtype(np.float32)

            # --- Post process ------------------------------------------------
            tstamp, proc_delay = self._record_proc_time(pre_proc_time)
            fname = self._img_name(fmri_img)
            self._log_proc(f"#{vol_idx+1};Regression;{fname}", tstamp,
                           proc_delay)

            fmri_img.set_filename('regRes.' + fname)

            if self.next_proc:
                # Keep the processed data, since the next process can
                # modify fmri_img.
                self.proc_data = np.asanyarray(fmri_img.dataobj)
                save_name = fmri_img.get_filename()

                self.next_proc.do_proc(fmri_img, vol_idx=vol_idx,
                                       pre_proc_time=self._proc_time[-1],
                                       **self._qc_kwargs(qc))

            if self.save_proc:
                if self.next_proc is not None:
                    # Restore the processed data of this module
                    fmri_img.uncache()
                    fmri_img._dataobj = self.proc_data
                    fmri_img.set_data_dtype(np.float32)
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
    def _init_X_buffer(self):
        """
        Allocate the X buffer (self.desMtx) with the template regressors
        and room for the maximum number of polynomial regressors.
        """
        # Update self.desMtx0 and self.reg_names
        self.setup_regressor_template(self.desMtx_read,
                                      self.max_scan_length,
                                      self.col_names_read)
        # Number of rows of the X and Y buffers: max_scan_length, or fewer
        # if the memory is not enough (fail-safe; the buffers are then used
        # as ring buffers).
        nt = self.desMtx0.shape[0]
        pnum = min(1 + int(nt*self.TR/150), self.max_poly_order)
        ncols = self.desMtx0.shape[1] + pnum + 1
        nt = self._buffer_length(nt, int(np.sum(self.maskV)), ncols)

        desMtx = np.concatenate(
            [self.desMtx0[:nt], np.zeros((nt, pnum+1), dtype=np.float32)],
            axis=1)
        self.desMtx = self._to_device(desMtx)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _init_Y_buffer(self):
        """
        Allocate the Y buffer (self.YMtx) with the same number of rows as
        the X buffer. If the GPU memory cannot be allocated, the processing
        falls back to the CPU and the exception is raised again (the current
        volume is not processed).
        """
        vox_num = np.sum(self.maskV)
        self.YMtx = torch.empty(self.desMtx.shape[0], vox_num,
                                dtype=torch.float32)
        try:
            self.YMtx = self.YMtx.to(self.device)
        except Exception as e:
            self._logger.error(str(e))
            if self.device != 'cpu':
                self._error("Failed to allocate GPU memory for Y.")
                self.onGPU = False  # self.device is changed to 'cpu'
                self.desMtx = self.desMtx.to(self.device)
                self.YMtx = self.YMtx.to(self.device)
            else:
                self._error("Failed to allocate memory for Y.")
            raise e

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _volreg_enabled(self):
        """True if the motion parameters of volreg can be used for FD."""
        return self.volreg is not None and \
            getattr(self.volreg, 'enabled', True) and \
            hasattr(self.volreg, '_motion')

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _qc_init_state(self):
        """Reset the running QC state (not the QC table self.qc)."""
        self._qc_M = None  # Welford running mean (tensor, voxels)
        self._qc_M2 = None  # Welford sum of squared deviations
        self._qc_n = 0  # Number of volumes in the running statistics
        self._qc_prev_y = None  # Input data of the previous volume (tensor)
        self._qc_prev_mot = None  # Motion of the previous volume
        self._qc_first_idx = -1  # First and last volume indices in self.qc
        self._qc_last_idx = -1
        self._qc_nomot_warned = False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _init_qc(self):
        """
        Allocate the QC table (self.qc) for max_scan_length volumes (or the
        length of the design matrix) and the running statistics on the
        device. Called at the first received volume of a run.
        """
        self._qc_init_state()
        n = max(self.max_scan_length,
                self.desMtx0.shape[0] if self.desMtx0 is not None else 0)
        self.qc = {
            'FD': np.full(n, np.nan),
            'DVARS': np.full(n, np.nan),
            'tSNR': np.full(n, np.nan),
            'censored': np.zeros(n, dtype=bool),
            'received': np.zeros(n, dtype=bool)}
        vox_num = int(np.sum(self.maskV))
        self._qc_M = torch.zeros(vox_num, dtype=torch.float32,
                                 device=self.device)
        self._qc_M2 = torch.zeros_like(self._qc_M)
        self.qc_last = None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _update_qc(self, vol_idx, ydata):
        """
        Compute the QC measures (FD, DVARS, tSNR, censored; see the class
        docstring) of volume vol_idx and store them in self.qc.

        Parameters
        ----------
        vol_idx : int
            0-based volume index.
        ydata : 1D tensor
            Input data of the volume in the mask (not scaled). It is not
            modified.

        Returns
        -------
        dict
            {'FD': float, 'DVARS': float, 'tSNR': float, 'censored': bool}
        """
        # Extend the table if the scan is longer than max_scan_length
        nq = len(self.qc['FD'])
        if vol_idx >= nq:
            n = max(vol_idx + 1, int(nq * 1.25) + 100)
            self._logger.warning(
                f"The scan is longer than max_scan_length ({nq}). The QC"
                f" table is extended to {n} volumes.")
            for k, arr in self.qc.items():
                ext = np.full(n, np.nan) if arr.dtype.kind == 'f' else \
                    np.zeros(n, dtype=arr.dtype)
                ext[:nq] = arr
                self.qc[k] = ext

        # --- FD ---
        FD = np.nan
        if self._volreg_enabled() and \
                vol_idx < self.volreg._motion.shape[0]:
            mot = self.volreg._motion[vol_idx, :].astype(np.float64)
            if self._qc_prev_mot is None:
                FD = 0.0
            else:
                dmot = np.abs(mot - self._qc_prev_mot)
                # Columns: roll, pitch, yaw (degrees), dS, dL, dP (mm)
                FD = float(dmot[3:].sum() +
                           FD_HEAD_RADIUS * np.deg2rad(dmot[:3]).sum())
            self._qc_prev_mot = mot
        elif self.censor_FD > 0 and not self._qc_nomot_warned:
            self._logger.warning(
                "FD is not available (no VOLREG). Motion censoring is not"
                " applied.")
            self._qc_nomot_warned = True

        censored = bool(self.censor_FD > 0 and FD > self.censor_FD)

        # --- Running mean and variance (Welford) ---
        self._qc_n += 1
        delta = ydata - self._qc_M
        self._qc_M.add_(delta / self._qc_n)
        self._qc_M2.addcmul_(delta, ydata - self._qc_M)
        del delta

        # --- DVARS and tSNR (one device-to-host copy) ---
        nan_t = torch.tensor(float('nan'), device=ydata.device)
        if self._qc_prev_y is not None:
            ymean = ydata.mean()
            diff = ydata - self._qc_prev_y
            dvars = torch.sqrt(torch.mean(diff * diff)) / ymean * 100
            dvars = torch.where(ymean.abs() > 1e-6, dvars, nan_t)
            del diff
        else:
            dvars = nan_t
        # The input tensor is not modified later; keep the reference.
        self._qc_prev_y = ydata

        if self._qc_n >= 3:
            sd = torch.sqrt(self._qc_M2 / (self._qc_n - 1))
            tsnr_v = torch.where(sd > 0, self._qc_M / sd,
                                 torch.full_like(sd, float('nan')))
            tsnr = torch.nanmedian(tsnr_v)
            del sd, tsnr_v
        else:
            tsnr = nan_t

        DVARS, tSNR = torch.stack([dvars, tsnr]).cpu().tolist()

        # --- Store ---
        self.qc['FD'][vol_idx] = FD
        self.qc['DVARS'][vol_idx] = DVARS
        self.qc['tSNR'][vol_idx] = tSNR
        self.qc['censored'][vol_idx] = censored
        self.qc['received'][vol_idx] = True
        if self._qc_first_idx < 0:
            self._qc_first_idx = vol_idx
        self._qc_last_idx = max(self._qc_last_idx, vol_idx)

        self.qc_last = {'FD': FD, 'DVARS': DVARS, 'tSNR': tSNR,
                        'censored': censored}
        return self.qc_last

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _qc_at(self, vol_idx):
        """QC dict of volume vol_idx from self.qc."""
        return {'FD': float(self.qc['FD'][vol_idx]),
                'DVARS': float(self.qc['DVARS'][vol_idx]),
                'tSNR': float(self.qc['tSNR'][vol_idx]),
                'censored': bool(self.qc['censored'][vol_idx])}

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _qc_kwargs(self, qc):
        """
        Keyword arguments for next_proc.do_proc: {'qc': qc} if its do_proc
        accepts a 'qc' keyword (or **kwargs), otherwise {} (e.g., an
        application class overriding do_proc without **kwargs; it can read
        the QC from this object's qc_last or qc).
        """
        nproc, ok = self._qc_kw_cache
        if nproc is not self.next_proc:
            ok = False
            try:
                params = inspect.signature(
                    self.next_proc.do_proc).parameters.values()
                ok = any(pp.kind == pp.VAR_KEYWORD or pp.name == 'qc'
                         for pp in params)
            except (TypeError, ValueError):
                pass
            self._qc_kw_cache = (self.next_proc, ok)
        return {'qc': qc} if ok else {}

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_qc_table(self):
        """
        Return the QC table of the received volumes as a pandas DataFrame
        with the columns vol_idx (0-based), FD, DVARS, tSNR, and censored
        (0/1). None if no volume has been received. The table of the last
        run is kept after end_reset until the next run starts.
        """
        if self.qc is None or self._qc_first_idx < 0:
            return None
        idx = np.arange(self._qc_first_idx, self._qc_last_idx + 1)
        idx = idx[self.qc['received'][idx]]
        return pd.DataFrame({
            'vol_idx': idx,
            'FD': self.qc['FD'][idx],
            'DVARS': self.qc['DVARS'][idx],
            'tSNR': self.qc['tSNR'][idx],
            'censored': self.qc['censored'][idx].astype(int)},
            columns=QC_COLS)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save_qc(self, image_file=None):
        """
        Save the QC table as a CSV file. If image_file (the saved processed
        image) is given, the file is <image_file without .nii*>_QC.csv;
        otherwise, work_dir/RTP/RtpRegress_QC_<date_time>.csv.

        Returns
        -------
        Path or None
            Saved file (None if there is no QC data).
        """
        qc_df = self.get_qc_table()
        if qc_df is None or len(qc_df) == 0:
            return None

        if image_file is not None:
            save_f = Path(re.sub(r'\.nii(\.gz)?$', '', str(image_file)) +
                          '_QC.csv')
        else:
            save_f = Path(self.work_dir) / 'RTP' / \
                f"{self.__class__.__name__}_QC_" \
                f"{time.strftime('%Y%m%d_%H%M%S')}.csv"
        save_f.parent.mkdir(parents=True, exist_ok=True)
        qc_df.to_csv(save_f, index=False, float_format='%.6g')
        self._logger.info(f"Saved the QC table as {save_f}")
        return save_f

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _update_retroicor(self, vol_idx, win_vols, win_rows):
        """
        Set the RETROICOR regressors of the volumes in the window.

        Returns
        -------
        bool
            False if the regressors are not available; the current volume is
            then processed without them.
        """
        retrots = self.rtp_physio.get_retrots(
            self.TR, vol_idx+1, self.tshift, timeout=self.TR)
        if retrots is not None and (
                retrots.ndim != 2 or
                retrots.shape[0] < vol_idx+1 or
                retrots.shape[1] != len(self.retrocols)):
            self._logger.error(
                "RETROTS regressors have an unexpected shape"
                f" {retrots.shape}.")
            retrots = None

        if retrots is None:
            self._logger.warning(
                "RETROTS regressors cannot be made."
                f" Volume #{vol_idx+1} is processed without"
                " physio regressors.")
            return False

        retro_win = self._to_device(retrots[win_vols, :])
        for ii, icol in enumerate(self.retrocols):
            self.desMtx[win_rows, icol] = retro_win[:, ii]

        return True

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _retro_proc(self, fmri_img, vol_idx, Resids):
        """
        Pass the residuals of the volumes before the current one (rows of
        Resids, ending at vol_idx - 1) to the next process and keep them for
        saving.
        """
        vi0 = vol_idx - Resids.shape[0]
        self._logger.info(
            f"Retrospective regression for {vi0}-{vol_idx - 1}")

        # Filename template. The volume number in the current filename
        # (with its zero padding) is replaced with that of each volume.
        save_name_temp = 'regRes.' + self._img_name(fmri_img)
        ma = re.search(f"0*{vol_idx+1}", save_name_temp)
        vol_str = ma.group() if ma is not None else None

        for ii in range(Resids.shape[0]):
            vi = vi0 + ii  # 0-based volume index
            vol_data = np.zeros_like(self.maskV, dtype=np.float32)
            vol_data[self.maskV] = Resids[ii, :]
            retro_fmri_img = nib.Nifti1Image(vol_data, fmri_img.affine,
                                             header=fmri_img.header)
            if vol_str is not None:
                save_name = save_name_temp.replace(
                    vol_str, f"{vi+1}".zfill(len(vol_str)))
            else:
                save_name = save_name_temp.replace(
                    '.nii', f"_{vi+1:04d}.nii")

            retro_fmri_img.set_filename(save_name)

            # Keep the volume before the next process, which may change
            # the data and the filename of retro_fmri_img.
            if self.save_proc:
                self.keep_processed_image(retro_fmri_img, vol_num=vi)

            if self.next_proc:
                self.next_proc.do_proc(retro_fmri_img, vi, time.time(),
                                       **self._qc_kwargs(self._qc_at(vi)))

            time.sleep(0.005)  # Not to send the data too fast

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _record_proc_time(self, pre_proc_time):
        """
        Record the time stamp of the end of the processing.

        Returns
        -------
        tstamp : float
            Time stamp.
        proc_delay : float or None
            Time from pre_proc_time; None if pre_proc_time is None.
        """
        tstamp = time.time()
        self._proc_time.append(tstamp)
        proc_delay = None
        if pre_proc_time is not None:
            proc_delay = tstamp - pre_proc_time
            if self.save_delay:
                self.proc_delay.append(proc_delay)

        return tstamp, proc_delay

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _log_proc(self, msg, tstamp, proc_delay):
        msg += f";tstamp={tstamp}"
        if proc_delay is not None:
            msg += f";took {proc_delay:.4f}s"
        self._logger.info(msg)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def _img_name(fmri_img):
        """Filename (without the directory) of fmri_img."""
        if fmri_img.get_filename():
            return Path(fmri_img.get_filename()).name
        return "unknown.nii.gz"

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end_reset(self):
        """End the process and reset the run state."""

        self._logger.info(f"Reset {self.__class__.__name__} module.")

        # A mask from a file is kept; a mask made from the first volume is
        # made again in the next run.
        if not isinstance(self.mask_file, (str, Path)):
            self.maskV = None

        self.desMtx = None
        self.mot0 = None
        self.YMtx = None
        self.Y_mean = None
        self._ring_warned = False
        self.Y_mean_mask = None
        self.GS_maskdata = None
        self.WM_maskdata = None
        self.Vent_maskdata = None
        self._censor_row = None
        self._censor_warned = False

        out_files = super(RtpRegress, self).end_reset()

        # Save the QC table next to the processed image (when save_proc is
        # True, as the processed image). The table (self.qc) is kept until
        # the next run starts; the running state is reset.
        try:
            if self.save_proc:
                if out_files is None:
                    out_files = {}
                key = f"{self.__class__.__name__}:processed image"
                qc_f = self.save_qc(out_files.get(key))
                if qc_f is not None:
                    out_files[f"{self.__class__.__name__}:QC"] = qc_f
        except Exception:
            self._logger.error("Failed to save the QC table:\n"
                               + traceback.format_exc())

        first_idx, last_idx = self._qc_first_idx, self._qc_last_idx
        self._qc_init_state()
        # Keep the range of the table for get_qc_table
        self._qc_first_idx, self._qc_last_idx = first_idx, last_idx

        return out_files

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_mask(self, maskdata, sub_i=0, method='zero_out'):
        """
        Set the processing mask (self.maskV) to the non-zero voxels of
        maskdata.

        Parameters
        ----------
        maskdata : str, Path, or array
            Mask file or volume data.
        sub_i : int, optional
            Sub-volume index of a 4D mask file. The default is 0.
        method : str, optional
            Only 'zero_out' is supported. The default is 'zero_out'.
        """
        if isinstance(maskdata, (str, Path)):
            if not Path(maskdata).is_file():
                self._error(f"Mask file is not found: {maskdata}")
                self.mask_file = 0
                return

            self._logger.info(f"Mask = {maskdata}")
            maskdata = np.squeeze(nib.load(maskdata).get_fdata())

            if maskdata.ndim > 3:
                maskdata = maskdata[:, :, :, sub_i]

        if method == 'zero_out':
            self.maskV = maskdata != 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def read_roi_mask(self, reg, dataV):
        """
        Read the ROI mask for the mean signal regressor reg ('GS', 'WM', or
        'Vent'). If the mask is not available or its shape does not match the
        functional image, the regressor option (e.g., self.GS_reg) is reset to
        False with an error message.
        """
        mask_f = getattr(self, f"{reg}_mask")
        errmsg = None
        if mask_f is None or not Path(mask_f).is_file():
            errmsg = f"{reg}_mask file is not found: {mask_f}"
        else:
            mask_img = nib.load(mask_f)
            if not np.all(dataV.shape == mask_img.shape):
                errmsg = f"{reg} mask shape {mask_img.shape} !="
                errmsg += f" functional image shape {dataV.shape}"
            else:
                setattr(self, f"{reg}_maskdata",
                        (mask_img.get_fdata() != 0)[self.maskV])

        if errmsg is not None:
            self._error(errmsg + f"\n{reg}_reg is reset to False.")
            setattr(self, f"{reg}_reg", False)
            # Rebuild the design matrix without the regressor
            self.desMtx = None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def setup_regressor_template(self, desMtx_read=None, max_scan_length=0,
                                 col_names_read=[]):
        """
        Make the regressor template self.desMtx0 (desMtx_read followed by
        the motion, physio, and ROI signal regressors, without the
        polynomials) and the regressor names self.reg_names. The column
        indices of the regressors are set to self.motcols, self.retrocols,
        and self.<GS|WM|Vent>_col.

        The template has max_scan_length rows, or the length of desMtx_read
        if it is longer. The nuisance regressor columns are zero; they are
        filled during the scan.
        """
        if desMtx_read is None and max_scan_length == 0:
            self._error('Either desMtx or max_scan_length must be given.')
            return

        col_names = col_names_read.copy()

        # Design matrix given by the user
        if desMtx_read is None:
            desMtx = np.zeros([max_scan_length, 0], dtype=np.float32)
        else:
            if max_scan_length <= desMtx_read.shape[0]:
                max_scan_length = desMtx_read.shape[0]
                desMtx = np.asarray(desMtx_read, dtype=np.float32).copy()
            else:
                desMtx = np.zeros([max_scan_length, desMtx_read.shape[1]],
                                  dtype=np.float32)
                desMtx[:desMtx_read.shape[0], :] = desMtx_read

            # Match the number of column names to the matrix columns
            ncols = desMtx.shape[1]
            col_names = col_names[:ncols]
            col_names += [f"Reg{ii+2}" for ii in range(len(col_names), ncols)]

        def append_regs(names):
            nonlocal desMtx
            desMtx = np.concatenate(
                [desMtx, np.zeros([max_scan_length, len(names)])], axis=1)
            col_names.extend(names)

        # Append the nuisance regressors
        if self.mot_reg != 'None':
            self.motcols = []
            if self.mot_reg in ('mot6', 'mot12'):
                append_regs(MOT_NAMES)
                self.motcols = [ii for ii, cn in enumerate(col_names)
                                if cn in MOT_NAMES]

            if self.mot_reg in ('mot12', 'dmot6'):
                append_regs(DMOT_NAMES)
                self.motcols.extend([ii for ii, cn in enumerate(col_names)
                                     if cn in DMOT_NAMES])

        if self.phys_reg != 'None':
            col_add = PHYS_REG_NAMES[self.phys_reg]
            append_regs(col_add)
            self.retrocols = \
                [ii for ii, cn in enumerate(col_names) if cn in col_add]

        for reg in ROI_REGS:
            if getattr(self, f"{reg}_reg"):
                append_regs([reg])
                setattr(self, f"{reg}_col", col_names.index(reg))

        self.desMtx0 = desMtx
        self.reg_names = col_names

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def legendre(self, x, m):
        """
        Legendre polynomial of order m, ported from misc_math.c in the AFNI
        source.

        Parameters
        ----------
        x : 1D array
            Values in [-1, 1].
        m : int
            Polynomial order.
        """

        if m < 0:
            return None  # bad input
        elif m == 0:
            return np.ones_like(x)
        elif m == 1:
            return x
        elif m == 2:
            return (3.0 * x * x - 1.0)/2.0
        elif m == 3:
            return (5.0 * x * x - 3.0) * x/2.0
        elif m == 4:
            return ((35.0 * x * x - 30.0) * x * x + 3.0)/8.0
        elif m == 5:
            return ((63.0 * x * x - 70.0) * x * x + 15.0) * x/8.0
        elif m == 6:
            return (((231.0 * x * x - 315.0) * x * x + 105.0) * x * x - 5.0) \
                   / 16.0
        elif m == 7:
            return (((429.0 * x * x - 693.0) * x * x + 315.0) * x * x - 35.0) \
                   * x/16.0
        elif m == 8:
            return ((((6435.0 * x * x - 12012.0) * x * x + 6930.0) * x * x
                    - 1260.0) * x * x + 35.0) / 128.0
        elif m == 9:
            #  Feb 2005: this part generated by Maple, then hand massaged
            return (0.24609375e1 +
                    (-0.3609375e2 +
                     (0.140765625e3 +
                      (-0.20109375e3 +
                       0.949609375e2 * x * x) * x * x) * x * x) * x * x) * x
        elif m == 10:
            return -0.24609375e0 + \
                (0.1353515625e2 +
                 (-0.1173046875e3 +
                  (0.3519140625e3 +
                   (-0.42732421875e3 +
                    0.18042578125e3 * x * x)
                   * x * x) * x * x) * x * x) * x * x
        elif m == 11:
            return (-0.270703125e1 +
                    (0.5865234375e2 +
                     (-0.3519140625e3 +
                      (0.8546484375e3 +
                       (-0.90212890625e3 +
                        0.34444921875e3 * x * x)
                       * x * x) * x * x) * x * x) * x * x)
        elif m == 12:
            return 0.2255859375e0 + \
                (-0.17595703125e2 +
                 (0.2199462890625e3 +
                  (-0.99708984375e3 +
                   (0.20297900390625e4 +
                    (-0.1894470703125e4 +
                     0.6601943359375e3 * x * x) * x * x)
                   * x * x) * x * x) * x * x) * x * x
        elif m == 13:
            return (0.29326171875e1 +
                    (-0.87978515625e2 +
                     (0.7478173828125e3 +
                      (-0.270638671875e4 +
                       (0.47361767578125e4 +
                        (-0.3961166015625e4
                         + 0.12696044921875e4 * x * x) * x * x) * x * x)
                      * x * x) * x * x) * x * x) * x
        elif m == 14:
            return -0.20947265625e0 + \
                (0.2199462890625e2 +
                 (-0.37390869140625e3 +
                  (0.236808837890625e4 +
                   (-0.710426513671875e4 +
                    (0.1089320654296875e5 +
                     (-0.825242919921875e4 +
                      0.244852294921875e4 * x * x) * x * x) * x * x)
                   * x * x) * x * x) * x * x) * x * x
        elif m == 15:
            return (-0.314208984375e1 +
                    (0.12463623046875e3 +
                     (-0.142085302734375e4 +
                      (0.710426513671875e4 +
                       (-0.1815534423828125e5 +
                        (0.2475728759765625e5 +
                         (-0.1713966064453125e5 +
                          0.473381103515625e4 * x * x) * x * x) * x * x)
                       * x * x) * x * x) * x * x) * x * x) * x
        elif m == 16:
            return 0.196380615234375e0 + \
                (-0.26707763671875e2 +
                 (0.5920220947265625e3 +
                  (-0.4972985595703125e4 +
                   (0.2042476226806641e5 +
                    (-0.4538836059570312e5 +
                     (0.5570389709472656e5 +
                      (-0.3550358276367188e5 +
                       0.9171758880615234e4 * x * x) * x * x) * x * x)
                    * x * x) * x * x) * x * x) * x * x) * x * x
        elif m == 17:
            return (0.3338470458984375e1 +
                    (-0.169149169921875e3 +
                     (0.2486492797851562e4 +
                      (-0.1633980981445312e5 +
                       (0.5673545074462891e5 +
                        (-0.1114077941894531e6 +
                         (0.1242625396728516e6 +
                          (-0.7337407104492188e5 +
                           0.1780400253295898e5 * x * x) * x * x) * x * x)
                        * x * x) * x * x) * x * x) * x * x) * x * x) * x
        elif m == 18:
            return -0.1854705810546875e0 + \
                (0.3171546936035156e2 +
                 (-0.8880331420898438e3 +
                  (0.9531555725097656e4 +
                   (-0.5106190567016602e5 +
                    (0.153185717010498e6 +
                     (-0.2692355026245117e6 +
                      (0.275152766418457e6 +
                       (-0.1513340215301514e6 +
                        0.3461889381408691e5 * x * x) * x * x) * x * x)
                     * x * x) * x * x) * x * x) * x * x) * x * x) * x * x
        elif m == 19:
            return (-0.3523941040039062e1 +
                    (0.2220082855224609e3 +
                     (-0.4084952453613281e4 +
                      (0.3404127044677734e5 +
                       (-0.153185717010498e6 +
                        (0.4038532539367676e6 +
                         (-0.6420231216430664e6 +
                          (0.6053360861206055e6 +
                           (-0.3115700443267822e6 +
                            0.6741574058532715e5 * x * x) * x * x)
                          * x * x) * x * x) * x * x) * x * x) * x * x)
                        * x * x) * x * x) * x
        elif m == 20:
            return 0.1761970520019531e0 + \
                (-0.3700138092041016e2 +
                 (0.127654764175415e4 +
                  (-0.1702063522338867e5 +
                   (0.1148892877578735e6 +
                    (-0.4442385793304443e6 +
                     (0.1043287572669983e7 +
                      (-0.1513340215301514e7 +
                       (0.1324172688388824e7 +
                        (-0.6404495355606079e6 +
                         0.1314606941413879e6 * x * x) * x * x)
                       * x * x) * x * x) * x * x) * x * x) * x * x) * x * x)
                    * x * x) * x * x
        else:
            # if here, m > 20 ==> use recurrence relation
            pk = 0
            pkm2 = self.legendre(x, 19)
            pkm1 = self.legendre(x, 20)
            for k in range(21, m+1):
                pk = ((2.0 * k - 1.0) * x * pkm1 - (k - 1.0) * pkm2) / k
                pkm2 = pkm1
                pkm1 = pk

            return pk

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _buffer_length(self, nt, vox_num, ncols):
        """Number of rows (volumes) of the X and Y buffers.

        nt (max_scan_length) rows are used if the memory allows. Otherwise,
        as a fail-safe, the rows that fit in half of the free memory are
        used; the buffers are then ring buffers and the GLM is fitted with
        the latest volumes.
        """
        row_bytes = 4 * (vox_num + ncols)  # float32
        try:
            if self.max_buffer_bytes is not None:
                budget = self.max_buffer_bytes
            elif str(self.device).startswith('cuda'):
                free, _ = torch.cuda.mem_get_info(torch.device(self.device))
                budget = 0.5 * free
            else:
                budget = 0.5 * psutil.virtual_memory().available
        except Exception:
            return nt

        n_fit = int(budget // row_bytes)
        if n_fit >= nt:
            return nt

        # Keep enough rows for the GLM
        n_fit = max(n_fit, self.wait_num + 10)
        self._logger.warning(
            f"Memory is not enough for {nt} volumes (max_scan_length)."
            f" The GLM is fitted with the latest {n_fit} volumes.")
        return n_fit

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def poly_reg(self, nt, TR):
        """
        Make the Legendre polynomial regressors for nt volumes. The order is
        min(1 + int(nt * TR / 150), self.max_poly_order).

        Parameters
        ----------
        nt : int
            Number of volumes.
        TR : float
            Repetition time (seconds).

        Returns
        -------
        polyreg : 2D array (nt x (order + 1)) or None
            Legendre polynomial regressors (None if nt < 1).
        """
        if nt < 1:
            return None

        pnum = min(1 + int(nt*TR/150), self.max_poly_order)
        polyreg = np.ndarray((nt, pnum+1), dtype=np.float32)
        xx = np.linspace(-1.0, 1.0, nt)
        for po in range(pnum+1):
            polyreg[:, po] = self.legendre(xx, po)

        return polyreg

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_reg_num(self):
        """
        Update the regressor template (self.desMtx0) and return the number of
        regressors including the polynomials at the scan length where the
        regression can start, i.e., the smallest length n with more volumes
        than regressors (the polynomial order increases with n as in
        poly_reg).
        """
        self.setup_regressor_template(self.desMtx_read,
                                      max_scan_length=self.max_scan_length,
                                      col_names_read=self.col_names_read)
        numReg = self.desMtx0.shape[1]

        def poly_num(nt):
            return min(1 + int(nt*self.TR/150), self.max_poly_order) + 1

        # The number of regressors increases by at most one per volume
        # (TR < 150 s), so the first n with n > numReg + poly_num(n) is
        # found by increasing n.
        nt = numReg + poly_num(0) + 1
        while nt <= numReg + poly_num(nt):
            nt += 1

        return numReg + poly_num(nt)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_wait_num(self, wait_num=None):
        """
        Set self.wait_num. It is at least the number of regressors + 1.
        If wait_num is None, the current self.wait_num is checked.
        """
        min_wait_num = self.get_reg_num() + 1

        if wait_num is None:
            wait_num = max(self.wait_num, min_wait_num)

        elif wait_num < min_wait_num:
            self._logger.info(f"Wait {min_wait_num} volumes"
                              " (number of regressors + 1).")
            wait_num = min_wait_num

        if wait_num != self.wait_num:
            if hasattr(self, 'ui_waitNum_lb'):
                self.ui_waitNum_lb.setText(
                    f"Wait REGRESS until receiving {wait_num} volumes")

            self.wait_num = wait_num

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _update_wait_num(self):
        """Update wait_num after the number of regressors has changed."""
        if hasattr(self, 'ui_waitNum_cmbBx'):
            if 'regressor' in self.ui_waitNum_cmbBx.currentText():
                self.set_wait_num(self.get_reg_num()+1)
        else:
            self.set_wait_num()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def _poly_order_text(max_order):
        """Label text for the maximum polynomial order."""
        txt = 'Increase the polynomial order with the scan length'
        if not np.isinf(max_order):
            txt += f' up to {max_order}'
        return txt

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_param(self, attr, val=None, reset_fn=None, echo=False):
        """
        Set a parameter and update the UI.

        When reset_fn is None, set_param is considered to be called from the
        load_parameters function (not from the UI). reset_fn is the UI
        widget's setter used to restore its value when the input is
        cancelled or invalid.
        """
        self._logger.debug(f"set_param: {attr} = {val}")

        # -- Check the value --
        if attr == 'enabled':
            if hasattr(self, 'ui_enabled_rdb'):
                self.ui_enabled_rdb.setChecked(val)

            if hasattr(self, 'ui_objs'):
                for ui in self.ui_objs:
                    ui.setEnabled(val)

            if self.desMtx_read is None and hasattr(self, 'ui_showDesMtx_btn'):
                self.ui_showDesMtx_btn.setEnabled(False)

        elif attr == 'work_dir':
            if val is None or not Path(val).is_dir():
                return

            val = Path(val)
            setattr(self, attr, val)

            if self.main_win is not None:
                self.main_win.set_workDir(val)

        elif attr == 'mask_file':
            if type(val) is int and val == 0:
                if hasattr(self, 'ui_mask_lnEd'):
                    self.ui_mask_lnEd.setText(MASK0_TEXT)

            elif type(val) is str and 'initial volume' in val:
                val = 0
                if hasattr(self, 'ui_mask_lnEd'):
                    self.ui_mask_lnEd.setText(MASK0_TEXT)

            elif type(val) is str and 'external file' in val:
                fname = self.select_file_dlg('REGRESS: Select mask volume',
                                             self.work_dir, "*.BRIK* *.nii*")
                if fname[0] == '':
                    if reset_fn:
                        reset_fn(1)
                    return -1

                mask_img = nib.load(fname[0])
                mask_fname = fname[0]
                if len(mask_img.shape) > 3 and mask_img.shape[3] > 1:
                    num, okflag = QtWidgets.QInputDialog.getInt(
                        None, "Select sub-volume",
                        "sub-volume index (0 is first)", 0, 0,
                        mask_img.shape[3])
                    if not okflag:
                        if reset_fn:
                            reset_fn(1)
                        return

                    mask_fname += f"[{num}]"
                else:
                    num = 0

                self.set_mask(fname[0], num)
                if hasattr(self, 'ui_mask_lnEd'):
                    self.ui_mask_lnEd.setText(str(mask_fname))
                val = mask_fname

            elif type(val) is str or isinstance(val, Path):
                # Mask file, optionally with a sub-volume index: 'file[n]'
                val = str(val)
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
                    self.ui_mask_cmbBx.setCurrentIndex(0)

                self.set_mask(fname, num)
                if hasattr(self, 'ui_mask_lnEd'):
                    self.ui_mask_lnEd.setText(str(val))

        elif attr == 'wait_num':
            if type(val) is int:
                self.set_wait_num(val)
                if reset_fn is None and hasattr(self, 'ui_waitNum_cmbBx'):
                    if self.wait_num == self.get_reg_num()+1:
                        self.ui_waitNum_cmbBx.setCurrentIndex(0)
                    else:
                        self.ui_waitNum_cmbBx.setCurrentIndex(1)
                return

            elif 'regressor' in val:
                self.set_wait_num(self.get_reg_num()+1)

            elif 'set' in val:
                num0 = self.get_reg_num()+1
                num1, okflag = QtWidgets.QInputDialog.getInt(
                    None, "Wait REGRESS until", "volume",
                    self.wait_num, num0)
                if not okflag:
                    return

                self.set_wait_num(num1)

            val = self.wait_num

        elif attr == 'max_scan_length':
            if self.desMtx_read is not None and \
                    self.desMtx_read.shape[0] > val:
                val = self.desMtx_read.shape[0]
                if reset_fn:
                    reset_fn(val)

            # The scan must be longer than wait_num for the regression
            if val <= self.wait_num:
                val = self.wait_num + 1
                if reset_fn:
                    reset_fn(val)

            # Update self.desMtx0
            self.setup_regressor_template(self.desMtx_read,
                                          max_scan_length=val,
                                          col_names_read=self.col_names_read)

            if reset_fn is None and hasattr(self, 'ui_maxLen_spBx'):
                self.ui_maxLen_spBx.setValue(val)

        elif attr == 'max_poly_order':
            if val == 'auto':
                if hasattr(self, 'ui_maxPoly_lb'):
                    self.ui_maxPoly_lb.setText(self._poly_order_text(np.inf))
                val = np.inf
            elif val == 'set':
                num, okflag = QtWidgets.QInputDialog.getInt(
                    None, "Maximum polynomial order", "enter value")
                if not okflag:
                    if np.isinf(self.max_poly_order) and reset_fn:
                        reset_fn(0)
                    return

                if hasattr(self, 'ui_maxPoly_lb'):
                    self.ui_maxPoly_lb.setText(self._poly_order_text(num))
                val = num
            elif np.isinf(val) and reset_fn is None:
                if hasattr(self, 'ui_maxPoly_cmbBx'):
                    self.ui_maxPoly_cmbBx.setCurrentIndex(0)
                    self.ui_maxPoly_lb.setText(self._poly_order_text(val))
            elif type(val) is int and reset_fn is None:
                if hasattr(self, 'ui_maxPoly_cmbBx'):
                    self.ui_maxPoly_cmbBx.setCurrentIndex(1)
                    self.ui_maxPoly_lb.setText(self._poly_order_text(val))

            setattr(self, attr, val)
            self._update_wait_num()

        elif attr == 'mot_reg':
            # Convert a UI text to the option value
            if val.lower() == 'none':
                val = 'None'
            elif val.startswith('6 motions'):
                val = 'mot6'
            elif val.startswith('12 motions'):
                val = 'mot12'
            elif val.startswith('6 motion derivatives'):
                val = 'dmot6'

            if reset_fn is None and hasattr(self, 'ui_motReg_cmbBx') and \
                    val in MOT_REG_OPTS:
                self.ui_motReg_cmbBx.setCurrentIndex(MOT_REG_OPTS.index(val))

            setattr(self, attr, val)
            self._update_wait_num()

        elif attr in ('GS_reg', 'WM_reg', 'Vent_reg'):
            setattr(self, attr, val)
            if hasattr(self, f"ui_{attr}_chb"):
                getattr(self, f"ui_{attr}_chb").setChecked(val)

            self._update_wait_num()

        elif attr in ('GS_mask', 'WM_mask', 'Vent_mask'):
            if reset_fn is not None:
                # Called from the UI: select the file with a dialog that
                # opens at val (the directory of the current file).
                if str(val) != '.' and Path(val).is_dir():
                    startdir = val
                else:
                    startdir = self.work_dir

                roi_name = {'GS_mask': 'global signal',
                            'WM_mask': 'white matter',
                            'Vent_mask': 'ventricle'}[attr]
                fname = self.select_file_dlg(
                    f"REGRESS: Select {roi_name} mask", startdir,
                    "*.BRIK* *.nii*")
                if fname[0] == '':
                    return -1

                val = fname[0]
                reset_fn(str(val))

            elif type(val) is str or isinstance(val, Path):
                if not Path(val).is_file():
                    val = ''

                if hasattr(self, f"ui_{attr}_lnEd"):
                    getattr(self, f"ui_{attr}_lnEd").setText(str(val))

        elif attr == 'phys_reg':
            # Convert a UI text to the option value
            if val.lower() == 'none':
                val = 'None'
            elif val.startswith('8 RICOR'):
                val = 'RICOR8'
            elif val.startswith('5 RVT'):
                val = 'RVT5'
            elif val.startswith('13 RVT+RICOR'):
                val = 'RVT+RICOR13'

            if val not in ('None', 'RICOR8'):
                # RetroTs makes only the 8 RETROICOR regressors.
                self._error(
                    f"Physio regressor '{val}' is not supported."
                    f" phys_reg is kept as '{self.phys_reg}'.")
                if reset_fn:
                    reset_fn({'None': 0, 'RICOR8': 1}.get(self.phys_reg, 0))
                return

            if reset_fn is None and hasattr(self, 'ui_physReg_cmbBx'):
                self.ui_physReg_cmbBx.setCurrentIndex(
                    PHYS_REG_OPTS.index(val))

            setattr(self, attr, val)
            self._update_wait_num()
            return

        elif attr == 'desMtx_f':
            if val is None:
                return

            # 'set' from the button toggles loading and unsetting
            btn_unset = hasattr(self, 'ui_loadDesMtx_btn') and \
                'Unset' in self.ui_loadDesMtx_btn.text()
            if (val == 'set' and not btn_unset) or Path(val).is_file():
                # Load a design matrix
                if val == 'set':
                    fname = self.select_file_dlg(
                        'REGRESS: Select design matrix file',
                        self.work_dir, "*.csv")
                    if fname[0] == '':
                        return -1
                    fname = fname[0]
                else:
                    fname = val

                self.desMtx_f = fname

                # A non-numeric first line is a header of column names
                with open(fname, 'r') as fd:
                    ll = fd.readline()
                cells = [cc.strip() for cc in ll.split(',')]
                try:
                    [float(cc) for cc in cells]
                    skiprows = 0
                except ValueError:
                    self.col_names_read = cells
                    skiprows = 1

                self.desMtx_read = np.loadtxt(fname, delimiter=',',
                                              skiprows=skiprows)
                if self.desMtx_read.ndim == 1:
                    self.desMtx_read = self.desMtx_read[:, np.newaxis]

                if hasattr(self, 'ui_showDesMtx_btn'):
                    self.ui_showDesMtx_btn.setEnabled(True)

                if self.desMtx_read.shape[0] > self.max_scan_length:
                    self.set_param('max_scan_length',
                                   self.desMtx_read.shape[0])

                if hasattr(self, 'ui_loadDesMtx_btn'):
                    self.ui_loadDesMtx_btn.setText('Unset')

            elif val == 'unset' or (val == 'set' and btn_unset):
                # Unset the design matrix
                self.desMtx_read = None
                self.desMtx_f = None
                self.col_names_read = []
                if hasattr(self, 'ui_loadDesMtx_btn'):
                    self.ui_loadDesMtx_btn.setText('Set')

                if hasattr(self, 'ui_showDesMtx_btn'):
                    self.ui_showDesMtx_btn.setEnabled(False)

            # Update self.desMtx0
            if self.desMtx_read is not None or self.max_scan_length > 0:
                self.setup_regressor_template(
                    self.desMtx_read, max_scan_length=self.max_scan_length,
                    col_names_read=self.col_names_read)

            self._update_wait_num()
            return

        elif attr == 'showDesMtx':
            if self.desMtx_read is not None:
                self.plt_win = MatplotlibWindow()
                self.plt_win.setWindowTitle('Design matrix')
                ax = self.plt_win.canvas.figure.subplots(1, 1)
                ax.matshow(self.desMtx_read, cmap='gray')
                ax.set_aspect('auto')

                self.plt_win.show()
                self.plt_win.canvas.draw()
                self.plt_win.canvas.start_event_loop(0.005)

            return

        elif attr == 'censor_FD':
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = np.nan
            if not np.isfinite(val) or val < 0:
                self._logger.error(
                    f"censor_FD must be >= 0 (0: off). Got {val}.")
                if reset_fn:
                    reset_fn(self.censor_FD)
                return
            if reset_fn is None and hasattr(self, 'ui_censorFD_spBx'):
                self.ui_censorFD_spBx.setValue(val)

        elif attr == 'reg_retro_proc':
            if hasattr(self, 'ui_retroProc_chb'):
                self.ui_retroProc_chb.setChecked(val)

        elif attr == 'save_proc':
            if hasattr(self, 'ui_saveProc_chb'):
                self.ui_saveProc_chb.setChecked(val)

        elif reset_fn is None:
            # Ignore an unrecognized parameter
            if not hasattr(self, attr):
                self._logger.error(f"{attr} is not a recognized parameter.")
                return

        # -- Set the value --
        setattr(self, attr, val)
        if echo:
            print(f"{self.__class__.__name__}." + attr, '=',
                  getattr(self, attr))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ui_set_param(self):
        """
        Return the UI rows for setting the parameters. The UI widgets call
        set_param with their setter as reset_fn.
        """

        ui_rows = []
        self.ui_objs = []

        # enabled
        self.ui_enabled_rdb = QtWidgets.QRadioButton("Enable")
        self.ui_enabled_rdb.setChecked(self.enabled)
        self.ui_enabled_rdb.toggled.connect(
            lambda checked: self.set_param('enabled', checked,
                                           self.ui_enabled_rdb.setChecked))
        ui_rows.append((self.ui_enabled_rdb, None))

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

        self.ui_mask_lnEd = QtWidgets.QLineEdit()
        self.ui_mask_lnEd.setReadOnly(True)
        self.ui_mask_lnEd.setStyleSheet('border: 0px none;')
        self.ui_objs.extend([var_lb, self.ui_mask_cmbBx,
                             self.ui_mask_lnEd])

        if self.mask_file == 0:
            self.ui_mask_cmbBx.setCurrentIndex(1)
            self.ui_mask_lnEd.setText(MASK0_TEXT)
        else:
            self.ui_mask_cmbBx.setCurrentIndex(0)
            self.ui_mask_lnEd.setText(str(self.mask_file))

        mask_hLayout = QtWidgets.QHBoxLayout()
        mask_hLayout.addWidget(self.ui_mask_cmbBx)
        mask_hLayout.addWidget(self.ui_mask_lnEd)
        ui_rows.append((var_lb, mask_hLayout))

        # wait_num
        var_lb = QtWidgets.QLabel("Wait REGRESS until (volumes) :")
        self.ui_waitNum_cmbBx = QtWidgets.QComboBox()
        self.ui_waitNum_cmbBx.addItems(['number of regressors', 'set value'])
        self.ui_waitNum_cmbBx.activated.connect(
            lambda idx:
            self.set_param('wait_num',
                           self.ui_waitNum_cmbBx.currentText(),
                           self.ui_waitNum_cmbBx.setCurrentIndex))

        self.ui_waitNum_lb = QtWidgets.QLabel()
        self.set_wait_num()
        self.ui_waitNum_lb.setText(
            f'Wait REGRESS until receiving {self.wait_num} volumes')
        self.ui_objs.extend([var_lb, self.ui_waitNum_cmbBx,
                             self.ui_waitNum_lb])

        wait_num_hLayout = QtWidgets.QHBoxLayout()
        wait_num_hLayout.addWidget(self.ui_waitNum_cmbBx)
        wait_num_hLayout.addWidget(self.ui_waitNum_lb)
        ui_rows.append((var_lb, wait_num_hLayout))

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

        # max_poly_order
        var_lb = QtWidgets.QLabel("Maximum polynomial order :\n"
                                  "regressors for slow fluctuation")
        self.ui_maxPoly_cmbBx = QtWidgets.QComboBox()
        self.ui_maxPoly_cmbBx.addItems(['auto', 'set'])
        self.ui_maxPoly_cmbBx.activated.connect(
            lambda idx:
            self.set_param('max_poly_order',
                           self.ui_maxPoly_cmbBx.currentText(),
                           self.ui_maxPoly_cmbBx.setCurrentIndex))

        self.ui_maxPoly_lb = QtWidgets.QLabel()
        self.ui_objs.extend([var_lb, self.ui_maxPoly_cmbBx,
                             self.ui_maxPoly_lb])
        if np.isinf(self.max_poly_order):
            self.ui_maxPoly_cmbBx.setCurrentIndex(0)
        else:
            self.ui_maxPoly_cmbBx.setCurrentIndex(1)
        self.ui_maxPoly_lb.setText(
            self._poly_order_text(self.max_poly_order))

        maxPoly_hLayout = QtWidgets.QHBoxLayout()
        maxPoly_hLayout.addWidget(self.ui_maxPoly_cmbBx)
        maxPoly_hLayout.addWidget(self.ui_maxPoly_lb)
        ui_rows.append((var_lb, maxPoly_hLayout))

        # mot_reg (the item order follows MOT_REG_OPTS)
        var_lb = QtWidgets.QLabel("Motion regressor :")
        self.ui_motReg_cmbBx = QtWidgets.QComboBox()
        self.ui_motReg_cmbBx.addItems(
            ['None', '6 motions (yaw, pitch, roll, dS, dL, dP)',
             '12 motions (6 motions and their temporal derivatives)',
             '6 motion derivatives'])
        self.ui_motReg_cmbBx.setCurrentIndex(
            MOT_REG_OPTS.index(self.mot_reg))
        self.ui_motReg_cmbBx.currentIndexChanged.connect(
            lambda idx:
            self.set_param('mot_reg',
                           self.ui_motReg_cmbBx.currentText(),
                           self.ui_motReg_cmbBx.setCurrentIndex))
        ui_rows.append((var_lb, self.ui_motReg_cmbBx))
        self.ui_objs.extend([var_lb, self.ui_motReg_cmbBx])

        # ROI mean signal regressors
        ui_rows.append(self._ui_roi_reg_row('GS', "Regress global signal :"))
        ui_rows.append(self._ui_roi_reg_row('WM', "Regress WM signal :"))
        ui_rows.append(self._ui_roi_reg_row('Vent', "Regress Vent signal :"))

        # phys_reg (the item order follows PHYS_REG_OPTS; RVT is not
        # supported)
        var_lb = QtWidgets.QLabel("RICOR regressor :")
        self.ui_physReg_cmbBx = QtWidgets.QComboBox()
        self.ui_physReg_cmbBx.addItems(['None', '8 RICOR (4 Resp and 4 Card)'])
        self.ui_physReg_cmbBx.setCurrentIndex(
            PHYS_REG_OPTS.index(self.phys_reg))
        self.ui_physReg_cmbBx.currentIndexChanged.connect(
            lambda idx:
            self.set_param('phys_reg',
                           self.ui_physReg_cmbBx.currentText(),
                           self.ui_physReg_cmbBx.setCurrentIndex))
        ui_rows.append((var_lb, self.ui_physReg_cmbBx))
        self.ui_objs.extend([var_lb, self.ui_physReg_cmbBx])

        # desMtx
        var_lb = QtWidgets.QLabel("Design matrix :")

        desMtx_hBLayout = QtWidgets.QHBoxLayout()
        self.ui_loadDesMtx_btn = QtWidgets.QPushButton(
            'Set' if self.desMtx_read is None else 'Unset')
        self.ui_loadDesMtx_btn.clicked.connect(
            lambda: self.set_param('desMtx_f', 'set'))
        desMtx_hBLayout.addWidget(self.ui_loadDesMtx_btn)

        self.ui_showDesMtx_btn = QtWidgets.QPushButton()
        self.ui_showDesMtx_btn.clicked.connect(
            lambda: self.set_param('showDesMtx'))
        desMtx_hBLayout.addWidget(self.ui_showDesMtx_btn)

        self.ui_objs.extend([var_lb, self.ui_loadDesMtx_btn,
                             self.ui_showDesMtx_btn])
        ui_rows.append((var_lb, desMtx_hBLayout))
        self.ui_showDesMtx_btn.setText('Show design matrix')
        self.ui_showDesMtx_btn.setEnabled(self.desMtx_read is not None)

        # censor_FD
        var_lb = QtWidgets.QLabel("Censor FD > (mm, 0: off) :")
        self.ui_censorFD_spBx = QtWidgets.QDoubleSpinBox()
        self.ui_censorFD_spBx.setMinimum(0.0)
        self.ui_censorFD_spBx.setMaximum(10.0)
        self.ui_censorFD_spBx.setDecimals(2)
        self.ui_censorFD_spBx.setSingleStep(0.1)
        self.ui_censorFD_spBx.setValue(self.censor_FD)
        self.ui_censorFD_spBx.setToolTip(
            "Exclude volumes with framewise displacement (Power et al., 2012)"
            " larger than this value from the GLM fits.")
        self.ui_censorFD_spBx.editingFinished.connect(
            lambda: self.set_param('censor_FD',
                                   self.ui_censorFD_spBx.value(),
                                   self.ui_censorFD_spBx.setValue))
        ui_rows.append((var_lb, self.ui_censorFD_spBx))
        self.ui_objs.extend([var_lb, self.ui_censorFD_spBx])

        # --- Checkbox row ----------------------------------------------------
        # Retrospective process
        self.ui_retroProc_chb = QtWidgets.QCheckBox("Retrospective process")
        self.ui_retroProc_chb.setChecked(self.reg_retro_proc)
        self.ui_retroProc_chb.stateChanged.connect(
            lambda state: setattr(self, 'reg_retro_proc', state > 0))
        self.ui_objs.append(self.ui_retroProc_chb)

        # Save
        self.ui_saveProc_chb = QtWidgets.QCheckBox("Save processed image")
        self.ui_saveProc_chb.setChecked(self.save_proc)
        self.ui_saveProc_chb.stateChanged.connect(
            lambda state: setattr(self, 'save_proc', state > 0))
        self.ui_objs.append(self.ui_saveProc_chb)

        chb_hLayout = QtWidgets.QHBoxLayout()
        chb_hLayout.addStretch()
        chb_hLayout.addWidget(self.ui_saveProc_chb)
        ui_rows.append((self.ui_retroProc_chb, chb_hLayout))

        return ui_rows

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _ui_roi_reg_row(self, reg, label):
        """
        Make the UI row for the ROI mean signal regressor reg ('GS', 'WM', or
        'Vent'): a check box for <reg>_reg, and the <reg>_mask file with a
        button to select it. The widgets are set to self.ui_<reg>_reg_chb,
        self.ui_<reg>_mask_lnEd, and self.ui_<reg>mask_btn.
        """
        reg_chb = QtWidgets.QCheckBox(label)
        reg_chb.setChecked(getattr(self, f"{reg}_reg"))
        reg_chb.stateChanged.connect(
            lambda state: self.set_param(f"{reg}_reg", state > 0))
        setattr(self, f"ui_{reg}_reg_chb", reg_chb)

        mask_hBLayout = QtWidgets.QHBoxLayout()
        mask_lnEd = QtWidgets.QLineEdit()
        mask_lnEd.setText(str(getattr(self, f"{reg}_mask")))
        mask_lnEd.setReadOnly(True)
        mask_lnEd.setStyleSheet('border: 0px none;')
        mask_hBLayout.addWidget(mask_lnEd)
        setattr(self, f"ui_{reg}_mask_lnEd", mask_lnEd)

        mask_btn = QtWidgets.QPushButton('Set')
        mask_btn.clicked.connect(
            lambda: self.set_param(f"{reg}_mask",
                                   Path(mask_lnEd.text()).parent,
                                   mask_lnEd.setText))
        mask_hBLayout.addWidget(mask_btn)
        setattr(self, f"ui_{reg}mask_btn", mask_btn)

        self.ui_objs.extend([reg_chb, mask_lnEd, mask_btn])
        return (reg_chb, mask_hBLayout)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_params(self):
        all_opts = super().get_params()
        # Run-time data, regressor buffers, and objects are not saved
        excluded_opts = ('desMtx0', 'mot0', 'tshift', 'retrocols', 'volreg',
                         'YMtx', 'TR', 'desMtx', 'maskV', 'motcols',
                         'col_names_read', 'Y_mean', 'Y_mean_mask',
                         'GS_maskdata', 'WM_maskdata', 'Vent_maskdata',
                         'GS_col', 'WM_col', 'Vent_col', 'desMtx_read',
                         'mask_src_proc', 'work_dir', 'qc', 'qc_last')
        # Path values are already converted to str by RTP.get_params
        return {k: v for k, v in all_opts.items() if k not in excluded_opts}


# %% __main__ (test) ==========================================================
if __name__ == '__main__':

    # Set logging
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO,
        format='%(asctime)s.%(msecs)04d,[%(levelname)s],%(name)s,%(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S')

    # --- Test ---
    # Test data directory
    test_dir = Path(__file__).absolute().parent.parent / 'tests'

    # Set test data files
    testdata_f = test_dir / 'func_epi.nii.gz'
    ecg_f = test_dir / 'ECG.1D'
    resp_f = test_dir / 'Resp.1D'
    mask_data_f = test_dir / 'work' / 'RTP' / 'RTP_mask.nii.gz'

    # Load test data
    assert testdata_f.is_file()
    img = nib.load(testdata_f)
    img_data = np.asanyarray(img.dataobj)

    work_dir = test_dir / 'work'
    if not work_dir.is_dir():
        work_dir.mkdir()

    # --- Create RTP instances and set parameters ---
    from rtpspy.rtp_tshift import RtpTshift
    from rtpspy.rtp_volreg import RtpVolreg
    from rtpspy.rtp_smooth import RtpSmooth
    from rtpspy.rpc_socket_server import RPCSocketCom

    # RtpTshift
    rtp_tshift = RtpTshift()
    rtp_tshift.method = 'cubic'
    rtp_tshift.ignore_init = 3
    rtp_tshift.ref_time = 0
    rtp_tshift.set_from_sample(testdata_f)

    # RtpVolreg
    rtp_volreg = RtpVolreg(regmode='cubic')
    refname = str(testdata_f) + '[0]'
    rtp_volreg.set_ref_vol(refname)

    # RtpSmooth
    rtp_smooth = RtpSmooth(blur_fwhm=6)
    rtp_smooth.set_param('mask_file', mask_data_f)

    # Physio recorder (RtTTLPhysio socket server)
    rtp_physio = RPCSocketCom(
        ["localhost", None, "RtTTLPhysioSocketServer"],
        Path.home() / ".RTPSpy" / "rtmri_config.json"
    )

    # RtpRegress
    rtp_regress = RtpRegress()

    # Set parameters
    rtp_regress.TR = 2.0
    rtp_regress.max_poly_order = np.inf
    rtp_regress.mot_reg = 'mot12'
    rtp_regress.volreg = rtp_volreg
    rtp_regress.phys_reg = 'RICOR8'
    rtp_regress.tshift = 0.0
    rtp_regress.GS_reg = True
    rtp_regress.GS_mask = test_dir / 'work' / 'RTP' / 'GSR_mask.nii.gz'
    rtp_regress.WM_reg = True
    rtp_regress.WM_mask = test_dir / 'work' / 'RTP' / \
        'anat_mprage_WM_alFunc.nii.gz'
    rtp_regress.Vent_reg = True
    rtp_regress.Vent_mask = test_dir / 'work' / 'RTP' / \
        'anat_mprage_Vent_alFunc.nii.gz'
    rtp_regress.mask_src_proc = rtp_volreg
    rtp_regress.wait_num = 45
    rtp_regress.set_param('mask_file', mask_data_f)
    rtp_regress.onGPU = True
    rtp_regress.rtp_physio = rtp_physio

    # Save processed files
    rtp_tshift.work_dir = work_dir
    rtp_tshift.save_proc = True
    rtp_volreg.work_dir = work_dir
    rtp_volreg.save_proc = True
    rtp_smooth.work_dir = work_dir
    rtp_smooth.save_proc = True
    rtp_regress.work_dir = work_dir
    rtp_regress.save_proc = True
    rtp_regress.save_delay = True

    # Chain the modules: tshift -> volreg -> smooth -> regress
    rtp_tshift.next_proc = rtp_volreg
    rtp_volreg.next_proc = rtp_smooth
    rtp_smooth.next_proc = rtp_regress
    rtp_tshift.end_reset()

    # Wait for rtp_physio to start recording
    while rtp_regress.rtp_physio.scan_onset < 0:
        time.sleep(0.1)

    # Run
    N_vols = img.shape[-1]
    TR = float(img.header.get_zooms()[3])
    rtp_regress.rtp_physio.scan_onset = time.time()
    for ii in range(N_vols):
        save_filename = f"test_nr_{ii+1:04d}.nii.gz"
        fmri_img = nib.Nifti1Image(img_data[:, :, :, ii], affine=img.affine)
        fmri_img.set_filename(save_filename)
        st = time.time()
        rtp_tshift.do_proc(fmri_img, ii, st)  # run rtp_tshift -> rtp_regress
        time.sleep(TR)

    proc_delay = rtp_regress.proc_delay
    rtp_tshift.end_reset()  # The reset propagates to the chained modules

    # --- Plot ---
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(proc_delay[1:], bins='auto')
    print(f"Median processing delay: {np.median(proc_delay[1:]):.4f} s")
