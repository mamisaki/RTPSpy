#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTP regression

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import sys
import time
import re
import traceback
import logging

import numpy as np
import nibabel as nib
from six import string_types
from PyQt5 import QtWidgets
import torch

try:
    from .rtp_common import RTP, MatplotlibWindow
except Exception:
    from rtpspy.rtp_common import RTP, MatplotlibWindow

gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()


# %% lstsq_SVDsolver ==========================================================
def lstsq_SVDsolver(A, B, rcond=None):
    """
    Solve a linear system Ax = b in least square sense (minimize ||Ax-b||^2)
    with SVD.

    Parameters
    ----------
    A : 2D tensor (n x m)
        n: number of samples (time points), m: nuber of variables (regressors).
        Number of rows (n) must be > number of columns (m).
    B : 2D tensor (n x k)
        n: number of sammples, k: number of depedent values (e.g., voxels).
    rcond : float, optional
        Cut-off ratio for small singular values. For the purpose of rank
        determination, singular values are treated as zero if they are smaller
        than rcond times the largest singular value.
        Default will use the machine precision times len(s).

    Returns
    -------
    x: 2D tensor (n x k)
    """

    if rcond is None:
        rcond = np.finfo(np.float32).eps * min(A.shape)

    # SVD for A
    U, S, V = torch.svd(A)

    # Clip singular values < S[0] * rcond
    sthresh = S[0] * rcond
    for ii in range(len(S)):
        if S[ii] < sthresh:
            break
        else:
            r = ii+1
    # r is rank of A

    # Diagnal matrix with Sinv: D = diag(1/S)
    Sinv = torch.zeros(len(S)).to(S.device)
    Sinv[:r] = 1.0/S[:r]
    D = torch.diag(Sinv)

    # X = V*D*UT*B
    X = torch.linalg.multi_dot((V, D, U.transpose(1, 0), B))

    return X


# %% RtpRegress class ========================================================
class RtpRegress(RTP):
    """
    RTP online regression analysis.
    Cumulative GLM that recalculates GLM with all available time points is
    used.
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, max_poly_order=np.inf, TR=2.0, mot_reg='None',
                 volreg=None, GS_reg=False, GS_mask=None, WM_reg=False,
                 WM_mask=None, Vent_reg=False, Vent_mask=None,
                 mask_src_proc=None, phys_reg='None', rtp_physio=None,
                 tshift=0.0, desMtx=None, wait_num=0, mask_file=0,
                 max_scan_length=800, onGPU=gpu_available, reg_retro_proc=True,
                 **kwargs):
        """
        Parameters
        ----------
        max_poly_order : int or np.inf, optional
            Maximum order of the polynomial regressors. Polynomial regressor is
            increasing automatically with the scan length. The default is
            np.inf.
        TR : float, optional
            fMRI TR. This is used for calculating the polynomial order as;
            order = 1 + int(TR*current_number_of_volume/150)
            The default is 2.0.
        mot_reg : str, ['None'|'mot6'|'mot12'|'dmot6'], optional
            Motion regressors type.
            'None': no motion regressor
            'mot6': six motions; yaw, pitch, roll, dx, dy, dz
            'mot12': mot6 plus their temporal derivative
            'dmot6': six motion derivatives
            The default is 'None'.
        volreg : RtpVolreg object instance, optional
            RtpVolreg instance to read motion parameters. The default is None.
        GS_reg : bool, optional
            Flag to use the global signal regressor.
            The default is False.
        GS_mask : str or Path, optional
            Mask file for the global signal caluculation. The default is None.
        WM_reg : bool, optional
            Flag to use the mean white matter regressor. The default is False.
        WM_mask : str or Path, optional
            Mask file for the white matter region. The default is None.
        Vent_reg : TYPE, optional
            Flag to use the mean ventricle regressor. The default is False.
        Vent_mask : str or Path, optional
            Mask file for the ventricle region. The default is None.
        mask_src_proc : RTP class object, optional
            Region mask regressors (GS_mask, WM_mask, Vent_mask) should be
            calculated with unsmoothed signals. If the input to RtpRegress is
            a smoothed one, this option allows using other RTP module's output
            for calculating masked signal regressors. The default is None.
        phys_reg : str, ['None'|'RICOR8'|'RVT5'|'RVT+RICOR13'], optional
            Physiological signal regressor type.
            None: no physio regressor
            RICOR8: four Resp and four Card regressors
            RVT5: five RVT regressors
            RVT+RICOR13: both RVT5 and RICOR8
            RVT is not recomended for RTP (see Misaki and Bodrka, 2021.)
            The default is 'None'.
        rtp_physio : RtpPyshio object, optional
            RtpPyshio object to get retrots regressors. The default is None.
        tshift : float, optional
            Slice timing offset (second) for calculating the restrots
            regressors. The default is 0.0.
        desMtx : 2D array, optional
            Additional design matrix other than motion, retrots, and polynomial
            regressors. The array must include the initial ignored volumes,
            while those are removed in the regression. The default is None.
        wait_num : int, optional
            The minimum number of volumes (excluding the ignored volumes) to
            wait before starting REGRESS. If the value is smaller than the
            number of regressors, this value is ignored and the regress waits
            for receiving the number of regressors volumes. The default is 12.
        mask_file : 0 or str (Path), optional
            Mask for the processing region. 0 means a zero-out mask with the
            initial volume. The default is 0.
        max_scan_length : int, optional
            Maximum scan length. This is used for pre-allocating memory space
            for X and Y data. If desMtx is not None max_scan_length is set by
            the length of desMtx.
        onGPU : bool, optional
            Run REGRESS on GPU when available. The default is the output of
            torch.cuda.is_available() or torch.backends.mps.is_available().
        reg_retro_proc : bool, optional
            Flag to retroactively process the volumes before starting regress.
            Default is True.

        """
        super(RtpRegress, self).__init__(**kwargs)

        # --- Set parameters ---
        # Polynomial regressors
        self.max_poly_order = max_poly_order
        self.TR = TR
        # Motion regressors
        self.mot_reg = mot_reg
        self.volreg = volreg
        # Region mask regressors
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
        # Other regressors design matrix
        self.desMtx_read = desMtx
        # Operational paramterss
        self.mask_file = mask_file
        self.max_scan_length = max_scan_length
        self.onGPU = onGPU
        self.reg_retro_proc = reg_retro_proc
        self.wait_num = wait_num

        # --- Initialize working data ---
        # Mask matrices
        self.maskV = None
        self.GS_maskdata = None
        self.WM_maskdata = None
        self.Vent_maskdata = None
        # Regressor variables
        self.reg_names = []
        self.col_names_read = []
        self.desMtx0 = None  # Initial design matrix including ignored volumes.
        self.desMtx = None
        # Previous motion parameters for calculating the motion derivatives.
        self.mot0 = None
        # Data matrix
        self.YMtx = None  # Data matrix
        self.Y_mean = None  # Mean signal for data scaling

        # --- Set the number of volumes to wait ---
        self.set_wait_num()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # onGPU getter, setter
    @property
    def onGPU(self):
        return self.device != 'cpu'

    @onGPU.setter
    def onGPU(self, _onGPU):
        if _onGPU:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                errmsg = "GPU is not available."
                self._logger.error(errmsg)
                self.err_popup(errmsg)
                self.device = 'cpu'
        else:
            self.device = 'cpu'

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ready_proc(self):
        self._proc_ready = True

        if self.TR is None:
            errmsg = 'TR is not set.'
            self._logger.error(errmsg)
            self.err_popup(errmsg)
            self._proc_ready = False

        if self.mot_reg != 'None' and self.volreg is None:
            errmsg = 'RtpVolreg object is not set.'
            self._logger.error(errmsg)
            self.err_popup(errmsg)
            self._proc_ready = False

        if self.phys_reg != 'None':
            if self.rtp_physio is None:
                errmsg = 'RtpTTLPhysio object is not set.'
                self._logger.error(errmsg)
                self.err_popup(errmsg)
                self._proc_ready = False

        if self.desMtx0 is None and self.max_scan_length is None:
            errmsg = 'Either design matrix or max scanlength must be set.'
            self._logger.error(errmsg)
            self.err_popup(errmsg)
            self._proc_ready = False

        if self.next_proc:
            self._proc_ready &= self.next_proc.ready_proc()

        if self._proc_ready:
            # Prepare design matrix
            self.setup_regressor_template(self.desMtx_read,
                                          self.max_scan_length,
                                          self.col_names_read)
            # Set mask
            if isinstance(self.mask_file, string_types) or \
                    isinstance(self.mask_file, Path):
                self.set_mask(self.mask_file)

            # Set the number of volumes to wait
            self.set_wait_num()

        return self._proc_ready

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def do_proc(self, fmri_img, vol_idx=None, pre_proc_time=None, **kwargs):
        """
        vol_idx: 0-based index of imaging volume from the scan start
        self._vol_num: Number of volumes received in this module.
        """

        try:
            # Increment the number of received volumes
            self._vol_num += 1  # 1-base number of volumes recieved by this
            if vol_idx is None:
                vol_idx = self._vol_num - 1  # 0-base index

            if vol_idx < self.ignore_init:
                # Skip ignore_init volumes
                return

            if self._proc_start_idx < 0:
                self._proc_start_idx = vol_idx

            dataV = fmri_img.get_fdata()
            if dataV.ndim > 3:
                dataV = np.squeeze(dataV)

            # --- Initialize --------------------------------------------------
            # Set maskV (process region mask)
            if self.maskV is None:
                # Make mask with the first received volume
                self.set_mask(dataV)
                msg = f"Mask is set with a volume index {vol_idx}."
                self._logger.info(msg)

            # Read global signal mask
            if self.GS_reg and self.GS_maskdata is None:
                if Path(self.GS_mask).is_file():
                    GSimg = nib.load(self.GS_mask)
                    if not np.all(dataV.shape == GSimg.shape):
                        errmsg = f"GS mask shape {GSimg.shape} !="
                        errmsg += " function image shape"
                        errmsg += f" {dataV.shape}"
                        self._logger.error(errmsg)
                        self.err_popup(errmsg)
                        return
                    else:
                        self.GS_maskdata = (GSimg.get_fdata() != 0)[self.maskV]
                else:
                    errmsg = f"Not found GS_mask file {self.GS_mask}"
                    self._logger.error(errmsg)
                    self._logger.error("GS_reg is reset to False.")
                    self.err_popup(errmsg)
                    return

            # Read WM mask
            if self.WM_reg and self.WM_maskdata is None:
                if Path(self.WM_mask).is_file():
                    WMimg = nib.load(self.WM_mask)
                    if not np.all(dataV.shape == WMimg.shape):
                        errmsg = f"WM mask shape {WMimg.shape} !="
                        errmsg += " function image shape"
                        errmsg += f" {dataV.shape}"
                        self._logger.error(errmsg)
                        self.err_popup(errmsg)
                        return
                    else:
                        self.WM_maskdata = (WMimg.get_fdata() != 0)[self.maskV]
                else:
                    errmsg = f"Not found WM_mask file {self.WM_mask}"
                    self._logger.error(errmsg)
                    self.err_popup(errmsg)
                    return

            # Read ventricle mask
            if self.Vent_reg and self.Vent_maskdata is None:
                if Path(self.Vent_mask).is_file():
                    Ventimg = nib.load(self.Vent_mask)
                    if not np.all(dataV.shape == Ventimg.shape):
                        errmsg = f"Vet mask shape {Ventimg.shape} !="
                        errmsg += " function image shape"
                        errmsg += f" {dataV.shape}"
                        self._logger.error(errmsg)
                        self.err_popup(errmsg)
                        return
                    else:
                        self.Vent_maskdata = \
                            (Ventimg.get_fdata() != 0)[self.maskV]
                else:
                    errmsg = f"Not found Vent_mask file {self.Vent_mask}"
                    self._logger.error(errmsg)
                    self.err_popup(errmsg)
                    return

            # Initialize design matrix
            if self.desMtx is None:
                # Update self.desMtx0 and self.reg_names
                self.setup_regressor_template(self.desMtx_read,
                                              self.max_scan_length,
                                              self.col_names_read)
                desMtx = self.desMtx0.copy()

                # Add maximum number of polynomial regressors
                nt = desMtx.shape[0]
                pnum = min(1 + int(nt*self.TR/150), self.max_poly_order)
                desMtx = np.concatenate(
                        [desMtx, np.zeros((nt, pnum+1), dtype=np.float32)],
                        axis=1)
                self.desMtx = torch.from_numpy(
                        desMtx.astype(np.float32)).to(self.device)

            # Initialize Y matrix
            if self.YMtx is None:
                vox_num = np.sum(self.maskV)
                self.YMtx = torch.empty(
                        self.desMtx0.shape[0]-self._proc_start_idx, vox_num,
                        dtype=torch.float32)
                try:
                    self.YMtx = self.YMtx.to(self.device)
                except Exception as e:
                    self._logger.error(str(e))
                    if self.device != 'cpu':
                        errmsg = "Failed to keep GPU memory for Y."
                        self._logger.error(errmsg)
                        self.err_popup(errmsg)
                        self.onGPU = False  # self.device is changed to 'cpu'
                        self.desMtx = self.desMtx.to(self.device)
                        self.YMtx = self.YMtx.to(self.device)
                    else:
                        errmsg = "Failed to keep memory for Y."
                        self._logger.error(errmsg)
                        self.err_popup(errmsg)
                    raise e

            # --- Update data -------------------------------------------------
            # Y matrix
            ydata = dataV[self.maskV]
            ydata = torch.from_numpy(ydata.astype(np.float32)).to(self.device)
            if self.Y_mean is not None:
                # Scaling data
                ydata[self.Y_mean_mask] = \
                    ydata[self.Y_mean_mask]/self.Y_mean[self.Y_mean_mask]*100
                ydata[ydata > 200] = 200

            self.YMtx[vol_idx, :] = ydata

            # -- Design matrix --
            # Append motion parameter
            if self.mot_reg != 'None':
                mot = self.volreg._motion[vol_idx, :]
                if self.mot_reg in ('mot12', 'dmot6'):
                    if self._vol_num > 1 and self.mot0 is not None:
                        dmot = mot - self.mot0
                    else:
                        dmot = np.zeros(6, dtype=np.float32)
                    self.mot0 = mot

                if self.mot_reg in ('mot6', 'mot12'):
                    mot = torch.from_numpy(
                        mot.astype(np.float32)).to(self.device)
                if self.mot_reg in ('mot12', 'dmot6'):
                    dmot = torch.from_numpy(
                        dmot.astype(np.float32)).to(self.device)

                # Assuming self.motcols is contiguous
                if self.mot_reg in ('mot6', 'mot12'):
                    self.desMtx[vol_idx,
                                self.motcols[0]:self.motcols[0]+6] = mot
                    if self.mot_reg == 'mot12':
                        self.desMtx[vol_idx,
                                    self.motcols[6]:self.motcols[6]+6] = dmot
                elif self.mot_reg == 'dmot6':
                    self.desMtx[vol_idx,
                                self.motcols[0]:self.motcols[0]+6] = dmot

            # Append mask mean signal regressor from mask_src_proc
            if self.GS_reg or self.WM_reg or self.Vent_reg:
                msk_src_data = self.mask_src_proc.proc_data[self.maskV]
                if self.GS_reg:
                    self.desMtx[vol_idx, self.GS_col] =  \
                        float(msk_src_data[self.GS_maskdata].mean())

                if self.WM_reg:
                    self.desMtx[vol_idx, self.WM_col] =  \
                        float(msk_src_data[self.WM_maskdata].mean())

                if self.Vent_reg:
                    self.desMtx[vol_idx, self.Vent_col] = \
                        float(msk_src_data[self.Vent_maskdata].mean())

            # --- If the number of samples is not enough, return --------------
            if self._vol_num <= self.wait_num:
                wait_idx = self._proc_start_idx+self.wait_num
                msg = f"#{vol_idx+1}:Wait until volume #{wait_idx+1}"
                self._logger.info(msg)
                return

            # --- Update retroicor regressors ---------------------------------
            if self.phys_reg != 'None' and self.rtp_physio is not None:
                retrots = self.rtp_physio.get_retrots(
                    self.TR, vol_idx+1, self.tshift,
                    timeout=self.TR)
                if retrots is None:
                    errmsg = "RETROTS regressors cannot be made."
                    self._logger.error(errmsg)
                    return

                for ii, icol in enumerate(self.retrocols):
                    self.desMtx[:vol_idx+1, icol] = \
                            torch.from_numpy(
                                    retrots[:, ii].astype(np.float32)
                                    ).to(self.device)

            # --- Run the procress --------------------------------------------
            # Set Y_mean for scaling data
            if self.Y_mean is None:
                # Scaling
                YMtx = self.YMtx[self._proc_start_idx:vol_idx+1, :]
                # YMtx is the reference to the original data, self.YMtx
                self.Y_mean = YMtx.mean(axis=0)
                self.Y_mean_mask = self.Y_mean.abs() > 1e-6

                YMtx[:, self.Y_mean_mask] = \
                    YMtx[:, self.Y_mean_mask] / \
                    self.Y_mean[self.Y_mean_mask] * 100
                YMtx[YMtx > 200] = 200
                YMtx[:, ~self.Y_mean_mask] = 0.0
                # The operation is done on the original data,
                # so no need to return like below
                # ydata = self.YMtx[vol_idx, :]

            # Add polynomials to the design matrix
            polyreg = self.poly_reg(vol_idx-self._proc_start_idx+1, self.TR)
            reg0_num = self.desMtx0.shape[1]
            polyreg_num = polyreg.shape[1]
            self.desMtx[self._proc_start_idx:vol_idx+1,
                        reg0_num:reg0_num+polyreg_num] = \
                torch.from_numpy(polyreg).to(self.device)

            # Extract a part of regressors (until the current volume)
            Xp = self.desMtx[self._proc_start_idx:vol_idx+1,
                             :reg0_num+polyreg_num].clone()

            # Standardizing regressors of motion, GS, WM, Vent
            norm_regs = ('roll', 'pitch', 'yaw', 'dS', 'dL', 'dP',
                         'dtroll', 'dtpitch', 'dtyaw', 'dtdS', 'dtdL', 'dtdP',
                         'GS', 'WM', 'Vent')
            for ii, reg_name in enumerate(self.reg_names):
                if reg_name in norm_regs:
                    reg = Xp[:, ii]
                    reg = (reg - reg.mean()) / reg.std()
                    Xp[:, ii] = reg

            # Extract a part of Y (until the current volume)
            Yp = self.YMtx[self._proc_start_idx:vol_idx+1, :]

            # Calculate Beta with the least sqare error, ||Y - XB||^2
            Beta = lstsq_SVDsolver(Xp, Yp[:, self.Y_mean_mask])
            Yh = torch.matmul(Xp, Beta)

            if self.reg_retro_proc and self._vol_num == self.wait_num+1:
                # Process (and save) the previous volumes retrospectively.
                Resids = Yp[:, self.Y_mean_mask] - Yh

                vi0 = vol_idx - (Resids.shape[0]-1)
                vilast = vol_idx - 1
                msg = f"Retrospective regression for {vi0}-{vilast}"
                self._logger.info(msg)

                # Save filename template
                save_name_temp = Path(fmri_img.get_filename()).name
                save_name_temp = 'regRes.' + save_name_temp
                ma = re.search(f"0*{vol_idx+1}", save_name_temp)
                if ma is not None:
                    dstr = ma.group()
                else:
                    dstr = None

                # Data space
                vol_data = np.zeros_like(self.maskV, dtype=np.float32)
                flat_data = np.zeros(np.sum(self.maskV), dtype=np.float32)
                for ii in range(Resids.shape[0]-1):
                    vi = vol_idx - (Resids.shape[0]-1) + ii
                    resid = Resids[ii, :]
                    flat_data[self.Y_mean_mask.cpu().numpy()] = \
                        resid.cpu().numpy()
                    vol_data[self.maskV] = flat_data
                    # temporay nibabel image for retroactive process
                    retro_fmri_img = nib.Nifti1Image(vol_data, fmri_img.affine,
                                                     header=fmri_img.header)
                    if dstr is not None:
                        dstr_i0 = f"{vi+1}"
                        dstr_i = '0' * (len(dstr)-len(dstr_i0)) + dstr_i0
                        save_name = save_name_temp.replace(dstr, dstr_i)
                    else:
                        save_name = save_name_temp.replace(
                            '.nii', f"_{vi+1:04d}.nii")

                    retro_fmri_img.set_filename(save_name)

                    # Run the next process
                    if self.next_proc:
                        self.next_proc.do_proc(retro_fmri_img, vi, time.time())

                    # Save processed image
                    if self.save_proc:
                        self.keep_processed_image(retro_fmri_img, vi)

                    time.sleep(0.01)  # To avoid too busy data send.

                Resid = Resids[-1, :]
                del Resids
            else:
                # Get only the last (current) volume
                Resid = ydata[self.Y_mean_mask] - Yh[-1, :]

            del Beta
            del Yh

            # Set processed data in fmri_img
            vol_data = np.zeros_like(self.maskV, dtype=np.float32)
            flat_data = np.zeros(np.sum(self.maskV), dtype=np.float32)
            flat_data[self.Y_mean_mask.cpu().numpy()] = Resid.cpu().numpy()
            vol_data[self.maskV] = flat_data
            fmri_img.uncache()
            fmri_img._dataobj = vol_data
            fmri_img.set_data_dtype = vol_data.dtype

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
            msg = f"#{vol_idx+1};Regression;{f}"
            msg += f";tstamp={tstamp}"
            if pre_proc_time is not None:
                msg += f";took {proc_delay:.4f}s"
            self._logger.info(msg)

            # Set filename
            fmri_img.set_filename('regRes.' +
                                  Path(fmri_img.get_filename()).name)

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

        if not isinstance(self.mask_file, string_types) and \
                not isinstance(self.mask_file, Path):
            self.maskV = None

        self.desMtx = None
        self.mot0 = None
        self.YMtx = None
        self.Y_mean = None
        self.Y_mean_mask = None
        self.GS_maskdata = None
        self.WM_maskdata = None
        self.Vent_maskdata = None

        return super(RtpRegress, self).end_reset()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_mask(self, maskdata, sub_i=0, method='zero_out'):
        if isinstance(maskdata, string_types) or isinstance(maskdata, Path):
            if not Path(maskdata).is_file():
                errmsg = f"Not found mask file: {maskdata}"
                self._logger.error(errmsg)
                self.err_popup(errmsg)
                self.mask_file = 0
                return

            msg = f"Mask = {maskdata}"
            self._logger.info(msg)
            maskdata = np.squeeze(nib.load(maskdata).get_fdata())

            if maskdata.ndim > 3:
                maskdata = maskdata[:, :, :, sub_i]

        if method == 'zero_out':
            self.maskV = maskdata != 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def setup_regressor_template(self, desMtx_read=None, max_scan_length=0,
                                 col_names_read=[]):
        if desMtx_read is None and max_scan_length == 0:
            errmsg = 'Either desMtx or max_scan_length must be given.'
            self._logger.error(errmsg)
            self.err_popup(errmsg)
            return

        col_names = col_names_read.copy()

        # Set provided design matrix
        if desMtx_read is None:
            desMtx = np.zeros([max_scan_length, 0], dtype=np.float32)
        else:
            if max_scan_length <= desMtx_read.shape[0]:
                max_scan_length = desMtx_read.shape[0]
            else:
                desMtx = np.zeros([max_scan_length, desMtx_read.shape[1]],
                                  dtype=np.float32)
                desMtx[:desMtx_read.shape[0], :] = desMtx_read

            # Adjust col_names and desMtx columns
            if len(col_names) > desMtx.shape[1]:
                col_names = col_names[:desMtx.shape[1]]
            elif len(col_names) < desMtx.shape[1]:
                while len(col_names) < desMtx.shape[1]:
                    col_names.append("Reg{len(col_names)+2}")

        # Append nuisunce regressors
        if self.mot_reg != 'None':
            if self.mot_reg in ('mot6', 'mot12'):
                # Append 6 motion parameters
                desMtx = np.concatenate(
                    [desMtx, np.zeros([max_scan_length, 6])], axis=1)
                col_names.extend(['roll', 'pitch', 'yaw', 'dS', 'dL', 'dP'])
                self.motcols = \
                    [ii for ii, cn in enumerate(col_names)
                     if cn in ('roll', 'pitch', 'yaw', 'dS', 'dL', 'dP')]
            else:
                self.motcols = []

            if self.mot_reg in ('mot12', 'dmot6'):
                # Append 6 motion derivative parameters
                desMtx = np.concatenate(
                        [desMtx, np.zeros([max_scan_length, 6])], axis=1)
                col_names.extend(['dtroll', 'dtpitch', 'dtyaw', 'dtdS', 'dtdL',
                                  'dtdP'])
                self.motcols.extend(
                        [ii for ii, cn in enumerate(col_names)
                         if cn in ('dtroll', 'dtpitch', 'dtyaw', 'dtdS',
                                   'dtdL', 'dtdP')])

        if self.phys_reg != 'None':
            # Append RVT, RETROICOR regresors
            if self.phys_reg == 'RVT5':
                nreg = 5
                col_add = ['RVT0', 'RVT1', 'RVT2', 'RVT3', 'RVT4']
            elif self.phys_reg == 'RICOR8':
                nreg = 8
                col_add = ['Resp0', 'Resp1', 'Resp2', 'Resp3',
                           'Card0', 'Card1', 'Card2', 'Card3']
            elif self.phys_reg == 'RVT+RICOR13':
                nreg = 13
                col_add = ['RVT0', 'RVT1', 'RVT2', 'RVT3', 'RVT4',
                           'Resp0', 'Resp1', 'Resp2', 'Resp3',
                           'Card0', 'Card1', 'Card2', 'Card3']
            desMtx = np.concatenate([desMtx,
                                     np.zeros([max_scan_length, nreg])],
                                    axis=1)
            col_names.extend(col_add)
            self.retrocols = \
                [ii for ii, cn in enumerate(col_names) if cn in col_add]

        if self.GS_reg:
            # Append global signal regressor
            desMtx = np.concatenate([desMtx,
                                     np.zeros([max_scan_length, 1])],
                                    axis=1)
            col_names.append('GS')
            self.GS_col = col_names.index('GS')

        if self.WM_reg:
            # Append mean WM signal regressor
            desMtx = np.concatenate([desMtx,
                                     np.zeros([max_scan_length, 1])],
                                    axis=1)
            col_names.append('WM')
            self.WM_col = col_names.index('WM')

        if self.Vent_reg:
            # Append mean ventricle signal regressor
            desMtx = np.concatenate([desMtx,
                                     np.zeros([max_scan_length, 1])],
                                    axis=1)
            col_names.append('Vent')
            self.Vent_col = col_names.index('Vent')

        self.desMtx0 = desMtx
        self.reg_names = col_names

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def legendre(self, x, m):
        """
        Legendre polynomial calculator
        Copied from misc_math.c in afni src

        Parameters
        ----------
        x: 1D array
            series of -1 .. 1 values
        m: int
            polynomial order
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
    def poly_reg(self, nt, TR):
        """
        Make legendre polynomial regressor for nt length data

        Option
        ------
        nt: int
            data length (must be > 1)

        Retrun
        ------
        polyreg: nt * x array
            Matrix of Legendre polynomial regressors

        """

        # If nt is not enough even for linear trend, return
        if nt < 1:
            return None

        # Set polynomial order
        pnum = min(1 + int(nt*TR/150), self.max_poly_order)
        polyreg = np.ndarray((nt, pnum+1), dtype=np.float32)

        for po in range(pnum+1):
            xx = np.linspace(-1.0, 1.0, nt)
            polyreg[:, po] = self.legendre(xx, po)

        return polyreg

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_reg_num(self):
        # update self.desMtx0 (regressor template)
        self.setup_regressor_template(self.desMtx_read,
                                      max_scan_length=self.max_scan_length,
                                      col_names_read=self.col_names_read)
        numReg = self.desMtx0.shape[1]
        numPolyReg = min(1 + int(numReg*self.TR/150), self.max_poly_order) + 1

        return numReg + numPolyReg

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_wait_num(self, wait_num=None):
        reg_num = self.get_reg_num()
        min_wait_num = reg_num+1

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
    def set_param(self, attr, val=None, reset_fn=None, echo=False):
        self._logger.debug(f"set_param: {attr} = {val}")

        # -- check value --
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
                    self.ui_mask_lnEd.setText(
                        'zero-out initial received volume')

            elif type(val) is str and 'initial volume' in val:
                val = 0
                if hasattr(self, 'ui_mask_lnEd'):
                    self.ui_mask_lnEd.setText(
                        'zero-out initial received volume')

            elif type(val) is str and 'external file' in val:
                fname = self.select_file_dlg('REGRESS: Selct mask volume',
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
                        return

                    mask_fname += f"[{num}]"
                else:
                    num = 0

                self.set_mask(fname[0], num)
                if hasattr(self, 'ui_mask_lnEd'):
                    self.ui_mask_lnEd.setText(str(mask_fname))
                val = mask_fname

            elif type(val) is str or isinstance(val, Path):
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
                if reset_fn is None:
                    if hasattr(self, 'ui_waitNum_cmbBx'):
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
                        self.wait_num,  num0)
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

            # Update self.desMtx0
            self.setup_regressor_template(self.desMtx_read,
                                          max_scan_length=val,
                                          col_names_read=self.col_names_read)
            if val < self.wait_num:
                val = self.wait_num + 1

            if reset_fn is None:
                if hasattr(self, 'ui_maxLen_spBx'):
                    self.ui_maxLen_spBx.setValue(val)

        elif attr == 'max_poly_order':
            if val == 'auto':
                if hasattr(self, 'ui_maxPoly_lb'):
                    self.ui_maxPoly_lb.setText('Increase polynomial order ' +
                                               'with the scan length')
                val = np.inf
            elif val == 'set':
                num, okflag = QtWidgets.QInputDialog.getInt(
                        None, "Maximum polynomial order", "enter value")
                if not okflag:
                    if np.isinf(self.max_poly_order):
                        if reset_fn:
                            reset_fn(0)
                    return

                if hasattr(self, 'ui_maxPoly_lb'):
                    self.ui_maxPoly_lb.setText('Increase polynomial order ' +
                                               'with the scan length' +
                                               f' up to {num}')
                val = num
            elif np.isinf(val) and reset_fn is None:
                if hasattr(self, 'ui_maxPoly_cmbBx'):
                    self.ui_maxPoly_cmbBx.setCurrentIndex(0)
                    self.ui_maxPoly_lb.setText('Increase polynomial order ' +
                                               'with the scan length')
            elif type(val) is int and reset_fn is None:
                if hasattr(self, 'ui_maxPoly_cmbBx'):
                    self.ui_maxPoly_cmbBx.setCurrentIndex(1)
                    self.ui_maxPoly_lb.setText('Increase polynomial order ' +
                                               'with the scan length' +
                                               f' up to {val}')

            setattr(self, attr, val)

            # Update wait_num
            if hasattr(self, 'ui_waitNum_cmbBx'):
                if 'regressor' in self.ui_waitNum_cmbBx.currentText():
                    self.set_wait_num(self.get_reg_num()+1)
            else:
                self.set_wait_num()

        elif attr == 'mot_reg':
            if val.lower() == 'none':
                val = 'None'
            elif val.startswith('6 motions'):
                val = 'mot6'
            elif val.startswith('12 motions'):
                val = 'mot12'
            elif val.startswith('6 motion derivatives'):
                val = 'dmot6'

            if reset_fn is None and hasattr(self, 'ui_motReg_cmbBx'):
                if val == 'None':
                    self.ui_motReg_cmbBx.setCurrentIndex(0)
                elif val == 'mot6':
                    self.ui_motReg_cmbBx.setCurrentIndex(1)
                elif val == 'mot12':
                    self.ui_motReg_cmbBx.setCurrentIndex(2)
                elif val == 'dmot6':
                    self.ui_motReg_cmbBx.setCurrentIndex(3)

            setattr(self, attr, val)

            # Update wait_num
            if hasattr(self, 'ui_waitNum_cmbBx'):
                if 'regressor' in self.ui_waitNum_cmbBx.currentText():
                    self.set_wait_num(self.get_reg_num()+1)
            else:
                self.set_wait_num()

        elif attr == 'GS_reg':
            setattr(self, attr, val)
            if hasattr(self, 'ui_GS_reg_chb'):
                self.ui_GS_reg_chb.setChecked(self.GS_reg)

            # Update wait_num
            if hasattr(self, 'ui_waitNum_cmbBx'):
                if 'regressor' in self.ui_waitNum_cmbBx.currentText():
                    self.set_wait_num(self.get_reg_num()+1)
            else:
                self.set_wait_num()

        elif attr == 'GS_mask':
            if reset_fn is not None:
                if str(val) != '.' and Path(val).is_dir():
                    startdir = val
                else:
                    startdir = self.work_dir

                dlgMdg = "REGRESS: Select global signal mask"
                fname = self.select_file_dlg(dlgMdg, startdir,
                                             "*.BRIK* *.nii*")
                if fname[0] == '':
                    return -1

                val = fname[0]
                if reset_fn:
                    reset_fn(str(val))

            elif type(val) is str or isinstance(val, Path):
                if not Path(val).is_file():
                    val = ''

                if hasattr(self, f"ui_{attr}_lnEd"):
                    obj = getattr(self, f"ui_{attr}_lnEd")
                    obj.setText(str(val))

        elif attr == 'WM_reg':
            setattr(self, attr, val)
            if hasattr(self, 'ui_WM_reg_chb'):
                self.ui_WM_reg_chb.setChecked(self.WM_reg)

            # Update wait_num
            if hasattr(self, 'ui_waitNum_cmbBx'):
                if 'regressor' in self.ui_waitNum_cmbBx.currentText():
                    self.set_wait_num(self.get_reg_num()+1)
            else:
                self.set_wait_num()

        elif attr == 'WM_mask':
            if reset_fn is not None:
                if str(val) != '.' and Path(val).is_dir():
                    startdir = val
                else:
                    startdir = self.work_dir

                dlgMdg = "REGRESS: Select white matter mask"
                fname = self.select_file_dlg(dlgMdg, startdir,
                                             "*.BRIK* *.nii*")
                if fname[0] == '':
                    return -1

                val = fname[0]
                if reset_fn:
                    reset_fn(str(val))

            elif type(val) is str or isinstance(val, Path):
                if not Path(val).is_file():
                    val = ''

                if hasattr(self, f"ui_{attr}_lnEd"):
                    obj = getattr(self, f"ui_{attr}_lnEd")
                    obj.setText(str(val))

        elif attr == 'Vent_reg':
            setattr(self, attr, val)
            if hasattr(self, 'ui_Vent_reg_chb'):
                self.ui_Vent_reg_chb.setChecked(self.Vent_reg)

            # Update wait_num
            if hasattr(self, 'ui_waitNum_cmbBx'):
                if 'regressor' in self.ui_waitNum_cmbBx.currentText():
                    self.set_wait_num(self.get_reg_num()+1)
            else:
                self.set_wait_num()

        elif attr == 'Vent_mask':
            if reset_fn is not None:
                if str(val) != '.' and Path(val).is_dir():
                    startdir = val
                else:
                    startdir = self.work_dir

                dlgMdg = "REGRESS: Select ventricle mask"
                fname = self.select_file_dlg(dlgMdg, startdir,
                                             "*.BRIK* *.nii*")
                if fname[0] == '':
                    return -1

                val = fname[0]
                if reset_fn:
                    reset_fn(str(val))

            elif type(val) is str or isinstance(val, Path):
                if not Path(val).is_file():
                    val = ''

                if hasattr(self, f"ui_{attr}_lnEd"):
                    obj = getattr(self, f"ui_{attr}_lnEd")
                    obj.setText(str(val))

        elif attr == 'phys_reg':
            if val.lower() == 'none':
                val = 'None'
            elif val.startswith('8 RICOR'):
                val = 'RICOR8'
            elif val.startswith('5 RVT'):
                val = 'RVT5'
            elif val.startswith('13 RVT+RICOR'):
                val = 'RVT+RICOR13'

            if reset_fn is None and hasattr(self, 'ui_physReg_cmbBx'):
                if val == 'None':
                    self.ui_physReg_cmbBx.setCurrentIndex(0)
                elif val == 'RICOR8':
                    self.ui_physReg_cmbBx.setCurrentIndex(1)
                elif val == 'RVT5':
                    self.ui_physReg_cmbBx.setCurrentIndex(2)
                elif val == 'RVT+RICOR13':
                    self.ui_physReg_cmbBx.setCurrentIndex(3)

            setattr(self, attr, val)

            # Update wait_num
            if hasattr(self, 'ui_waitNum_cmbBx'):
                if 'regressor' in self.ui_waitNum_cmbBx.currentText():
                    self.set_wait_num(self.get_reg_num()+1)
            else:
                self.set_wait_num()

            return

        elif attr == 'desMtx_f':
            if val is None:
                return

            elif (val == 'set' and
                  'Unset' not in self.ui_loadDesMtx_btn.text()) or \
                    Path(val).is_file():
                if val == 'set':
                    fname = self.select_file_dlg(
                            'REGRESS: Selct design matrix file',
                            self.work_dir, "*.csv")
                    if fname[0] == '':
                        return -1
                    fname = fname[0]
                else:
                    fname = val

                self.desMtx_f = fname

                # Have a header?
                ll = open(fname, 'r').readline()
                if np.any([isinstance(cc, string_types)
                           for cc in ll.split(',')]):
                    self.col_names_read = ll.split()
                    skiprows = 1
                else:
                    skiprows = 0

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

            elif val == 'unset' or (val == 'set' and
                                    'Unset' in self.ui_loadDesMtx_btn.text()):
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

            # Update wait_num
            if hasattr(self, 'ui_waitNum_cmbBx'):
                if 'regressor' in self.ui_waitNum_cmbBx.currentText():
                    self.set_wait_num(self.get_reg_num()+1)
            else:
                self.set_wait_num()

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

        elif attr == 'reg_retro_proc':
            if hasattr(self, 'ui_retroProc_chb'):
                self.ui_retroProc_chb.setChecked(val)

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
        """
        When reset_fn is None, set_param is considered to be called from
        load_parameters function.
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
        self.ui_mask_lnEd.setStyleSheet(
            'border: 0px none;')
        self.ui_objs.extend([var_lb, self.ui_mask_cmbBx,
                             self.ui_mask_lnEd])

        if self.mask_file == 0:
            self.ui_mask_cmbBx.setCurrentIndex(1)
            self.ui_mask_lnEd.setText('zero-out initial received volume')
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
        regNum = self.get_reg_num()
        self.ui_waitNum_lb.setText(
                f'Wait REGRESS until receiving {regNum} volumes')
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
            self.ui_maxPoly_lb.setText('Increase polynomial order ' +
                                       'with the scan length')
        else:
            self.ui_maxPoly_cmbBx.setCurrentIndex(1)
            self.ui_maxPoly_lb.setText('Increase polynomial order ' +
                                       'with the scan length' +
                                       f' up to {self.max_poly_order}')

        maxPoly_hLayout = QtWidgets.QHBoxLayout()
        maxPoly_hLayout.addWidget(self.ui_maxPoly_cmbBx)
        maxPoly_hLayout.addWidget(self.ui_maxPoly_lb)
        ui_rows.append((var_lb, maxPoly_hLayout))

        # mot_reg
        var_lb = QtWidgets.QLabel("Motion regressor :")
        self.ui_motReg_cmbBx = QtWidgets.QComboBox()
        self.ui_motReg_cmbBx.addItems(
                ['None', '6 motions (yaw, pitch, roll, dS, dL, dP)',
                 '12 motions (6 motions and their temporal derivatives)',
                 '6 motion derivatives'])
        ci = {'None': 0, 'mot6': 1, 'mot12': 2, 'dmot6': 3}[self.mot_reg]
        self.ui_motReg_cmbBx.setCurrentIndex(ci)
        self.ui_motReg_cmbBx.currentIndexChanged.connect(
                lambda idx:
                self.set_param('mot_reg',
                               self.ui_motReg_cmbBx.currentText(),
                               self.ui_motReg_cmbBx.setCurrentIndex))
        ui_rows.append((var_lb, self.ui_motReg_cmbBx))
        self.ui_objs.extend([var_lb, self.ui_motReg_cmbBx])

        # GS ROI regressor
        self.ui_GS_reg_chb = QtWidgets.QCheckBox("Regress global signal :")
        self.ui_GS_reg_chb.setChecked(self.GS_reg)
        self.ui_GS_reg_chb.stateChanged.connect(
                lambda state: self.set_param('GS_reg', state > 0))

        GSmask_hBLayout = QtWidgets.QHBoxLayout()
        self.ui_GS_mask_lnEd = QtWidgets.QLineEdit()
        self.ui_GS_mask_lnEd.setText(str(self.GS_mask))
        self.ui_GS_mask_lnEd.setReadOnly(True)
        self.ui_GS_mask_lnEd.setStyleSheet(
            'border: 0px none;')
        GSmask_hBLayout.addWidget(self.ui_GS_mask_lnEd)

        self.ui_GSmask_btn = QtWidgets.QPushButton('Set')
        self.ui_GSmask_btn.clicked.connect(
                lambda: self.set_param(
                        'GS_mask',
                        Path(self.ui_GS_mask_lnEd.text()).parent,
                        self.ui_GS_mask_lnEd.setText))
        GSmask_hBLayout.addWidget(self.ui_GSmask_btn)

        self.ui_objs.extend([self.ui_GS_reg_chb, self.ui_GS_mask_lnEd,
                             self.ui_GSmask_btn])
        ui_rows.append((self.ui_GS_reg_chb, GSmask_hBLayout))

        # WM ROI regressor
        self.ui_WM_reg_chb = QtWidgets.QCheckBox("Regress WM signal :")
        self.ui_WM_reg_chb.setChecked(self.WM_reg)
        self.ui_WM_reg_chb.stateChanged.connect(
                lambda state: self.set_param('WM_reg', state > 0))

        WMmask_hBLayout = QtWidgets.QHBoxLayout()
        self.ui_WM_mask_lnEd = QtWidgets.QLineEdit()
        self.ui_WM_mask_lnEd.setText(str(self.WM_mask))
        self.ui_WM_mask_lnEd.setReadOnly(True)
        self.ui_WM_mask_lnEd.setStyleSheet(
            'border: 0px none;')
        WMmask_hBLayout.addWidget(self.ui_WM_mask_lnEd)

        self.ui_WMmask_btn = QtWidgets.QPushButton('Set')
        self.ui_WMmask_btn.clicked.connect(
                lambda: self.set_param(
                        'WM_mask',
                        Path(self.ui_WM_mask_lnEd.text()).parent,
                        self.ui_WM_mask_lnEd.setText))
        WMmask_hBLayout.addWidget(self.ui_WMmask_btn)

        self.ui_objs.extend([self.ui_WM_reg_chb, self.ui_WM_mask_lnEd,
                             self.ui_WMmask_btn])
        ui_rows.append((self.ui_WM_reg_chb, WMmask_hBLayout))

        # Vent ROI regressor
        self.ui_Vent_reg_chb = QtWidgets.QCheckBox("Regress Vent signal :")
        self.ui_Vent_reg_chb.setChecked(self.Vent_reg)
        self.ui_Vent_reg_chb.stateChanged.connect(
                lambda state: self.set_param('Vent_reg', state > 0))

        Ventmask_hBLayout = QtWidgets.QHBoxLayout()

        self.ui_Vent_mask_lnEd = QtWidgets.QLineEdit()
        self.ui_Vent_mask_lnEd.setText(str(self.Vent_mask))
        self.ui_Vent_mask_lnEd.setReadOnly(True)
        self.ui_Vent_mask_lnEd.setStyleSheet(
            'border: 0px none;')
        Ventmask_hBLayout.addWidget(self.ui_Vent_mask_lnEd)

        self.ui_Ventmask_btn = QtWidgets.QPushButton('Set')
        self.ui_Ventmask_btn.clicked.connect(
                lambda: self.set_param(
                        'Vent_mask',
                        Path(self.ui_Vent_mask_lnEd.text()).parent,
                        self.ui_Vent_mask_lnEd.setText))
        Ventmask_hBLayout.addWidget(self.ui_Ventmask_btn)

        self.ui_objs.extend([self.ui_Vent_reg_chb, self.ui_Vent_mask_lnEd,
                             self.ui_Ventmask_btn])
        ui_rows.append((self.ui_Vent_reg_chb, Ventmask_hBLayout))

        # phys_reg
        var_lb = QtWidgets.QLabel("RICOR regressor :")
        self.ui_physReg_cmbBx = QtWidgets.QComboBox()
        self.ui_physReg_cmbBx.addItems(
                ['None', '8 RICOR (4 Resp and 4 Card)']
                )
        ci = {'None': 0, 'RICOR8': 1, 'RVT5': 2,
              'RVT+RICOR13': 3}[self.phys_reg]
        self.ui_physReg_cmbBx.setCurrentIndex(ci)
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
        self.ui_loadDesMtx_btn = QtWidgets.QPushButton('Set')
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
        self.ui_showDesMtx_btn.setText('Show desing matrix')
        if self.desMtx_read is None:
            self.ui_showDesMtx_btn.setEnabled(False)
        else:
            self.ui_showDesMtx_btn.setEnabled(True)

        # --- Checkbox row ----------------------------------------------------
        # Restrocpective process
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
    def get_params(self):
        all_opts = super().get_params()
        excld_opts = ('desMtx0', 'mot0', 'tshift',
                      'mask_byte', 'retrocols', 'volreg', 'YMtx', 'TR',
                      'desMtx', 'maskV', 'motcols', 'rtp_retrots',
                      'col_names_read', 'Y_mean', 'Y_mean_mask', 'GS_maskdata',
                      'WM_maskdata', 'Vent_maskdata', 'GS_col', 'WM_col',
                      'Vent_col', 'desMtx_read', 'mask_src_proc', 'work_dir')
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

    # Set logging
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO,
        format='%(asctime)s.%(msecs)04d,[%(levelname)s],%(name)s,%(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S')

    # --- Test ---
    # test data directory
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
    N_vols = img.shape[-1]

    work_dir = test_dir / 'work'
    if not work_dir.is_dir():
        work_dir.mkdir()

    # --- Create RTP instances and set parameters ---
    from rtpspy.rtp_tshift import RtpTshift
    from rtpspy.rtp_volreg import RtpVolreg
    from rtpspy.rtp_smooth import RtpSmooth
    from rtpspy.rtp_ttl_physio import RtpTTLPhysio

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

    # Set RtpTTLPhysio
    rtp_physio = RtpTTLPhysio(
        sample_freq=40, device='Dummy', sim_card_f=ecg_f, sim_resp_f=resp_f)

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
        'anat_mprage_WM_al_func.nii.gz'
    rtp_regress.Vent_reg = True
    rtp_regress.Vent_mask = test_dir / 'work' / 'RTP' / \
        'anat_mprage_Vent_al_func.nii.gz'
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

    # Chain RTPS: tshift -> volreg -> smooth -> regress
    rtp_tshift.next_proc = rtp_volreg
    rtp_volreg.next_proc = rtp_smooth
    rtp_smooth.next_proc = rtp_regress
    rtp_tshift.end_reset()

    # Wait for the rtp_physio to start recording
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
    rtp_tshift.end_reset()  # Reset propagetes to the chained processes

    # --- Plot ---
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(proc_delay[1:], bins='auto')
    np.median(proc_delay[1:])
