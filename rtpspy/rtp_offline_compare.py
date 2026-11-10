#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare the real-time (online) RTPSpy processing with standard offline
processing of the same run.

This is the evaluation used in Misaki and Bodurka (2022, Front Neurosci,
doi:10.3389/fnins.2022.834827) to justify the online pipeline, made reusable
for your own data and parameters.

1. Online: the RTPSpy modules are chained as in RtpApp
   (VOLREG -> TSHIFT -> SMOOTH -> REGRESS) and the volumes are fed one by
   one. The RETROICOR regressors of each volume are made only from the
   physio signals recorded up to that volume (causal), as in a real scan.
   The output of each module is saved (online/RTP/), with the online QC.
2. Offline: the whole run is processed with the AFNI programs, with the
   settings matched to the online ones:
       VOLREG  3dvolreg -base <same volume> -<same interpolation>
       TSHIFT  3dTshift -<same method> -tzero <ref_time> -ignore <n>
               -tpattern @<same slice timing>
       SMOOTH  3dBlurInMask -FWHM <same> -mask <RTP mask> -preserve
               (the same algorithm as RtpSmooth: mri_blur3D_inmask)
       REGRESS ordinary least squares (numpy) over the whole run with the
               same regressors: Legendre polynomials (RtpRegress.poly_reg
               for the run length), motion (mot6/mot12/dmot6 from the
               offline 3dvolreg), mean GS/WM/Vent signals of the offline
               volume-registered data in the same masks, and RICOR8 made by
               RtpRetroTS from the whole physio signals. The data are scaled
               to percent of the run mean, as RtpRegress does.
3. The online and offline outputs are compared over the volumes that the
   online pipeline output (see compare_runs).
4. metrics.csv, roi_series.csv, report.png, and report.html (self-contained)
   are written to the output directory.

Usage
-----
python -m rtpspy.rtp_offline_compare --input func.nii.gz \
    --masks_dir work/RTP --roi work/RTP/ROI_alFunc.nii.gz \
    --card ECG.1D --resp Resp.1D --physio_freq 40 --out compare_out

The AFNI programs (3dvolreg, 3dTshift, 3dBlurInMask) must be on PATH.

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import argparse
import base64
import copy
import html
import json
import codecs
import logging
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
import nibabel as nib

try:
    from .rtp_volreg import RtpVolreg
    from .rtp_tshift import RtpTshift
    from .rtp_smooth import RtpSmooth
    from .rtp_regress import RtpRegress, NORM_REGS, MOT_NAMES, DMOT_NAMES
    from .rtp_retrots import RtpRetroTS
except ImportError:
    from rtpspy.rtp_volreg import RtpVolreg
    from rtpspy.rtp_tshift import RtpTshift
    from rtpspy.rtp_smooth import RtpSmooth
    from rtpspy.rtp_regress import RtpRegress, NORM_REGS, MOT_NAMES, \
        DMOT_NAMES
    from rtpspy.rtp_retrots import RtpRetroTS


# %% Constants ================================================================
# Processing stages in the order of the RtpApp chain
STAGES = ('VOLREG', 'TSHIFT', 'SMOOTH', 'REGRESS')

# Default parameters: those of rtpspy_system_check.py
DEFAULT_PARAMS = {
    "VOLREG": {"regmode": "cubic"},
    "TSHIFT": {"method": "cubic", "ignore_init": 3},
    "SMOOTH": {"blur_fwhm": 6.0},
    "REGRESS": {"mot_reg": "mot12", "GS_reg": True, "WM_reg": True,
                "Vent_reg": True, "phys_reg": "RICOR8", "wait_num": 30},
}

# RtpVolreg regmode / RtpTshift method -> AFNI option
AFNI_VOLREG_INTERP = {'linear': '-linear', 'cubic': '-cubic',
                      'fourier': '-Fourier', 'quintic': '-quintic',
                      'heptic': '-heptic'}
AFNI_TSHIFT_INTERP = {'linear': '-linear', 'cubic': '-cubic'}

# Mask filenames made by RtpApp.make_masks (glob patterns in masks_dir)
MASK_PATTERNS = {'RTP': 'RTP_mask.nii*', 'GS': 'GSR_mask.nii*',
                 'WM': '*_WM_alFunc.nii*', 'Vent': '*_Vent_alFunc.nii*'}

# Plot colors: online, offline, full-run GLM of the online input
COLORS = {'online': '#2a78d6', 'offline': '#eb6834', 'fullGLM': '#1baf7a'}

_logger = logging.getLogger('rtp_offline_compare')


# %% Physio ===================================================================
class CausalSavedPhysio():
    """
    RETROICOR regressors from saved physio signals, made at each volume only
    from the samples recorded up to the end of that volume, as
    RtpTTLPhysio.get_retrots does during a scan (the phases near the end of
    the data are estimated without future samples).

    Replaces RtpTTLPhysio as RtpRegress.rtp_physio.
    """

    def __init__(self, card, resp, sample_freq):
        self.card = np.asarray(card, dtype=float)
        self.resp = np.asarray(resp, dtype=float)
        self.sample_freq = float(sample_freq)
        self._retrots = RtpRetroTS()

    def get_retrots(self, tr, n_vol=np.inf, tshift=0, timeout=None):
        n_all = min(len(self.card), len(self.resp))
        if np.isinf(n_vol):
            n_vol = int(n_all / self.sample_freq / tr)
        n_smp = int(round(n_vol * tr * self.sample_freq))
        if n_smp > n_all:
            return None  # Not recorded yet
        return self._retrots.RetroTs(self.resp[:n_smp], self.card[:n_smp],
                                     tr, self.sample_freq, tshift, n_vol)


# %% Helpers ==================================================================
def load_params(param_f=None):
    """
    RTP parameters {module: {parameter: value}} from a JSON file (the format
    of rtp_sim.py; WATCH is ignored and TTLPHYSIO may give ecg_f, resp_f,
    and sample_freq). DEFAULT_PARAMS if param_f is None.
    """
    if param_f is None:
        return copy.deepcopy(DEFAULT_PARAMS)
    try:
        with open(param_f, 'r') as fp:
            params = json.load(fp)
    except json.JSONDecodeError:
        params = json.load(codecs.open(param_f, 'r', 'utf-8-sig'))
    params.pop('WATCH', None)
    reg = params.get('REGRESS', {})
    if reg.get('max_poly_order', 0) is None:
        reg['max_poly_order'] = np.inf
    return params


def find_masks(masks_dir=None, **override):
    """
    Mask files {'RTP', 'GS', 'WM', 'Vent'}: the files made by
    RtpApp.make_masks in masks_dir (MASK_PATTERNS), or the files given as
    keyword arguments (e.g., RTP='mask.nii.gz'). A missing mask is None.
    """
    masks = {}
    for key, pat in MASK_PATTERNS.items():
        if override.get(key) is not None:
            masks[key] = Path(override[key])
            continue
        masks[key] = None
        if masks_dir is not None:
            cands = sorted(Path(masks_dir).glob(pat))
            if len(cands) > 1:
                _logger.warning(f"Several {key} masks in {masks_dir}:"
                                f" {cands}. {cands[0]} is used.")
            if len(cands):
                masks[key] = cands[0]
    return masks


def _enabled(params, stage):
    return stage in params and params[stage].get('enabled', True)


def _set_module_params(pobj, params):
    """Set the parameters as RtpApp._set_module_params does."""
    for attr, val in params.items():
        if attr in ('enabled', 'ref_vol', 'ref_fname', 'mask_file',
                    'GS_mask', 'WM_mask', 'Vent_mask'):
            continue  # Set from the masks / ref_vol arguments
        pobj.set_param(attr, val)


def _ref_vol_str(func_f, ref_vol):
    """Reference volume as 'file[index]' (int ref_vol: index in func_f)."""
    if isinstance(ref_vol, (int, np.integer)):
        return f"{func_f}[{int(ref_vol)}]"
    return str(ref_vol) if '[' in str(ref_vol) else f"{ref_vol}[0]"


def _roi_mean(data, roi_in_mask):
    """
    Mean over the ROI voxels with nonzero values at each volume, as
    RtpApp._process does (data: volumes x mask voxels).
    """
    d = data[:, roi_in_mask]
    nz = np.abs(d) > 0
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(nz, d, 0).sum(1) / nz.sum(1)


class _Sink():
    """End of the online chain: records the indices of the output volumes."""

    def __init__(self):
        self.next_proc = None
        self.vol_idx = []

    def ready_proc(self):
        return True

    def do_proc(self, fmri_img, vol_idx=None, pre_proc_time=None, **kwargs):
        self.vol_idx.append(vol_idx)

    def end_reset(self):
        return None


# %% Online ===================================================================
def run_online(func_f, masks, params, physio=None, ref_vol=0, work_dir='.'):
    """
    Run the RTPSpy modules on func_f volume by volume, as RtpApp does.

    Parameters
    ----------
    func_f : str or Path
        fMRI run (4D NIfTI). TR and slice timing are read from it (or from
        its JSON sidecar) by RtpTshift.set_from_sample.
    masks : dict
        Mask files {'RTP', 'GS', 'WM', 'Vent'} (see find_masks).
    params : dict
        RTP parameters {module: {parameter: value}}.
    physio : CausalSavedPhysio, optional
        Physio signals for the RICOR8 regressors.
    ref_vol : int or str, optional
        VOLREG reference: volume index in func_f or 'file[index]'.
    work_dir : Path
        The module outputs are saved in work_dir/RTP.

    Returns
    -------
    dict
        'stages': enabled stages, '<STAGE>': output data (volumes x
        RTP-mask voxels, rows by volume index), 'motion': (volumes x 6),
        'out_idx': indices of the volumes output by the chain,
        'Y_mean': scaling mean of REGRESS, 'proc': module objects,
        'saved_files': {description: file}, 'TR', 'ref_time'.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    img = nib.load(func_f)
    n_vols = img.shape[-1]
    maskV = np.asanyarray(nib.load(masks['RTP']).dataobj) != 0

    # --- Make and set up the modules (RtpApp.RTP_setup) ---
    procs = {}
    if _enabled(params, 'VOLREG'):
        procs['VOLREG'] = RtpVolreg()
        _set_module_params(procs['VOLREG'], params['VOLREG'])
        procs['VOLREG'].set_ref_vol(_ref_vol_str(func_f, ref_vol))
    tshift = RtpTshift()  # TR and slice timing are needed also by REGRESS
    if _enabled(params, 'TSHIFT'):
        _set_module_params(tshift, params['TSHIFT'])
    tshift.set_from_sample(str(func_f))
    if _enabled(params, 'TSHIFT'):
        procs['TSHIFT'] = tshift
    if _enabled(params, 'SMOOTH'):
        procs['SMOOTH'] = RtpSmooth()
        _set_module_params(procs['SMOOTH'], params['SMOOTH'])
        procs['SMOOTH'].set_mask(str(masks['RTP']))
    if _enabled(params, 'REGRESS'):
        reg = RtpRegress()
        _set_module_params(reg, params['REGRESS'])
        # RtpApp._setup_regress
        reg.set_param('TR', tshift.TR)
        reg.set_param('tshift', tshift.ref_time)
        reg.mask_file = str(masks['RTP'])
        if reg.mot_reg != 'None':
            if 'VOLREG' not in procs:
                raise ValueError('mot_reg needs VOLREG.')
            reg.volreg = procs['VOLREG']
        if reg.phys_reg != 'None':
            if physio is None:
                raise ValueError(f"phys_reg={reg.phys_reg} needs the physio"
                                 " signals (--card, --resp).")
            reg.rtp_physio = physio
        for roi in ('GS', 'WM', 'Vent'):
            if getattr(reg, f"{roi}_reg"):
                if masks.get(roi) is None:
                    raise ValueError(f"{roi}_reg needs the {roi} mask.")
                setattr(reg, f"{roi}_mask", str(masks[roi]))
        # Noise ROI signals from the image before smoothing
        # (RtpApp.ready_to_run)
        if reg.GS_reg or reg.WM_reg or reg.Vent_reg:
            src = procs.get('VOLREG', procs.get('TSHIFT'))
            if src is None:
                raise ValueError("GS/WM/Vent regressors need VOLREG or"
                                 " TSHIFT.")
            reg.mask_src_proc = src
        reg.max_scan_length = max(reg.max_scan_length, n_vols)
        procs['REGRESS'] = reg

    stages = [st for st in STAGES if st in procs]
    if not stages:
        raise ValueError('No RTP module is enabled.')

    # --- Chain the modules (RtpApp.ready_to_run) ---
    sink = _Sink()
    for ii, st in enumerate(stages):
        pobj = procs[st]
        pobj.work_dir = work_dir
        pobj.save_proc = True
        pobj.max_scan_length = max(n_vols, getattr(pobj, 'max_scan_length',
                                                   0) or 0)
        pobj.next_proc = procs[stages[ii+1]] if ii+1 < len(stages) else sink
    chain = procs[stages[0]]
    chain.end_reset()
    if not chain.ready_proc():
        raise RuntimeError('Failed to get the RTP modules ready.')

    # --- Feed the volumes ---
    fmri_data = np.asanyarray(img.dataobj)
    fbase = Path(func_f).name.split('.nii')[0]
    for ii in range(n_vols):
        fmri_img = nib.Nifti1Image(fmri_data[:, :, :, ii], affine=img.affine)
        fmri_img.set_filename(f"{fbase}_nr_{ii+1:04d}.nii.gz")
        chain.do_proc(fmri_img, ii, time.time())

    # --- Collect the outputs (before end_reset clears them) ---
    out = {'stages': stages, 'proc': procs, 'TR': float(tshift.TR),
           'ref_time': float(tshift.ref_time),
           'out_idx': np.array(sorted(set(sink.vol_idx)), dtype=int)}
    for st in stages:
        sd = procs[st].saved_data
        if sd is None:
            raise RuntimeError(f"{st} output no volume.")
        arr = np.full((n_vols, int(maskV.sum())), np.nan, dtype=np.float32)
        last = min(procs[st]._saved_last_idx + 1, n_vols)
        first = max(procs[st]._proc_start_idx, 0)
        arr[first:last] = sd[maskV][:, first:last].T
        out[st] = arr
    if 'VOLREG' in procs:
        out['motion'] = procs['VOLREG']._motion[:n_vols].astype(float).copy()
    if 'REGRESS' in procs and procs['REGRESS'].Y_mean is not None:
        out['Y_mean'] = procs['REGRESS'].Y_mean.cpu().numpy()
    out['saved_files'] = {k: str(v) for k, v in chain.end_reset().items()}
    return out


# %% Offline ==================================================================
def _run_afni(cmd, log_fd):
    """Run an AFNI command and append its output to log_fd. The output
    file (-prefix) is checked, as some write errors do not change the exit
    code."""
    log_fd.write('$ ' + ' '.join(cmd) + '\n')
    log_fd.flush()
    ret = subprocess.run(cmd, stdout=log_fd, stderr=subprocess.STDOUT)
    out_f = Path(cmd[cmd.index('-prefix') + 1])
    if ret.returncode != 0 or not out_f.is_file() or \
            out_f.stat().st_size == 0:
        raise RuntimeError(f"{cmd[0]} failed (exit {ret.returncode});"
                           f" see {log_fd.name}")


def design_matrix(n, TR, mot_reg='None', motion=None, roi_sigs={},
                  ricor=None, max_poly_order=np.inf):
    """
    Regressors of a whole-run GLM with the RtpRegress definitions.

    Parameters
    ----------
    n : int
        Number of volumes.
    motion : array (n x 6)
        Motion parameters (3dvolreg order).
    roi_sigs : dict
        {'GS'|'WM'|'Vent': mean signal (n,)}.
    ricor : array (n x 8) or None
        RICOR8 regressors.

    Returns
    -------
    X : array (n x regressors), names : list
    """
    cols, names = [], []
    if mot_reg != 'None':
        dmot = np.vstack([np.zeros((1, 6)), np.diff(motion, axis=0)])
        if mot_reg in ('mot6', 'mot12'):
            cols.append(motion)
            names += list(MOT_NAMES)
        if mot_reg in ('mot12', 'dmot6'):
            cols.append(dmot)
            names += list(DMOT_NAMES)
    if ricor is not None:
        cols.append(ricor)
        names += ['Resp0', 'Resp1', 'Resp2', 'Resp3',
                  'Card0', 'Card1', 'Card2', 'Card3']
    for roi, sig in roi_sigs.items():
        cols.append(np.asarray(sig)[:, None])
        names.append(roi)
    poly = RtpRegress(max_poly_order=max_poly_order, onGPU=False).poly_reg(
        n, TR)
    cols.append(poly)
    names += [f"Poly{ii}" for ii in range(poly.shape[1])]
    X = np.concatenate(cols, axis=1).astype(float)

    # Standardize the motion and ROI regressors as RtpRegress does (this
    # does not change the residuals)
    for ii, nm in enumerate(names):
        if nm in NORM_REGS:
            sd = X[:, ii].std()
            X[:, ii] = (X[:, ii] - X[:, ii].mean()) / sd if sd > 1e-6 \
                else 0.0
    return X, names


def full_run_glm(Y, X):
    """
    Residuals of the whole-run GLM (numpy least squares). Y (volumes x
    voxels) is scaled to percent of its mean (clipped at 200) as in
    RtpRegress.

    Returns
    -------
    resid : array, Yp : scaled Y, Y_mean : array
    """
    Y_mean = Y.mean(axis=0)
    valid = np.abs(Y_mean) > 1e-6
    Yp = np.zeros_like(Y, dtype=float)
    Yp[:, valid] = np.minimum(Y[:, valid] / Y_mean[valid] * 100, 200)
    beta = np.linalg.lstsq(X, Yp, rcond=None)[0]
    return Yp - X @ beta, Yp, Y_mean


def run_offline(func_f, masks, params, online, card=None, resp=None,
                physio_freq=None, ref_vol=0, work_dir='.'):
    """
    Process the whole run with AFNI and a whole-run GLM, with the settings
    of the online run (see the module docstring).

    Returns
    -------
    dict with the same keys as run_online ('<STAGE>': volumes x RTP-mask
    voxels), plus 'X' and 'reg_names' of the GLM and 'commands_log'.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    for prog in ('3dvolreg', '3dTshift', '3dBlurInMask'):
        if shutil.which(prog) is None:
            raise RuntimeError(f"AFNI {prog} is not found on PATH.")

    procs = online['proc']
    stages = online['stages']
    maskV = np.asanyarray(nib.load(masks['RTP']).dataobj) != 0
    n_vols = nib.load(func_f).shape[-1]
    TR = online['TR']
    # First volume processed by the chain
    i0 = max([procs[st].ignore_init for st in stages])
    fbase = Path(func_f).name.split('.nii')[0]
    out = {'stages': stages}
    log_f = work_dir / 'afni_commands.log'

    def masked(fname):
        data = np.asanyarray(nib.load(fname).dataobj)
        return data[maskV].T.astype(np.float32)  # volumes x voxels

    cur = str(func_f)
    vr_data = None
    with open(log_f, 'w') as log_fd:
        if 'VOLREG' in stages:
            vr = procs['VOLREG']
            mot_f = work_dir / f"motion.{fbase}.1D"
            dst = work_dir / f"vr.{fbase}.nii.gz"
            _run_afni(['3dvolreg', '-overwrite', '-float',
                       AFNI_VOLREG_INTERP[vr.regmode],
                       '-base', _ref_vol_str(func_f, ref_vol),
                       '-1Dfile', str(mot_f), '-prefix', str(dst), cur],
                      log_fd)
            out['motion'] = np.loadtxt(mot_f)
            cur = str(dst)
            out['VOLREG'] = vr_data = masked(cur)

        if 'TSHIFT' in stages:
            ts = procs['TSHIFT']
            if ts.slice_dim != 2:
                raise NotImplementedError(
                    '3dTshift needs the slices along the third axis.')
            tpat_f = work_dir / 'slice_timing.1D'
            np.savetxt(tpat_f, np.asarray(ts.slice_timing)[None, :],
                       fmt='%.6f')
            dst = work_dir / f"ts.{Path(cur).name}"
            _run_afni(['3dTshift', '-overwrite',
                       AFNI_TSHIFT_INTERP[ts.method],
                       '-tzero', f"{ts.ref_time}", '-ignore', f"{i0}",
                       '-TR', f"{TR}s", '-tpattern', f"@{tpat_f}",
                       '-prefix', str(dst), cur], log_fd)
            cur = str(dst)
            out['TSHIFT'] = masked(cur)

        if 'SMOOTH' in stages:
            dst = work_dir / f"sm.{Path(cur).name}"
            _run_afni(['3dBlurInMask', '-overwrite', '-quiet', '-preserve',
                       '-FWHM', f"{procs['SMOOTH'].blur_fwhm}",
                       '-mask', str(masks['RTP']), '-input', cur,
                       '-prefix', str(dst)], log_fd)
            cur = str(dst)
            out['SMOOTH'] = masked(cur)
    out['commands_log'] = str(log_f)

    if 'REGRESS' in stages:
        reg = procs['REGRESS']
        idx = np.arange(i0, n_vols)
        prev = [st for st in stages if st != 'REGRESS']
        Y = out[prev[-1]][idx] if prev else masked(func_f)[idx]
        # ROI signals from the data before smoothing (as mask_src_proc)
        src = vr_data if vr_data is not None else out.get('TSHIFT')
        roi_sigs = {}
        for roi in ('GS', 'WM', 'Vent'):
            if getattr(reg, f"{roi}_reg"):
                rmask = np.asanyarray(
                    nib.load(masks[roi]).dataobj)[maskV] != 0
                roi_sigs[roi] = src[idx][:, rmask].mean(axis=1)
        ricor = None
        if reg.phys_reg != 'None':
            # Phases from the whole physio signals (offline)
            ricor = RtpRetroTS().RetroTs(resp, card, TR, physio_freq,
                                         reg.tshift, n_vols)[idx]
        motion = out['motion'][idx] if 'motion' in out else None
        X, names = design_matrix(len(idx), TR, reg.mot_reg, motion,
                                 roi_sigs, ricor, reg.max_poly_order)
        resid, _, Y_mean = full_run_glm(Y.astype(float), X)
        arr = np.full((n_vols, Y.shape[1]), np.nan, dtype=np.float32)
        arr[idx] = resid
        out['REGRESS'] = arr
        out['Y_mean'] = Y_mean
        out['X'], out['reg_names'] = X, names
        pd.DataFrame(X, columns=names, index=idx).rename_axis(
            'vol_idx').to_csv(work_dir / 'offline_design_matrix.csv',
                              float_format='%.6g')
        img = nib.load(masks['RTP'])
        vol = np.zeros((*maskV.shape, len(idx)), dtype=np.float32)
        vol[maskV] = resid.T
        nib.Nifti1Image(vol, img.affine).to_filename(
            work_dir / f"regRes.{Path(cur).name}")

        # Causality of the GLM: the whole-run GLM of the online REGRESS input
        # with the online regressors (motion and ROI signals of the online
        # VOLREG/TSHIFT output; RICOR8 from the whole physio signals).
        Y_on = online[prev[-1]][idx] if prev else masked(func_f)[idx]
        src_on = online.get('VOLREG', online.get('TSHIFT'))
        roi_on = {}
        for roi in roi_sigs:
            rmask = np.asanyarray(nib.load(masks[roi]).dataobj)[maskV] != 0
            roi_on[roi] = src_on[idx][:, rmask].mean(axis=1)
        mot_on = online['motion'][idx] if 'motion' in online else None
        X_on, _ = design_matrix(len(idx), TR, reg.mot_reg, mot_on, roi_on,
                                ricor, reg.max_poly_order)
        # Scaled with the online Y_mean so that the residuals are in the
        # units of the online residuals
        scale = online.get('Y_mean')
        if scale is not None:
            valid = np.abs(scale) > 1e-6
            Yp = np.zeros_like(Y_on, dtype=float)
            Yp[:, valid] = np.minimum(Y_on[:, valid] / scale[valid] * 100,
                                      200)
            beta = np.linalg.lstsq(X_on, Yp, rcond=None)[0]
            resid_on = Yp - X_on @ beta
        else:
            resid_on = full_run_glm(Y_on.astype(float), X_on)[0]
        arr = np.full((n_vols, Y.shape[1]), np.nan, dtype=np.float32)
        arr[idx] = resid_on
        out['REGRESS_fullGLM_online_input'] = arr

    return out


# %% Comparison ===============================================================
def compare_series(a, b):
    """
    Voxelwise comparison of two time series arrays (volumes x voxels).

    Returns
    -------
    r : array (voxels,)
        Pearson correlation (NaN for a constant voxel).
    rmse : array (voxels,)
        Root mean square difference.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    am = a - a.mean(axis=0)
    bm = b - b.mean(axis=0)
    den = np.sqrt((am ** 2).sum(axis=0) * (bm ** 2).sum(axis=0))
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.where(den > 0, (am * bm).sum(axis=0) / den, np.nan)
    rmse = np.sqrt(((a - b) ** 2).mean(axis=0))
    return r, rmse


def _corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def _var_removed(y_in, resid):
    """Median over voxels of the % of the input variance removed."""
    v_in = np.var(y_in, axis=0)
    ok = v_in > 0
    return float(np.median(100 * (1 - np.var(resid, axis=0)[ok] / v_in[ok])))


def compare_runs(online, offline, roi_in_mask=None):
    """
    Compare the online and offline outputs over the volumes that the
    online pipeline output (online['out_idx']).

    Returns
    -------
    metrics : DataFrame (stage, metric, value)
    roi_df : DataFrame of the ROI mean series (None without ROI)
    vox_r : {stage: voxelwise correlation array}
    """
    idx = online['out_idx']
    rows = []

    def add(stage, metric, value):
        rows.append({'stage': stage, 'metric': metric, 'value': value})

    add('ALL', 'n_volumes_compared', len(idx))
    add('ALL', 'first_volume_compared', int(idx[0]) if len(idx) else -1)
    if 'REGRESS' in online['stages']:
        add('REGRESS', 'first_realtime_fit_volume',
            int(idx[0]) + online['proc']['REGRESS'].wait_num)

    # Motion parameters
    if 'motion' in online and 'motion' in offline:
        mon, moff = online['motion'][idx], offline['motion'][idx]
        for ii, nm in enumerate(MOT_NAMES):
            add('VOLREG', f"motion_{nm}_r", _corr(mon[:, ii], moff[:, ii]))
            add('VOLREG', f"motion_{nm}_maxabsdiff",
                float(np.max(np.abs(mon[:, ii] - moff[:, ii]))))

        def fd(m):
            d = np.abs(np.diff(m, axis=0))
            return d[:, 3:].sum(1) + 50 * np.deg2rad(d[:, :3]).sum(1)
        add('VOLREG', 'meanFD_online_mm', float(fd(mon).mean()))
        add('VOLREG', 'meanFD_offline_mm', float(fd(moff).mean()))

    # Voxelwise and ROI comparisons per stage
    vox_r = {}
    roi_cols = {'vol_idx': idx}
    pairs = [(st, online[st], offline[st]) for st in online['stages']
             if st in offline]
    if 'REGRESS_fullGLM_online_input' in offline:
        pairs.append(('REGRESS_causal', online['REGRESS'],
                      offline['REGRESS_fullGLM_online_input']))
    # REGRESS outputs the volumes before its first fit retrospectively
    # (fitted together); the later ones are fitted one by one in real time.
    first_fit = None
    if 'REGRESS' in online['stages']:
        reg = online['proc']['REGRESS']
        first_fit = int(idx[0]) + reg.wait_num
    for st, a, b in pairs:
        a, b = a[idx], b[idx]
        r, rmse = compare_series(a, b)
        vox_r[st] = r
        add(st, 'voxel_r_median', float(np.nanmedian(r)))
        add(st, 'voxel_r_p5', float(np.nanpercentile(r, 5)))
        add(st, 'voxel_rmse_median', float(np.median(rmse)))
        add(st, 'voxel_rmse_p5', float(np.percentile(rmse, 5)))
        if roi_in_mask is not None:
            ra, rb = _roi_mean(a, roi_in_mask), _roi_mean(b, roi_in_mask)
            add(st, 'roi_r', _corr(ra, rb))
        if st.startswith('REGRESS') and first_fit is not None:
            late = idx >= first_fit
            if late.sum() >= 3:
                r_l = compare_series(a[late], b[late])[0]
                add(st, 'voxel_r_median_realtime_vols',
                    float(np.nanmedian(r_l)))
                if roi_in_mask is not None:
                    add(st, 'roi_r_realtime_vols',
                        _corr(ra[late], rb[late]))
            if st == 'REGRESS_causal':
                roi_cols['REGRESS_fullGLM_online_input'] = rb
            else:
                roi_cols[f"{st}_online"] = ra
                roi_cols[f"{st}_offline"] = rb

    # Noise variance removed by REGRESS (% of the variance of the REGRESS
    # input, in the same units as the residuals)
    if 'REGRESS' in online['stages']:
        prev = [st for st in online['stages'] if st != 'REGRESS']
        if prev:
            for label, run, resid in (
                    ('online', online, online['REGRESS']),
                    ('offline', offline, offline['REGRESS']),
                    ('fullGLM_online_input', online,
                     offline.get('REGRESS_fullGLM_online_input'))):
                if resid is None or run.get('Y_mean') is None:
                    continue
                ym = run['Y_mean']
                ok = np.abs(ym) > 1e-6
                y_in = run[prev[-1]][idx][:, ok] / ym[ok] * 100
                add('REGRESS', f"var_removed_pct_{label}",
                    _var_removed(y_in, resid[idx][:, ok]))

    metrics = pd.DataFrame(rows, columns=['stage', 'metric', 'value'])
    roi_df = pd.DataFrame(roi_cols) if roi_in_mask is not None else None
    return metrics, roi_df, vox_r


# %% Report ===================================================================
def make_figure(online, offline, roi_df, vox_r, out_f):
    """Plot the motion, the ROI series, and the voxelwise correlations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    idx = online['out_idx']
    has_mot = 'motion' in online and 'motion' in offline
    nrow = 1 + int(has_mot) * 2 + int(roi_df is not None)
    fig = plt.figure(figsize=(11, 2.6 * nrow + 0.5), layout='constrained')
    gs = fig.add_gridspec(nrow, 3)
    row = 0
    style = dict(lw=1.5)
    if has_mot:
        units = ['deg'] * 3 + ['mm'] * 3
        for ii, nm in enumerate(MOT_NAMES):
            ax = fig.add_subplot(gs[row + ii // 3, ii % 3])
            ax.plot(idx, online['motion'][idx, ii], color=COLORS['online'],
                    label='online', **style)
            ax.plot(idx, offline['motion'][idx, ii],
                    color=COLORS['offline'], label='offline', **style)
            ax.set_title(f"{nm} ({units[ii]})", fontsize=10)
            if ii == 0:
                ax.legend(fontsize=8, frameon=False)
        row += 2

    if roi_df is not None:
        ax = fig.add_subplot(gs[row, :])
        last = [st for st in STAGES if f"{st}_online" in roi_df][-1]
        ax.plot(roi_df.vol_idx, roi_df[f"{last}_online"],
                color=COLORS['online'], label=f"online {last}", **style)
        ax.plot(roi_df.vol_idx, roi_df[f"{last}_offline"],
                color=COLORS['offline'], label=f"offline {last}", **style)
        if 'REGRESS_fullGLM_online_input' in roi_df:
            ax.plot(roi_df.vol_idx, roi_df['REGRESS_fullGLM_online_input'],
                    color=COLORS['fullGLM'], ls='--',
                    label='whole-run GLM of the online input', **style)
        ax.set_title('ROI mean signal', fontsize=10)
        ax.set_xlabel('Volume index')
        ax.legend(fontsize=8, frameon=False, ncol=3)
        row += 1

    ax = fig.add_subplot(gs[row, :])
    labels = list(vox_r.keys())
    data = [v[np.isfinite(v)] for v in vox_r.values()]
    try:
        ax.boxplot(data, orientation='horizontal', showfliers=False,
                   widths=0.5)
    except TypeError:  # matplotlib < 3.10
        ax.boxplot(data, vert=False, showfliers=False, widths=0.5)
    ax.set_yticks(range(1, len(labels) + 1), labels)
    ax.invert_yaxis()
    ax.set_xlabel('Voxelwise correlation, online vs offline')
    ax.grid(axis='x', alpha=0.3)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    fig.savefig(out_f, dpi=100)
    plt.close(fig)


def write_html(out_f, metrics, png_f, info):
    """Self-contained HTML report (metrics table and the figure)."""
    with open(png_f, 'rb') as fd:
        png64 = base64.b64encode(fd.read()).decode()
    tbl = metrics.copy()
    tbl['value'] = [f"{v:.4g}" if isinstance(v, float) else str(v)
                    for v in tbl['value']]
    info_rows = ''.join(
        f"<tr><th>{html.escape(str(k))}</th>"
        f"<td>{html.escape(str(v))}</td></tr>" for k, v in info.items())
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Online vs offline</title>
<style>
:root {{ --bg:#fcfcfb; --fg:#0b0b0b; --muted:#52514e; --line:#dddcd7; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#1a1a19; --fg:#f0efec; --muted:#c3c2b7; --line:#383835; }}
}}
body {{ background:var(--bg); color:var(--fg); margin:0 auto; padding:16px;
  max-width:1100px; font:14px/1.45 system-ui, sans-serif; }}
table {{ border-collapse:collapse; margin:8px 0 20px; }}
th, td {{ border-bottom:1px solid var(--line); padding:3px 10px;
  text-align:left; font-variant-numeric:tabular-nums; }}
th {{ color:var(--muted); font-weight:500; }}
img {{ max-width:100%; background:#fff; }}
p.note {{ color:var(--muted); }}
</style></head><body>
<h1>RTPSpy real-time vs offline processing</h1>
<table>{info_rows}</table>
<img alt="Motion, ROI signal, and voxelwise correlation"
 src="data:image/png;base64,{png64}">
<h2>Metrics</h2>
<p class="note">Compared over the volumes output by the online pipeline,
within the RTP mask. REGRESS_causal compares the online (cumulative GLM)
residuals with the whole-run GLM residuals of the same online input.</p>
{tbl.to_html(index=False, border=0)}
</body></html>
"""
    Path(out_f).write_text(doc)


# %% Main function ============================================================
def compare_online_offline(func_f, out_dir, masks_dir=None, roi_f=None,
                           param_f=None, params=None, card_f=None,
                           resp_f=None, physio_freq=None, ref_vol=0,
                           mask_files={}):
    """
    Run the online and offline processing of func_f and compare them.

    Parameters
    ----------
    func_f : str or Path
        fMRI run.
    out_dir : str or Path
        Output directory (online/, offline/, metrics.csv, roi_series.csv,
        report.png, report.html).
    masks_dir : str or Path, optional
        Directory of the masks made by RtpApp.make_masks (see find_masks).
    roi_f : str or Path, optional
        ROI mask for the ROI mean signal.
    param_f : str or Path, optional
        JSON file of the RTP parameters (see load_params). params (dict) can
        be given instead. The default is DEFAULT_PARAMS.
    card_f, resp_f : str or Path, optional
        Cardiac and respiration signals from the scan onset (one value per
        line), needed for phys_reg='RICOR8'.
    physio_freq : float, optional
        Sampling frequency (Hz) of the physio signals.
    ref_vol : int or str, optional
        VOLREG reference: volume index of func_f or 'file[index]'.
    mask_files : dict, optional
        Mask files overriding those in masks_dir ({'RTP', 'GS', 'WM',
        'Vent'}).

    Returns
    -------
    dict of the output files and the metrics DataFrame.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if params is None:
        params = load_params(param_f)
    params = copy.deepcopy(params)
    ttl = params.pop('TTLPHYSIO', None) or {}
    card_f = card_f or ttl.get('ecg_f')
    resp_f = resp_f or ttl.get('resp_f')
    physio_freq = physio_freq or ttl.get('sample_freq')

    masks = find_masks(masks_dir, **mask_files)
    if masks['RTP'] is None:
        raise ValueError('RTP mask is not found.')

    card = resp = physio = None
    if card_f is not None and resp_f is not None:
        if physio_freq is None:
            raise ValueError('physio_freq must be given.')
        card, resp = np.loadtxt(card_f), np.loadtxt(resp_f)
        physio = CausalSavedPhysio(card, resp, physio_freq)

    t0 = time.time()
    online = run_online(func_f, masks, params, physio, ref_vol,
                        out_dir / 'online')
    t1 = time.time()
    offline = run_offline(func_f, masks, params, online, card, resp,
                          physio_freq, ref_vol, out_dir / 'offline')
    t2 = time.time()

    maskV = np.asanyarray(nib.load(masks['RTP']).dataobj) != 0
    roi_in_mask = None
    if roi_f is not None:
        roi_in_mask = np.asanyarray(nib.load(roi_f).dataobj)[maskV] > 0
        if not roi_in_mask.any():
            raise ValueError('ROI has no voxel in the RTP mask.')
    metrics, roi_df, vox_r = compare_runs(online, offline, roi_in_mask)

    files = {'metrics': out_dir / 'metrics.csv',
             'figure': out_dir / 'report.png',
             'report': out_dir / 'report.html'}
    metrics.to_csv(files['metrics'], index=False, float_format='%.6g')
    if roi_df is not None:
        files['roi_series'] = out_dir / 'roi_series.csv'
        roi_df.to_csv(files['roi_series'], index=False, float_format='%.6g')
    if 'motion' in online:
        pd.DataFrame(online['motion'], columns=list(MOT_NAMES)).to_csv(
            out_dir / 'online' / 'motion_online.csv', index_label='vol_idx',
            float_format='%.6g')
    make_figure(online, offline, roi_df, vox_r, files['figure'])

    procs = online['proc']
    info = {'Input': Path(func_f).name,
            'Volumes': online[online['stages'][0]].shape[0],
            'TR (s)': online['TR'], 'Stages': ' -> '.join(online['stages']),
            'VOLREG reference': _ref_vol_str(Path(func_f).name, ref_vol)}
    for st in online['stages']:
        info[f"{st} parameters"] = ', '.join(
            f"{k}={v}" for k, v in params.get(st, {}).items())
    if 'REGRESS' in procs:
        info['REGRESS wait_num (used)'] = procs['REGRESS'].wait_num
        if 'reg_names' in offline:
            info['Offline GLM regressors'] = ', '.join(offline['reg_names'])
    info['Offline'] = ('3dvolreg, 3dTshift, 3dBlurInMask (AFNI); whole-run'
                       ' GLM by numpy least squares')
    info['Run time (s)'] = f"online {t1 - t0:.1f}, offline {t2 - t1:.1f}"
    write_html(files['report'], metrics, files['figure'], info)
    with open(out_dir / 'run_info.json', 'w') as fd:
        json.dump({k: str(v) for k, v in info.items()} |
                  {'online_files': online['saved_files'],
                   'afni_log': offline['commands_log']}, fd, indent=2)

    files['metrics_df'] = metrics
    return files


# %% __main__ (command line interface) ========================================
def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Compare RTPSpy real-time processing with standard'
        ' offline (AFNI) processing of the same run.')
    parser.add_argument('--input', required=True, help='fMRI run (NIfTI).')
    parser.add_argument('--masks_dir', help='Directory of the masks made by'
                        ' RtpApp.make_masks (RTP_mask, GSR_mask,'
                        ' *_WM_alFunc, *_Vent_alFunc).')
    parser.add_argument('--rtp_mask', help='Processing mask (overrides'
                        ' masks_dir).')
    parser.add_argument('--gs_mask', help='Global signal mask.')
    parser.add_argument('--wm_mask', help='White matter mask.')
    parser.add_argument('--vent_mask', help='Ventricle mask.')
    parser.add_argument('--roi', help='ROI mask for the ROI mean signal.')
    parser.add_argument('--param', help='JSON file of RTP parameters'
                        ' (default: the rtpspy_system_check.py settings).')
    parser.add_argument('--card', help='Cardiac signal file (1D).')
    parser.add_argument('--resp', help='Respiration signal file (1D).')
    parser.add_argument('--physio_freq', type=float,
                        help='Physio sampling frequency (Hz).')
    parser.add_argument('--ref_vol', default='0',
                        help="VOLREG reference: volume index of the input"
                        " or 'file[index]' (default 0).")
    parser.add_argument('--out', required=True, help='Output directory.')
    parser.add_argument('--verbose', action='store_true',
                        help='Show the INFO logs of the RTP modules.')
    args = parser.parse_args(argv)

    logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s',
                        level=logging.INFO if args.verbose
                        else logging.WARNING)
    ref_vol = int(args.ref_vol) if args.ref_vol.isdigit() else args.ref_vol
    files = compare_online_offline(
        args.input, args.out, masks_dir=args.masks_dir, roi_f=args.roi,
        param_f=args.param, card_f=args.card, resp_f=args.resp,
        physio_freq=args.physio_freq, ref_vol=ref_vol,
        mask_files={'RTP': args.rtp_mask, 'GS': args.gs_mask,
                    'WM': args.wm_mask, 'Vent': args.vent_mask})
    with pd.option_context('display.max_rows', None, 'display.width', 120):
        print(files.pop('metrics_df').to_string(index=False))
    for k, v in files.items():
        print(f"{k}: {v}")


if __name__ == '__main__':
    main()
