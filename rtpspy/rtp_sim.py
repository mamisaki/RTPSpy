#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTP simulation: run the RTP processing offline on a saved fMRI file

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import os
import time
import sys
from datetime import datetime
import argparse
import json
import codecs
import traceback
import logging

import numpy as np
import nibabel as nib

try:
    from .rtp_retrots import RtpRetroTS
    from .rtp_app import RtpApp
except Exception:
    from rtpspy.rtp_retrots import RtpRetroTS
    from rtpspy.rtp_app import RtpApp


# %% SavedPhysio ==============================================================
class SavedPhysio():
    """RETROICOR regressors from saved physio signals.

    Offline replacement of RtpTTLPhysio for RtpRegress (rtp_physio), which
    calls get_retrots().
    """

    def __init__(self, card, resp, sample_freq):
        self.card = card
        self.resp = resp
        self.sample_freq = sample_freq
        self._retrots = RtpRetroTS()

    def get_retrots(self, tr, n_vol=np.inf, tshift=0, timeout=None):
        """Return the RETROICOR regressors of n_vol volumes from the scan
        onset (the start of the signals). None if the signals are not long
        enough."""
        max_vol = int(min(len(self.card), len(self.resp)) /
                      self.sample_freq / tr)
        if np.isinf(n_vol):
            n_vol = max_vol
        elif n_vol > max_vol:
            return None

        return self._retrots.RetroTs(self.resp, self.card, tr,
                                     self.sample_freq, tshift, n_vol)


# %% RTP_SIM class ============================================================
class RTP_SIM(RtpApp):
    """RtpApp that runs the RTP processing chain on a saved fMRI file."""

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def do_proc(self, fmri_img, vol_idx=None, pre_proc_time=0):
        """Extract the mean signal in the ROI and append it to a file."""
        try:
            self._vol_num += 1  # 1-based number of the received volumes
            if vol_idx is None:
                vol_idx = self._vol_num - 1  # 0-base index

            if vol_idx < self.ignore_init:
                return

            if self._proc_start_idx < 0:
                self._proc_start_idx = vol_idx

            dataV = fmri_img.get_fdata()

            # --- Initialize --------------------------------------------------
            if self.ROI_mask is None:
                self.ROI_mask = np.asarray(nib.load(self.ROI_orig).dataobj)

            # --- Run the process ---------------------------------------------
            roimask = (self.ROI_mask > 0) & (np.abs(dataV) > 0.0)
            mean_sig = np.nanmean(dataV[roimask])

            # Append the value to the file at each volume
            save_vals = (f"{vol_idx},{mean_sig:.6f}")
            with open(self.sig_save_file, 'a') as save_fd:
                print(save_vals, file=save_fd)
            self._logger.info(f"Write ROI data '{save_vals}'")

            # --- Post process ------------------------------------------------
            # Record the processing time
            tstamp = time.time()
            self._proc_time.append(tstamp)
            if pre_proc_time is not None:
                proc_delay = self._proc_time[-1] - pre_proc_time
                if self.save_delay:
                    self.proc_delay.append(proc_delay)

            # Log message
            if fmri_img.get_filename():
                fname = Path(fmri_img.get_filename()).name
            else:
                fname = "unknown.nii.gz"
            msg = f"#{vol_idx+1}"
            msg += f";tstamp={tstamp}"
            msg += f";ROI signal extraction;{fname}"
            if pre_proc_time is not None:
                msg += f';took {proc_delay:.4f}s'
            self._logger.info(msg)

        except Exception as e:
            self._logger.error(str(e) + '\n' + traceback.format_exc())

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ready_to_run(self):
        """Connect the enabled RTP modules and make them ready to run."""
        self.proc_chain = None
        last_proc = None
        for proc in (['TSHIFT', 'VOLREG', 'SMOOTH', 'REGRESS']):
            pobj = self.rtp_objs[proc]
            if pobj.enabled:
                if self.proc_chain is None:
                    self.proc_chain = pobj
                else:
                    last_proc.next_proc = pobj

                if proc == 'REGRESS':
                    # The global signal, WM, and Vent regressors are
                    # extracted from the output of the preceding module.
                    if pobj.GS_reg or pobj.WM_reg or pobj.Vent_reg:
                        if self.rtp_objs['VOLREG'].enabled:
                            pobj.mask_src_proc = self.rtp_objs['VOLREG']
                        elif self.rtp_objs['TSHIFT'].enabled:
                            pobj.mask_src_proc = self.rtp_objs['TSHIFT']
                        else:
                            pobj.mask_src_proc = self.rtp_objs['WATCH']

                last_proc = pobj

        last_proc.save_proc = True

        if self.ROI_orig and Path(self.ROI_orig).is_file():
            last_proc.next_proc = self  # Add self (RTP_SIM) at the end

        # Reset process status
        self.proc_chain.end_reset()

        # ready_proc() calls ready_proc() of the next module in the chain
        if not self.proc_chain.ready_proc():
            return

        # Log the process parameters
        log_str = 'RTP parameters:\n'
        rtp = self.proc_chain
        while rtp is not None:
            log_str += f"# {type(rtp).__name__}\n"
            for k, v in rtp.get_params().items():
                if k == 'ignore_init' and v == 0:
                    continue
                log_str += f"#     {k}: {v}\n"

            rtp = rtp.next_proc
        self._logger.info(log_str.rstrip())

        # Start a new ROI signal file (run_simulation renames it).
        self.sig_save_file = Path(self.work_dir) / 'rtp_ROI_signal.csv'
        if self.ROI_mask is not None:
            open(self.sig_save_file, 'w').close()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end_run(self, quit_btn=False):
        """Save the parameters in a text file and reset the modules."""
        all_params = {}
        for rtp in ('TSHIFT', 'VOLREG', 'SMOOTH', 'REGRESS'):
            if rtp not in self.rtp_objs or not self.rtp_objs[rtp].enabled:
                continue
            all_params[rtp] = self.rtp_objs[rtp].get_params()

        all_params[self.__class__.__name__] = self.get_params()

        save_dir = self.work_dir
        scan_name = datetime.isoformat(datetime.now()).split('.')[0]
        save_f = save_dir / f'rtp_sim_params_run{scan_name}.txt'
        with open(save_f, 'w') as fd:
            for rtp, opt_dict in all_params.items():
                fd.write('# {}\n'.format(rtp))
                for k in sorted(opt_dict.keys()):
                    val = opt_dict[k]
                    fd.write('{}: {}\n'.format(k, val))

        # End and reset all processes
        self.proc_chain.end_reset()


# %% run_simulation ===========================================================
def run_simulation(raw_fmri_f, anat_mri_f, rtp_param_f=None,
                   outprefix='rtpsim', fastSeg_batch_size=1, template_f=None,
                   ROI_template_f=None, verb=True, overwrite=False):
    """
    Run the RTP simulation.

    Parameters
    ----------
    raw_fmri_f : str or Path
        Raw fMRI filename.
    anat_mri_f : str or Path
        Anatomical MRI filename.
    rtp_param_f : str or Path, optional
        JSON file of RTP parameters. The default is None.
    outprefix : str or Path, optional
        Output path and filename prefix. The default is 'rtpsim'.
    fastSeg_batch_size : int, optional
        Batch size of FastSeg. The default is 1.
    template_f : str or Path, optional
        Template brain filename. The default is None.
    ROI_template_f : str or Path, optional
        ROI mask filename on the template brain. The default is None.
    verb : bool, optional
        Show the INFO logs of the RTP modules. The default is True.
    overwrite : bool, optional
        Overwrite processed anatomy files. The default is False.

    Returns
    -------
    out_files : list
        Output files.

    """
    print("RTP simulation")
    logging.getLogger().setLevel(logging.INFO if verb else logging.WARNING)

    # Set work_dir
    if outprefix is not None:
        work_dir = Path(outprefix).parent
        if not work_dir.is_dir():
            os.makedirs(work_dir)
    else:
        work_dir = Path('.')

    # --- Make an RTP_SIM instance ---
    rtp_sim = RTP_SIM()
    rtp_sim.set_param('work_dir', work_dir)

    # --- Set RTP parameters ---
    rtp_params = {}
    if rtp_param_f is not None:
        assert Path(rtp_param_f).is_file()
        try:
            with open(rtp_param_f, 'r') as fp:
                rtp_params = json.load(fp)
        except json.JSONDecodeError:
            rtp_params = json.load(codecs.open(rtp_param_f, 'r', 'utf-8-sig'))

    # The volumes are fed directly, not via WATCH.
    rtp_params.setdefault('WATCH', {})['enabled'] = False

    # Slice timing parameters are read from the fMRI file (RTP_setup).
    rtp_sim.func_param_ref = str(raw_fmri_f)

    # RETROICOR regressors from the saved physio signals
    saved_physio = None
    ttl_params = rtp_params.pop('TTLPHYSIO', None)

    # REGRESS parameters
    if 'REGRESS' in rtp_params and \
            'enabled' in rtp_params['REGRESS'] and \
            rtp_params['REGRESS']['enabled']:
        if 'max_poly_order' in rtp_params['REGRESS']:
            if rtp_params['REGRESS']['max_poly_order'] is None:
                rtp_params['REGRESS']['max_poly_order'] = np.inf

        if 'phys_reg' in rtp_params['REGRESS']:
            if rtp_params['REGRESS']['phys_reg'] != 'None':
                if ttl_params is None:
                    errmsg = '"TTLPHYSIO" must be given when REGRESS'
                    errmsg += ' phys_reg is not "None"\n'
                    sys.stderr.write(errmsg)
                    sys.exit()

                # RETROICOR regressors from the saved signals
                ecg_f = ttl_params['ecg_f']
                assert Path(ecg_f).is_file()
                resp_f = ttl_params['resp_f']
                assert Path(resp_f).is_file()
                sample_freq = ttl_params['sample_freq']
                resp = np.loadtxt(resp_f)
                card = np.loadtxt(ecg_f)
                saved_physio = SavedPhysio(card, resp, sample_freq)

    # --- RTP setup -----------------------------------------------------------
    print("   Creating masks ...")
    rtp_sim.fastSeg_batch_size = fastSeg_batch_size
    rtp_sim.make_masks(func_orig=str(raw_fmri_f)+"'[0]'", anat_orig=anat_mri_f,
                       template=template_f, ROI_template=ROI_template_f,
                       overwrite=overwrite)

    # The mask files made by make_masks() are set in RTP_setup().
    if rtp_sim.RTP_setup(rtp_params=rtp_params) != 0:
        sys.stderr.write("RTP setup failed.\n")
        sys.exit(1)
    if saved_physio is not None:
        rtp_sim.rtp_objs['REGRESS'].set_param('rtp_physio', saved_physio)

    # --- Run RTP simulation --------------------------------------------------
    print(f"   Running simulation with {raw_fmri_f} ...")

    img = nib.load(raw_fmri_f)
    fmri_data = np.asanyarray(img.dataobj)
    N_vols = img.shape[-1]

    rtp_sim.ready_to_run()

    if outprefix is not None:
        outprefix = Path(outprefix)
        if '.gz' in outprefix.suffixes:
            out_fbase = Path(outprefix.stem).stem
        else:
            out_fbase = outprefix.stem
    else:
        out_fbase = 'rtp_sim'

    for ii in range(N_vols):
        save_filename = f"{out_fbase}_nr_{ii:04d}.nii.gz"
        fmri_img = nib.Nifti1Image(fmri_data[:, :, :, ii], affine=img.affine)
        fmri_img.set_filename(save_filename)
        st = time.time()
        rtp_sim.proc_chain.do_proc(fmri_img, ii, st)

    rtp_sim.end_run()

    # --- Move the result files to work_dir ---
    out_files = []
    proc = rtp_sim.proc_chain
    while proc is not None:
        if hasattr(proc, 'saved_filename') and proc.saved_filename is not None:
            saved_f = Path(proc.saved_filename)
            if saved_f.is_file():
                dst_f = work_dir / saved_f.name
                saved_f.rename(dst_f)
                out_files.append(dst_f)
        proc = proc.next_proc

    roi_f = Path(rtp_sim.sig_save_file)
    if roi_f.is_file():
        dst_f = work_dir / (out_fbase + '_' + roi_f.name)
        roi_f.rename(dst_f)
        out_files.append(dst_f)

    # Remove the RTP directory only if it is empty (make_masks puts the
    # mask files there).
    RTP_dir = work_dir / 'RTP'
    if RTP_dir.is_dir() and not any(RTP_dir.iterdir()):
        RTP_dir.rmdir()

    return out_files


# %% __main__ (command line interface) ========================================
if __name__ == '__main__':
    # --- Parse arguments ---
    parser = argparse.ArgumentParser(description='RTP simulation.')
    parser.add_argument('--input', metavar='raw_fMRI_file', required=True,
                        help='Raw fMRI to be processed.')
    parser.add_argument('--anat', metavar='anatomy_MRI_file', required=True,
                        help='Anatomical MRI for creating masks.')
    parser.add_argument('--param', metavar='rtp_params.json',
                        help='JSON file of RTP parameters.')
    parser.add_argument('--prefix', metavar='out_dir/out_filename',
                        help='Output prefix.')
    parser.add_argument('--fastSeg_batch_size', metavar='N', type=int,
                        default=8, help='Batch size of FastSeg.')
    parser.add_argument('--template', help='Template brain.')
    parser.add_argument('--ROI_template',
                        help='ROI mask on the template brain.')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Do not print logs.')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite processed anatomy files.')
    args = parser.parse_args()

    logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s')

    raw_fmri_f = args.input
    anat_mri_f = args.anat
    rtp_param_f = args.param
    outprefix = args.prefix
    fastSeg_batch_size = args.fastSeg_batch_size
    template_f = args.template
    ROI_template_f = args.ROI_template
    verb = not args.quiet
    overwrite = args.overwrite

    # --- Check files and directory ---
    assert Path(raw_fmri_f).is_file()
    assert Path(anat_mri_f).is_file()

    out_files = run_simulation(
        raw_fmri_f, anat_mri_f, rtp_param_f=rtp_param_f, outprefix=outprefix,
        fastSeg_batch_size=fastSeg_batch_size, template_f=template_f,
        ROI_template_f=ROI_template_f, verb=verb, overwrite=overwrite)

    for out_f in out_files:
        print(f"Output: {out_f}")
