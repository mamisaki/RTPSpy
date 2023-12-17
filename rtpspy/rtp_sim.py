#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTP simulation

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

import numpy as np
import nibabel as nib

try:
    from .rtp_physio import RTP_PHYSIO_DUMMY
    from .rtp_app import RtpApp
except Exception:
    from rtpspy.rtp_physio import RTP_PHYSIO_DUMMY
    from rtpspy.rtp_app import RtpApp


# %% RTP_SIM class ============================================================
class RTP_SIM(RtpApp):
    """ RTP simulation class """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def do_proc(self, fmri_img, vol_idx=None, pre_proc_time=0):
        """ Extract ROI average signal.
        """
        try:
            # Increment the number of received volume
            self.vol_num += 1
            if vol_idx is None:
                vol_idx = self.vol_num

            if vol_idx < self.ignore_init:
                # Skip ignore_init volumes
                return

            if self.proc_start_idx < 0:
                self.proc_start_idx = vol_idx

            dataV = fmri_img.get_fdata()

            # --- Initialize --------------------------------------------------
            if self.ROI_mask is None:
                # Load ROI mask
                self.ROI_mask = nib.load(self.ROI_orig).get_data()

            # --- Run the procress --------------------------------------------
            # Get mean signal in the ROI
            roimask = (self.ROI_mask > 0) & (np.abs(dataV) > 0.0)
            mean_sig = np.nanmean(dataV[roimask])

            # Online saving in a file
            save_vals = (f"{vol_idx},{mean_sig:.6f}")
            with open(self.sig_save_file, 'a') as save_fd:
                print(save_vals, file=save_fd)
            if self._verb:
                self.logmsg(f"Write data '{save_vals}'")

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
                msg = f'#{vol_idx}, ROI signal extraction is done for {f}'
                if pre_proc_time is not None:
                    msg += f' (took {proc_delay:.4f}s)'
                msg += '.'
                self.logmsg(msg)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errmsg = '{}, {}:{}'.format(
                    exc_type, exc_tb.tb_frame.f_code.co_filename,
                    exc_tb.tb_lineno)
            self.errmsg(str(e) + '\n' + errmsg, no_pop=True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ready_to_run(self):
        """ Ready running the process """

        # connect RTP modules
        self.proc_chain = None
        last_proc = None
        for proc in (['TSHIFT', 'VOLREG', 'SMOOTH', 'REGRESS']):
            pobj = self.rtp_objs[proc]
            if pobj.enabled:
                if self.proc_chain is None:
                    # First proc
                    self.proc_chain = pobj
                else:
                    last_proc.next_proc = pobj

                if proc == 'REGRESS':
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
            last_proc.next_proc = self  # Add self (RTP_SIM) at the last.

        # Reset process status
        self.proc_chain.end_reset()

        # Ready process sequence: proc_ready calls its child's proc_ready
        if not self.proc_chain.ready_proc():
            return

        if self._verb:
            # Print process parameters
            log_str = 'RTP parameters:\n'
            rtp = self.proc_chain
            while rtp is not None:
                log_str += f"# {type(rtp).__name__}\n"
                for k, v in rtp.get_params().items():
                    if k == 'ignore_init' and v == 0:
                        continue
                    log_str += f"#     {k}: {v}\n"

                rtp = rtp.next_proc
            log_str = log_str.rstrip()
            self.logmsg(log_str, show_ui=False)

        if self.ROI_mask is not None:
            self.sig_save_file = Path(self.work_dir) / 'ROI_signal.csv'

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end_run(self, quit_btn=False):

        # Save parameter list
        # Get parameters
        all_params = {}
        for rtp in ('TSHIFT', 'VOLREG', 'SMOOTH', 'REGRESS', 'EXTSIG'):
            if rtp not in self.rtp_objs or not self.rtp_objs[rtp].enabled:
                continue

            if not self.rtp_objs[rtp].enabled:
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


# %% run_simulation
def run_simulation(raw_fmri_f, anat_mri_f, rtp_param_f=None,
                   outprefix='rtpsim', fastSeg_batch_size=1, template_f=None,
                   ROI_template_f=None, verb=True, overwrite=False):
    """
    Run RTP simulation

    Parameters
    ----------
    raw_fmri_f : str or Path
        Raw fMRI filename.
    anat_mri_f : str or Path
        Anatomy MRI filename.
    rtp_param_f : str or Path, optional
        JSON file of RTP parameters. The default is None.
    outprefix : str or Path, optional
        Output path and filename prefix. The default is 'rtpsim'.
    fastSeg_batch_size : TYPE, optional
        DESCRIPTION. The default is 8.
    template_f : TYPE, optional
        Template brain filename. The default is None.
    ROI_template_f : TYPE, optional
        ROI mask filename on the template brain. The default is None.
    verb : bool, optional
        Print progress. The default is True.
    overwrite : TYPE, optional
        Overwrite processed anatomy files. The default is False.

    Returns
    -------
    out_files : list
        Output files.

    """
    print("RTP simulation")

    # Set work_dir
    if outprefix is not None:
        work_dir = Path(outprefix).parent
        if not work_dir.is_dir():
            os.makedirs(work_dir)
    else:
        work_dir = Path('.')

    # --- Make rtp_sim instance ---
    rtp_sim = RTP_SIM()
    rtp_sim.set_param('work_dir', work_dir)
    rtp_sim.verb = verb
    for proc in rtp_sim.rtp_objs.values():
        proc.verb = verb

    # --- Set RTP parameters ---
    if rtp_param_f is not None:
        assert Path(rtp_param_f).is_file()
        try:
            with open(rtp_param_f, 'r') as fp:
                rtp_params = json.load(fp)
        except json.JSONDecodeError:
            rtp_params = json.load(codecs.open(rtp_param_f, 'r', 'utf-8-sig'))

    # slice timing parameters
    if 'TSHIFT' not in rtp_params:
        rtp_params['TSHIFT'] = {}
    if 'slice_timing' not in rtp_params['TSHIFT']:
        rtp_params['TSHIFT']['slice_timing_from_sample'] = raw_fmri_f

    # REGRESS parameters
    if 'REGRESS' in rtp_params and \
            'enabled' in rtp_params['REGRESS'] and \
            rtp_params['REGRESS']['enabled']:
        if 'max_poly_order' in rtp_params['REGRESS']:
            if rtp_params['REGRESS']['max_poly_order'] is None:
                rtp_params['REGRESS']['max_poly_order'] = np.inf

        if 'phys_reg' in rtp_params['REGRESS']:
            if rtp_params['REGRESS']['phys_reg'] != 'None':
                if 'PHYSIO' not in rtp_params:
                    errmsg = '"PHYSIO" must be given when REGRESS phys_reg'
                    errmsg += ' is not "None"\n'
                    sys.stderr.write(errmsg)
                    sys.exit()

                # Set PHYSIO
                ecg_f = rtp_params['PHYSIO']['ecg_f']
                assert Path(ecg_f).is_file()
                resp_f = rtp_params['PHYSIO']['resp_f']
                assert Path(resp_f).is_file()
                sample_freq = rtp_params['PHYSIO']['sample_freq']
                rtp_sim.rtp_objs['PHYSIO'] = RTP_PHYSIO_DUMMY(
                    ecg_f, resp_f, sample_freq,
                    rtp_sim.rtp_objs['RETROTS'])

    # --- RTP setup -----------------------------------------------------------
    print("   mask creation ...")
    # Make mask images
    rtp_sim.fastSeg_batch_size = fastSeg_batch_size
    rtp_sim.make_masks(func_orig=str(raw_fmri_f)+"'[0]'", anat_orig=anat_mri_f,
                       template=template_f, ROI_template=ROI_template_f,
                       overwrite=overwrite)

    # RTP setup
    rtp_sim.RTP_setup(rtp_params=rtp_params)
    # Mask files made by make_masks() are automatically set in RTP_setup().

    # --- Run RTP simulation --------------------------------------------------
    print(f"   run simulation with {raw_fmri_f} ...")

    # Load data
    img = nib.load(raw_fmri_f)
    fmri_data = np.asanyarray(img.dataobj)
    N_vols = img.shape[-1]

    # Ready rtp_sim
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

    # --- Move result files to outprefix ---
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

    roi_f = work_dir / 'rtp_ROI_signal.csv'
    if roi_f.is_file():
        dst_f = work_dir / (out_fbase + '_' + roi_f.name)
        roi_f.rename(dst_f)
        out_files.append(dst_f)

    RTP_dir = work_dir / 'RTP'
    if RTP_dir.is_dir():
        RTP_dir.rmdir()

    return out_files


# %% __main__ (command line interface) ========================================
if __name__ == '__main__':
    # --- Parse arguments ---
    parser = argparse.ArgumentParser(description='RTP simulation.')
    parser.add_argument('--input', metavar='raw_fMRI_file', required=True,
                        help='Raw fMRI to be processed.')
    parser.add_argument('--anat', metavar='anatomy_MRI_file', required=True,
                        help='Anatomy MRI for creating masks.')
    parser.add_argument('--param', metavar='rtp_params.json',
                        help='json file for RTP options.')
    parser.add_argument('--prefix', metavar='out_dir/out_filename',
                        help='Output prefix.')
    parser.add_argument('--fastSeg_batch_size', metavar='N', type=int,
                        default=8, help='batch size of FastSeg.')
    parser.add_argument('--template', help='Template brain.')
    parser.add_argument('--ROI_template',
                        help='ROI mask on the template brain.')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='No log print.')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='overwite proceesed anatomy files.')
    args = parser.parse_args()

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
