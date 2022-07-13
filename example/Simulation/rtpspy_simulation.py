#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import time
import sys

import numpy as np
import nibabel as nib

from rtpspy import RTP_APP
from rtpspy import RTP_PHYSIO_DUMMY


# %% main =====================================================================
if __name__ == '__main__':
    print("=" * 80)
    print("RTPSpy simulation")
    sys.stdout.flush()

    # --- Filenames -------------------------------------------------------
    # test data directory
    test_dir = Path(__file__).absolute().parent.parent.parent / 'test'

    # Set test data files
    testdata_f = test_dir / 'func_epi.nii.gz'
    anat_f = test_dir / 'anat_mprage.nii.gz'
    template_f = test_dir / 'MNI152_2009_template.nii.gz'
    ROI_template_f = test_dir / 'MNI152_2009_template_LAmy.nii.gz'
    ecg_f = test_dir / 'ECG.1D'
    resp_f = test_dir / 'Resp.1D'

    work_dir = Path(__file__).absolute().parent / 'work'
    if not work_dir.is_dir():
        work_dir.mkdir()

    # --- Create RTP_APP instance -----------------------------------------
    rtp_app = RTP_APP(work_dir=work_dir)

    # --- Make mask images ------------------------------------------------
    rtp_app.fastSeg_batch_size = 2  # Adjust the size according to GPU
    rtp_app.make_masks(func_orig=str(testdata_f)+"'[0]'", anat_orig=anat_f,
                       template=template_f, ROI_template=ROI_template_f,
                       overwrite=False)

    # --- Set up RTP ------------------------------------------------------
    # Set RTP_PHYSIO to RTP_PHYSIO_DUMMY
    sample_freq = 40
    rtp_app.rtp_objs['PHYSIO'] = RTP_PHYSIO_DUMMY(
        ecg_f, resp_f, sample_freq, rtp_app.rtp_objs['RETROTS'])

    # RTP parameters
    rtp_params = {'WATCH': {'enabled': False},
                  'TSHIFT': {'slice_timing_from_sample': testdata_f,
                             'method': 'cubic', 'ignore_init': 3},
                  'VOLREG': {'regmode': 'cubic'},
                  'SMOOTH': {'blur_fwhm': 6.0},
                  'REGRESS': {'mot_reg': 'mot12',
                              'GS_reg': True, 'WM_reg': True,
                              'Vent_reg': True, 'phys_reg': 'RICOR8',
                              'wait_num': 40}}

    # RTP setup
    rtp_app.RTP_setup(rtp_params=rtp_params)
    # Mask files made by make_masks() are automatically set in RTP_setup()

    # Ready to run the pipeline
    proc_chain = rtp_app.ready_to_run()

    # --- Simulate scan (Feed data volume-by-volume) ----------------------
    # Load data
    img = nib.load(testdata_f)
    fmri_data = np.asanyarray(img.dataobj)
    N_vols = img.shape[-1]

    for ii in range(N_vols):
        name = f"sim_example_nr_{ii:04d}.nii.gz"
        fmri_img = nib.Nifti1Image(fmri_data[:, :, :, ii], affine=img.affine)
        fmri_img.set_filename(name)
        proc_chain.do_proc(fmri_img, ii, time.time())

    saved_fnames = rtp_app.end_run(scan_name='sim_test')

    # End
    print('-' * 80)
    print('End process')
    for k, pp in saved_fnames.items():
        print(f'{k} is saved in,\n  {pp}')
