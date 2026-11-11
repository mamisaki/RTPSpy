#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTPSpy simulation: run the RTP pipeline offline on the test data, feeding
the volumes one by one.

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import time
import sys

import numpy as np
import nibabel as nib

from rtpspy import RtpApp
from rtpspy.rtp_sim import SavedPhysio


# %% main =====================================================================
if __name__ == '__main__':
    print("=" * 80)
    print("RTPSpy simulation")
    sys.stdout.flush()

    # --- Filenames -----------------------------------------------------------
    test_dir = Path(__file__).absolute().parent.parent.parent / 'tests'

    testdata_f = test_dir / 'func_epi.nii.gz'
    anat_f = test_dir / 'anat_mprage.nii.gz'
    template_f = test_dir / 'MNI152_2009_template.nii.gz'
    ROI_template_f = test_dir / 'MNI152_2009_template_LAmy.nii.gz'
    ecg_f = test_dir / 'ECG.1D'
    resp_f = test_dir / 'Resp.1D'

    work_dir = Path(__file__).absolute().parent / 'work'
    if not work_dir.is_dir():
        work_dir.mkdir()

    # --- Create an RtpApp instance -------------------------------------------
    rtp_app = RtpApp(work_dir=work_dir)

    # --- Make mask images ----------------------------------------------------
    # Adjust the batch size to the GPU memory
    rtp_app.fastSeg_batch_size = 2
    rtp_app.make_masks(func_orig=str(testdata_f)+"'[0]'", anat_orig=anat_f,
                       template=template_f, ROI_template=ROI_template_f,
                       overwrite=False)

    # --- Set up RTP ----------------------------------------------------------
    # Slice timing and TR are read from the fMRI file in RTP_setup().
    rtp_app.func_param_ref = str(testdata_f)

    # RTP parameters
    rtp_params = {'WATCH': {'enabled': False},
                  'TSHIFT': {'method': 'cubic', 'ignore_init': 3},
                  'VOLREG': {'regmode': 'cubic'},
                  'SMOOTH': {'blur_fwhm': 6.0},
                  'REGRESS': {'mot_reg': 'mot12',
                              'GS_reg': True, 'WM_reg': True,
                              'Vent_reg': True, 'phys_reg': 'RICOR8',
                              'wait_num': 40}}

    # The mask files made by make_masks() are set in RTP_setup().
    rtp_app.RTP_setup(rtp_params=rtp_params)

    # RETROICOR regressors from the saved physio signals (40 Hz)
    resp = np.loadtxt(resp_f)
    card = np.loadtxt(ecg_f)
    rtp_app.rtp_objs['REGRESS'].set_param(
        'rtp_physio', SavedPhysio(card, resp, sample_freq=40))

    proc_chain = rtp_app.ready_to_run()

    # --- Simulate scan (feed the data volume by volume) ----------------------
    img = nib.load(testdata_f)
    fmri_data = np.asanyarray(img.dataobj)
    N_vols = img.shape[-1]

    for ii in range(N_vols):
        name = f"sim_example_nr_{ii:04d}.nii.gz"
        fmri_img = nib.Nifti1Image(fmri_data[:, :, :, ii], affine=img.affine)
        fmri_img.set_filename(name)
        proc_chain.do_proc(fmri_img, ii, time.time())

    saved_fnames = rtp_app.end_run(scan_name='sim_test')

    # Stop the physio recording process (rt_physio) explicitly; at the
    # interpreter exit the QUIT message cannot be sent, and it is terminated
    # only after a timeout.
    rtp_app.rtp_objs['TTLPHYSIO'].quit()

    print('-' * 80)
    print('End process')
    for k, pp in saved_fnames.items():
        print(f'{k} is saved in\n  {pp}')
