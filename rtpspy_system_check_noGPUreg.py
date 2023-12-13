#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import time
import sys
import traceback

import numpy as np
import nibabel as nib
import torch

from rtpspy import RTP_APP
from rtpspy import RTP_PHYSIO_DUMMY


# %% main =====================================================================
if __name__ == '__main__':
    print("=" * 80)
    print("Test RTPSpy installation")
    sys.stdout.flush()

    try:
        # --- Filenames -------------------------------------------------------
        # test data directory
        test_dir = Path(__file__).absolute().parent / 'tests'

        # Set test data files
        testdata_f = test_dir / 'func_epi.nii.gz'
        anat_f = test_dir / 'anat_mprage.nii.gz'
        template_f = test_dir / 'MNI152_2009_template.nii.gz'
        ROI_template_f = test_dir / 'MNI152_2009_template_LAmy.nii.gz'
        WM_template_f = test_dir / 'MNI152_2009_template_WM.nii.gz'
        Vent_template_f = test_dir / 'MNI152_2009_template_Vent.nii.gz'
        ecg_f = test_dir / 'ECG.1D'
        resp_f = test_dir / 'Resp.1D'

        work_dir = test_dir / 'work'
        if not work_dir.is_dir():
            work_dir.mkdir()

        # Prepare watch dir
        watch_dir = test_dir / 'watch'
        if not watch_dir.is_dir():
            watch_dir.mkdir()
        else:
            # Clean up watch_dir
            for ff in watch_dir.glob('*'):
                if ff.is_dir():
                    for fff in ff.glob('*'):
                        fff.unlink()
                    ff.rmdir()
                else:
                    ff.unlink()

        # --- Create RTP_APP instance -----------------------------------------
        rtp_app = RTP_APP(work_dir=work_dir)

        # --- Make mask images ------------------------------------------------
        if torch.cuda.is_available():
            print('GPU is utilized.')
            no_FastSeg = False
            rtp_app.fastSeg_batch_size = 8  # Adjust the size according to GPU
        else:
            print('GPU is not avilable.')
            no_FastSeg = True

        rtp_app.make_masks(
            func_orig=str(testdata_f)+"'[0]'",  anat_orig=anat_f,
            template=template_f, ROI_template=ROI_template_f,
            no_FastSeg=no_FastSeg, WM_template=WM_template_f,
            Vent_template=Vent_template_f, overwrite=True)

        # --- Set up RTP ------------------------------------------------------
        # Set RTP_PHYSIO to RTP_PHYSIO_DUMMY
        sample_freq = 40
        rtp_app.rtp_objs['PHYSIO'] = RTP_PHYSIO_DUMMY(
            ecg_f, resp_f, sample_freq, rtp_app.rtp_objs['RETROTS'])

        # RTP parameters
        rtp_params = {'WATCH': {'watch_dir': watch_dir,
                                'watch_file_pattern': r'nr_\d+.*\.nii'},
                      'TSHIFT': {'slice_timing_from_sample': testdata_f,
                                 'method': 'cubic', 'ignore_init': 3},
                      'VOLREG': {'regmode': 'cubic'},
                      'SMOOTH': {'blur_fwhm': 6.0},
                      'REGRESS': {'mot_reg': 'mot12',
                                  'GS_reg': True, 'WM_reg': True,
                                  'Vent_reg': True, 'phys_reg': 'RICOR8',
                                  'onGPU': False, 'wait_num': 40}}

        # RTP setup
        rtp_app.RTP_setup(rtp_params=rtp_params)
        # Mask files made by make_masks() are automatically set in RTP_setup()

        # Ready to run the pipeline
        rtp_app.ready_to_run()

        # --- Simulate scan (Copy data volume-by-volume) ----------------------
        # Load data
        img = nib.load(testdata_f)
        fmri_data = np.asanyarray(img.dataobj)
        N_vols = img.shape[-1]

        # Start
        scan_onset_time = rtp_app.manual_start()

        next_tr = 2.0
        for ii in range(N_vols):
            next_tr = (ii+1)*2.0
            while time.time() - scan_onset_time < next_tr:
                # Wait ofr next TR
                time.sleep(0.001)

            # Copy volume file to watch_dir
            save_filename = watch_dir / f"system_test_nr_{ii:04d}.nii.gz"
            nib.save(
                nib.Nifti1Image(fmri_data[:, :, :, ii], affine=img.affine),
                save_filename)

        time.sleep(2.0)
        rtp_app.end_run()

        # End
        print('-' * 80)
        print('End process')
        print('RTPSpy system check finished successfully.')
        print(f'See the processed data in {work_dir}/RTP')

    except Exception:
        print('x' * 80)
        print('RTPSpy system check failed.')
        exc_type, exc_obj, exc_tb = sys.exc_info()
        exc_type, exc_obj, exc_tb = sys.exc_info()
        traceback.print_exception(exc_type, exc_obj, exc_tb)
