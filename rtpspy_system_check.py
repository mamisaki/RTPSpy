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
import argparse
import logging

import numpy as np
import nibabel as nib
import torch

from rtpspy import RtpApp


# %% main =====================================================================
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="rtp_system_check")
    parser.add_argument(
        "--log_file", help="Write log to specified file," + " instead of console."
    )
    parser.add_argument(
        "--keep_masks", action="store_true", help="Keep existing processd mask files"
    )
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    log_file = args.log_file
    overwrite = not args.keep_masks
    debug = args.debug

    if debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    # -------------------------------------------------------------------------
    print("=" * 80)
    print("Test RTPSpy installation")
    sys.stdout.flush()

    # Set logging
    fmt = "%(asctime)s.%(msecs)04d,[%(levelname)s],%(name)s,%(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"
    if log_file is None:
        logging.basicConfig(
            stream=sys.stdout, level=log_level, format=fmt, datefmt=datefmt
        )
    else:
        logging.basicConfig(
            filename=log_file, level=log_level, format=fmt, datefmt=datefmt
        )

    logger = logging.getLogger("rtpspy_system_check")

    try:
        # --- Filenames -------------------------------------------------------
        # test data directory
        test_dir = Path(__file__).absolute().parent / "tests"

        # Set test data files
        testdata_f = test_dir / "func_epi.nii.gz"
        anat_f = test_dir / "anat_mprage.nii.gz"
        template_f = test_dir / "MNI152_2009_template.nii.gz"
        ROI_template_f = test_dir / "MNI152_2009_template_LAmy.nii.gz"
        WM_template_f = test_dir / "MNI152_2009_template_WM.nii.gz"
        Vent_template_f = test_dir / "MNI152_2009_template_Vent.nii.gz"
        ecg_f = test_dir / "ECG.1D"
        resp_f = test_dir / "Resp.1D"

        work_dir = test_dir / "work"
        if not work_dir.is_dir():
            work_dir.mkdir()

        # Prepare watch dir
        watch_dir = test_dir / "watch"
        if not watch_dir.is_dir():
            watch_dir.mkdir()
        else:
            # Clean up watch_dir
            for ff in watch_dir.glob("*"):
                if ff.is_dir():
                    for fff in ff.glob("*"):
                        fff.unlink()
                    ff.rmdir()
                else:
                    ff.unlink()

        # --- Create RtpApp instance -----------------------------------------
        rtp_app = RtpApp(work_dir=work_dir)

        # --- Make mask images ------------------------------------------------
        logger.debug("### Start creating mask images ###")
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            logger.info("GPU is utilized.")
            no_FastSeg = False
            rtp_app.fastSeg_batch_size = 1  # Adjust the size according to GPU
        else:
            logger.info("GPU is not avilable.")
            no_FastSeg = True

        rtp_app.make_masks(
            func_orig=str(testdata_f) + "'[0]'",
            anat_orig=anat_f,
            template=template_f,
            ROI_template=ROI_template_f,
            no_FastSeg=no_FastSeg,
            WM_template=WM_template_f,
            Vent_template=Vent_template_f,
            overwrite=overwrite,
        )

        logger.debug("### End creating mask images ###")

        # --- Set up RTP ------------------------------------------------------
        logger.debug("### Start preparing the dummy physio recorder ###")
        # Set RtpTTLPhysio
        rtp_app.rtp_objs["TTLPHYSIO"].stop_recording()
        for k, val in {
            "sample_freq": 40, "sim_card_f": ecg_f, "sim_resp_f": resp_f
        }.items():
            rtp_app.rtp_objs["TTLPHYSIO"].set_param(k, val)
        rtp_app.rtp_objs["TTLPHYSIO"].set_device("Dummy")

        # Wait for the rtp_physio to start recording
        while not rtp_app.rtp_objs["TTLPHYSIO"].is_recording():
            time.sleep(0.1)

        logger.debug("### End preparing the dummy physio recorder ###")

        # RTP parameters
        logger.debug("### Start RTP setup ###")
        rtp_app.func_param_ref = Path(testdata_f)
        rtp_params = {
            "WATCH": {
                "watch_dir": watch_dir,
                "watch_file_pattern": r"nr_\d+.*\.nii",
                "file_type": "Nifti",
            },
            "VOLREG": {"regmode": "cubic"},
            "TSHIFT": {"method": "cubic", "ignore_init": 3},
            "SMOOTH": {"blur_fwhm": 6.0},
            "REGRESS": {
                "mot_reg": "mot12",
                "GS_reg": True,
                "WM_reg": True,
                "Vent_reg": True,
                "phys_reg": "RICOR8",
                "wait_num": 30,
            },
        }

        # RTP setup
        rtp_app.RTP_setup(rtp_params=rtp_params)
        # Mask files made by make_masks() are automatically set in RTP_setup()

        # Ready to run the pipeline
        proc_chain = rtp_app.ready_to_run()

        logger.debug("### End RTP setup ###")

        # --- Simulate scan (Copy data volume-by-volume) ----------------------
        logger.debug("### Start simulating real-time fMRI imaging ###")
        # Load data
        img = nib.load(testdata_f)
        TR = img.header.get_zooms()[3]
        fmri_data = np.asanyarray(img.dataobj)
        N_vols = img.shape[-1]

        if debug:
            rtp_app.rtp_objs["WATCH"].stop_watching()

        # Start
        rtp_app.manual_start()
        scan_onset_time = rtp_app.scan_onset

        for ii in range(N_vols):
            next_tr = (ii + 1) * TR
            while time.time() - scan_onset_time < next_tr:
                # Wait for next TR
                time.sleep(0.001)

            # Copy volume file to watch_dir
            save_filename = watch_dir / f"system_test_nr_{ii + 1:04d}.nii.gz"
            nib.save(
                nib.Nifti1Image(fmri_data[:, :, :, ii], affine=img.affine),
                save_filename,
            )

            if debug:
                proc_chain.do_proc(save_filename)

        logger.debug("### End simulating real-time fMRI imaging ###")

        logger.debug("### Close RTP ###")
        rtp_app.end_run()
        logger.debug("### Close dummy physio recorder ###")
        rtp_app.rtp_objs["TTLPHYSIO"].end()

        # End
        logger.info("-" * 80)
        logger.info("End process")
        logger.info("RTPSpy system check finished.")
        logger.info(f"See the processed data in {work_dir}/RTP")

    except Exception:
        logger.error("x" * 80)
        logger.error("RTPSpy system check failed.")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        exc_type, exc_obj, exc_tb = sys.exc_info()
        traceback.print_exception(exc_type, exc_obj, exc_tb)
