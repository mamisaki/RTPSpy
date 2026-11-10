#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTPSpy system check: run the full RTP pipeline on the test data in tests/,
simulating real-time fMRI by writing the volumes one by one at the TR.

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
    # region: Parse arguments -------------------------------------------------
    parser = argparse.ArgumentParser(description="rtp_system_check")
    parser.add_argument(
        "--log_file",
        help="Write log to specified file instead of console.",
    )
    parser.add_argument(
        "--keep_masks",
        action="store_true",
        help="Keep existing processed mask files",
    )
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    log_file = args.log_file
    overwrite = not args.keep_masks
    debug = args.debug

    log_level = logging.DEBUG if debug else logging.INFO
    # endregion: Parse arguments ----------------------------------------------

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
        # region: Set paths ---------------------------------------------------
        test_dir = Path(__file__).absolute().parent / "tests"

        # Test data files
        testdata_f = test_dir / "func_epi.nii.gz"
        anat_f = test_dir / "anat_mprage.nii.gz"
        template_f = test_dir / "MNI152_2009_template.nii.gz"
        ROI_template_f = test_dir / "MNI152_2009_template_LAmy.nii.gz"
        WM_template_f = test_dir / "MNI152_2009_template_WM.nii.gz"
        Vent_template_f = test_dir / "MNI152_2009_template_Vent.nii.gz"
        ecg_f = test_dir / "ECG.1D"
        resp_f = test_dir / "Resp.1D"
        for label, ff in (
            ("Functional test data file", testdata_f),
            ("Anatomical test data file", anat_f),
            ("Template file", template_f),
            ("ROI template file", ROI_template_f),
            ("WM template file", WM_template_f),
            ("Vent template file", Vent_template_f),
            ("ECG file", ecg_f),
            ("Resp file", resp_f),
        ):
            assert ff.is_file(), f"{label} not found: {ff}"

        work_dir = test_dir / "work"
        if not work_dir.is_dir():
            work_dir.mkdir()

        # Prepare an empty watch directory
        watch_dir = test_dir / "watch"
        if not watch_dir.is_dir():
            watch_dir.mkdir()
        else:
            for ff in watch_dir.glob("*"):
                if ff.is_dir():
                    for fff in ff.glob("*"):
                        fff.unlink()
                    ff.rmdir()
                else:
                    ff.unlink()
        # endregion: Set paths ------------------------------------------------

        rtp_app = RtpApp(work_dir=work_dir)

        # region: Make mask images --------------------------------------------
        logger.debug("### Start creating mask images ###")
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            logger.info("GPU is utilized.")
            no_FastSeg = False
            # Increase the batch size if the GPU memory allows
            rtp_app.fastSeg_batch_size = 1
        else:
            logger.info("GPU is not available.")
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
        # endregion: Make mask images -----------------------------------------

        # region: Set up RTP --------------------------------------------------
        logger.debug("### Start preparing the dummy physio recorder ###")
        ttl_physio = rtp_app.rtp_objs["TTLPHYSIO"]
        assert ttl_physio.available, "rt_physio process is not available."
        # Feed the dummy recorder of the rt_physio process with the test files
        ttl_physio.TTLPhysioCom.call_rt_proc(
            ("START_DUMMY_FEEDING_WITH_FILES", (str(ecg_f), str(resp_f), 40)),
            pkl=True,
        )

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

        # The mask files made by make_masks() are set in RTP_setup().
        rtp_app.RTP_setup(rtp_params=rtp_params)

        # Ready to run the pipeline
        proc_chain = rtp_app.ready_to_run()

        logger.debug("### End RTP setup ###")
        # endregion: Set up RTP -----------------------------------------------

        # region: Simulate scan (write the data volume by volume) -------------
        logger.debug("### Start simulating real-time fMRI imaging ###")
        img = nib.load(testdata_f)
        TR = img.header.get_zooms()[3]
        fmri_data = np.asanyarray(img.dataobj)
        N_vols = img.shape[-1]

        # In debug mode, the volumes are processed directly in this thread
        # (not by the watchdog) so that errors can be traced.
        if debug:
            rtp_app.rtp_objs["WATCH"].stop_watching()

        rtp_app.manual_start()
        scan_onset_time = rtp_app.scan_onset

        for ii in range(N_vols):
            next_tr = (ii + 1) * TR
            while time.time() - scan_onset_time < next_tr:
                time.sleep(0.001)  # Wait for the next TR

            # Write a volume file to the watch directory
            save_filename = watch_dir / f"system_test_nr_{ii + 1:04d}.nii.gz"
            nib.save(
                nib.Nifti1Image(fmri_data[:, :, :, ii], affine=img.affine),
                save_filename,
            )

            if debug:
                proc_chain.do_proc(save_filename)

        logger.debug("### End simulating real-time fMRI imaging ###")
        # endregion: Simulate scan --------------------------------------------

        logger.debug("### Close RTP ###")
        rtp_app.end_run()
        logger.debug("### Close dummy physio recorder ###")
        rtp_app.rtp_objs["TTLPHYSIO"].quit()

        logger.info("-" * 80)
        logger.info("End process")
        logger.info("RTPSpy system check finished.")
        logger.info(f"See the processed data in {work_dir}/RTP")

    except Exception:
        logger.error("x" * 80)
        logger.error("RTPSpy system check failed.")
        traceback.print_exc()
        # Stop the rt_physio process so that it does not remain orphaned
        if "rtp_app" in locals() and "TTLPHYSIO" in rtp_app.rtp_objs:
            rtp_app.rtp_objs["TTLPHYSIO"].quit()
        sys.exit(1)
