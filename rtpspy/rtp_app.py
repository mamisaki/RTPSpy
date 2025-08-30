#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTP application

@author: mmisaki@laureateinstitute.org
"""

# %% import ===================================================================
from pathlib import Path
import os
import sys
import subprocess
import shlex
import time
import re
from datetime import datetime, timedelta
from functools import partial
import socket
import traceback
import signal
import shutil
import json
import logging

import numpy as np
import nibabel as nib
import pandas as pd
import torch

from PyQt5 import QtWidgets, QtCore
import matplotlib.pyplot as plt

try:
    # Load modules from the same directory
    from .rtp_common import (
        RTP,
        boot_afni,
        MatplotlibWindow,
        DlgProgressBar,
        excepthook,
        load_parameters,
        save_parameters,
    )
    from .rtp_watch import RtpWatch
    from .rtp_volreg import RtpVolreg
    from .rtp_tshift import RtpTshift
    from .rtp_smooth import RtpSmooth
    from .rtp_regress import RtpRegress
    from .rtp_imgproc import RtpImgProc
    from .rtp_serve import boot_RTP_SERVE_app, pack_data
    from .rtp_ttl_physio import RtpTTLPhysio

except Exception:
    # # For DEBUG environment
    # from rtpspy import (
    #     RtpWatch,
    #     RtpTshift,
    #     RtpVolreg,
    #     RtpSmooth,
    #     RtpRegress,
    #     RtpImgProc,
    #     RtpTTLPhysio
    # )
    # from rtpspy.rtp_common import (
    #     RTP,
    #     boot_afni,
    #     MatplotlibWindow,
    #     DlgProgressBar,
    #     excepthook,
    #     load_parameters,
    #     save_parameters,
    # )
    # from rtpspy.rtp_serve import boot_RTP_SERVE_app, pack_data
    pass


# %% RtpApp class ============================================================
class RtpApp(RTP):
    """RTP application class"""

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(
        self,
        default_rtp_params=None,
        work_dir="",
        physio_log_file=None,
        rt_physio_address_name=["localhost", None, "RtTTLPhysioSocketServer"],
        config_path=Path.home() / ".RTPSpy" / "rtmri_config.json",
        **kwargs
    ):
        """
        Parameters:
            default_rtp_params : dictionary, optional
                Parameter dictionary. Defaults to None.
            work_dir : Path or str
                Working directory.
        """
        super(RtpApp, self).__init__(**kwargs)

        self._logger.debug("### Initialize RtpApp ###")

        # region: Initialize parameters ---------------------------------------
        self.work_dir = work_dir

        # Template images
        self.template = ""
        self.ROI_template = ""
        self.WM_template = ""
        self.Vent_template = ""

        # Image files
        self.func_param_ref = ""  # fMRI parameter reference image
        self.func_orig = ""  # Reference function image
        self.anat_orig = ""  # Anatomy image
        self.alAnat = ""  # Anatomy image aligned to the reference function

        self.WM_orig = ""
        self.Vent_orig = ""
        self.ROI_orig = ""
        self.aseg_oirg = ""
        self.ROI_mask = ""

        self.RTP_mask = ""
        self.GSR_mask = ""

        # Fieldmap images for distortion correction (PEPOLAR)
        self.fieldmap_posi = ""  # Same phase encoding EPI
        self.fieldmap_nega = ""  # Opposite phase encoding EPI
        self.sdc_invwarp = ""  # Warp field for inverse distortion correction
        self.func_orig_sdc = ""  # Distortion-corrected functional image

        self.enable_RTP = 0

        self.AFNIRT_TRUSTHOST = None
        self._isRunning_end_run_proc = False
        self._isReadyRun = False

        # Mask creation parameters
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            self.no_FastSeg = False
            self.fastSeg_batch_size = 1
            # total_memory = torch.cuda.get_device_properties(0).total_memory
            # self.fastSeg_batch_size = int((total_memory-1e9) // (6 * 1e8))
        else:
            self.no_FastSeg = True
            self.fastSeg_batch_size = 1

        # Default processing times for proc_anat progress bar (seconds)
        self.proc_times = {
            "FastSeg": 100,
            "SkullStrip": 100,
            "SDC": 10,
            "AlAnat": 40,
            "RTP_GSR_mask": 3,
            "ANTs": 120,
            "Resample_WM_mask": 1,
            "Resample_Vent_mask": 1,
            "Resample_aseg_mask": 1,
        }
        self.prtime_keys = list(self.proc_times.keys())

        # Interpolation option for antsApplyTransforms when resampling the
        # warped ROI: ['linear'|'nearestNeighbor'|'bSpline']
        self.ROI_resample_opts = ["nearestNeighbor", "linear", "bSpline"]
        self.ROI_resample = "nearestNeighbor"

        # Scan onset
        self._wait_start = False
        self._scanning = False
        self.scan_onset = 0.0

        # Prepare the timer to check the running status
        self.chk_run_timer = QtCore.QTimer()
        self.chk_run_timer.setSingleShot(True)
        self.chk_run_timer.timeout.connect(self.chkRunTimerEvent)
        self.max_watch_wait = 5  # seconds

        # endregion

        # region: External application ----------------------------------------
        # Define an external application that receives the RTP signal
        self.run_extApp = False
        self.extApp_cmd = ""
        self.extApp_addr = None
        self.extApp_proc = None
        self.extApp_sock = None
        self.extApp_sock_timeout = 3

        self.sig_save_file = Path(self.work_dir) / "rtp_ROI_signal.csv"
        # endregion

        # region: Initialize signal plot --------------------------------------
        self.num_ROIs = 1
        self.roi_labels = ["ROI"]
        self._plt_xi = []

        self._roi_sig = []
        for ii in range(self.num_ROIs):
            self._roi_sig.append(list([]))
        # endregion

        # region: RTP module instances ----------------------------------------
        rtp_objs = dict()
        rtp_objs["WATCH"] = RtpWatch()
        rtp_objs["VOLREG"] = RtpVolreg()
        rtp_objs["TSHIFT"] = RtpTshift()
        rtp_objs["SMOOTH"] = RtpSmooth()
        rtp_objs["TTLPHYSIO"] = RtpTTLPhysio(
            physio_log_file=physio_log_file,
            rt_physio_address_name=rt_physio_address_name,
            config_path=config_path
        )
        rtp_objs["REGRESS"] = RtpRegress(
            volreg=rtp_objs["VOLREG"],
            rtp_physio=rtp_objs["TTLPHYSIO"]
        )

        self.rtp_objs = rtp_objs
        self.rtp_gui = None

        # endregion

        # region: Clean temporary DICOM files ---------------------------------
        tmp_dcm = list(Path("/tmp").glob("**/*.dcm"))
        for rmf in tmp_dcm:
            if rmf.is_file():
                shutil.rmtree(rmf.parent)
        # endregion

        # region: Set the default RTP parameters ------------------------------
        if default_rtp_params is not None:
            # Set default parameters
            for proc, params in default_rtp_params.items():
                if proc in self.rtp_objs:
                    for attr, val in params.items():
                        self.rtp_objs[proc].set_param(attr, val)

            if "APP" in default_rtp_params:
                for attr, val in default_rtp_params["APP"].items():
                    self.set_param(attr, val)

        self._logger.debug("### Complete RtpApp initialization ###")
        # endregion

    # --- Override these functions for a custom application -------------------
    # ready_proc, do_proc, end_reset, end_proc
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ready_proc(self):
        """Prepare the process."""

        self._proc_ready = True

        if not Path(self.ROI_orig).is_file():
            errmsg = f"ROI mask in original space not found: {self.ROI_orig}."
            self._logger.error(errmsg)
            self.err_popup(errmsg)
            self._proc_ready = False
        self.ROI_mask = None

        if self._proc_ready and self.ROI_mask is None:
            # Load ROI mask
            self.ROI_mask = np.asarray(nib.load(self.ROI_orig).dataobj)

        # Reset plot values
        self._plt_xi[:] = []
        for ii in range(self.num_ROIs):
            self._roi_sig[ii][:] = []

        if hasattr(self, "pltROISig"):
            self.pltROISig.reset_plot()

        return self._proc_ready

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def do_proc(self, fmri_img, vol_idx=None, pre_proc_time=0):
        """Process the RTP image (e.g., extracting the ROI signal)."""
        try:
            # Increment the number of received volumes
            self._vol_num += 1  # 1-based number of volumes received
            if vol_idx is None:
                vol_idx = self._vol_num - 1  # 0-based index

            if vol_idx < self.ignore_init:
                # Skip ignore_init volumes
                return

            if self._proc_start_idx < 0:
                self._proc_start_idx = vol_idx

            dataV = fmri_img.get_fdata()

            # --- Initialize --------------------------------------------------
            if Path(self.ROI_orig).is_file() and self.ROI_mask is None:
                # Load ROI mask
                self.ROI_mask = np.asarray(nib.load(self.ROI_orig).dataobj)

            # --- Run the process ---------------------------------------------
            # Get mean signal in the ROI
            roimask = (self.ROI_mask > 0) & (np.abs(dataV) > 0.0)
            mean_sig = np.nanmean(dataV[roimask])

            val_str = (
                f"{time.time() - self.scan_onset:.4f},{vol_idx},{mean_sig:.6f}"
            )
            if self.extApp_sock is not None:
                # Send data to the external application via socket
                try:
                    msg = f"NF {val_str};"
                    self.send_extApp(msg.encode())
                    self._logger.info(f"Sent '{msg}' to an external app")

                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    errmsg = "".join(
                        traceback.format_exception(exc_type, exc_obj, exc_tb)
                    )
                    self._logger.error(str(e) + "\n" + errmsg)

            else:
                # Save data to a file in real-time
                with open(self.sig_save_file, "a") as save_fd:
                    print(val_str, file=save_fd)
                self._logger.info(f"Saved ROI data '{val_str}'")

            # --- Post-process -----------------------------------------------
            tstamp = time.time()
            self._proc_time.append(tstamp)
            if pre_proc_time is not None:
                proc_delay = self._proc_time[-1] - pre_proc_time
                if self.save_delay:
                    self.proc_delay.append(proc_delay)

            # Log message
            f = Path(fmri_img.get_filename()).name
            msg = f"#{vol_idx + 1};ROI signal extraction;{f}"
            msg += f";tstamp={tstamp}"
            if pre_proc_time is not None:
                msg += f";took {proc_delay:.4f}s"
            self._logger.info(msg)

            # Update signal plot
            self._plt_xi.append(vol_idx + 1)
            self._roi_sig[0].append(mean_sig)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errmsg = "".join(
                traceback.format_exception(exc_type, exc_obj, exc_tb)
            )
            self._logger.error(str(e) + "\n" + errmsg)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end_reset(self):
        """End process and reset process parameters."""
        self._logger.info(f"Reset {self.__class__.__name__} module.")

        # Reset ROI_mask
        self.ROI_mask = None

        return super(RtpApp, self).end_reset()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end_proc(self):
        """Placeholder for a custom end process.
        This is called at the beginning of end_run().
        """
        pass

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run_dcm2nii(self):
        """Convert DICOM to NIfTI using dcm2niix."""
        # Select DICOM directory
        watch_dir = self.rtp_objs["WATCH"].watch_dir
        dcm_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self.main_win, "Select DICOM file directory", str(watch_dir)
        )

        if dcm_dir == "":
            return

        out_dir = Path(self.work_dir).absolute()
        if out_dir == "":
            out_dir = "./"

        cmd = "dcm2niix -f sub-%n_ses-%t_ser-%s_desc-%d"
        cmd += f" -i n -z o -w 0 -o {out_dir} {dcm_dir}"
        try:
            # Progress dialog
            msgBox = QtWidgets.QMessageBox(self.main_win)
            msgBox.setWindowTitle("dcm2niix")
            msgBox.setText("Converting DICOM to NIfTI ...")
            msgBox.setStandardButtons(QtWidgets.QMessageBox.NoButton)
            msgBox.show()
            time.sleep(0.1)
            msgBox.repaint()
            QtWidgets.QApplication.processEvents()

            proc = subprocess.Popen(
                shlex.split(cmd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            while proc.poll() is None:
                QtWidgets.QApplication.processEvents()
                time.sleep(0.5)

            stdout, stderr = proc.communicate()
            ostr = "\n".join([stdout.decode(), stderr.decode()])

            # Copy DICOMs
            src_dir = dcm_dir
            dst_dir = out_dir / "dicom"
            cmd = f"rsync -auvz {src_dir}/ {dst_dir}/"
            subprocess.check_call(shlex.split(cmd), stdout=subprocess.DEVNULL)

            # # sanitize filename
            # for fpath in out_dir.glob('*'):
            #     fname = fpath.name
            #     sanitized = re.sub(
            #         r'[\^|\s|<>:"/\\|?*\x00-\x1F]', '_', fname)
            #     if sanitized != fname:
            #         fpath.rename(out_dir / sanitized)

            msgBox.accept()
            QtWidgets.QMessageBox.information(
                self.main_win, "dcm2niix", ostr, QtWidgets.QMessageBox.Ok
            )

        except Exception as e:
            try:
                msgBox.accept()
            except Exception:
                pass
            errmsg = f"Error during dcm2niix: {e}"
            self._logger.error(errmsg)
            self.err_popup(errmsg)
            ostr = errmsg.encode()

        return

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def make_masks(
        self,
        func_orig=None,
        anat_orig=None,
        template=None,
        ROI_template=None,
        no_FastSeg=False,
        WM_template=None,
        Vent_template=None,
        ask_cmd=False,
        overwrite=False,
        progress_dlg=False,
    ):
        """
        Make mask images for RTP

        Parameters
        ----------
        func_orig : str or Path, optional
            Reference function image file. The default is None.
        anat_orig : str or Path, optional
            Anatomy image file. The default is None.
        template : str or Path, optional
            Template image file. The default is None.
        ROI_template : str or Path, optional
            ROI mask image on template. The default is None.
        no_FastSeg : bool, oprional
            Not using FastSeg. Use 3dSkullStrip and templete WM, Vent masks
            warped into a subject brain.
        ask_cmd : bool, optional
            Flag to ask editing the processing command line.
            The default is False.
        overwrite : boolr, optional
            Falg to overwrite existing files. The default is False.
        progress_dlg : DlgProgressBar object, optional
            Progress dialog. The default is False.

        """
        # Check work_dir
        if type(self.work_dir) is str and self.work_dir == "":
            errmsg = "Working directory is not set."
            self._logger.error(errmsg)
            self.err_popup(errmsg)
            return

        # region: Initialize --------------------------------------------------
        if func_orig is not None:
            self.set_param("func_orig", func_orig)
        else:
            func_orig = self.func_orig

        if anat_orig is not None:
            self.set_param("anat_orig", anat_orig)
        else:
            anat_orig = self.anat_orig

        if template is not None:
            self.set_param("template", template)
        else:
            template = self.template

        if ROI_template is not None:
            self.set_param("ROI_template", ROI_template)
        else:
            if self.ROI_template == "":
                ROI_template = None
            else:
                ROI_template = self.ROI_template

        if no_FastSeg:
            if WM_template is not None:
                self.set_param("WM_template", WM_template)
            else:
                WM_template = self.WM_template

            if Vent_template is not None:
                self.set_param("Vent_template", Vent_template)
            else:
                Vent_template = self.Vent_template

        # Check attributes
        for attr in (
            "func_orig",
            "anat_orig",
            "template",
            "ROI_template",
            "WM_template",
            "Vent_template",
        ):
            if not hasattr(self, attr) or getattr(self, attr) is None:
                if attr in ("func_orig", "anat_orig") or (
                    no_FastSeg and attr in ("WM_template", "Vent_template")
                ):
                    errmsg = f"\n{attr} is not set.\n"
                    self._logger.error(errmsg)
                    self.err_popup(errmsg)
                    return -1
                else:
                    continue

            fpath = getattr(self, attr)
            if fpath == "":
                continue
            fpath = Path(fpath)
            fpath = fpath.parent / re.sub(
                "'", "", re.sub(r"\'*\[\d+\]\'*$", "", fpath.name)
            )

            if not Path(fpath).is_file() and len(str(fpath)) > 0:
                if (
                    attr
                    in (
                        "template",
                        "ROI_template",
                        "WM_template",
                        "Vent_template",
                    )
                    and not no_FastSeg
                ):
                    self.set_param(attr, "")
                    continue
                else:
                    errmsg = f"\nNot found {attr}:{getattr(self, attr)}.\n"
                    self._logger.error(errmsg)
                    self.err_popup(errmsg)
                    return -1

            self.set_param(attr, Path(getattr(self, attr)))

        # Disable CreateMasks button in GUI
        if hasattr(self, "ui_CreateMasks_btn"):
            self.ui_CreateMasks_btn.setEnabled(False)

        if self.main_win is not None:
            # Check shift (overwrite) and ctrl (ask_cmd) key press
            modifiers = QtWidgets.QApplication.keyboardModifiers()
            if modifiers == QtCore.Qt.ShiftModifier:
                overwrite = True
            elif modifiers == QtCore.Qt.ControlModifier:
                ask_cmd = True
            elif modifiers == (
                QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier
            ):
                overwrite = True
                ask_cmd = True

        # Progress bar
        if progress_dlg:
            sys.stdout.flush()
            original_stdout = sys.stdout
            progress_bar = DlgProgressBar(
                title="Mask image creation",
                modal=False,
                parent=self.main_win,
                st_time=time.time(),
                win_size=(750, 320),
            )
            sys.stdout = progress_bar.ostream
            progress_bar.set_value(0)
        else:
            progress_bar = None

        # Initialize the processing time for progress information,
        # if it is not set.
        for k in self.prtime_keys:
            if k not in self.proc_times:
                self.proc_times[k] = 1

        if no_FastSeg:
            del self.proc_times["FastSeg"]
        else:
            del self.proc_times["SkullStrip"]

        # Make image processor object
        improc = RtpImgProc(main_win=self.main_win)
        improc.proc_times = self.proc_times

        # Set total_ETA for the progress report.
        total_ETA = np.sum(list(self.proc_times.values()))
        if not Path(self.template).is_file():
            total_ETA -= self.proc_times["ANTs"]

        # Make RTP work_dir
        work_dir = self.work_dir
        if work_dir.name != 'RTP':
            out_dir = work_dir / 'RTP'
            if not out_dir.is_dir():
                out_dir.mkdir(parents=True)
        else:
            out_dir = work_dir

        st0 = time.time()  # start time
        OK = True
        # endregion: Initialize

        try:
            # region: Check image space ---------------------------------------
            func_space = None
            try:
                func_space = (
                    subprocess.check_output(
                        shlex.split(f"3dinfo -space {func_orig}")
                    )
                    .decode()
                    .rstrip()
                )
            except Exception:
                pass

            # Deoblique self.anat_orig
            anat_orig = out_dir / Path(self.anat_orig).name
            improc.copy_deoblique(
                self.anat_orig, anat_orig, progress_bar=progress_bar,
                overwrite=overwrite
            )
            # endregion: Check image space

            # region: 0. Copy func_orig to out_dir as vr_base_* --------------
            if (
                Path(func_orig).parent != out_dir
                or not Path(func_orig).stem.startswith("vr_base_")
                or overwrite
            ):
                src_f_stem = Path(func_orig).with_suffix('').stem
                suffix = Path(func_orig).suffix
                ma = re.search(r"[\'*|\"*]\[(\d+)\][\'*|\"*]", suffix)
                vidx = None
                if ma is not None:
                    vidx = int(ma.groups()[0])
                else:
                    vidx = -1

                if not src_f_stem.startswith("vr_base_"):
                    src_save_fname = "vr_base_" + src_f_stem
                else:
                    src_save_fname = src_f_stem

                dst_f = out_dir / f"{src_save_fname}.nii.gz"
                if (
                    dst_f != Path(func_orig) and
                    (not dst_f.is_file() or overwrite)
                ):
                    src_img = improc.load_image(Path(func_orig), vidx=vidx)
                    if src_img is None:
                        if progress_dlg:
                            sys.stdout = original_stdout
                        return
                    print("#" * 80)
                    print(f"Copy {func_orig} to {dst_f}")
                    nib.save(src_img, dst_f)
                    if func_space is not None:
                        ostr = subprocess.check_output(
                            shlex.split(
                                f"3drefit -space {func_space} {dst_f}"
                            ),
                            stderr=subprocess.STDOUT,
                        )
                        print(ostr.decode())

                # json sidecar file
                dst_json_f = dst_f.with_suffix('.json')
                if not dst_json_f.is_file() or overwrite:
                    # If a json file exists, copy it
                    json_f = func_orig.with_suffix('').with_suffix('.json')
                    if not json_f.is_file():
                        json_f = (
                            json_f.parent /
                            json_f.name.replace("vr_base_", "")
                        )
                        if not json_f.is_file():
                            json_f = (
                                json_f.parent.parent /
                                json_f.name.replace("vr_base_", "")
                            )

                    if json_f.is_file() and json_f != dst_json_f:
                        shutil.copy(json_f, dst_json_f)
                    else:
                        # Save TR and slice timing
                        header = nib.load(func_orig).header

                        slice_timing = []
                        if hasattr(header, "get_slice_times"):
                            try:
                                slice_timing = header.get_slice_times()
                            except nib.spatialimages.HeaderDataError:
                                pass
                        elif (
                            hasattr(header, "info")
                            and "TAXIS_FLOATS" in header.info
                        ):
                            slice_timing = header.info["TAXIS_OFFSETS"]

                        if len(slice_timing):
                            tr = subprocess.check_output(
                                shlex.split(f"3dinfo -tr {func_orig}")
                            )
                            TR = float(tr.decode().rstrip())

                            img_info = {
                                "RepetitionTime": TR,
                                "SliceTiming": slice_timing,
                            }
                            with open(dst_json_f, "w") as fd:
                                json.dump(img_info, fd, indent=4)

                print("\n")
                self.set_param("func_orig", dst_f)
            # endregion: Copy func_orig as vr_base_*

            # region: 1. Brain segmentation -----------------------------------
            if not no_FastSeg:
                # FastSeg
                # Make Brain, WM, Vent segmentations
                improc.fastSeg_batch_size = self.fastSeg_batch_size

                seg_files = improc.run_fast_seg(
                    out_dir,
                    anat_orig,
                    total_ETA,
                    progress_bar=progress_bar,
                    overwrite=overwrite,
                )
                assert seg_files is not None

                # Use self.set_param() to update GUI fields
                self.set_param("brain_anat_orig", seg_files[0])
                WM_seg = seg_files[1]
                Vent_seg = seg_files[2]
                aseg_seg = seg_files[3]
            else:
                # 3dSkullStrip
                # Make Brain segmentations
                brain_anat_orig = improc.skullStrip(
                    out_dir,
                    anat_orig,
                    total_ETA,
                    progress_bar=progress_bar,
                    ask_cmd=ask_cmd,
                    overwrite=overwrite,
                )
                assert brain_anat_orig is not None, "skullStrip failed.\n"

                # Use self.set_param() to update GUI fields
                self.set_param("brain_anat_orig", brain_anat_orig)
            # endregion

            # region: 2. Suceptibility distortion correction ------------------
            if (
                self.fieldmap_posi and Path(self.fieldmap_posi).is_file() and
                self.fieldmap_nega and Path(self.fieldmap_nega).is_file()
            ):
                sdc_invwarp, func_orig_sdc = improc.run_pepolar_sdc(
                    out_dir,
                    self.fieldmap_posi,
                    self.fieldmap_nega,
                    self.func_orig,
                    total_ETA,
                    progress_bar=progress_bar,
                    overwrite=overwrite,
                )
                assert sdc_invwarp is not None, (
                    "Distortion correction failed.\n"
                )

                # Use self.set_param() to update GUI fields
                self.set_param("sdc_invwarp", sdc_invwarp)
                self.set_param("func_orig_sdc", func_orig_sdc)
                func_base = func_orig_sdc
            else:
                self.set_param("sdc_invwarp", "")
                self.set_param("func_orig_sdc", "")
                func_base = self.func_orig
            # endregion

            # region: 3. Align anatomy to function ----------------------------
            alAnat = improc.align_anat2epi(
                out_dir,
                self.brain_anat_orig,
                func_base,
                total_ETA,
                progress_bar=progress_bar,
                ask_cmd=ask_cmd,
                overwrite=overwrite,
            )
            assert alAnat is not None, "align_anat2epi failed.\n"

            # Use self.set_param() to update GUI fields
            self.set_param("alAnat", alAnat)

            alAnat_f_stem = self.alAnat.stem.replace(".nii", "")
            aff1D_f = out_dir / (alAnat_f_stem + "_mat.aff12.1D")
            assert aff1D_f.is_file()
            # endregion

            # region: 4. Make RTP and GSR masks -------------------------------
            mask_files = improc.make_RTP_GSR_masks(
                out_dir,
                func_base,
                total_ETA,
                ref_vi=0,
                alAnat=self.alAnat,
                progress_bar=progress_bar,
                ask_cmd=ask_cmd,
                overwrite=overwrite,
            )
            assert mask_files is not None

            if self.sdc_invwarp and Path(self.sdc_invwarp).is_file():
                # Apply inverse distortion correction to the masks
                mask_files = list(mask_files)
                for ii, mask_f in enumerate(mask_files):
                    mask_files[ii] = improc.ants_warp_resample(
                        out_dir,
                        mask_f,
                        self.func_orig,
                        [self.sdc_invwarp],
                        suffix="_invsdc",
                        interpolator="nearestNeighbor",
                        progress_bar=progress_bar,
                        overwrite=overwrite,
                    )
                    assert mask_files[ii] is not None

            self.set_param("RTP_mask", mask_files[0])
            self.set_param("GSR_mask", mask_files[1])
            # endregion

            # region: 5. Warp template ----------------------------------------
            if Path(self.template).is_file():
                if Path(self.ROI_template).is_file() or no_FastSeg:
                    warp_params = improc.warp_template(
                        out_dir,
                        self.alAnat,
                        self.template,
                        total_ETA,
                        progress_bar=progress_bar,
                        ask_cmd=ask_cmd,
                        overwrite=overwrite,
                    )
            else:
                warp_params = None
            # endregion

            # region: 6. Apply warp -------------------------------------------
            # ROI_template
            if warp_params is not None and Path(self.ROI_template).is_file():
                ROI_orig = improc.ants_warp_resample(
                    out_dir,
                    self.ROI_template,
                    self.alAnat,
                    warp_params,
                    res_master_f=self.func_orig,
                    interpolator=self.ROI_resample,
                    progress_bar=progress_bar,
                    overwrite=overwrite,
                )
                assert ROI_orig is not None
                self.set_param("ROI_orig", ROI_orig)

                if self.sdc_invwarp and Path(self.sdc_invwarp).is_file():
                    # Apply inverse distortion correction to the ROI
                    ROI_orig_invsdc = improc.ants_warp_resample(
                        out_dir,
                        ROI_orig,
                        self.func_orig,
                        [self.sdc_invwarp],
                        suffix="_invsdc",
                        interpolator="linear",
                        progress_bar=progress_bar,
                        overwrite=overwrite,
                    )
                    assert ROI_orig_invsdc is not None
                    self.set_param("ROI_orig", ROI_orig_invsdc)
            # endregion

            # region: 7. Make white matter and ventricle masks ----------------
            if warp_params is not None and Path(self.ROI_template).is_file():
                for segname in ("WM", "Vent", "aseg"):
                    if segname == "WM":
                        erode = 2
                        if no_FastSeg:
                            seg_anat_f = self.WM_template
                        else:
                            seg_anat_f = WM_seg
                        interpolator = "linear"
                    elif segname == "Vent":
                        erode = 1
                        if no_FastSeg:
                            seg_anat_f = self.Vent_template
                        else:
                            seg_anat_f = Vent_seg
                        interpolator = "linear"
                    elif segname == "aseg":
                        erode = 0
                        if no_FastSeg:
                            continue
                        else:
                            seg_anat_f = aseg_seg
                        interpolator = "nearestNeighbor"

                    assert seg_anat_f.is_file()

                    if no_FastSeg:
                        # warp template seg_anat_f
                        seg_anat_f = improc.ants_warp_resample(
                            out_dir,
                            seg_anat_f,
                            self.alAnat,
                            warp_params,
                            interpolator=interpolator,
                            progress_bar=progress_bar,
                            ask_cmd=ask_cmd,
                            overwrite=overwrite,
                        )
                        aff1D_f = None

                    seg_al_f = improc.resample_segmasks(
                        out_dir,
                        seg_anat_f,
                        segname,
                        erode,
                        self.func_orig,
                        total_ETA,
                        aff1D_f=aff1D_f,
                        progress_bar=progress_bar,
                        ask_cmd=ask_cmd,
                        overwrite=overwrite,
                    )

                    if self.sdc_invwarp and Path(self.sdc_invwarp).is_file():
                        # Apply inverse distortion correction to the ROI
                        seg_al_f = improc.ants_warp_resample(
                            out_dir,
                            seg_al_f,
                            self.func_orig,
                            [self.sdc_invwarp],
                            suffix="_invsdc",
                            interpolator="linear",
                            progress_bar=progress_bar,
                            overwrite=overwrite,
                        )
                        assert seg_al_f is not None

                    # Use self.set_param() to update GUI fields
                    self.set_param(f"{segname}_orig", seg_al_f)
            # endregion

        except Exception:
            OK = False
            exc_type, exc_obj, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_obj, exc_tb)
            if progress_bar is not None:
                progress_bar.set_msgTxt("Exit with error.")
                progress_bar.add_desc("!!! Exit with error.")
                progress_bar.setWindowTitle(progress_bar.title)
                progress_bar.btnCancel.setText("Close")

        # region: All done ----------------------------------------------------
        if progress_bar is not None:
            etstr = str(timedelta(seconds=time.time() - st0))
            etstr = ":".join(etstr.split(".")[0].split(":")[1:])
            if OK:
                progress_bar.set_msgTxt("Done.")
                progress_bar.add_desc(f"All done (took {etstr}).")
                progress_bar.set_value(100)
            else:
                progress_bar.set_msgTxt("Exit with error.")
                progress_bar.add_desc(f"!!! Exit with error (took {etstr}).")

            progress_bar.setWindowTitle(progress_bar.title)
            progress_bar.btnCancel.setText("Close")
            sys.stdout = original_stdout

        if OK:
            self.proc_times = improc.proc_times

            # Message dialog
            if self.main_win is not None:
                QtWidgets.QMessageBox.information(
                    self.main_win,
                    "Mask creation is complete.",
                    "Mask creation is complete.",
                )

        # Enable CreateMasks button
        if hasattr(self, "ui_CreateMasks_btn"):
            self.ui_CreateMasks_btn.setEnabled(True)
        # endregion

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def RTP_setup(self, *args, rtp_params={}, ignore_error=False):
        """
        Setup and enable RTP modules.

        Parameters
        ----------
        rtp_params : dictionary, optional
            Dictionary of RTP parameters dictionary. The default is {}.
        ignore_error : bool, optional
            Ignore error to allow incomplete setup. The default is False.

        Returns
        -------
        int
             0 : normal
            -1 : error

        """
        if "enable_RTP" in rtp_params:
            self.set_param("enable_RTP", bool(rtp_params["enable_RTP"]) * 2)
        else:
            self.set_param("enable_RTP", 2)

        show_progress = (self.main_win is not None) and (self.enable_RTP > 0)
        if show_progress:
            # progress bar
            progress_bar = DlgProgressBar(
                title="Session setup ...", parent=self.main_win
            )
            progress = 0
            time.sleep(0.01)  # Give UI time to update the dialog

        # --- Set RTP parameters ----------------------------------------------
        if self.enable_RTP > 0:
            # Set parameters in rtp_params
            for proc, params in rtp_params.items():
                if proc in self.rtp_objs:
                    for attr, val in params.items():
                        self.rtp_objs[proc].set_param(attr, val)

            for proc in ["WATCH", "VOLREG", "TSHIFT", "SMOOTH", "REGRESS"]:
                if show_progress:
                    progress += 100 // 5
                    progress_bar.set_value(progress)
                    progress_bar.add_desc("Set {} parameters\n".format(proc))
                    time.sleep(0.01)  # Give time for UI to update the dialog

                if proc not in self.rtp_objs:
                    continue

                pobj = self.rtp_objs[proc]
                if pobj.enabled:
                    if hasattr(pobj, "work_dir"):
                        setattr(pobj, "work_dir", Path(self.work_dir))
                        if not Path(pobj.work_dir).is_dir():
                            Path(pobj.work_dir).mkdir()

                    if proc == "WATCH":
                        if not Path(pobj.watch_dir).is_dir():
                            errmsg = "'Watching directory' must be set."
                            self._logger.error(errmsg)
                            self.err_popup(errmsg)
                            if not ignore_error:
                                if show_progress:
                                    progress_bar.close()
                                    return -1

                        if not Path(self.func_orig).is_file():
                            errmsg = (
                                "Not found 'Base function image'"
                                + f" {self.func_orig}."
                            )
                            self._logger.error(errmsg)
                            self.err_popup(errmsg)
                            if show_progress and progress_bar.isVisible():
                                progress_bar.close()
                                return -1

                    elif proc == "VOLREG":
                        if Path(self.func_orig).is_file():
                            pobj.set_param("ref_vol", self.func_orig)
                        else:
                            ret = pobj.set_param("ref_vol", "external")
                            if ret is not None and ret == -1:
                                # Canceled
                                if show_progress and progress_bar.isVisible():
                                    progress_bar.close()
                                return -1

                    elif proc == "TSHIFT":
                        if not Path(self.func_param_ref).is_file():
                            if not ignore_error:
                                errmsg = (
                                    "Not found 'fMRI parameter reference'"
                                    + f" {self.func_param_ref}."
                                )
                                self._logger.error(errmsg)
                                self.err_popup(errmsg)
                                if show_progress and progress_bar.isVisible():
                                    progress_bar.close()
                                    return -1
                        else:
                            # Get parameters from a paremeter reference image
                            self.rtp_objs["TSHIFT"].set_from_sample(
                                self.func_param_ref
                            )

                    elif proc == "SMOOTH":
                        if Path(self.RTP_mask).is_file():
                            pobj.set_param("mask_file", self.RTP_mask)
                        else:
                            if not ignore_error:
                                ret = pobj.set_param("mask_file", "external")
                                if ret is not None and ret == -1:
                                    # Canceled
                                    if (
                                        show_progress
                                        and progress_bar.isVisible()
                                    ):
                                        progress_bar.close()
                                    return -1

                    elif proc == "REGRESS":
                        pobj.set_param("TR", self.rtp_objs["TSHIFT"].TR)
                        pobj.set_param(
                            "tshift", self.rtp_objs["TSHIFT"].ref_time
                        )
                        pobj.set_param("mask_file", self.RTP_mask)
                        if pobj.mot_reg != "None":
                            if not self.rtp_objs["VOLREG"].enabled:
                                pobj.ui_motReg_cmbBx.setCurrentText("None")
                                errmsg = "VOLREG is not enabled."
                                errmsg += "Motion regressor is set to None."
                                self.rtp_objs["VOLREG"].errmsg(errmsg)
                            else:
                                pobj.set_param(
                                    "volreg", self.rtp_objs["VOLREG"]
                                )

                        if pobj.phys_reg != "None":
                            pobj.set_param(
                                "rtp_physio", self.rtp_objs["TTLPHYSIO"]
                            )

                        if pobj.GS_reg:
                            if Path(self.GSR_mask).is_file():
                                pobj.set_param("GS_mask", self.GSR_mask)
                            else:
                                if not ignore_error and hasattr(
                                    pobj, "ui_GS_mask_lnEd"
                                ):
                                    ret = pobj.set_param(
                                        "GS_mask",
                                        self.work_dir,
                                        pobj.ui_GS_mask_lnEd.setText,
                                    )
                                    if ret == -1:
                                        if (
                                            show_progress
                                            and progress_bar.isVisible()
                                        ):
                                            progress_bar.close()
                                        return -1

                        if pobj.WM_reg:
                            if Path(self.WM_orig).is_file():
                                pobj.set_param("WM_mask", self.WM_orig)
                            else:
                                if not ignore_error and hasattr(
                                    pobj, "ui_WM_mask_lnEd"
                                ):
                                    ret = pobj.set_param(
                                        "WM_mask",
                                        self.work_dir,
                                        pobj.ui_WM_mask_lnEd.setText,
                                    )
                                    if ret == -1:
                                        if (
                                            show_progress
                                            and progress_bar.isVisible()
                                        ):
                                            progress_bar.close()
                                        return -1

                        if pobj.Vent_reg:
                            if Path(self.Vent_orig).is_file():
                                pobj.set_param("Vent_mask", self.Vent_orig)
                            else:
                                if not ignore_error and hasattr(
                                    pobj, "ui_Vent_mask_lnEd"
                                ):
                                    ret = pobj.set_param(
                                        "Vent_mask",
                                        self.work_dir,
                                        pobj.ui_Vent_mask_lnEd.setText,
                                    )
                                    if ret == -1:
                                        if (
                                            show_progress
                                            and progress_bar.isVisible()
                                        ):
                                            progress_bar.close()
                                        return -1

                        if pobj.reg_retro_proc:
                            self.max_watch_wait = np.max(
                                [self.max_watch_wait, pobj.wait_num / 2]
                            )

                if show_progress and not progress_bar.isVisible():
                    self._logger.info("Cancel experiment setup")
                    return -1

        # --- End -------------------------------------------------------------
        if show_progress:
            progress = 100
            progress_bar.set_value(progress)
            progress_bar.add_desc("Done")
            time.sleep(0.1)  # Give time for UI to update the dialog

        if self.main_win is not None:
            # Set ready button
            self.ui_ready_btn.setEnabled(True)
            self.ui_ready_btn.setText("Ready")
            self.ui_quit_btn.setEnabled(True)

            if show_progress:
                progress_bar.close()

            self.main_win.show_options_list()
            self.main_win.options_tab.setCurrentIndex(2)

        return 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _is_running_dcm2nii(self):
        # Check if dcm2niix is in progress
        try:
            ostr = subprocess.check_output(shlex.split("pgrep -f dcm2niix"))
            if len(ostr.decode().rstrip()):
                errmsg = "DICOM to NIfTI conversion is still in progress."
                self._logger.error(errmsg)
                self.err_popup(errmsg)
                return True
        except Exception:
            pass

        return False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ready_to_run(self):
        """Ready to run the process"""
        self._logger.debug("ready_to_run")

        #  --- Disable ui to block parameters change --------------------------
        if self.enable_RTP > 0 and self.main_win is not None:
            self.ui_setEnabled(False)

        # --- Ready RTP -------------------------------------------------------
        proc_chain = None
        if self.enable_RTP > 0:
            # connect RTP modules
            if self.rtp_objs["WATCH"].enabled:
                proc_chain = self.rtp_objs["WATCH"]
                last_proc = proc_chain
            else:
                proc_chain = None

            for proc in ["VOLREG", "TSHIFT", "SMOOTH", "REGRESS"]:
                if proc not in self.rtp_objs:
                    continue

                pobj = self.rtp_objs[proc]
                if pobj.enabled:
                    if proc_chain is None:
                        proc_chain = pobj
                        last_proc = proc_chain
                    else:
                        last_proc.next_proc = pobj

                    if proc == "TSHIFT":
                        self.max_watch_wait = max(
                            pobj.TR * 3, self.max_watch_wait
                        )

                    elif proc == "REGRESS":
                        if pobj.GS_reg or pobj.WM_reg or pobj.Vent_reg:
                            if self.rtp_objs["VOLREG"].enabled:
                                pobj.set_param(
                                    "mask_src_proc", self.rtp_objs["VOLREG"]
                                )
                            elif self.rtp_objs["TSHIFT"].enabled:
                                pobj.set_param(
                                    "mask_src_proc", self.rtp_objs["TSHIFT"]
                                )
                            else:
                                pobj.set_param(
                                    "mask_src_proc", self.rtp_objs["WATCH"]
                                )

                        self.max_watch_wait = max(
                            pobj.wait_num * 0.5, self.max_watch_wait
                        )

                    last_proc = pobj

            last_proc.save_proc = True
            last_proc.next_proc = self  # self (RtpApp) is the last process

            # Show plot windows
            if self.main_win is not None:
                if self.rtp_objs["VOLREG"].enabled:
                    self.main_win.chbShowMotion.setCheckState(2)

                if (
                    self.rtp_objs["REGRESS"].enabled
                    and self.rtp_objs["REGRESS"].phys_reg != "None"
                ):
                    # Start physio recording
                    self.main_win.chbShowPhysio.setCheckState(2)

            # Reset process chain status
            proc_chain.end_reset()

            # Ready process sequence: proc_ready calls its child's proc_ready
            if not proc_chain.ready_proc():
                return

            # Logging process parameters
            log_str = "RTP parameters:\n"
            rtp = proc_chain
            while rtp is not None:
                log_str += f"# {type(rtp).__name__}\n"
                for k, v in rtp.get_params().items():
                    if k == "ignore_init" and v == 0:
                        continue
                    log_str += f"#     {k}: {v}\n"

                rtp = rtp.next_proc

            log_str = log_str.rstrip()
            self._logger.info(log_str)

        # --- Ready application -----------------------------------------------
        if self.run_extApp:
            # Ready external application
            if not self.isAlive_extApp():
                self.boot_extApp()
                if not self.isAlive_extApp():
                    errmsg = "Cannot get response from extApp."
                    self._logger.error(errmsg)
                    self.err_popup(errmsg)
                    return

        else:
            if not self.sig_save_file.parent.is_dir():
                os.makedirs(self.sig_save_file.parent)
            open(self.sig_save_file, "w").write("Time,Index,Value\n")

        # Start running-status checking timer
        if self.main_win is not None:
            # Timers and QThread can only be used with QApplication
            self.chk_run_timer.start(1000)

            # Stand by scan onset monitor
            if (
                "TTLPHYSIO" in self.rtp_objs
                and self.rtp_objs["TTLPHYSIO"] is not None
            ):
                self.rtp_objs["TTLPHYSIO"].standby_scan()

            self._scanning = False
            self._wait_start = True

            # Run wait_onset thread
            self.th_wait_onset = QtCore.QThread()
            self.wait_onset = RtpApp.WAIT_ONSET(
                self, self.rtp_objs["TTLPHYSIO"], self.extApp_sock
            )
            self.wait_onset.moveToThread(self.th_wait_onset)
            self.th_wait_onset.started.connect(self.wait_onset.run)
            self.wait_onset.finished.connect(self.th_wait_onset.quit)
            self.th_wait_onset.start()

            # Change button text
            self.ui_ready_btn.setText("Waiting for scan start ...")
            self.ui_ready_btn.setEnabled(False)
            self.ui_manStart_btn.setEnabled(True)
            self.ui_quit_btn.setEnabled(True)

        self._isReadyRun = True
        return proc_chain

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def manual_start(self):
        """Manual scan start."""
        self.scan_onset = time.time()
        self._scanning = True
        self._wait_start = False

        if (
            "TTLPHYSIO" not in self.rtp_objs or
            self.rtp_objs["TTLPHYSIO"] is None
        ):
            self.rtp_objs["TTLPHYSIO"].scan_onset = self.scan_onset
            self.rtp_objs["TTLPHYSIO"].release_standby_scan()

        if self.extApp_sock is not None:
            # Send message to self.extApp_sock
            onset_str = datetime.fromtimestamp(self.scan_onset).isoformat()
            msg = f"SCAN_START {onset_str};"
            try:
                self.extApp_sock.send(msg.encode())
            except BrokenPipeError:
                pass

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end_run(self, quit_btn=False, scan_name=None):
        """End the RTP run and save results."""
        if self._isRunning_end_run_proc:
            """
            Check if end_run is running.
            end_run could be called simultaneously by the timer thread's'
            chkRunTimerEvent function,
            """
            return

        self._isRunning_end_run_proc = True

        self.chk_run_timer.stop()

        if quit_btn:
            self._logger.info("Quit button is pressed.")

        if self.main_win is not None:
            # Disable buttons
            self.ui_ready_btn.setText("Ready")
            self.ui_ready_btn.setEnabled(False)
            self.ui_quit_btn.setEnabled(False)
            self.ui_manStart_btn.setEnabled(False)
            self.main_win.repaint()
            QtWidgets.QApplication.instance().processEvents()

        try:
            save_fnames = {}

            # Run custom end process
            self.end_proc()

            # Abort WAIT_ONSET thread if it is running
            if (
                hasattr(self, "th_wait_onset")
                and self.th_wait_onset.isRunning()
            ):
                self.wait_onset.abort = True
                if not self.th_wait_onset.wait(1):
                    self.wait_onset.finished.emit()
                    self.th_wait_onset.wait()

            # Send 'END' message to an external application
            if self.isAlive_extApp():
                if self.send_extApp("END;".encode()):
                    recv = self.recv_extApp(timeout=3)
                    if recv is not None:
                        self._logger.info(f"Recv {recv.decode()}")

            # Save parameter list
            if self.enable_RTP > 0:
                if scan_name is None or scan_name is False:
                    if (
                        self.rtp_objs["WATCH"].enabled
                        and self.rtp_objs["WATCH"].scan_name is not None
                    ):
                        scan_name = self.rtp_objs["WATCH"].scan_name
                    else:
                        scan_name = f"scan_{time.strftime('%Y%m%d%H%M')}"

                # Get parameters
                all_params = {}
                for rtp in ("WATCH", "VOLREG", "TSHIFT", "SMOOTH", "REGRESS"):
                    if (
                        rtp not in self.rtp_objs
                        or not self.rtp_objs[rtp].enabled
                    ):
                        continue

                    if not self.rtp_objs[rtp].enabled:
                        continue

                    all_params[rtp] = self.rtp_objs[rtp].get_params()

                all_params[self.__class__.__name__] = self.get_params()

                save_dir = self.work_dir / "log"
                if not save_dir.is_dir():
                    save_dir.mkdir()

                save_f = save_dir / f"rtp_params_{scan_name}.txt"
                with open(save_f, "w") as fd:
                    for rtp, opt_dict in all_params.items():
                        fd.write("# {}\n".format(rtp))
                        for k in sorted(opt_dict.keys()):
                            val = opt_dict[k]
                            fd.write("{}: {}\n".format(k, val))
                save_fnames["RTP parameters"] = save_f

                # Save ROI signals
                if len(self._roi_sig) > 0:
                    roi_save_f = save_dir / f"ROI_sig_{scan_name}.csv"
                    self.save_ROI_sig(
                        roi_save_f,
                        self._plt_xi,
                        self._roi_sig,
                        self.roi_labels
                    )
                    save_fnames["ROI signals"] = roi_save_f

                # Get the root and last processes
                root_proc = None
                for rtp, obj in self.rtp_objs.items():
                    if rtp in ("TTLPHYSIO"):
                        continue
                    if obj.enabled:
                        if root_proc is None:
                            root_proc = obj

                # Reset all process chain
                out_files = root_proc.end_reset()
                if out_files is not None:
                    save_fnames.update(out_files)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errmsg = "".join(
                traceback.format_exception(exc_type, exc_obj, exc_tb)
            )
            self._logger.error(str(e) + "\n" + errmsg)

        if self.main_win is not None:
            # Enable ui
            self.ui_setEnabled(True)
            self.main_win.options_tab.setCurrentIndex(0)

        if self.rtp_objs["TTLPHYSIO"].available:
            self.rtp_objs["TTLPHYSIO"].release_standby_scan()
            self.rtp_objs["TTLPHYSIO"].end_scan()

        self._isReadyRun = False
        self._isRunning_end_run_proc = False
        self._wait_start = False
        self._scanning = False

        return save_fnames

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def check_onAFNI(self, base=None, ovl=None):
        """Check and display images in AFNI."""

        # Set underlay and overlay image file
        base_img = None
        if base == "anat":
            base_img = self.alAnat
        elif base == "func":
            base_img = self.func_orig

        ovl_img = None
        if ovl == "func":
            if base == "anat":
                if self.sdc_invwarp and Path(self.sdc_invwarp).is_file():
                    ovl_img = self.func_orig_sdc
                else:
                    ovl_img = self.func_orig
            else:
                ovl_img = self.func_orig

        elif ovl == "wm":
            if base == "anat":
                if self.sdc_invwarp and Path(self.sdc_invwarp).is_file():
                    ovl_img = (
                        Path(self.WM_orig).parent /
                        Path(self.WM_orig).name.replace("_invsdc", "")
                    )
                else:
                    ovl_img = self.WM_orig
            elif base == "func":
                ovl_img = self.WM_orig

        elif ovl == "vent":
            if base == "anat":
                if self.sdc_invwarp and Path(self.sdc_invwarp).is_file():
                    ovl_img = (
                        Path(self.Vent_orig).parent /
                        Path(self.Vent_orig).name.replace("_invsdc", "")
                    )
                else:
                    ovl_img = self.Vent_orig
            elif base == "func":
                ovl_img = self.Vent_orig

        elif ovl == "roi":
            if base == "anat":
                if self.sdc_invwarp and Path(self.sdc_invwarp).is_file():
                    ovl_img = (
                        Path(self.ROI_orig).parent /
                        Path(self.ROI_orig).name.replace("_invsdc", "")
                    )
                else:
                    ovl_img = self.ROI_orig
            elif base == "func":
                ovl_img = self.ROI_orig

        elif ovl == "RTPmask":
            if base == "anat":
                if self.sdc_invwarp and Path(self.sdc_invwarp).is_file():
                    ovl_img = (
                        Path(self.RTP_mask).parent /
                        Path(self.RTP_mask).name.replace("_invsdc", "")
                    )
                else:
                    ovl_img = self.RTP_mask
            elif base == "func":
                ovl_img = self.RTP_mask

        elif ovl == "GSRmask":
            if base == "anat":
                if self.sdc_invwarp and Path(self.sdc_invwarp).is_file():
                    ovl_img = (
                        Path(self.GSR_mask).parent /
                        Path(self.GSR_mask).name.replace("_invsdc", "")
                    )
                else:
                    ovl_img = self.GSR_mask
            elif base == "func":
                ovl_img = self.GSR_mask

        if base_img is None or not Path(base_img).is_file():
            errmsg = f"Base image {base_img.name} is not set or found."
            self.err_popup(errmsg)
            return
        ses_dir = base_img.parent

        if ovl_img is None or not Path(ovl_img).is_file():
            errmsg = f"Base image {ovl_img.name} is not set or found."
            self.err_popup(errmsg)
            return

        # Check if afni is ready
        cmd0 = "afni"
        pret = subprocess.run(
            shlex.split(f"pgrep -af '{cmd0}'"), stdout=subprocess.PIPE
        )
        procs = pret.stdout
        procs = [
            ll
            for ll in procs.decode().rstrip().split("\n")
            if "pgrep -af 'afni" not in ll
            and "RTafni" not in ll
            and len(ll) > 0
        ]
        if len(procs) == 0:
            # Boot AFNI
            boot_afni(
                main_win=self.main_win,
                boot_dir=ses_dir,
                TRUSTHOST=self.AFNIRT_TRUSTHOST,
            )

        if base_img is not None:
            # Get volume shape to adjust window size
            baseWinSize = 480  # height of axial image window
            bimg = nib.load(base_img)
            vsize = np.diag(bimg.affine)[:3]
            if np.any(vsize == 0):
                vsize = np.abs(
                    [r[np.argmax(np.abs(r))] for r in bimg.affine[:3, :3]]
                )
            vshape = bimg.shape[:3] * vsize
            wh_ax = np.abs(
                [int(baseWinSize * vshape[0] // vshape[1]), baseWinSize]
            )
            wh_sg = np.abs(
                [baseWinSize, int(baseWinSize * vshape[2] // vshape[1])]
            )
            wh_cr = np.abs(
                [
                    int(baseWinSize * vshape[0] // vshape[1]),
                    int(baseWinSize * vshape[2] // vshape[1]),
                ]
            )

            # Get cursor position
            if ovl in ["roi", "wm", "vent", "mask"]:
                ovl_v = nib.load(ovl_img).get_fdata()
                if ovl_v.ndim > 3:
                    ovl_v = ovl_v[:, :, :, 0]
                ijk = np.mean(np.argwhere(ovl_v != 0), axis=0)
                ijk = np.concatenate([ijk, [1]])
                xyz = np.dot(nib.load(ovl_img).affine, ijk)[:3]

            # Run plugout_drive to drive afni
            le = 800  # left end
            tp = 500  # top
            cmd = "plugout_drive"
            cmd += f" -com 'SWITCH_SESSION {Path(ses_dir).name}'"
            cmd += " -com 'RESCAN_THIS'"
            cmd += f" -com 'SWITCH_UNDERLAY {Path(base_img).name}'"
            if ovl_img is not None:
                cmd += f" -com 'SWITCH_OVERLAY {Path(ovl_img).name}'"
                cmd += " -com 'SEE_OVERLAY +'"
            cmd += " -com 'OPEN_WINDOW A.axialimage"
            cmd += f" geom={wh_ax[0]}x{wh_ax[1]}+{le}+{tp} opacity=6'"
            cmd += " -com 'OPEN_WINDOW A.sagittalimage"
            if le + wh_ax[0] + wh_sg[0] + wh_cr[0] - 100 > 1920:
                cmd += (
                    f" geom={wh_sg[0]}x{wh_sg[1]}+"
                    f"{le + wh_ax[0] - 50}+{tp - 100}"
                )
                cmd += " opacity=6'"
                cmd += " -com 'OPEN_WINDOW A.coronalimage"
                cmd += f" geom={wh_cr[0]}x{wh_cr[1]}+{le + wh_ax[0] - 50}"
                cmd += f"+{wh_sg[1] + tp - 100} opacity=6'"
            else:
                cmd += f" geom={wh_sg[0]}x{wh_sg[1]}+{le + wh_ax[0] - 50}+{tp}"
                cmd += " opacity=6'"
                cmd += " -com 'OPEN_WINDOW A.coronalimage"
                cmd += f" geom={wh_cr[0]}x{wh_cr[1]}"
                cmd += f"+{le + wh_ax[0] + wh_sg[0] - 100}"
                cmd += f"+{tp} opacity=6'"

            if ovl in ["roi", "wm", "vent", "mask"]:
                cmd += f" -com 'SET_SPM_XYZ {xyz[0]} {xyz[1]} {xyz[2]}'"
            cmd += " -quit"
            subprocess.run(cmd, shell=True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def delete_file(self, attr, keepfile=False):
        """Delete specified file or all processed files."""
        if not hasattr(self, attr) and attr != "AllProc":
            return

        if attr != "AllProc":
            ff = Path(getattr(self, attr))
            if not ff.is_file():
                self.set_param(attr, "")
                return

        if not keepfile:
            # Warning dialog
            msgBox = QtWidgets.QMessageBox()
            msgBox.setIcon(QtWidgets.QMessageBox.Question)
            if attr != "AllProc":
                msgBox.setText(f"Are you sure to delete {ff}?")
                msgBox.setWindowTitle(f"Delete {attr} file")
            else:
                msgBox.setText("Are you sure to delete ALL processed files?")
                msgBox.setWindowTitle("Delete all processed files")

            msgBox.setStandardButtons(
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            msgBox.setDefaultButton(QtWidgets.QMessageBox.No)
            ret = msgBox.exec()
            if ret != QtWidgets.QMessageBox.Yes:
                return

        if attr != "AllProc":
            delFiles = {attr: ff}
        else:
            delFiles = {}

            # Set all processed images to delete
            for rmattr in (
                "alAnat",
                "WM_orig",
                "Vent_orig",
                "ROI_orig",
                "RTP_mask",
                "GSR_mask",
                "brain_anat_orig",
            ):
                if (
                    hasattr(self, rmattr)
                    and Path(getattr(self, rmattr)).is_file()
                ):
                    delFiles[rmattr] = Path(getattr(self, rmattr))

            # Delete interim files
            anat_prefix = self.anat_orig.name.replace(
                "".join(self.anat_orig.suffixes[-2:]), ""
            )
            anat_prefix = anat_prefix.replace("+orig", "").replace("+tlrc", "")
            if not hasattr(self, "brain_anat_orig") and hasattr(
                self, "anat_orig"
            ):
                brain_anat_orig = self.work_dir / (
                    anat_prefix + "_Brain.nii.gz"
                )
                if brain_anat_orig.is_file():
                    self.brain_anat_orig = brain_anat_orig
                    delFiles["brain_anat_orig"] = brain_anat_orig

            if hasattr(self, "brain_anat_orig"):
                WM_anat = (
                    self.brain_anat_orig.parent
                    / self.brain_anat_orig.name.replace("Brain", "WM")
                )
                if WM_anat.is_file():
                    delFiles["WM_anat"] = WM_anat

                Vent_anat = (
                    self.brain_anat_orig.parent
                    / self.brain_anat_orig.name.replace("Brain", "Vent")
                )
                if Vent_anat.is_file():
                    delFiles["Vent_anat"] = Vent_anat

                del_tmep = anat_prefix + "_Brain_alFunc"
                for rmf in self.brain_anat_orig.parent.glob(f"*{del_tmep}*"):
                    attr = rmf.stem.replace("del_tmep", "alFunc_")
                    delFiles[attr] = rmf

            for rmf in self.work_dir.glob("template2orig_*"):
                attr = rmf.stem.replace("template2orig_", "")
                delFiles[attr] = rmf

            if hasattr(self, "func_orig") and Path(self.func_orig).is_file():
                fbase = re.sub(r"\+.*", "", Path(self.func_orig).name)
                func_mask = (
                    Path(self.func_orig).parent / f"automask_{fbase}.nii.gz"
                )
                if func_mask.is_file():
                    delFiles["func_mask"] = func_mask

        # Delete files
        for attr, ff in delFiles.items():
            if not keepfile:
                if ".HEAD" in ff.suffixes or ".BRIK" in ff.suffixes:
                    ff_stem = ff.stem
                    if ff.suffix == ".gz":
                        ff_stem = Path(ff_stem).stem

                    for rmf in ff.parent.glob(ff_stem + ".*"):
                        rmf.unlink()
                else:
                    if ff.is_file():
                        ff.unlink()

            if hasattr(self, attr):
                self.set_param(attr, "")

        return

    # --- Signal plot ---------------------------------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    class PlotROISignal(QtCore.QObject):
        finished = QtCore.pyqtSignal()

        def __init__(self, root, num_ROIs=1, roi_labels=[], main_win=None):
            super().__init__()

            self.root = root
            self.main_win = main_win
            self.abort = False

            # Initialize figure
            plt_winname = "ROI signal"
            self.plt_win = MatplotlibWindow()
            self.plt_win.setWindowTitle(plt_winname)

            # set position
            if main_win is not None:
                main_geom = main_win.geometry()
                x = main_geom.x() + main_geom.width() + 10
                y = main_geom.y() + 735
            else:
                x, y = (0, 0)
            win_height = 80 + 70 * num_ROIs
            self.plt_win.setGeometry(x, y, 500, win_height)

            # Set axis
            self.roi_labels = roi_labels
            self._axes = self.plt_win.canvas.figure.subplots(num_ROIs, 1)
            if num_ROIs == 1:
                self._axes = [self._axes]

            self.plt_win.canvas.figure.subplots_adjust(
                left=0.15, bottom=0.18, right=0.95, top=0.96, hspace=0.35
            )
            self._color_cycle = plt.get_cmap("tab10")

            self.reset_plot()

            # show window
            self.plt_win.show()
            self.plt_win.canvas.draw()

        # ---------------------------------------------------------------------
        def reset_plot(self):
            self._ln = []
            for ii, ax in enumerate(self._axes):
                ax.cla()
                if len(self.roi_labels) > ii:
                    ax.set_ylabel(self.roi_labels[ii])
                ax.set_xlim(0, 10)
                self._ln.append(ax.plot(0, 0, color=self._color_cycle(ii + 1)))
            self._axes[-1].set_xlabel("TR")

        # ---------------------------------------------------------------------
        def run(self):
            plt_xi = self.root._plt_xi.copy()
            while self.plt_win.isVisible() and not self.abort:
                if self.main_win is not None and not self.main_win.isVisible():
                    break

                if len(self.root._plt_xi) == len(plt_xi):
                    time.sleep(0.1)
                    continue

                try:
                    # Plot signal
                    plt_xi = self.root._plt_xi.copy()
                    plt_roi_sig = self.root._roi_sig
                    for ii, ax in enumerate(self._axes):
                        ll = min(len(plt_xi), len(plt_roi_sig[ii]))
                        if ll == 0:
                            continue

                        y = plt_roi_sig[ii][:ll]
                        self._ln[ii][0].set_data(plt_xi[:ll], y)

                        # Adjust y scale
                        ax.relim()
                        if np.sum(~np.isnan(y)) > 10:
                            yl_orig = np.array(ax.get_ylim())
                            yl = yl_orig.copy()
                            sd = np.nanstd(y)
                            mu = np.nanmean(y)
                            if np.nanmin(y) < yl[0]:
                                yl[0] = max(np.nanmin(y), mu - 4 * sd)

                            if np.nanmax(y) > yl[0]:
                                yl[1] = min(np.nanmax(y), mu + 4 * sd)

                            if np.any(yl != yl_orig):
                                ax.set_ylim(yl)

                        ax.autoscale_view()

                        xl = ax.get_xlim()
                        if (plt_xi[-1] // 10 + 1) * 10 > xl[1]:
                            ax.set_xlim([0, (plt_xi[-1] // 10 + 1) * 10])
                    self.plt_win.canvas.draw()

                except IndexError:
                    continue

                except Exception as e:
                    self.root.errmsg(str(e), no_pop=True)
                    sys.stdout.flush()
                    continue

            self.end_thread()

        # ---------------------------------------------------------------------
        def end_thread(self):
            if self.plt_win.isVisible():
                self.plt_win.close()

            self.finished.emit()

            if hasattr(self.root, "ui_showROISig_cbx"):
                self.root.ui_showROISig_cbx.setCheckState(0)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def open_ROISig_plot(self, num_ROIs=1, roi_labels=[]):
        """Open the ROI signal plot."""
        if hasattr(self, "thPltROISig") and self.thPltROISig.isRunning():
            return

        self.thPltROISig = QtCore.QThread()
        self.pltROISig = RtpApp.PlotROISignal(
            self,
            num_ROIs=num_ROIs,
            roi_labels=roi_labels,
            main_win=self.main_win,
        )
        self.pltROISig.moveToThread(self.thPltROISig)
        self.thPltROISig.started.connect(self.pltROISig.run)
        self.pltROISig.finished.connect(self.thPltROISig.quit)
        self.thPltROISig.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def close_ROISig_plot(self):
        """Close the ROI signal plot."""
        if hasattr(self, "thPltROISig") and self.thPltROISig.isRunning():
            self.pltROISig.abort = True
            if not self.thPltROISig.wait(1):
                self.pltROISig.finished.emit()
                self.thPltROISig.wait()

            del self.thPltROISig

        if hasattr(self, "pltROISig"):
            del self.pltROISig

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def show_ROIsig_chk(self, state):
        """Checkbox to show or hide the ROI signal plot."""
        if state == 2:
            self.open_ROISig_plot(
                num_ROIs=self.num_ROIs, roi_labels=self.roi_labels
            )
        else:
            self.close_ROISig_plot()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save_ROI_sig(self, save_f, ti, roi_sig, roi_label=[]):
        """
        Save ROI signal extracted in RTP.

        Parameters
        ----------
        save_f : Path or str
            Saving file path/name.
        ti : int list
            Time (TR) indices.
        roi_sig : array like
            Array of ROI signals; Nr. ROIs x Nr. TRs
        roi_label : str list, optional
            ROI names. The default is [].

        Returns
        -------
        None.

        """
        if len(roi_label) < len(roi_sig):
            labs = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            roi_label = [labs[ii] for ii in range(len(roi_sig))]

        roi_sig = np.array(roi_sig).T
        ti = ti[: roi_sig.shape[0]]

        savedf = pd.DataFrame(columns=["TR"] + roi_label)
        savedf.loc[:, "TR"] = ti
        savedf.iloc[:, 1:] = roi_sig

        savedf.to_csv(save_f, index=False)

    # --- Session control -----------------------------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    class WAIT_ONSET(QtCore.QObject):
        """Send scan start message to an external application"""

        finished = QtCore.pyqtSignal()

        def __init__(self, parent, onsetObj=None, extApp_sock=None):
            super().__init__()
            self.parent = parent
            self.onsetObj = onsetObj
            self.extApp_sock = extApp_sock
            self.abort = False

        def run(self):
            if self.onsetObj is not None and self.onsetObj.available:
                onset_time = self.onsetObj.scan_onset

            while not self.abort and onset_time == 0:
                if self.onsetObj is not None and self.onsetObj.available:
                    onset_time = self.onsetObj.scan_onset
                time.sleep(1 / 60)

            if self.abort:
                self.finished.emit()
                return

            self.parent._scanning = True
            self.parent._wait_start = False
            self.parent.scan_onset = onset_time
            if self.onsetObj is not None and self.onsetObj.available:
                self.onsetObj.start_scan()

            if self.extApp_sock is not None:
                # Send message to self.extApp_sock
                onset_str = datetime.fromtimestamp(onset_time).isoformat()
                msg = f"SCAN_START {onset_str};"
                try:
                    self.extApp_sock.send(msg.encode())
                except BrokenPipeError:
                    pass

            self.finished.emit()
            return

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ui_setEnabled(self, enabled):
        """Enable or disable the UI elements."""
        if self.main_win is not None:
            objs = list(self.main_win.rtp_objs.values())
            objs += list(self.main_win.rtp_apps.values())
            for pobj in objs:
                if not hasattr(pobj, "ui_objs"):
                    continue

                for ui in pobj.ui_objs:
                    ui.setEnabled(enabled)

                if hasattr(pobj, "ui_enabled_rdb"):
                    pobj.ui_enabled_rdb.setEnabled(enabled)

            if hasattr(self.main_win, "chbRecSignal"):
                self.main_win.chbRecSignal.setEnabled(enabled)
            if hasattr(self.main_win, "chbUseGPU"):
                self.main_win.chbUseGPU.setEnabled(enabled)
            self.main_win.btnSetWorkDir.setEnabled(enabled)
            self.main_win.btnSetWatchDir.setEnabled(enabled)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def chkRunTimerEvent(self):
        """Check the running process"""

        if self.extApp_sock is not None:
            # Check a message from an external application
            msg = self.recv_extApp(timeout=0.001)
            if msg is not None and "END_SESSION" in msg.decode():
                if self.ui_quit_btn.isEnabled():
                    self._logger.info("Recv END_SESSION. End session.")
                    if not self._isRunning_end_run_proc:
                        self.end_run()
                    return
                else:
                    self.ui_ready_btn.setEnabled(False)

                return

        # Check delay in WATCH
        if (
            self.enable_RTP
            and not np.isnan(self.max_watch_wait)
            and len(self.rtp_objs["WATCH"]._proc_time)
        ):
            delay = time.time() - self.rtp_objs["WATCH"]._proc_time[-1]
            if delay > self.max_watch_wait:
                self._logger.info(
                    f"No new file was seen for {delay:.3f} s. End session."
                )
                if not self._isRunning_end_run_proc:
                    self.end_run()
                return

        # Check scan start
        if (
            hasattr(self, "ui_ready_btn")
            and "waiting" in self.ui_ready_btn.text().lower()
        ):
            if self._scanning:
                self.ui_ready_btn.setText("Session is running")
                self.ui_manStart_btn.setEnabled(False)

        # schedule the next check
        self.chk_run_timer.start(1000)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def keyPressEvent(self, event):
        if self._wait_start:
            if event.key() == QtCore.Qt.Key_T:
                # Scan start
                self.manual_start()

    # --- Communication with an external application via RTP_SERVE ----------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def boot_extApp(self, cmd="", timeout=30):
        """
        Setup external appllication with RTP_SERVE server

        Parameters
        ----------
        cmd : str
            Command line string to boot an application.

        Returns
        -------
        self.extApp_addr and self.extApp_proc are set.
        """

        # --- Initialize ---
        if len(cmd) == 0:
            cmd = self.extApp_cmd

        if self.main_win is not None:
            if len(cmd):
                cmdstr = Path(cmd.split()[0]).name
            else:
                cmd = "external application"
            progress_bar = DlgProgressBar(
                title=f"Run {cmdstr} ...", parent=self.main_win
            )
            progress_bar.btnCancel.setVisible(False)
            progress_bar.set_msgTxt(f"Run {cmdstr} ...")
            time.sleep(0.1)  # Give time for UI to update the dialog
        else:
            progress_bar = None

        # --- Kill a running process ---
        if self.extApp_sock is not None:
            self.send_extApp("QUIT;".encode("utf-8"), no_err_pop=True)

            try:
                self.extApp_sock.close()
            except Exception:
                pass
            self.extApp_sock = None

        for kill_cmd in cmd.split():
            kill_cmd = Path(kill_cmd).name
            if kill_cmd.endswith(".sh") or kill_cmd.endswith(".py"):
                kill_cmd = Path(kill_cmd).stem
                break

        if sys.platform == "darwin":
            pret = subprocess.run(
                f"ps -A | grep {kill_cmd}", shell=True, stdout=subprocess.PIPE
            )
        else:
            pret = subprocess.run(
                f"ps -A ww | grep {kill_cmd}",
                shell=True,
                stdout=subprocess.PIPE,
            )
        procs = pret.stdout
        procs = [
            ll
            for ll in procs.decode().rstrip().split("\n")
            if f"grep {kill_cmd}" not in ll and len(ll) > 0
        ]
        for ll in procs:
            pid = int(ll.split()[0])
            try:
                os.killpg(pid, 9)
            except Exception:
                pass

        self.extApp_proc = None

        # --- Boot externel application ---
        if progress_bar is not None:
            progress_bar.add_desc(f"Run {cmd} ...")
            time.sleep(0.1)  # Give time for UI to update the dialog

        extApp_addr, extApp_proc = boot_RTP_SERVE_app(cmd, timeout=timeout)

        # Check app boot failure.
        if extApp_addr is None:
            errmsg = extApp_proc
            errmsg = "Failed to run the external application.\n" + errmsg
            self._logger.error(errmsg)
            self.err_popup(errmsg)
            if progress_bar is not None:
                progress_bar.close()
            return -1

        if progress_bar is not None:
            progress_bar.add_desc(" done.")

        # Set extApp properties
        self.set_param("extApp_addr", extApp_addr)
        self.extApp_proc = extApp_proc
        self.connect_extApp()

        if progress_bar is not None:
            progress_bar.close()

        return 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def connect_extApp(self, no_err_pop=False):
        """Connect to the external application socket."""
        if self.extApp_addr is None:
            errmsg = "No address is set for the external application."
            self._logger.error(errmsg)
            self.err_popup(errmsg)
            return False

        if self.extApp_sock is not None:
            if self.isAlive_extApp():
                return True
            else:
                try:
                    self.extApp_sock.close()
                except Exception:
                    pass

        self.extApp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.extApp_sock.connect(self.extApp_addr)
            self.extApp_sock.settimeout(self.extApp_sock_timeout)
        except ConnectionRefusedError:
            errmsg = f"Failed connecting {self.extApp_addr}"
            self._logger.error(errmsg)
            if not no_err_pop:
                self.err_popup(errmsg)

            return -1

        return 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def isAlive_extApp(self):
        """Check if the external application is alive."""
        if self.extApp_addr is None:
            return False

        if self.extApp_sock is None:
            ret = self.connect_extApp(no_err_pop=True)
            if ret != 0:
                return False

        try:
            # Clear receive buffer
            self.extApp_sock.settimeout(0.01)
            while True:
                try:
                    recv = self.extApp_sock.recv(1024)
                    if len(recv) == 0:
                        break

                except Exception:
                    break
            self.extApp_sock.settimeout(self.extApp_sock_timeout)

            self.extApp_sock.send("IsAlive?;".encode())
            recv = self.extApp_sock.recv(1024)
            if "Yes." in recv.decode():
                return True
            else:
                return False

        except socket.timeout:
            return False

        except BrokenPipeError:
            self.extApp_sock = None
            return False

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_obj, exc_tb)
            return False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def send_extApp(self, data, pkl=False, no_err_pop=False):
        """
        Send data to an external RTP_SERVE application.

        Parameters
        ----------
        data : bytes
            message to send.
        pkl : bool, optional
            send data as a pickled binary.
        """

        if self.extApp_sock is None:
            return False

        if pkl:
            data = pack_data(data)

        try:
            self.extApp_sock.sendall(data)
            return True

        except BrokenPipeError:
            self.extApp_sock = None
            errmsg = "No connection to external app."
            self._logger.debug(errmsg)
            if not no_err_pop:
                self.err_popup(errmsg)
            return False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def recv_extApp(self, bufsize=1024, timeout=None):
        """
        Receive data from an external RTP_SERVE application.

        Parameters
        ----------
        bufsize : int, optional
            reciving buffer size
        timeout : float, optional
            response waiting timeout (s). The default is None ==
            self.extApp_sock_timeout.

        Returns
        -------
        received : TYPE
            DESCRIPTION.

        Returns
        -------
        Returned message from the server.

        """

        if self.extApp_sock is None:
            return None

        if timeout is not None:
            self.extApp_sock.settimeout(timeout)

        try:
            received = self.extApp_sock.recv(bufsize)
        except socket.timeout:
            received = None

        except BrokenPipeError:
            self.extApp_sock = None
            self._logger.error("No connection to external app.")
            received = None

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_obj, exc_tb)
            received = None

        if timeout is not None:
            self.extApp_sock.settimeout(self.extApp_sock_timeout)

        return received

    # --- user interface ------------------------------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_param(
        self, attr, val=None, reset_fn=None, echo=False, unk_warning=True
    ):
        """
        When reset_fn is None, set_param is considered to be called from
        load_parameters function.
        """
        self._logger.debug(f"set_param: {attr} = {val}")

        if attr == "enable_RTP":
            if reset_fn is None and hasattr(self, "ui_enableRTP_chb"):
                if val == 0:
                    self.ui_enableRTP_chb.setCheckState(0)
                else:
                    self.ui_enableRTP_chb.setCheckState(2)

        elif attr == "work_dir":
            if val is None or not Path(val).is_dir():
                return

            val = Path(val)
            setattr(self, attr, val)

            if self.main_win is not None:
                self.main_win.set_workDir(val)

        elif attr == "extApp_cmd":
            if self.main_win is not None and hasattr(
                self, "ui_extApp_cmd_lnEd"
            ):
                self.ui_extApp_cmd_lnEd.setText(val)

        elif attr == "extApp_addr":
            if val is None:
                return

            if type(val) is tuple:
                if self.main_win is not None:
                    self.extApp_addr = val
                    if not self.isAlive_extApp():
                        self.extApp_addr = None
                        return
                    address_str = "{}:{}".format(*val)
                    self.ui_extApp_addr_lnEd.blockSignals(True)
                    self.ui_extApp_addr_lnEd.setText(address_str)
                    self.ui_extApp_addr_lnEd.blockSignals(False)

            elif type(val) is str:
                try:
                    host, port = val.split(":")
                    port = int(port)
                    val = (host, port)

                except Exception:
                    if self.extApp_addr is not None:
                        errmsg = f"{val} is not a valid host:port string"
                        self._logger.error(errmsg)
                        self.err_popup(errmsg)
                        if reset_fn is not None:
                            self.ui_extApp_addr_lnEd.blockSignals(True)
                            addr_str = "{}:{}".format(*self.extApp_addr)
                            reset_fn(addr_str)
                            self.ui_extApp_addr_lnEd.blockSignals(False)
                    return

                if self.extApp_sock is not None:
                    if (
                        self.extApp_addr is None
                        or self.extApp_addr[0] != val[0]
                        or self.extApp_addr[1] != val[1]
                    ):
                        # Address is changed. Close the current socket.
                        self.extApp_sock.close()
                        self.extApp_sock = None

        elif attr == "extApp_isAlive":
            if self.extApp_addr is None:
                extApp_addr = self.ui_extApp_addr_lnEd.text()
                self.set_param("extApp_addr", extApp_addr)
                if self.extApp_addr is None:
                    msgStr = f"!No valid address is set. ({time.ctime()})"
                    reset_fn(msgStr)
                    return

            isAlive = self.isAlive_extApp()
            address_str = "{}:{}".format(*self.extApp_addr)
            if isAlive:
                msgStr = f"{address_str} is alive. ({time.ctime()})"
            else:
                msgStr = f"!Failed to connect {address_str}. ({time.ctime()})"
            reset_fn(msgStr)
            return

        elif attr == "sig_save_file":
            if val == "":
                fname = QtWidgets.QFileDialog.getSaveFileName(
                    None,
                    "Real-time signal save filename",
                    str(self.work_dir),
                    "*.csv",
                )
                if fname[0] == "":
                    return -1

                val = fname[0]
                if reset_fn:
                    reset_fn(val)

            elif type(val) is str:
                try:
                    val = Path(val)
                except Exception:
                    errmsg = f"{val} is not a valid filename."
                    self._logger.error(errmsg)
                    self.err_popup(errmsg)
                    if reset_fn is not None:
                        reset_fn(self.sig_save_file)
                    return

        elif attr in (
            "anat_orig",
            "func_orig",
            "func_param_ref",
            "template",
            "ROI_template",
            "WM_template",
            "Vent_template",
            "sdc_invwarp",
            "func_orig_sdc",
            "alAnat",
            "brain_anat_orig",
            "ROI_orig",
            "WM_orig",
            "Vent_orig",
            "aseg_orig",
            "RTP_mask",
            "GSR_mask",
            "fieldmap_posi",
            "fieldmap_nega",
        ):
            msglab = {
                "anat_orig": "anatomy image in original space",
                "func_orig": "function image in original space",
                "func_param_ref": "fMRI parameter reference image",
                "template": "template image",
                "ROI_template": "ROI mask on template",
                "WM_template": "white matter mask on template",
                "Vent_template": "ventricle mask on template",
                "sdc_invwarp": "distortion correction inverse warp",
                "func_orig_sdc": "distortion-corrected functional image",
                "alAnat": "aligned anatomy image in original space",
                "brain_anat_orig":
                    "skull-stripped brain image in original space",
                "ROI_orig": "ROI mask in original space",
                "WM_orig": "white matter mask in original space",
                "Vent_orig": "ventricle mask in original space",
                "aseg_orig": "aseg in original space",
                "RTP_mask": "mask for real-time processing",
                "GSR_mask": "mask for global signal regression",
                "fieldmap_posi": "fieldmap with same phase encoding",
                "fieldmap_nega": "fieldmap with opposite phase encoding",
            }
            if reset_fn is not None:
                # Set start directory
                startdir = self.work_dir
                if val is not None and os.path.isdir(val):
                    startdir = val
                else:
                    if "template" in attr and Path(self.template).is_file():
                        startdir = Path(self.template).parent
                    else:
                        startdir = self.work_dir

                dlgMdg = "RtpApp: Select {}".format(msglab[attr])
                filt = "*.BRIK* *.nii*;;*.*"
                fname = self.select_file_dlg(
                    dlgMdg, startdir, filt, parent=self.rtp_gui
                )
                if fname[0] == "":
                    return -1

                val = fname[0]
                if reset_fn:
                    reset_fn(val)

                val = Path(val).absolute()
            else:
                if val is None:
                    val = ""
                else:
                    fpath = Path(val).absolute()
                    fpath = fpath.parent / re.sub(
                        "'", "", re.sub(r"\'*\[\d+\]\'*$", "", fpath.name)
                    )
                    if Path(fpath).is_file():
                        val = Path(val).absolute()
                    else:
                        val = ""

                if hasattr(self, f"ui_{attr}_lnEd"):
                    obj = getattr(self, f"ui_{attr}_lnEd")
                    obj.setText(str(val))

        elif attr == "ROI_resample":
            if val not in self.ROI_resample_opts:
                return

            if reset_fn is None and hasattr(self, "ui_ROI_resample_cmbBx"):
                self.ui_ROI_resample_cmbBx.setCurrentText(val)

        elif attr == "no_FastSeg":
            if hasattr(self, "ui_no_FastSeg_chb"):
                self.ui_no_FastSeg_chb.setChecked(val)

        elif attr in ("proc_times", "fastSeg_batch_size", "run_extApp"):
            # Just set the attribute
            pass

        else:
            # Ignore an unrecognized parameter
            if not hasattr(self, attr):
                if unk_warning:
                    self._logger.error(f"{attr} is unrecognized parameter.")
            return

        setattr(self, attr, val)
        if echo:
            print(
                "{}.".format(self.__class__.__name__) + attr,
                "=",
                getattr(self, attr),
            )

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ui_set_param(self):
        """Set up the UI parameters."""
        ui_rows = []
        self.ui_objs = []

        # enabled
        self.ui_enableRTP_chb = QtWidgets.QCheckBox("Enable RTP")
        self.ui_enableRTP_chb.setChecked(self.enable_RTP)
        self.ui_enableRTP_chb.stateChanged.connect(
            lambda x: self.set_param(
                "enable_RTP", x, self.ui_enableRTP_chb.setCheckState
            )
        )
        ui_rows.append((self.ui_enableRTP_chb, None))
        self.ui_objs.append(self.ui_enableRTP_chb)

        # tab
        self.ui_top_tabs = QtWidgets.QTabWidget()
        ui_rows.append((self.ui_top_tabs,))

        # region: Preprocessing tab -------------------------------------------
        self.ui_preprocessingTab = QtWidgets.QWidget()
        self.ui_top_tabs.addTab(self.ui_preprocessingTab, "Preprocessing")
        self.ui_preprocessing_fLayout = QtWidgets.QFormLayout(
            self.ui_preprocessingTab
        )

        # region: dcm2nii --
        self.ui_dcm2nii_btn = QtWidgets.QPushButton("dcm2niix")
        self.ui_dcm2nii_btn.clicked.connect(self.run_dcm2nii)
        self.ui_preprocessing_fLayout.addRow(self.ui_dcm2nii_btn)
        self.ui_objs.append(self.ui_dcm2nii_btn)
        # endregion

        # region: Reference images group ---
        self.ui_RefImg_grpBx = QtWidgets.QGroupBox("Reference images")
        RefImg_gLayout = QtWidgets.QGridLayout(self.ui_RefImg_grpBx)
        self.ui_preprocessing_fLayout.addRow(self.ui_RefImg_grpBx)
        self.ui_objs.append(self.ui_RefImg_grpBx)

        # -- Anatomy orig image --
        ri = 0
        var_lb = QtWidgets.QLabel("Anatomy image :")
        RefImg_gLayout.addWidget(var_lb, ri, 0)

        self.ui_anat_orig_lnEd = QtWidgets.QLineEdit()
        self.ui_anat_orig_lnEd.setReadOnly(True)
        self.ui_anat_orig_lnEd.setStyleSheet("border: 0px none;")
        RefImg_gLayout.addWidget(self.ui_anat_orig_lnEd, ri, 1)
        if self.anat_orig and Path(self.anat_orig).exists():
            self.ui_anat_orig_lnEd.setText(self.anat_orig)

        self.ui_anat_orig_btn = QtWidgets.QPushButton("Set")
        self.ui_anat_orig_btn.clicked.connect(
            lambda: self.set_param(
                "anat_orig", "", self.ui_anat_orig_lnEd.setText
            )
        )
        self.ui_anat_orig_btn.setStyleSheet(
            "background-color: rgb(151,217,235);"
        )
        RefImg_gLayout.addWidget(self.ui_anat_orig_btn, ri, 2)

        self.ui_anat_orig_del_btn = QtWidgets.QPushButton("Unset")
        self.ui_anat_orig_del_btn.clicked.connect(
            lambda: self.delete_file("anat_orig", keepfile=True)
        )
        RefImg_gLayout.addWidget(self.ui_anat_orig_del_btn, ri, 3)

        # -- function orig image --
        ri += 1
        var_lb = QtWidgets.QLabel("Base function image : ")
        RefImg_gLayout.addWidget(var_lb, ri, 0)

        self.ui_func_orig_lnEd = QtWidgets.QLineEdit()
        self.ui_func_orig_lnEd.setReadOnly(True)
        self.ui_func_orig_lnEd.setStyleSheet("border: 0px none;")
        RefImg_gLayout.addWidget(self.ui_func_orig_lnEd, ri, 1)

        self.ui_func_orig_btn = QtWidgets.QPushButton("Set")
        self.ui_func_orig_btn.clicked.connect(
            lambda: self.set_param(
                "func_orig", "", self.ui_func_orig_lnEd.setText
            )
        )
        self.ui_func_orig_btn.setStyleSheet(
            "background-color: rgb(151,217,235);"
        )
        RefImg_gLayout.addWidget(self.ui_func_orig_btn, ri, 2)

        self.ui_func_orig_del_btn = QtWidgets.QPushButton("Unset")
        self.ui_func_orig_del_btn.clicked.connect(
            lambda: self.delete_file("func_orig", keepfile=True)
        )
        RefImg_gLayout.addWidget(self.ui_func_orig_del_btn, ri, 3)

        # -- paremeter reference image --
        ri += 1
        var_lb = QtWidgets.QLabel("fMRI parameter reference : ")
        RefImg_gLayout.addWidget(var_lb, ri, 0)

        self.ui_func_param_ref_lnEd = QtWidgets.QLineEdit()
        self.ui_func_param_ref_lnEd.setReadOnly(True)
        self.ui_func_param_ref_lnEd.setStyleSheet("border: 0px none;")
        RefImg_gLayout.addWidget(self.ui_func_param_ref_lnEd, ri, 1)

        self.ui_param_ref_btn = QtWidgets.QPushButton("Set")
        self.ui_param_ref_btn.clicked.connect(
            lambda: self.set_param(
                "func_param_ref", "", self.ui_func_param_ref_lnEd.setText
            )
        )
        self.ui_param_ref_btn.setStyleSheet(
            "background-color: rgb(151,217,235);"
        )
        RefImg_gLayout.addWidget(self.ui_param_ref_btn, ri, 2)

        self.ui_param_ref_del_btn = QtWidgets.QPushButton("Unset")
        self.ui_param_ref_del_btn.clicked.connect(
            lambda: self.delete_file("func_param_ref", keepfile=True)
        )
        RefImg_gLayout.addWidget(self.ui_param_ref_del_btn, ri, 3)
        # endregion

        # region: Fieldmap PEPOLAR images group --
        self.ui_Fmap_grpBx = QtWidgets.QGroupBox("Fieldmap PEPOLAR images")
        Fmap_gLayout = QtWidgets.QGridLayout(self.ui_Fmap_grpBx)
        self.ui_preprocessing_fLayout.addRow(self.ui_Fmap_grpBx)
        self.ui_objs.append(self.ui_Fmap_grpBx)

        # -- Fieldmap with same phase encoding --
        ri = 0
        var_lb = QtWidgets.QLabel("Same PE direction :")
        Fmap_gLayout.addWidget(var_lb, ri, 0)

        self.ui_fieldmap_posi_lnEd = QtWidgets.QLineEdit()
        self.ui_fieldmap_posi_lnEd.setReadOnly(True)
        self.ui_fieldmap_posi_lnEd.setStyleSheet("border: 0px none;")
        Fmap_gLayout.addWidget(self.ui_fieldmap_posi_lnEd, ri, 1)

        self.ui_fieldmap_posi_btn = QtWidgets.QPushButton("Set")
        self.ui_fieldmap_posi_btn.clicked.connect(
            lambda: self.set_param(
                "fieldmap_posi", "", self.ui_fieldmap_posi_lnEd.setText
            )
        )
        self.ui_fieldmap_posi_btn.setStyleSheet(
            "background-color: rgb(151,217,235);"
        )
        Fmap_gLayout.addWidget(self.ui_fieldmap_posi_btn, ri, 2)

        self.ui_fieldmap_posi_del_btn = QtWidgets.QPushButton("Unset")
        self.ui_fieldmap_posi_del_btn.clicked.connect(
            lambda: self.delete_file("fieldmap_posi", keepfile=True)
        )
        Fmap_gLayout.addWidget(self.ui_fieldmap_posi_del_btn, ri, 3)

        # -- Fieldmap with opposite phase encoding --
        ri += 1
        var_lb = QtWidgets.QLabel("Opposite PE direction :")
        Fmap_gLayout.addWidget(var_lb, ri, 0)

        self.ui_fieldmap_nega_lnEd = QtWidgets.QLineEdit()
        self.ui_fieldmap_nega_lnEd.setReadOnly(True)
        self.ui_fieldmap_nega_lnEd.setStyleSheet("border: 0px none;")
        Fmap_gLayout.addWidget(self.ui_fieldmap_nega_lnEd, ri, 1)

        self.ui_fieldmap_nega_btn = QtWidgets.QPushButton("Set")
        self.ui_fieldmap_nega_btn.clicked.connect(
            lambda: self.set_param(
                "fieldmap_nega", "", self.ui_fieldmap_nega_lnEd.setText
            )
        )
        self.ui_fieldmap_nega_btn.setStyleSheet(
            "background-color: rgb(151,217,235);"
        )
        Fmap_gLayout.addWidget(self.ui_fieldmap_nega_btn, ri, 2)

        self.ui_fieldmap_nega_del_btn = QtWidgets.QPushButton("Unset")
        self.ui_fieldmap_nega_del_btn.clicked.connect(
            lambda: self.delete_file("fieldmap_nega", keepfile=True)
        )
        Fmap_gLayout.addWidget(self.ui_fieldmap_nega_del_btn, ri, 3)
        # endregion

        # region: CreateMasks button --
        ui_CreateMasks_hLayout = QtWidgets.QHBoxLayout()
        self.ui_preprocessing_fLayout.addRow(ui_CreateMasks_hLayout)

        self.ui_CreateMasks_btn = QtWidgets.QPushButton(
            "Create masks (+shift=overwrite)"
        )
        self.ui_CreateMasks_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.ui_CreateMasks_btn.clicked.connect(
            lambda x: self.make_masks(
                progress_dlg=True, no_FastSeg=self.no_FastSeg
            )
        )
        self.ui_CreateMasks_btn.setStyleSheet(
            "background-color: rgb(151,217,235);"
        )
        ui_CreateMasks_hLayout.addWidget(self.ui_CreateMasks_btn)
        self.ui_objs.append(self.ui_CreateMasks_btn)

        # no_FastSeg checkbox
        self.ui_no_FastSeg_chb = QtWidgets.QCheckBox("No FastSeg")
        self.ui_no_FastSeg_chb.setChecked(self.no_FastSeg)
        self.ui_no_FastSeg_chb.stateChanged.connect(
            lambda x: self.set_param(
                "no_FastSeg", x, self.ui_no_FastSeg_chb.setCheckState
            )
        )
        ui_CreateMasks_hLayout.addWidget(self.ui_no_FastSeg_chb)
        self.ui_objs.append(self.ui_no_FastSeg_chb)
        # endregion

        # region: check ROIs groups ---
        self.ui_ChkMask_grpBx = QtWidgets.QGroupBox(
            "Display the masks in AFNI"
        )
        ChkMask_gLayout = QtWidgets.QGridLayout(self.ui_ChkMask_grpBx)
        self.ui_preprocessing_fLayout.addRow(self.ui_ChkMask_grpBx)
        self.ui_objs.append(self.ui_ChkMask_grpBx)

        self.ui_openAFNI_btn = QtWidgets.QPushButton("Open AFNI")
        self.ui_openAFNI_btn.clicked.connect(lambda: self.check_onAFNI())
        ChkMask_gLayout.addWidget(self.ui_openAFNI_btn, 0, 0)

        self.ui_chkFuncAnat_btn = QtWidgets.QPushButton("func-anat align")
        self.ui_chkFuncAnat_btn.clicked.connect(
            lambda: self.check_onAFNI("anat", "func")
        )
        ChkMask_gLayout.addWidget(self.ui_chkFuncAnat_btn, 0, 1)

        self.ui_chkROIFunc_btn = QtWidgets.QPushButton("ROI on function")
        self.ui_chkROIFunc_btn.clicked.connect(
            lambda: self.check_onAFNI("func", "roi")
        )
        ChkMask_gLayout.addWidget(self.ui_chkROIFunc_btn, 0, 2)

        self.ui_chkROIAnat_btn = QtWidgets.QPushButton("ROI on anatomy")
        self.ui_chkROIAnat_btn.clicked.connect(
            lambda: self.check_onAFNI("anat", "roi")
        )
        ChkMask_gLayout.addWidget(self.ui_chkROIAnat_btn, 0, 3)

        self.ui_chkWMFunc_btn = QtWidgets.QPushButton("WM on anatomy")
        self.ui_chkWMFunc_btn.clicked.connect(
            lambda: self.check_onAFNI("anat", "wm")
        )
        ChkMask_gLayout.addWidget(self.ui_chkWMFunc_btn, 1, 0)

        self.ui_chkVentFunc_btn = QtWidgets.QPushButton("Vent on anatomy")
        self.ui_chkVentFunc_btn.clicked.connect(
            lambda: self.check_onAFNI("anat", "vent")
        )
        ChkMask_gLayout.addWidget(self.ui_chkVentFunc_btn, 1, 1)

        self.ui_chkGSRmask_btn = QtWidgets.QPushButton("GSR mask")
        self.ui_chkGSRmask_btn.clicked.connect(
            lambda: self.check_onAFNI("func", "GSRmask")
        )
        ChkMask_gLayout.addWidget(self.ui_chkGSRmask_btn, 1, 2)

        self.ui_chkRTPmask_btn = QtWidgets.QPushButton("RTP mask")
        self.ui_chkRTPmask_btn.clicked.connect(
            lambda: self.check_onAFNI("func", "RTPmask")
        )
        ChkMask_gLayout.addWidget(self.ui_chkRTPmask_btn, 1, 3)
        # endregion

        # endregion: Preprocessing tab ----------------------------------------

        # --- Template tab ----------------------------------------------------
        self.ui_templateTab = QtWidgets.QWidget()
        self.ui_top_tabs.addTab(self.ui_templateTab, "Template")
        template_fLayout = QtWidgets.QFormLayout(self.ui_templateTab)

        # -- Template group box --
        self.ui_Template_grpBx = QtWidgets.QGroupBox("Template images")
        Template_gLayout = QtWidgets.QGridLayout(self.ui_Template_grpBx)
        template_fLayout.addWidget(self.ui_Template_grpBx)
        self.ui_objs.append(self.ui_Template_grpBx)

        ri = 0

        # -- Templete image --
        var_lb = QtWidgets.QLabel("Template brain :")
        Template_gLayout.addWidget(var_lb, ri, 0)

        self.ui_template_lnEd = QtWidgets.QLineEdit()
        self.ui_template_lnEd.setText(str(self.template))
        self.ui_template_lnEd.setReadOnly(True)
        self.ui_template_lnEd.setStyleSheet("border: 0px none;")
        Template_gLayout.addWidget(self.ui_template_lnEd, ri, 1)

        self.ui_template_btn = QtWidgets.QPushButton("Set")
        self.ui_template_btn.clicked.connect(
            lambda: self.set_param(
                "template",
                os.path.dirname(self.ui_template_lnEd.text()),
                self.ui_template_lnEd.setText,
            )
        )
        Template_gLayout.addWidget(self.ui_template_btn, ri, 2)

        self.ui_template_del_btn = QtWidgets.QPushButton("Unset")
        self.ui_template_del_btn.clicked.connect(
            lambda: self.delete_file("template", keepfile=True)
        )
        Template_gLayout.addWidget(self.ui_template_del_btn, ri, 3)

        # -- ROI on template --
        ri += 1
        self.ui_ROI_template_lb = QtWidgets.QLabel("ROI on template :")
        Template_gLayout.addWidget(self.ui_ROI_template_lb, ri, 0)

        self.ui_ROI_template_lnEd = QtWidgets.QLineEdit()
        self.ui_ROI_template_lnEd.setText(str(self.ROI_template))
        self.ui_ROI_template_lnEd.setReadOnly(True)
        self.ui_ROI_template_lnEd.setStyleSheet("border: 0px none;")
        Template_gLayout.addWidget(self.ui_ROI_template_lnEd, ri, 1)

        self.ui_ROI_template_btn = QtWidgets.QPushButton("Set")
        self.ui_ROI_template_btn.clicked.connect(
            lambda: self.set_param(
                "ROI_template",
                os.path.dirname(self.ui_ROI_template_lnEd.text()),
                self.ui_ROI_template_lnEd.setText,
            )
        )
        Template_gLayout.addWidget(self.ui_ROI_template_btn, ri, 2)

        self.ui_ROI_template_del_btn = QtWidgets.QPushButton("Unset")
        self.ui_ROI_template_del_btn.clicked.connect(
            lambda: self.delete_file("ROI_template", keepfile=True)
        )
        Template_gLayout.addWidget(self.ui_ROI_template_del_btn, ri, 3)

        # ROI resampling mode
        ri += 1
        var_ROI_resample_lb = QtWidgets.QLabel("ROI resampling :")
        Template_gLayout.addWidget(var_ROI_resample_lb, ri, 0)

        self.ui_ROI_resample_cmbBx = QtWidgets.QComboBox()
        self.ui_ROI_resample_cmbBx.addItems(self.ROI_resample_opts)
        self.ui_ROI_resample_cmbBx.setCurrentText(self.ROI_resample)
        self.ui_ROI_resample_cmbBx.currentIndexChanged.connect(
            lambda idx: self.set_param(
                "ROI_resample",
                self.ui_ROI_resample_cmbBx.currentText(),
                self.ui_ROI_resample_cmbBx.setCurrentIndex,
            )
        )
        Template_gLayout.addWidget(self.ui_ROI_resample_cmbBx, ri, 1)

        # -- WM on template --
        ri += 1
        self.ui_WM_template_lb = QtWidgets.QLabel("White matter on template :")
        Template_gLayout.addWidget(self.ui_WM_template_lb, ri, 0)

        self.ui_WM_template_lnEd = QtWidgets.QLineEdit()
        self.ui_WM_template_lnEd.setText(str(self.WM_template))
        self.ui_WM_template_lnEd.setReadOnly(True)
        self.ui_WM_template_lnEd.setStyleSheet("border: 0px none;")
        Template_gLayout.addWidget(self.ui_WM_template_lnEd, ri, 1)

        self.ui_WM_template_btn = QtWidgets.QPushButton("Set")
        self.ui_WM_template_btn.clicked.connect(
            lambda: self.set_param(
                "WM_template",
                os.path.dirname(self.ui_WM_template_lnEd.text()),
                self.ui_WM_template_lnEd.setText,
            )
        )
        Template_gLayout.addWidget(self.ui_WM_template_btn, ri, 2)

        self.ui_WM_template_del_btn = QtWidgets.QPushButton("Unset")
        self.ui_WM_template_del_btn.clicked.connect(
            lambda: self.delete_file("WM_template", keepfile=True)
        )
        Template_gLayout.addWidget(self.ui_WM_template_del_btn, ri, 3)

        # -- Ventricle on template --
        ri += 1
        self.ui_Vent_template_lb = QtWidgets.QLabel("Ventricle on template :")
        Template_gLayout.addWidget(self.ui_Vent_template_lb, ri, 0)

        self.ui_Vent_template_lnEd = QtWidgets.QLineEdit()
        self.ui_Vent_template_lnEd.setText(str(self.Vent_template))
        self.ui_Vent_template_lnEd.setReadOnly(True)
        self.ui_Vent_template_lnEd.setStyleSheet("border: 0px none;")
        Template_gLayout.addWidget(self.ui_Vent_template_lnEd, ri, 1)

        self.ui_Vent_template_btn = QtWidgets.QPushButton("Set")
        self.ui_Vent_template_btn.clicked.connect(
            lambda: self.set_param(
                "Vent_template",
                os.path.dirname(self.ui_Vent_template_lnEd.text()),
                self.ui_Vent_template_lnEd.setText,
            )
        )
        Template_gLayout.addWidget(self.ui_Vent_template_btn, ri, 2)

        self.ui_Vent_template_del_btn = QtWidgets.QPushButton("Unset")
        self.ui_Vent_template_del_btn.clicked.connect(
            lambda: self.delete_file("Vent_template", keepfile=True)
        )
        Template_gLayout.addWidget(self.ui_Vent_template_del_btn, ri, 3)

        # --- Processed image tab ---------------------------------------------
        self.ui_procImgTab = QtWidgets.QWidget()
        self.ui_top_tabs.addTab(self.ui_procImgTab, "Processed images")
        procImg_gLayout = QtWidgets.QGridLayout(self.ui_procImgTab)

        # -- distortion correction warp --
        ri0 = 0
        var_lb = QtWidgets.QLabel("Distortion warp :")
        procImg_gLayout.addWidget(var_lb, ri0, 0)

        self.ui_sdc_invwarp_lnEd = QtWidgets.QLineEdit()
        self.ui_sdc_invwarp_lnEd.setText(str(self.sdc_invwarp))
        self.ui_sdc_invwarp_lnEd.setReadOnly(True)
        self.ui_sdc_invwarp_lnEd.setStyleSheet("border: 0px none;")
        procImg_gLayout.addWidget(self.ui_sdc_invwarp_lnEd, ri0, 1)

        self.ui_sdc_invwarp_btn = QtWidgets.QPushButton("Set")
        self.ui_sdc_invwarp_btn.clicked.connect(
            lambda: self.set_param(
                "sdc_invwarp", self.work_dir,
                self.ui_sdc_invwarp_lnEd.setText
            )
        )
        procImg_gLayout.addWidget(self.ui_sdc_invwarp_btn, ri0, 2)

        self.ui_sdc_invwarp_del_btn = QtWidgets.QPushButton("Unset")
        self.ui_sdc_invwarp_del_btn.clicked.connect(
            lambda: self.delete_file("sdc_invwarp", keepfile=True)
        )
        procImg_gLayout.addWidget(self.ui_sdc_invwarp_del_btn, ri0, 3)

        # -- distortion-corrected functional image --
        ri0 += 1
        var_lb = QtWidgets.QLabel("Corrected function :")
        procImg_gLayout.addWidget(var_lb, ri0, 0)

        self.ui_func_orig_sdc_lnEd = QtWidgets.QLineEdit()
        self.ui_func_orig_sdc_lnEd.setText(str(self.func_orig_sdc))
        self.ui_func_orig_sdc_lnEd.setReadOnly(True)
        self.ui_func_orig_sdc_lnEd.setStyleSheet("border: 0px none;")
        procImg_gLayout.addWidget(self.ui_func_orig_sdc_lnEd, ri0, 1)

        self.ui_func_orig_sdc_btn = QtWidgets.QPushButton("Set")
        self.ui_func_orig_sdc_btn.clicked.connect(
            lambda: self.set_param(
                "func_orig_sdc", self.work_dir,
                self.ui_func_orig_sdc_lnEd.setText
            )
        )
        procImg_gLayout.addWidget(self.ui_func_orig_sdc_btn, ri0, 2)

        self.ui_func_orig_sdc_del_btn = QtWidgets.QPushButton("Unset")
        self.ui_func_orig_sdc_del_btn.clicked.connect(
            lambda: self.delete_file("func_orig_sdc", keepfile=True)
        )
        procImg_gLayout.addWidget(self.ui_func_orig_sdc_del_btn, ri0, 3)

        # -- aligned anatomy --
        ri0 += 1
        var_lb = QtWidgets.QLabel("Aligned anatomy :")
        procImg_gLayout.addWidget(var_lb, ri0, 0)

        self.ui_alAnat_lnEd = QtWidgets.QLineEdit()
        self.ui_alAnat_lnEd.setText(str(self.alAnat))
        self.ui_alAnat_lnEd.setReadOnly(True)
        self.ui_alAnat_lnEd.setStyleSheet("border: 0px none;")
        procImg_gLayout.addWidget(self.ui_alAnat_lnEd, ri0, 1)

        self.ui_alAnat_btn = QtWidgets.QPushButton("Set")
        self.ui_alAnat_btn.clicked.connect(
            lambda: self.set_param(
                "alAnat", self.work_dir, self.ui_alAnat_lnEd.setText
            )
        )
        procImg_gLayout.addWidget(self.ui_alAnat_btn, ri0, 2)

        self.ui_alAnat_del_btn = QtWidgets.QPushButton("Unset")
        self.ui_alAnat_del_btn.clicked.connect(
            lambda: self.delete_file("alAnat", keepfile=True)
        )
        procImg_gLayout.addWidget(self.ui_alAnat_del_btn, ri0, 3)

        self.ui_objs.extend(
            [
                var_lb,
                self.ui_alAnat_lnEd,
                self.ui_alAnat_btn,
                self.ui_alAnat_del_btn,
            ]
        )

        # -- warped images --
        ri0 += 1
        self.ui_wrpImg_grpBx = QtWidgets.QGroupBox("Warped images")
        wrpImg_gLayout = QtWidgets.QGridLayout(self.ui_wrpImg_grpBx)
        procImg_gLayout.addWidget(self.ui_wrpImg_grpBx, ri0, 0, 1, 4)
        self.ui_objs.append(self.ui_wrpImg_grpBx)

        # -- WM in the original space --
        ri = 0
        var_lb = QtWidgets.QLabel("WM mask in original :")
        wrpImg_gLayout.addWidget(var_lb, ri, 0)

        self.ui_WM_orig_lnEd = QtWidgets.QLineEdit()
        self.ui_WM_orig_lnEd.setText(str(self.WM_orig))
        self.ui_WM_orig_lnEd.setReadOnly(True)
        self.ui_WM_orig_lnEd.setStyleSheet("border: 0px none;")
        wrpImg_gLayout.addWidget(self.ui_WM_orig_lnEd, ri, 1)

        self.ui_WM_orig_btn = QtWidgets.QPushButton("Set")
        self.ui_WM_orig_btn.clicked.connect(
            lambda: self.set_param(
                "WM_orig", self.work_dir, self.ui_WM_orig_lnEd.setText
            )
        )
        wrpImg_gLayout.addWidget(self.ui_WM_orig_btn, ri, 2)

        self.ui_WM_orig_del_btn = QtWidgets.QPushButton("Unset")
        self.ui_WM_orig_del_btn.clicked.connect(
            lambda: self.delete_file("WM_orig", keepfile=True)
        )
        wrpImg_gLayout.addWidget(self.ui_WM_orig_del_btn, ri, 3)

        # -- Vent in the original space --
        ri += 1
        var_lb = QtWidgets.QLabel("Vent mask in original :")
        wrpImg_gLayout.addWidget(var_lb, ri, 0)

        self.ui_Vent_orig_lnEd = QtWidgets.QLineEdit()
        self.ui_Vent_orig_lnEd.setText(str(self.Vent_orig))
        self.ui_Vent_orig_lnEd.setReadOnly(True)
        self.ui_Vent_orig_lnEd.setStyleSheet("border: 0px none;")
        wrpImg_gLayout.addWidget(self.ui_Vent_orig_lnEd, ri, 1)

        self.ui_Vent_orig_btn = QtWidgets.QPushButton("Set")
        self.ui_Vent_orig_btn.clicked.connect(
            lambda: self.set_param(
                "Vent_orig", self.work_dir, self.ui_Vent_orig_lnEd.setText
            )
        )
        wrpImg_gLayout.addWidget(self.ui_Vent_orig_btn, ri, 2)

        self.ui_Vent_orig_del_btn = QtWidgets.QPushButton("Unset")
        self.ui_Vent_orig_del_btn.clicked.connect(
            lambda: self.delete_file("Vent_orig", keepfile=True)
        )
        wrpImg_gLayout.addWidget(self.ui_Vent_orig_del_btn, ri, 3)

        # -- ROI in the original space --
        ri += 1
        self.ui_ROI_orig_lb = QtWidgets.QLabel("ROI mask in original :")
        wrpImg_gLayout.addWidget(self.ui_ROI_orig_lb, ri, 0)

        self.ui_ROI_orig_lnEd = QtWidgets.QLineEdit()
        self.ui_ROI_orig_lnEd.setText(str(self.ROI_orig))
        self.ui_ROI_orig_lnEd.setReadOnly(True)
        self.ui_ROI_orig_lnEd.setStyleSheet("border: 0px none;")
        wrpImg_gLayout.addWidget(self.ui_ROI_orig_lnEd, ri, 1)

        self.ui_ROI_orig_btn = QtWidgets.QPushButton("Set")
        self.ui_ROI_orig_btn.clicked.connect(
            lambda: self.set_param(
                "ROI_orig", self.work_dir, self.ui_ROI_orig_lnEd.setText
            )
        )
        wrpImg_gLayout.addWidget(self.ui_ROI_orig_btn, ri, 2)

        self.ui_ROI_orig_del_btn = QtWidgets.QPushButton("Unset")
        self.ui_ROI_orig_del_btn.clicked.connect(
            lambda: self.delete_file("ROI_orig", keepfile=True)
        )
        wrpImg_gLayout.addWidget(self.ui_ROI_orig_del_btn, ri, 3)

        # --- RTP_mask ---
        ri0 += 1
        var_lb = QtWidgets.QLabel("RTP mask :")
        procImg_gLayout.addWidget(var_lb, ri0, 0)

        self.ui_RTP_mask_lnEd = QtWidgets.QLineEdit()
        self.ui_RTP_mask_lnEd.setText(str(self.RTP_mask))
        self.ui_RTP_mask_lnEd.setReadOnly(True)
        self.ui_RTP_mask_lnEd.setStyleSheet("border: 0px none;")
        procImg_gLayout.addWidget(self.ui_RTP_mask_lnEd, ri0, 1)

        self.ui_RTP_mask_btn = QtWidgets.QPushButton("Set")
        self.ui_RTP_mask_btn.clicked.connect(
            lambda: self.set_param(
                "RTP_mask", self.work_dir, self.ui_RTP_mask_lnEd.setText
            )
        )
        procImg_gLayout.addWidget(self.ui_RTP_mask_btn, ri0, 2)

        self.ui_RTP_mask_del_btn = QtWidgets.QPushButton("Unset")
        self.ui_RTP_mask_del_btn.clicked.connect(
            lambda: self.delete_file("RTP_mask", keepfile=True)
        )
        procImg_gLayout.addWidget(self.ui_RTP_mask_del_btn, ri0, 3)

        self.ui_objs.extend(
            [
                var_lb,
                self.ui_RTP_mask_lnEd,
                self.ui_RTP_mask_btn,
                self.ui_RTP_mask_del_btn,
            ]
        )

        # --- GSR mask ---
        ri0 += 1
        var_lb = QtWidgets.QLabel("GSR mask :")
        procImg_gLayout.addWidget(var_lb, ri0, 0)

        self.ui_GSR_mask_lnEd = QtWidgets.QLineEdit()
        self.ui_GSR_mask_lnEd.setText(str(self.GSR_mask))
        self.ui_GSR_mask_lnEd.setReadOnly(True)
        self.ui_GSR_mask_lnEd.setStyleSheet("border: 0px none;")
        procImg_gLayout.addWidget(self.ui_GSR_mask_lnEd, ri0, 1)

        self.ui_GSR_mask_btn = QtWidgets.QPushButton("Set")
        self.ui_GSR_mask_btn.clicked.connect(
            lambda: self.set_param(
                "GSR_mask", self.work_dir, self.ui_GSR_mask_lnEd.setText
            )
        )
        procImg_gLayout.addWidget(self.ui_GSR_mask_btn, ri0, 2)

        self.ui_GSR_mask_del_btn = QtWidgets.QPushButton("Unset")
        self.ui_GSR_mask_del_btn.clicked.connect(
            lambda: self.delete_file("GSR_mask", keepfile=True)
        )
        procImg_gLayout.addWidget(self.ui_GSR_mask_del_btn, ri0, 3)

        self.ui_objs.extend(
            [
                var_lb,
                self.ui_GSR_mask_lnEd,
                self.ui_GSR_mask_btn,
                self.ui_GSR_mask_del_btn,
            ]
        )

        # --- Delete all processed images ---
        ri0 += 1
        self.ui_delAllProcImgs_btn = QtWidgets.QPushButton("Delete All")
        self.ui_delAllProcImgs_btn.setStyleSheet(
            "background-color: rgb(255,0,0);"
        )
        self.ui_delAllProcImgs_btn.clicked.connect(
            lambda: self.delete_file("AllProc")
        )
        procImg_gLayout.addWidget(self.ui_delAllProcImgs_btn, ri0, 3)
        self.ui_objs.append(self.ui_delAllProcImgs_btn)

        if self.run_extApp:
            # --- External App tab --------------------------------------------
            self.ui_extAppTab = QtWidgets.QWidget()
            self.ui_top_tabs.addTab(self.ui_extAppTab, "Ext App")
            self.ui_extApp_fLayout = QtWidgets.QFormLayout(self.ui_extAppTab)

            # Command line
            var_lb = QtWidgets.QLabel("App command:")
            self.ui_extApp_cmd_lnEd = QtWidgets.QLineEdit()
            self.ui_extApp_cmd_lnEd.setText(str(self.extApp_cmd))
            self.ui_extApp_cmd_lnEd.editingFinished.connect(
                lambda: self.set_param(
                    "extApp_cmd", self.ui_extApp_cmd_lnEd.text()
                )
            )
            self.ui_extApp_fLayout.addRow(var_lb, self.ui_extApp_cmd_lnEd)
            self.ui_objs.append(self.ui_extApp_cmd_lnEd)

            # Boot App
            self.ui_extApp_run_btn = QtWidgets.QPushButton("Run App")
            self.ui_extApp_run_btn.clicked.connect(
                lambda x: self.boot_extApp()
            )
            self.ui_extApp_fLayout.addRow(self.ui_extApp_run_btn, None)
            self.ui_objs.append(self.ui_extApp_run_btn)

            # address
            var_lb = QtWidgets.QLabel("App server address\n(host:port):")
            self.ui_extApp_addr_lnEd = QtWidgets.QLineEdit()
            self.ui_extApp_addr_lnEd.editingFinished.connect(
                lambda: self.set_param(
                    "extApp_addr",
                    self.ui_extApp_addr_lnEd.text(),
                    self.ui_extApp_addr_lnEd.setText,
                )
            )
            if self.extApp_addr is not None:
                addr_str = "{}:{}".format(*self.extApp_addr)
                self.ui_extApp_addr_lnEd.setText(addr_str)
            self.ui_extApp_fLayout.addRow(var_lb, self.ui_extApp_addr_lnEd)
            self.ui_objs.append(self.ui_extApp_addr_lnEd)

            # Check alive
            self.ui_extApp_isAlive_btn = QtWidgets.QPushButton(
                "Check connection"
            )
            self.ui_extApp_isAlive_btn.clicked.connect(
                lambda x: self.set_param(
                    "extApp_isAlive", "", self.ui_extApp_isAlive_lb.setText
                )
            )
            self.ui_extApp_isAlive_lb = QtWidgets.QLabel("")
            self.ui_extApp_fLayout.addRow(
                self.ui_extApp_isAlive_btn, self.ui_extApp_isAlive_lb
            )
            self.ui_objs.append(self.ui_extApp_isAlive_btn)

        else:
            # --- Output --------------------------------------------
            self.ui_extAppTab = QtWidgets.QWidget()
            self.ui_top_tabs.addTab(self.ui_extAppTab, "Output")
            self.ui_extApp_gLayout = QtWidgets.QGridLayout(self.ui_extAppTab)

            # real-time signal save file
            var_lb = QtWidgets.QLabel("Real-time output file :")
            self.ui_extApp_gLayout.addWidget(var_lb, 0, 0)

            self.ui_sigSaveFile_lnEd = QtWidgets.QLineEdit()
            self.ui_extApp_gLayout.addWidget(self.ui_sigSaveFile_lnEd, 0, 1)
            self.ui_sigSaveFile_lnEd.setText(str(self.sig_save_file))
            self.ui_sigSaveFile_lnEd.editingFinished.connect(
                lambda: self.set_param(
                    "sig_save_file",
                    self.ui_sigSaveFile_lnEd.text(),
                    self.ui_sigSaveFile_lnEd.setText,
                )
            )

            self.ui_sigSaveFile_btn = QtWidgets.QPushButton("Set")
            self.ui_sigSaveFile_btn.clicked.connect(
                lambda: self.set_param(
                    "sig_save_file", "", self.ui_sigSaveFile_lnEd.setText
                )
            )
            self.ui_sigSaveFile_btn.setStyleSheet(
                "background-color: rgb(151,217,235);"
            )
            self.ui_extApp_gLayout.addWidget(self.ui_sigSaveFile_btn, 0, 2)

        # --- Setup experiment button -----------------------------------------
        self.ui_setRTP_btn = QtWidgets.QPushButton("RTP setup")
        self.ui_setRTP_btn.setStyleSheet(
            "background-color: rgb(151,217,235);height: 20px;"
        )

        self.ui_setRTP_btn.clicked.connect(self.RTP_setup)
        ui_rows.append((self.ui_setRTP_btn,))
        self.ui_objs.append(self.ui_setRTP_btn)

        # --- Show Extracted Signal checkbox ----------------------------------
        self.ui_showROISig_cbx = QtWidgets.QCheckBox("Show Extracted Signal")
        self.ui_showROISig_cbx.setCheckState(0)
        self.ui_showROISig_cbx.stateChanged.connect(
            lambda x: self.show_ROIsig_chk(x)
        )
        # This checkbox will be placed by RtpGUI.layout_ui

        # --- Ready and Quit buttons ------------------------------------------
        readyQuitWdt = QtWidgets.QWidget()
        self.main_win.hBoxExpCtrls.addWidget(readyQuitWdt)
        ui_readyQiut_gLayout = QtWidgets.QGridLayout(readyQuitWdt)

        # -- Ready button --
        self.ui_ready_btn = QtWidgets.QPushButton("Ready")
        self.ui_ready_btn.setStyleSheet(
            "font: bold; background-color: rgb(237,45,135);"
        )
        self.ui_ready_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        self.ui_ready_btn.clicked.connect(self.ready_to_run)
        self.ui_ready_btn.setEnabled(False)
        self.ui_ready_btn.setStyleSheet(
            "background-color: rgb(255,201,32);height: 20px;"
        )
        ui_readyQiut_gLayout.addWidget(self.ui_ready_btn, 0, 0, 1, 4)

        # -- Manual start button --
        self.ui_manStart_btn = QtWidgets.QPushButton("Manual start")
        self.ui_manStart_btn.setStyleSheet("background-color: rgb(94,63,153);")
        self.ui_manStart_btn.clicked.connect(self.manual_start)
        self.ui_manStart_btn.setEnabled(False)
        ui_readyQiut_gLayout.addWidget(self.ui_manStart_btn, 0, 4, 1, 1)

        # -- Quit button --
        self.ui_quit_btn = QtWidgets.QPushButton("Quit session")
        self.ui_quit_btn.setStyleSheet("background-color: rgb(255,0,0);")
        self.ui_quit_btn.clicked.connect(partial(self.end_run, True))
        self.ui_quit_btn.setEnabled(False)
        ui_readyQiut_gLayout.addWidget(self.ui_quit_btn, 0, 5, 1, 1)

        return ui_rows

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_params(self):
        opts = super().get_params()

        excld_opts = (
            "ROI_mask",
            "isReadyRun",
            "chk_run_timer",
            "brain_anat_orig",
            "roi_sig",
            "plt_xi",
            "num_ROIs",
            "roi_labels",
            "enable_RTP",
            "proc_times0",
            "extApp_proc",
            "extApp_sock",
            "prtime_keys",
            "run_extApp",
            "scan_onset",
            "extApp_isAlive",
            "ROI_resample_opts",
        )

        for k in excld_opts:
            if k in opts or k[0] == "_":
                del opts[k]

        return opts

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        # Kill self.extApp_proc
        if (
            hasattr(self, "extApp_proc")
            and self.extApp_proc is not None
            and self.extApp_proc.poll() is None
        ):
            # Kill running App
            if self.extApp_sock is not None:
                if self.send_extApp("QUIT;".encode(), no_err_pop=True):
                    while self.extApp_proc.poll() is None:
                        time.sleep(0.1)

                    self.extApp_sock.close()
                    self.extApp_sock = None
            else:
                os.killpg(os.getpgid(self.extApp_proc.pid), signal.SIGTERM)


# %% __main__ =================================================================
if __name__ == "__main__":
    from rtpspy import RtpGUI

    DEBUG = False

    # --- Set logging ---------------------------------------------------------
    dstr = datetime.now().strftime("%Y%m%dT%H%M%S")
    log_file = Path(f"log/RtpApp_{dstr}.log")

    if not log_file.parent.is_dir():
        log_file.parent.mkdir()

    if DEBUG:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        filename=log_file,
        filemode="a",
        format="%(asctime)s.%(msecs)04d,[%(levelname)s],%(name)s,%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # --- Start app -----------------------------------------------------------
    app = QtWidgets.QApplication(sys.argv)

    # Make RtpApp instance
    rtp_app = RtpApp()

    # Make RtpGUI instance
    app_obj = {"RTP App": rtp_app}
    rtp_ui = RtpGUI(rtp_app.rtp_objs, app_obj, log_file=log_file)

    # Keep RTP objects for loading and saving the parameters
    all_rtp_objs = rtp_app.rtp_objs
    all_rtp_objs.update(app_obj)

    if Path("RNT_CNF_params.pkl").is_file():
        load_parameters(all_rtp_objs, fname="RNT_CNF_params.pkl")

    # Run the application
    sys.excepthook = excepthook
    try:
        rtp_ui.show()
        rtp_ui.show_physio_chk(2)
        exit_code = app.exec_()

    except Exception as e:
        with open("rtpspy.error", "w") as fd:
            fd.write(str(e))

        print(str(e))
        exit_code = -1

    # --- End ---
    save_parameters(all_rtp_objs, fname="RNT_CNF_params.pkl")
    sys.exit(exit_code)
