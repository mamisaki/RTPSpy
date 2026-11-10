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
import tempfile

import numpy as np
import nibabel as nib
import pandas as pd
import torch

from PyQt5 import QtWidgets, QtCore
import matplotlib.pyplot as plt

try:
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

except ImportError:
    # Run as a script (no parent package)
    from rtpspy.rtp_common import (
        RTP,
        boot_afni,
        MatplotlibWindow,
        DlgProgressBar,
        excepthook,
        load_parameters,
        save_parameters,
    )
    from rtpspy.rtp_watch import RtpWatch
    from rtpspy.rtp_volreg import RtpVolreg
    from rtpspy.rtp_tshift import RtpTshift
    from rtpspy.rtp_smooth import RtpSmooth
    from rtpspy.rtp_regress import RtpRegress
    from rtpspy.rtp_imgproc import RtpImgProc
    from rtpspy.rtp_serve import boot_RTP_SERVE_app, pack_data
    from rtpspy.rtp_ttl_physio import RtpTTLPhysio


# %% Constants and helper functions ===========================================
# Labels of the image file parameters, used in the file selection dialog
_FILE_PARAM_LABELS = {
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
    "brain_anat_orig": "skull-stripped brain image in original space",
    "ROI_orig": "ROI mask in original space",
    "WM_orig": "white matter mask in original space",
    "Vent_orig": "ventricle mask in original space",
    "aseg_orig": "aseg in original space",
    "RTP_mask": "mask for real-time processing",
    "GSR_mask": "mask for global signal regression",
    "fieldmap_posi": "fieldmap with same phase encoding",
    "fieldmap_nega": "fieldmap with opposite phase encoding",
}

# Set button color in the UI
_SET_BTN_STYLE = "background-color: rgb(151,217,235);"


def _strip_subbrick_selector(fpath):
    """Return the file path without an AFNI sub-brick selector.

    e.g., func.nii.gz'[0]' -> func.nii.gz
    """
    fpath = Path(fpath)
    fname = re.sub("'", "", re.sub(r"\'*\[\d+\]\'*$", "", fpath.name))
    return fpath.parent / fname


def _send_scan_start(sock, onset):
    """Send the scan start time to an external application."""
    onset_str = datetime.fromtimestamp(onset).isoformat()
    try:
        sock.send(f"SCAN_START {onset_str};".encode())
    except BrokenPipeError:
        pass


def _own_process_tree():
    """Return the PIDs of this process and its ancestors."""
    pids = set()
    pid = os.getpid()
    while pid > 0 and pid not in pids:
        pids.add(pid)
        try:
            with open(f"/proc/{pid}/stat") as fd:
                # The command name (field 2) can contain spaces. The parent
                # PID is the second field after the closing parenthesis.
                pid = int(fd.read().rsplit(")", 1)[1].split()[1])
        except Exception:
            break
    return pids


def _find_processes(prog, args=()):
    """Return the PIDs of this user's processes running prog.

    A process matches when the base name of its argv[0] equals prog and its
    argv[1:] contains every element of args exactly. The argv is read from
    /proc/<pid>/cmdline, so a shell whose command line merely contains prog
    as text does not match. This process and its ancestors are excluded.
    """
    own_tree = _own_process_tree()
    pids = []
    for proc_d in Path("/proc").glob("[0-9]*"):
        pid = int(proc_d.name)
        if pid in own_tree:
            continue
        try:
            if proc_d.stat().st_uid != os.getuid():
                continue
            argv = (proc_d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue  # The process ended or is not accessible

        argv = [arg.decode(errors="replace") for arg in argv if len(arg)]
        if len(argv) == 0 or Path(argv[0]).name != prog:
            continue
        if all(arg in argv[1:] for arg in args):
            pids.append(pid)

    return pids


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
        Parameters
        ----------
        default_rtp_params : dict, optional
            Default parameters: {module name: {parameter: value}}. The
            module name is a key of self.rtp_objs or "APP" for this object.
            The default is None.
        work_dir : Path or str, optional
            Working directory. The default is "".
        physio_log_file : Path or str, optional
            Physio log file passed to RtpTTLPhysio. The default is None.
        rt_physio_address_name : list, optional
            Address and socket name of the rt_physio process, passed to
            RtpTTLPhysio.
        config_path : Path, optional
            Configuration file passed to RtpTTLPhysio.
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
        self.aseg_orig = ""
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

        # Mask creation parameters. FastSeg needs a GPU.
        self.no_FastSeg = not (
            torch.cuda.is_available() or torch.backends.mps.is_available()
        )
        self.fastSeg_batch_size = 1

        # Default processing times (seconds) for the progress bar of
        # make_masks
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
        # warped ROI
        self.ROI_resample_opts = ["nearestNeighbor", "linear", "bSpline"]
        self.ROI_resample = "nearestNeighbor"

        # Scan onset
        self._wait_start = False
        self._scanning = False
        self.scan_onset = 0.0

        # Timer to check the running status
        self.chk_run_timer = QtCore.QTimer()
        self.chk_run_timer.setSingleShot(True)
        self.chk_run_timer.timeout.connect(self.chkRunTimerEvent)
        self.max_watch_wait = 5  # seconds

        # endregion

        # region: External application ----------------------------------------
        # An external application that receives the RTP signal
        self.run_extApp = False
        self.extApp_cmd = ""
        self.extApp_addr = None
        self.extApp_proc = None
        self._extApp_pgid = None  # process group ID of extApp_proc
        self.extApp_sock = None
        self.extApp_sock_timeout = 3

        self.sig_save_file = Path(self.work_dir) / "rtp_ROI_signal.csv"
        # endregion

        # region: Initialize signal plot --------------------------------------
        self.num_ROIs = 1
        self.roi_labels = ["ROI"]
        self._plt_xi = []

        self._roi_sig = [[] for _ in range(self.num_ROIs)]
        # endregion

        # region: RTP module instances ----------------------------------------
        self.rtp_objs = {}
        self.rtp_objs["WATCH"] = RtpWatch()
        self.rtp_objs["VOLREG"] = RtpVolreg()
        self.rtp_objs["TSHIFT"] = RtpTshift()
        self.rtp_objs["SMOOTH"] = RtpSmooth()
        self.rtp_objs["TTLPHYSIO"] = RtpTTLPhysio(
            physio_log_file=physio_log_file,
            rt_physio_address_name=rt_physio_address_name,
            config_path=config_path
        )
        self.rtp_objs["REGRESS"] = RtpRegress(
            volreg=self.rtp_objs["VOLREG"],
            rtp_physio=self.rtp_objs["TTLPHYSIO"]
        )
        self.rtp_gui = None
        # endregion

        # region: Clean temporary DICOM files ---------------------------------
        # Remove only the temporary directories that DicomConverter made and
        # left (e.g., after a crash): owned by this user and not modified for
        # an hour (another RTPSpy process may be using a newer one).
        tmp_root = Path(tempfile.gettempdir())
        for tmp_d in tmp_root.glob("dicom_converter_*"):
            try:
                st = tmp_d.lstat()
                if (
                    tmp_d.is_dir() and not tmp_d.is_symlink()
                    and st.st_uid == os.getuid()
                    and time.time() - st.st_mtime > 3600
                ):
                    shutil.rmtree(tmp_d)
            except Exception as e:
                self._logger.warning(f"Failed to remove {tmp_d}: {e}")
        # endregion

        # region: Set the default RTP parameters ------------------------------
        if default_rtp_params is not None:
            self._set_module_params(default_rtp_params)

            if "APP" in default_rtp_params:
                for attr, val in default_rtp_params["APP"].items():
                    self.set_param(attr, val)

        self._logger.debug("### Complete RtpApp initialization ###")
        # endregion

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_module_params(self, rtp_params):
        """Set parameters of the RTP modules.

        rtp_params : {module name: {parameter: value}}. Keys that are not in
        self.rtp_objs are ignored.
        """
        for proc, params in rtp_params.items():
            if proc in self.rtp_objs:
                for attr, val in params.items():
                    self.rtp_objs[proc].set_param(attr, val)

    # --- Override these functions for a custom application -------------------
    # ready_proc, do_proc, end_reset, end_proc
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ready_proc(self):
        """Prepare the process."""
        self.ROI_mask = None
        if Path(self.ROI_orig).is_file():
            self.ROI_mask = np.asarray(nib.load(self.ROI_orig).dataobj)
            self._proc_ready = True
        else:
            errmsg = f"ROI mask in original space not found: {self.ROI_orig}."
            self._logger.error(errmsg)
            self.err_popup(errmsg)
            self._proc_ready = False

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
                return

            if self._proc_start_idx < 0:
                self._proc_start_idx = vol_idx

            dataV = fmri_img.get_fdata()

            # --- Initialize --------------------------------------------------
            if Path(self.ROI_orig).is_file() and self.ROI_mask is None:
                self.ROI_mask = np.asarray(nib.load(self.ROI_orig).dataobj)

            # --- Run the process ---------------------------------------------
            # Mean signal in the ROI, excluding voxels with zero signal
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
                    self._logger.error(str(e) + "\n" + traceback.format_exc())

            else:
                # Save data to a file in real-time
                with open(self.sig_save_file, "a") as save_fd:
                    print(val_str, file=save_fd)
                self._logger.info(f"Saved ROI data '{val_str}'")

            # --- Post-process ------------------------------------------------
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
            msg = f"#{vol_idx + 1};ROI signal extraction;{fname}"
            msg += f";tstamp={tstamp}"
            if pre_proc_time is not None:
                msg += f";took {proc_delay:.4f}s"
            self._logger.info(msg)

            # Update signal plot
            self._plt_xi.append(vol_idx + 1)
            self._roi_sig[0].append(mean_sig)

        except Exception as e:
            self._logger.error(str(e) + "\n" + traceback.format_exc())

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end_reset(self):
        """End process and reset process parameters."""
        self._logger.info(f"Reset {self.__class__.__name__} module.")
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
        cmd = "dcm2niix -f sub-%n_ses-%t_ser-%s_desc-%d"
        cmd += f" -i n -z o -w 0 -o {out_dir} {dcm_dir}"
        try:
            # Progress dialog without buttons
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

            # Copy DICOMs to the working directory
            dst_dir = out_dir / "dicom"
            cmd = f"rsync -auvz {dcm_dir}/ {dst_dir}/"
            subprocess.check_call(shlex.split(cmd), stdout=subprocess.DEVNULL)

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
        Make mask images for RTP.

        Parameters
        ----------
        func_orig : str or Path, optional
            Reference function image file. An AFNI sub-brick selector (e.g.,
            func.nii.gz'[0]') can be added. The default is None.
        anat_orig : str or Path, optional
            Anatomy image file. The default is None.
        template : str or Path, optional
            Template image file. The default is None.
        ROI_template : str or Path, optional
            ROI mask image on template. The default is None.
        no_FastSeg : bool, optional
            Do not use FastSeg. Use 3dSkullStrip and the template WM and Vent
            masks warped into the subject brain. The default is False.
        WM_template : str or Path, optional
            White matter mask on template. Used only with no_FastSeg.
            The default is None.
        Vent_template : str or Path, optional
            Ventricle mask on template. Used only with no_FastSeg.
            The default is None.
        ask_cmd : bool, optional
            Ask the user to edit the processing command lines.
            The default is False.
        overwrite : bool, optional
            Overwrite existing files. The default is False.
        progress_dlg : bool, optional
            Show a progress dialog. The default is False.

        Returns
        -------
        -1 if a required input is missing, otherwise None.
        """
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

        input_files = {
            "anat_orig": anat_orig,
            "template": template,
            "ROI_template": ROI_template,
        }
        if no_FastSeg:
            input_files["WM_template"] = WM_template
            input_files["Vent_template"] = Vent_template
        for attr, val in input_files.items():
            if val is not None:
                self.set_param(attr, val)

        # Check the input files
        required = ["func_orig", "anat_orig"]
        if no_FastSeg:
            required += ["WM_template", "Vent_template"]
        for attr in (
            "func_orig",
            "anat_orig",
            "template",
            "ROI_template",
            "WM_template",
            "Vent_template",
        ):
            if getattr(self, attr, None) in (None, ""):
                if attr in required:
                    errmsg = f"\n{attr} is not set.\n"
                    self._logger.error(errmsg)
                    self.err_popup(errmsg)
                    return -1
                continue

            if not _strip_subbrick_selector(getattr(self, attr)).is_file():
                if attr not in ("func_orig", "anat_orig") and not no_FastSeg:
                    # Optional template file. Unset it.
                    self.set_param(attr, "")
                    continue

                errmsg = f"\n{attr} not found: {getattr(self, attr)}.\n"
                self._logger.error(errmsg)
                self.err_popup(errmsg)
                return -1

            self.set_param(attr, Path(getattr(self, attr)))

        if hasattr(self, "ui_CreateMasks_btn"):
            self.ui_CreateMasks_btn.setEnabled(False)

        if self.main_win is not None:
            # Shift-click: overwrite. Ctrl-click: ask_cmd.
            shift = QtCore.Qt.ShiftModifier
            ctrl = QtCore.Qt.ControlModifier
            modifiers = QtWidgets.QApplication.keyboardModifiers()
            if modifiers in (shift, ctrl | shift):
                overwrite = True
            if modifiers in (ctrl, ctrl | shift):
                ask_cmd = True

        # Progress bar. The processing output to stdout is shown in the
        # dialog.
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

        # Processing times for the progress report. Only the steps to run are
        # counted.
        for k in self.prtime_keys:
            if k not in self.proc_times:
                self.proc_times[k] = 1

        if no_FastSeg:
            del self.proc_times["FastSeg"]
        else:
            del self.proc_times["SkullStrip"]

        improc = RtpImgProc(main_win=self.main_win)
        improc.proc_times = self.proc_times

        total_ETA = np.sum(list(self.proc_times.values()))
        if not Path(self.template).is_file():
            total_ETA -= self.proc_times["ANTs"]

        # Output directory: work_dir/RTP
        work_dir = self.work_dir
        if work_dir.name != 'RTP':
            out_dir = work_dir / 'RTP'
            if not out_dir.is_dir():
                out_dir.mkdir(parents=True)
        else:
            out_dir = work_dir

        st0 = time.time()
        OK = True
        # endregion: Initialize

        try:
            # region: Get the image space and deoblique the anatomy -----------
            # The space label (e.g., ORIG) is copied to vr_base_*.
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

            anat_orig = out_dir / Path(self.anat_orig).name
            improc.copy_deoblique(
                self.anat_orig, anat_orig, progress_bar=progress_bar,
                overwrite=overwrite
            )
            # endregion

            # region: 0. Copy func_orig to out_dir as vr_base_* ---------------
            if (
                Path(func_orig).parent != out_dir
                or not Path(func_orig).stem.startswith("vr_base_")
                or overwrite
            ):
                # AFNI sub-brick selector (e.g., func.nii.gz'[0]' or
                # func.nii.gz[0]). The last volume is used if not specified.
                sel_pat = r"[\'\"]*\[(\d+)\][\'\"]*$"
                ma = re.search(sel_pat, Path(func_orig).name)
                if ma is not None:
                    vidx = int(ma.groups()[0])
                else:
                    vidx = -1

                src_f = Path(func_orig)
                src_f = src_f.parent / re.sub(sel_pat, "", src_f.name)
                # File name without the image file extension
                src_f_stem = re.sub(
                    r"(\.nii(\.gz)?|\+\w+\.(HEAD|BRIK)(\.gz)?)$", "",
                    src_f.name
                )

                if not src_f_stem.startswith("vr_base_"):
                    src_save_fname = "vr_base_" + src_f_stem
                else:
                    src_save_fname = src_f_stem

                dst_f = out_dir / f"{src_save_fname}.nii.gz"
                if (
                    dst_f != Path(func_orig) and
                    (not dst_f.is_file() or overwrite)
                ):
                    src_img = improc.load_image(src_f, vidx=vidx)
                    assert src_img is not None, f"Failed to load {src_f}.\n"
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

                # json sidecar file (the name RtpTshift looks for)
                dst_json_f = out_dir / f"{src_save_fname}.json"
                if not dst_json_f.is_file() or overwrite:
                    # Copy the json file of the source image if it exists.
                    # Look also for the name without 'vr_base_' and in the
                    # parent directory.
                    json_name = f"{src_f_stem}.json"
                    json_name_nobase = json_name.replace("vr_base_", "")
                    json_cands = [
                        src_f.parent / json_name,
                        src_f.parent / json_name_nobase,
                        src_f.parent.parent / json_name_nobase,
                    ]
                    json_f = next(
                        (ff for ff in json_cands if ff.is_file()),
                        json_cands[-1]
                    )

                    if json_f.is_file() and json_f != dst_json_f:
                        shutil.copy(json_f, dst_json_f)
                    else:
                        # Save TR and slice timing from the image header
                        header = nib.load(src_f).header

                        slice_timing = []
                        if hasattr(header, "get_slice_times"):
                            try:
                                slice_timing = header.get_slice_times()
                            except nib.spatialimages.HeaderDataError:
                                pass
                        elif (
                            hasattr(header, "info")
                            and "TAXIS_OFFSETS" in header.info
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
            # endregion

            # Use self.set_param() to set the result files so that the GUI
            # fields are updated.

            # region: 1. Brain segmentation -----------------------------------
            if not no_FastSeg:
                # FastSeg makes the brain, WM, Vent, and aseg segmentations.
                improc.fastSeg_batch_size = self.fastSeg_batch_size
                seg_files = improc.run_fast_seg(
                    out_dir,
                    anat_orig,
                    total_ETA,
                    progress_bar=progress_bar,
                    overwrite=overwrite,
                )
                assert seg_files is not None
                self.set_param("brain_anat_orig", seg_files[0])
                fastseg_files = {
                    "WM": seg_files[1],
                    "Vent": seg_files[2],
                    "aseg": seg_files[3],
                }
            else:
                brain_anat_orig = improc.skullStrip(
                    out_dir,
                    anat_orig,
                    total_ETA,
                    progress_bar=progress_bar,
                    ask_cmd=ask_cmd,
                    overwrite=overwrite,
                )
                # skullStrip returns -1 when the command edit is canceled
                assert brain_anat_orig not in (None, -1), \
                    "skullStrip failed or canceled.\n"
                self.set_param("brain_anat_orig", brain_anat_orig)
            # endregion

            # region: 2. Susceptibility distortion correction -----------------
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
                self.set_param("sdc_invwarp", sdc_invwarp)
                self.set_param("func_orig_sdc", func_orig_sdc)
                func_base = func_orig_sdc
            else:
                self.set_param("sdc_invwarp", "")
                self.set_param("func_orig_sdc", "")
                func_base = self.func_orig

            # With distortion correction, the masks are made in the corrected
            # space and warped back to the space of func_orig.
            do_invsdc = bool(self.sdc_invwarp) and \
                Path(self.sdc_invwarp).is_file()

            def apply_invsdc(img_f, interpolator):
                """Warp img_f into the uncorrected func_orig space."""
                return improc.ants_warp_resample(
                    out_dir,
                    img_f,
                    self.func_orig,
                    [self.sdc_invwarp],
                    suffix="_invsdc",
                    interpolator=interpolator,
                    progress_bar=progress_bar,
                    overwrite=overwrite,
                )
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

            if do_invsdc:
                mask_files = [
                    apply_invsdc(mask_f, "nearestNeighbor")
                    for mask_f in mask_files
                ]
                assert None not in mask_files

            self.set_param("RTP_mask", mask_files[0])
            self.set_param("GSR_mask", mask_files[1])
            # endregion

            # region: 5. Warp template ----------------------------------------
            # The template registration is needed for the ROI template and,
            # without FastSeg, for the WM and Vent templates.
            warp_params = None
            if Path(self.template).is_file() and (
                Path(self.ROI_template).is_file() or no_FastSeg
            ):
                warp_params = improc.warp_template(
                    out_dir,
                    self.alAnat,
                    self.template,
                    total_ETA,
                    progress_bar=progress_bar,
                    ask_cmd=ask_cmd,
                    overwrite=overwrite,
                )
                # warp_template returns None or -1 on failure or cancel
                assert warp_params not in (None, -1), \
                    "warp_template failed or canceled.\n"
            # endregion

            # region: 6. Warp ROI template ------------------------------------
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

                if do_invsdc:
                    ROI_orig_invsdc = apply_invsdc(ROI_orig, self.ROI_resample)
                    assert ROI_orig_invsdc is not None
                    self.set_param("ROI_orig", ROI_orig_invsdc)
            # endregion

            # region: 7. Make white matter and ventricle masks ----------------
            # FastSeg masks are aligned with aff1D_f. Template masks (no
            # FastSeg) need the template registration.
            if not no_FastSeg or warp_params is not None:
                # {segment name: (erosion, interpolator)}
                # Binary masks and label maps are resampled without
                # interpolation not to dilate or mix the labels.
                seg_opts = {
                    "WM": (2, "nearestNeighbor"),
                    "Vent": (1, "nearestNeighbor"),
                    "aseg": (0, "genericLabel"),
                }
                for segname, (erode, interpolator) in seg_opts.items():
                    if no_FastSeg:
                        if segname == "aseg":
                            # aseg is made only by FastSeg
                            self.set_param("aseg_orig", "")
                            continue

                        # Warp the template mask to the aligned anatomy
                        seg_anat_f = getattr(self, f"{segname}_template")
                        assert seg_anat_f.is_file()
                        seg_anat_f = improc.ants_warp_resample(
                            out_dir,
                            seg_anat_f,
                            self.alAnat,
                            warp_params,
                            interpolator=interpolator,
                            progress_bar=progress_bar,
                            overwrite=overwrite,
                        )
                        assert seg_anat_f is not None
                        seg_aff1D_f = None
                    else:
                        seg_anat_f = fastseg_files[segname]
                        assert seg_anat_f.is_file()
                        seg_aff1D_f = aff1D_f

                    seg_al_f = improc.resample_segmasks(
                        out_dir,
                        seg_anat_f,
                        segname,
                        erode,
                        self.func_orig,
                        total_ETA,
                        aff1D_f=seg_aff1D_f,
                        progress_bar=progress_bar,
                        ask_cmd=ask_cmd,
                        overwrite=overwrite,
                    )
                    assert seg_al_f is not None

                    if do_invsdc:
                        seg_al_f = apply_invsdc(seg_al_f, interpolator)
                        assert seg_al_f is not None

                    self.set_param(f"{segname}_orig", seg_al_f)
            else:
                # Do not leave the masks of a previous run.
                self._logger.warning(
                    "WM and Vent masks are not made: template is not set."
                )
                for segname in ("WM", "Vent", "aseg"):
                    self.set_param(f"{segname}_orig", "")
            # endregion

        except Exception:
            OK = False
            traceback.print_exc()

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
            if self.main_win is not None:
                QtWidgets.QMessageBox.information(
                    self.main_win,
                    "Mask creation is complete.",
                    "Mask creation is complete.",
                )

        if hasattr(self, "ui_CreateMasks_btn"):
            self.ui_CreateMasks_btn.setEnabled(True)
        # endregion

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def RTP_setup(self, *args, rtp_params={}, ignore_error=False):
        """
        Set up and enable the RTP modules.

        Parameters
        ----------
        rtp_params : dict, optional
            Parameters of the RTP modules: {module name: {parameter: value}}.
            The default is {}.
        ignore_error : bool, optional
            Ignore errors to allow an incomplete setup. The default is False.

        Returns
        -------
        int
             0 : normal
            -1 : error or canceled
        """
        # *args absorbs the 'checked' argument of the button click signal.
        if "enable_RTP" in rtp_params:
            self.set_param("enable_RTP", bool(rtp_params["enable_RTP"]) * 2)
        else:
            self.set_param("enable_RTP", 2)

        show_progress = (self.main_win is not None) and (self.enable_RTP > 0)
        if show_progress:
            progress_bar = DlgProgressBar(
                title="Session setup ...", parent=self.main_win
            )
            progress = 0
            time.sleep(0.01)  # Give the UI time to update the dialog

        def cancel_setup():
            if show_progress and progress_bar.isVisible():
                progress_bar.close()
            return -1

        # region: Set RTP parameters ------------------------------------------
        if self.enable_RTP > 0:
            self._set_module_params(rtp_params)

            for proc in ["WATCH", "VOLREG", "TSHIFT", "SMOOTH", "REGRESS"]:
                if show_progress:
                    progress += 100 // 5
                    progress_bar.set_value(progress)
                    progress_bar.add_desc(f"Set {proc} parameters\n")
                    time.sleep(0.01)  # Give the UI time to update the dialog

                if proc not in self.rtp_objs:
                    continue

                pobj = self.rtp_objs[proc]
                if pobj.enabled:
                    if hasattr(pobj, "work_dir"):
                        pobj.work_dir = Path(self.work_dir)
                        if not pobj.work_dir.is_dir():
                            pobj.work_dir.mkdir()

                    if proc == "WATCH":
                        if not Path(pobj.watch_dir).is_dir():
                            errmsg = "'Watching directory' must be set."
                            self._logger.error(errmsg)
                            self.err_popup(errmsg)
                            if not ignore_error:
                                return cancel_setup()

                        if not Path(self.func_orig).is_file():
                            errmsg = (
                                "'Base function image' not found:"
                                + f" {self.func_orig}."
                            )
                            self._logger.error(errmsg)
                            self.err_popup(errmsg)
                            if not ignore_error:
                                return cancel_setup()

                    elif proc == "VOLREG":
                        if Path(self.func_orig).is_file():
                            pobj.set_param("ref_vol", self.func_orig)
                        else:
                            # Ask the user to select a reference volume
                            ret = pobj.set_param("ref_vol", "external")
                            if ret == -1:  # Canceled
                                return cancel_setup()

                    elif proc == "TSHIFT":
                        if Path(self.func_param_ref).is_file():
                            # Get the slice timing and TR from the reference
                            pobj.set_from_sample(self.func_param_ref)
                        elif not ignore_error:
                            errmsg = (
                                "'fMRI parameter reference' not found:"
                                + f" {self.func_param_ref}."
                            )
                            self._logger.error(errmsg)
                            self.err_popup(errmsg)
                            return cancel_setup()

                    elif proc == "SMOOTH":
                        if Path(self.RTP_mask).is_file():
                            pobj.set_param("mask_file", self.RTP_mask)
                        elif not ignore_error:
                            # Ask the user to select a mask file
                            ret = pobj.set_param("mask_file", "external")
                            if ret == -1:  # Canceled
                                return cancel_setup()

                    elif proc == "REGRESS":
                        if self._setup_regress(pobj, ignore_error) == -1:
                            return cancel_setup()

                if show_progress and not progress_bar.isVisible():
                    self._logger.info("Cancel experiment setup")
                    return -1

        # endregion: Set RTP parameters

        if show_progress:
            progress_bar.set_value(100)
            progress_bar.add_desc("Done")
            time.sleep(0.1)  # Give the UI time to update the dialog

        if self.main_win is not None:
            self.ui_ready_btn.setEnabled(True)
            self.ui_ready_btn.setText("Ready")
            self.ui_quit_btn.setEnabled(True)

            if show_progress:
                progress_bar.close()

            # Show the parameter list tab
            self.main_win.show_options_list()
            self.main_win.options_tab.setCurrentIndex(2)
            QtWidgets.QApplication.processEvents()
            time.sleep(0.1)

        return 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _setup_regress(self, pobj, ignore_error):
        """Set the REGRESS parameters that depend on the other modules.

        Returns -1 if the user canceled a mask file selection.
        """
        pobj.set_param("TR", self.rtp_objs["TSHIFT"].TR)
        pobj.set_param("tshift", self.rtp_objs["TSHIFT"].ref_time)
        pobj.set_param("mask_file", self.RTP_mask)
        if pobj.mot_reg != "None":
            if not self.rtp_objs["VOLREG"].enabled:
                pobj.ui_motReg_cmbBx.setCurrentText("None")
                errmsg = "VOLREG is not enabled. "
                errmsg += "Motion regressor is set to None."
                self.rtp_objs["VOLREG"].errmsg(errmsg)
            else:
                pobj.set_param("volreg", self.rtp_objs["VOLREG"])

        if pobj.phys_reg != "None":
            pobj.set_param("rtp_physio", self.rtp_objs["TTLPHYSIO"])

        # Noise ROI masks
        for use_reg, mask_param, mask_f in (
            (pobj.GS_reg, "GS_mask", self.GSR_mask),
            (pobj.WM_reg, "WM_mask", self.WM_orig),
            (pobj.Vent_reg, "Vent_mask", self.Vent_orig),
        ):
            if not use_reg:
                continue

            if Path(mask_f).is_file():
                pobj.set_param(mask_param, mask_f)
            elif not ignore_error and hasattr(pobj, f"ui_{mask_param}_lnEd"):
                # Ask the user to select a mask file
                ret = pobj.set_param(
                    mask_param,
                    self.work_dir,
                    getattr(pobj, f"ui_{mask_param}_lnEd").setText,
                )
                if ret == -1:
                    return -1

        if pobj.reg_retro_proc:
            self.max_watch_wait = np.max(
                [self.max_watch_wait, pobj.wait_num / 2]
            )

        return 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _is_running_dcm2nii(self):
        """Return True (with an error popup) if dcm2niix is running."""
        if len(_find_processes("dcm2niix")):
            errmsg = "DICOM to NIfTI conversion is still in progress."
            self._logger.error(errmsg)
            self.err_popup(errmsg)
            return True

        return False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ready_to_run(self):
        """Get ready to run the process.

        Returns
        -------
        The first module of the process chain (None if RTP is disabled).
        """
        self._logger.debug("ready_to_run")

        # Disable the UI to block parameter changes
        if self.enable_RTP > 0 and self.main_win is not None:
            self.ui_setEnabled(False)

        # --- Ready RTP -------------------------------------------------------
        proc_chain = None
        if self.enable_RTP > 0:
            # Use the maximum scan length of REGRESS (a realistic scan
            # length, set by the user) for all modules, so that the buffers
            # are allocated once before the scan.
            if "REGRESS" in self.rtp_objs:
                max_len = self.rtp_objs["REGRESS"].max_scan_length
                for proc, pobj in self.rtp_objs.items():
                    if proc == "REGRESS" or \
                            not hasattr(pobj, "max_scan_length"):
                        continue
                    if proc == "VOLREG":
                        pobj.set_param("max_scan_length", max_len)
                    else:
                        pobj.max_scan_length = max_len

            # Connect the enabled RTP modules into a chain
            last_proc = None
            for proc in ["WATCH", "VOLREG", "TSHIFT", "SMOOTH", "REGRESS"]:
                if proc not in self.rtp_objs:
                    continue

                pobj = self.rtp_objs[proc]
                if not pobj.enabled:
                    continue

                if proc_chain is None:
                    proc_chain = pobj
                else:
                    last_proc.next_proc = pobj

                # Wait longer than the expected interval of the input before
                # chkRunTimerEvent regards the scan as ended.
                if proc == "TSHIFT":
                    self.max_watch_wait = max(
                        pobj.TR * 3, self.max_watch_wait
                    )

                elif proc == "REGRESS":
                    if pobj.GS_reg or pobj.WM_reg or pobj.Vent_reg:
                        # Noise ROI signals are taken from the image before
                        # smoothing.
                        if self.rtp_objs["VOLREG"].enabled:
                            mask_src = self.rtp_objs["VOLREG"]
                        elif self.rtp_objs["TSHIFT"].enabled:
                            mask_src = self.rtp_objs["TSHIFT"]
                        else:
                            mask_src = self.rtp_objs["WATCH"]
                        pobj.set_param("mask_src_proc", mask_src)

                    self.max_watch_wait = max(
                        pobj.wait_num * 0.5, self.max_watch_wait
                    )

                last_proc = pobj

            if proc_chain is None:
                errmsg = "No RTP module is enabled."
                self._logger.error(errmsg)
                self.err_popup(errmsg)
                self.ui_setEnabled(True)
                return

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
                    self.main_win.chbShowPhysio.setCheckState(2)

            # Reset process chain status
            proc_chain.end_reset()

            # ready_proc of each module calls ready_proc of the next one.
            if not proc_chain.ready_proc():
                self.ui_setEnabled(True)
                return

            # Log the process parameters
            log_str = "RTP parameters:\n"
            rtp = proc_chain
            while rtp is not None:
                log_str += f"# {type(rtp).__name__}\n"
                for k, v in rtp.get_params().items():
                    if k == "ignore_init" and v == 0:
                        continue
                    log_str += f"#     {k}: {v}\n"

                rtp = rtp.next_proc

            self._logger.info(log_str.rstrip())

        # --- Ready application -----------------------------------------------
        if self.run_extApp:
            if not self.isAlive_extApp():
                self.boot_extApp()
                if not self.isAlive_extApp():
                    errmsg = "No response from the external application."
                    self._logger.error(errmsg)
                    self.err_popup(errmsg)
                    self.ui_setEnabled(True)
                    return

        else:
            # Start a new signal file
            if not self.sig_save_file.parent.is_dir():
                os.makedirs(self.sig_save_file.parent)
            with open(self.sig_save_file, "w") as fd:
                fd.write("Time,Index,Value\n")

        if self.main_win is not None:
            # Start the running-status check timer. Timers and QThread work
            # only with a QApplication.
            self.chk_run_timer.start(1000)

            # Wait for the scan onset signal
            if (
                "TTLPHYSIO" in self.rtp_objs
                and self.rtp_objs["TTLPHYSIO"] is not None
            ):
                self.rtp_objs["TTLPHYSIO"].standby_scan()

            self._scanning = False
            self._wait_start = True

            self.th_wait_onset = QtCore.QThread()
            self.wait_onset = RtpApp.WAIT_ONSET(
                self, self.rtp_objs["TTLPHYSIO"], self.extApp_sock
            )
            self.wait_onset.moveToThread(self.th_wait_onset)
            self.th_wait_onset.started.connect(self.wait_onset.run)
            self.wait_onset.finished.connect(self.th_wait_onset.quit)
            self.th_wait_onset.start()

            self.ui_ready_btn.setText("Waiting for scan start ...")
            self.ui_ready_btn.setEnabled(False)
            self.ui_manStart_btn.setEnabled(True)
            self.ui_quit_btn.setEnabled(True)

        self._isReadyRun = True
        return proc_chain

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def manual_start(self):
        """Start the scan manually."""
        self.scan_onset = time.time()
        self._scanning = True
        self._wait_start = False

        if (
            "TTLPHYSIO" in self.rtp_objs and
            self.rtp_objs["TTLPHYSIO"] is not None
        ):
            self.rtp_objs["TTLPHYSIO"].scan_onset = self.scan_onset
            self.rtp_objs["TTLPHYSIO"].release_standby_scan()

        if self.extApp_sock is not None:
            _send_scan_start(self.extApp_sock, self.scan_onset)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end_run(self, quit_btn=False, scan_name=None):
        """End the RTP run and save the results.

        Returns
        -------
        save_fnames : dict
            {description: saved file path}
        """
        # end_run can also be called by chkRunTimerEvent. Do not run it twice
        # at the same time.
        if self._isRunning_end_run_proc:
            return

        self._isRunning_end_run_proc = True

        self.chk_run_timer.stop()

        if quit_btn:
            self._logger.info("Quit button is pressed.")

        if self.main_win is not None:
            self.ui_ready_btn.setText("Ready")
            self.ui_ready_btn.setEnabled(False)
            self.ui_quit_btn.setEnabled(False)
            self.ui_manStart_btn.setEnabled(False)
            self.main_win.repaint()
            QtWidgets.QApplication.instance().processEvents()

        try:
            save_fnames = {}

            self.end_proc()

            # Abort the WAIT_ONSET thread if it is running
            if (
                hasattr(self, "th_wait_onset")
                and self.th_wait_onset.isRunning()
            ):
                self.wait_onset.abort = True
                if not self.th_wait_onset.wait(1):
                    self.wait_onset.finished.emit()
                    self.th_wait_onset.wait()

            # Send 'END' message to the external application
            if self.isAlive_extApp():
                if self.send_extApp("END;".encode()):
                    recv = self.recv_extApp(timeout=3)
                    if recv is not None:
                        self._logger.info(f"Recv {recv.decode()}")

            if self.enable_RTP > 0:
                if scan_name is None or scan_name is False:
                    if (
                        self.rtp_objs["WATCH"].enabled
                        and self.rtp_objs["WATCH"].scan_name is not None
                    ):
                        scan_name = self.rtp_objs["WATCH"].scan_name
                    else:
                        scan_name = f"scan_{time.strftime('%Y%m%d%H%M')}"

                # Save the parameters
                all_params = {}
                for rtp in ("WATCH", "VOLREG", "TSHIFT", "SMOOTH", "REGRESS"):
                    if rtp in self.rtp_objs and self.rtp_objs[rtp].enabled:
                        all_params[rtp] = self.rtp_objs[rtp].get_params()

                all_params[self.__class__.__name__] = self.get_params()

                save_dir = self.work_dir / "log"
                if not save_dir.is_dir():
                    save_dir.mkdir()

                save_f = save_dir / f"rtp_params_{scan_name}.txt"
                with open(save_f, "w") as fd:
                    for rtp, opt_dict in all_params.items():
                        fd.write(f"# {rtp}\n")
                        for k in sorted(opt_dict.keys()):
                            fd.write(f"{k}: {opt_dict[k]}\n")
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

                # Reset the process chain from its first module
                root_proc = next(
                    obj for rtp, obj in self.rtp_objs.items()
                    if rtp != "TTLPHYSIO" and obj.enabled
                )
                out_files = root_proc.end_reset()
                if out_files is not None:
                    save_fnames.update(out_files)

        except Exception as e:
            self._logger.error(str(e) + "\n" + traceback.format_exc())

        if self.main_win is not None:
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
        """Display images in AFNI to check them.

        Parameters
        ----------
        base : str, optional
            Underlay image: 'anat' (aligned anatomy) or 'func' (reference
            function image).
        ovl : str, optional
            Overlay image: 'func', 'wm', 'vent', 'roi', 'RTPmask', or
            'GSRmask'.
        """
        # With distortion correction, the masks in func_orig space are
        # '*_invsdc' files. The anatomy is aligned to the corrected function,
        # so the corrected function and masks are shown on it.
        sdc_applied = bool(self.sdc_invwarp) and \
            Path(self.sdc_invwarp).is_file()

        base_img = None
        if base == "anat":
            base_img = self.alAnat
        elif base == "func":
            base_img = self.func_orig

        ovl_mask_attrs = {
            "wm": "WM_orig",
            "vent": "Vent_orig",
            "roi": "ROI_orig",
            "RTPmask": "RTP_mask",
            "GSRmask": "GSR_mask",
        }
        ovl_img = None
        if ovl == "func":
            if base == "anat" and sdc_applied:
                ovl_img = self.func_orig_sdc
            else:
                ovl_img = self.func_orig

        elif ovl in ovl_mask_attrs and base in ("anat", "func"):
            ovl_img = getattr(self, ovl_mask_attrs[ovl])
            if base == "anat" and sdc_applied:
                ovl_img = (
                    Path(ovl_img).parent /
                    Path(ovl_img).name.replace("_invsdc", "")
                )

        if base is None and ovl is None:
            # 'Open AFNI' button: only boot AFNI in work_dir
            ses_dir = Path(self.work_dir)
        else:
            if base_img is None or not Path(base_img).is_file():
                errmsg = f"Base image ({base}) is not set or found: {base_img}"
                self.err_popup(errmsg)
                return
            ses_dir = Path(base_img).parent

            if ovl_img is None or not Path(ovl_img).is_file():
                errmsg = (
                    f"Overlay image ({ovl}) is not set or found: {ovl_img}"
                )
                self.err_popup(errmsg)
                return

        # Boot AFNI if it is not running. plugout_drive needs an AFNI started
        # with -yesplugouts (boot_afni adds it).
        if len(_find_processes("afni", ["-yesplugouts"])) == 0:
            boot_afni(
                main_win=self.main_win,
                boot_dir=ses_dir,
                TRUSTHOST=self.AFNIRT_TRUSTHOST,
            )

        if base is None and ovl is None:
            return

        # Image window sizes adjusted to the volume shape
        baseWinSize = 480  # height of the axial image window
        bimg = nib.load(base_img)
        vsize = np.diag(bimg.affine)[:3]
        if np.any(vsize == 0):
            # Oblique or rotated affine
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

        # Put the cursor at the center of the mask
        xyz = None
        if ovl in ovl_mask_attrs:
            ovl_v = nib.load(ovl_img).get_fdata()
            if ovl_v.ndim > 3:
                ovl_v = ovl_v[:, :, :, 0]
            if np.any(ovl_v != 0):
                ijk = np.mean(np.argwhere(ovl_v != 0), axis=0)
                ijk = np.concatenate([ijk, [1]])
                xyz = np.dot(nib.load(ovl_img).affine, ijk)[:3]

        # Drive AFNI with plugout_drive
        le = 800  # left end
        tp = 500  # top
        cmd = "plugout_drive"
        cmd += f" -com 'SWITCH_SESSION {Path(ses_dir).name}'"
        cmd += " -com 'RESCAN_THIS'"
        cmd += f" -com 'SWITCH_UNDERLAY {Path(base_img).name}'"
        cmd += f" -com 'SWITCH_OVERLAY {Path(ovl_img).name}'"
        cmd += " -com 'SEE_OVERLAY +'"
        cmd += " -com 'OPEN_WINDOW A.axialimage"
        cmd += f" geom={wh_ax[0]}x{wh_ax[1]}+{le}+{tp} opacity=6'"
        cmd += " -com 'OPEN_WINDOW A.sagittalimage"
        if le + wh_ax[0] + wh_sg[0] + wh_cr[0] - 100 > 1920:
            # Too wide for the screen. Put the coronal window below the
            # sagittal one.
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

        if xyz is not None:
            cmd += f" -com 'SET_SPM_XYZ {xyz[0]} {xyz[1]} {xyz[2]}'"
        cmd += " -quit"
        subprocess.run(cmd, shell=True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def delete_file(self, attr, keepfile=False):
        """Delete a file parameter's file, or all processed files.

        Parameters
        ----------
        attr : str
            Name of the file parameter, or 'AllProc' for all processed
            files.
        keepfile : bool, optional
            Only unset the parameter without deleting the file.
            The default is False.
        """
        if not hasattr(self, attr) and attr != "AllProc":
            return

        if attr != "AllProc":
            ff = Path(getattr(self, attr))
            if not ff.is_file():
                self.set_param(attr, "")
                return

        if not keepfile:
            # Confirmation dialog
            msgBox = QtWidgets.QMessageBox()
            msgBox.setIcon(QtWidgets.QMessageBox.Question)
            if attr != "AllProc":
                msgBox.setText(f"Are you sure you want to delete {ff}?")
                msgBox.setWindowTitle(f"Delete {attr} file")
            else:
                msgBox.setText(
                    "Are you sure you want to delete ALL processed files?"
                )
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

            # All processed images
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

            # Intermediate files in the make_masks output directory
            # (work_dir/RTP). Nothing is searched if work_dir is not set.
            work_dir = Path(self.work_dir)
            if str(self.work_dir) == "":
                out_dir = None
            elif work_dir.name != "RTP":
                out_dir = work_dir / "RTP"
            else:
                out_dir = work_dir

            brain_anat_orig = Path(getattr(self, "brain_anat_orig", ""))
            if (
                not brain_anat_orig.is_file()
                and self.anat_orig
                and out_dir is not None
            ):
                # Brain image name made by make_masks from anat_orig
                anat_orig = Path(self.anat_orig)
                anat_prefix = anat_orig.name.replace(
                    "".join(anat_orig.suffixes[-2:]), ""
                )
                anat_prefix = anat_prefix.replace("+orig", "").replace(
                    "+tlrc", ""
                )
                brain_anat_orig = out_dir / (anat_prefix + "_Brain.nii.gz")
                if brain_anat_orig.is_file():
                    self.brain_anat_orig = brain_anat_orig
                    delFiles["brain_anat_orig"] = brain_anat_orig

            if brain_anat_orig.is_file():
                WM_anat = (
                    brain_anat_orig.parent
                    / brain_anat_orig.name.replace("Brain", "WM")
                )
                if WM_anat.is_file():
                    delFiles["WM_anat"] = WM_anat

                Vent_anat = (
                    brain_anat_orig.parent
                    / brain_anat_orig.name.replace("Brain", "Vent")
                )
                if Vent_anat.is_file():
                    delFiles["Vent_anat"] = Vent_anat

                # e.g., anat_Brain_alFunc_mat.aff12.1D
                alfunc_pat = brain_anat_orig.name.replace(
                    ".nii.gz", ""
                ).replace(".nii", "") + "_alFunc"
                for rmf in brain_anat_orig.parent.glob(f"*{alfunc_pat}*"):
                    attr = rmf.stem.replace(alfunc_pat, "alFunc")
                    delFiles[attr] = rmf

            if out_dir is not None:
                for rmf in out_dir.glob("template2orig_*"):
                    attr = rmf.stem.replace("template2orig_", "")
                    delFiles[attr] = rmf

            if hasattr(self, "func_orig") and Path(self.func_orig).is_file():
                fbase = re.sub(r"\+.*", "", Path(self.func_orig).name)
                func_mask = (
                    Path(self.func_orig).parent / f"automask_{fbase}.nii.gz"
                )
                if func_mask.is_file():
                    delFiles["func_mask"] = func_mask

        for attr, ff in delFiles.items():
            if not keepfile:
                if ".HEAD" in ff.suffixes or ".BRIK" in ff.suffixes:
                    # Delete both the AFNI .HEAD and .BRIK files
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

    # --- Signal plot ---------------------------------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    class PlotROISignal(QtCore.QObject):
        """Window to plot the ROI signal in real time."""

        finished = QtCore.pyqtSignal()

        def __init__(self, root, num_ROIs=1, roi_labels=[], main_win=None):
            super().__init__()

            self.root = root
            self.main_win = main_win
            self.abort = False

            self.plt_win = MatplotlibWindow()
            self.plt_win.setWindowTitle("ROI signal")

            # Place the window to the right of the main window
            if main_win is not None:
                main_geom = main_win.geometry()
                x = main_geom.x() + main_geom.width() + 10
                y = main_geom.y() + 735
            else:
                x, y = (0, 0)
            win_height = 80 + 70 * num_ROIs
            self.plt_win.setGeometry(x, y, 500, win_height)

            self.roi_labels = roi_labels
            self._axes = self.plt_win.canvas.figure.subplots(num_ROIs, 1)
            if num_ROIs == 1:
                self._axes = [self._axes]

            self.plt_win.canvas.figure.subplots_adjust(
                left=0.15, bottom=0.18, right=0.95, top=0.96, hspace=0.35
            )
            self._color_cycle = plt.get_cmap("tab10")

            self.reset_plot()

            self.plt_win.show()
            self.plt_win.canvas.draw()

            # Qt widgets must be updated in the GUI thread. A timer in the
            # GUI thread polls the ROI signal and updates the plot.
            self._plt_xi = []
            self._running = False
            self._timer = QtCore.QTimer(self)
            self._timer.setInterval(100)
            self._timer.timeout.connect(self._update_plot)

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
            """Start updating the plot."""
            self._plt_xi = self.root._plt_xi.copy()
            self._running = True
            self._timer.start()

        # ---------------------------------------------------------------------
        def isRunning(self):
            return self._running

        # ---------------------------------------------------------------------
        def _update_plot(self):
            main_closed = (
                self.main_win is not None and not self.main_win.isVisible()
            )
            if not self.plt_win.isVisible() or self.abort or main_closed:
                self.end_thread()
                return

            if len(self.root._plt_xi) == len(self._plt_xi):
                return

            try:
                # Plot signal
                plt_xi = self.root._plt_xi.copy()
                self._plt_xi = plt_xi
                plt_roi_sig = self.root._roi_sig
                for ii, ax in enumerate(self._axes):
                    ll = min(len(plt_xi), len(plt_roi_sig[ii]))
                    if ll == 0:
                        continue

                    y = plt_roi_sig[ii][:ll]
                    self._ln[ii][0].set_data(plt_xi[:ll], y)

                    # Adjust the y range, ignoring outliers beyond 4 SD
                    ax.relim()
                    if np.sum(~np.isnan(y)) > 10:
                        yl_orig = np.array(ax.get_ylim())
                        yl = yl_orig.copy()
                        sd = np.nanstd(y)
                        mu = np.nanmean(y)
                        if np.nanmin(y) < yl[0]:
                            yl[0] = max(np.nanmin(y), mu - 4 * sd)

                        if np.nanmax(y) > yl[1]:
                            yl[1] = min(np.nanmax(y), mu + 4 * sd)

                        if np.any(yl != yl_orig):
                            ax.set_ylim(yl)

                    ax.autoscale_view()

                    # Extend the x range in steps of 10
                    xl = ax.get_xlim()
                    if (plt_xi[-1] // 10 + 1) * 10 > xl[1]:
                        ax.set_xlim([0, (plt_xi[-1] // 10 + 1) * 10])
                self.plt_win.canvas.draw()

            except IndexError:
                pass

            except Exception as e:
                self.root.errmsg(str(e), no_pop=True)
                sys.stdout.flush()

        # ---------------------------------------------------------------------
        def end_thread(self):
            if not self._running:
                return
            self._running = False
            self._timer.stop()

            if self.plt_win.isVisible():
                self.plt_win.close()

            self.finished.emit()

            if hasattr(self.root, "ui_showROISig_cbx"):
                self.root.ui_showROISig_cbx.setCheckState(0)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def open_ROISig_plot(self, num_ROIs=1, roi_labels=[]):
        """Open the ROI signal plot."""
        if hasattr(self, "pltROISig") and self.pltROISig.isRunning():
            return

        self.pltROISig = RtpApp.PlotROISignal(
            self,
            num_ROIs=num_ROIs,
            roi_labels=roi_labels,
            main_win=self.main_win,
        )
        self.pltROISig.run()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def close_ROISig_plot(self):
        """Close the ROI signal plot."""
        if hasattr(self, "pltROISig"):
            pltROISig = self.pltROISig
            del self.pltROISig
            pltROISig.end_thread()

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
        Save the ROI signals extracted in RTP.

        Parameters
        ----------
        save_f : Path or str
            Output file path.
        ti : list of int
            Time (TR) indices.
        roi_sig : array-like
            ROI signals; Nr. ROIs x Nr. TRs.
        roi_label : list of str, optional
            ROI names. A, B, C, ... are used if it has fewer names than
            ROIs. The default is [].
        """
        if len(roi_label) < len(roi_sig):
            labs = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            roi_label = [labs[ii] for ii in range(len(roi_sig))]

        # ROI signals can have different lengths; shorter ones are padded
        # with NaN.
        n_rows = max([len(sig) for sig in roi_sig], default=0)
        savedf = pd.DataFrame(
            {"TR": pd.Series(list(ti[:n_rows]))}
            | {
                lab: pd.Series(list(sig), dtype=float)
                for lab, sig in zip(roi_label, roi_sig)
            }
        )

        savedf.to_csv(save_f, index=False)

    # --- Session control -----------------------------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    class WAIT_ONSET(QtCore.QObject):
        """Wait for the scan onset and notify the external application.

        Run in a QThread. The onset time is read from onsetObj
        (RtpTTLPhysio).
        """

        finished = QtCore.pyqtSignal()

        def __init__(self, parent, onsetObj=None, extApp_sock=None):
            super().__init__()
            self.parent = parent
            self.onsetObj = onsetObj
            self.extApp_sock = extApp_sock
            self.abort = False

        def run(self):
            onset_time = 0
            while not self.abort:
                if self.onsetObj is not None and self.onsetObj.available:
                    onset_time = self.onsetObj.scan_onset
                    if onset_time != 0:
                        break
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
                _send_scan_start(self.extApp_sock, onset_time)

            self.finished.emit()

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
        """Check the running status every second and end the session if
        requested by the external application or no new image arrives."""
        if self.extApp_sock is not None:
            # Check a message from the external application
            msg = self.recv_extApp(timeout=0.001)
            if msg is not None and "END_SESSION" in msg.decode():
                if self.ui_quit_btn.isEnabled():
                    self._logger.info("Recv END_SESSION. End session.")
                    if not self._isRunning_end_run_proc:
                        self.end_run()
                else:
                    self.ui_ready_btn.setEnabled(False)
                return

        # Check the time since the last image in WATCH
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

        # Update the button text when the scan has started
        if (
            hasattr(self, "ui_ready_btn")
            and "waiting" in self.ui_ready_btn.text().lower()
        ):
            if self._scanning:
                self.ui_ready_btn.setText("Session is running")
                self.ui_manStart_btn.setEnabled(False)

        # Schedule the next check
        self.chk_run_timer.start(1000)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def keyPressEvent(self, event):
        # The 'T' key starts the scan while waiting for the scan onset.
        if self._wait_start and event.key() == QtCore.Qt.Key_T:
            self.manual_start()

    # --- Communication with an external application via RTP_SERVE ----------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def boot_extApp(self, cmd="", timeout=30):
        """
        Start an external application with the RTP_SERVE server.

        A running process of the same application is killed first.

        Parameters
        ----------
        cmd : str, optional
            Command line to start the application. The default is
            self.extApp_cmd.
        timeout : float, optional
            Timeout (s) to wait for the server to start. The default is 30.

        Returns
        -------
        0 on success (self.extApp_addr and self.extApp_proc are set), -1 on
        failure.
        """
        if len(cmd) == 0:
            cmd = self.extApp_cmd

        if len(cmd.split()) == 0:
            errmsg = "No command is set for the external application."
            self._logger.error(errmsg)
            self.err_popup(errmsg)
            return -1

        if self.main_win is not None:
            cmdstr = Path(cmd.split()[0]).name
            progress_bar = DlgProgressBar(
                title=f"Run {cmdstr} ...", parent=self.main_win
            )
            progress_bar.btnCancel.setVisible(False)
            progress_bar.set_msgTxt(f"Run {cmdstr} ...")
            time.sleep(0.1)  # Give the UI time to update the dialog
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

        # Kill the process this object started before
        self._kill_extApp_proc(signal.SIGKILL)

        # --- Start the external application ---
        if progress_bar is not None:
            progress_bar.add_desc(f"Run {cmd} ...")
            time.sleep(0.1)  # Give the UI time to update the dialog

        extApp_addr, extApp_proc = boot_RTP_SERVE_app(cmd, timeout=timeout)

        if extApp_addr is None:
            # extApp_proc holds the error message
            errmsg = "Failed to run the external application.\n" + extApp_proc
            self._logger.error(errmsg)
            self.err_popup(errmsg)
            if progress_bar is not None:
                progress_bar.close()
            return -1

        if progress_bar is not None:
            progress_bar.add_desc(" done.")

        self.set_param("extApp_addr", extApp_addr)
        # Keep the process and its group ID to kill it later
        self.extApp_proc = extApp_proc
        self._extApp_pgid = os.getpgid(extApp_proc.pid)
        self.connect_extApp()

        if progress_bar is not None:
            progress_bar.close()

        return 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _kill_extApp_proc(self, sig=signal.SIGTERM):
        """Kill the external application process started by boot_extApp.

        Its process group is killed if it leads its own group. Otherwise
        (e.g., it shares the group of this process), only the process itself
        is killed.
        """
        proc = self.extApp_proc
        self.extApp_proc = None
        pgid = self._extApp_pgid
        self._extApp_pgid = None

        # Popen keeps an unreaped child, so its PID cannot be reused while
        # poll() returns None.
        if proc is None or proc.poll() is not None:
            return

        try:
            if pgid == proc.pid and pgid != os.getpgrp():
                os.killpg(pgid, sig)
            else:
                proc.send_signal(sig)
            proc.wait(timeout=5)
        except Exception as e:
            self._logger.error(f"Failed to kill the external app: {e}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def connect_extApp(self, no_err_pop=False):
        """Connect to the external application socket.

        Returns
        -------
        True if already connected, 0 if newly connected, False if no address
        is set, -1 if the connection failed.
        """
        if self.extApp_addr is None:
            errmsg = "No address is set for the external application."
            self._logger.error(errmsg)
            self.err_popup(errmsg)
            return False

        if self.extApp_sock is not None:
            if self.isAlive_extApp():
                return True

            try:
                self.extApp_sock.close()
            except Exception:
                pass

        self.extApp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.extApp_sock.connect(self.extApp_addr)
            self.extApp_sock.settimeout(self.extApp_sock_timeout)
        except ConnectionRefusedError:
            errmsg = f"Failed to connect to {self.extApp_addr}"
            self._logger.error(errmsg)
            if not no_err_pop:
                self.err_popup(errmsg)
            return -1

        return 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def isAlive_extApp(self):
        """Check if the external application responds."""
        if self.extApp_addr is None:
            return False

        if self.extApp_sock is None:
            ret = self.connect_extApp(no_err_pop=True)
            if ret != 0:
                return False

        try:
            # Clear the receive buffer
            self.extApp_sock.settimeout(3)
            while True:
                try:
                    recv = self.extApp_sock.recv(1024)
                    if len(recv) == 0:
                        break
                except Exception:
                    break

            self.extApp_sock.send("IsAlive?;".encode())
            recv = self.extApp_sock.recv(1024)
            self.extApp_sock.settimeout(self.extApp_sock_timeout)
            return "Yes." in recv.decode()

        except Exception as e:
            self._logger.error(str(e) + "\n" + traceback.format_exc())
            if isinstance(e, BrokenPipeError):
                self.extApp_sock = None
            return False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def send_extApp(self, data, pkl=False, no_err_pop=False):
        """
        Send data to the external RTP_SERVE application.

        Parameters
        ----------
        data : bytes
            Message to send.
        pkl : bool, optional
            Send data as a pickled binary.
        no_err_pop : bool, optional
            Do not show an error popup.

        Returns
        -------
        True if sent.
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
            errmsg = "No connection to the external application."
            self._logger.debug(errmsg)
            if not no_err_pop:
                self.err_popup(errmsg)
            return False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def recv_extApp(self, bufsize=1024, timeout=None):
        """
        Receive data from the external RTP_SERVE application.

        Parameters
        ----------
        bufsize : int, optional
            Receive buffer size.
        timeout : float, optional
            Timeout (s) to wait for a response. The default is None (no
            timeout).

        Returns
        -------
        bytes or None
            Message from the server. None if nothing is received.
        """
        if self.extApp_sock is None:
            return None

        self.extApp_sock.settimeout(timeout)

        try:
            received = self.extApp_sock.recv(bufsize)
        except socket.timeout:
            received = None

        except BrokenPipeError:
            self.extApp_sock = None
            self._logger.error("No connection to the external application.")
            received = None

        except Exception:
            traceback.print_exc()
            received = None

        # The socket is None after a broken connection
        if self.extApp_sock is not None:
            self.extApp_sock.settimeout(self.extApp_sock_timeout)

        return received

    # --- user interface ------------------------------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_param(
        self, attr, val=None, reset_fn=None, echo=False, unk_warning=True
    ):
        """
        Set a parameter and update the UI.

        Parameters
        ----------
        attr : str
            Parameter name.
        val : optional
            Parameter value.
        reset_fn : function, optional
            Function to set the value to the UI widget. It is given when
            set_param is called from the UI. None means a call from the
            program (e.g., load_parameters).
        echo : bool, optional
            Print the set value.
        unk_warning : bool, optional
            Log an error for an unknown parameter.
        """
        self._logger.debug(f"set_param: {attr} = {val}")

        if attr == "aseg_oirg":  # Misspelled name in old parameter files
            attr = "aseg_orig"

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
                    self.ui_extApp_addr_lnEd.blockSignals(True)
                    self.ui_extApp_addr_lnEd.setText("{}:{}".format(*val))
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

                if self.extApp_sock is not None and (
                    self.extApp_addr is None
                    or tuple(self.extApp_addr[:2]) != val
                ):
                    # The address is changed. Close the current socket.
                    self.extApp_sock.close()
                    self.extApp_sock = None

        elif attr == "extApp_isAlive":
            # Check the connection and show the result with reset_fn
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
                    "File name to save the real-time signal",
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

        elif attr in _FILE_PARAM_LABELS:
            if reset_fn is not None:
                # Called from the UI. Select the file with a dialog.
                if val is not None and os.path.isdir(val):
                    startdir = val
                elif "template" in attr and Path(self.template).is_file():
                    startdir = Path(self.template).parent
                else:
                    startdir = self.work_dir

                dlg_msg = f"RtpApp: Select {_FILE_PARAM_LABELS[attr]}"
                filt = "*.BRIK* *.nii*;;*.*"
                fname = self.select_file_dlg(
                    dlg_msg, startdir, filt, parent=self.rtp_gui
                )
                if fname[0] == "":
                    return -1

                val = fname[0]
                reset_fn(val)
                val = Path(val).absolute()
            else:
                # Unset the parameter if the file is not found.
                if val is None:
                    val = ""
                elif _strip_subbrick_selector(
                    Path(val).absolute()
                ).is_file():
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

        elif not hasattr(self, attr):
            # Other existing attributes (e.g., max_watch_wait) are just set
            if unk_warning:
                self._logger.error(f"{attr} is an unknown parameter.")
            return

        setattr(self, attr, val)
        if echo:
            print(f"{self.__class__.__name__}.{attr} = {getattr(self, attr)}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _add_file_row(
        self, layout, row, label, attr, set_dir, show_value=True,
        highlight=False,
    ):
        """Add a row of [label, file path, Set, Unset] widgets for a file
        parameter to a grid layout.

        Parameters
        ----------
        layout : QGridLayout
            Layout to add the widgets.
        row : int
            Row index in the layout.
        label : str
            Label text.
        attr : str
            File parameter name.
        set_dir : function
            Returns the value given to set_param when Set is clicked (the
            start directory of the file dialog).
        show_value : bool, optional
            Show the current value in the path field.
        highlight : bool, optional
            Color the Set button.

        Returns
        -------
        (label, path field, Set button, Unset button) widgets
        """
        lb = QtWidgets.QLabel(label)
        layout.addWidget(lb, row, 0)

        lnEd = QtWidgets.QLineEdit()
        if show_value:
            lnEd.setText(str(getattr(self, attr)))
        lnEd.setReadOnly(True)
        lnEd.setStyleSheet("border: 0px none;")
        layout.addWidget(lnEd, row, 1)

        set_btn = QtWidgets.QPushButton("Set")
        set_btn.clicked.connect(
            lambda: self.set_param(attr, set_dir(), lnEd.setText)
        )
        if highlight:
            set_btn.setStyleSheet(_SET_BTN_STYLE)
        layout.addWidget(set_btn, row, 2)

        del_btn = QtWidgets.QPushButton("Unset")
        del_btn.clicked.connect(
            lambda: self.delete_file(attr, keepfile=True)
        )
        layout.addWidget(del_btn, row, 3)

        return lb, lnEd, set_btn, del_btn

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ui_set_param(self):
        """Make the UI widgets.

        Returns
        -------
        ui_rows : list of tuples
            Widget rows to be placed by RtpGUI.
        """
        ui_rows = []
        self.ui_objs = []  # Widgets disabled while running

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

        # Values given to set_param by the Set buttons of the file rows
        # (the start directory of the file dialog)
        def default_dir():
            return ""

        def work_dir():
            return self.work_dir

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
        (
            _, self.ui_anat_orig_lnEd, self.ui_anat_orig_btn,
            self.ui_anat_orig_del_btn
        ) = self._add_file_row(
            RefImg_gLayout, 0, "Anatomy image :", "anat_orig", default_dir,
            show_value=False, highlight=True,
        )
        if self.anat_orig and Path(self.anat_orig).exists():
            self.ui_anat_orig_lnEd.setText(str(self.anat_orig))

        # -- function orig image --
        (
            _, self.ui_func_orig_lnEd, self.ui_func_orig_btn,
            self.ui_func_orig_del_btn
        ) = self._add_file_row(
            RefImg_gLayout, 1, "Base function image : ", "func_orig",
            default_dir, show_value=False, highlight=True,
        )

        # -- parameter reference image --
        (
            _, self.ui_func_param_ref_lnEd, self.ui_param_ref_btn,
            self.ui_param_ref_del_btn
        ) = self._add_file_row(
            RefImg_gLayout, 2, "fMRI parameter reference : ",
            "func_param_ref", default_dir, show_value=False, highlight=True,
        )
        # endregion

        # region: Fieldmap PEPOLAR images group --
        self.ui_Fmap_grpBx = QtWidgets.QGroupBox("Fieldmap PEPOLAR images")
        Fmap_gLayout = QtWidgets.QGridLayout(self.ui_Fmap_grpBx)
        self.ui_preprocessing_fLayout.addRow(self.ui_Fmap_grpBx)
        self.ui_objs.append(self.ui_Fmap_grpBx)

        # -- Fieldmap with the same phase encoding --
        (
            _, self.ui_fieldmap_posi_lnEd, self.ui_fieldmap_posi_btn,
            self.ui_fieldmap_posi_del_btn
        ) = self._add_file_row(
            Fmap_gLayout, 0, "Same PE direction :", "fieldmap_posi",
            default_dir, show_value=False, highlight=True,
        )

        # -- Fieldmap with the opposite phase encoding --
        (
            _, self.ui_fieldmap_nega_lnEd, self.ui_fieldmap_nega_btn,
            self.ui_fieldmap_nega_del_btn
        ) = self._add_file_row(
            Fmap_gLayout, 1, "Opposite PE direction :", "fieldmap_nega",
            default_dir, show_value=False, highlight=True,
        )
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
        self.ui_CreateMasks_btn.setStyleSheet(_SET_BTN_STYLE)
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

        # region: Check masks group ---
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

        self.ui_Template_grpBx = QtWidgets.QGroupBox("Template images")
        Template_gLayout = QtWidgets.QGridLayout(self.ui_Template_grpBx)
        template_fLayout.addWidget(self.ui_Template_grpBx)
        self.ui_objs.append(self.ui_Template_grpBx)

        # The file dialog for a template file starts from the directory of
        # the current file.
        # -- Template image --
        (
            _, self.ui_template_lnEd, self.ui_template_btn,
            self.ui_template_del_btn
        ) = self._add_file_row(
            Template_gLayout, 0, "Template brain :", "template",
            lambda: os.path.dirname(self.ui_template_lnEd.text()),
        )

        # -- ROI on template --
        (
            self.ui_ROI_template_lb, self.ui_ROI_template_lnEd,
            self.ui_ROI_template_btn, self.ui_ROI_template_del_btn
        ) = self._add_file_row(
            Template_gLayout, 1, "ROI on template :", "ROI_template",
            lambda: os.path.dirname(self.ui_ROI_template_lnEd.text()),
        )

        # ROI resampling mode
        var_ROI_resample_lb = QtWidgets.QLabel("ROI resampling :")
        Template_gLayout.addWidget(var_ROI_resample_lb, 2, 0)

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
        Template_gLayout.addWidget(self.ui_ROI_resample_cmbBx, 2, 1)

        # -- WM on template --
        (
            self.ui_WM_template_lb, self.ui_WM_template_lnEd,
            self.ui_WM_template_btn, self.ui_WM_template_del_btn
        ) = self._add_file_row(
            Template_gLayout, 3, "White matter on template :", "WM_template",
            lambda: os.path.dirname(self.ui_WM_template_lnEd.text()),
        )

        # -- Ventricle on template --
        (
            self.ui_Vent_template_lb, self.ui_Vent_template_lnEd,
            self.ui_Vent_template_btn, self.ui_Vent_template_del_btn
        ) = self._add_file_row(
            Template_gLayout, 4, "Ventricle on template :", "Vent_template",
            lambda: os.path.dirname(self.ui_Vent_template_lnEd.text()),
        )

        # --- Processed image tab ---------------------------------------------
        self.ui_procImgTab = QtWidgets.QWidget()
        self.ui_top_tabs.addTab(self.ui_procImgTab, "Processed images")
        procImg_gLayout = QtWidgets.QGridLayout(self.ui_procImgTab)

        # -- Distortion correction warp --
        (
            _, self.ui_sdc_invwarp_lnEd, self.ui_sdc_invwarp_btn,
            self.ui_sdc_invwarp_del_btn
        ) = self._add_file_row(
            procImg_gLayout, 0, "Distortion warp :", "sdc_invwarp", work_dir
        )

        # -- Distortion-corrected functional image --
        (
            _, self.ui_func_orig_sdc_lnEd, self.ui_func_orig_sdc_btn,
            self.ui_func_orig_sdc_del_btn
        ) = self._add_file_row(
            procImg_gLayout, 1, "Corrected function :", "func_orig_sdc",
            work_dir,
        )

        # -- Aligned anatomy --
        alAnat_row = self._add_file_row(
            procImg_gLayout, 2, "Aligned anatomy :", "alAnat", work_dir
        )
        (
            _, self.ui_alAnat_lnEd, self.ui_alAnat_btn, self.ui_alAnat_del_btn
        ) = alAnat_row
        self.ui_objs.extend(alAnat_row)

        # -- Warped images --
        self.ui_wrpImg_grpBx = QtWidgets.QGroupBox("Warped images")
        wrpImg_gLayout = QtWidgets.QGridLayout(self.ui_wrpImg_grpBx)
        procImg_gLayout.addWidget(self.ui_wrpImg_grpBx, 3, 0, 1, 4)
        self.ui_objs.append(self.ui_wrpImg_grpBx)

        # -- WM in the original space --
        (
            _, self.ui_WM_orig_lnEd, self.ui_WM_orig_btn,
            self.ui_WM_orig_del_btn
        ) = self._add_file_row(
            wrpImg_gLayout, 0, "WM mask in original :", "WM_orig", work_dir
        )

        # -- Vent in the original space --
        (
            _, self.ui_Vent_orig_lnEd, self.ui_Vent_orig_btn,
            self.ui_Vent_orig_del_btn
        ) = self._add_file_row(
            wrpImg_gLayout, 1, "Vent mask in original :", "Vent_orig",
            work_dir,
        )

        # -- ROI in the original space --
        (
            self.ui_ROI_orig_lb, self.ui_ROI_orig_lnEd, self.ui_ROI_orig_btn,
            self.ui_ROI_orig_del_btn
        ) = self._add_file_row(
            wrpImg_gLayout, 2, "ROI mask in original :", "ROI_orig", work_dir
        )

        # -- RTP mask --
        RTP_mask_row = self._add_file_row(
            procImg_gLayout, 4, "RTP mask :", "RTP_mask", work_dir
        )
        (
            _, self.ui_RTP_mask_lnEd, self.ui_RTP_mask_btn,
            self.ui_RTP_mask_del_btn
        ) = RTP_mask_row
        self.ui_objs.extend(RTP_mask_row)

        # -- GSR mask --
        GSR_mask_row = self._add_file_row(
            procImg_gLayout, 5, "GSR mask :", "GSR_mask", work_dir
        )
        (
            _, self.ui_GSR_mask_lnEd, self.ui_GSR_mask_btn,
            self.ui_GSR_mask_del_btn
        ) = GSR_mask_row
        self.ui_objs.extend(GSR_mask_row)

        # -- Delete all processed images --
        self.ui_delAllProcImgs_btn = QtWidgets.QPushButton("Delete All")
        self.ui_delAllProcImgs_btn.setStyleSheet(
            "background-color: rgb(255,0,0);"
        )
        self.ui_delAllProcImgs_btn.clicked.connect(
            lambda: self.delete_file("AllProc")
        )
        procImg_gLayout.addWidget(self.ui_delAllProcImgs_btn, 6, 3)
        self.ui_objs.append(self.ui_delAllProcImgs_btn)

        if self.run_extApp:
            # --- External App tab ------------------------------------------
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

            # Address
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
            # --- Output tab ------------------------------------------------
            self.ui_extAppTab = QtWidgets.QWidget()
            self.ui_top_tabs.addTab(self.ui_extAppTab, "Output")
            self.ui_extApp_gLayout = QtWidgets.QGridLayout(self.ui_extAppTab)

            # Real-time signal output file
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
            self.ui_sigSaveFile_btn.setStyleSheet(_SET_BTN_STYLE)
            self.ui_extApp_gLayout.addWidget(self.ui_sigSaveFile_btn, 0, 2)

        # --- RTP setup button ------------------------------------------------
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
        # RtpGUI.layout_ui places this checkbox.

        # --- Ready and Quit buttons ------------------------------------------
        readyQuitWdt = QtWidgets.QWidget()
        self.main_win.hBoxExpCtrls.addWidget(readyQuitWdt)
        readyQuit_gLayout = QtWidgets.QGridLayout(readyQuitWdt)

        # -- Ready button --
        self.ui_ready_btn = QtWidgets.QPushButton("Ready")
        self.ui_ready_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        self.ui_ready_btn.clicked.connect(self.ready_to_run)
        self.ui_ready_btn.setEnabled(False)
        self.ui_ready_btn.setStyleSheet(
            "background-color: rgb(255,201,32);height: 20px;"
        )
        readyQuit_gLayout.addWidget(self.ui_ready_btn, 0, 0, 1, 4)

        # -- Manual start button --
        self.ui_manStart_btn = QtWidgets.QPushButton("Manual start")
        self.ui_manStart_btn.setStyleSheet("background-color: rgb(94,63,153);")
        self.ui_manStart_btn.clicked.connect(self.manual_start)
        self.ui_manStart_btn.setEnabled(False)
        readyQuit_gLayout.addWidget(self.ui_manStart_btn, 0, 4, 1, 1)

        # -- Quit button --
        self.ui_quit_btn = QtWidgets.QPushButton("Cancel")
        self.ui_quit_btn.setStyleSheet("background-color: rgb(255,0,0);")
        self.ui_quit_btn.clicked.connect(partial(self.end_run, True))
        self.ui_quit_btn.setEnabled(False)
        readyQuit_gLayout.addWidget(self.ui_quit_btn, 0, 5, 1, 1)

        return ui_rows

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_params(self):
        """Return the parameters to save, excluding run-time states."""
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
            opts.pop(k, None)

        return opts

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        # Quit the external application
        if (
            hasattr(self, "extApp_proc")
            and self.extApp_proc is not None
            and self.extApp_proc.poll() is None
        ):
            if self.extApp_sock is not None:
                if self.send_extApp("QUIT;".encode(), no_err_pop=True):
                    while self.extApp_proc.poll() is None:
                        time.sleep(0.1)

                    self.extApp_sock.close()
                    self.extApp_sock = None
            else:
                self._kill_extApp_proc(signal.SIGTERM)


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

    # Keep RTP objects for loading and saving the parameters. A new dict is
    # made not to add the application to rtp_app.rtp_objs.
    all_rtp_objs = {**rtp_app.rtp_objs, **app_obj}

    if Path("RTPSpy_params.pkl").is_file():
        load_parameters(all_rtp_objs, fname="RTPSpy_params.pkl")

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
    save_parameters(all_rtp_objs, fname="RTPSpy_params.pkl")
    sys.exit(exit_code)
