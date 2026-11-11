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

try:
    from .rtp_common import (
        RTP,
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
    from .rtp_latency import VolumeLatency

except ImportError:
    # Run as a script (no parent package)
    from rtpspy.rtp_common import (
        RTP,
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
    from rtpspy.rtp_latency import VolumeLatency


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

    # Log label (see RTP.do_proc). The image filename is not changed.
    _proc_label = "ROI signal extraction"

    # GUI panel (rtpspy.ui.app_ui). In the GUI, the panel also runs the
    # session monitor and the ROI signal plot.
    _ui_class = "app_ui:RtpAppUI"

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

        # Time (seconds) without a new image after which the GUI ends the
        # session (see RtpAppUI.check_run_status)
        self.max_watch_wait = 5

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

        # Latency of each volume: TTL onset -> image read -> feedback sent
        # (see rtp_latency.py and _record_feedback)
        self.latency = VolumeLatency()

        # Online QC of the latest volume passed from REGRESS (see do_proc)
        self.qc_last = None
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
    # ready_proc, _process (or do_proc), end_reset, end_proc
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
        self.latency.reset()

        self._ui_call("reset_ROISig_plot")

        return self._proc_ready

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def do_proc(self, fmri_img, vol_idx=None, pre_proc_time=0, **kwargs):
        """Process the RTP image (e.g., extracting the ROI signal).

        The processing step is self._process(); the common bookkeeping
        (volume count, ignore_init, processing time log, error handling) is
        done by RTP.do_proc. As the last process of the chain, RtpApp does
        not pass the image on nor keep it (self._output does nothing).

        kwargs can include qc, the online QC of the volume from RtpRegress
        ({'FD': mm, 'DVARS': %, 'tSNR': median tSNR, 'censored': bool};
        see RtpRegress). It is kept in self.qc_last. The message sent to the
        external application is not changed; an application class
        overriding _process or do_proc can use self.qc_last (or accept the
        qc keyword in do_proc) to use the QC, e.g., to skip the feedback of
        a censored volume. If an overriding do_proc has no qc parameter or
        **kwargs, REGRESS does not pass qc; the QC is then available in
        self.rtp_objs["REGRESS"].qc_last.
        """
        if "qc" in kwargs:
            self.qc_last = kwargs["qc"]

        super().do_proc(fmri_img, vol_idx=vol_idx,
                        pre_proc_time=pre_proc_time, **kwargs)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _process(self, fmri_img, vol_idx, pre_proc_time):
        """Extract the mean signal in the ROI and send it to the external
        application or save it to a file (see RTP.do_proc)."""
        dataV = fmri_img.get_fdata()

        # --- Initialize ------------------------------------------------------
        if Path(self.ROI_orig).is_file() and self.ROI_mask is None:
            self.ROI_mask = np.asarray(nib.load(self.ROI_orig).dataobj)

        # --- Run the process -------------------------------------------------
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
            self._record_feedback(vol_idx)
            self._logger.info(f"Saved ROI data '{val_str}'")

        # Update signal plot
        self._plt_xi.append(vol_idx + 1)
        self._roi_sig[0].append(mean_sig)

        return fmri_img

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _output(self, fmri_img, vol_idx, tstamp, **kwargs):
        """RtpApp is the end of the chain: nothing is passed or kept."""
        pass

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _proc_error(self):
        """Log the exception raised in do_proc with the traceback."""
        e = sys.exc_info()[1]
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
        """Convert DICOM to NIfTI using dcm2niix (GUI only: the DICOM
        directory is selected in a dialog; see RtpAppUI.run_dcm2nii)."""
        self._ui_call("run_dcm2nii")

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
            Ask the user to edit the processing command lines (GUI only).
            The default is False.
        overwrite : bool, optional
            Overwrite existing files. The default is False.
        progress_dlg : bool, optional
            Show a progress dialog (GUI only). The default is False.

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

        self._ui_call("set_create_masks_enabled", False)

        # Progress bar (None without a GUI). The processing output to stdout
        # is shown in the dialog.
        progress_bar = None
        if progress_dlg:
            progress_bar = self._notifier().progress_dialog(
                title="Mask image creation",
                modal=False,
                parent=self.main_win,
                st_time=time.time(),
                win_size=(750, 320),
            )
        if progress_bar is not None:
            sys.stdout.flush()
            original_stdout = sys.stdout
            sys.stdout = progress_bar.ostream
            progress_bar.set_value(0)

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
                self._notifier().info(
                    "Mask creation is complete.",
                    "Mask creation is complete.",
                    parent=self.main_win,
                )

        self._ui_call("set_create_masks_enabled", True)
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

        progress_bar = None
        if (self.main_win is not None) and (self.enable_RTP > 0):
            progress_bar = self._notifier().progress_dialog(
                title="Session setup ...", parent=self.main_win
            )
        show_progress = progress_bar is not None
        if show_progress:
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

        if show_progress:
            progress_bar.close()

        if self.main_win is not None:
            # Enable the Ready button and show the parameter list
            self._ui_call("on_RTP_setup_done")

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
                pobj.set_param("mot_reg", "None")
                errmsg = "VOLREG is not enabled. "
                errmsg += "Motion regressor is set to None."
                self._logger.error(errmsg)
                self.err_popup(errmsg)
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
            elif not ignore_error:
                # Ask the user to select a mask file (GUI only)
                ret = pobj._ui_call("select_mask", mask_param, self.work_dir)
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
                self._ui_call("show_run_plots")

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
            # In the GUI, the running status is checked every second (the
            # session ends when no new image arrives) and the scan onset is
            # waited for in a thread (see RtpAppUI). Without the GUI, the
            # scan is started with manual_start() and ended with end_run().
            self._ui_call("start_run_check")

            # Wait for the scan onset signal
            if (
                "TTLPHYSIO" in self.rtp_objs
                and self.rtp_objs["TTLPHYSIO"] is not None
            ):
                self.rtp_objs["TTLPHYSIO"].standby_scan()

            self._scanning = False
            self._wait_start = True

            self._ui_call("start_wait_onset")

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

        self._ui_call("stop_run_check")

        if quit_btn:
            self._logger.info("Quit button is pressed.")

        if self.main_win is not None:
            self._ui_call("on_end_run_start")

        try:
            save_fnames = {}

            self.end_proc()

            # Abort waiting for the scan onset (GUI)
            self._ui_call("abort_wait_onset")

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

                # Save the latency of each volume
                lat_f = self.latency.save_csv(
                    save_dir / f"latency_{scan_name}.csv")
                if lat_f is not None:
                    save_fnames["Latency"] = lat_f

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
            self._ui_call("on_end_run_done")

        if self.rtp_objs["TTLPHYSIO"].available:
            self.rtp_objs["TTLPHYSIO"].release_standby_scan()
            self.rtp_objs["TTLPHYSIO"].end_scan()

        self._isReadyRun = False
        self._isRunning_end_run_proc = False
        self._wait_start = False
        self._scanning = False

        return save_fnames

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def show_images(self, base=None, ovl=None):
        """Show images in the image viewer of the GUI (rtpspy.ui.image_viewer)
        to check them.

        Parameters
        ----------
        base : str, optional
            Base image: 'anat' (aligned anatomy) or 'func' (reference
            function image).
        ovl : str, optional
            Overlay image: 'func', 'wm', 'vent', 'roi', 'RTPmask', or
            'GSRmask'.
        With neither ('Image viewer' button), the reference function image
        is shown (or, if it is not set, the viewer asks for an image file).
        The Base and Overlay menus of the viewer choose other images.
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

        if getattr(self, "_ui", None) is None:
            self._logger.warning(f"No GUI to show the images ({base}, {ovl}).")
            return
        choices = self._viewer_choices()
        start_dir = Path(self.work_dir)

        if base is None and ovl is None:
            # 'Image viewer' button: the reference function image, or
            # choose a file in work_dir
            if self.func_orig and Path(self.func_orig).is_file():
                self._ui_call("show_images", self.func_orig, None, None,
                              choices, start_dir)
            else:
                self._ui_call("open_image_dialog", start_dir)
            return

        if base_img is None or not Path(base_img).is_file():
            errmsg = f"Base image ({base}) is not set or found: {base_img}"
            self.err_popup(errmsg)
            return
        if ovl_img is None or not Path(ovl_img).is_file():
            errmsg = f"Overlay image ({ovl}) is not set or found: {ovl_img}"
            self.err_popup(errmsg)
            return

        self._ui_call("show_images", base_img, ovl_img, None, choices,
                      start_dir)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _viewer_choices(self):
        """{name: file} of the images in the Base and Overlay menus of the
        image viewer (the images that are set)."""
        sdc_applied = bool(self.sdc_invwarp) and \
            Path(self.sdc_invwarp).is_file()
        items = [
            ("Base function image", self.func_orig),
            ("Function image (distortion corrected)",
             self.func_orig_sdc if sdc_applied else ""),
            ("Anatomy image (aligned)", self.alAnat),
            ("ROI mask", self.ROI_orig),
            ("WM mask", self.WM_orig),
            ("Vent mask", self.Vent_orig),
            ("RTP mask", self.RTP_mask),
            ("GSR mask", self.GSR_mask),
        ]
        choices = {}
        for name, fname in items:
            if fname and Path(fname).is_file():
                choices[name] = str(fname)
            # The masks in the corrected function space (on the anatomy)
            if sdc_applied and fname and "_invsdc" in Path(fname).name:
                f2 = Path(fname).parent / \
                    Path(fname).name.replace("_invsdc", "")
                if f2.is_file():
                    choices[f"{name} (distortion corrected)"] = str(f2)
        return choices

    # Former name (the images were shown in AFNI)
    check_onAFNI = show_images

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
            # Confirmation dialog (without a GUI, the answer is No)
            if attr != "AllProc":
                text = f"Are you sure you want to delete {ff}?"
                title = f"Delete {attr} file"
            else:
                text = "Are you sure you want to delete ALL processed files?"
                title = "Delete all processed files"

            ret = self._notifier().question(
                text, title, buttons=("Yes", "No"), default="No")
            if ret != "Yes":
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
    # The ROI signal plot window is in the GUI (rtpspy.ui.app_ui).
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def open_ROISig_plot(self, num_ROIs=1, roi_labels=[]):
        """Open the ROI signal plot (GUI only)."""
        self._ui_call("open_ROISig_plot", num_ROIs=num_ROIs,
                      roi_labels=roi_labels)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def close_ROISig_plot(self):
        """Close the ROI signal plot (GUI only)."""
        self._ui_call("close_ROISig_plot")

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

    # --- Session control in the GUI ------------------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ui_setEnabled(self, enabled):
        """Enable or disable the widgets of all the modules in the GUI."""
        if self.main_win is not None:
            self._ui_call("set_enabled_all", enabled)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def keyPressEvent(self, event):
        """Key press in the GUI window (passed by RtpGUI). The 'T' key
        starts the scan while waiting for the scan onset."""
        # Qt key code of a letter is its uppercase ASCII code (Qt.Key_T).
        if self._wait_start and event.key() == ord("T"):
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

        progress_bar = None
        if self.main_win is not None:
            cmdstr = Path(cmd.split()[0]).name
            progress_bar = self._notifier().progress_dialog(
                title=f"Run {cmdstr} ...", parent=self.main_win
            )
        if progress_bar is not None:
            progress_bar.btnCancel.setVisible(False)
            progress_bar.set_msgTxt(f"Run {cmdstr} ...")
            time.sleep(0.1)  # Give the UI time to update the dialog

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
    def _record_feedback(self, vol_idx, t_sent=None):
        """
        Record the latency of volume vol_idx, whose feedback signal was sent
        (or saved) at t_sent (default: now). Called by send_extApp for an
        "NF ..." message, so applications that override do_proc or _process
        (e.g., example/ROI-NF) need nothing more. See rtp_latency.py.
        """
        try:
            if t_sent is None:
                t_sent = time.time()

            # Time the image of the volume was read (RtpWatch)
            t_read = None
            watch = self.rtp_objs.get("WATCH")
            if watch is not None and 0 <= vol_idx < len(watch._proc_time):
                t_read = watch._proc_time[vol_idx]

            # Output in the retrospective batch of RtpRegress
            regress = self.rtp_objs.get("REGRESS")
            retro = bool(getattr(regress, "in_retro_proc", False))

            # Acquisition onset from the scanner TTL
            onset, est = None, True
            if self.scan_onset > 0:
                TR = getattr(self.rtp_objs.get("TSHIFT"), "TR", None)
                if not TR and watch is not None and \
                        len(watch._proc_time) > 2:
                    TR = float(np.median(np.diff(watch._proc_time)))
                ttl_obj = self.rtp_objs.get("TTLPHYSIO")
                ttl = (ttl_obj.get_ttl_onsets() if ttl_obj is not None
                       else np.array([]))
                onset, est = self.latency.match_onset(
                    vol_idx, ttl, self.scan_onset, TR or 2.0)

            self.latency.record(vol_idx, onset, est, t_read, t_sent, retro)
            d = self.latency.data
            self._logger.debug(
                f"#{vol_idx + 1};Latency;acq_to_read="
                f"{d['acq_to_read'][-1]:.3f}s"
                f"{'(est.)' if est else ''};read_to_fb="
                f"{d['read_to_fb'][-1]:.3f}s{';retro' if retro else ''}")

        except Exception as e:
            self._logger.error(
                f"Failed to record the latency: {e}\n"
                + traceback.format_exc())

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
            if not pkl and data[:3] == b"NF ":
                # Feedback signal "NF {time},{volume_index},...;"
                try:
                    vol_idx = int(data[3:].split(b",")[1])
                except (IndexError, ValueError):
                    vol_idx = None
                if vol_idx is not None:
                    self._record_feedback(vol_idx)
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
            if reset_fn is None:
                self._ui_update(attr, val)

        elif attr == "work_dir":
            if val is None or not Path(val).is_dir():
                return

            val = Path(val)
            setattr(self, attr, val)

            if self.main_win is not None:
                self.main_win.set_workDir(val)

        elif attr == "extApp_cmd":
            if self.main_win is not None:
                self._ui_update(attr, val)

        elif attr == "extApp_addr":
            if val is None:
                return

            if type(val) is tuple:
                if self.main_win is not None:
                    self.extApp_addr = val
                    if not self.isAlive_extApp():
                        self.extApp_addr = None
                        return
                    self._ui_update(attr, val)

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
                            # Restore the address in the GUI
                            self._ui_update(attr, self.extApp_addr)
                    return

                if self.extApp_sock is not None and (
                    self.extApp_addr is None
                    or tuple(self.extApp_addr[:2]) != val
                ):
                    # The address is changed. Close the current socket.
                    self.extApp_sock.close()
                    self.extApp_sock = None

        elif attr == "extApp_isAlive":
            # Check the connection and show the result with reset_fn. val is
            # the address ('host:port') to use if extApp_addr is not set.
            if self.extApp_addr is None:
                self.set_param("extApp_addr", val)
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
                fname = self._notifier().get_save_file(
                    "File name to save the real-time signal",
                    self.work_dir,
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

                self._ui_update(attr, val)

        elif attr == "ROI_resample":
            if val not in self.ROI_resample_opts:
                return

            if reset_fn is None:
                self._ui_update(attr, val)

        elif attr == "no_FastSeg":
            self._ui_update(attr, val)

        elif not hasattr(self, attr):
            # Other existing attributes (e.g., max_watch_wait) are just set
            if unk_warning:
                self._logger.error(f"{attr} is an unknown parameter.")
            return

        setattr(self, attr, val)
        if echo:
            print(f"{self.__class__.__name__}.{attr} = {getattr(self, attr)}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_params(self):
        """Return the parameters to save, excluding run-time states."""
        opts = super().get_params()

        excld_opts = (
            "ROI_mask",
            "isReadyRun",
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
            "qc_last",
            "latency",
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
    from PyQt6 import QtWidgets
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
        exit_code = app.exec()

    except Exception as e:
        with open("rtpspy.error", "w") as fd:
            fd.write(str(e))

        print(str(e))
        exit_code = -1

    # --- End ---
    save_parameters(all_rtp_objs, fname="RTPSpy_params.pkl")
    sys.exit(exit_code)
