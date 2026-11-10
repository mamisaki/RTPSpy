#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image processing steps to make the mask and ROI images for real-time fMRI
processing (brain extraction, segmentation, alignment, and warping).

@author: mmisaki
"""

# %% import ===================================================================
import subprocess
import shutil
from pathlib import Path
import re
import time
import os
import sys
import shlex
import traceback

import numpy as np
import nibabel as nib
import ants
from ants.registration import build_template

from .rtp_common import RTP
from .fast_seg import FastSeg


# %% RtpImgProc class =========================================================
class RtpImgProc(RTP):
    """
    Image processing utility class.

    Each processing method skips the process and uses the existing output
    files unless overwrite is True. The expected processing time of each step
    (self.proc_times) is used to update the progress bar, and is updated with
    the actual time after the step is run.

    The processing methods return the output file(s) on success and None on
    failure or when the user cancels (e.g., in the command edit dialog).
    """

    # Interpolation options of antsApplyTransforms to resample the warped ROI
    ROI_resample_opts = ["nearestNeighbor", "linear", "bSpline"]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, main_win=None):
        super().__init__()

        self.main_win = main_win
        self.fastSeg_batch_size = 1

        # Default processing times (seconds) for the progress bar
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

    # --- Internal utility methods --------------------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def _out_prefix(fname):
        """
        Return the file name without the extensions and the AFNI view
        (+orig/+tlrc), e.g., 'anat+orig.HEAD' -> 'anat'.
        """
        fname = Path(fname)
        prefix = fname.name.replace("".join(fname.suffixes[-2:]), "")
        return prefix.replace("+orig", "").replace("+tlrc", "")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def _print_step_header(title, progress_bar=None, msgTxt=None):
        """
        Print the header of a processing step. When progress_bar is given,
        msgTxt (the default is title) is shown under the bar.
        """
        if progress_bar is not None:
            progress_bar.set_msgTxt(title if msgTxt is None else msgTxt)
        print("+" * 70 + "\n" + f"+++ {title}\n")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _report_existing(self, files, progress_bar=None, total_ETA=None,
                         time_key=None):
        """
        Report that the existing files are used and the step is skipped.
        When time_key is given, the progress bar is advanced by the expected
        time of the step, self.proc_times[time_key].
        """
        label = "Use existing file: " if len(files) == 1 \
            else "Use existing files: "
        msg = label + ("\n" + " " * len(label)).join(
            [str(ff) for ff in files]) + "\n"

        if progress_bar is None:
            print(msg)
            return

        if time_key is not None:
            bar_inc = self.proc_times[time_key] / total_ETA * 100
            progress_bar.set_value(progress_bar.progBar.value() + bar_inc)
        progress_bar.add_desc(msg + "\n")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _edit_command(self, labelTxt="Command line:", cmdTxt=""):
        """
        Show a dialog to edit a command line (the notifier's text input;
        canceled without a GUI).

        Parameters
        ----------
        labelTxt : str, optional
            Description of the command. The default is 'Command line:'.
        cmdTxt : str, optional
            Initial command line. The default is ''.

        Returns
        -------
        cmd : str
            Edited command.
        okflag : bool
            True if OK was pressed, False if canceled.
        """

        return self._notifier().get_text(labelTxt, cmdTxt,
                                         parent=self.main_win)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _show_cmd_progress(self, cmd, progress_bar=None, msgTxt="", desc=""):
        """
        Run a command and print its output. This blocks until the command
        finishes, so use it only for a short command.

        For a long command, spawn a child process with subprocess.Popen and
        use self._show_proc_progress to show the progress online.

        Parameters
        ----------
        cmd : str
            Command line.
        progress_bar : rtpspy.ui.DlgProgressBar, optional
            Progress bar dialog. The default is None.
        msgTxt : str, optional
            Message text shown in the progress_bar (under the bar).
            The default is ''.
        desc : str, optional
            Description shown in the progress_bar (in the text box).
            The default is ''.

        Returns
        -------
        int
            0: no error.
            1: error.
        """

        if progress_bar is not None:
            if len(msgTxt):
                progress_bar.set_msgTxt(msgTxt)

            if len(desc):
                progress_bar.add_desc(desc)

        try:
            ostr = subprocess.check_output(
                cmd, stderr=subprocess.STDOUT, shell=True
            ).decode()
            if progress_bar is not None:
                progress_bar.add_desc(ostr)
            else:
                print(ostr)

        except Exception:
            errmsg = f"Failed to run:\n{cmd}"
            self._logger.error(errmsg)
            self.err_popup(errmsg)
            return 1

        return 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _proc_print_progress(self, proc):
        """
        Print the output of a spawned child process to stdout until it ends.

        Parameters
        ----------
        proc : subprocess.Popen object
            Child process opened by subprocess.Popen.

        Returns
        -------
        int
            Return code of proc.
        """

        out0 = bytearray()
        while proc.poll() is None:
            # Read a few bytes at a time to show the output online. A
            # multi-byte character can be split; then read more bytes before
            # decoding.
            out0 += proc.stdout.read(4)
            try:
                out1 = out0.decode()
            except UnicodeDecodeError:
                continue

            out = "\n".join(out1.splitlines())
            if len(out1) and out1[-1] == "\n":
                out += "\n"
            print(out, end="")
            out0 = bytearray()

        # Print the rest of the output
        out0 += proc.stdout.read()
        out = out0.decode(errors="replace")
        print("\n".join(out.splitlines()) + "\n")

        return proc.returncode

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _show_proc_progress(
        self,
        proc,
        progress_bar=None,
        msgTxt="",
        desc="",
        ETA=None,
        total_ETA=None,
    ):
        """
        Wait for a spawned child process to end, showing its progress in the
        progress bar dialog if available.

        Parameters
        ----------
        proc : subprocess.Popen object
            Child process opened by subprocess.Popen.
        progress_bar : rtpspy.ui.DlgProgressBar, optional
            Progress bar dialog. The default is None.
        msgTxt : str, optional
            Message text shown in the progress_bar (under the bar).
            The default is ''.
        desc : str, optional
            Description shown in the progress_bar (in the text box).
            The default is ''.
        ETA : float, optional
            Expected time (seconds) of the process, used to update the bar
            online. The default is None.
        total_ETA : float, optional
            Expected time (seconds) of all the processes when proc is one of
            many. The bar is advanced by the fraction ETA / total_ETA.
            The default is None.

        Returns
        -------
        ret : int
            0: no error.
            -1: canceled.
            Other: return code of the failed process.
        """

        if progress_bar is not None:
            if len(msgTxt):
                progress_bar.set_msgTxt(msgTxt)

            if len(desc):
                progress_bar.add_desc(desc)

            if ETA is not None:
                if total_ETA is None:
                    total_ETA = ETA
                bar_inc = 100 * (ETA / total_ETA)
            else:
                bar_inc = None
            ret = progress_bar.proc_print_progress(
                proc, bar_inc=bar_inc, ETA=ETA
            )

            if not progress_bar.isVisible():
                # The dialog was closed (canceled).
                ret = -1
        else:
            ret = self._proc_print_progress(proc)

        if ret != 0:
            if ret == -1:
                self._logger.info(f"Canceled: {msgTxt}")
            else:
                errmsg = f"Failed: {msgTxt}"
                self._logger.error(errmsg)
                self.err_popup(errmsg)

        return ret

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _spawn(self, cmd, work_dir):
        """Spawn cmd in work_dir, with stderr merged into stdout."""
        return subprocess.Popen(
            shlex.split(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=work_dir,
        )

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def load_image(self, fname, vidx=-1):
        """
        Load an image file and return a nibabel NIfTI image object.
        If vidx is not None, only the volume vidx of a 4D image is returned.
        None is returned on error.
        """
        try:
            img = nib.load(fname)
            if not isinstance(img, (nib.Nifti1Image, nib.Nifti2Image)):
                # e.g., AFNI BRIK
                img = nib.Nifti1Image.from_image(img)

            if vidx is not None and img.ndim > 3:
                # slicer keeps the header and the data scaling
                img = img.slicer[..., vidx]
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errmsg = "".join(
                traceback.format_exception(exc_type, exc_obj, exc_tb)
            )
            self._logger.error(str(e) + "\n" + errmsg)
            self.err_popup(errmsg)
            return None

        return img

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def copy_deoblique(
        self, src_f, dst_f, progress_bar=None, overwrite=False
    ):
        """
        Copy src_f to dst_f, deobliquing it if it is oblique.

        If dst_f exists and is not oblique, nothing is done unless overwrite
        is True. When dst_f is src_f itself, it is never copied onto itself
        and is only deobliqued if needed.
        """
        src_f, dst_f = Path(src_f), Path(dst_f)
        same_file = dst_f.exists() and dst_f.resolve() == src_f.resolve()
        if dst_f.exists() and (not overwrite or same_file):
            oblique = subprocess.check_output(
                shlex.split(f"3dinfo -is_oblique {dst_f}")
            )
            if not int(oblique.decode().rstrip()):
                return

        oblique = subprocess.check_output(
            shlex.split(f"3dinfo -is_oblique {src_f}")
        )
        if int(oblique.decode().rstrip()):
            cmd = f"3dWarp -overwrite -deoblique -prefix {dst_f} {src_f}"
        else:
            cmd = f"3dcopy -overwrite {src_f} {dst_f}"

        ret = self._show_cmd_progress(
            cmd,
            progress_bar,
            msgTxt=f"Deoblique {Path(src_f).name}",
            desc="\n== Deoblique anatomy ==",
        )
        assert ret == 0, f"Failed to deoblique {src_f}.\n"

    # --- Image processing methods --------------------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run_fast_seg(
        self, work_dir, anat_orig, total_ETA, progress_bar=None,
        overwrite=False
    ):
        """
        Segment the anatomy image (anat_orig) with FastSeg to make the brain,
        white matter, ventricle, and aseg images.

        Parameters
        ----------
        work_dir : Path
            Working directory.
        anat_orig : Path
            Anatomy image file in the individual's original space.
        total_ETA : float
            Expected time (seconds) of all the processing steps, for the
            progress bar.
        progress_bar : rtpspy.ui.DlgProgressBar, optional
            Progress bar dialog. The default is None.
        overwrite : bool, optional
            Overwrite existing files. When this is False and the results
            exist, the process is skipped. The default is False.

        Returns
        -------
        brain_anat_orig, wm_anat_orig, vent_anat_orig, aseg_anat_orig : Path
            Filenames of the segmented images. None on failure.
        """

        out_prefix = self._out_prefix(anat_orig)

        self._print_step_header(
            "Brain extraction and WM/Vent segmentation", progress_bar)

        brain_anat_orig = work_dir / (out_prefix + "_Brain.nii.gz")
        wm_anat_orig = work_dir / (out_prefix + "_WM.nii.gz")
        vent_anat_orig = work_dir / (out_prefix + "_Vent.nii.gz")
        aseg_anat_orig = work_dir / (out_prefix + "_aseg.nii.gz")
        out_files = (brain_anat_orig, wm_anat_orig, vent_anat_orig,
                     aseg_anat_orig)

        if not overwrite and all([ff.is_file() for ff in out_files]):
            self._report_existing(out_files, progress_bar, total_ETA,
                                  "FastSeg")
            return out_files

        fastSeg = FastSeg()
        st = time.time()

        def show_proc_progress(proc, ETA):
            return self._show_proc_progress(
                proc,
                progress_bar,
                msgTxt="FastSeg image segmentation",
                ETA=ETA,
                total_ETA=total_ETA,
            )

        # Prepare the input file (convert BRIK to NIfTI and correct the bias
        # field unless it has been done)
        bias_correction = "N4BCOR.nii" not in anat_orig.name
        in_f, prefix = fastSeg.prep_files(
            anat_orig,
            bias_correction=bias_correction,
            prefix=(work_dir / out_prefix),
            show_proc_progress=show_proc_progress,
            ETA=None,
        )
        if in_f is None:
            return None

        # Run the segmentation
        fsSeg_mgz = fastSeg.run_seg_only(
            in_f,
            prefix,
            self.fastSeg_batch_size,
            show_proc_progress=show_proc_progress,
            ETA=self.proc_times["FastSeg"],
        )
        if fsSeg_mgz is None:
            return None
        subj_dir = fsSeg_mgz.parent.parent

        # Make the mask images
        out_fs = fastSeg.make_seg_images(
            in_f,
            fsSeg_mgz,
            prefix,
            segs=["Brain", "WM", "Vent", "aseg"],
            show_proc_progress=show_proc_progress,
        )
        if out_fs is None:
            return None

        # Delete the FastSurfer subject directory
        if subj_dir.is_dir():
            shutil.rmtree(subj_dir)

        self.proc_times["FastSeg"] = np.ceil(time.time() - st)

        if progress_bar is not None:
            progress_bar.add_desc("\n")

        return out_files

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def skullStrip(
        self,
        work_dir,
        anat_orig,
        total_ETA,
        progress_bar=None,
        ask_cmd=False,
        overwrite=False,
    ):
        """
        Skull-strip the anatomy image (anat_orig) with 3dSkullStrip.

        Parameters
        ----------
        work_dir : Path
            Working directory.
        anat_orig : Path
            Anatomy image file in the individual's original space.
        total_ETA : float
            Expected time (seconds) of all the processing steps, for the
            progress bar.
        progress_bar : rtpspy.ui.DlgProgressBar, optional
            Progress bar dialog. The default is None.
        ask_cmd : bool, optional
            Allow editing the command line. The default is False.
        overwrite : bool, optional
            Overwrite existing files. When this is False and the results
            exist, the process is skipped. The default is False.

        Returns
        -------
        brain_anat_orig : Path
            Filename of the skull-stripped image. None on failure or
            cancel.
        """

        self._print_step_header("Brain extraction", progress_bar)

        out_prefix = self._out_prefix(anat_orig)
        brain_anat_orig = work_dir / (out_prefix + "_Brain.nii.gz")
        if brain_anat_orig.is_file() and not overwrite:
            self._report_existing([brain_anat_orig], progress_bar, total_ETA,
                                  "SkullStrip")
            return brain_anat_orig

        cmd = f"3dSkullStrip -overwrite -input {anat_orig}"
        cmd += f" -prefix {brain_anat_orig}"
        if ask_cmd:
            cmd, okflag = self._edit_command(cmdTxt=cmd)
            if not okflag:
                return None

        st = time.time()
        proc = self._spawn(cmd, work_dir)
        ret = self._show_proc_progress(
            proc,
            progress_bar,
            msgTxt="Skull stripping",
            ETA=self.proc_times["SkullStrip"],
            total_ETA=total_ETA,
        )
        if ret != 0:
            return None

        self.proc_times["SkullStrip"] = np.ceil(time.time() - st)

        if progress_bar is not None:
            progress_bar.add_desc("\n")

        assert brain_anat_orig.is_file(), "Failed in skullStrip."

        return brain_anat_orig

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def _mean_aligned_image(aimg_4d, fixed_aimg):
        """
        Rigidly align each volume of aimg_4d to fixed_aimg and return the
        mean of the aligned volumes on the fixed_aimg grid.
        """
        reg_ndimgs = []
        for aimg in ants.ndimage_to_list(aimg_4d):
            areg = ants.registration(
                fixed=fixed_aimg,
                moving=aimg,
                type_of_transform="Rigid",
                verbose=True,
            )
            reg_ndimgs.append(areg["warpedmovout"].numpy())
        return fixed_aimg.new_image_like(np.mean(reg_ndimgs, axis=0))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run_pepolar_sdc(
        self,
        work_dir,
        fieldmap_posi,
        fieldmap_nega,
        func_orig,
        total_ETA,
        progress_bar=None,
        overwrite=False,
    ):
        """
        Susceptibility distortion correction (SDC) with the reversed
        phase-encoding (PEPOLAR) field map images.

        The positive and negative phase-encoding images are aligned to
        func_orig, and a nonlinear template of the two is built as the
        undistorted image. The warp from func_orig to the template is the
        distortion correction.

        Parameters
        ----------
        work_dir : Path
            Working directory.
        fieldmap_posi, fieldmap_nega : str or Path
            Field map images with the positive and negative phase-encoding
            directions.
        func_orig : Path
            Functional image file in the individual's original space.
        total_ETA : float
            Expected time (seconds) of all the processing steps, for the
            progress bar.
        progress_bar : rtpspy.ui.DlgProgressBar, optional
            Progress bar dialog. The default is None.
        overwrite : bool, optional
            Overwrite existing files. When this is False and the results
            exist, the process is skipped. The default is False.

        Returns
        -------
        sdc_invwarp : Path
            Composite transform file mapping the corrected space to the
            distorted func_orig space.
        func_orig_sdc : Path
            Distortion-corrected func_orig.
        """
        func_stem = func_orig.with_suffix("").stem
        func_orig_sdc = work_dir / (func_stem + "_sdc.nii.gz")
        sdc_invwarp = work_dir / (func_stem + "_sdc_invwarp_comptx.nii.gz")
        if func_orig_sdc.is_file() and sdc_invwarp.is_file() \
                and not overwrite:
            self._report_existing([sdc_invwarp], progress_bar, total_ETA,
                                  "SDC")
            return sdc_invwarp, func_orig_sdc

        self._print_step_header("Distortion correction", progress_bar)

        st = time.time()

        posi_aimg = ants.image_read(str(fieldmap_posi))
        nega_aimg = ants.image_read(str(fieldmap_nega))
        func_orig_aimg = ants.image_read(str(func_orig))

        # Average of the posi and nega volumes aligned to func_orig
        posi_reg_mean_aimg = self._mean_aligned_image(
            posi_aimg, func_orig_aimg)
        nega_reg_mean_aimg = self._mean_aligned_image(
            nega_aimg, func_orig_aimg)

        # The nonlinear template of the posi and nega images is the
        # distortion-corrected image.
        sdc_template_aimg = build_template(
            posi_reg_mean_aimg, [posi_reg_mean_aimg, nega_reg_mean_aimg]
        )

        # Rigidly align the template to func_orig
        reg_sdc_to_orig = ants.registration(
            fixed=func_orig_aimg,
            moving=sdc_template_aimg,
            type_of_transform="BOLDRigid",
        )
        sdc_template_reg_orig = reg_sdc_to_orig["warpedmovout"]

        # Warp between the template and func_orig
        warp_reg = ants.registration(
            fixed=func_orig_aimg,
            moving=sdc_template_reg_orig,
            type_of_transform="SyNOnly",
        )

        # func_orig warped to the template is the corrected func_orig.
        func_orig_sdc_aimg = warp_reg["warpedfixout"]
        ants.image_write(func_orig_sdc_aimg, str(func_orig_sdc))

        # Save the warp as a single composite transform file
        # (sdc_invwarp_stem + 'comptx.nii.gz').
        sdc_invwarp_stem = work_dir / sdc_invwarp.name.replace(
            "comptx.nii.gz", ""
        )
        ants.apply_transforms(
            fixed=func_orig_aimg,
            moving=sdc_template_reg_orig,
            transformlist=warp_reg["fwdtransforms"],
            compose=str(sdc_invwarp_stem),
        )

        self.proc_times["SDC"] = np.ceil(time.time() - st)
        assert sdc_invwarp.is_file()

        return sdc_invwarp, func_orig_sdc

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def align_anat2epi(
        self,
        work_dir,
        brain_anat_orig,
        func_orig,
        total_ETA,
        progress_bar=None,
        ask_cmd=False,
        overwrite=False,
    ):
        """
        Align the anatomy image to the base functional image using
        align_epi_anat.py in AFNI.
        https://afni.nimh.nih.gov/pub/dist/doc/program_help/align_epi_anat.py.html

        Parameters
        ----------
        work_dir : Path
            Working directory.
        brain_anat_orig : str or Path
            Brain-extracted anatomy image file.
        func_orig : str or Path
            Functional image file.
        total_ETA : float
            Expected time (seconds) of all the processing steps, for the
            progress bar.
        progress_bar : rtpspy.ui.DlgProgressBar, optional
            Progress bar dialog. The default is None.
        ask_cmd : bool, optional
            Allow editing the command line. The default is False.
        overwrite : bool, optional
            Overwrite existing files. When this is False and the results
            exist, the process is skipped. The default is False.

        Returns
        -------
        alAnat : Path
            Aligned anatomy image file. None on failure or cancel.
        """

        self._print_step_header("Align anat to func", progress_bar)

        brain_anat_orig = Path(brain_anat_orig)
        suffs = [
            ext
            for ext in brain_anat_orig.suffixes
            if ext in (".nii", ".gz", ".HEAD", ".BRIK")
        ]
        alAnat = work_dir / (
            brain_anat_orig.name.replace("".join(suffs), "_alFunc.nii.gz")
        )
        if alAnat.is_file() and not overwrite:
            self._report_existing([alAnat], progress_bar, total_ETA,
                                  "AlAnat")
            return alAnat

        if alAnat.is_file():
            alAnat.unlink()

        anat_orig_rel = os.path.relpath(brain_anat_orig, work_dir)
        func_orig_rel = os.path.relpath(func_orig, work_dir)

        cmd = "align_epi_anat.py -overwrite -anat2epi"
        cmd += f" -anat {anat_orig_rel} -epi {func_orig_rel} -epi_base 0"
        cmd += " -suffix _alFunc -epi_strip 3dAutomask -anat_has_skull no"
        cmd += f" -master_anat {brain_anat_orig}"
        cmd += " -volreg off -tshift off -ginormous_move"
        if ask_cmd:
            labelTxt = "Command line: (see "
            labelTxt += "https://afni.nimh.nih.gov/pub/dist/doc/program_help/"
            labelTxt += "align_epi_anat.py.html)"
            labelTxt += "\nConsider the -ginormous_move or"
            labelTxt += " -partial_coverage option."
            cmd, okflag = self._edit_command(labelTxt=labelTxt, cmdTxt=cmd)
            if not okflag:
                return None

        st = time.time()
        proc = self._spawn(cmd, work_dir)
        ret = self._show_proc_progress(
            proc,
            progress_bar,
            msgTxt="Align anat to func",
            ETA=self.proc_times["AlAnat"],
            total_ETA=total_ETA,
        )
        if ret != 0:
            return None

        # align_epi_anat.py writes an AFNI BRIK file; convert it to NIfTI.
        alAnat_f_stem = alAnat.stem.replace(".nii", "")
        alAnat_brik = list(work_dir.glob(alAnat_f_stem + "*.HEAD"))
        if len(alAnat_brik):
            cmd = f"3dAFNItoNIFTI -overwrite -prefix {alAnat}"
            cmd += f" {alAnat_brik[0]}"
            self._show_cmd_progress(
                cmd,
                progress_bar,
                msgTxt="Convert alAnat to NIfTI",
                desc="++ Convert alAnat to NIfTI\n",
            )

        self.proc_times["AlAnat"] = np.ceil(time.time() - st)
        assert alAnat.is_file()

        if progress_bar is not None:
            progress_bar.add_desc("\n")

        return alAnat

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def erode_ROI(self, work_dir, src_f, out_f, erode=0, ask_cmd=False):
        """
        Spawn 3dmask_tool to erode an ROI mask. The WM and Vent masks are
        eroded to exclude voxels partially in the GM.

        Parameters
        ----------
        work_dir : str or Path
            Working directory.
        src_f : str or Path
            Source file.
        out_f : str or Path
            Output file.
        erode : int, optional
            Number of voxels to erode. A negative value dilates the mask.
            The default is 0.
        ask_cmd : bool, optional
            Allow editing the command line. The default is False.

        Returns
        -------
        proc : subprocess.Popen object
            Child process opened by subprocess.Popen. None on failure or
            cancel.
        """

        src = Path(os.path.relpath(src_f, work_dir))
        out_f = Path(os.path.relpath(out_f, work_dir))

        cmd = f"3dmask_tool -overwrite -input {src} -dilate_input {-erode}"
        cmd += f" -prefix {out_f}"
        if ask_cmd:
            labelTxt = "Command line: (see "
            labelTxt += "https://afni.nimh.nih.gov/pub/dist/doc/"
            labelTxt += "program_help/3dmask_tool.html)"
            cmd, okflag = self._edit_command(labelTxt=labelTxt, cmdTxt=cmd)
            if not okflag:
                return None

        try:
            return self._spawn(cmd, work_dir)
        except Exception as e:
            errmsg = str(e) + "\n"
            errmsg += f"'{cmd}' failed."
            self._logger.error(errmsg)
            self.err_popup(errmsg)
            return None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def resample_segmasks(
        self,
        work_dir,
        seg_anat_f,
        segname,
        erode,
        func_orig,
        total_ETA,
        aff1D_f=None,
        progress_bar=None,
        ask_cmd=False,
        overwrite=False,
    ):
        """
        Erode a segmentation mask and resample it to the func_orig grid.

        Parameters
        ----------
        work_dir : Path
            Working directory.
        seg_anat_f : Path
            Segmentation mask in the anatomy space.
        segname : str
            Mask name (e.g., 'WM', 'Vent', 'aseg'), used in the messages and
            as the key of self.proc_times (f'Resample_{segname}_mask').
        erode : int
            Number of voxels to erode (see erode_ROI). 0 for no erosion.
        func_orig : str or Path
            Functional image file defining the output grid.
        total_ETA : float
            Expected time (seconds) of all the processing steps, for the
            progress bar.
        aff1D_f : str or Path, optional
            Affine matrix file (e.g., *_mat.aff12.1D) to align seg_anat_f to
            func_orig. If None, seg_anat_f is only resampled.
        progress_bar : rtpspy.ui.DlgProgressBar, optional
            Progress bar dialog. The default is None.
        ask_cmd : bool, optional
            Allow editing the erosion command line. The default is False.
        overwrite : bool, optional
            Overwrite existing files. When this is False and the results
            exist, the process is skipped. The default is False.

        Returns
        -------
        seg_al_f : Path
            Mask resampled to func_orig. None on failure or cancel.
        """
        print("+" * 70 + "\n" + f"+++ Resample {segname} mask\n")
        if progress_bar is not None:
            progress_bar.set_msgTxt(f"Align {segname} to func")

        # A template mask warped to alAnat by ants_warp_resample has the
        # '_alAnat' suffix, which is replaced with '_alFunc'.
        seg_prefix = re.sub(r"_alAnat$", "", self._out_prefix(seg_anat_f))
        seg_al_f = work_dir / (seg_prefix + "_alFunc.nii.gz")
        time_key = f"Resample_{segname}_mask"
        if seg_al_f.is_file() and not overwrite:
            self._report_existing([seg_al_f], progress_bar, total_ETA,
                                  time_key)
            return seg_al_f

        st = time.time()

        # --- Erode the segmentation mask ---
        if erode != 0:
            # Temporary file, deleted below with the other rm.* files
            src_f = seg_al_f.parent / ("rm.1." + seg_al_f.name)
            proc = self.erode_ROI(
                work_dir, seg_anat_f, src_f, erode=erode, ask_cmd=ask_cmd
            )
            if proc is None:
                return None

            ret = self._show_proc_progress(
                proc, progress_bar, desc=f"\n== Erode {segname} ==\n"
            )
            if ret != 0:
                return None
        else:
            src_f = seg_anat_f

        # --- Align and resample to func_orig ---
        # seg_anat_f is on the grid of the anatomy image that has been
        # deobliqued by copy_deoblique (or on the alAnat grid), so no
        # deoblique is needed here.
        if aff1D_f is not None:
            cmd = f"3dAllineate -overwrite -final NN -input {src_f}"
            cmd += f" -1Dmatrix_apply {aff1D_f}"
            cmd += f" -master {func_orig} -prefix {seg_al_f}"
        else:
            cmd = f"3dresample -overwrite -rmode NN -input {src_f}"
            cmd += f" -master {func_orig} -prefix {seg_al_f} "

        ret = self._show_cmd_progress(
            cmd,
            progress_bar,
            msgTxt=f"Resample {segname} mask",
            desc=f"\n== Resample {segname} mask ==",
        )
        assert ret == 0, f"Failed at resample_segmasks {segname}.\n"

        for rmf in work_dir.glob("rm.*"):
            rmf.unlink()

        assert seg_al_f.is_file(), (
            f"Failed at resample_segmasks {segname}.\n"
        )

        self.proc_times[time_key] = np.ceil(time.time() - st)

        if progress_bar is not None:
            progress_bar.add_desc("\n")

        return seg_al_f

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def make_RTP_GSR_masks(
        self,
        work_dir,
        func_orig,
        total_ETA,
        ref_vi=0,
        alAnat=None,
        progress_bar=None,
        ask_cmd=False,
        overwrite=False,
    ):
        """
        Make the masks for RTP and global signal regression (GSR).

        3dAutomask is used to make a functional image mask.
        If alAnat is None, the functional automask is used for both the RTP
        and GSR masks.
        If alAnat is given, the RTP mask is the union of the functional
        automask and the nonzero voxels of alAnat, and the GSR mask is their
        intersection.

        The RTP mask is used in SMOOTH and REGRESS. The GSR mask is used in
        REGRESS if GSR is enabled.

        Parameters
        ----------
        work_dir : Path
            Working directory.
        func_orig : str or Path
            Functional image file in the individual's original space.
        total_ETA : float
            Expected time (seconds) of all the processing steps, for the
            progress bar.
        ref_vi : int, optional
            Volume index in func_orig used to make the functional automask.
            The default is 0.
        alAnat : str or Path, optional
            Anatomy image file aligned to func_orig.
        progress_bar : rtpspy.ui.DlgProgressBar, optional
            Progress bar dialog. The default is None.
        ask_cmd : bool, optional
            Allow editing the command line. The default is False.
        overwrite : bool, optional
            Overwrite existing files. When this is False and the results
            exist, the process is skipped. The default is False.

        Returns
        -------
        RTP_mask, GSR_mask : Path
            None on cancel.
        """

        RTP_mask = work_dir / "RTP_mask.nii.gz"
        GSR_mask = work_dir / "GSR_mask.nii.gz"

        self._print_step_header("Make RTP and GSR masks", progress_bar)

        if RTP_mask.is_file() and GSR_mask.is_file() and not overwrite:
            self._report_existing([RTP_mask, GSR_mask], progress_bar,
                                  total_ETA, "RTP_GSR_mask")
            return RTP_mask, GSR_mask

        rm_fs = []  # Temporary files

        func_orig = Path(func_orig)
        fbase = re.sub(r"\+.*", "", func_orig.name)
        fbase = re.sub(r"(\.nii(\.gz)?|\.HEAD|\.BRIK(\.gz)?)$", "", fbase)
        func_mask = work_dir / f"automask_{fbase}.nii.gz"
        cmd = f"3dAutomask -overwrite -prefix {func_mask}"
        cmd += f" '{func_orig}[{ref_vi}]' && "
        rm_fs.append(func_mask)

        if alAnat is not None and Path(alAnat).is_file():
            alAnat = Path(alAnat)
            fbase = re.sub(r"\+.*", "", alAnat.name)
            fbase = re.sub(r"\.nii(\.gz)?$", "", fbase)
            temp_out = work_dir / "rm.anat_mask_tmp.nii.gz"
            anat_mask = work_dir / f"anatmask_{fbase}.nii.gz"

            # Anatomy mask on the func_orig grid
            cmd += f"3dmask_tool -overwrite -input {alAnat}"
            cmd += f" -prefix {temp_out} -frac 0.0 -fill_holes && "
            cmd += f"3dresample -overwrite -rmode NN -master {func_orig}"
            cmd += f" -prefix {anat_mask} -input {temp_out} && "

            # RTP mask: union of the func automask and the anat mask
            cmd += "3dmask_tool -overwrite"
            cmd += f" -input {anat_mask} {func_mask}"
            cmd += f" -prefix {RTP_mask} -frac 0.0 && "

            # GSR mask: intersection of the func automask and the anat mask
            cmd += f"3dmask_tool -overwrite -input {func_mask}"
            cmd += f" {anat_mask} -prefix {GSR_mask} -frac 1.0"

            rm_fs.append(anat_mask)
            rm_fs.append(temp_out)
        else:
            # RTP and GSR masks: func automask
            cmd += "3dAFNItoNIFTI -overwrite"
            cmd += f" -prefix {RTP_mask} {func_mask} && "
            cmd += f"cp {RTP_mask} {GSR_mask}"

        if ask_cmd:
            labelTxt = "Command line: (see "
            labelTxt += "https://afni.nimh.nih.gov/pub/dist/doc/"
            labelTxt += "program_help/3dAutomask.html)"
            cmd, okflag = self._edit_command(labelTxt=labelTxt, cmdTxt=cmd)
            if not okflag:
                return None

        st = time.time()
        ret = self._show_cmd_progress(
            cmd, progress_bar=progress_bar, msgTxt="Making RTP/GSR masks"
        )
        assert ret == 0

        self.proc_times["RTP_GSR_mask"] = np.ceil(time.time() - st)

        if progress_bar is not None:
            progress_bar.add_desc("\n")

        for rm_f in rm_fs:
            if rm_f.is_file():
                rm_f.unlink()

        return RTP_mask, GSR_mask

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def warp_template(
        self,
        work_dir,
        alAnat,
        template_f,
        total_ETA,
        progress_bar=None,
        ask_cmd=False,
        overwrite=False,
    ):
        """
        Register the template image to alAnat with ANTs (ants_run.py).

        Parameters
        ----------
        work_dir : Path
            Working directory.
        alAnat : Path
            Anatomy image aligned to the functional image.
        template_f : str or Path
            Template image file.
        total_ETA : float
            Expected time (seconds) of all the processing steps, for the
            progress bar.
        progress_bar : rtpspy.ui.DlgProgressBar, optional
            Progress bar dialog. The default is None.
        ask_cmd : bool, optional
            Allow editing the command line. The default is False.
        overwrite : bool, optional
            Overwrite existing files. When this is False and the results
            exist, the process is skipped. The default is False.

        Returns
        -------
        warp_params : list of str
            Transform files [warp, affine] from the template to alAnat.
            None on failure or cancel.
        """
        self._print_step_header("ANTs registration", progress_bar)

        aff_f = work_dir / "template2orig_0GenericAffine.mat"
        wrp_f = work_dir / "template2orig_1Warp.nii.gz"
        if aff_f.is_file() and wrp_f.is_file() and not overwrite:
            self._report_existing([aff_f, wrp_f], progress_bar, total_ETA,
                                  "ANTs")
            return [str(wrp_f), str(aff_f)]

        fix_f = os.path.relpath(alAnat.absolute(), work_dir)
        move_f = os.path.relpath(Path(template_f).absolute(), work_dir)
        outprefix = "template2orig_"

        ants_run = shutil.which("ants_run.py")
        if ants_run is None:
            ants_run = Path(__file__).absolute().parent / "ants_run.py"
            ants_run = f"python3 {ants_run}"

        cmd = f"{ants_run} registration -f {fix_f} -m {move_f}"
        cmd += f" -o {outprefix} -v"

        if ask_cmd:
            cmd, okflag = self._edit_command(cmdTxt=cmd)
            if not okflag:
                return None

        try:
            st = time.time()
            proc = self._spawn(cmd, work_dir)
        except Exception as e:
            errmsg = str(e) + "\n"
            errmsg += f"'{cmd}' failed."
            self._logger.error(errmsg)
            self.err_popup(errmsg)
            return None

        ret = self._show_proc_progress(
            proc,
            progress_bar,
            msgTxt="ANTs registration ...",
            ETA=self.proc_times["ANTs"],
            total_ETA=total_ETA,
        )
        if ret != 0:
            return None

        self.proc_times["ANTs"] = np.ceil(time.time() - st)

        if progress_bar is not None:
            progress_bar.add_desc("\n")

        return [str(wrp_f), str(aff_f)]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ants_warp_resample(
        self,
        work_dir,
        move_f,
        fix_f,
        transformlist,
        res_master_f=None,
        suffix=None,
        interpolator="nearestNeighbor",
        imagetype=0,
        progress_bar=None,
        overwrite=False,
    ):
        """
        Apply ANTs transforms to warp move_f into the fix_f space, and
        resample it to res_master_f if given.

        Parameters
        ----------
        work_dir : Path
            Working directory.
        move_f : str or Path
            Image file to be warped.
        fix_f : Path
            Target image file.
        transformlist : list
            ANTs transform files (e.g., the output of warp_template).
        res_master_f : str or Path, optional
            Master image file to which the warped image is resampled.
        suffix : str, optional
            Suffix added to the move_f name for the output file. The default
            (None) is '_alFunc' when the image is resampled to res_master_f
            (the functional grid), and '_alAnat' otherwise (the fix_f grid,
            e.g., alAnat).
        interpolator : str ['nearestNeighbor', 'linear', 'bSpline'], optional
            Interpolation method. The default is 'nearestNeighbor'.
        imagetype : int, optional
            ANTs image type code: 0/1/2/3 for scalar/vector/tensor/
            time-series. The default is 0.
        progress_bar : rtpspy.ui.DlgProgressBar, optional
            Progress bar dialog. The default is None.
        overwrite : bool, optional
            Overwrite existing files. When this is False and the results
            exist, the process is skipped. The default is False.

        Returns
        -------
        warped_f : Path
            Warped move_f file. None on failure.
        """
        self._print_step_header("Apply warp", progress_bar, "Apply Warp")

        if suffix is None:
            suffix = "_alFunc" if res_master_f is not None else "_alAnat"

        move_name = Path(move_f).name
        if ".nii" in move_name:
            warped_name = move_name.replace(".nii", f"{suffix}.nii")
        else:
            # AFNI BRIK: write a NIfTI file not to overwrite the input
            warped_name = re.sub(
                r"(\+\w+)?\.(HEAD|BRIK)(\.gz)?$", "", move_name
            ) + f"{suffix}.nii.gz"
        warped_f = work_dir / warped_name
        tmp_out = work_dir / f"rm_tmp_{warped_name}"
        if warped_f.is_file() and not overwrite:
            self._report_existing([warped_f], progress_bar)
        else:
            st = time.time()
            try:
                print(f"Apply transform to {move_name} ...")

                # Warp move_f to fix_f
                fix_img = ants.image_read(str(fix_f))
                move_img = ants.image_read(str(move_f))
                transformlist = [str(t) for t in transformlist]
                out_img = ants.apply_transforms(
                    fix_img,
                    move_img,
                    transformlist,
                    imagetype=imagetype,
                    interpolator=interpolator,
                    verbose=False,
                )

                if res_master_f is not None:
                    # Resample the warped image to res_master_f
                    ants.image_write(out_img, str(tmp_out))
                    cmd = f"3dresample -overwrite -master {res_master_f}"
                    cmd += f" -input {tmp_out} -prefix {warped_f}"
                    ret = self._show_cmd_progress(
                        cmd,
                        progress_bar,
                        msgTxt=f"Warp and resample {move_f}",
                        desc=f"\n== Resample {move_f} mask ==",
                    )
                    assert ret == 0, f"Failed to resample {move_f}.\n"
                else:
                    ants.image_write(out_img, str(warped_f))

            except Exception as e:
                errmsg = str(e) + "\n"
                errmsg += "'ants.apply_transforms' failed."
                self._logger.error(errmsg)
                self.err_popup(errmsg)
                return None

            finally:
                # Delete the temporary file
                tmp_out.unlink(missing_ok=True)

            assert warped_f.is_file()
            print(f"Done. (took {np.ceil(time.time() - st)} s)\n")

        return warped_f
