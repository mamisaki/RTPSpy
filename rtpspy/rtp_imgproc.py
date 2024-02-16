#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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

import numpy as np
import nibabel as nib
import ants

from PyQt5 import QtWidgets
from .rtp_common import RTP
from .fast_seg import FastSeg


# %% RtpImgProc class =======================================================
class RtpImgProc(RTP):
    """
    Image processing utility class.

    """

    # Interpolation option for antsApplyTransforms at resampleing the
    # warped ROI: ['linear'|'nearestNeighbor'|'bSpline']
    ROI_resample_opts = ['nearestNeighbor', 'linear', 'bSpline']

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, main_win=None):
        super().__init__()  # call __init__() in RTP class

        # --- Initialize parameters -------------------------------------------
        self.main_win = main_win

        self.fastSeg_batch_size = 1

        # Set the default processing times for proc_anat progress bar
        self.proc_times = {
            "FastSeg": 100,
            "SkullStrip": 100,
            "AlAnat": 40,
            "RTP_GSR_mask": 3,
            "ANTs": 120,
            "ApplyWarp": 10,
            "Resample_WM_mask": 1,
            "Resample_Vent_mask": 1,
            "Resample_aseg_mask": 1
            }

    # --- Internal utility methods --------------------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _edit_command(self, labelTxt='Commdand line:', cmdTxt=''):
        '''
        Show command line edit dialog

        Parameters
        ----------
        labelTxt : str, optional
            Description about the command. The default is 'Commdand line:'.
        cmdTxt : str, optional
            Initial command line. The default is ''.

        Returns
        -------
        cmd : str
            Edited command.
        okflag : bool
            OK is pressed or not (canceled).

        '''

        dlg = QtWidgets.QInputDialog(self.main_win)
        dlg.setInputMode(QtWidgets.QInputDialog.TextInput)
        dlg.setLabelText(labelTxt)
        dlg.setTextValue(cmdTxt)
        dlg.resize(640, 100)
        okflag = dlg.exec_()
        cmd = dlg.textValue()

        return cmd, okflag

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _show_cmd_progress(self, cmd, progress_bar=None, msgTxt='', desc=''):
        """
        Run a command and print its output. This should be used for a cammand
        that can finish in short time.

        For a command taking long time, a child process should be opened by
        subprocess.Popen, then use self._show_proc_progress to show the
        progress online wihtout blocking other process.

        Parameters
        ----------
        cmd : str
            Command line.
        progress_bar : rtp_common.DlgProgressBar, optional
            Progress_bar dialog object. The default is None.
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
            1: with error.

        """

        if progress_bar is not None:
            if len(msgTxt):
                progress_bar.set_msgTxt(msgTxt)

            if len(desc):
                progress_bar.add_desc(desc)

        try:
            ostr = subprocess.check_output(cmd, stderr=subprocess.STDOUT,
                                           shell=True).decode()
            if progress_bar is not None:
                progress_bar.add_desc(ostr)
            else:
                print(ostr)

        except Exception:
            errmsg = f"Failed execution:\n{cmd}"
            self._logger.error(errmsg)
            self.err_popup(errmsg)

            if hasattr(self, 'ui_procAnat_btn'):
                self.ui_procAnat_btn.setEnabled(True)
            # if progress_bar is not None and progress_bar.isVisible():
            #    progress_bar.close()
            return 1

        return 0

        if progress_bar is not None:
            progress_bar.set_msgTxt("Align anat to func ... done.")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _proc_print_progress(self, proc):
        """
        Pring progress of a spawned child process in stdout.

        Parameters
        ----------
        proc : subprocess.Popen object
            child process object opened by subprocess.Popen.

        Returns
        -------
        proc.exitstatus
            Exit status of the proc.

        """

        while proc.poll() is None:
            try:
                out0 = proc.stdout.read(4).decode()
                out = '\n'.join(out0.splitlines())
                if len(out0) and out0[-1] == '\n':
                    out += '\n'

                print(out, end='')

            except subprocess.TimeoutExpired:
                pass

            if hasattr(self, 'isVisible') and not self.isVisible():
                break

        try:
            out = proc.stdout.read().decode()
            print('\n'.join(out.splitlines()) + '\n')
        except subprocess.TimeoutExpired:
            pass

        proc.terminate()
        return proc.returncode

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _show_proc_progress(self, proc, progress_bar=None, msgTxt='', desc='',
                            ETA=None, total_ETA=None):
        """
        Show progress of a spawned child process in a dialog if available.

        Parameters
        ----------
        proc : subprocess.Popen object
            child process object opened by subprocess.Popen.
        progress_bar : rtp_common.DlgProgressBar, optional
            Progress_bar dialog object. The default is None.
        msgTxt : str, optional
            Message text shown in the progress_bar (under the bar).
            The default is ''.
        desc : str, optional
            Description shown in the progress_bar (in the text box).
            The default is ''.
        ETA : float, optional
            Estimated time of arrival (seconds). This is used for online bar
            update. The default is None.
        total_ETA : float, optional
            Estimated time of arrival (seconds) for all processes if the proc
            is a part of many processes. Only a part of a progress bar in the
            dialog will be updated accordingly. The default is None.

        Returns
        -------
        ret : int
            0: no error.
            -1: with error.

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
                proc, bar_inc=bar_inc, ETA=ETA)

            if not progress_bar.isVisible():
                ret = -1
        else:
            ret = self._proc_print_progress(proc)

        # Check error
        if ret != 0:
            if ret == -1:
                self._logger.info(f"Cancel {msgTxt}")
            else:
                errmsg = f"Failed in {msgTxt}"
                self._logger.error(errmsg)
                self.err_popup(errmsg)

            if hasattr(self, 'ui_procAnat_btn'):
                self.ui_procAnat_btn.setEnabled(True)

        return ret

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def load_image(self, fname):
        """Load image file and retur nibabel Nifti1Image object
        """
        try:
            img = nib.load(fname)

            suffix = Path(fname).suffix
            if suffix == '.gz':
                suffix = Path(Path(fname).stem).suffix

            if suffix == '.nii':
                img_data = img.get_fdata().astype(img.header.get_data_dtype())
                img = nib.Nifti1Image(img_data, affine=img.affine)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errmsg = '{}, {}:{}'.format(
                    exc_type, exc_tb.tb_frame.f_code.co_filename,
                    exc_tb.tb_lineno)
            self._logger.error(str(e) + '\n' + errmsg)
            self.err_popup(errmsg)
            return None

        return img

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def copy_deoblique(self, input, prefix, progress_bar=None):
        oblique = subprocess.check_output(
            shlex.split(f"3dinfo -is_oblique {input}"))
        oblique = int(oblique.decode().rstrip())
        if oblique:
            cmd = f"3dWarp -overwrite -deoblique -prefix {prefix} {input}"
        else:
            cmd = f"3dcopy -overwrite {input} {prefix}"

        # Run cmd
        ret = self._show_cmd_progress(
            cmd, progress_bar, msgTxt=f"Deobliqu {Path(input).name}",
            desc='\n== Deobplique anatomy ==')
        assert ret == 0, f'Failed at deoblique {input}.\n'

    # --- Image processing methos ---------------------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run_fast_seg(self, work_dir, anat_orig, total_ETA, progress_bar=None,
                     overwrite=False):
        """
        Segment anatomy image (anat_orig) with FasSeg to extract brain, white
        matter, and ventricle masks.

        Parameters
        ----------
        work_dir : stirng or Path object
            Wroking directory.
        anat_orig : stirng or Path object
            anaotmy image file in individual's original space.
        progress_bar : rtp_common.DlgProgressBar, optional
            Progress_bar dialog object. The default is None.
        overwrite : bool, optional
            Overwrite existing files. When this is False and results exist,
            the process is skipped. The default is False.

        Returns
        -------
        brain_anat_orig, wm_anat_orig, vent_anat_orig: Path
            Filenames of the segmented images.

        """

        # Set FastSeg output prefix.
        out_prefix = anat_orig.name.replace(
            ''.join(anat_orig.suffixes[-2:]), '')
        out_prefix = out_prefix.replace('+orig', '').replace('+tlrc', '')

        # Print job description.
        descStr = '+' * 70 + '\n'
        descStr += "+++ Brain extraction and WM/Vent segmentation\n"
        if progress_bar is not None:
            progress_bar.set_msgTxt(
                'Brain extraction and WM/Vent segmentation')
        print(descStr)

        # Check if the result files exist.
        brain_anat_orig = work_dir / (out_prefix + '_Brain.nii.gz')
        wm_anat_orig = work_dir / (out_prefix + '_WM.nii.gz')
        vent_anat_orig = work_dir / (out_prefix + '_Vent.nii.gz')
        aseg_anat_orig = work_dir / (out_prefix + '_aseg.nii.gz')
        if not brain_anat_orig.is_file() or not wm_anat_orig.is_file() or \
                not vent_anat_orig.is_file() or \
                not aseg_anat_orig.is_file() or overwrite:
            # Make FastSeg instance
            fastSeg = FastSeg()

            # Prepare files (Convert BRIK to NIfTI)
            in_f, prefix = fastSeg.prep_files(anat_orig,
                                              work_dir / out_prefix)
            subj_dir = prefix

            # Spawn the process
            st = time.time()
            proc, fsSeg_mgz = fastSeg.run_seg_only(
                in_f, prefix, self.fastSeg_batch_size)

            assert proc.returncode is None or proc.returncode == 0, \
                'Failed at FastSeg.\n'

            # Wait for the process to finish with showing the progress.
            ret = self._show_proc_progress(
                proc, progress_bar, msgTxt='FastSeg image segnemtation',
                ETA=self.proc_times['FastSeg'], total_ETA=total_ETA)

            if ret != 0:
                return None

            # make_seg_images
            def show_proc_progress(proc):
                return self._show_proc_progress(
                    proc, progress_bar, msgTxt='FastSeg image segmentation',
                    total_ETA=total_ETA)

            out_fs = fastSeg.make_seg_images(
                in_f, fsSeg_mgz, prefix,
                segs=['Brain', 'WM', 'Vent', 'aseg'],
                show_proc_progress=show_proc_progress)

            if out_fs is None:
                return None

            # Delete woring files
            if subj_dir.is_dir():
                shutil.rmtree(subj_dir)

            # Record processing time
            self.proc_times["FastSeg"] = np.ceil(time.time() - st)

            if progress_bar is not None:
                progress_bar.add_desc('\n')

        else:
            # Report that existing files are used.
            if progress_bar is not None:
                bar_inc = (self.proc_times["FastSeg"]) / total_ETA * 100
                progress_bar.set_value(bar_inc)
                progress_bar.add_desc(
                    f"Use existing files: {brain_anat_orig}\n" +
                    f"                    {wm_anat_orig}\n"
                    f"                    {vent_anat_orig}\n\n")
            else:
                print(f"Use existing files: {brain_anat_orig}\n" +
                      f"                    {wm_anat_orig}\n"
                      f"                    {vent_anat_orig}\n")

        return brain_anat_orig, wm_anat_orig, vent_anat_orig, aseg_anat_orig

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def skullStrip(self, work_dir, anat_orig, total_ETA, progress_bar=None,
                   ask_cmd=False, overwrite=False):
        """
        Skull-stripping anatomy image (anat_orig) with 3dSkullStrip.

        Parameters
        ----------
        work_dir : stirng or Path object
            Wroking directory.
        anat_orig : stirng or Path object
            anaotmy image file in individual's original space.
        progress_bar : rtp_common.DlgProgressBar, optional
            Progress_bar dialog object. The default is None.
        ask_cmd : TYPE, optional
            Allow to edit command line. The default is False.
        overwrite : bool, optional
            Overwrite existing files. When this is False and results exist,
            the process is skipped. The default is False.

        Returns
        -------
        brain_anat_orig : Path
            Filename of the skull-stripped  image.

        """

        # Set output prefix.
        out_prefix = anat_orig.name.replace(
            ''.join(anat_orig.suffixes[-2:]), '')
        out_prefix = out_prefix.replace('+orig', '').replace('+tlrc', '')

        # Print job description.
        descStr = '+' * 70 + '\n'
        descStr += "+++ Brain extraction\nRunning 3dSkullStrip ..."
        if progress_bar is not None:
            progress_bar.set_msgTxt('Brain extraction')
            progress_bar.add_desc(descStr)
        else:
            sys.stdout.write(descStr)

        # Check if the result files exist.
        brain_anat_orig = work_dir / (out_prefix + '_Brain.nii.gz')
        if not brain_anat_orig.is_file() or overwrite:
            # Run the process
            cmd = f"3dSkullStrip -overwrite -input {anat_orig}"
            cmd += f" -prefix {brain_anat_orig}"

            if ask_cmd:
                # Edit the command line.
                labelTxt = 'Commdand line:'
                cmd, okflag = self._edit_command(labelTxt=labelTxt, cmdTxt=cmd)
                if not okflag:
                    return -1

            # Spawn the process
            st = time.time()
            proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, cwd=work_dir)
            assert proc.returncode is None or proc.returncode == 0, \
                f'Failed running skullStrip command, {cmd}.\n'

            # Wait for the process to finish with showing the progress.
            ret = self._show_proc_progress(
                proc, progress_bar, msgTxt='Skull stripping',
                ETA=self.proc_times['SkullStrip'], total_ETA=total_ETA)

            if ret != 0:
                return None

            self.proc_times["SkullStrip"] = np.ceil(time.time() - st)

            if progress_bar is not None:
                progress_bar.add_desc('\n')

        else:
            # Report using existing files.
            if progress_bar is not None:
                bar_inc = (self.proc_times["SkullStrip"]) / total_ETA * 100
                progress_bar.set_value(bar_inc)
                progress_bar.add_desc(
                    f"Use existing file: {brain_anat_orig}\n\n")
            else:
                print(f"Use existing files: {brain_anat_orig}\n")

        assert brain_anat_orig.is_file(), "Failed in skullStrip."

        return brain_anat_orig

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def align_anat2epi(self, work_dir, brain_anat_orig, func_orig, total_ETA,
                       progress_bar=None, ask_cmd=False, overwrite=False):
        """
        Align anatomy image to the base functional image using
        align_epi_anat.py in AFNI.
        https://afni.nimh.nih.gov/pub/dist/doc/program_help/align_epi_anat.py.html

        Parameters
        ----------
        work_dir : stirng or Path object
            Working directory.
        brain_anat_orig : stirng or Path object
            Brain-extracted anatomy image file.
        func_orig : stirng or Path object
            Functional image file.
        progress_bar : rtp_common.DlgProgressBar, optional
            Progress_bar dialog object. The default is None.
        ask_cmd : TYPE, optional
            Allow to edit command line. The default is False.
        overwrite : bool, optional
            Overwrite existing files. When this is False and results exist,
            the process is skipped. The default is False.

        Returns
        -------
        proc : subprocess.Popen object
            child process object opened by subprocess.Popen.

        """

        # Print job description
        descStr = '+' * 70 + '\n'
        descStr += "+++ Align anat to func\n"
        if progress_bar is not None:
            progress_bar.set_msgTxt('Align anat to func')

        print(descStr)

        brain_anat_orig = Path(brain_anat_orig)
        suffs = brain_anat_orig.suffixes
        alAnat = work_dir / \
            (brain_anat_orig.name.replace(''.join(suffs), '_al_func.nii.gz'))
        if not alAnat.is_file() or overwrite:
            if alAnat.is_file():
                alAnat.unlink()

            anat_orig_rel = os.path.relpath(brain_anat_orig, work_dir)
            func_orig_rel = os.path.relpath(func_orig, work_dir)

            cmd = "align_epi_anat.py -overwrite -anat2epi"
            cmd += f" -anat {anat_orig_rel} -epi {func_orig_rel} -epi_base 0"
            cmd += " -suffix _al_func -epi_strip 3dAutomask -anat_has_skull no"
            cmd += f" -master_anat {brain_anat_orig}"
            cmd += " -volreg off -tshift off -giant_move"
            if ask_cmd:
                labelTxt = 'Commdand line: (see '
                labelTxt += \
                    'https://afni.nimh.nih.gov/pub/dist/doc/program_help/'
                labelTxt += 'align_epi_anat.py.html)'
                labelTxt += "\nConsider -ginormous_move or"
                labelTxt += " -partial_coverage option"
                cmd, okflag = self._edit_command(labelTxt=labelTxt, cmdTxt=cmd)
                if not okflag:
                    return None

            # Spawn the process
            st = time.time()  # start time
            proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, cwd=work_dir)
            assert proc.returncode is None or proc.returncode == 0, \
                'Failed at align_anat2epi.\n'

            # Wait for the process to finish with showing the progress.
            ret = self._show_proc_progress(
                proc, progress_bar, msgTxt='Align anat to func',
                ETA=self.proc_times["AlAnat"], total_ETA=total_ETA)
            if ret != 0:
                return None

            # Convert aligned anatomy to NIfTI
            alAnat_f_stem = alAnat.stem.replace('.nii', '')
            alAnat_brik = list(work_dir.glob(alAnat_f_stem + '*.HEAD'))
            if len(alAnat_brik):
                cmd = f"3dAFNItoNIFTI -overwrite -prefix {alAnat}"
                cmd += f" {alAnat_brik[0]}"
                ret = self._show_cmd_progress(
                    cmd, progress_bar, msgTxt='Convert alAnat to NIfTI',
                    desc="++ Convert alAnat to NIfTI\n")

            self.proc_times["AlAnat"] = np.ceil(time.time() - st)
            assert alAnat.is_file()

            if progress_bar is not None:
                progress_bar.add_desc('\n')

        else:
            # Use existing file.
            if progress_bar is not None:
                bar_inc = self.proc_times["AlAnat"] / total_ETA * 100
                bar_val0 = progress_bar.progBar.value()
                progress_bar.set_value(bar_val0+bar_inc)
                progress_bar.add_desc(f"Use existing file: {alAnat}\n\n")
            else:
                print(f"Use existing file: {alAnat}\n")

        return alAnat

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def erode_ROI(self, work_dir, src_f, out_f, erode=0, ask_cmd=False):
        """
        Erode ROI with 3dmask_tool. WM and Vent masks will be eroded to exclude
        partial GM voxles.

        Parameters
        ----------
        work_dir : stirng or Path object
            Working directory.
        src_f : stirng or Path object
            source file.
        out_f : stirng or Path object
            Output file.
        erode : TYPE, optional
            Number of voxels to erode. Negative value means dilating.
            The default is 0.
        ask_cmd : TYPE, optional
            Allow to edit command line. The default is False.

        Returns
        -------
        proc : subprocess.Popen object
            child process object opened by subprocess.Popen.

        """

        src = Path(os.path.relpath(src_f, work_dir))
        out_f = Path(os.path.relpath(out_f, work_dir))

        # --- Erode mask ---
        cmd = f"3dmask_tool -overwrite -input {src} -dilate_input {-erode}"
        cmd += f" -prefix {out_f}"
        if ask_cmd:
            labelTxt = 'Commdand line: (see '
            labelTxt += 'https://afni.nimh.nih.gov/pub/dist/doc/'
            labelTxt += 'program_help/3dmask_tool.html)'
            cmd, okflag = self._edit_command(labelTxt=labelTxt, cmdTxt=cmd)
            if not okflag:
                return None

        try:
            proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, cwd=work_dir)
            return proc
        except Exception as e:
            errmsg = str(e)+'\n'
            errmsg += f"'{cmd}' failed."
            self._logger.error(errmsg)
            self.err_popup(errmsg)
            return None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def resample_segmasks(self, work_dir, seg_anat_f, segname, erode,
                          func_orig, total_ETA, aff1D_f=None,
                          progress_bar=None, ask_cmd=False, overwrite=False):

        # Print job description
        descStr = '+' * 70 + '\n'
        descStr += f"+++ Resample {segname} mask\n"
        if progress_bar is not None:
            progress_bar.set_msgTxt(f"Align {segname} to func")
            progress_bar.add_desc(descStr)
        else:
            sys.stdout.write(descStr)

        prefix = seg_anat_f.name.replace(''.join(seg_anat_f.suffixes[-2:]), '')
        prefix = prefix.replace('+orig', '').replace('+tlrc', '')

        seg_al_f = work_dir / (prefix + '_al_func.nii.gz')
        if not seg_al_f.is_file() or overwrite:
            if erode != 0:
                # -- Erode the segmentation mask ---
                out_f0 = seg_al_f.parent / ('rm.1.' + seg_al_f.name)

                # Spawn the processseg_files
                st = time.time()
                proc = self.erode_ROI(work_dir, seg_anat_f, out_f0,
                                      erode=erode, ask_cmd=ask_cmd)
                assert proc.returncode is None or proc.returncode == 0, \
                    f'Failed at resample_segmasks {segname}.\n'

                # Wait for the process to finish with showing the progress.
                ret = self._show_proc_progress(
                    proc, progress_bar, desc=f"\n== Erode {segname} ==\n")
                if ret != 0:
                    return None
            else:
                st = time.time()
                out_f0 = seg_anat_f

            if aff1D_f is not None:
                # --- deoblique ---
                out_f1 = seg_al_f.parent / ('rm.2.' + seg_al_f.name)
                cmd = f"3dWarp -deoblique -prefix {out_f1} {out_f0} && "

                # --- Apply mat.aff12.1D ---
                cmd += f"3dAllineate -overwrite -final NN -input {out_f1}"
                cmd += f" -1Dmatrix_apply {aff1D_f}"
                cmd += f" -master {func_orig} -prefix {seg_al_f}"
            else:
                # --- Resample in func_orig space ---
                cmd = f"3dresample -overwrite -rmode NN -master {func_orig}"
                cmd += f" -prefix {seg_al_f} -input {out_f0}"

            # Run cmd
            ret = self._show_cmd_progress(
                cmd, progress_bar, msgTxt=f"Resample {segname} mask",
                desc=f'\n== Resample {segname} mask ==')
            assert ret == 0, f'Failed at resample_segmasks {segname}.\n'

            for rmf in work_dir.glob('rm*'):
                rmf.unlink()

            assert seg_al_f.is_file(), \
                f'Failed at resample_segmasks {segname}.\n'

            self.proc_times[f'Resample_{segname}_mask'] = \
                np.ceil(time.time() - st)

            if progress_bar is not None:
                progress_bar.add_desc('\n')

        else:
            # Use existing file
            if progress_bar is not None:
                bar_inc = (
                    self.proc_times[f'Resample_{segname}_mask']
                    / total_ETA) * 100
                bar_val0 = progress_bar.progBar.value()
                progress_bar.set_value(bar_val0+bar_inc)
                progress_bar.add_desc(
                    f"Use existing file: {seg_al_f}\n\n")
            else:
                print(f"Use existing file: {seg_al_f}\n")

        return seg_al_f

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def make_RTP_GSR_masks(self, work_dir, func_orig, total_ETA, ref_vi=0,
                           alAnat=None, progress_bar=None, ask_cmd=False,
                           overwrite=None):
        """
        Make masks for RTP and Global Signal Regression (GSR).
        3dAutomask is used to make a functional image mask.
        If alAnat is None, automask (3sAutomask in AFNI) for func_orig is used
        for both the RTP and GSR masks.
        If alAnat is not None, RTP mask is a union of function automask and
        zero-out alAnat mask, and GSR mask is an intersect of them.

        RTP mask will be used as a mask in SMOOTH and REGRESS.
        GSR mask will be used in REGRESS if GSR is enabled.

        Options
        -------
        work_dir : stirng or Path object
            Wroking directory.
        func_orig : stirng or Path object
            Functional image file in individual's original space.
        ref_vi : int
            Reference volume index in func_orig
        alAnat: stirng or Path object, optional
            Anatomy image file aligned to func_orig
        progress_bar : rtp_common.DlgProgressBar, optional
            Progress_bar dialog object. The default is None.
        ask_cmd : TYPE, optional
            Allow to edit command line. The default is False.
        overwrite : bool, optional
            Overwrite existing files. When this is False and results exist,
            the process is skipped. The default is False.

        Return
        ------
        RTP_mask, GSR_mask : Path or str

        """

        rm_fs = []
        RTP_mask = work_dir / "RTP_mask.nii.gz"
        GSR_mask = work_dir / 'GSR_mask.nii.gz'

        # Print job description
        descStr = '+' * 70 + '\n'
        descStr += "+++ Make GSR mask\n"
        if progress_bar is not None:
            progress_bar.set_msgTxt('Make RTP, GSR masks')
            progress_bar.add_desc(descStr)
        else:
            if self.verb:
                sys.stdout.write(descStr)

        if not RTP_mask.is_file() or not GSR_mask.is_file() or overwrite:
            func_orig = Path(func_orig)
            fbase = re.sub(r'\+.*', '', func_orig.name)
            func_mask = work_dir / f"automask_{fbase}.nii.gz"
            cmd = f"3dAutomask -overwrite -prefix {func_mask} {func_orig} && "
            rm_fs.append(func_mask)

            if alAnat is not None and Path(alAnat).is_file():
                alAnat = Path(alAnat)
                fbase = re.sub(r'\+.*', '', alAnat.name)
                temp_out = work_dir / 'rm.anat_mask_tmp.nii.gz'

                anat_mask = work_dir / f"anatmask_{fbase}.nii.gz"
                cmd += f"3dmask_tool -overwrite -input {alAnat}"
                cmd += f" -prefix {temp_out} -frac 0.0 -fill_holes && "

                cmd += f"3dresample -overwrite -rmode NN -master {func_orig}"
                cmd += f" -prefix {anat_mask} -input {temp_out} && "

                # RTP mask : union of func automask and anat mask
                cmd += "3dmask_tool -overwrite"
                cmd += f" -input {anat_mask} {func_mask}"
                cmd += f" -prefix {RTP_mask} -frac 0.0 && "

                # GSR mask : intersect of func automask and anat mask
                cmd += f"3dmask_tool -overwrite -input {func_mask}"
                cmd += f" {anat_mask} -prefix {GSR_mask} -frac 1.0"

                rm_fs.append(anat_mask)
                rm_fs.append(temp_out)
            else:
                # RTP and GSR masks :func automask
                cmd += " && 3dAFNItoNIFTI -overwrite"
                cmd += f" -prefix {RTP_mask} {func_mask} && "
                cmd = f"cp {GSR_mask} {RTP_mask}"

            # Print job description
            descStr = '+' * 70 + '\n'
            descStr += "+++ Make RTP, GSR masks\n"
            if progress_bar is not None:
                progress_bar.set_msgTxt('Make RTP and GSR masks')
                progress_bar.add_desc(descStr)
            else:
                sys.stdout.write(descStr)

            if ask_cmd:
                labelTxt = 'Commdand line: (see '
                labelTxt += 'https://afni.nimh.nih.gov/pub/dist/doc/'
                labelTxt += 'program_help/3dAutomask.html)'
                cmd, okflag = self._edit_command(labelTxt=labelTxt, cmdTxt=cmd)
                if not okflag:
                    return None

            # Spawn the process
            st = time.time()  # start time
            proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, cwd=work_dir)
            assert proc.returncode is None or proc.returncode == 0, \
                'Failed at RTP/GSR mask creation.\n'

            # Wait for the process to finish with showing the progress.
            ret = self._show_cmd_progress(
                cmd, progress_bar=progress_bar, msgTxt='Making RTP/GSR masks')
            assert ret == 0

            self.proc_times["RTP_GSR_mask"] = np.ceil(time.time() - st)

            if progress_bar is not None:
                progress_bar.add_desc('\n')

        else:
            # Use existing file.
            if progress_bar is not None:
                bar_inc = (self.proc_times["RTP_GSR_mask"]) / total_ETA * 100
                bar_val0 = progress_bar.progBar.value()
                progress_bar.set_value(bar_val0 + bar_inc)
                progress_bar.add_desc(
                    f"Use existing file: {RTP_mask} and {GSR_mask}\n\n")
            else:
                print(f"Use existing files: {RTP_mask} and {GSR_mask}\n")

        for rm_f in rm_fs:
            if rm_f.is_file():
                rm_f.unlink()

        return RTP_mask, GSR_mask

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def warp_template(self, work_dir, alAnat, template_f, total_ETA,
                      progress_bar=None, ask_cmd=False, overwrite=False):

        # Print job description
        descStr = '+' * 70 + '\n'
        descStr += "+++ ANTs registration\n"
        if progress_bar is not None:
            progress_bar.set_msgTxt('ANTs registraion')
        print(descStr)

        # --- Warp template to alAnat -----------------------------------------
        aff_f = work_dir / 'template2orig_0GenericAffine.mat'
        wrp_f = work_dir / 'template2orig_1Warp.nii.gz'
        if not aff_f.is_file() or not wrp_f.is_file() or overwrite:
            fix_f = os.path.relpath(alAnat.absolute(), work_dir)
            move_f = os.path.relpath(Path(template_f).absolute(),
                                     work_dir)
            outprefix = 'template2orig_'

            # Prepare the command
            ants_run = shutil.which("ants_run.py")
            if ants_run is None:
                ants_run = Path(__file__).absolute().parent / 'ants_run.py'
                ants_run = f"python3 {ants_run}"

            cmd = f"{ants_run} registration -f {fix_f} -m {move_f}"
            cmd += f" -o {outprefix} -v"

            if ask_cmd:
                labelTxt = 'Commdand line:'
                cmd, okflag = self._edit_command(labelTxt=labelTxt, cmdTxt=cmd)
                if not okflag:
                    return None
                # Spawn the process

            try:
                st = time.time()
                proc = subprocess.Popen(
                    shlex.split(cmd), stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, cwd=work_dir)

                assert proc.returncode is None or proc.returncode == 0, \
                    'Failed at ANTs registration.\n'

            except Exception as e:
                errmsg = str(e)+'\n'
                errmsg += "'{}' failed.".format(cmd)
                self._logger.error(errmsg)
                self.err_popup(errmsg)
                return -1

            # Wait for the process to finish with showing the progress.
            ret = self._show_proc_progress(
                proc, progress_bar, msgTxt="ANTs registraion ...",
                ETA=self.proc_times["ANTs"], total_ETA=total_ETA)
            if ret != 0:
                return None

            self.proc_times["ANTs"] = np.ceil(time.time() - st)

            if progress_bar is not None:
                progress_bar.add_desc('\n')

            # try:
            #     # Warp template to anat
            #     print("Running antsRegistrationSyNQuickRepro[b] ...")

            #     st = time.time()
            #     anat_img = ants.image_read(str(alAnat))
            #     template_img = ants.image_read(str(template_f))
            #     t2a_reg = ants.registration(
            #         anat_img, template_img,
            #         'antsRegistrationSyNQuickRepro[b]',
            #         verbose=False)

            #     shutil.copy(t2a_reg['fwdtransforms'][0], wrp_f)
            #     shutil.copy(t2a_reg['fwdtransforms'][1], aff_f)
            #     invwrp_f = work_dir / 'template2orig_1InverseWarp.nii.gz'
            #     shutil.copy(t2a_reg['invtransforms'][1], invwrp_f)

            # except Exception as e:
            #     errmsg = str(e)+'\n'
            #     errmsg += "'ants.registration' failed."
            #     self._logger.error(errmsg)
            #     self.err_popup(errmsg)
            #     return -1

            # self.proc_times["ANTs"] = np.ceil(time.time() - st)
            # if progress_bar is not None:
            #     bar_inc = self.proc_times["ANTs"] / total_ETA * 100
            #     bar_val0 = progress_bar.progBar.value()
            #     progress_bar.set_value(bar_val0+bar_inc)

            # print(f'Done (took {self.proc_times["ANTs"]} s)\n')

        else:
            # Use existing file
            if progress_bar is not None:
                bar_inc = self.proc_times["ANTs"] / total_ETA * 100
                bar_val0 = progress_bar.progBar.value()
                progress_bar.set_value(bar_val0+bar_inc)
                progress_bar.add_desc(
                    f"Use existing files: {aff_f}\n" +
                    f"                    {wrp_f}\n\n")
            else:
                print(f"Use existing files: {aff_f}\n" +
                      f"                    {wrp_f}\n")

        warp_params = [str(wrp_f), str(aff_f)]
        return warp_params

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ants_warp_resample(self, work_dir, fix_f, move_f, total_ETA,
                           transformlist, interpolator='nearestNeighbor',
                           imagetype=0, progress_bar=None, ask_cmd=False,
                           overwrite=False):
        """
        Apply ANTs warping with resampling to the fix_f image resolution.

        Parameters
        ----------
        work_dir : stirng or Path object
            Wroking directory.
        fix_f : stirng or Path object
            Target image file.
        out_f : stirng or Path object
            Output file.
        total_ETA : float
            Total estimated time arrival.
        transformlist : list
            List of ANTs transform files, which should be made at
            self.ants_registration.
        interpolator : str ['nearestNeighbor', 'linear', 'bSpline'], optional
            Resampling interpolation option. The default is 'nearestNeighbor'.
        imagetype : int, optional
            image type code for ANTs. 0/1/2/3 mapping to
            scalar/vector/tensor/time-series. The default is 0.
        progress_bar : rtp_common.DlgProgressBar, optional
            Progress_bar dialog object. The default is None.
        ask_cmd : TYPE, optional
            Allow to edit command line. The default is False.
        overwrite : bool, optional
            Overwrite existing files. When this is False and results exist,
            the process is skipped. The default is False.

        Returns
        -------
        warped_f : Path object
            Warped move_f file.

        """
        # Print job description
        descStr = '+' * 70 + '\n'
        descStr += "+++ Apply warp\n"
        if progress_bar is not None:
            progress_bar.set_msgTxt('Apply Warp')
        print(descStr)

        warped_f = work_dir / \
            Path(move_f).name.replace('.nii', '_inOrig.nii')
        if not warped_f.is_file() or overwrite:
            st = time.time()

            try:
                # Apply warp with resampling in fix_f space
                print(f'Apply transform to {move_f.name} ...')

                fix_img = ants.image_read(str(fix_f))
                move_img = ants.image_read(str(move_f))
                out_img = ants.apply_transforms(
                    fix_img, move_img, transformlist, imagetype=imagetype,
                    interpolator=interpolator, verbose=False)
                ants.image_write(out_img, str(warped_f))
            except Exception as e:
                errmsg = str(e)+'\n'
                errmsg += "'ants.apply_transforms' failed."
                self._logger.error(errmsg)
                self.err_popup(errmsg)
                return None

            self.proc_times['ApplyWarp'] = np.ceil(time.time() - st)
            if progress_bar is not None:
                bar_inc = self.proc_times["ApplyWarp"] / total_ETA * 100
                bar_val0 = progress_bar.progBar.value()
                progress_bar.set_value(bar_val0+bar_inc)

            assert warped_f.is_file()
            print(f"Done. (took {self.proc_times['ApplyWarp']} s)\n")

        else:
            # Use existing file
            if progress_bar is not None:
                bar_inc = (self.proc_times['ApplyWarp']
                           / total_ETA) * 100
                bar_val0 = progress_bar.progBar.value()
                progress_bar.set_value(bar_val0+bar_inc)
                progress_bar.add_desc(
                    f"Use existing file: {warped_f}\n\n")
            else:
                print(f"Use existing file: {warped_f}\n")

        # Clean rm.* files
        for rmf in work_dir.glob('rm*'):
            rmf.unlink()

        return warped_f
