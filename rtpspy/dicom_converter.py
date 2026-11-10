#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert DICOM files to NIfTI (and AFNI BRIK) with dcm2niix.

mmisaki@libr.net
"""

# %% import ===================================================================
from pathlib import Path
import logging
import subprocess
import shlex
import sys
import traceback
import shutil
import re
import os
import time

import numpy as np
import pandas as pd
import nibabel as nib
import pydicom
import tempfile


# %% DicomConverter ===========================================================
class DicomConverter():
    """List DICOM files by series and convert each series to NIfTI."""

    def __init__(self, study_prefix='P', cmd_pipe=None):
        self._logger = logging.getLogger('DicomConverter')
        self.study_prefix = study_prefix

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _make_path_safe(self, input_string, max_length=255):
        # Replace each run of characters that are not safe for paths with
        # one underscore
        cleaned_string = re.sub(r'[\\/:"*?<>|]+', '_', input_string)

        # Replace spaces and other non-alphanumeric characters with underscores
        cleaned_string = re.sub(r'[^a-zA-Z0-9._-]', '_', cleaned_string)

        # Ensure the resulting string is within length limits
        if len(cleaned_string) > max_length:
            cleaned_string = cleaned_string[:max_length]

        return cleaned_string

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def rt_convert_dicom(
        self,
        dicom_dir,
        out_dir,
        make_brik=False,
        rt_mrib_com=None,
        overwrite=False
    ):
        """Convert the series with the lowest series number in dicom_dir
        (used at the end of a real-time scan).

        The files in dicom_dir are copied to a temporary directory and to
        'out_dir / dicom'. The source files of the series that were copied
        are then removed, and the series is converted to out_dir.
        """
        if not dicom_dir.is_dir():
            return

        try:
            dicom_dst = out_dir / 'dicom'
            if not dicom_dst.is_dir():
                os.makedirs(dicom_dst)

            # Copy DICOM files to tmp for processing
            with tempfile.TemporaryDirectory(
                prefix="dicom_converter_"
            ) as tmp_dicom_dir:
                self._logger.debug(
                    f"Copy DICOM files from {dicom_dir} to {tmp_dicom_dir}"
                )
                cmd = f"rsync -auz {dicom_dir}/ {tmp_dicom_dir}/"
                try:
                    subprocess.check_call(shlex.split(cmd))
                except Exception as e:
                    errstr = str(e) + "\n" + traceback.format_exc()
                    sys.stderr.write(errstr)
                    return

                # Copy the same files from tmp to dicom_dst, so that every
                # file processed (and removed from the source below) is kept
                # in dicom_dst. Files arriving at dicom_dir after the copy
                # above stay in the source.
                self._logger.debug(
                    f"Copy DICOM files from {tmp_dicom_dir} to {dicom_dst}"
                )
                cmd = f"rsync -au {tmp_dicom_dir}/ {dicom_dst}/"
                try:
                    subprocess.check_call(shlex.split(cmd))
                except Exception as e:
                    errstr = str(e) + "\n" + traceback.format_exc()
                    sys.stderr.write(errstr)
                    return

                # Get the list of DICOM files
                dcm_info = self._list_dicom_files(
                    tmp_dicom_dir, out_dir, rt_mode=True)

                # Clean the source files
                self._logger.debug(
                    f"Remove source DICOM files in {dicom_dir}"
                )
                ser_nr = dcm_info['SeriesNumber'].min()
                rm_files = dcm_info[dcm_info['SeriesNumber'] == ser_nr].index
                for ff in rm_files:
                    src_f = dicom_dir / Path(ff).name
                    # Remove only the files kept in dicom_dst
                    if src_f.is_file() and (dicom_dst / src_f.name).is_file():
                        src_f.unlink()

                # Process files
                self.convert_dicom(
                    dcm_info,
                    tmp_dicom_dir,
                    out_dir,
                    ser_nrs=[ser_nr],
                    make_brik=make_brik,
                    rt_mrib_com=rt_mrib_com,
                    overwrite=overwrite
                )

        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            self._logger.error(errstr)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _list_dicom_files(self, dicom_dir, out_dir, timeout=3, rt_mode=False):
        """List the DICOM files in dicom_dir with their header information.

        Unless rt_mode is True, the list is saved in
        'out_dir / dicom / DICOM_INFO.csv' and reused at the next call, so
        that only new files are read.

        Returns
        -------
        pandas.DataFrame
            File information indexed by the filename.
        """
        self._logger.info(f'List DICOM files in {dicom_dir} ...')

        # --- Initialize the file list ---
        dicom_dir = Path(dicom_dir)

        dcm_info_f = Path(out_dir) / 'dicom' / 'DICOM_INFO.csv'
        if not rt_mode and dcm_info_f.is_file():
            dcm_info = pd.read_csv(dcm_info_f, index_col=0)
            # Remove non-existent files
            fmask = [(Path(out_dir) / 'dicom' / ff).is_file()
                     for ff in dcm_info.index]
            dcm_info = dcm_info[fmask]
        else:
            # Initialize the list
            dcm_info = pd.DataFrame(
                columns=('SOPInstanceUID', 'Patient', 'SeriesNumber',
                         'SeriesDescription', 'AcquisitionDateTime',
                         'InstanceNumber', 'IsImage',
                         'NumberOfTemporalPositions'))
            dcm_info.index.name = 'Filename'

        # --- Update the DICOM file list ---
        all_files = list(dicom_dir.glob('*.dcm'))

        # List new files
        exist_fs = list(dcm_info.index)
        all_fs = [str(ff) for ff in all_files]
        new_flist = sorted(np.setdiff1d(all_fs, exist_fs))

        # Add new files in the list
        for ff in new_flist:
            # Read DICOM headers
            try:
                dcm = pydicom.dcmread(ff)
            except Exception:
                # Not a DICOM file
                self._logger.debug(f"{ff} is not a DICOM file.")
                continue

            # For an image DICOM, wait until the pixel data is complete
            is_image = 'PixelData' in dcm
            st = time.time()
            read_fail = False
            while is_image:
                try:
                    # Verify pixel data is complete by accessing it
                    _ = dcm.pixel_array
                    read_fail = False
                    break
                except Exception:
                    # File is incomplete or corrupted
                    read_fail = True

                if (time.time() - st) > timeout:
                    break

                # Re-read the file
                time.sleep(0.1)
                try:
                    dcm = pydicom.dcmread(ff)
                except Exception:
                    pass

            if read_fail:
                self._logger.warning(f"Failed to read pixel data in {ff}")
                continue

            # Append file info
            try:
                addrow = pd.Series()
                addrow['SOPInstanceUID'] = str(dcm.SOPInstanceUID)
                patient = str(dcm.PatientName).split('^')
                if re.match(r'\w\w\d\d\d', patient[0]):
                    # Subject ID format of two letters and three digits
                    addrow['Patient'] = patient[0]
                else:
                    addrow['Patient'] = '_'.join(patient)
                addrow['SeriesNumber'] = int(dcm.SeriesNumber)
                addrow['SeriesDescription'] = dcm.SeriesDescription
                if 'AcquisitionDateTime' in dcm:
                    acq_dt = dcm.AcquisitionDateTime
                else:
                    acq_dt = dcm.get('ContentDate', '') + \
                        dcm.get('ContentTime', '')
                try:
                    addrow['AcquisitionDateTime'] = float(acq_dt)
                except (TypeError, ValueError):
                    addrow['AcquisitionDateTime'] = np.nan
                addrow['InstanceNumber'] = int(dcm.InstanceNumber)
                addrow['IsImage'] = is_image
                if hasattr(dcm, 'NumberOfTemporalPositions'):
                    nt = int(dcm.NumberOfTemporalPositions)
                else:
                    nt = 0
                addrow['NumberOfTemporalPositions'] = nt
            except Exception as e:
                self._logger.warning(f"Failed to read DICOM header {ff}: {e}")
                continue
            dcm_info.loc[ff, :] = addrow

        if len(new_flist):
            # Rows without SeriesNumber (e.g., non-DICOM files in an old
            # list) cannot be converted to int.
            dcm_info = dcm_info[dcm_info.SeriesNumber.notna()]
            dcm_info.sort_values(
                ['SeriesNumber', 'IsImage', 'AcquisitionDateTime'],
                inplace=True)
            dcm_info['SeriesNumber'] = dcm_info.SeriesNumber.astype(int)
            dcm_info['IsImage'] = dcm_info.IsImage.astype(bool)

            if not rt_mode:
                # Save the list
                if not dcm_info_f.parent.is_dir():
                    os.makedirs(dcm_info_f.parent)
                dcm_info.to_csv(dcm_info_f)

        return dcm_info

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def convert_dicom(
        self,
        dcm_info,
        dicom_dir,
        out_dir,
        ser_nrs=None,
        make_brik=False,
        rt_mrib_com=None,
        overwrite=False
    ):
        """Convert each series in dcm_info to NIfTI in out_dir with dcm2niix.

        Parameters
        ----------
        dcm_info : pandas.DataFrame
            File list made by _list_dicom_files.
        dicom_dir : Path
            DICOM directory (used only for the log message).
        out_dir : Path
            Output directory.
        ser_nrs : list of int, optional
            Series numbers to convert. If None, all series are converted.
        make_brik : bool, optional
            Also make AFNI BRIK files.
        rt_mrib_com : RPCSocketCom, optional
            If given, the NIfTI file is opened on the MRI browser.
        overwrite : bool, optional
            Overwrite existing files.
        """
        self._logger.debug(f'Process DICOM files in {dicom_dir} ...')

        dicom_dir = Path(dicom_dir)
        out_dir = Path(out_dir)

        # Process each series
        created_nii = {}

        if ser_nrs is None:
            ser_nrs = dcm_info.SeriesNumber.unique()

        for ser in ser_nrs:
            ser_df = dcm_info[(dcm_info.SeriesNumber == ser) &
                              dcm_info.IsImage]
            if len(ser_df) == 0:
                continue

            sub = ser_df.Patient.iloc[0]
            ses = out_dir.name.replace('_', '')
            serDesc = ser_df.SeriesDescription.iloc[0]

            # Set nii filename
            out_fname = f"sub-{sub}_ses-{ses}_ser-{int(ser):02d}"
            if not pd.isnull(serDesc) and len(serDesc):
                out_fname += f"_desc-{serDesc}"
            out_fname = self._make_path_safe(out_fname)

            nii_f = out_dir / (out_fname + '.nii.gz')
            # dcm2niix may add a suffix to the filename (e.g., '_e2').
            nii_fs = list(out_dir.glob(out_fname + '*.nii.gz'))
            if len(nii_fs) == 1:
                nii_f = nii_fs[0]

            if nii_f.is_file() and (pd.isnull(serDesc) or
                                    'localizer' not in serDesc):
                # Check the integrity of the existing nii_f: the number of
                # volumes (4D) or slices (3D, one slice per file) must match
                # the number of files. Remove it to convert again if not.
                nii = nib.load(nii_f)
                if len(nii.shape) < 4:
                    if len(ser_df) != 1 and len(ser_df) != nii.shape[-1]:
                        nii_f.unlink()
                elif nii.shape[-1] != len(ser_df):
                    nii_f.unlink()

            # -- Convert DICOM to NIfTI ---------------------------------------
            if not nii_f.is_file() or overwrite:
                self._logger.info(
                    f"Processing series {out_dir.name}:{ser}:{serDesc}")
                tmpdir = out_dir / f'tmp_dcm2niix_ser{int(ser)}'
                if tmpdir.is_dir():
                    shutil.rmtree(tmpdir)
                tmpdir.mkdir()
                src_fs = []
                for ff in ser_df.index:
                    orig_f = Path(ff)
                    if orig_f.is_file():
                        dst = tmpdir / orig_f.name
                        dst.symlink_to(orig_f.absolute())
                        src_fs.append(dst)

                if len(src_fs):
                    cmd = f"dcm2niix -m n -f {out_fname} -o {out_dir}"
                    if overwrite:
                        cmd += " -w 1"
                    else:
                        cmd += " -w 2"
                    cmd += f" -z y {tmpdir}"
                    try:
                        proc = subprocess.Popen(
                            shlex.split(cmd), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
                        stdout, stderr = proc.communicate()
                        logstr = stdout.decode().rstrip()
                        self._logger.info(logstr)
                        if proc.returncode != 0:
                            errstr = stderr.decode().rstrip()
                            self._logger.error(errstr)

                    except Exception as e:
                        errstr = str(e) + "\n" + traceback.format_exc()
                        self._logger.error(errstr)

                if tmpdir.is_dir():
                    shutil.rmtree(tmpdir)

            if rt_mrib_com is not None:
                # Open NIfTI file on MRI Browser
                if rt_mrib_com.rpc_ping():
                    rt_mrib_com.call_rt_proc(
                        ('BROWSER_OPEN_FILE', nii_f),
                        pkl=True
                    )

            created_nii[ser] = nii_f
            if not make_brik:
                continue

            # -- Convert NIfTI into BRIK -------------------------------------
            # Include the files with a suffix added by dcm2niix
            nii_fs = list(out_dir.glob(out_fname + '*.nii.gz'))
            for nii_f in nii_fs:
                out_prefix = nii_f.stem.replace('.nii', '')
                brik_f = out_dir / (out_prefix + '+orig.HEAD')
                if not brik_f.is_file() or overwrite:
                    cmd = "AFNI_COMPRESSOR=GZIP "
                    cmd += f"3dcopy -overwrite {nii_f} {brik_f}"
                    try:
                        proc = subprocess.Popen(
                            cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, shell=True)
                        stdout, stderr = proc.communicate()
                        logstr = stdout.decode().rstrip()
                        errstr = stderr.decode().rstrip()
                        if proc.returncode == 0:
                            self._logger.info(logstr + errstr)
                        else:
                            self._logger.error(logstr + errstr)

                    except Exception as e:
                        errstr = str(e) + "\n" + traceback.format_exc()
                        self._logger.error(errstr)
