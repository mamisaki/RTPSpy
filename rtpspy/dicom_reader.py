#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% import ===================================================================
from pathlib import Path
import time
from datetime import datetime
import logging
import sys
import traceback
import re

import numpy as np
import pydicom
import nibabel as nib


# %% DicomReader ==============================================================
class DicomReader():
    """Read scan information from DICOM headers and convert DICOM images to
    NIfTI. Siemens (XA and mosaic) and GE DICOM are supported."""

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self):
        self.logger = logging.getLogger('DicomReader')

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def _software_versions(dcm):
        """Return SoftwareVersions as a string (it can be a MultiValue)."""
        soft_ver = dcm.get('SoftwareVersions', '')
        if isinstance(soft_ver, pydicom.multival.MultiValue):
            soft_ver = '\\'.join(soft_ver)
        return str(soft_ver)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _read_scan_info(self, dcm):
        """Read scan information with the reader for the manufacturer and
        the software version. Return None if they are not supported."""
        manufacturer = dcm.Manufacturer.lower()
        if 'siemens' in manufacturer:
            soft_ver = self._software_versions(dcm)
            if "XA" in soft_ver:
                return self.read_siemens_XA_dicom_info(dcm)
            elif re.search(r"E\d+", soft_ver):
                return self.read_siemens_mosaic_dicom_info(dcm)
        elif 'ge' in manufacturer:
            return self.read_ge_dicom_info(dcm)

        return None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def safe_read_dicom_file(self, dicom_file, root_dir=None, timeout=1):
        """Read a DICOM file and its scan information, retrying until the
        file is completely written or the timeout (seconds) passes.

        Returns
        -------
        dcm : pydicom.dataset.FileDataset or None
        scan_info : dict or None
            None if the file could not be read or is not supported.
        """
        dcm = None
        scan_info = None

        st = time.time()
        while time.time() - st < timeout:
            try:
                dcm = pydicom.dcmread(dicom_file)
                # Fails if the pixel data is incomplete
                assert hasattr(dcm, 'pixel_array')

                # Unsupported manufacturer: stop without retrying
                manufacturer = dcm.Manufacturer.lower()
                if 'siemens' not in manufacturer and \
                        'ge' not in manufacturer:
                    break

                scan_info = self._read_scan_info(dcm)
                assert scan_info is not None
                break
            except Exception:
                time.sleep(0.05)

        if scan_info is not None:
            if root_dir is not None:
                dcm_dir = Path(dicom_file).relative_to(root_dir).parent
            else:
                dcm_dir = Path(dicom_file).parent
            scan_info['DICOM directory'] = str(dcm_dir)

        return dcm, scan_info

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def read_dicom_info(self, dicom, root_dir=None, timeout=1):
        """Read scan information.

        Parameters
        ----------
        dicom : str, Path, or pydicom.dataset.Dataset
            DICOM file or dataset.
        root_dir : Path, optional
            If given, 'DICOM directory' in the returned info is relative to
            root_dir. Used only for a file.
        timeout : float, optional
            Timeout (seconds) for reading a file.

        Returns
        -------
        dict or None
        """
        if isinstance(dicom, pydicom.dataset.Dataset):
            return self._read_scan_info(dicom)

        dcm, scan_info = self.safe_read_dicom_file(dicom, root_dir, timeout)
        if dcm is None:
            self.logger.error(f"Failed to read {dicom} as DICOM")
            return None

        return scan_info

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def dcm2nii(self, dicom_files=None, dcms=None, timeout=2):
        """Convert DICOM images of one volume to a NIfTI image.

        Parameters
        ----------
        dicom_files : list of str or Path, optional
            DICOM files to read.
        dcms : list of pydicom.dataset.Dataset, optional
            DICOM datasets. The datasets read from dicom_files are added
            to these.
        timeout : float, optional
            Timeout (seconds) for reading each file.

        Returns
        -------
        nibabel.Nifti1Image or None
            For an enhanced multi-frame DICOM, the last dataset is
            converted. Otherwise, all datasets are used as the slices of
            one volume.

        References
        ----------
        https://stackoverflow.com/questions/21759013/dicom-affine-matrix-transformation-from-image-space-to-patient-space-in-matlab
        https://nipy.org/nibabel/dicom/dicom_orientation.html#dicom-voxel-to-patient-coordinate-system-mapping
        """

        # --- Load DICOM files ---
        dcms = [] if dcms is None else list(dcms)
        for dcm_f in (dicom_files if dicom_files is not None else []):
            # Wait until the file is readable.
            dcm, scan_info = self.safe_read_dicom_file(dcm_f, timeout=timeout)
            if dcm is None or scan_info is None:
                self.logger.error(f"Failed to read DICOM file {dcm_f}")
                return None
            dcms.append(dcm)

        if len(dcms) == 0:
            self.logger.error("No DICOM data is given.")
            return None

        dcm = dcms[-1]

        # --- Read position and pixel_array data ---
        if 'PerFrameFunctionalGroupsSequence' in dcm:
            rets = self.load_dicom_data_volume(dcm)
        else:
            # Classic DICOM: one slice per file
            rets = self.load_dicom_data_multislice(dcms)

        if rets is None:
            return None

        img_pix_space, img_ornt_pat, img_pos, pixel_array, scan_info = rets

        # --- Affine matrix from voxel index to RAS+ (mm) ---
        F = np.reshape(img_ornt_pat, (2, 3)).T
        # PixelSpacing = [row spacing (between rows), column spacing]
        dr, dc = [float(v) for v in img_pix_space]
        T1 = img_pos[0]
        TN = img_pos[-1]
        if len(img_pos) > 1:
            k = (TN - T1)/(len(img_pos)-1)
        else:
            # Single slice: use the slice normal and thickness
            k = np.cross(F[:, 0], F[:, 1]) * float(
                scan_info.get('Slice Thickness', 1.0))
        A = np.concatenate([F * [[dc, dr]], k[:, None], T1[:, None]],
                           axis=1)
        A = np.concatenate([A, np.array([[0, 0, 0, 1]])], axis=0)
        # DICOM patient coordinates are LPS+; flip x and y to RAS+.
        A[:2, :] *= -1

        # Flip the row axis to LPI: move the origin to the last row.
        point = np.array([[0, pixel_array.shape[1]-1, 0, 1]]).T
        k = np.dot(A, point)
        A[:, 1] *= -1
        A[:, 3] = k.ravel()

        # Transpose and flip to LPI
        img_array = np.transpose(pixel_array, (2, 1, 0))
        img_array = np.flip(img_array, axis=1)

        nii_image = nib.Nifti1Image(img_array, A)

        # Set TR (seconds) in pixdim
        TR = scan_info.get('TR')
        TE = scan_info.get('TE')
        if TR is not None:
            nii_image.header.structarr['pixdim'][4] = TR / 1000.0

        nii_image.header.set_slope_inter(1, 0)
        # Set units for xyz (leave t as unknown)
        nii_image.header.set_xyzt_units(2)

        # Get slice orientation from the dominant axis of the slice normal
        # (Axial: z (2), Sagittal: x (0), Coronal: y (1); also for oblique)
        slice = int(np.argmax(np.abs(np.cross(F[:, 0], F[:, 1]))))
        nii_image.header.set_dim_info(slice=slice)

        # Set the image type, TE, creation time, and multiband factor in
        # the descrip field
        imgType = scan_info['ImageType']
        descrip = f"{imgType};TE={TE};"
        if 'InstanceCreationTime' in dcm:
            descrip += f"Time={float(dcm.InstanceCreationTime):.3f}"
        if 'Slice Timing' in scan_info:
            # Multiband factor: number of slices per acquisition time
            NrSlices = dcm.NumberOfFrames
            NrSliceSampling = len(np.unique(scan_info['Slice Timing']))
            mb = int(NrSlices//NrSliceSampling)
            descrip += f";mb={mb}"
        nii_image.header['descrip'] = descrip.encode()

        return nii_image

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def read_ge_dicom_info(self, dcm):
        # Get info from headers
        scan_info = {}
        try:
            scan_info['Series Information'] = f"scan_{dcm.SeriesNumber}"
            date = datetime.strptime(dcm.StudyDate, '%Y%m%d')
            scan_info['Scan date'] = date.strftime('%a %b %d %Y')
            if '.' in dcm.AcquisitionTime:
                tfmt = '%H%M%S.%f'
            else:
                tfmt = '%H%M%S'
            acqtime = datetime.strptime(dcm.AcquisitionTime, tfmt)
            scan_info['Scan started'] = acqtime.strftime('%H:%M:%S')

            Service_id = dcm[(0x0009, 0x1030)].value
            scan_info['Scanner'] = f'GE-{Service_id[-3:]}'
            if (0x0043, 0x1096) in dcm:
                scan_info['Mode'] = dcm[(0x0043, 0x1096)].value

            if hasattr(dcm, 'StudyID'):
                scan_info['Exam'] = dcm.StudyID
            else:
                scan_info['Exam'] = dcm.PatientID[1:]

            scan_info['SeriesNumber'] = int(dcm.SeriesNumber)
            scan_info['Protocol'] = dcm.ProtocolName
            scan_info['Patient Initials'] = str(dcm.PatientName)[:2]
            scan_info['ID'] = dcm.PatientID
            scan_info['Exam description'] = dcm.StudyDescription
            scan_info['Series De'] = dcm.SeriesDescription
            if hasattr(dcm, 'OperatorsName'):
                scan_info['Operator'] = dcm.OperatorsName
            scan_info['Receive Coil Name'] = dcm.ReceiveCoilName
            scan_info['Pulse Sequence Name'] = dcm[(0x0019, 0x109c)].value
            plane_types = {2: 'Axial',
                           4: 'Sagittal',
                           8: 'Coronal',
                           16: 'Oblique',
                           32: '3-Plane'
                           }
            pltype = dcm[(0x0027, 0x1035)].value
            if pltype in plane_types:
                scan_info['Image Plane'] = plane_types[pltype]
            else:
                scan_info['Image Plane'] = 'Oblique'
            if (0x0027, 0x1060) in dcm:
                scan_info['Image DimX'] = dcm[(0x0027, 0x1060)].value
                scan_info['Image DimY'] = dcm[(0x0027, 0x1061)].value
            scan_info['Xres'] = dcm.Columns
            scan_info['Yres'] = dcm.Rows
            scan_info['ImageType'] = '\\'.join(dcm[(0x0008, 0x0008)].value)

            if (0x0021, 0x104f) in dcm:
                # Read 'Locations in acquisition'
                scan_info['Nr of slices'] = dcm[(0x0021, 0x104f)].value
            elif (0x0025, 0x1007) in dcm:
                scan_info['Nr of slices'] = dcm[(0x0025, 0x1007)].value

            scan_info['Slice Thickness'] = float(dcm.SliceThickness)
            scan_info['Image FOV'] = float(dcm[(0x0019, 0x101e)].value)
            scan_info['TR'] = float(dcm.RepetitionTime)
            scan_info['TE'] = float(dcm.EchoTime)
            scan_info['Number of Echoes'] = int(dcm.EchoNumbers)
            scan_info['FA'] = float(dcm.FlipAngle)
            if hasattr(dcm, 'InversionTime'):
                scan_info['TI'] = dcm.InversionTime

            AdditionalAssetData = dcm[(0x0043, 0x1084)].value
            if AdditionalAssetData[5] == 'NO':
                scan_info['Asset-slice'] = 0.0
                scan_info['Asset-phase'] = 0.0
            else:
                if (0x0043, 0x1083) in dcm:
                    AssetRFactors = dcm[(0x0043, 0x1083)].value
                    if type(AssetRFactors) is pydicom.multival.MultiValue:
                        scan_info['Asset-slice'] = 1.0/AssetRFactors[1]
                        scan_info['Asset-phase'] = 1.0/AssetRFactors[0]
                    else:
                        scan_info['Asset-phase'] = 1.0/AssetRFactors

            if (0x0019, 0x105a) in dcm:
                scan_info['Acquisition Duration'] = dcm[(0x0019, 0x105a)].value

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errstr = ''.join(
                traceback.format_exception(exc_type, exc_obj, exc_tb))
            msg = f"Failed to read DICOM info: {errstr}"
            self.logger.error(msg)
            scan_info = None

        return scan_info

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def read_siemens_XA_dicom_info(self, dcm):
        scan_info = {}
        try:
            scan_info['Series Information'] = f"scan_{dcm.SeriesNumber}"
            if 'AcquisitionDateTime' in dcm:
                acqdtime = dcm.AcquisitionDateTime
            elif 'SeriesDate' in dcm and 'SeriesTime' in dcm:
                acqdtime = dcm.SeriesDate + dcm.SeriesTime
            else:
                acqdtime = None

            if acqdtime is not None:
                if '.' in acqdtime:
                    tfmt = '%Y%m%d%H%M%S.%f'
                else:
                    tfmt = '%Y%m%d%H%M%S'
                acqdtime = datetime.strptime(acqdtime, tfmt)
                scan_info['Scan date'] = acqdtime.strftime('%a %b %d %Y')
                scan_info['Scan started'] = acqdtime.strftime('%H:%M:%S')

            dateTime = datetime.strptime(dcm.StudyDate+dcm.StudyTime,
                                         '%Y%m%d%H%M%S.%f')
            scan_info['StudyDateTime'] = dateTime.isoformat()

            scan_info['ContentTime'] = dcm.ContentTime
            scan_info["InstanceCreationTime"] = dcm.InstanceCreationTime

            StationName = dcm.StationName
            if 'ManufacturerModelName' in dcm:
                ModelName = dcm.ManufacturerModelName.replace(' ', '_')
            else:
                ModelName = ''
            scan_info['Scanner'] = f"{StationName}_{ModelName}"
            if hasattr(dcm, 'StudyID'):
                scan_info['StudyID'] = dcm.StudyID

            scan_info['SeriesNumber'] = int(dcm.SeriesNumber)
            scan_info['Protocol'] = dcm.ProtocolName
            scan_info['PatientName'] = dcm.PatientName
            scan_info['PatientID'] = dcm.PatientID
            scan_info['StudyDescription'] = dcm.StudyDescription
            scan_info['SeriesDescription'] = dcm.SeriesDescription
            if hasattr(dcm, 'OperatorsName'):
                scan_info['Operator'] = dcm.OperatorsName
            if 'SequenceName' in dcm:
                scan_info['Pulse Sequence Name'] = dcm.SequenceName
            elif 'PulseSequenceName' in dcm:
                scan_info['Pulse Sequence Name'] = dcm.PulseSequenceName
            scan_info['Protocol Name'] = dcm.ProtocolName

            # Get Acquisition Number and Instance Number to judge multi-echo
            if 'AcquisitionNumber' in dcm:
                scan_info['AcquisitionNumber'] = int(dcm.AcquisitionNumber)
            if 'InstanceNumber' in dcm:
                scan_info['InstanceNumber'] = int(dcm.InstanceNumber)

            if 'Columns' in dcm and 'Rows' in dcm:
                scan_info['Xres'] = dcm.Columns
                scan_info['Yres'] = dcm.Rows
            scan_info['ImageType'] = '\\'.join(dcm.ImageType)
            if 'NumberOfFrames' in dcm:
                scan_info['Nr of slices'] = int(dcm.NumberOfFrames)
            if 'SpacingBetweenSlices' in dcm:
                scan_info['Slice Thickness'] = float(dcm.SpacingBetweenSlices)
            elif 'SliceThickness' in dcm:
                scan_info['Slice Thickness'] = float(dcm.SliceThickness)

            # Calculate FOV
            if 'PixelSpacing' in dcm:
                PixelSpacing = dcm.PixelSpacing
            elif (0x5200, 0x9230) in dcm:
                PixelSpacing = dcm[(0x5200, 0x9230)
                                   ][0].PixelMeasuresSequence[0].PixelSpacing
            else:
                PixelSpacing = None
            if PixelSpacing is not None:
                # PixelSpacing = [row spacing, column spacing]
                FOV = [scan_info['Xres'] * PixelSpacing[1],
                       scan_info['Yres'] * PixelSpacing[0]]
                scan_info['FOV'] = FOV

            MRTiming = None
            if 'SharedFunctionalGroupsSequence' in dcm:
                if 'MRTimingAndRelatedParametersSequence' in \
                        dcm.SharedFunctionalGroupsSequence[0]:
                    MRTiming = dcm.SharedFunctionalGroupsSequence[
                        0].MRTimingAndRelatedParametersSequence[0]

            if 'RepetitionTime' in dcm:
                scan_info['TR'] = float(dcm.RepetitionTime)
            elif MRTiming is not None:
                scan_info['TR'] = float(MRTiming.RepetitionTime)

            if 'EchoTime' in dcm:
                scan_info['TE'] = float(dcm.EchoTime)
            elif (0x5200, 0x9230) in dcm:
                scan_info['TE'] = float(
                    dcm[(0x5200, 0x9230)
                        ][0].MREchoSequence[0].EffectiveEchoTime)

            if 'EchoTrainLength' in dcm:
                scan_info['EchoTrainLength'] = int(dcm.EchoTrainLength)
            elif MRTiming is not None:
                scan_info['EchoTrainLength'] = int(MRTiming.EchoTrainLength)

            if 'FlipAngle' in dcm:
                scan_info['FA'] = float(dcm.FlipAngle)
            elif MRTiming is not None:
                scan_info['FA'] = float(MRTiming.FlipAngle)

            if hasattr(dcm, 'InversionTime'):
                scan_info['TI'] = float(dcm.InversionTime)
            if 'AcquisitionDuration' in dcm:
                scan_info['Acquisition Duration'] = dcm.AcquisitionDuration

            # Parallel acquisition
            if 'SharedFunctionalGroupsSequence' in dcm:
                if 'MRModifierSequence' in \
                        dcm.SharedFunctionalGroupsSequence[0]:
                    MRModifierSequence = dcm.SharedFunctionalGroupsSequence[
                        0].MRModifierSequence[0]
                    if 'ParallelAcquisition' in MRModifierSequence:
                        scan_info['Parallel Acquisition'] = \
                            MRModifierSequence.ParallelAcquisition
                        if scan_info['Parallel Acquisition'] != 'NO':
                            scan_info['Parallel Acquisition Technique'] = \
                                MRModifierSequence.ParallelAcquisitionTechnique
                            scan_info['Parallel Reduction Factor In-plane'] = (
                                MRModifierSequence
                                .ParallelReductionFactorInPlane
                            )

            # Get slice timing
            if 'NumberOfFrames' in dcm:
                acq_times = []
                for frame in dcm.PerFrameFunctionalGroupsSequence:
                    acq_time = frame.FrameContentSequence[
                        0].FrameAcquisitionDateTime
                    acq_times.append(
                        datetime.strptime(acq_time, '%Y%m%d%H%M%S.%f'))

                slice_timing = [dt.total_seconds() for dt
                                in np.array(acq_times) - np.min(acq_times)]
                if len(np.unique(slice_timing)) > 1:
                    scan_info['Slice Timing'] = slice_timing

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errstr = ''.join(
                traceback.format_exception(exc_type, exc_obj, exc_tb))
            msg = f"Failed to read DICOM info: {errstr}"
            self.logger.error(msg)
            scan_info = None

        return scan_info

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def read_siemens_mosaic_dicom_info(self, dcm):
        scan_info = {}
        try:
            scan_info['Series Information'] = f"scan_{dcm.SeriesNumber}"
            date = datetime.strptime(dcm.StudyDate, '%Y%m%d')
            scan_info['Scan date'] = date.strftime('%a %b %d %Y')
            if '.' in dcm.AcquisitionTime:
                tfmt = '%H%M%S.%f'
            else:
                tfmt = '%H%M%S'
            acqtime = datetime.strptime(dcm.AcquisitionTime, tfmt)
            scan_info['Scan started'] = acqtime.strftime('%H:%M:%S')

            StationName = dcm[(0x0008, 0x1010)].value
            ModelName = dcm[(0x0008, 0x1090)].value
            scan_info['Scanner'] = f"{StationName}_{ModelName}"

            if hasattr(dcm, 'StudyID'):
                scan_info['Exam'] = dcm.StudyID

            scan_info['SeriesNumber'] = int(dcm.SeriesNumber)
            scan_info['ProtocolName'] = dcm.ProtocolName
            scan_info['StudyDescription'] = dcm.StudyDescription
            scan_info['SeriesDescription'] = dcm.SeriesDescription
            if hasattr(dcm, 'OperatorsName'):
                scan_info['Operator'] = dcm.OperatorsName
            scan_info['Transmit Coil Name'] = dcm.TransmitCoilName
            scan_info['SequenceName'] = dcm.SequenceName
            scan_info['Image Plane'] = dcm[(0x0051, 0x100e)].value
            scan_info['Xres'] = dcm.Columns
            scan_info['Yres'] = dcm.Rows
            scan_info['ImageType'] = '\\'.join(dcm.ImageType)

            # NumberOfImagesInMosaic
            scan_info['Nr of slices'] = dcm[(0x0019, 0x100a)].value

            scan_info['Slice Thickness'] = float(dcm.SliceThickness)
            scan_info['FOV'] = dcm[(0x0051, 0x100c)].value
            scan_info['TR'] = float(dcm.RepetitionTime)
            scan_info['TE'] = float(dcm.EchoTime)
            scan_info['Number of Echoes'] = int(dcm.EchoNumbers)
            scan_info['FA'] = float(dcm.FlipAngle)
            if hasattr(dcm, 'InversionTime'):
                scan_info['TI'] = dcm.InversionTime

            if (0x0019, 0x105a) in dcm:
                scan_info['Acquisition Duration'] = dcm[(0x0019, 0x105a)].value

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errstr = ''.join(
                traceback.format_exception(exc_type, exc_obj, exc_tb))
            msg = f"Failed to read DICOM info: {errstr}"
            self.logger.error(msg)
            scan_info = None

        return scan_info

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def load_dicom_data_volume(self, dcm):
        """Read the geometry and pixel data of an enhanced multi-frame DICOM
        volume (with PerFrameFunctionalGroupsSequence), e.g., Siemens XA
        functional data.

        Parameters
        ----------
        dcm : pydicom.dataset.FileDataset
            DICOM dataset read by pydicom.

        Returns
        -------
        tuple or None
            (img_pix_space, img_ornt_pat, img_pos, pixel_array, scan_info)
        """
        try:
            # Check multi-frame field
            assert 'PerFrameFunctionalGroupsSequence' in dcm

            # Pixel spacing (should be the same for all slices)
            img_pix_space = dcm.PerFrameFunctionalGroupsSequence[
                0].PixelMeasuresSequence[0].PixelSpacing

            # ImageOrientationPatient (should be the same for all slices):
            # direction cosines of the first row and the first column with
            # respect to the patient; the row values for the x, y, and z
            # axes, followed by the column values for the x, y, and z axes.
            img_ornt_pat = dcm.PerFrameFunctionalGroupsSequence[
                0].PlaneOrientationSequence[0].ImageOrientationPatient

            # Slice positions: x, y, and z coordinates of the upper left-hand
            # corner (the center of the first voxel) of each slice.
            img_pos = []
            for frame in dcm.PerFrameFunctionalGroupsSequence:
                img_pos.append(
                    frame.PlanePositionSequence[0].ImagePositionPatient)

            img_pos = np.concatenate(
                [np.array(pos)[None, :] for pos in img_pos], axis=0)

            pixel_array = dcm.pixel_array
            if pixel_array.ndim == 2:
                # Single-frame file: pixel_array is (rows, columns)
                pixel_array = pixel_array[None, :, :]
            scan_info = self.read_dicom_info(dcm)
            assert scan_info is not None

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errstr = ''.join(
                traceback.format_exception(exc_type, exc_obj, exc_tb))
            self.logger.error(errstr)
            return None

        return img_pix_space, img_ornt_pat, img_pos, pixel_array, scan_info

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def load_dicom_data_multislice(self, dcms):
        """Read the geometry and pixel data of a volume from classic DICOM
        datasets with one slice each. The slices are sorted by position.

        Parameters
        ----------
        dcms : list of pydicom.dataset.Dataset
            DICOM datasets of the slices.

        Returns
        -------
        tuple or None
            (img_pix_space, img_ornt_pat, img_pos, pixel_array, scan_info)
        """
        try:
            # Check image position fields
            assert 'ImageOrientationPatient' in dcms[0]
            assert 'PixelSpacing' in dcms[0]
            assert 'ImagePositionPatient' in dcms[0]

            # Pixel spacing (should be the same for all slices)
            img_pix_space = dcms[0].PixelSpacing

            # ImageOrientationPatient (should be the same for all slices):
            # see load_dicom_data_volume.
            img_ornt_pat = dcms[0].ImageOrientationPatient

            # Slice positions (the center of the first voxel) and pixel
            # arrays
            img_pos = []
            pixel_array = []
            for frame in dcms:
                img_pos.append(frame.ImagePositionPatient)
                pixel_array.append(frame.pixel_array)

            img_pos = np.concatenate(
                [np.array(pos, dtype=float)[None, :] for pos in img_pos],
                axis=0)
            pixel_array = np.concatenate(
                [np.array(arr)[None, :, :] for arr in pixel_array], axis=0)

            # Sort slices by the position along the slice normal
            F = np.reshape(np.array(img_ornt_pat, dtype=float), (2, 3)).T
            slice_pos = img_pos @ np.cross(F[:, 0], F[:, 1])
            sidx = np.argsort(slice_pos, kind='stable')
            img_pos = img_pos[sidx]
            pixel_array = pixel_array[sidx]

            scan_info = self.read_dicom_info(dcms[0])
            assert scan_info is not None

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errstr = ''.join(
                traceback.format_exception(exc_type, exc_obj, exc_tb))
            self.logger.error(errstr)
            return None

        return img_pix_space, img_ornt_pat, img_pos, pixel_array, scan_info
