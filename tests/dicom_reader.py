#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DicomReader for the test scripts (an older copy of rtpspy/dicom_reader.py)
"""

# %% import ===================================================================
from pathlib import Path
import time
from datetime import datetime
import logging
import traceback

import numpy as np
import pydicom
import nibabel as nib


# %% DicomReader ==============================================================
class DicomReader():

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self):
        self.logger = logging.getLogger('DicomReader')

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _read_scan_info(self, dcm):
        """Read scan information with the reader for the manufacturer.
        Return None if it is not supported."""
        manufacturer = dcm.Manufacturer.lower()
        if 'siemens' in manufacturer:
            return self.read_siemens_dicom_info(dcm)
        elif 'ge' in manufacturer:
            return self.read_ge_dicom_info(dcm)

        return None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def safe_read_dicom_file(self, dicom_file, root_dir=None, timeout=1):
        """Read a DICOM file, retrying until timeout (s) while the file is
        being written. Returns the dataset and the scan information."""
        dcm = None
        scan_info = None

        st = time.time()
        while time.time() - st < timeout:
            # Wait until the file is readable.
            try:
                dcm = pydicom.dcmread(dicom_file)
                assert hasattr(dcm, 'pixel_array')

                scan_info = self._read_scan_info(dcm)
                if scan_info is None:
                    break  # Not supported

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
    def read_dicom_info(self, dicom, is_file=True, root_dir=None,
                        timeout=1):
        """Return the scan information of a DICOM file (is_file=True) or a
        dataset. None if it cannot be read."""
        if is_file:
            dcm, scan_info = self.safe_read_dicom_file(
                dicom, root_dir, timeout)
            if dcm is None:
                self.logger.error(f"Failed to read {dicom} as DICOM")
                return None
        else:
            scan_info = self._read_scan_info(dicom)

        # For a file, safe_read_dicom_file() has set 'DICOM directory'.
        return scan_info

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def dcm2nii(self, dicom_files=[], dcms=None, time_out=2):
        """Convert DICOM files (or datasets in dcms) of a volume to a NIfTI
        image.

        See
        https://stackoverflow.com/questions/21759013/dicom-affine-matrix-transformation-from-image-space-to-patient-space-in-matlab
        https://nipy.org/nibabel/dicom/dicom_orientation.html#dicom-voxel-to-patient-coordinate-system-mapping
        """

        # --- Load the DICOM files ---
        dcms = [] if dcms is None else list(dcms)
        for dcm_f in dicom_files:
            dcm = None
            st = time.time()
            while time.time() - st < time_out:
                # Wait until the file is readable.
                try:
                    dcm = pydicom.dcmread(dcm_f)
                    assert hasattr(dcm, 'pixel_array')
                    assert self._read_scan_info(dcm) is not None
                    dcms.append(dcm)
                    break
                except Exception:
                    dcm = None
                    time.sleep(0.01)
            if dcm is None:
                self.logger.error(f"Failed to read DICOM file {dcm_f}")
                return None

        dcm = dcms[-1]

        # --- Read position and pixel_array data ---
        if 'PerFrameFunctionalGroupsSequence' in dcms[0]:
            rets = self.load_dicom_data_volume(dcms[0])
        else:
            rets = self.load_dicom_data_multislice(dcms)

        if rets is None:
            return None

        img_pix_space, img_ornt_pat, img_pos, pixel_array, scan_info = rets

        # --- Affine matrix of image to patient space (LPI) (mm) translation --
        F = np.reshape(img_ornt_pat, (2, 3)).T
        dc, dr = [float(v) for v in img_pix_space]
        T1 = img_pos[0]
        TN = img_pos[-1]
        k = (TN - T1)/(len(img_pos)-1)
        A = np.concatenate([F * [[dc, dr]], k[:, None], T1[:, None]],
                           axis=1)
        A = np.concatenate([A, np.array([[0, 0, 0, 1]])], axis=0)
        A[:2, :] *= -1

        # -> LPI
        point = np.array([[0, pixel_array.shape[1]-1, 0, 1]]).T
        k = np.dot(A, point)
        A[:, 1] *= -1
        A[:, 3] = k.ravel()

        # Transpose and flip to LPI
        img_array = np.transpose(pixel_array, (2, 1, 0))
        img_array = np.flip(img_array, axis=1)

        nii_image = nib.Nifti1Image(img_array, A)

        # Set TR in pixdim
        TR = scan_info['TR']
        TE = scan_info['TE']
        nii_image.header.structarr['pixdim'][4] = TR / 1000.0

        nii_image.header.set_slope_inter(1, 0)
        # Set units for xyz (leave t as unknown)
        nii_image.header.set_xyzt_units(2)

        # Get slice orientation
        ori_code = np.round(np.abs(F)).sum(axis=1)
        if ori_code[0] > 0 and ori_code[1] > 0 and ori_code[2] == 0:
            # Axial
            slice = 2  # z is slice
        elif ori_code[0] == 0 and ori_code[1] > 0 and ori_code[2] > 0:
            # Sagittal
            slice = 0  # x is slice
        elif ori_code[0] > 0 and ori_code[1] == 0 and ori_code[2] > 0:
            # Coronal
            slice = 1  # y is slice
        nii_image.header.set_dim_info(slice=slice)

        imgType = scan_info['ImageType']
        descrip = f"{imgType};TE={TE};"
        descrip += f"Time={float(dcm.InstanceCreationTime):.3f}"
        if 'Slice Timing' in scan_info:
            NrSlices = dcm.NumberOfFrames
            NrSliceSampling = len(np.unique(scan_info['Slice Timing']))
            mb = int(NrSlices//NrSliceSampling)
            descrip += f";mb={mb}"
        nii_image.header['descrip'] = descrip.encode()

        return nii_image

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def read_ge_dicom_info(self, dcm):
        """Read the scan information from GE DICOM headers."""
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
            errstr = traceback.format_exc()
            msg = f"Failed to read DICOM info: {errstr}"
            self.logger.error(msg)
            scan_info = None

        return scan_info

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def read_siemens_dicom_info(self, dcm):
        """Read the scan information from Siemens DICOM headers."""
        scan_info = {}
        try:
            scan_info['SeriesNumber'] = int(dcm.SeriesNumber)
            date = datetime.strptime(dcm.StudyDate, '%Y%m%d')
            scan_info['Scan date'] = date.strftime('%a %b %d %Y')
            scan_info['ContentTime'] = dcm.ContentTime
            if hasattr(dcm, 'AcquisitionTime'):
                scan_info['AcquisitionTime'] = dcm.AcquisitionTime
            elif hasattr(dcm, 'AcquisitionDateTime'):
                scan_info['AcquisitionTime'] = dcm.AcquisitionDateTime[8:]
            scan_info['SeriesDescription'] = dcm.SeriesDescription

        except Exception:
            errstr = traceback.format_exc()
            msg = f"Failed to read DICOM info: {errstr}"
            self.logger.error(msg)
            scan_info = None

        return scan_info

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def load_dicom_data_volume(self, dcm):
        """Read a multi-frame DICOM volume (with
        PerFrameFunctionalGroupsSequence), e.g., Siemens XA30 functional
        data.

        Args:
            dcm (pydicom.dataset.FileDataset): DICOM dataset read by pydicom
        """
        try:
            # Check multi-frame field
            assert 'PerFrameFunctionalGroupsSequence' in dcm

            # Pixel spacing (should be the same for all slices)
            img_pix_space = dcm.PerFrameFunctionalGroupsSequence[
                0].PixelMeasuresSequence[0].PixelSpacing

            # Get ImageOrientationPatient (should be the same for all slices)
            # img_ornt_pat: direction cosines of the first row and the first
            # column with respect to the patient.
            # These attributes are provided as a pair.
            # Row value for the x, y, and z axes respectively followed by the
            # Column value for the x, y, and z axes respectively.
            img_ornt_pat = dcm.PerFrameFunctionalGroupsSequence[
                0].PlaneOrientationSequence[0].ImageOrientationPatient

            # Get slice positions
            # img_pos: x, y, and z coordinates of the upper left hand corner of
            # the image; it is the center of the first voxel transmitted.
            img_pos = []
            for frame in dcm.PerFrameFunctionalGroupsSequence:
                img_pos.append(
                    frame.PlanePositionSequence[0].ImagePositionPatient)

            img_pos = np.concatenate(
                [np.array(pos)[None, :] for pos in img_pos], axis=0)

            pixel_array = dcm.pixel_array
            scan_info = self._read_scan_info(dcm)

        except Exception:
            errstr = traceback.format_exc()
            self.logger.error(errstr)
            return None

        return img_pix_space, img_ornt_pat, img_pos, pixel_array, scan_info

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def load_dicom_data_multislice(self, dcms):
        """Read a volume from single-slice DICOM datasets."""
        try:
            # Check the image position fields
            assert 'ImageOrientationPatient' in dcms[0]
            assert 'PixelSpacing' in dcms[0]
            assert 'ImagePositionPatient' in dcms[0]

            # Pixel spacing (should be the same for all slices)
            img_pix_space = dcms[0].PixelSpacing

            # Get ImageOrientationPatient (should be the same for all slices)
            # img_ornt_pat: direction cosines of the first row and the first
            # column with respect to the patient.
            # These attributes are provided as a pair.
            # Row value for the x, y, and z axes respectively followed by the
            # Column value for the x, y, and z axes respectively.
            img_ornt_pat = dcms[0].ImageOrientationPatient

            # Get slice positions and pixel array
            # img_pos: x, y, and z coordinates of the upper left hand corner of
            # the image; it is the center of the first voxel transmitted.
            img_pos = []
            pixel_array = []
            for frame in dcms:
                img_pos.append(frame.ImagePositionPatient)
                pixel_array.append(frame.pixel_array)

            img_pos = np.concatenate(
                [np.array(pos)[None, :] for pos in img_pos], axis=0)
            pixel_array = np.concatenate(
                [np.array(arr)[None, :, :] for arr in pixel_array], axis=0)
            scan_info = self._read_scan_info(dcms[0])

        except Exception:
            errstr = traceback.format_exc()
            self.logger.error(errstr)
            return None

        return img_pix_space, img_ornt_pat, img_pos, pixel_array, scan_info


# %% __main__ =================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Print the scan information of the DICOM files in a'
        ' directory, sorted by content time.')
    parser.add_argument('dicom_dir', help='DICOM file directory')
    args = parser.parse_args()

    dcmReader = DicomReader()
    scan_infos = []
    for ff in sorted(Path(args.dicom_dir).glob('*')):
        if not ff.is_file():
            continue
        scan_info = dcmReader.read_dicom_info(ff, timeout=0.1)
        if scan_info is None:
            continue  # Not a DICOM file
        scan_infos.append((scan_info.get('ContentTime', ''), ff.name,
                           scan_info))

    for content_time, fname, scan_info in sorted(scan_infos):
        print(f"{fname}: Series {scan_info['SeriesNumber']},"
              f" ContentTime {content_time}")
