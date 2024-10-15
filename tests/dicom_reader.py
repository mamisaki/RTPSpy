#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% import ===================================================================
from pathlib import Path
import time
from datetime import datetime
import logging
import sys
import traceback

import numpy as np
import pydicom
import nibabel as nib


# %% DicomReader ==============================================================
class DicomReader():

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, vendor='SiemensXA30'):
        self.logger = logging.getLogger('DicomReader')
        self.vendor = vendor

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def safe_read_dicom_file(self, dicom_file, root_dir=None, timeout=1):
        dcm = None
        scan_info = None

        st = time.time()
        while time.time() - st < timeout:
            # Wait until the file is readable.
            try:
                dcm = pydicom.dcmread(dicom_file)
                assert hasattr(dcm, 'pixel_array')

                manufacturer = dcm.Manufacturer.lower()
                if 'siemens' in manufacturer:
                    scan_info = self.read_siemens_dicom_info(dcm)
                elif 'GE' in manufacturer:
                    scan_info = self.read_ge_dicom_info(dcm)
                else:
                    break

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
    def read_dicom_info(self, dicom, is_file=True, root_dir=None,
                        timeout=1):
        if is_file:
            dcm, scan_info = self.safe_read_dicom_file(
                dicom, root_dir, timeout)
            if dcm is None:
                self.logger.error(f"Failed to read {dicom} as DICOM")
                return None
        else:
            dcm = dicom
            if self.vendor == 'GE':
                scan_info = self.read_ge_dicom_info(dcm)
            elif self.vendor == 'SiemensMosaic':
                scan_info = self.read_mosaic_dicom_info(dcm)
            elif self.vendor == 'SiemensXA30':
                scan_info = self.read_siemensXA30_dicom_info(dcm)

        if scan_info is not None:
            if is_file:
                if root_dir is not None:
                    dcm_dir = Path(dicom).relative_to(root_dir).parent
                else:
                    dcm_dir = Path(dicom).parent
                scan_info['DICOM directory'] = str(dcm_dir)

        return scan_info

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def dcm2nii(self, dicom_files=[], dcms=[], time_out=2):
        """_summary_

        Args:
            img_ornt_pat (_type_): _description_
            img_pix_space (_type_): _description_
            img_pos (_type_): _description_
            pixel_array (_type_): _description_

        Refered to
        https://stackoverflow.com/questions/21759013/dicom-affine-matrix-transformation-from-image-space-to-patient-space-in-matlab
        https://nipy.org/nibabel/dicom/dicom_orientation.html#dicom-voxel-to-patient-coordinate-system-mapping
        """

        # --- Load dicom files ---
        for dcm_f in dicom_files:
            st = time.time()
            while time.time() - st < time_out:
                # Wait until the file is readable.
                try:
                    dcm = pydicom.dcmread(dcm_f)
                    assert hasattr(dcm, 'pixel_array')
                    if self.vendor == 'GE':
                        scan_info = self.read_ge_dicom_info(dcm)
                    elif self.vendor == 'SiemensMosaic':
                        scan_info = self.read_mosaic_dicom_info(dcm)
                    elif self.vendor == 'SiemensXA30':
                        scan_info = self.read_siemensXA30_dicom_info(dcm)
                    assert scan_info is not None
                    dcms.append(dcm)
                    break
                except Exception:
                    time.sleep(0.01)
            if dcm is None:
                self.logger.error(f"Failed to read DICOM file {dcm_f}")
                return None

        dcm = dcms[-1]

        # --- Read position and pixel_array data ---
        if 'PerFrameFunctionalGroupsSequence' in dcms[0]:
            rets = self.load_dicom_data_volume(dcms[0])
        else:
            rets = self.load_dicom_data_volume(dcms)

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
        # image_pos = _multiframe_get_image_position(dicoms[0], 0)
        point = np.array([[0, pixel_array.shape[1]-1, 0, 1]]).T
        k = np.dot(A, point)
        A[:, 1] *= -1
        A[:, 3] = k.ravel()

        # Transpose and flip to LPI
        img_array = np.transpose(pixel_array, (2, 1, 0))
        img_array = np.flip(img_array, axis=1)

        nii_image = nib.Nifti1Image(img_array, A)

        # Set TR and TE in pixdim and db_name field
        TR = scan_info['TR']
        TE = scan_info['TE']
        nii_image.header.structarr['pixdim'][4] = TR / 1000.0

        nii_image.header.set_slope_inter(1, 0)
        # set units for xyz (leave t as unknown)
        nii_image.header.set_xyzt_units(2)

        # Get slice orientation
        ori_code = np.round(np.abs(F)).sum(axis=1)
        if ori_code[0] > 0 and ori_code[1] > 0 and ori_code[2] == 0:
            # Axial
            slice = 2  # z is slice
        elif ori_code[0] == 0 and ori_code[1] > 0 and ori_code[2] > 0:
            # Saggital
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
            scan_info['Scanner'] = f'LIBR-{Service_id[-3:]}'
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
    def read_siemens_dicom_info(self, dcm):
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
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errstr = ''.join(
                traceback.format_exception(exc_type, exc_obj, exc_tb))
            msg = f"Failed to read DICOM info: {errstr}"
            self.logger.error(msg)
            scan_info = None

        return scan_info

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def load_dicom_data_volume(self, dcm):
        """Read DICOM data that contains multiframe volume, e.g. Siemense XA30
        function data that has multi-frame field (i.e.,
        PerFrameFunctionalGroupsSequence)

        Args:
            dcm (pydicom.dataset.FileDataset): dicom dataset read by pydicom
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
            # These Attributes shall be provide as a pair.
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
            if self.vendor == 'GE':
                scan_info = self.read_ge_dicom_info(dcm)
            elif self.vendor == 'SiemensMosaic':
                scan_info = self.read_mosaic_dicom_info(dcm)
            elif self.vendor == 'SiemensXA30':
                scan_info = self.read_siemensXA30_dicom_info(dcm)

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errstr = ''.join(
                traceback.format_exception(exc_type, exc_obj, exc_tb))
            self.logger.error(errstr)
            return None

        return img_pix_space, img_ornt_pat, img_pos, pixel_array, scan_info

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def load_dicom_data_multislice(self, dcms):
        try:
            # Check imae position fields
            assert 'ImageOrientationPatient' in dcms[0]
            assert 'PixelSpacing' in dcms[0]
            assert 'ImagePositionPatient' in dcms[0]

            # Pixel spacing (should be the same for all slices)
            img_pix_space = dcms[0].PixelSpacing

            # Get ImageOrientationPatient (should be the same for all slices)
            # img_ornt_pat: direction cosines of the first row and the first
            # column with respect to the patient.
            # These Attributes shall be provide as a pair.
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
            scan_info = self.load_dicom_data_volume(dcms[0])

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errstr = ''.join(
                traceback.format_exception(exc_type, exc_obj, exc_tb))
            self.logger.error(errstr)
            return None

        return img_pix_space, img_ornt_pat, img_pos, pixel_array, scan_info


# %% __main__ =================================================================
if __name__ == '__main__':
    '''Sample files
    # --- GE MR750 ---
    # localizer
    dicom_file = Path('test_data/p234/e235/s2445/i14853692.MRDC.1')
    dicom_file = Path('test_data/p235/e236/s2452/i14863892.MRDC.1')
    # corrupted

    # calibration scans
    dicom_file = Path('test_data/p235/e236/s2453/i14863900.MRDC.1')
    dicom_file = Path('test_data/p235/e236/s2454/i14863929.MRDC.1')

    # Functions (epi)
    dicom_file = Path('test_data/p235/e236/s2455/i14864051.MRDC.1')
    dicom_file = Path('test_data/p235/e236/s2456/i14871069.MRDC.1')

    # mprage
    dicom_file = Path('test_data/p235/e236/s2459/i14906665.MRDC.1')

    # DWPROP
    dicom_file = Path('test_data/p235/e236/s2463/i14907121.MRDC.1')

    # epi2
    dicom_file = Path('test_data/p235/e236/s2468/i14907274.MRDC.1')

    # ? epi2
    dicom_file = Path('test_data/p235/e236/s2469/i14907319.MRDC.1')
    dicom_file = Path('test_data/p235/e236/s2470/i14907353.MRDC.1')

    # ABCD
    # DTI : muxepi2
    dicom_file = Path('test_data/p238/e239/s2489/i15000472.MRDC.1')

    # muxepi2
    dicom_file = Path('test_data/p238/e239/s2486/i14953445.MRDC.1')

    # fieldmap
    dicom_file = Path('test_data/p238/e239/s2491/i15009108.MRDC.1')

    # ABCD_Scan_Session Fieldmap
    dicom_file = Path('test_data/p246/e247/s2574/i15644607.MRDC.1')
    dicom_file1 = Path('test_data/p246/e247/s2574/i15644842.MRDC.240')

    dicom_file = Path('test_data/p246/e247/s2575/i15645743.MRDC.1')
    dicom_file1 = Path('test_data/p246/e247/s2575/i15671992.MRDC.27180')

    dicom_file = Path('test_data/p239/e240/s2518/i15323002.MRDC.1')

    # QA
    dicom_file = Path('test_data/p296/e297/s3091/i18948214.MRDC.1')

    # --- Siemense XA30 ---
    # UCSD (UC Riverside)
    # fcMRI
    dicom_file = Path(
        'test_data/XA30/URBANE_Axial_fcMRI/'
        'URBANE_Pilot_01.MR.URBANE_Final'
        '.3.1.2022.09.20.14.01.15.768.22081276.dcm')

    # DTI
    dicom_file = Path(
        'test_data/XA30/URBANE_Axial_DTI_AP/'
        'URBANE_Pilot_01.MR.URBANE_Final'
        '.7.1.2022.09.20.14.02.37.881.46058563.dcm')

    # T1W
    dicom_file = Path(
        'test_data/XA30/URBANE_T1W_3DMPRAGE/'
        'URBANE_Pilot_01.MR.URBANE_Final'
        '.2001.1.2022.09.20.13.58.17.841.11445763.dcm')

    # PCASL
    dicom_file = Path(
        'test_data/XA30/URBANE_PCASL/'
        'URBANE_Pilot_01.MR.URBANE_Final'
        '.13.1.2022.09.20.14.04.12.453.16837788.dcm')

    # Mclean
    # fMRI
    dicom_file = Path(
        'test_data/XA30-anon-hierarchical/Patientname_19800101/E26580742/'
        'S011/IM-0011-0001.dcm')
    dicom_files = [dicom_file]
    nii_file = Path(
        'test_data/XA30-anon-hierarchical/Patientname_19800101/E26580742/'
        'S011/S011_lfms-mb_720ms_20220427121855_23.nii')
    img = nib.load(nii_file)
    affine = img.affine
    img_data = img.get_fdata()[:, :, :, 0]

    img = nib.load('23_lfms-mb_720ms.nii.gz')
    affine = img.affine
    img_data = img.get_fdata()

    # MPRAGE
    dicom_dir = Path(
        'test_data/XA30-anon-hierarchical/Patientname_19800101/E26580742/S013')
    dicom_files = np.array(list(dicom_dir.glob('IM-0013-*.dcm')))
    fidx = [int(fname.stem.split('-')[-1]) for fname in dicom_files]
    dicom_files = dicom_files[np.argsort(fidx).ravel()]
    dicom_file = dicom_files[0]

    '''
    # MR1 Prisma
    from tqdm import tqdm
    dcmReader = DicomReader()
    dicom_dir = Path(
        'test_data/MR1_PRISMA_ser' +
        '/ba3beee3a4aaa60032bfa6c4314ca5fde0112dff_MP0_2023_20231018/Ser_08')
    dicom_files = np.array(list(dicom_dir.glob('*')))
    content_times = {}
    derived = {}
    for ii, ff in tqdm(enumerate(dicom_files), total=len(dicom_files),
                       desc=f'read files in {dicom_dir.name}'):
        scan_info = dcmReader.read_dicom_info(ff, timeout=0.1)
        if scan_info is None:
            # Not a DICOM file
            continue
        at = datetime.strptime(scan_info['ContentTime'], '%H%M%S.%f')
        content_times[ii] = at
        derived[ii] = 'DERIVED' in scan_info['ImageType']
    # sort with content_time
    t_order = np.argsort(list(content_times.values())).ravel()
    sidx = np.array(list(content_times.keys()))[list(t_order)]
    dicom_files = np.array(dicom_files)[sidx]
    derived = [derived[idx] for idx in sidx]
    acq_time = np.array([content_times[idx] for idx in sidx])

    dcmReader = DicomReader()
    dcm_f = Path('/RTMRI/ba3beee3a4aaa60032bfa6c4314ca5fde0112dff_MP0_2023_20231018/1.3.12.2.1107.5.2.43.166321.30000023101809592153400000020_9cadb06a-8a32-449c-b34f-ef42e750dfa8')
    dcm = dcmReader.safe_read_dicom_file(dcm_f)
    # 20230902140053.387500
    # 'Wed Oct 18 03:10:45 2023'
    # 1697616645.6933825

    dicom_files_orig = dicom_files[np.logical_not(derived)]
    np.diff(acq_time[np.logical_not(derived)])

    dcm0 = dcmReader.safe_read_dicom_file(dicom_files[2])
    dcm1 = dcmReader.safe_read_dicom_file(dicom_files[0])
    dcm2 = dcmReader.safe_read_dicom_file(dicom_files[1])
    dcm3 = dcmReader.safe_read_dicom_file(dicom_files[3])

    dcm_l1 = dcmReader.safe_read_dicom_file(dicom_files_orig[-1])
    dcm_l2 = dcmReader.safe_read_dicom_file(dicom_files_orig[-2])
    with open('dcm_header_l1.txt', 'w') as fd:
        print(dcm_l1, file=fd)

    with open('dcm_header_l2.txt', 'w') as fd:
        print(dcm_l2, file=fd)

    # dicom_file = Path(
    #     'test_data/XA30-anon-hierarchical/Patientname_19800101/E26580742/'
    #     'S011/IM-0011-0001.dcm')
    # dicom_files = [dicom_file]
    # nii_img = dcmReader.dcm2nii(dicom_files)
