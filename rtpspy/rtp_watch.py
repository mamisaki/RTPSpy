#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Watch for new image files and feed them into the real-time processing
pipeline. DICOM is supported only for Siemens (XA enhanced multi-frame and
classic mosaic).

@author: mmisaki@libr.net
"""

# %% import ==================================================================#
from pathlib import Path
import os
import sys
import time
import re
import traceback
import shutil

import numpy as np
import nibabel as nib
import pydicom

from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

try:
    from .rtp_common import RTP
    from .rtp_volume import RtpVolume
except Exception:
    # For DEBUG environment
    try:
        sys.path.append("./")
        from rtpspy.rtp_common import RTP
        from rtpspy.rtp_volume import RtpVolume
    except Exception:
        from rtp_common import RTP
        from rtp_volume import RtpVolume


# %% class RtpWatch ===========================================================
class RtpWatch(RTP):
    """Read image files in real time, convert them to NIfTI, and feed the
    data into the RTP pipeline.
    Procedures:
    - Monitor a directory (self.watch_dir) to which DICOM files are exported
      in real time.
    - Read a DICOM file and convert it to a nibabel.Nifti1Image object. The
      object is passed to the downstream process connected to
      self.next_proc.
    - If self.clean_ready is True, the files in self.watch_dir are moved to
      'self.work_dir / dicom' by self.ready_proc() at the beginning.

    Refer also to the python watchdog package:
    https://pypi.org/project/watchdog/
    """

    # GUI panel (rtpspy.ui.watch_ui)
    _ui_class = "watch_ui:RtpWatchUI"

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(
        self,
        watch_dir=None,
        watch_file_pattern=r".+\.dcm$",
        file_type="Dicom",
        read_timeout=3,
        polling_observer=False,
        polling_timeout=1,
        **kwargs,
    ):
        """
        Parameters
        ----------
        watch_dir : Path or str
            Directory to which DICOM images are exported in real time.
            Watchdog monitors the creation of files in this directory
            recursively, filtering the filename with watch_file_pattern
            (regular expression).
        watch_file_pattern : str (regular expression)
            Regular expression to filter the filename monitored by a watchdog.
            The default is r'.+\\.dcm$'.
        file_type : str
            Type of image file. The following types are supported:
            'Dicom', 'Nifti', and 'AFNI_BRIK'.
        read_timeout : float, optional
            Timeout (seconds) for reading a file.
        polling_observer : bool, optional
            Flag to use a PollingObserver if watch_dir is not on a local
            file system. The default is False.
        polling_timeout : float, optional
            Timeout (seconds) for polling watch_dir.
        """
        super().__init__(**kwargs)  # call __init__() in RTP base class

        self._supported_file_types = ["Dicom", "Nifti", "AFNI_BRIK"]
        # Initialize parameters
        self.watch_dir = watch_dir
        self.watch_file_pattern = watch_file_pattern
        self.file_type = file_type
        self.read_timeout = read_timeout
        self.polling_observer = polling_observer
        self.polling_timeout = polling_timeout
        self.clean_ready = False

        self._last_proc_f = ""  # Last processed filename
        self._done_proc = -1  # Number of the processed volumes
        self._proc_ready = False
        self.scan_name = None  # Used for saving a physio signal file.
        self.nii_save_filename = None

        if self.file_type not in self._supported_file_types:
            errmsg = f"file_type = {self.file_type} is not supported."
            errmsg += " file_type is reset to 'Nifti'."
            self._logger.error(errmsg)
            self.file_type = "Nifti"

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ready_proc(self):
        if self.clean_ready:
            self.clean_files()

        self._last_proc_f = ""
        self._done_proc = -1
        self._vol_num = 0
        self.scan_name = None
        self.nii_save_filename = None

        self._proc_ready = True
        if self.next_proc is not None:
            self._proc_ready &= self.next_proc.ready_proc()

        if self._proc_ready:
            self.start_watching()

        return self._proc_ready

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end_reset(self):
        """End process and reset process parameters."""
        self._logger.info(f"Reset {self.__class__.__name__} module.")

        # Stop the observer, waiting for the running process to finish.
        self.stop_watching()
        return super().end_reset()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def start_watching(self):
        """Start watchdog observer monitoring the watch_dir directory."""
        if self.watch_dir is None or not Path(self.watch_dir).is_dir():
            errmsg = f"No directory: {self.watch_dir}"
            self._logger.error(errmsg)
            self.err_popup(errmsg)
            return
        self.watch_dir = Path(self.watch_dir)

        # Stop the running observer, if any, before starting a new one
        if hasattr(self, "_observer"):
            self.stop_watching()

        # Start observer
        if self.polling_observer:
            self._observer = PollingObserver(timeout=self.polling_timeout)
        else:
            self._observer = Observer()

        self._event_handler = RtpWatch.RTPFileHandler(
            self.watch_file_pattern, callback=self.do_proc
        )

        self._observer.schedule(
            self._event_handler, str(self.watch_dir), recursive=True
        )
        self._observer.start()
        self._logger.info(
            "Start observer monitoring "
            + f"{self.watch_dir}/**{self.watch_file_pattern}"
        )

    # /////////////////////////////////////////////////////////////////////////
    class RTPFileHandler(FileSystemEventHandler):
        """Call the callback for a new file that matches the pattern."""

        def __init__(self, watch_file_pattern, callback):
            """
            Parameters
            ----------
            watch_file_pattern : str
                Regular expression to filter the file.
            callback : function
                Function to apply to the file.
            """
            super().__init__()
            self.watch_file_pattern = watch_file_pattern
            self.callback = callback

        def on_created(self, event):
            if event.is_directory:
                return

            if re.search(self.watch_file_pattern, Path(event.src_path).name):
                self.callback(event.src_path)

        def on_moved(self, event):
            # A writer may save to a temporary name and rename it.
            if event.is_directory:
                return

            if re.search(self.watch_file_pattern, Path(event.dest_path).name):
                self.callback(event.dest_path)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def stop_watching(self):
        if hasattr(self, "_observer"):
            if self._observer.is_alive():
                self._observer.stop()
                self._observer.join()
            del self._observer
        self._logger.info("Stop watchdog observer.")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _make_path_safe(self, input_string, max_length=255):
        # Replace each run of characters that are not safe for paths with
        # one underscore
        cleaned_string = re.sub(r'[\\/:"*?<>|]+', "_", input_string)

        # Replace spaces and other non-alphanumeric characters with underscores
        cleaned_string = re.sub(r"[^a-zA-Z0-9._-]", "_", cleaned_string)

        # Ensure the resulting string is within length limits
        if len(cleaned_string) > max_length:
            cleaned_string = cleaned_string[:max_length]

        return cleaned_string

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def do_proc(self, file_path):
        """
        Read a new image file and pass the volume to the next process.

        This overrides the template RTP.do_proc, as this module is the head
        of the processing chain: it receives a file path (from the watchdog
        observer) instead of an image, it counts only the files that make a
        valid volume, and its processing time is the delay from the file
        creation. The image is passed as an RtpVolume (with vol_idx,
        acq_time = file creation time, and proc_time = time stamp set). The
        output step is shared with RTP (_pass_to_next).

        Parameters
        ----------
        file_path : str or Path
            File path found by a watchdog observer.
        """
        try:
            file_path = Path(file_path)

            # Ignore the previously processed file
            if self._last_proc_f == str(file_path):
                self._logger.debug(f"{file_path.name} has been processed.")
                return
            self._last_proc_f = str(file_path)

            # --- Process the file --------------------------------------------
            if self._proc_start_idx < 0:
                self._proc_start_idx = 0  # 0-base index

            # 1-base number of this volume. self._vol_num is incremented only
            # when a valid image is made, so that skipped or failed files
            # do not consume a volume index.
            vol_num = self._vol_num + 1

            fmri_img = None
            if self.file_type == "Dicom":
                fmri_img = self.process_Dicom(file_path, vol_num)
            elif self.file_type in ("Nifti", "AFNI_BRIK"):
                fmri_img = self.process_nibabel(file_path, vol_num)

            if fmri_img is None:
                return

            # Increment the number of received volumes
            self._vol_num = vol_num  # 1-base

            # Record process time
            tstamp = time.time()
            self._proc_time.append(tstamp)
            acq_time = file_path.stat().st_ctime
            proc_delay = tstamp - acq_time
            if self.save_delay:
                self.proc_delay.append(proc_delay)

            # Log message
            if len(self._proc_time) > 1:
                t_interval = self._proc_time[-1] - self._proc_time[-2]
            else:
                t_interval = -1
            msg = f"#{self._vol_num};Read file;{file_path.name}"
            msg += f";tstamp={tstamp}"
            msg += f";took {proc_delay:.4f}s"
            msg += f";interval {t_interval:.4f}s"
            self._logger.info(msg)

            # The volume made in this module (an image returned by an
            # overridden reader is converted)
            fmri_img = RtpVolume.as_volume(fmri_img)
            fmri_img.vol_idx = self._vol_num - 1
            fmri_img.acq_time = acq_time
            fmri_img.proc_time = tstamp

            # Run the next process. self.proc_data is a copy of the data,
            # as the next process may modify the array in place.
            self._pass_to_next(
                fmri_img, self._vol_num - 1, self._proc_time[-1], copy=True)

            # Record the number of the processed volume
            self._done_proc = self._vol_num

            # Keep the image read in this module (its data copy, if it was
            # passed to the next process)
            if self.save_proc:
                if self.next_proc:
                    fmri_img = fmri_img.with_data(self.proc_data)
                self.keep_processed_image(fmri_img, vol_num=None,
                                          save_temp=self.online_saving)

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errmsg = "".join(
                traceback.format_exception(exc_type, exc_obj, exc_tb))
            self._logger.error(errmsg)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _read_dicom_file(self, file_path):
        """Read a DICOM file, retrying until the file is completely written.

        Returns the pydicom dataset, or None if the file could not be read
        within self.read_timeout seconds.
        """
        st = time.time()
        last_size = -1
        while True:
            sleep = 0.01
            try:
                dcm = pydicom.dcmread(file_path)
                if "PixelData" in dcm:
                    # Decoding fails if the pixel data is incomplete.
                    _ = dcm.pixel_array
                    return dcm

                # No pixel data: accept it when the file size is stable.
                fsize = Path(file_path).stat().st_size
                if fsize == last_size:
                    return dcm
                last_size = fsize
                errmsg = "No pixel data"
                sleep = 0.05
            except Exception as e:
                errmsg = str(e)

            if time.time() - st > self.read_timeout:
                self._logger.error(
                    f"Failed to read {file_path} as DICOM: {errmsg}")
                return None
            time.sleep(sleep)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def process_Dicom(self, file_path, vol_num):
        # --- Read DICOM file -------------------------------------------------
        dcm = self._read_dicom_file(file_path)
        if dcm is None:
            return

        if "PixelData" not in dcm:
            self._logger.debug(f"{file_path.name} is not an image file.")
            return

        manufacturer = str(dcm.get("Manufacturer", "")).lower()
        if "siemens" not in manufacturer:
            self._logger.debug(f"{file_path.name} is not a Siemens DICOM.")
            return

        soft_ver = dcm.get("SoftwareVersions", "")
        if isinstance(soft_ver, pydicom.multival.MultiValue):
            soft_ver = "\\".join(soft_ver)
        is_XA = "XA" in str(soft_ver).upper()

        fmri_img = self._proc_SiemensDicom(file_path, dcm, is_XA)
        if fmri_img is None:
            return

        if self.nii_save_filename is None:
            # Set the save filename from the first volume of the series
            patient = str(dcm.PatientName).split("^")
            if re.match(r"\w\w\d\d\d", patient[0]):
                # Subject ID format of two letters and three digits
                sub = patient[0]
            else:
                sub = "_".join(patient)
            ser = dcm.SeriesNumber
            serDesc = dcm.SeriesDescription
            save_filename = f"sub-{sub}_ser-{int(ser)}"
            if len(serDesc):
                save_filename += f"_desc-{serDesc}"
            self.nii_save_filename = self._make_path_safe(save_filename)
            self.scan_name = f"Ser-{ser}"

        nii_fname = self.nii_save_filename + f"_{vol_num:04d}.nii.gz"
        fmri_img.set_filename(nii_fname)

        return fmri_img

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _proc_SiemensDicom(self, file_path, dcm, is_XA):
        """Check whether dcm is an fMRI volume to process, and convert it to
        a NIfTI image.

        Parameters
        ----------
        file_path : Path
            DICOM file path.
        dcm : pydicom.dataset.Dataset
            DICOM dataset read from file_path.
        is_XA : bool
            True for XA software (enhanced multi-frame DICOM); False for
            older software (mosaic DICOM).

        Returns
        -------
        nibabel.Nifti1Image or None
            None if the file is not processed.
        """
        try:
            imageType = "\\".join(dcm.ImageType)
            # Ignore non-fMRI file
            if is_XA:
                is_fmri = (
                    "FMRI" in imageType
                    or "ep2d_fid_" in dcm.ProtocolName
                    or "ep2d_bold_" in dcm.ProtocolName
                )
            else:
                is_fmri = (
                    "FMRI" in imageType or "epfid2d" in dcm.SequenceName
                )
            if not is_fmri:
                self._logger.debug(f"{file_path.name} is not an fMRI file.")
                return

            # Ignore derived file
            if "DERIVED" in imageType:
                self._logger.debug(f"{file_path.name} is a derived file.")
                return

            seriesDescription = dcm.SeriesDescription
            # Ignore localizer and scout
            if "localizer" in seriesDescription:
                self._logger.debug(f"{file_path.name} is a localizer series.")
                return

            if "scout" in seriesDescription:
                self._logger.debug(f"{file_path.name} is a scout series.")
                return

            # Ignore MoCoSeries
            if seriesDescription == "MoCoSeries":
                self._logger.debug(f"{file_path.name} is a MoCo series.")
                return

            # Ignore phase image
            if seriesDescription.endswith("_Pha"):
                self._logger.debug(f"{file_path.name} is a phase series.")
                return

            if is_XA:
                return self.dcm2nii_SiemensXADicom(dcm, file_path)
            else:
                return self.dcm2nii_SiemensMosaicDicom(dcm, file_path)

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errmsg = "".join(traceback.format_exception(
                exc_type, exc_obj, exc_tb))
            self._logger.error(errmsg)
            return

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def process_nibabel(self, file_path, vol_num):
        file_path = Path(file_path)
        st = time.time()
        fsize = -1
        while True:
            # Wait until the file is completely written: the file size is
            # stable and the data can be read.
            try:
                if file_path.stat().st_size == fsize:
                    # Load file
                    load_img = nib.load(file_path)

                    # get_fdata will fail if the file is incomplete.
                    load_img.get_fdata()
                    break
                fsize = file_path.stat().st_size
                errmsg = "File size is changing"
            except Exception as e:
                errmsg = str(e)

            if time.time() - st > self.read_timeout:
                self._logger.error(f"Failed to read {file_path}: {errmsg}")
                return
            time.sleep(0.001)

        try:
            # Set save_filename
            if file_path.suffix == ".gz":
                save_filename = Path(file_path.stem).stem
            else:
                save_filename = file_path.stem
            save_filename = re.sub(r"\+orig.*", "", save_filename) + ".nii.gz"
            self.nii_save_filename = self._make_path_safe(save_filename)

            fmri_img = RtpVolume(
                load_img.dataobj, load_img.affine, header=load_img.header
            )
            fmri_img.set_filename(self.nii_save_filename)

            if self.scan_name is None:
                # scan_name is used for saving a physio signal file.
                ma = re.search(r".+scan_\d+__\d+", file_path.stem)
                if ma:
                    self.scan_name = ma.group()

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errmsg = "".join(traceback.format_exception(
                exc_type, exc_obj, exc_tb))
            self._logger.error(errmsg)
            return

        return fmri_img

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _get_pixel_array(self, dcm, file_path=None):
        """Get the pixel array, re-reading file_path (with self.read_timeout)
        if the pixel data in dcm is incomplete."""
        try:
            return dcm.pixel_array
        except Exception:
            if file_path is None:
                raise

        dcm = self._read_dicom_file(file_path)
        if dcm is None or "PixelData" not in dcm:
            raise RuntimeError(f"Failed to read pixel data in {file_path}")
        return dcm.pixel_array

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _make_nifti(self, img_array, affine, ori_row, ori_col, TR):
        """Make a NIfTI image in LPI orientation from a DICOM volume.

        Parameters
        ----------
        img_array : 3D array
            Volume data in the DICOM order (slice, row, column).
        affine : 4x4 array
            Affine matrix from the (column, row, slice) voxel index to the
            RAS+ coordinates (mm), with the translation at the first voxel.
        ori_row, ori_col : array
            Direction cosines of the row and the column
            (ImageOrientationPatient).
        TR : float or None
            Repetition time (ms).

        Returns
        -------
        nibabel.Nifti1Image
        """
        # Flip the row axis to LPI: move the origin to the last row.
        point = np.array([[0, img_array.shape[1] - 1, 0, 1]]).T
        k = np.dot(affine, point)
        affine[:, 1] *= -1
        affine[:, 3] = k.ravel()

        # Transpose and flip to LPI
        img_array = np.transpose(img_array, (2, 1, 0))
        img_array = np.flip(img_array, axis=1)

        nii_img = RtpVolume(img_array, affine)

        # Set TR (seconds) in pixdim
        if TR is not None:
            nii_img.header.structarr["pixdim"][4] = TR / 1000.0

        nii_img.header.set_slope_inter(1, 0)
        # Set units for xyz (leave t as unknown)
        nii_img.header.set_xyzt_units(2)

        # Slice orientation: the dominant axis of the slice normal
        # (Axial: z (2), Sagittal: x (0), Coronal: y (1); also for oblique
        # slices)
        slice = int(np.argmax(np.abs(np.cross(ori_row, ori_col))))
        nii_img.header.set_dim_info(slice=slice)

        return nii_img

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def dcm2nii_SiemensXADicom(self, dcm, file_path=None):
        """Convert a Siemens XA (enhanced multi-frame) DICOM to NIfTI.

        Parameters
        ----------
        dcm : pydicom.dataset.Dataset
            DICOM dataset.
        file_path : Path, optional
            DICOM file path, used to re-read the file if the pixel data is
            incomplete.

        Returns
        -------
        nibabel.Nifti1Image
        """
        # --- Read position and pixel_array data from dcm ---------------------
        assert "PerFrameFunctionalGroupsSequence" in dcm
        frames = dcm.PerFrameFunctionalGroupsSequence

        # Pixel spacing (should be the same for all slices)
        img_pix_space = frames[0].PixelMeasuresSequence[0].PixelSpacing

        # ImageOrientationPatient (should be the same for all slices):
        # direction cosines of the first row and the first column with
        # respect to the patient; the row values for the x, y, and z axes,
        # followed by the column values for the x, y, and z axes.
        img_ornt_pat = \
            frames[0].PlaneOrientationSequence[0].ImageOrientationPatient

        # Slice positions: x, y, and z coordinates of the upper left-hand
        # corner (the center of the first voxel) of each slice.
        img_pos = np.concatenate(
            [np.array(frame.PlanePositionSequence[0].ImagePositionPatient)[
                None, :] for frame in frames], axis=0)

        pixel_array = self._get_pixel_array(dcm, file_path)
        if pixel_array.ndim == 2:
            # Single-frame file: pixel_array is (rows, columns)
            pixel_array = pixel_array[None, :, :]

        # --- Affine matrix from voxel index to RAS+ (mm) ---------------------
        F = np.reshape(img_ornt_pat, (2, 3)).T
        # PixelSpacing = [row spacing (between rows), column spacing]
        dr, dc = [float(v) for v in img_pix_space]
        T1 = img_pos[0]
        TN = img_pos[-1]
        if len(img_pos) > 1:
            k = (TN - T1) / (len(img_pos) - 1)
        else:
            # Single slice: slice normal times the slice spacing
            pm = frames[0].PixelMeasuresSequence[0]
            ds = float(getattr(pm, "SpacingBetweenSlices", 0) or
                       getattr(pm, "SliceThickness", 1) or 1)
            k = np.cross(F[:, 0], F[:, 1]) * ds
        A = np.concatenate([F * [[dc, dr]], k[:, None], T1[:, None]], axis=1)
        A = np.concatenate([A, np.array([[0, 0, 0, 1]])], axis=0)
        # DICOM patient coordinates are LPS+; flip x and y to RAS+.
        A[:2, :] *= -1

        # --- TR --------------------------------------------------------------
        TR = None
        if "RepetitionTime" in dcm:
            TR = float(dcm.RepetitionTime)
        elif "SharedFunctionalGroupsSequence" in dcm and \
                "MRTimingAndRelatedParametersSequence" in \
                dcm.SharedFunctionalGroupsSequence[0]:
            MRTiming = dcm.SharedFunctionalGroupsSequence[
                0].MRTimingAndRelatedParametersSequence[0]
            TR = float(MRTiming.RepetitionTime)

        return self._make_nifti(pixel_array, A, F[:, 0], F[:, 1], TR)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def dcm2nii_SiemensMosaicDicom(self, dcm, file_path=None):
        """Convert a Siemens mosaic DICOM to NIfTI.

        Parameters
        ----------
        dcm : pydicom.dataset.Dataset
            DICOM dataset.
        file_path : Path, optional
            DICOM file path, used to re-read the file if the pixel data is
            incomplete.

        Returns
        -------
        nibabel.Nifti1Image

        This code is based on
        https://github.com/icometrix/dicom2nifti/blob/main/dicom2nifti/convert_siemens.py
        """
        pixel_array = self._get_pixel_array(dcm, file_path)

        # --- Read the slice layout from the ASCCONV protocol header ----------
        asconv_headers = re.findall(
            r"### ASCCONV BEGIN(.*)### ASCCONV END ###",
            dcm[(0x0029, 0x1020)].value.decode(encoding="ISO-8859-1"),
            re.DOTALL,
        )[0]

        size_z = int(
            re.findall(r"sSliceArray\.lSize\s*=\s*(\d+)", asconv_headers)[0])

        # Transverse position of each slice (0 if not given)
        slice_location = []
        for index in range(size_z):
            axial_result = re.findall(
                r"sSliceArray\.asSlice\[%s\]\.s"
                r"Position\.dTra\s*=\s*([-+]?[0-9]*\.?[0-9]*)"
                % index,
                asconv_headers,
            )
            if len(axial_result) > 0:
                slice_location.append(float(axial_result[0]))
            else:
                slice_location.append(0.0)

        invert = False
        invert_result = re.findall(
            r"sSliceArray\.ucImageNumbTra\s*=\s*([-+]?0?x?[0-9]+)",
            asconv_headers
        )
        if len(invert_result) > 0:
            invert_value = int(invert_result[0], 16)
            if invert_value >= 0:
                invert = True

        # Slices are stored in ascending order of the location unless the
        # image numbering is inverted.
        ascending = (slice_location[0] <= slice_location[1]) != invert

        # --- Reshape the mosaic to a volume ----------------------------------
        # Tiles are arranged row by row in a number_x x number_x grid.
        number_x = int(np.ceil(np.sqrt(size_z)))
        size_x = int(dcm.Columns / number_x)
        size_y = int(dcm.Rows / number_x)
        img_array = np.zeros([size_z, size_y, size_x], dtype=pixel_array.dtype)
        for z_index in range(size_z):
            y_index, x_index = divmod(z_index, number_x)
            if ascending:
                dst_z = z_index
            else:
                dst_z = size_z - (z_index + 1)
            img_array[dst_z, :, :] = pixel_array[
                size_y * y_index: size_y * (y_index + 1),
                size_x * x_index: size_x * (x_index + 1),
            ]

        # --- Affine matrix from voxel index to RAS+ (mm) ---------------------
        img_ori1 = np.array(dcm.ImageOrientationPatient)[0:3]
        img_ori2 = np.array(dcm.ImageOrientationPatient)[3:6]
        nrm = np.cross(img_ori1, img_ori2)
        dr = float(dcm.PixelSpacing[0])
        dc = float(dcm.PixelSpacing[1])
        image_pos = dcm.ImagePositionPatient
        ds = dcm.SpacingBetweenSlices
        affine = np.array(
            [
                [-img_ori1[0] * dc,
                 -img_ori2[0] * dr,
                 -ds * nrm[0],
                 -image_pos[0]],
                [-img_ori1[1] * dc,
                 -img_ori2[1] * dr,
                 -ds * nrm[1],
                 -image_pos[1]],
                [img_ori1[2] * dc,
                 img_ori2[2] * dr,
                 ds * nrm[2],
                 image_pos[2]],
                [0, 0, 0, 1],
            ]
        )

        # ImagePositionPatient of a mosaic is for the whole mosaic image.
        # Shift it to the first voxel of one tile.
        affine[0:3, [3]] += np.dot(
            affine[0:3, 0:2],
            np.array(
                [[(dcm.Columns - size_x) / 2], [(dcm.Rows - size_y) / 2]]),
        )

        # --- TR --------------------------------------------------------------
        if "RepetitionTime" in dcm:
            TR = float(dcm.RepetitionTime)
        else:
            TR = None

        return self._make_nifti(img_array, affine, img_ori1, img_ori2, TR)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def clean_files(self, *args, warning=None):
        """Move the files matching watch_file_pattern in watch_dir to
        'work_dir / dicom', keeping the subdirectory structure."""
        if self.watch_dir is None or not Path(self.watch_dir).is_dir():
            return

        rt_mv_dst = Path(self.work_dir) / "dicom"
        if not rt_mv_dst.is_dir():
            os.makedirs(rt_mv_dst)

        fs = [
            ff
            for ff in Path(self.watch_dir).glob("**/*")
            if re.search(self.watch_file_pattern, ff.name)
        ]

        for src_f in fs:
            dst_f = rt_mv_dst / src_f.relative_to(self.watch_dir)
            if not dst_f.parent.is_dir():
                os.makedirs(dst_f.parent)
            shutil.copy(src_f, dst_f)
            src_f.unlink()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_param(self, attr, val=None, reset_fn=None, echo=False):
        """
        When reset_fn is None, set_param is considered to be called from
        the load_parameters function.
        """
        self._logger.debug(f"set_param: {attr} = {val}")

        # -- Check value --
        if attr == "enabled":
            self._ui_update(attr, val)

        elif attr == "watch_dir":
            if val is None or not Path(val).is_dir():
                return

            self._ui_update(attr, val)

            val = Path(val)
            setattr(self, attr, val)

            if self.main_win is not None:
                self.main_win.set_watchDir(val)

            if reset_fn is not None:
                reset_fn(val)

        elif attr == "work_dir":
            if val is None or not Path(val).is_dir():
                return

            val = Path(val)
            setattr(self, attr, val)

            if self.main_win is not None:
                self.main_win.set_workDir(val)

        elif attr == "file_type":
            if val not in self._supported_file_types:
                self._logger.error(f"file_type = {val} is not supported")
                return

            if reset_fn is None:
                self._ui_update(attr, val)

        elif attr == "watch_file_pattern":
            if len(val) == 0:
                if reset_fn:
                    reset_fn(str(self.watch_file_pattern))
                return
            if reset_fn is None:
                self._ui_update(attr, val)

        elif attr in ("read_timeout", "polling_observer", "polling_timeout"):
            if reset_fn is None:
                self._ui_update(attr, val)

        elif attr in ("save_proc", "clean_ready"):
            self._ui_update(attr, val)

        elif attr == "rtp_ttl_physio_address":
            if type(val) is str:
                host, port = val.split(":")
                val = (host, int(port))

        elif reset_fn is None:
            # Ignore an unrecognized parameter
            if not hasattr(self, attr):
                errmsg = f"{attr} is an unrecognized parameter."
                self._logger.error(errmsg)
                return

        # -- Set value --
        setattr(self, attr, val)
        if echo:
            print(
                "{}.".format(self.__class__.__name__) + attr,
                "=", getattr(self, attr)
            )

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_params(self):
        all_opts = super().get_params()
        excld_opts = ("scan_name", "nii_save_filename")
        sel_opts = {}
        for k, v in all_opts.items():
            if k[0] == "_" or k in excld_opts:
                continue
            if isinstance(v, Path):
                v = str(v)
            sel_opts[k] = v

        sel_opts["watch_dir"] = str(self.watch_dir)

        return sel_opts

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        # Stop the observer thread
        if hasattr(self, "_observer") and self._observer.is_alive():
            self._observer.stop()
            self._observer.join()


# %% __main__ (test) ==========================================================
if __name__ == "__main__":
    # Copy DICOM files of a test series to the watch directory one by one,
    # restarting the watcher every 10 files.
    test_dir = Path(__file__).resolve().parent.parent / "tests"
    src_data = test_dir / "20240916.OrangePhantom.7T2024091600PH"
    watch_file_pattern = r".+\.dcm"
    test_work_dir = test_dir
    file_type = "Dicom"

    # --- Prepare the test data -----------------------------------------------
    try:
        assert src_data.is_dir()
        src_files = sorted(
            ff for ff in src_data.glob("*09_*")
            if re.search(watch_file_pattern, ff.name)
        )

        # Set export directory
        watch_dir = test_work_dir / "watch"
        if not watch_dir.is_dir():
            watch_dir.mkdir()
        else:
            for rmf in watch_dir.glob("*"):
                if rmf.is_dir():
                    shutil.rmtree(rmf)
                else:
                    rmf.unlink()

        work_dir = test_work_dir / "work"
        if not work_dir.is_dir():
            work_dir.mkdir()

        # Create RtpWatch instance
        rtp_watch = RtpWatch(watch_dir, watch_file_pattern)

        rtp_watch.save_proc = True  # Save result
        rtp_watch.online_saving = True  # Online saving
        rtp_watch.save_delay = True
        rtp_watch.work_dir = work_dir
        rtp_watch.file_type = file_type

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        errstr = "".join(traceback.format_exception(exc_type, exc_obj, exc_tb))
        sys.stderr.write(errstr)
        sys.exit()

    # --- Start simulation ----------------------------------------------------
    # Copy test data volume by volume
    n_copy = 0
    for src_f in src_files:
        dst_f = watch_dir / src_f.relative_to(src_data)
        if not dst_f.parent.is_dir():
            os.makedirs(dst_f.parent)

        shutil.copy(src_f, dst_f)
        n_copy += 1
        time.sleep(1)

        if (n_copy + 1) % 10 == 0:
            rtp_watch.end_reset()
            time.sleep(3)
            rtp_watch.ready_proc()

    # Clean up watch_dir
    for ff in watch_dir.glob("*"):
        if ff.is_dir():
            continue
        ff.unlink()

    watch_dir.rmdir()
