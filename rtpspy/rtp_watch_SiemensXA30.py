#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Watching new file creation for real-time processing
Support only for Siemens XA30 DICOM

@author: mmisaki@libr.net
"""


# %% import ==================================================================#
from pathlib import Path
import os
import sys
import time
import re
import traceback
import logging
import shutil

import numpy as np
import nibabel as nib
import pydicom

from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
from PyQt5 import QtWidgets

try:
    from .rtp_common import RTP
except Exception:
    # For DEBUG environment
    try:
        sys.path.append("./")
        from rtpspy.rtp_common import RTP
    except Exception:
        from rtp_common import RTP


# %% class RtpWatch ===========================================================
class RtpWatch(RTP):
    """Read the DICOM file in real time, convert to NIfTI and feed the data
    into the RTP pipeline.
    Procedures:
    - Monitor a directory (self.watch_dir) to which DICOM files are exported
      in real time.
    - Reads a dicom file and converts it to a nibable.Nifti1Image object. The
      object is passed to the downsterm process connected to self.next_proc.
    - The DICOM files in self.watch_dir are moved to another location in the
      'self.work_dir / dicom' directory by a self.ready_proc() call at the
      beginning.

    Refer also to the python watchdog package:
    https://pypi.org/project/watchdog/
    """
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, watch_dir=None, watch_file_pattern=r'.+\.dcm',
                 dcmread_timeout=1, polling_observer=False, polling_timeout=1,
                 **kwargs):
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
            The default is r'.+\\.dcm'.
        dcmread_timeout : float, optional
            Timeout (second) for reading a DICOM file.
        polling_observer : bool, optional
            Flag to use a PollingObserver if the dcm_dir is not on a local
            file system. The default is False.
        polling_timeout : float, optional
            Timeout (second) for polling watch_dir.
        """
        super().__init__(**kwargs)  # call __init__() in RTP base class
        self._logger = logging.getLogger('RtpWatch')

        # Initialize parameters
        self.watch_dir = watch_dir
        self.watch_file_pattern = watch_file_pattern
        self.dcmread_timeout = dcmread_timeout
        self.polling_observer = polling_observer
        self.polling_timeout = polling_timeout
        self.clean_ready = False
        self._dcmread_timeout = 1

        self._last_proc_f = ''  # Last processed filename
        self._done_proc = -1  # Number of the processed volume
        self._proc_ready = False

        self.scan_name = None  # Used for saving a physio signal file.
        self.nii_save_filename = None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ready_proc(self):
        if self.clean_ready:
            self.clean_files()

        self._last_proc_f = ''
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
        """ End process and reset process parameters. """
        self._logger.info(f"Reset {self.__class__.__name__} module.")

        # Wait for process in watch thread finish.
        self.stop_watching()
        super().end_reset()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def start_watching(self):
        """ Start watchdog observer monitoring the watch_dir directory.
        """
        if self.watch_dir is None or not self.watch_dir.is_dir():
            errmsg = f'No directory: {self.watch_dir}'
            self._logger.error(errmsg)
            self.err_popup(errmsg)
            return

        # Start observer
        if self.polling_observer:
            self._observer = PollingObserver(timeout=self.polling_timeout)
        else:
            self._observer = Observer()

        self._event_handler = \
            RtpWatch.RTPFileHandler(
                self.watch_file_pattern, callback=self.do_proc)

        self._observer.schedule(self._event_handler, self.watch_dir,
                                recursive=True)
        self._observer.start()
        self._logger.info(
            "Start observer monitoring " +
            f"{self.watch_dir}/**{self.watch_file_pattern}")

    # /////////////////////////////////////////////////////////////////////////
    class RTPFileHandler(FileSystemEventHandler):
        """ File handling class """

        def __init__(self, watch_file_pattern, callback):
            """
            Parameters
            ----------
            watch_file_pattern : str
                Regular expression to filter the file.
            callback : function
                Applied function to the file.
            """
            super().__init__()
            self.watch_file_pattern = watch_file_pattern
            self.callback = callback

        def on_created(self, event):
            if event.is_directory:
                return

            if re.search(self.watch_file_pattern, Path(event.src_path).name):
                self.callback(event.src_path)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def stop_watching(self):
        if hasattr(self, '_observer'):
            if self._observer.is_alive():
                self._observer.stop()
                self._observer.join()
            del self._observer
        self._logger.info("Stop watchdog observer.")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _make_path_safe(self, input_string, max_length=255):
        # Remove characters that are not safe for paths
        cleaned_string = re.sub(r'[\\/:"*?<>|]+', '_', input_string)

        # Replace spaces and other non-alphanumeric characters with underscores
        cleaned_string = re.sub(r'[^a-zA-Z0-9._-]', '_', cleaned_string)

        # Ensure the resulting string is within length limits
        if len(cleaned_string) > max_length:
            cleaned_string = cleaned_string[:max_length]

        return cleaned_string

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def do_proc(self, file_path):
        """
        Parameters
        ----------
        file_path : str or Path
            File path found by a watchdog observer.
        """
        try:
            file_path = Path(file_path)

            # --- Read header and check if this file should be processed ------
            # Ignore the previously processed file
            if self._last_proc_f == str(file_path):
                self._logger.debug(f"{file_path.name} has been processed.")
                return
            self._last_proc_f = str(file_path)

            # Read dicom file
            st = time.time()
            dcm = None
            try:
                dcm = pydicom.dcmread(file_path)
                _ = dcm.pixel_array
            except Exception:
                pass

            while dcm is None and time.time() - st < self._dcmread_timeout:
                # Wait until the file is readable.
                try:
                    dcm = pydicom.dcmread(file_path)
                    _ = dcm.pixel_array
                except Exception:
                    time.sleep(0.01)

            if dcm is None:
                errmsg = f"Failed to read {file_path} as DICOM"
                self._logger.error(errmsg)
                return

            imageType = '\\'.join(dcm.ImageType)
            # Ignore non fMRI file
            if 'FMRI' not in imageType and \
                    'ep2d_fid_' not in dcm.ProtocolName and \
                    'ep2d_bold_' not in dcm.ProtocolName:
                self._logger.debug(f"{file_path.name} is not a fMRI file.")
                return

            # Ignore derived file
            if 'DERIVED' in imageType:
                self._logger.debug(f"{file_path.name} is a derived file.")
                return

            seriesDescription = dcm.SeriesDescription
            # Ignore localizer and scout
            if 'localizer' in seriesDescription:
                self._logger.debug(f"{file_path.name} is a localizer series.")
                return

            if 'scout' in seriesDescription:
                self._logger.debug(f"{file_path.name} is a scout series.")
                return

            # Ignore MoCoSeries
            if seriesDescription == 'MoCoSeries':
                self._logger.debug(f"{file_path.name} is a MoCo series.")
                return

            # Ignore Phase image
            if seriesDescription.endswith('_Pha'):
                self._logger.debug(f"{file_path.name} is a Phase series.")
                return

            # --- Process the file --------------------------------------------
            # Increment the number of received volume
            self._vol_num += 1  # 1-base
            if self._proc_start_idx < 0:
                self._proc_start_idx = 0  # 0-base index

            fmri_img = self.dcm2nii(dcm, file_path)

            if self.nii_save_filename is None:
                # Set save_filename
                patinet = dcm.PatientName.split('^')
                if re.match(r'\w\w\d\d\d', patinet[0]):  # LIBR ID
                    sub = patinet[0]
                else:
                    sub = '_'.join(patinet)
                ser = dcm.SeriesNumber
                serDesc = dcm.SeriesDescription
                self.nii_save_filename = f"sub-{sub}_ser-{int(ser)}"
                if len(serDesc):
                    self.nii_save_filename += f"_desc-{serDesc}"
                self.nii_save_filename = self._make_path_safe(
                    self.nii_save_filename)
                self.scan_name = f"Ser-{ser}"

            nii_fname = self.nii_save_filename + \
                f"_{self._vol_num+1:04d}.nii.gz"
            fmri_img.set_filename(nii_fname)

            # Record process time
            tstamp = time.time()
            self._proc_time.append(tstamp)
            proc_delay = tstamp - file_path.stat().st_ctime
            if self.save_delay:
                self.proc_delay.append(proc_delay)

            # log
            f = file_path.name
            if len(self._proc_time) > 1:
                t_interval = self._proc_time[-1] - self._proc_time[-2]
            else:
                t_interval = -1
            msg = f"#{self._vol_num};tstamp={tstamp}"
            msg += f";Read {f};took {proc_delay:.4f}s"
            msg += f";interval {t_interval:.4f}s"
            self._logger.info(msg)

            if self.next_proc:
                # Keep the current processed data
                self.proc_data = np.asanyarray(fmri_img.dataobj)
                save_name = fmri_img.get_filename()

                # Run the next process
                self.next_proc.do_proc(fmri_img, vol_idx=self._vol_num-1,
                                       pre_proc_time=self._proc_time[-1])

            # Record the number of the processed volume
            self._done_proc = self._vol_num

            # Save processed image
            if self.save_proc:
                if self.next_proc is not None:
                    # Recover the processed data in this module
                    fmri_img.uncache()
                    fmri_img._dataobj = self.proc_data
                    fmri_img.set_data_dtype = self.proc_data.dtype
                    fmri_img.set_filename(save_name)

                self.keep_processed_image(fmri_img,
                                          save_temp=self.online_saving)

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errmsg = ''.join(
                traceback.format_exception(exc_type, exc_obj, exc_tb))
            self._logger.error(errmsg)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def dcm2nii(self, dcm, file_path=None):
        """Comver dicom (Siemense XA30) to NIfTI

        Args:
            dcm (_type_): _description_

        Returns:
            _type_: _description_
        """
        # --- Read position and pixel_array data from dcm---
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

        pix_data_len = int((dcm.Rows * dcm.Columns * dcm.BitsAllocated *
                            dcm.SamplesPerPixel * dcm.NumberOfFrames) / 8)
        while True:  # Wait for PixelData to be ready
            try:
                assert len(dcm.PixelData) == pix_data_len
                pixel_array = dcm.pixel_array
                break
            except Exception as e:
                if file_path is not None:
                    dcm = pydicom.dcmread(file_path)
                else:
                    raise e

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

        nii_img = nib.Nifti1Image(img_array, A)

        # Set TR and TE in pixdim and db_name field
        if 'RepetitionTime' in dcm:
            TR = float(dcm.RepetitionTime)
        elif 'SharedFunctionalGroupsSequence' in dcm:
            if 'MRTimingAndRelatedParametersSequence' in \
                    dcm.SharedFunctionalGroupsSequence[0]:
                MRTiming = dcm.SharedFunctionalGroupsSequence[
                    0].MRTimingAndRelatedParametersSequence[0]
                TR = float(MRTiming.RepetitionTime)
        else:
            TR = None
        if TR is not None:
            nii_img.header.structarr['pixdim'][4] = TR / 1000.0

        nii_img.header.set_slope_inter(1, 0)
        # set units for xyz (leave t as unknown)
        nii_img.header.set_xyzt_units(2)

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
        nii_img.header.set_dim_info(slice=slice)

        return nii_img

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def clean_files(self, *args, warning=None):
        if self.watch_dir is None or not Path(self.watch_dir).is_dir():
            return

        rt_mv_dst = Path(self.work_dir) / 'dicom'
        if not rt_mv_dst.is_dir():
            os.makedirs(rt_mv_dst)

        fs = [ff for ff in Path(self.watch_dir).glob('**/*')
              if re.search(self.watch_file_pattern, ff.name)]

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
        load_parameters function.
        """

        # -- Check value --
        if attr == 'enabled':
            if hasattr(self, 'ui_enabled_rdb'):
                self.ui_enabled_rdb.setChecked(val)

            if hasattr(self, 'ui_objs'):
                for ui in self.ui_objs:
                    ui.setEnabled(val)

        elif attr == 'watch_dir':
            if val is None or not Path(val).is_dir():
                return

            if hasattr(self, 'ui_wdir_lnEd'):
                self.ui_wdir_lnEd.setText(str(val))

            val = Path(val)
            setattr(self, attr, val)

            if self.main_win is not None:
                self.main_win.set_watchDir(val)

            if reset_fn is not None:
                reset_fn(val)

        elif attr == 'work_dir':
            if val is None or not Path(val).is_dir():
                return

            val = Path(val)
            setattr(self, attr, val)

            if self.main_win is not None:
                self.main_win.set_workDir(val)

        elif attr == 'watch_file_pattern':
            if len(val) == 0:
                if reset_fn:
                    reset_fn(str(self.watch_file_pattern))
                return
            if reset_fn is None:
                if hasattr(self, 'ui_watchPat_lnEd'):
                    self.ui_watchPat_lnEd.setText(str(val))

        elif attr == 'dcmread_timeout' and reset_fn is None:
            if hasattr(self, 'ui_dcmreadTimeout_dSpBx'):
                self.ui_dcmreadTimeout_dSpBx.setValue(val)

        elif attr == 'polling_observer':
            if reset_fn is None:
                if hasattr(self, 'ui_pollingObserver_chb'):
                    self.ui_pollingObserver_chb.setChecked(val)

        elif attr == 'polling_timeout' and reset_fn is None:
            if hasattr(self, 'ui_pollingTimeout_dSpBx'):
                self.ui_pollingTimeout_dSpBx.setValue(val)

        elif attr == 'save_proc':
            if hasattr(self, 'ui_saveProc_chb'):
                self.ui_saveProc_chb.setChecked(val)

        elif attr == 'clean_ready':
            if hasattr(self, 'ui_cleanReady_chb'):
                self.ui_cleanReady_chb.setChecked(val)

        elif attr == 'rtp_physio_address':
            if type(val) is str:
                host, port = val.split(':')
                val = (host, int(port))

        elif reset_fn is None:
            # Ignore an unrecognized parameter
            if not hasattr(self, attr):
                errmsg = f"{attr} is unrecognized parameter."
                self._logger.error(errmsg)
                return

        # -- Set value --
        setattr(self, attr, val)
        if echo:
            print("{}.".format(self.__class__.__name__) + attr, '=',
                  getattr(self, attr))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ui_set_param(self):

        ui_rows = []
        self.ui_objs = []

        # enabled
        self.ui_enabled_rdb = QtWidgets.QRadioButton("Enable")
        self.ui_enabled_rdb.setChecked(self.enabled)
        self.ui_enabled_rdb.toggled.connect(
                lambda checked:
                self.set_param('enabled', checked,
                               self.ui_enabled_rdb.setChecked))
        ui_rows.append((self.ui_enabled_rdb, None))

        # watch_dir
        var_lb = QtWidgets.QLabel("Watch directory :")
        self.ui_wdir_lnEd = QtWidgets.QLineEdit()
        self.ui_wdir_lnEd.setReadOnly(True)
        self.ui_wdir_lnEd.setStyleSheet(
            'background: white; border: 0px none;')
        if self.watch_dir is not None and Path(self.watch_dir).is_dir():
            self.ui_wdir_lnEd.setText(str(self.watch_dir))
        ui_rows.append((var_lb, self.ui_wdir_lnEd))
        self.ui_objs.extend([var_lb, self.ui_wdir_lnEd])

        # watch_file_pattern
        var_lb = QtWidgets.QLabel("Watch pattern :")
        self.ui_watchPat_lnEd = QtWidgets.QLineEdit()
        self.ui_watchPat_lnEd.setText(str(self.watch_file_pattern))
        self.ui_watchPat_lnEd.editingFinished.connect(
                lambda: self.set_param('watch_file_pattern',
                                       self.ui_watchPat_lnEd.text(),
                                       self.ui_watchPat_lnEd.setText))
        ui_rows.append((var_lb, self.ui_watchPat_lnEd))
        self.ui_objs.extend([var_lb, self.ui_watchPat_lnEd])

        # dcmread_timeout
        var_lb = QtWidgets.QLabel("File reading timeout :")
        self.ui_dcmreadTimeout_dSpBx = QtWidgets.QDoubleSpinBox()
        self.ui_dcmreadTimeout_dSpBx.setMinimum(0.0)
        self.ui_dcmreadTimeout_dSpBx.setSingleStep(0.1)
        self.ui_dcmreadTimeout_dSpBx.setDecimals(3)
        self.ui_dcmreadTimeout_dSpBx.setSuffix(" seconds")
        self.ui_dcmreadTimeout_dSpBx.setValue(self.dcmread_timeout)
        self.ui_dcmreadTimeout_dSpBx.valueChanged.connect(
                lambda x: self.set_param(
                    'dcmread_timeout', x,
                    self.ui_dcmreadTimeout_dSpBx.setValue))
        ui_rows.append((var_lb, self.ui_dcmreadTimeout_dSpBx))
        self.ui_objs.extend([var_lb, self.ui_dcmreadTimeout_dSpBx])

        # polling_observer check
        self.ui_pollingObserver_chb = QtWidgets.QCheckBox(
            "Use PollingObserver")
        self.ui_pollingObserver_chb.setChecked(self.polling_observer)
        self.ui_pollingObserver_chb.stateChanged.connect(
             lambda: self.set_param('polling_observer',
                                    self.ui_pollingObserver_chb.isChecked(),
                                    self.ui_pollingObserver_chb.setChecked))
        ui_rows.append((None, self.ui_pollingObserver_chb))
        self.ui_objs.extend([self.ui_pollingObserver_chb])

        # polling_timeout
        var_lb = QtWidgets.QLabel("Polling timeout :")
        self.ui_pollingTimeout_dSpBx = QtWidgets.QDoubleSpinBox()
        self.ui_pollingTimeout_dSpBx.setMinimum(0.0)
        self.ui_pollingTimeout_dSpBx.setSingleStep(0.1)
        self.ui_pollingTimeout_dSpBx.setDecimals(3)
        self.ui_pollingTimeout_dSpBx.setSuffix(" seconds")
        self.ui_pollingTimeout_dSpBx.setValue(self.polling_timeout)
        self.ui_pollingTimeout_dSpBx.valueChanged.connect(
                lambda x: self.set_param(
                    'polling_timeout', x,
                    self.ui_pollingTimeout_dSpBx.setValue))
        ui_rows.append((var_lb, self.ui_pollingTimeout_dSpBx))
        self.ui_objs.extend([var_lb, self.ui_pollingTimeout_dSpBx])

        # clean_ready
        self.ui_cleanReady_chb = QtWidgets.QCheckBox(
            "Clean watch dir at ready")
        self.ui_cleanReady_chb.setChecked(self.clean_ready)
        self.ui_cleanReady_chb.stateChanged.connect(
                lambda state: setattr(self, 'clean_ready', state > 0))
        self.ui_objs.append(self.ui_cleanReady_chb)

        # --- Checkbox row ----------------------------------------------------
        # Save
        self.ui_saveProc_chb = QtWidgets.QCheckBox("Save processed image")
        self.ui_saveProc_chb.setChecked(self.save_proc)
        self.ui_saveProc_chb.stateChanged.connect(
                lambda state: setattr(self, 'save_proc', state > 0))
        self.ui_objs.append(self.ui_saveProc_chb)

        chb_hLayout = QtWidgets.QHBoxLayout()
        chb_hLayout.addStretch()
        chb_hLayout.addWidget(self.ui_saveProc_chb)
        ui_rows.append((None, chb_hLayout))

        return ui_rows

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_params(self):
        all_opts = super().get_params()
        excld_opts = ('work_dir', 'scan_name', 'nii_save_filename')
        sel_opts = {}
        for k, v in all_opts.items():
            if k[0] == '_' or k in excld_opts:
                continue
            if isinstance(v, Path):
                v = str(v)
            sel_opts[k] = v

        sel_opts['watch_dir'] = self.watch_dir

        return sel_opts

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        # Kill observer process
        if hasattr(self, '_observer') and self._observer.isAlive():
            self._observer.stop()
            self._observer.join()


# %% __main__ (test) ==========================================================
if __name__ == '__main__':
    if '__file__' not in locals():
        __file__ == 'this.py'

    # --- Test ---
    # test data
    test_dir = Path(__file__).parent.parent / 'tests'
    dcm_dir = test_dir / 'dicomXA30'
    dcm_dir1 = dcm_dir / \
        '20231108.Phantom.2023.11.08_09_12_32_STD_1.3.12.2.1107.5.99.3'
    dcm_files = sorted(list(dcm_dir1.glob('*.dcm')))

    watch_dir = test_dir / 'watch'
    if not watch_dir.is_dir():
        watch_dir.mkdir()
    else:
        for rmf in watch_dir.glob('*'):
            if rmf.is_dir():
                shutil.rmtree(rmf)
            else:
                rmf.unlink()

    work_dir = test_dir / 'work'
    if not work_dir.is_dir():
        work_dir.mkdir()

    # Create RtpWatch instance
    watch_file_pattern = r'.+\.dcm'
    rtp_watch = RtpWatch(watch_dir, watch_file_pattern)

    rtp_watch.save_proc = True  # save result
    rtp_watch.online_saving = True  # Onlline saving
    rtp_watch.save_delay = True
    rtp_watch.work_dir = work_dir

    # Start watching
    rtp_watch.ready_proc()

    # Test nifti: copy the test data volume-by-volume
    n_copy = 0
    for dcm_f in dcm_files:
        dst_f = watch_dir / dcm_f.relative_to(dcm_dir)
        if not dst_f.parent.is_dir():
            os.makedirs(dst_f.parent)

        shutil.copy(dcm_f, dst_f)
        n_copy += 1
        time.sleep(1)

        if (n_copy + 1) % 10 == 0:
            rtp_watch.end_reset()
            time.sleep(3)
            rtp_watch.ready_proc()

    # End and reset module
    rtp_watch.end_reset()

    # Clean up watch_dir
    for ff in watch_dir.glob('*'):
        if ff.is_dir():
            continue
        ff.unlink()

    watch_dir.rmdir()
