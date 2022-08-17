#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Watching new file creation for real-time processing

@author: mmisaki@laureateinstitute.org
"""


# %% import ==================================================================#
from pathlib import Path
import os
import sys
import time
import re
from datetime import datetime
import traceback

import numpy as np
import nibabel as nib
import pydicom

from watchdog.observers.polling import PollingObserverVFS
from watchdog.events import FileSystemEventHandler
from PyQt5 import QtWidgets, QtCore

from dicom2nifti.convert_dicom import dicom_array_to_nifti
from dicom2nifti.image_volume import load, ImageVolume
from dicom2nifti.image_reorientation import _reorient_4d, _reorient_3d

try:
    from .rtp_common import RTP
except Exception:
    from rtpspy.rtp_common import RTP


# %% class RTP_WATCH ==========================================================
class RTP_WATCH(RTP):
    """
    Watching new file creation, reading data from the file, and sending data to
    a next process.

    e.g.
    # Make an instance
    rtp_watch = RTP_WATCH(watch_dir='watching/path',
                          watch_file_pattern='nr_\d+.+\.BRIK')

    # Start wathing
    rtp_watch.start_watching()

    When the name of a new file in watch_dir matches watch_file_pattern,
    the file is read by nibabel.

    Refer also to the python watchdog package:
    https://pypi.org/project/watchdog/
    """

    # -------------------------------------------------------------------------
    class RTPFileHandler(FileSystemEventHandler):
        """ File handling class """

        def __init__(self, watch_file_pattern, data_proc, scan_onset=None):
            """
            Parameters
            ----------
            watch_file_pattern : str
                Regular expression to filter the file.
            data_proc : function
                Applied function to the file.
            scan_onset : RTP_SCANNONSET object, optional
                If scan_onset is not None, and the scan is not running
                (scan_onset.is_scan_on() is False), the file creation event is
                ignored. The default is None.
            """
            super().__init__()
            self.watch_file_pattern = watch_file_pattern
            self.data_proc = data_proc
            self.scan_onset = scan_onset

        def on_created(self, event):
            if event.is_directory:
                return

            if self.scan_onset is not None and \
                    not self.scan_onset.is_scan_on():
                return

            if re.search(self.watch_file_pattern, Path(event.src_path).name):
                self.data_proc(event.src_path)

    # -------------------------------------------------------------------------
    class RTP_Observer(PollingObserverVFS):
        """ Observer class with a custom thread function.
        PollingObserverVFS is used to work wiht a network-shared drive. If the
        number of file in the watching directory is large, this observer takes
        long time to find a new one. So the watching directory should be
        cleaned at every scan run.
        """

        def __init__(self, stat=os.stat, watch_file_pattern=None,
                     polling_interval=0.001, verb=False, std_out=sys.stdout):
            """
            Parameters
            ----------
            See watchdog document.
            https://pythonhosted.org/watchdog/api.html#watchdog.observers.polling.PollingObserver

            verb : bool
                Print process log.
            std_out : output stream
                Log output.
            """
            super().__init__(stat, self.listdir, polling_interval)

            self._watch_file_pattern = watch_file_pattern
            self._verb = verb
            self._std_out = std_out

        def listdir(self, root):
            if self._watch_file_pattern is not None:
                paths = [pp.name for pp in Path(root).glob('*')
                         if re.search(self._watch_file_pattern, pp.name)]
            else:
                paths = [pp.name for pp in Path(root).glob('*')]

            return paths

        def logmsg(self, msg, ret_str=False):
            # Add datetime and class name
            dtstr = datetime.now().isoformat()
            msg = f"{dtstr}:[{self.__class__.__name__}]: {msg}"
            if ret_str:
                return msg

            print(msg, file=self._std_out)
            if hasattr(self._std_out, 'flush'):
                self._std_out.flush()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, watch_dir='', watch_file_pattern=r'nr_\d+.+\.BRIK',
                 polling_interval=0.001, siemens_mosaic_dicom=False,
                 clean_rt_src=False, clean_warning=True, scan_onset=None,
                 **kwargs):
        """
        Parameters
        ----------
        watch_dir : str
            Watching directory.
        watch_file_pattern : str
            Regular expression for watching filename.
        polling_interval : float, optional
            Interval to check new file creation (second). The default is
            0.001.
        siemens_mosaic_dicom : bool, optional
            The watching file is Siemens mosaic dicom. The default is False.
        clean_rt_src : bool, optional
            Delete real-time created soouce files. At starting the wachdog
            thread, files matching watch_file_pattern in watch_dir are deleted.
            The default is False.
        clean_warning : bool, optional,
            Show warning when cleaning the watch_dir. The default is True.
        scan_onset : RTP_SCANONSET object, optional
            If scan_onset is not None and the scan is not running
            (scan_onset.is_scan_on() is False), the file creation event is
            ignored. The default is None.
        """

        super().__init__(**kwargs)  # call __init__() in RTP base class

        # Set parameters
        self.watch_dir = watch_dir
        self.watch_file_pattern = watch_file_pattern
        self.polling_interval = polling_interval
        self.siemens_mosaic_dicom = siemens_mosaic_dicom
        self.scan_onset = scan_onset

        self.last_proc_f = ''  # Last processed filename
        self.done_proc = -1  # Number of the processed volume
        self.clean_rt_src = clean_rt_src
        self.clean_warning = True  # Show warning when cleaning the watch_dir.
        self.imgType = 'AFNI BRIK'
        self.NSlices = 0
        self.dicom_list = {}
        self.do_proc = None

        # if watch_dir does not exist, set _proc_ready False.
        if not Path(self.watch_dir).is_dir():
            self._proc_ready = False

        self.scan_name = None  # For LIBR, used at saving a physio signal file.

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ready_proc(self):
        self.dicom_list = {}
        self._proc_ready  = True

        if self.imgType == 'GE DICOM':
            if self.NSlices < 1:
                self.errmsg("Nr. slices is not set.")
                self._proc_ready = False

        if not Path(self.watch_dir).is_dir():
            self.errmsg(f"watch_dir ({self.watch_dir}) is not set properly.")
            self._proc_ready = False

        if self.imgType == 'AFNI BRIK':
            self.do_proc = self.do_proc_volImg
        elif self.imgType == 'NIfTI':
            self.do_proc = self.do_proc_volImg
        elif self.imgType == 'GE DICOM':
            self.do_proc = self.do_proc_GEDICOM
        elif self.imgType == 'Siemense Mosaic':
            self.do_proc = self.do_proc_SiemensMosaic

        if self.do_proc is None:
            self.errmsg(f"Process for {self.imgType} is not defined.")
            self._proc_ready = False

        if self.clean_rt_src:
            self.clean_files()

        if self.next_proc:
            self._proc_ready &= self.next_proc.ready_proc()

        if self._proc_ready:
            self.start_watching()

        return self._proc_ready

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def do_proc_volImg(self, fpath):
        """
        Parameters
        ----------
        fpath : str
            File path found by a watchdog observer.
        """

        try:
            # Avoid processing the same file multiple times
            if self.last_proc_f == fpath:
                return

            self.last_proc_f = fpath

            # Increment the number of received volume
            self.vol_num += 1

            if self.proc_start_idx < 0:
                self.proc_start_idx = 0

            fpath = Path(fpath)
            fsize = fpath.stat().st_size
            time.sleep(0.001)
            while True:
                # Wait for completing file creation
                if fpath.stat().st_size != fsize:
                    fsize = fpath.stat().st_size
                    time.sleep(0.001)
                    continue

                try:
                    # Load file
                    load_img = nib.load(fpath)

                    # get_fdata will fail if the file is incomplete.
                    load_img.get_fdata()
                    break
                except Exception:
                    continue

            # Create Nifti1Image
            dataV = np.asanyarray(load_img.dataobj)
            if dataV.ndim > 3:
                dataV = np.squeeze(dataV)

            # Set save_filename
            if fpath.suffix == '.gz':
                save_filename = Path(fpath.stem).stem
            else:
                save_filename = fpath.stem
            save_filename = re.sub(r'\+orig.*', '', save_filename) + '.nii.gz'
            fmri_img = nib.Nifti1Image(load_img.dataobj, load_img.affine,
                                       header=load_img.header)
            fmri_img.set_filename(save_filename)

            if hasattr(self, 'scan_name') and self.scan_name is None:
                # For LIBR: scan_name is used at saving a physio signal.
                ma = re.search(r'.+scan_\d+__\d+', fpath.stem)
                if ma:
                    self.scan_name = ma.group()

            # Record process time
            self.proc_time.append(time.time())
            proc_delay = self.proc_time[-1] - fpath.stat().st_ctime
            if self.save_delay:
                self.proc_delay.append(proc_delay)

            # log
            if self._verb:
                f = fpath.name
                if len(self.proc_time) > 1:
                    t_interval = self.proc_time[-1] - self.proc_time[-2]
                else:
                    t_interval = -1
                msg = f'#{self.vol_num}, Read {f}'
                msg += f' (took {proc_delay:.4f}s,'
                msg += f' interval {t_interval:.4f}s).'
                self.logmsg(msg)

            if self.next_proc:
                # Keep the current processed data
                self.proc_data = np.asanyarray(fmri_img.dataobj)
                save_name = fmri_img.get_filename()

                # Run the next process
                self.next_proc.do_proc(fmri_img, vol_idx=self.vol_num,
                                       pre_proc_time=self.proc_time[-1])

            # Record the number of the processed volume
            self.done_proc = self.vol_num

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
            errmsg = (f'{exc_type}, {exc_tb.tb_frame.f_code.co_filename}' +
                      f':{exc_tb.tb_lineno}')
            self.errmsg(errmsg, no_pop=True)
            traceback.print_exc(file=self._err_out)
            print(self.saved_data.shape)
            print(fmri_img.get_fdata().shape)

            raise

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def do_proc_GEDICOM(self, fpath):
        """
        Parameters
        ----------
        fpath : str
            File path found by a watchdog observer.
        """

        try:
            # Avoid processing the same file multiple times
            if self.last_proc_f == fpath:
                return

            self.last_proc_f = fpath

            fpath = Path(fpath)
            fnum = int(re.search('\d+$', fpath.name).group())

            fsize = fpath.stat().st_size
            time.sleep(0.001)
            while True:
                # Wait for completing file creation
                if fpath.stat().st_size != fsize:
                    fsize = fpath.stat().st_size
                    time.sleep(0.001)
                    continue

                try:
                    dcm = pydicom.read_file(
                        fpath, defer_size="1 KB", stop_before_pixels=False,
                        force=False)
                    break
                except Exception:
                    continue

            self.dicom_list[fnum] = dcm

            if fnum % self.NSlices != 0:
                return

            # Increment volume index (0 base)
            self.vol_num += 1
            vol_idxs = np.arange(self.vol_num * self.NSlices + 1,
                                 (self.vol_num+1) * self.NSlices + 1,
                                 dtype=int)

            dcm_list = []
            for nn in vol_idxs:
                while nn not in self.dicom_list:
                    # Wait for another thread to read the data
                    time.sleep(0.001)

                dcm_list.append(self.dicom_list[nn])

            if self.proc_start_idx < 0:
                self.proc_start_idx = 0

            # Create Nifti1Image
            ret = dicom_array_to_nifti(dcm_list, None,
                                       reorient_nifti=False)
            load_img = self.reorient_image(ret['NII'], None)
            dataV = np.asanyarray(load_img.dataobj)
            if dataV.ndim > 3:
                dataV = np.squeeze(dataV)

            # Removed processed data
            for nn in vol_idxs:
                del self.dicom_list[nn]

            # Set save_filename
            if fpath.suffix == '.gz':
                save_filename = Path(fpath.stem).stem
            else:
                save_filename = '_'.join(
                    [dcm.PatientID, fpath.parts[-2], dcm.ProtocolName,
                     dcm.AcquisitionDate, dcm.AcquisitionTime,
                     f"{self.vol_num:04d}"])

            save_filename = re.sub(r'\+orig.*', '', save_filename) + '.nii.gz'
            fmri_img = nib.Nifti1Image(load_img.dataobj, load_img.affine,
                                       header=load_img.header)
            fmri_img.set_filename(save_filename)

            if hasattr(self, 'scan_name') and self.scan_name is None:
                # For LIBR: scan_name is used at saving a physio signal.
                ma = re.search(r'.+scan_\d+__\d+', fpath.stem)
                if ma:
                    self.scan_name = ma.group()

            # Record process time
            self.proc_time.append(time.time())
            proc_delay = self.proc_time[-1] - fpath.stat().st_ctime
            if self.save_delay:
                self.proc_delay.append(proc_delay)

            # log
            if self._verb:
                f = fpath.name
                if len(self.proc_time) > 1:
                    t_interval = self.proc_time[-1] - self.proc_time[-2]
                else:
                    t_interval = -1
                msg = f'#{self.vol_num}, Read {f}'
                msg += f' (took {proc_delay:.4f}s,'
                msg += f' interval {t_interval:.4f}s).'
                self.logmsg(msg)

            if self.next_proc:
                # Keep the current processed data
                self.proc_data = np.asanyarray(fmri_img.dataobj)
                save_name = fmri_img.get_filename()

                # Run the next process
                self.next_proc.do_proc(fmri_img, vol_idx=self.vol_num,
                                       pre_proc_time=self.proc_time[-1])

            # Record the number of the processed volume
            self.done_proc = self.vol_num

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
            errmsg = (f'{exc_type}, {exc_tb.tb_frame.f_code.co_filename}' +
                      f':{exc_tb.tb_lineno}')
            self.errmsg(errmsg, no_pop=True)
            traceback.print_exc(file=self._err_out)

            raise

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def do_proc_SiemensMosaic(self, fpath):
        """
        Parameters
        ----------
        fpath : str
            File path found by a watchdog observer.
        """

        try:
            # Avoid processing the same file multiple times
            if self.last_proc_f == fpath:
                return

            self.last_proc_f = fpath

            fpath = Path(fpath)
            while True:
                fsize = fpath.stat().st_size
                time.sleep(0.001)

                # Wait for completing file creation
                if fpath.stat().st_size != fsize:
                    fsize = fpath.stat().st_size
                    time.sleep(0.001)
                    continue
                try:
                    dcm = pydicom.read_file(
                        fpath, defer_size="1 KB", stop_before_pixels=False,
                        force=False)
                    break
                except Exception:
                    continue

            # Increment the number of received volume
            self.vol_num += 1

            if self.proc_start_idx < 0:
                self.proc_start_idx = 0

            # Create Nifti1Image
                ret = dicom_array_to_nifti([dcm], None,
                                           reorient_nifti=False)
            load_img = self.reorient_image(ret['NII'], None)

            # Create Nifti1Image
            dataV = np.asanyarray(load_img.dataobj)
            if dataV.ndim > 3:
                dataV = np.squeeze(dataV)

            # Set save_filename
            if fpath.suffix == '.gz':
                save_filename = Path(fpath.stem).stem
            else:
                save_filename = '_'.join(
                    [dcm.PatientID, dcm.ProtocolName, dcm.AcquisitionDate,
                     dcm.AcquisitionTime])

            save_filename = re.sub(r'\+orig.*', '', save_filename) + '.nii.gz'
            fmri_img = nib.Nifti1Image(load_img.dataobj, load_img.affine,
                                       header=load_img.header)
            fmri_img.set_filename(save_filename)

            if hasattr(self, 'scan_name') and self.scan_name is None:
                # For LIBR: scan_name is used at saving a physio signal.
                ma = re.search(r'.+scan_\d+__\d+', fpath.stem)
                if ma:
                    self.scan_name = ma.group()

            # Record process time
            self.proc_time.append(time.time())
            proc_delay = self.proc_time[-1] - fpath.stat().st_ctime
            if self.save_delay:
                self.proc_delay.append(proc_delay)

            # log
            if self._verb:
                f = fpath.name
                if len(self.proc_time) > 1:
                    t_interval = self.proc_time[-1] - self.proc_time[-2]
                else:
                    t_interval = -1
                msg = f'#{self.vol_num}, Read {f}'
                msg += f' (took {proc_delay:.4f}s,'
                msg += f' interval {t_interval:.4f}s).'
                self.logmsg(msg)

            if self.next_proc:
                # Keep the current processed data
                self.proc_data = np.asanyarray(fmri_img.dataobj)
                save_name = fmri_img.get_filename()

                # Run the next process
                self.next_proc.do_proc(fmri_img, vol_idx=self.vol_num,
                                       pre_proc_time=self.proc_time[-1])

            # Record the number of the processed volume
            self.done_proc = self.vol_num

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
            errmsg = (f'{exc_type}, {exc_tb.tb_frame.f_code.co_filename}' +
                      f':{exc_tb.tb_lineno}')
            self.errmsg(errmsg, no_pop=True)
            traceback.print_exc(file=self._err_out)
            print(self.saved_data.shape)
            print(fmri_img.get_fdata().shape)

            raise

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end_reset(self):
        """ End process and reset process parameters. """

        if self.verb:
            self.logmsg(f"Reset {self.__class__.__name__} module.")

        # Wait for process in watch thread finish.
        self.stop_watching()
        self.scan_name = None

        super().end_reset()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def start_watching(self):
        """
        Start watchdog observer process. The oberver will run on another
        thread.
        """

        # set event_handler
        self.event_handler = \
            RTP_WATCH.RTPFileHandler(self.watch_file_pattern, self.do_proc,
                                     scan_onset=self.scan_onset)

        # self.observer = Observer(timeout=0.001)
        self.observer = RTP_WATCH.RTP_Observer(
                stat=os.stat, watch_file_pattern=self.watch_file_pattern,
                polling_interval=self.polling_interval,
                verb=self._verb, std_out=self._std_out)

        self.observer.schedule(self.event_handler, self.watch_dir)

        self.observer.start()

        if self._verb:
            self.logmsg(f"Start watchdog observer on {self.watch_dir}.")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def stop_watching(self):
        if hasattr(self, 'observer'):
            if self.observer.is_alive():
                self.observer.stop()
                self.observer.join()
            del self.observer

        if self._verb:
            self.logmsg("Stop watchdog observer.")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def clean_files(self, *args, warning=None):
        if not Path(self.watch_dir).is_dir():
            return

        fs = [ff for ff in Path(self.watch_dir).glob('*')
              if re.search(self.watch_file_pattern, str(ff))]

        if len(fs) > 0:
            if warning is None:
                warning = self.clean_warning

            if warning and self.main_win is not None:
                # Warning dialog
                msgBox = QtWidgets.QMessageBox()
                msgBox.setIcon(QtWidgets.QMessageBox.Question)
                msgBox.setText("Delete {} watched files?".format(len(fs)))
                msgBox.setDetailedText('\n'.join([str(ff) for ff in fs]))
                msgBox.setWindowTitle("Delete temporary files")
                msgBox.setStandardButtons(QtWidgets.QMessageBox.Yes |
                                          QtWidgets.QMessageBox.No)
                msgBox.setDefaultButton(QtWidgets.QMessageBox.Yes)
                ret = msgBox.exec()
                if ret != QtWidgets.QMessageBox.Yes:
                    return

            # If pattern is BRIK file (.BRIK|.HEAD), remove paried files
            if re.search('BRIK', self.watch_file_pattern):
                pat = re.sub('BRIK.*', 'HEAD', self.watch_file_pattern)
                fs += [ff for ff in os.listdir(self.watch_dir)
                       if re.search(pat, ff)]
            elif re.search('HEAD', self.watch_file_pattern):
                pat = re.sub('HEAD', r'BRIK.*',
                             self.watch_file_pattern)
                fs += [ff for ff in os.listdir(self.watch_dir)
                       if re.search(pat, ff)]

            if self._verb:
                self.logmsg(
                    f"Remove {len(fs)} temporary files in {self.watch_dir}")
            for fbase in fs:
                (Path(self.watch_dir) / fbase).unlink()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def reorient_image(self, input_image, output_image=None):
        """
        Copied from dicom2nifti/image_reorientation.py with reviseing to skip
        savving the file when output_image is None.
        https://github.com/icometrix/dicom2nifti
        """

        # Use the imageVolume module to find which coordinate corresponds to
        # each plane and get the image data in RAS orientation print
        # 'Reading nifti'
        if isinstance(input_image, nib.Nifti1Image):
            image = ImageVolume(input_image)
        else:
            image = load(input_image)

        # 4d have a different conversion to 3d
        # print 'Reorganizing data'
        if image.nifti_data.squeeze().ndim == 4:
            new_image = _reorient_4d(image)
        elif image.nifti_data.squeeze().ndim == 3 or \
                image.nifti_data.ndim == 3 or \
                image.nifti_data.squeeze().ndim == 2:
            new_image = _reorient_3d(image)
        else:
            raise Exception('Only 3d and 4d images are supported')

        # print 'Recreating affine'
        affine = image.nifti.affine

        new_affine = np.eye(4)
        new_affine[:, 0] = affine[:, image.sagittal_orientation.normal_component]
        new_affine[:, 1] = affine[:, image.coronal_orientation.normal_component]
        new_affine[:, 2] = affine[:, image.axial_orientation.normal_component]
        point = [0, 0, 0, 1]

        # If the orientation of coordinates is inverted, then the origin of the "new" image
        # would correspond to the last voxel of the original image
        # First we need to find which point is the origin point in image coordinates
        # and then transform it in world coordinates
        if not image.axial_orientation.x_inverted:
            new_affine[:, 0] = - new_affine[:, 0]
            point[image.sagittal_orientation.normal_component] = \
                image.dimensions[image.sagittal_orientation.normal_component] - 1
            # new_affine[0, 3] = - new_affine[0, 3]
        if image.axial_orientation.y_inverted:
            new_affine[:, 1] = - new_affine[:, 1]
            point[image.coronal_orientation.normal_component] = \
                image.dimensions[image.coronal_orientation.normal_component] - 1
            # new_affine[1, 3] = - new_affine[1, 3]
        if image.coronal_orientation.y_inverted:
            new_affine[:, 2] = - new_affine[:, 2]
            point[image.axial_orientation.normal_component] = \
                image.dimensions[image.axial_orientation.normal_component] - 1
            # new_affine[2, 3] = - new_affine[2, 3]

        new_affine[:, 3] = np.dot(affine, point)

        # DONE: Needs to update new_affine, so that there is no translation
        # difference between the original and created image
        # (now there is 1-2 voxels translation)
        # print 'Creating new nifti image'
        if new_image.ndim > 3:  # do not squeeze single slice data
            new_image = new_image.squeeze()
        output = nib.nifti1.Nifti1Image(new_image, new_affine)
        output.header.set_slope_inter(1, 0)
        output.header.set_xyzt_units(2)  # set units for xyz (leave t as unknown)
        if output_image is not None:
            output.to_filename(output_image)
        return output

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

        elif attr == 'polling_interval' and reset_fn is None:
            if hasattr(self, 'ui_pollingIntv_dSpBx'):
                self.ui_pollingIntv_dSpBx.setValue(val)

        elif attr == 'watch_file_pattern':
            if len(val) == 0:
                if reset_fn:
                    reset_fn(str(self.watch_file_pattern))
                return
            if reset_fn is None:
                if hasattr(self, 'ui_watchPat_lnEd'):
                    self.ui_watchPat_lnEd.setText(str(val))

        elif attr == 'save_proc':
            if hasattr(self, 'ui_saveProc_chb'):
                self.ui_saveProc_chb.setChecked(val)

        elif attr == '_verb':
            if hasattr(self, 'ui_verb_chb'):
                self.ui_verb_chb.setChecked(val)

        elif attr == 'scan_onset':
            pass

        elif attr == 'clean_rt_src':
            if reset_fn is None:
                if hasattr(self, 'ui_cleanRtSrc_chb'):
                    self.ui_cleanRtSrc_chb.setChecked(val)
                
        elif attr == 'clean_warning':
            if reset_fn is None:
                if hasattr(self, 'ui_cleanWarning_chb'):
                    self.ui_cleanWarning_chb.setChecked(val)

        elif attr == 'imgType':
            if reset_fn is None and hasattr(self, 'ui_imgType_grpBox'):
                rdb = getattr(self, f"ui_imgType{val.replace(' ', '')}_rdb")
                if not rdb.isChecked():
                    rdb.setChecked(True)

            if val == 'AFNI BRIK':
                self.set_param('watch_file_pattern', '.*nr_\d+.*\.BRIK.*')
                self.do_proc = self.do_proc_volImg
            elif val == 'NIfTI':
                self.set_param('watch_file_pattern', '.+\.nii.*')
                self.do_proc = self.do_proc_volImg
            elif val == 'GE DICOM':
                self.set_param('watch_file_pattern', 'i\d+.*')
                self.do_proc = self.do_proc_GEDICOM
            elif val == 'Siemense Mosaic':
                self.set_param('watch_file_pattern', '.+\.dcm')
                self.do_proc = self.do_proc_SiemensMosaic

        if attr == 'NSlices':
            if hasattr(self, 'ui_NSlices_spBx') and reset_fn is None:
                self.ui_NSlices_spBx.setValue(val)

        elif reset_fn is None:
            # Ignore an unrecognized parameter
            if not hasattr(self, attr):
                self.errmsg(f"{attr} is unrecognized parameter.", no_pop=True)
                return

        # -- Set value--
        setattr(self, attr, val)
        if echo and self._verb:
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
        if Path(self.watch_dir).is_dir():
            self.ui_wdir_lnEd.setText(str(self.watch_dir))
        ui_rows.append((var_lb, self.ui_wdir_lnEd))
        self.ui_objs.extend([var_lb, self.ui_wdir_lnEd])

        # polling_interval
        var_lb = QtWidgets.QLabel("Polling interval :")
        self.ui_pollingIntv_dSpBx = QtWidgets.QDoubleSpinBox()
        self.ui_pollingIntv_dSpBx.setMinimum(0.0001)
        self.ui_pollingIntv_dSpBx.setSingleStep(0.001)
        self.ui_pollingIntv_dSpBx.setDecimals(4)
        self.ui_pollingIntv_dSpBx.setSuffix(" seconds")
        self.ui_pollingIntv_dSpBx.setValue(self.polling_interval)
        self.ui_pollingIntv_dSpBx.valueChanged.connect(
                lambda x: self.set_param('polling_interval', x,
                                         self.ui_pollingIntv_dSpBx.setValue))
        ui_rows.append((var_lb, self.ui_pollingIntv_dSpBx))
        self.ui_objs.extend([var_lb, self.ui_pollingIntv_dSpBx])

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

        # File type check box
        self.ui_imgType_grpBox = QtWidgets.QGroupBox("Image type :")
        imgType_hbox = QtWidgets.QHBoxLayout()

        # AFNI BRIK
        self.ui_imgTypeAFNIBRIK_rdb = QtWidgets.QRadioButton("AFNI BRIK")
        if self.imgType == 'AFNI BRIK':
            self.ui_imgTypeAFNIBRIK_rdb.setChecked(1)
        imgType_hbox.addWidget(self.ui_imgTypeAFNIBRIK_rdb)
        self.ui_imgTypeAFNIBRIK_rdb.toggled.connect(
             lambda: self.set_param('imgType', 'AFNI BRIK',
                                    self.ui_imgTypeAFNIBRIK_rdb.setChecked))

        # NIfTI
        self.ui_imgTypeNIfTI_rdb = QtWidgets.QRadioButton("NIfTI")
        if self.imgType == 'NIfTI':
            self.ui_imgTypeNIfTI_rdb.setChecked(1)
        imgType_hbox.addWidget(self.ui_imgTypeNIfTI_rdb)
        self.ui_imgTypeNIfTI_rdb.toggled.connect(
             lambda: self.set_param('imgType', 'NIfTI',
                                    self.ui_imgTypeNIfTI_rdb.setChecked))

        # GE DICOM
        self.ui_imgTypeGEDICOM_rdb = QtWidgets.QRadioButton("GE DICOM")
        if self.imgType == 'GE DICOM':
            self.ui_imgTypeGEDICOM_rdb.setChecked(1)
        imgType_hbox.addWidget(self.ui_imgTypeGEDICOM_rdb)
        self.ui_imgTypeGEDICOM_rdb.toggled.connect(
             lambda: self.set_param('imgType', 'GE DICOM',
                                    self.ui_imgTypeGEDICOM_rdb.setChecked))

        # Siemens mosaic DIOCM siemens_mosaic_dicom check
        self.ui_imgTypeSiemenseMosaic_rdb = \
            QtWidgets.QRadioButton("Siemense Mosaic")
        if self.imgType == 'Siemense Mosaic':
            self.ui_imgTypeSiemenseMosaic_rdb.setChecked(1)
        imgType_hbox.addWidget(self.ui_imgTypeSiemenseMosaic_rdb)
        self.ui_imgTypeSiemenseMosaic_rdb.toggled.connect(
             lambda: self.set_param(
                 'imgType', 'Siemense Mosaic',
                 self.ui_imgTypeSiemenseMosaic_rdb.setChecked))

        self.ui_imgType_grpBox.setLayout(imgType_hbox)
        ui_rows.append((None, self.ui_imgType_grpBox))
        self.ui_objs.extend([self.ui_imgType_grpBox])

        # Nr. slices for GE DICOM
        var_lb = QtWidgets.QLabel('Nr. slices')
        var_lb.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        self.ui_NSlices_spBx = QtWidgets.QSpinBox()
        self.ui_NSlices_spBx.setMinimum(0)
        if hasattr(self, 'NSlices') and self.NSlices is not None:
            self.ui_NSlices_spBx.setValue(self.NSlices)
        self.ui_NSlices_spBx.valueChanged.connect(
                    lambda x: self.set_param('NSlices', x,
                                             self.ui_NSlices_spBx.setValue))
        ui_rows.append((var_lb, self.ui_NSlices_spBx))
        self.ui_objs.extend([self.ui_NSlices_spBx])

        # clean_rt_src check
        self.ui_cleanRtSrc_chb = QtWidgets.QCheckBox(
            "Clean real-time MRI source")
        self.ui_cleanRtSrc_chb.setChecked(self.clean_rt_src)
        self.ui_cleanRtSrc_chb.stateChanged.connect(
             lambda: self.set_param('clean_rt_src',
                                    self.ui_cleanRtSrc_chb.isChecked(),
                                    self.ui_cleanRtSrc_chb.setChecked))
        ui_rows.append((None, self.ui_cleanRtSrc_chb))
        self.ui_objs.extend([self.ui_cleanRtSrc_chb])

        # clean_warning check
        self.ui_cleanWarning_chb = QtWidgets.QCheckBox(
            "Warn at cleaning real-time MRI source")
        self.ui_cleanWarning_chb.setChecked(self.clean_warning)
        self.ui_cleanWarning_chb.stateChanged.connect(
             lambda: self.set_param('clean_warning',
                                    self.ui_cleanWarning_chb.isChecked(),
                                    self.ui_cleanWarning_chb.setChecked))
        ui_rows.append((None, self.ui_cleanWarning_chb))
        self.ui_objs.extend([self.ui_cleanWarning_chb])

        # Clean_files
        self.ui_cleanFilest_btn = QtWidgets.QPushButton()
        self.ui_cleanFilest_btn.setText('Clean up existing watch files')
        self.ui_cleanFilest_btn.clicked.connect(self.clean_files)
        ui_rows.append((None, self.ui_cleanFilest_btn))
        self.ui_objs.append(self.ui_cleanFilest_btn)

        # --- Checkbox row ----------------------------------------------------
        # Save
        self.ui_saveProc_chb = QtWidgets.QCheckBox("Save processed image")
        self.ui_saveProc_chb.setChecked(self.save_proc)
        self.ui_saveProc_chb.stateChanged.connect(
                lambda state: setattr(self, 'save_proc', state > 0))
        self.ui_objs.append(self.ui_saveProc_chb)

        # verb
        self.ui_verb_chb = QtWidgets.QCheckBox("Verbose logging")
        self.ui_verb_chb.setChecked(self.verb)
        self.ui_verb_chb.stateChanged.connect(
                lambda state: setattr(self, 'verb', state > 0))
        self.ui_objs.append(self.ui_verb_chb)

        chb_hLayout = QtWidgets.QHBoxLayout()
        chb_hLayout.addStretch()
        chb_hLayout.addWidget(self.ui_saveProc_chb)
        chb_hLayout.addWidget(self.ui_verb_chb)
        ui_rows.append((None, chb_hLayout))

        return ui_rows

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_params(self):
        all_opts = super().get_params()
        excld_opts = ('work_dir', 'scan_name', 'scan_onset'
                      'done_proc', 'last_proc_f', 'clean_warning',
                      'dicom_list')
        sel_opts = {}
        for k, v in all_opts.items():
            if k in excld_opts:
                continue
            if isinstance(v, Path):
                v = str(v)
            sel_opts[k] = v

        sel_opts['watch_dir'] = self.watch_dir

        return sel_opts

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        # Kill observer process
        if hasattr(self, 'observer') and self.observer.isAlive():
            self.observer.stop()
            self.observer.join()


# %% __main__ (test) ==========================================================
if __name__ == '__main__':
    import shutil

    # --- Test ---
    # test data
    test_dir = Path(__file__).resolve().parent.parent / 'test'

    testdata_f = test_dir / 'func_epi.nii.gz' # NIfTI file
    assert testdata_f.is_file()

    testdata_d = test_dir / 'Siemens_mosaic_dicom'  # Siemens mosaic dicom
    assert testdata_d.is_dir()

    watch_dir = test_dir / 'watch'
    watch_dir = Path('/media/cephfs/labs/jbodurka/mmisaki/watch')
    if not watch_dir.is_dir():
        watch_dir.mkdir()
    else:
        for rmf in watch_dir.glob('*'):
            rmf.unlink()

    work_dir = test_dir / 'work'
    if not work_dir.is_dir():
        work_dir.mkdir()

    # Create RTP_WATCH instance
    watch_file_pattern = r'nr_\d+.+\.nii'
    watch_file_pattern = r'\d+_\d+_\d+\.dcm'
    siemens_mosaic_dicom = True
    rtp_watch = RTP_WATCH(watch_dir, watch_file_pattern,
                          siemens_mosaic_dicom=siemens_mosaic_dicom)
    rtp_watch.verb = True
    rtp_watch.save_proc = True  # save result
    rtp_watch.online_saving = True  # Onlline saving
    rtp_watch.save_delay = True
    rtp_watch.work_dir = work_dir

    # Start watching
    rtp_watch.ready_proc()

    # Test nifti: copy the test data volume-by-volume
    if siemens_mosaic_dicom:
        srs_fs = sorted([pp.name for pp in testdata_d.glob('*')
                         if pp.is_file() and
                         re.search(watch_file_pattern, pp.name)])
        N_vols = len(srs_fs)
    else:
        img = nib.load(testdata_f)
        fmri_data = np.asanyarray(img.dataobj)
        N_vols = img.shape[-1]

    for ii in range(N_vols):
        if siemens_mosaic_dicom:
            src_f = testdata_d / srs_fs[ii]
            save_f = watch_dir / srs_fs[ii]
            shutil.copy(src_f, save_f)
        else:
            save_f = watch_dir / f"test_nr_{ii:04d}.nii"
            nib.save(nib.Nifti1Image(fmri_data[:, :, :, ii],
                                     affine=img.affine),
                     save_f)
        time.sleep(1)

    # End and reset module
    rtp_watch.end_reset()

    # Clean up watch_dir
    for ff in watch_dir.glob('*'):
        if ff.is_dir():
            continue
        ff.unlink()

    watch_dir.rmdir()
