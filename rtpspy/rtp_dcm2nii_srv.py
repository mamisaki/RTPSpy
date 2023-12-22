#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert the exported DICOM files in 'watch_dir' to NIfTI files in the working
directory under 'work_root' and move the DICOM files to the working directory.

mmisaki@laureateinstitute.org
"""

# %% import ===================================================================
import argparse
from pathlib import Path
import logging
import sys
import traceback
from datetime import datetime
import re
from threading import Lock, Thread
from multiprocessing import Process
import time
import os
import socketserver
import socket

from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
import pydicom

from dicom_converter import DicomConverter

if '__file__' not in locals():
    __file__ = 'this.py'


# %% class RtpDcm2NiiServer ===================================================
class RtpDcm2NiiServer:
    """A server process that converts DICOM to NIfTI in real time or on demand.
    Procedures:
    - Monitor a directory (self.dcm_dir) to which DICOM files are exported in
      real time.
    - When a new study and/or series is created, create a study directory
      under 'self.work_root' to store the converted files and the original
      DICOM files.
    - Wait for a scan to complete, and when the scan is complete, copy the
      DICOM files to the study directory and convert them to NIfTI. The
      completion of the scan is determined by the NumberofTemporalPositions
      field in the DICOM header, or no new file is created for
      'self.series_timeout' seconds.
    - The conversion process can also be triggered by a remote request issued
      by a real-time processing process if a scan is aborted before the
      NumberofTemporalPositions. A TCPServer thread in the class receives such
      a request via a network socket at the 'rpc_port'.
    """
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, watch_dir, work_root, watch_file_pattern=r'.+\.dcm',
                 study_prefix='P', series_timeout=60, rpc_port=63210,
                 make_brik=False, polling_observer=False, **kwargs):
        """
        Parameters
        ----------
        watch_dir : Path or str
            Directory to which DICOM images are exported in real time.
            Watchdog monitors the creation of files in this directory
            recursively, filtering the filename with watch_file_pattern
            (regular expression).
        work_root : Path or str
            Directory where a study directory is created to store the
            converted files and the original DICOM files. The study directory
            name is created as '{self.study_prefix}_{%Y%m%d%H%M%S}'.
        watch_file_pattern : str (regular expression)
            Regular expression to filter the filename monitored by a watchdog.
            The default is r'.+\\.dcm'.
        study_prefix : str
            Prefix for a study directory name. The study directory name will
            be created as '{self.study_prefix}_{%Y%m%d%H%M%S}'.
            The default is 'P'.
        series_timeout : float
            Timeout period to consider as end of series to start conversion.
            The default is 60.
        rpc_port : int
            Port number to receive a remote request via a network socket.
            The default is 63210.
        make_brik : bool
            Flag to create both BRIK and NIfTI files during DICOM conversion.
        polling_observer : bool, optional
            Flag to use a PollingObserver if the dcm_dir is not on a local
            file system. The default is False.
        """
        self._logger = logging.getLogger('RtpDcm2NiiServer')

        # Initialize parameters
        self.watch_dir = watch_dir
        self.work_root = work_root
        self.watch_file_pattern = watch_file_pattern
        self.study_prefix = study_prefix
        self.series_timeout = series_timeout
        self.rpc_port = rpc_port
        self.make_brik = make_brik
        self.polling_observer = polling_observer

        self._dcmread_timeout = 3
        self._polling_timeout = 3

        # Prepare a DicomConverter
        self.dcm_converter = DicomConverter(study_prefix=study_prefix)

        # Intialize session parameters
        self._current_study = -1
        self._current_series = -1
        self._isRun_series = False
        self._last_dicom_time = None
        self._last_dicom_header = None
        self._last_proc_f = None
        self._study_dir = None

        # Parameters to set the physio recording length
        self._TR = None
        self._NVol = 0
        self._physio_prefix = './{}_scan.1D'
        self._process_lock = Lock()
        self._cancel = False
        self._end_complete = False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run(self):
        """main loop"""
        self._logger.info(
            '\n' + '#' * 80 + '\n' +
            f'#==== Start RtpDcm2NiiServer {time.ctime()} ====='
            )

        # Start watchdog
        self.start_watching(callback=self.do_proc)

        # Strat RPC server
        self.start_RPC_server()

        # Application loop
        while not self._cancel:
            try:
                # Check if the series ends
                if self._TR is not None:
                    time_out = max(self.series_timeout, self._TR*2)
                else:
                    time_out = self.series_timeout

                if self._isRun_series and self._last_dicom_header:
                    # Check if the series ends
                    ser_end = False
                    if hasattr(self._last_dicom_header,
                               'NumberOfTemporalPositions'):
                        nt = int(
                            self._last_dicom_header.NumberOfTemporalPositions)
                        if self._NVol == nt:
                            ser_end = True

                    if not ser_end and self._last_dicom_time is not None and \
                            time.time() - self._last_dicom_time > time_out:
                        ser_end = True
                        # No new image for time_out seconds.
                        self._logger.info(
                            f"No new DICOM file for {time_out}s. Close series."
                            )

                    if ser_end:
                        self.end_series()

                time.sleep(1)

            except KeyboardInterrupt:
                break

            except Exception:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                errstr = ''.join(
                    traceback.format_exception(exc_type, exc_obj, exc_tb))
                self._logger.error(errstr)

        self.exit()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def start_watching(self, callback):
        """ Start watchdog observer monitoring the watch_dir directory.
        """
        if not self.watch_dir.is_dir():
            self._logger.error(f'No directory: {self.watch_dir}')
            return

        # Start observer
        if self.polling_observer:
            self._observer = PollingObserver(timeout=self._polling_timeout)
        else:
            self._observer = Observer()

        self._event_handler = \
            RtpDcm2NiiServer._FileHandler(
                self.watch_file_pattern, callback=callback)

        self._observer.schedule(self._event_handler, self.watch_dir,
                                recursive=True)
        self._observer.start()
        self._logger.info(
            "Start observer monitoring " +
            f"{self.watch_dir}/**/{self.watch_file_pattern}")

    # /////////////////////////////////////////////////////////////////////////
    class _FileHandler(FileSystemEventHandler):
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
    def start_RPC_server(self):
        # Boot server
        socketserver.TCPServer.allow_reuse_address = True
        try:
            self._RPCserver = socketserver.TCPServer(
                ('0.0.0.0', self.rpc_port), RtpDcm2NiiServer._dataRecvHandler)
        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errstr = ''.join(
                traceback.format_exception(exc_type, exc_obj, exc_tb))
            self._logger.error(errstr)
            self.server = None
            return

        self._RPCserver._callback = self._RPC_handler
        self._RPCserver._cancel = False

        # Start the server on another thread.
        self._RPCserver_thread = Thread(
            target=self._RPCserver.serve_forever, args=(0.5,))
        # Make the server thread exit when the main thread terminates
        self._RPCserver_thread.daemon = True
        self._RPCserver_thread.start()

    # /////////////////////////////////////////////////////////////////////////
    class _dataRecvHandler(socketserver.BaseRequestHandler):
        def handle(self):
            self._logger = logging.getLogger('RtpDcm2NiiServer')

            # --- Initialize ---
            client_addr, port = self.client_address
            self._logger.info(f"Connected from {client_addr}:{port}")
            self.request.settimeout(1)
            call = None
            while not self.server._cancel:
                try:
                    call = self.request.recv(1024)
                    if len(call) == 0:
                        break
                except socket.timeout:
                    continue
                except BlockingIOError:
                    # No more data available at the moment
                    continue
                except BrokenPipeError:
                    break
                except Exception:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    errstr = ''.join(
                        traceback.format_exception(exc_type, exc_obj, exc_tb))
                    self._logger.error(f'Data receiving error: {errstr}')
                    break

                if call is not None:
                    call = call.decode()
                    if call == 'END':
                        break
                    self._logger.info(f"Received {call}")
                    self.server._callback(call)
                    call = None

            self._logger.info(f"Close connection from {client_addr}:{port}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _RPC_handler(self, call):
        if call == 'CLOSE_SERIES':
            self.end_series()

        elif call == 'QUIT':
            self._cancel = True

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
    def _make_study_id(self, dcm, study_prefix=None):
        if study_prefix is None:
            study_prefix = self.study_prefix
        dateTime = datetime.strptime(
                dcm.StudyDate+dcm.StudyTime, '%Y%m%d%H%M%S.%f')
        studyID = study_prefix+dateTime.strftime('_%Y%m%d%H%M%S')
        studyID = self._make_path_safe(studyID)

        return studyID

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def do_proc(self, dicom_file):
        """
        This is called by the self._observer thread.
        """
        self._last_dicom_time = time.time()  # Record new file arrival time
        dicom_file = Path(dicom_file)
        if self._last_proc_f is not None and \
                self._last_proc_f == dicom_file:
            return
        self._last_proc_f = dicom_file

        # Read dicom file
        st = time.time()
        dcm = None
        try:
            dcm = pydicom.dcmread(dicom_file)
        except Exception:
            pass

        while dcm is None and time.time() - st < self._dcmread_timeout:
            # Wait until the file is readable.
            try:
                dcm = pydicom.dcmread(dicom_file)
            except Exception:
                time.sleep(0.01)

        if dcm is None:
            self._logger.error(f"Failed to read {dicom_file} as DICOM")
            return

        imageType = '\\'.join(dcm.ImageType)

        # Ignore derived file
        if 'DERIVED' in imageType:
            self._logger.debug(f"{dicom_file.name} is a derived file.")
            return

        seriesDescription = dcm.SeriesDescription

        # Ignore MoCoSeries
        if seriesDescription == 'MoCoSeries':
            self._logger.debug(f"{dicom_file.name} is a MoCo series.")
            return

        # Ignore Phase image
        if seriesDescription.endswith('_Pha'):
            self._logger.debug(f"{dicom_file.name} is a Phase series.")
            return

        # Ignore replicated files
        if self._last_dicom_header is not None:
            if self._last_dicom_header == dcm:
                # The same dicom contents
                self._logger.debug(
                    f'Same DICOM contents. Skip {dicom_file.name}.')
                return

        #  --- Process --------------------------------------------------------
        try:
            self._last_dicom_header = dcm
            # Check if a new series starts
            if not self._isRun_series:
                if self._current_study != dcm.StudyInstanceUID:
                    self._current_study = dcm.StudyInstanceUID
                    self._current_series = dcm.SeriesInstanceUID
                    self.init_study(dcm)
                    self.init_series(dcm)

                elif self._current_series != dcm.SeriesInstanceUID:
                    # New series
                    self._current_series = dcm.SeriesInstanceUID
                    self.init_series(dcm)

            elif self._current_series != -1 and \
                    self._current_series != dcm.SeriesInstanceUID:
                # New series wihtout closing the previous one
                self.end_series()
                self._current_series = dcm.SeriesInstanceUID
                self.init_series(dcm)

            self._NVol += 1

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errstr = ''.join(
                traceback.format_exception(exc_type, exc_obj, exc_tb))
            self._logger.error(errstr)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def init_study(self, dcm):
        studyID = self._make_study_id(dcm, self.study_prefix)
        self._logger.info(
            '+' * 6 +
            f" Create a new study {studyID} +++\n#" + '+' * 79)

        # Create a study directory if one does not exist
        self._study_dir = self.work_root / studyID
        if not self._study_dir.is_dir():
            os.makedirs(self._study_dir)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def init_series(self, dcm):
        if self._study_dir is None:
            studyID = self.dcm_converter.make_study_id(dcm, self.study_prefix)
            self.init_study(self, studyID)

        self._logger.info(
            '-' * 6 + f" Create a new series {dcm.SeriesNumber} ---\n#" +
            '-' * 79)

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
            TR = self.series_timeout

        self._TR = TR / 1000

        # Start physio saving if FMRI
        imageType = '\\'.join(dcm.ImageType)
        if 'FMRI' in imageType:
            # if self.rt_mri_disp_ctrl_proc:
            #     self.rt_mri_disp_ctrl_pipe.send('PHYSIO_START_SAVING')
            #     self.rt_mri_disp_ctrl_pipe.send(self._TR)
            #     self.rt_mri_disp_ctrl_pipe.send('CHECK_SERIES_END')
            #     self.rt_mri_disp_ctrl_pipe.send(self._TR)
            self._physio_prefix = str(
                self._study_dir / ("{}_ser-" + f"{dcm.SeriesNumber}.1D"))
            self._NVol = 0

        self._isRun_series = True

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end_series(self):
        if not self._isRun_series:
            return

        self._logger.info("Closing series")

        get_lock = self._process_lock.acquire(timeout=1)
        if get_lock:
            try:
                if self._NVol > 1:
                    # Write saved physio data in a file
                    # if self._TR is not None:
                    #     series_duration = self._NVol * self._TR
                    # else:
                    #     series_duration = None

                    # Stop Physio saving
                    # if self.rt_mri_disp_ctrl_proc:
                    #     self.rt_mri_disp_ctrl_pipe.send('PHYSIO_STOP_SAVING')
                    #     self.rt_mri_disp_ctrl_pipe.send(
                    #         (self.physio_prefix, series_duration))

                    # Reset physio parameters
                    self._physio_prefix = './{}_scan.1D'
                    self._NVol = 0

                # Run DICOM convert process
                dicom_dir = self._last_proc_f.parent
                dcm_conv_proc = Process(
                    target=self.dcm_converter.rt_convert_dicom,
                    args=(dicom_dir, self._study_dir),
                    kwargs={'make_brik': self.make_brik}
                )
                dcm_conv_proc.start()

                last_ser = self._last_dicom_header.SeriesNumber
                last_ser_de = self._last_dicom_header.SeriesDescription
                self._logger.info(
                    f"Close series {last_ser} ({last_ser_de})\n#" +
                    '-' * 80 + '\n#\n#')

                self._isRun_series = False
                self._current_series = -1
                self._last_dicom_time = None
                self._TR = None
                self._last_proc_f = None
                self._last_dicom_header = None

            except Exception:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                errstr = ''.join(
                    traceback.format_exception(exc_type, exc_obj, exc_tb))
                self._logger.error(errstr)

            self._process_lock.release()

        else:
            self._logger.error("Failed to acquire a lock in end_series")
            self._isRun_series = False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def exit(self):
        # This can be called multiple times. To avoid multiple exits, keep a
        # completion record of the exit process.
        if not self._end_complete:
            self.stop_watching()
            if hasattr(self, '_RPCserver'):
                self._RPCserver._cancel = True
                self._RPCserver.shutdown()
                self._RPCserver_thread.join()

            try:
                self._logger.info('\n==== Close RtpDcm2NiiServer ====' +
                                  '\n#' + '=' * 80 + '\n#\n#')
            except Exception:
                pass

            self._end_complete = True

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        self.exit()


# %% __main__ =================================================================
if __name__ == '__main__':
    RTMRI_DIR = Path('/RTMRI/RTExport')
    WORK_DIR = Path('/data/rt')
    RT_COPY_DST_DIR = Path('/RTMRI/RTExport_rt')

    # --- Parse arguments -----------------------------------------------------
    parser = argparse.ArgumentParser(description='RTPSPy dcm2nii server')
    parser.add_argument('--watch_dir', default=RTMRI_DIR,
                        help='Watch directory, where MRI data is exported in' +
                        'real time')
    parser.add_argument('--work_root', default=RTMRI_DIR,
                        help='Converted data output directory root')
    parser.add_argument('--watch_file_pattern', default=r'.+\.dcm',
                        help='watch file pattern (regexp)')
    parser.add_argument('--study_prefix', default='P', help='Study ID prefix')
    parser.add_argument('--series_timeout', default=60,
                        help='Timeout period to close a series')
    parser.add_argument('--rpc_port', default=63210,
                        help='Port number to receive a remote request')
    parser.add_argument('--make_brik', action='store_true',
                        help='Make BRIK files')
    parser.add_argument('--polling_observer', action='store_true',
                        help='Use Polling observer')
    args = parser.parse_args()
    watch_dir = Path(args.watch_dir)
    work_root = Path(args.work_root)
    watch_file_pattern = args.watch_file_pattern
    study_prefix = args.study_prefix
    series_timeout = args.series_timeout
    rpc_port = args.rpc_port
    make_brik = args.make_brik
    polling_observer = args.polling_observer

    # Make log file
    dstr = datetime.now().strftime("%Y%m%d")
    log_file = Path(f'log/RtpDcm2NiiServer_{dstr}.log')
    if not log_file.parent.is_dir():
        os.makedirs(log_file.parent)

    logging.basicConfig(
        level=logging.INFO, filename=log_file, filemode='a',
        format='%(asctime)s.%(msecs)03d,[%(levelname)s],%(name)s,%(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S')

    # Run the server
    dcm2nii_serv = RtpDcm2NiiServer(
        watch_dir, work_root, watch_file_pattern=watch_file_pattern,
        study_prefix=study_prefix, series_timeout=series_timeout,
        rpc_port=rpc_port, make_brik=make_brik,
        polling_observer=polling_observer)

    # Run mainloop
    try:
        dcm2nii_serv.run()
    except Exception:
        pass

    del dcm2nii_serv
