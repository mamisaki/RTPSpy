#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitor the creation of new directories and DICOM files for real-time MRI.

@author: mmisaki@laureateinstitute.org
"""

# %% import ==================================================================#
from pathlib import Path
import time
import re
from datetime import datetime
import sys
from functools import partial
from threading import Lock
import logging
from collections import deque

import numpy as np
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
from dicom2nifti.convert_dicom import dicom_array_to_nifti
from dicom2nifti.image_volume import load, ImageVolume
from dicom2nifti.image_reorientation import _reorient_4d, _reorient_3
from PyQt5 import QtWidgets, QtCore


# %% class RtWatchDicom =======================================================
class RtWatchDicom():
    """
    Watch new dicom directory and file creation.

    Multiple obsevers for each directory level can run in parallel.
    A root observer monitoring creation of a folder is booted first.
    Observers for subdirectories are booted sequentially to monitor under
    the new directory.

    Refer also to the python watchdog package:
    https://pypi.org/project/watchdog/
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, dicom_callback=print,
                 callback_args=[], callback_kwargs={},
                 observer_boot_margin=2.0,
                 end_first_dicom=False, polling_observer=False,
                 polling_timeout=1, verb=True):
        """Initialize WatchDicom class object

        Args:
            dicom_callback (function):
                Function called to handle a dicom file. The function recieves
                Path of the file and callback_args.
            callback_args (list):
                Argument list to the handler.
            callback_kwargs (dict):
                Keyword argument dict to the handler.
            observer_boot_margin (float, optional):
                Time (seconds) to allow the observer to boot.
                If a directory/file was created during this period and the new
                observer missed it, an action defined to handle the
                file/directory creation event will be executed.
            end_first_dicom (bool):
                Process only the first DICOM file and end monitorig.
                The following process will be done by Dimon and AFNI.
            polling_observer (bool):
                Use PollingObserver. If the monitoring directory is on a
                network mount, standard Observer may not work.
            polling_timeout (int, optional):
                Polling timeoput in seconds. Defaults to 1.
            verb (bool, optional):
                Verbatim option. Defaults to False.
        """
        self.logger = logging.getLogger('RtWatchDicom')

        # Set parameters
        self.dicom_callback = dicom_callback
        self.callback_args = callback_args
        self.callback_kwargs = callback_kwargs
        self.observer_boot_margin = observer_boot_margin
        self.end_first_dicom = end_first_dicom
        self.polling_observer = polling_observer
        self.polling_timeout = polling_timeout
        self.verb = verb
        self.processed_que_lock = Lock()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run(self, watch_root, watch_patterns):
        """ Start observer monitoring the root directory.
        Args:
            watch_root (Path): DICOM data root directory
            watch_patterns (list):
                List of directory name patterns in regexs.
                Each item corresponds to the level of directory. Nunber of
                items must fit to the depth of DICOM directories.
        """
        if not watch_root.is_dir():
            self.logger.error(f'No directory: {watch_root}')
            return
        self.watch_root = watch_root
        self.watch_patterns = watch_patterns

        # Prepare the double-ended queue of recently processed items.
        # Keep up to 2 items
        self.processed_que = {lv: deque(['None'], maxlen=2)
                              for lv in range(len(watch_patterns))}

        # Start the root observer
        self.start_observer(watch_root, level=0)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def start_observer(self, watch_dir, level):
        """Start a watchdog observer thread.
        watch_dir (Path):
            Monitoring directory.
        level (int):
            Directory level.
        """
        # Set parameters
        obs_name = f"observer_{level}"
        watch_pattern = self.watch_patterns[level]
        if level != len(self.watch_patterns) - 1:
            directory = True
            create_action = partial(self.new_directory_action, level=level)
        else:
            directory = False
            create_action = self.new_file_action

        delete_action = partial(self.delete_action, level=level)
        event_handler = \
            RtWatchDicom.CreationHandler(
                watch_pattern, directory, create_action=create_action,
                delete_action=delete_action, verb=self.verb)

        # Start observer
        if self.polling_observer:
            observer = PollingObserver(timeout=self.polling_timeout)
        else:
            observer = Observer()
        setattr(self, obs_name, observer)
        observer.schedule(event_handler, watch_dir, recursive=False)
        observer.start()

        if self.verb:
            self.logger.info(
                f"Start {obs_name} monitoring {watch_dir}/{watch_pattern}")

        # ---------------------------------------------------------------------
        # Failsafe process in case the observer missed the creation of a new
        # directory/file during boot.

        # Check for self.observer_boot_margin seconds if a new directory or
        # file has been created that was not processed by the observer.
        missed_creation = False
        st = time.time()
        while not missed_creation and \
                time.time() - st < self.observer_boot_margin:
            # Get directoris/files
            dfs = list(Path(watch_dir).glob('*'))  # directory or files
            if len(dfs) == 0:
                time.sleep(0.1)
                continue

            if directory:
                dfs = [dd for dd in dfs if dd.is_dir() and
                       re.match(watch_pattern, dd.name)]
                # Take only the latest directory
                idx = np.argmax([dd.stat().st_mtime for dd in dfs])
                dfs = dfs[idx:idx+1]
            else:
                dfs = [ff for ff in dfs if ff.is_file() and
                       re.match(watch_pattern, ff.name)]
                # Take the files that are less than observer_boot_margin
                # seconds old.
                df_ages = [ff.stat().st_mtime - st for ff in dfs]
                sidx = np.argsort(df_ages).ravel()
                df_ages = np.array(df_ages)[sidx]
                dfs = np.array(dfs)[sidx]
                dfs = list(dfs[df_ages >= -self.observer_boot_margin])

            if len(dfs) == 0:
                time.sleep(0.1)
                continue

            # Process the found directoies/files
            for df in dfs:
                self.processed_que_lock.acquire()
                if df in self.processed_que[level]:
                    self.logger.debug(f"{df} is in que.")
                    # If it has been processed, ignore it.
                    self.processed_que_lock.release()
                    continue
                self.processed_que_lock.release()

                # Take an action handling the new diirectory/file missed by
                # the observer.
                if level+1 != len(self.watch_patterns):
                    self.new_directory_action(df, level)
                else:
                    self.new_file_action(df)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def new_directory_action(self, new_dir, level):
        """Actions to be performed when a new directory is found.
        Args:
            new_dir (Path): Path to a new patient directory
            level (int): level of new_dir
        """

        self.processed_que_lock.acquire()
        if new_dir in self.processed_que[level]:
            # This has been processed.
            self.processed_que_lock.release()
            return
        else:
            # Put new_dir to the processed_que
            self.processed_que[level].append(new_dir)
        self.processed_que_lock.release()

        # Stop observers monitoring the same and lower levels as the current
        for obs_level in range(level+1, len(self.watch_patterns)):
            self.stop_observer(obs_name=f'observer_{obs_level}')

        # Boot a child observer
        self.start_observer(new_dir, level+1)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def new_file_action(self, new_file):
        level = len(self.watch_patterns) - 1
        self.processed_que_lock.acquire()
        if new_file in self.processed_que[level]:
            # This has been processed by other threads.
            self.processed_que_lock.release()
            return
        else:
            # put new_file to the processed_que
            self.processed_que[level].append(new_file)
        self.processed_que_lock.release()

        # Call dicom_callback
        self.dicom_callback(new_file, *self.callback_args,
                            **self.callback_kwargs)

        # End file observer
        if self.end_first_dicom:
            file_obs_name = f"observer_{len(self.watch_patterns)-1}"
            if hasattr(self, file_obs_name):
                observer = getattr(self, file_obs_name)
                if observer is not None and observer.is_alive():
                    try:
                        observer.stop()
                        observer.join(3)
                    except Exception:
                        sys.exit()
                    if self.verb:
                        self.logger.info(f"Stop {file_obs_name}.")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def stop_observer(self, obs_name):
        if not hasattr(self, obs_name):
            return

        observer = getattr(self, obs_name)
        if observer is not None and observer.is_alive():
            observer.stop()
            observer.join(3)
            if self.verb:
                self.logger.info(f"Stop {obs_name}.")

        del observer
        setattr(self, obs_name, None)

        level = int(obs_name.split('_')[-1])
        self.processed_que_lock.acquire()
        self.processed_que[level].clear()
        self.processed_que_lock.release()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def delete_action(self, del_path, level):
        if del_path in self.processed_que[level]:
            # Deleted one is processed recently.
            if level < len(self.watch_patterns)-1:
                # When the observer level is not the deepest.
                idx = self.processed_que[level].index(del_path)
                if idx == len(self.processed_que[level])-1:
                    # The deleted one is the most recent (observing) directory
                    # Stop observers monitoring the lower levels
                    for obs_level in range(level+1, len(self.watch_patterns)):
                        self.stop_observer(obs_name=f'observer_{obs_level}')
                self.processed_que_lock.acquire()
                self.processed_que[level].remove(del_path)
                self.processed_que_lock.release()
            else:
                # remove the del_path from the processed que
                self.processed_que_lock.acquire()
                if del_path in self.processed_que[level]:
                    self.processed_que[level].remove(del_path)
                self.processed_que_lock.release()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    class CreationHandler(FileSystemEventHandler):
        """ Event handling class for directory/file creation """
        def __init__(self, watch_pattern, directory=True,
                     create_action=None, delete_action=None,
                     verb=False):
            """
            Parameters
            ----------
            watch_pattern : str
                Regular expression to filter the created directory/file name.
            directory : bool
                Check directory only.
            create_action : function
                Applied action to the created directory/file.
            delete_action : function
                Applied action to the deleted directory/file.
            verb : bool
                Flag to save log.
            """
            super().__init__()

            self.logger = logging.getLogger('RtWatchDicom')
            self.watch_pattern = watch_pattern
            self.create_action = create_action
            self.delete_action = delete_action
            self.directory = directory
            self.verb = verb

        def on_created(self, event):
            if self.directory and not event.is_directory:
                return
            elif not self.directory and event.is_directory:
                return

            name = Path(event.src_path).name
            if re.match(self.watch_pattern, name):
                if self.verb:
                    ctime = datetime.fromtimestamp(
                        Path(event.src_path).stat().st_ctime).isoformat()
                    self.logger.info(
                        f"{event.src_path} is created at {ctime}")

                if self.create_action is not None:
                    self.create_action(event.src_path)

        def on_deleted(self, event):
            if self.directory and not event.is_directory:
                return
            elif not self.directory and event.is_directory:
                return

            name = Path(event.src_path).name
            if re.match(self.watch_pattern, name):
                self.logger.debug(
                    f"{event.src_path} is deleted")

                if self.delete_action is not None:
                    self.delete_action(event.src_path)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def exit(self):
        # Kill observer process
        for obs_level in range(0, len(self.watch_patterns)):
            self.stop_observer(obs_name=f'observer_{obs_level}')

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        self.exit()


# %% __main__ (test) ==========================================================
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG, stream=sys.stdout,
        format='%(asctime)s.%(msecs)03d,%(name)s,%(levelname)s,%(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S')

    watch_root = Path('/RTMRI/RTExport')
    watch_patterns = [r'.+', r'.+\.dcm']

    # Create WatchDicom instance
    watcher = RtWatchDicom()

    # Start watching
    watcher.run(watch_root, watch_patterns)

    try:
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        del watcher
