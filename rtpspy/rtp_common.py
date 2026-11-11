# -*- coding: utf-8 -*-
"""
Common base class and utilities for the RTP modules

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import os
import sys
import re
import subprocess
import shlex
import pickle
import time
import traceback
import socket
import logging
import threading

import nibabel as nib
import numpy as np
import pandas as pd

import serial

try:
    from .rtp_volume import RtpVolume
    from .notifier import get_notifier, default_notifier
    from .mp_context import mp_ctx
    from . import bids_name as bn
except Exception:
    from rtpspy.rtp_volume import RtpVolume
    from rtpspy.notifier import get_notifier, default_notifier
    from rtpspy.mp_context import mp_ctx
    from rtpspy import bids_name as bn

# Shared memory objects of RingBuffer, for the processes started with the
# 'fork' start method (see mp_context.py)
Value, Array, Lock = mp_ctx.Value, mp_ctx.Array, mp_ctx.Lock


# %% RTP base class ===========================================================
# Default maximum scan length (number of volumes) used to pre-allocate the
# buffers (e.g., 800 volumes = 20 min with TR 1.5 s). Set max_scan_length to
# the realistic length of the scan.
DEFAULT_MAX_SCAN_LENGTH = 800


class RTP(object):
    """Base class for the real-time processing modules"""

    # GUI panel of the module: 'module:class' in rtpspy.ui (see
    # rtpspy.ui.module_ui). None if the module has no panel.
    _ui_class = None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, ignore_init=0, next_proc=None, save_proc=False,
                 online_saving=False, save_delay=False, work_dir='',
                 max_scan_length=None, main_win=None, **kwargs):

        self.ignore_init = ignore_init
        self.next_proc = next_proc
        self.save_proc = save_proc
        self.online_saving = online_saving
        self.save_delay = save_delay
        self.work_dir = work_dir
        if max_scan_length is None:
            max_scan_length = DEFAULT_MAX_SCAN_LENGTH
        self.max_scan_length = max_scan_length
        self.main_win = main_win

        # Run state
        self._vol_num = 0  # Number of received volumes
        self._proc_start_idx = -1  # 0-based index of the first processed vol
        self._proc_time = []
        self.proc_delay = []

        self.saved_files = []
        self._temp_saved_files = []  # files written by save_temp
        self.saved_filename = None  # File of the concatenated saved data
        # BIDS entities of the output files of the scan (set by
        # set_out_entities; see bids_name.resolve_scan_entities) and the
        # metadata for the image sidecar (e.g., RepetitionTime)
        self._out_entities = None
        self._out_meta = {}
        self._last_out_entities = None
        self.saved_data = None
        self._saved_last_idx = -1  # 0-based index of the last saved volume
        self.enabled = True

        self._logger = logging.getLogger(self.__class__.__name__)
        self._log_level = logging.getLevelName(
            self._logger.getEffectiveLevel())
        self._proc_ready = False

        self._std_out = sys.stdout
        self._err_out = sys.stderr

        # GUI panel bound by ui_set_param (None without a GUI)
        self._ui = None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @property
    def std_out(self):
        return self._std_out

    @std_out.setter
    def std_out(self, std_out):
        self._std_out = std_out

    @property
    def err_out(self):
        return self._err_out

    @err_out.setter
    def err_out(self, err_out):
        self._err_out = err_out

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ready_proc(self):
        """
        Prepare the process. If self.next_proc is set, its ready_proc() is
        also called.

        Subclasses should override this.

        Returns
        -------
        bool
            True if this and all the following processes are ready.
        """

        self._vol_num = 0
        self._proc_start_idx = -1
        self._proc_time = []
        self.proc_delay = []
        self.saved_files = []
        self._temp_saved_files = []
        self.saved_data = None
        self._saved_last_idx = -1

        if self.next_proc:
            self._proc_ready &= self.next_proc.ready_proc()

        return self._proc_ready

    # --- Volume processing ---------------------------------------------------
    # do_proc() is a template method that runs the bookkeeping common to the
    # modules of the processing chain. A module implements only its
    # processing step in _process() and sets the two class attributes below.
    # The other hooks (_next_proc_kwargs, _output, _proc_error) have defaults
    # that a module overrides only when it needs to.

    # Processing name in the log message
    # ("#<volume number>;<_proc_label>;<filename>;tstamp=<time>;took <s>s")
    _proc_label = "Process"
    # Prefix added to the filename of the processed image (e.g., 'vr.')
    _out_prefix = ""
    # desc entity of the saved processed image (desc-<_bids_desc>_bold).
    # None: the class name without 'Rtp', in lowercase.
    _bids_desc = None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def do_proc(self, fmri_img, vol_idx=None, pre_proc_time=None, **kwargs):
        """
        Process a volume and pass the result to self.next_proc.

        The steps are:
        1. Count the received volume (self._vol_num); vol_idx is set from
           the count if it is None. A volume with vol_idx < self.ignore_init
           is skipped. self._proc_start_idx is set to the index of the first
           volume that is not skipped.
        2. A nibabel image that is not an RtpVolume is converted to an
           RtpVolume (sharing the data; the given image is not changed).
           self._process() processes the volume. If it returns None (e.g.,
           the volume is held to be output later, or it cannot be
           processed), the steps below are not done.
        3. Record the processing time (self._proc_time, self.proc_delay with
           save_delay) and log it. self._out_prefix is added to the
           filename of the processed image, and its vol_idx and proc_time
           are set. The received volume is not changed: if _process()
           returns it and a prefix is to be added, a new volume is made.
        4. self._output() passes the image to self.next_proc.do_proc(), with
           the keyword arguments returned by self._next_proc_kwargs(), and
           keeps it for saving if self.save_proc is True.

        An exception is logged by self._proc_error() and not raised, so that
        an error in one volume does not stop the real-time processing.

        Parameters
        ----------
        fmri_img : RtpVolume or nibabel image object
            Volume to process.
        vol_idx : int, optional
            0-based index of the volume from the scan start. If None, the
            number of received volumes - 1 is used.
        pre_proc_time : float, optional
            Time stamp when the previous module finished. If None, the
            processing time is not logged.
        kwargs : dict
            Additional information from the previous module (e.g., qc from
            RtpRegress). It is not used here.
        """
        try:
            self._vol_num += 1  # 1-based number of received volumes
            if vol_idx is None:
                vol_idx = self._vol_num - 1  # 0-based index

            if vol_idx < self.ignore_init:
                return

            if self._proc_start_idx < 0:
                self._proc_start_idx = vol_idx

            # --- Run the process ---------------------------------------------
            in_img = RtpVolume.as_volume(fmri_img, vol_idx=vol_idx)
            out_img = self._process(in_img, vol_idx, pre_proc_time)
            if out_img is None:
                return

            # --- Post-process ------------------------------------------------
            tstamp, proc_delay = self._record_proc_time(pre_proc_time)
            fname = self._img_name(out_img)
            self._log_proc(f"#{vol_idx+1};{self._proc_label};{fname}",
                           tstamp, proc_delay)
            # The received volume is not changed. in_img is made here if
            # fmri_img was converted.
            own = out_img is not in_img or in_img is not fmri_img
            if not own and self._out_prefix:
                out_img = in_img.with_data(np.asanyarray(in_img.dataobj))
                own = True
            if own:
                # A volume made by this module
                out_img = RtpVolume.as_volume(out_img)
                if self._out_prefix:
                    out_img.set_filename(self._out_prefix + fname)
                out_img.vol_idx = vol_idx
                out_img.proc_time = tstamp

            self._output(out_img, vol_idx, tstamp,
                         **self._next_proc_kwargs(vol_idx))

        except Exception:
            self._proc_error()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _process(self, fmri_img, vol_idx, pre_proc_time):
        """
        Processing step of the module, called by do_proc(). Subclasses in
        the processing chain must override this.

        Parameters
        ----------
        fmri_img : RtpVolume
            Volume to process. It must not be changed; the processed data
            are returned in a new volume (e.g., fmri_img.with_data(data)).
        vol_idx : int
            0-based index of the volume (vol_idx >= self.ignore_init).
        pre_proc_time : float or None
            Time stamp when the previous module finished. It is needed only
            to log the processing time of a volume that is not returned
            (see self._record_proc_time and self._log_proc).

        Returns
        -------
        RtpVolume or None
            Processed image to be passed to the next process, or None if no
            image is output for this volume. A module that outputs earlier
            volumes retrospectively passes them with
            self._output_earlier_volume() within this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _process().")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _next_proc_kwargs(self, vol_idx):
        """
        Keyword arguments added to next_proc.do_proc() for volume vol_idx
        (e.g., qc in RtpRegress). The default is none.
        """
        return {}

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _output(self, fmri_img, vol_idx, tstamp, **kwargs):
        """
        Pass the processed image to the next process and keep it for saving
        (if self.save_proc). A module that is the end of the chain (e.g.,
        RtpApp) overrides this.

        Parameters
        ----------
        fmri_img : RtpVolume
            Processed image.
        vol_idx : int
            0-based index of the volume.
        tstamp : float
            Time stamp when this module finished, passed to the next process
            as pre_proc_time.
        kwargs : dict
            Keyword arguments added to next_proc.do_proc().
        """
        self._pass_to_next(fmri_img, vol_idx, tstamp, **kwargs)
        if self.save_proc:
            self.keep_processed_image(fmri_img, vol_num=vol_idx,
                                      save_temp=self.online_saving)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _pass_to_next(self, fmri_img, vol_idx, pre_proc_time, copy=False,
                      **kwargs):
        """
        Pass the processed image to self.next_proc.do_proc().

        The data of this module's output are kept in self.proc_data (read
        by other modules, e.g., RtpRegress.mask_src_proc).

        Parameters
        ----------
        copy : bool, optional
            Keep a copy of the data in self.proc_data, for a module whose
            data array may be modified in place by the next process.
        """
        if not self.next_proc:
            return

        if copy:
            self.proc_data = np.array(fmri_img.dataobj)
        else:
            self.proc_data = np.asanyarray(fmri_img.dataobj)

        self.next_proc.do_proc(fmri_img, vol_idx=vol_idx,
                               pre_proc_time=pre_proc_time, **kwargs)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _output_earlier_volume(self, fmri_img, vol_idx, save_temp, **kwargs):
        """
        Output a volume before the current one, processed retrospectively
        (e.g., by RtpTshift and RtpRegress) within self._process(). The
        volume is kept for saving (if self.save_proc) before it is passed to
        the next process. pre_proc_time of the next process is the current
        time, which is also set to fmri_img.proc_time.

        Parameters
        ----------
        fmri_img : RtpVolume
            Processed image of the earlier volume, made by this module.
        vol_idx : int
            0-based index of the earlier volume.
        save_temp : bool
            save_temp of keep_processed_image().
        kwargs : dict
            Keyword arguments added to next_proc.do_proc().
        """
        if self.save_proc:
            self.keep_processed_image(fmri_img, vol_num=vol_idx,
                                      save_temp=save_temp)

        tstamp = time.time()
        fmri_img.proc_time = tstamp
        if self.next_proc:
            self.next_proc.do_proc(fmri_img, vol_idx, tstamp, **kwargs)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _record_proc_time(self, pre_proc_time):
        """
        Record the time stamp of the end of the processing.

        Returns
        -------
        tstamp : float
            Time stamp.
        proc_delay : float or None
            Time from pre_proc_time; None if pre_proc_time is None.
        """
        tstamp = time.time()
        self._proc_time.append(tstamp)
        proc_delay = None
        if pre_proc_time is not None:
            proc_delay = tstamp - pre_proc_time
            if self.save_delay:
                self.proc_delay.append(proc_delay)

        return tstamp, proc_delay

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _log_proc(self, msg, tstamp, proc_delay):
        """Log msg with the time stamp and the processing time."""
        msg += f";tstamp={tstamp}"
        if proc_delay is not None:
            msg += f";took {proc_delay:.4f}s"
        self._logger.info(msg)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def _img_name(fmri_img):
        """Filename (without the directory) of fmri_img."""
        if fmri_img.get_filename():
            return Path(fmri_img.get_filename()).name
        return "unknown.nii.gz"

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _proc_error(self):
        """
        Log the exception raised in do_proc(); called in its except block.
        The error message gives the exception type and the location in the
        module's code (the frame called from do_proc), and the traceback is
        printed to self.err_out.
        """
        exc_type, exc_obj, exc_tb = sys.exc_info()
        tb = exc_tb.tb_next if exc_tb.tb_next is not None else exc_tb
        errmsg = f'{exc_type}, {tb.tb_frame.f_code.co_filename}' + \
            f':{tb.tb_lineno}'
        self._logger.error(errmsg)
        traceback.print_exc(file=self._err_out)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end_reset(self):
        """
        End the process, save the kept data, and reset the run state.
        If self.next_proc is set, its end_reset() is also called.

        Returns
        -------
        dict
            Saved files of this and the following processes
            ({'<class name>:<content>': file}).
        """

        out_files = {}

        try:
            # Save proc_delay
            if self.save_delay and len(self.proc_delay) > 0:
                out_f = self.save_proc_delay()
                out_files[f"{self.__class__.__name__}:process time"] = out_f

            # Save the kept volumes as a 4D file
            if self.saved_data is not None:
                self.save_processed_data(
                    self.saved_data, self.saved_data_affine,
                    saved_files=self.saved_files)
                out_files[f"{self.__class__.__name__}:processed image"] = \
                    self.saved_filename
        except Exception:
            self._logger.error(
                "Failed to save the processed data:\n"
                + traceback.format_exc())
        finally:
            # Reset running variables even if saving failed, so that the
            # state does not leak into the next run.
            self._vol_num = 0
            self._proc_start_idx = -1
            self._proc_time = []
            self.proc_delay = []
            self.saved_files = []
            self._temp_saved_files = []
            self.saved_data = None
            self._saved_last_idx = -1
            # Kept for the outputs saved after this (e.g., the QC table of
            # RtpRegress)
            self._last_out_entities = self._out_entities
            self._out_entities = None
            self._out_meta = {}

        # Reset the following process
        if self.next_proc:
            out_fs = self.next_proc.end_reset()
            if out_fs is not None:
                out_files.update(out_fs)

        return out_files

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_param(self, attr, val, echo=True):
        self._logger.debug(f"set_param: {attr} = {val}")

        setattr(self, attr, val)
        if echo:
            print("{}.".format(self.__class__.__name__) + attr, '=',
                  getattr(self, attr))

    # --- GUI interface -------------------------------------------------------
    # The GUI code is in rtpspy.ui. A module calls its GUI panel only
    # through the methods below, which do nothing without a GUI.

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ui_set_param(self):
        """
        Make the GUI panel of the module (rtpspy.ui; Qt is imported here)
        and return the list of its rows, each placed in a QFormLayout by
        RtpGUI. None if the module has no panel.
        """
        if self._ui_class is None:
            return None

        from .ui.module_ui import bind_ui
        return bind_ui(self).make_ui()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _ui_update(self, attr, val, **kwargs):
        """Show the value of parameter attr in the GUI panel, if any."""
        ui = getattr(self, '_ui', None)
        if ui is not None:
            ui.update_widget(attr, val, **kwargs)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _ui_value(self, key):
        """Return a state held only in the GUI panel (None without it)."""
        ui = getattr(self, '_ui', None)
        if ui is not None:
            return ui.widget_value(key)
        return None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _ui_call(self, method, *args, **kwargs):
        """Run a method of the GUI panel, if any."""
        ui = getattr(self, '_ui', None)
        if ui is not None:
            return getattr(ui, method)(*args, **kwargs)
        return None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _notifier(self):
        """Notifier to show messages and dialogs (rtpspy.notifier)."""
        return get_notifier()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def select_file_dlg(self, caption, directory, filt, parent=None):
        """
        Ask the user to select a file (a file dialog in the GUI). Returns the
        (filename, filter) tuple; filename is '' if canceled or without a
        GUI.
        """
        return self._notifier().get_open_file(
            caption, directory, filt, parent=parent)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def keep_processed_image(self, fmri_img, vol_num=None, save_temp=False):
        """
        Keep the processed volume in self.saved_data. At the end of the scan,
        self.saved_data is saved as a 4D file by self.save_processed_data().

        Parameters
        ----------
        fmri_img : nibabel image
            nibabel image object.
        vol_num : int, optional
            0-based index of the volume. If None, the latest received volume
            (self._vol_num - 1) is used. The default is None.
        save_temp : bool, optional
            Also save the volume immediately as a single-volume file (e.g., for
            debugging). These files are deleted after the 4D file is saved.
        """

        if vol_num is None:
            vol_num = self._vol_num - 1  # 0-based index of the latest volume

        # Save as work_dir/RTP/<image filename>. The image filename can be
        # the path of an existing file (e.g., the input file, or the file
        # written by nib.save with save_temp). The name is changed only not
        # to overwrite a file that is not an output of this module.
        savefname = fmri_img.get_filename()
        if savefname is not None:
            savefname = Path(self.work_dir) / 'RTP' / Path(savefname).name
        if savefname is None or (savefname.is_file() and
                                 str(savefname) not in self._temp_saved_files):
            savefname = Path(self.work_dir) / 'RTP' / \
                f"RTP_image_{vol_num + 1:04d}.nii.gz"

        self.saved_files.append(str(savefname))

        if save_temp:
            self.save_data(fmri_img, savefname)
            self._temp_saved_files.append(str(savefname))

        # The buffer is allocated once for max_scan_length volumes (np.zeros
        # reserves the memory lazily, so this is fast). It is extended only
        # when the scan is longer than max_scan_length, which should be set
        # to the realistic scan length.
        if self.saved_data is None:
            n = max(vol_num + 1, self.max_scan_length)
            self.saved_data = np.zeros([*fmri_img.shape[:3], n],
                                       dtype=np.float32)
        elif vol_num >= self.saved_data.shape[-1]:
            n = max(vol_num + 1, int(self.saved_data.shape[-1] * 1.25) + 100)
            self._logger.warning(
                f"The scan is longer than max_scan_length "
                f"({self.max_scan_length}). The save buffer is extended to "
                f"{n} volumes. Set max_scan_length to the scan length.")
            old_data = self.saved_data
            self.saved_data = np.zeros([*old_data.shape[:3], n],
                                       dtype=old_data.dtype)
            self.saved_data[..., :old_data.shape[-1]] = old_data

        self.saved_data[..., vol_num] = np.asarray(
            fmri_img.dataobj, dtype=np.float32).reshape(
                self.saved_data.shape[:3])
        self.saved_data_affine = fmri_img.affine
        self._saved_last_idx = max(self._saved_last_idx, vol_num)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save_data(self, fmri_img, savefname):
        save_dir = Path(savefname).parent
        if not save_dir.is_dir():
            save_dir.mkdir()

        nib.save(fmri_img, savefname)
        self._logger.info(f"Saved data as {savefname}")

        return savefname

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save_proc_delay(self):
        """Save the processing time of each volume as
        work_dir/RTP/<entities>_desc-<module>proctime_timeseries.tsv."""
        save_dir = Path(self.work_dir) / 'RTP'
        if not save_dir.is_dir():
            save_dir.mkdir(parents=True)

        save_fname = save_dir / bn.bids_name(
            self.out_entities(), "timeseries", ".tsv",
            desc=f"{self.bids_desc}proctime")
        bn.write_tsv(
            pd.DataFrame({"proc_delay": np.asarray(self.proc_delay,
                                                   dtype=float)}),
            save_fname,
            columns_info={"proc_delay": {
                "Description": f"Processing time of {self._proc_label} "
                "for each processed volume (from the end of the previous "
                "module)", "Units": "s"}})
        return save_fname

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @property
    def bids_desc(self):
        """desc entity of the outputs of this module (e.g., 'volreg')."""
        if self._bids_desc is not None:
            return self._bids_desc
        return bn.bids_label(
            re.sub(r"^Rtp", "", self.__class__.__name__).lower()) or "rtp"

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_out_entities(self, entities, meta=None):
        """
        Set the BIDS entities (and the image sidecar metadata, e.g.,
        {'RepetitionTime': 2.0}) of the output files of the scan, for this
        module and the following ones. RtpApp sets them when the scan
        starts to output (see bids_name.resolve_scan_entities); None
        resets them.
        """
        rtp = self
        while rtp is not None:
            rtp._out_entities = None if entities is None else dict(entities)
            rtp._out_meta = dict(meta or {})
            rtp = getattr(rtp, 'next_proc', None)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def out_entities(self):
        """
        BIDS entities of the output files of the scan: the ones set by
        set_out_entities, or else the entities read from the name of the
        first kept volume (resolved in work_dir/RTP, and kept for the other
        outputs of the scan).
        """
        if self._out_entities is None:
            src = self.saved_files[0] if len(self.saved_files) else ''
            ent = bn.entities_from_filename(src) if src else \
                {'sub': 'unknown', 'task': 'rtp'}
            ent = bn.resolve_scan_entities(
                ent, [Path(self.work_dir) / 'RTP',
                      Path(self.work_dir) / 'log'])
            # The same entities for the following modules of the scan
            self.set_out_entities(ent, self._out_meta)
        return self._out_entities

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_params(self):
        """
        Return the parameters to save (see save_parameters). Run-time data,
        private (underscore) attributes, UI objects, and objects that cannot
        be pickled are excluded.
        """
        excluded_opts = ('main_win', 'saved_files', 'next_proc', 'enabled',
                         'save_proc', 'save_delay', 'proc_delay', 'proc_data',
                         'saved_data', 'max_scan_length', 'done_proc',
                         'saved_data_affine', 'saved_filename',
                         'online_saving')

        opts = dict()
        for var_name, val in self.__dict__.items():
            if var_name in excluded_opts or var_name[0] == '_' or \
                    'ui_' in var_name or _is_unsaved_type(val):
                continue

            if isinstance(val, Path):
                val = str(val)

            if _is_picklable(val):
                opts[var_name] = val

        return opts

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save_processed_data(self, saved_data, affine, saved_files=[],
                            save_prefix=''):
        """
        Save the kept volumes (from the first processed volume to the last
        kept one) as a 4D NIfTI file in work_dir/RTP, named
        <entities>_desc-<module>_bold.nii.gz (see out_entities), with a
        JSON sidecar. save_prefix (deprecated) replaces the desc label.
        """

        # --- Start message ---
        if self.main_win is not None:
            self._logger.info(
                "<B>Saving the processed data. Please wait ...")
            self._notifier().process_events()
        else:
            self._logger.info("Saving the processed data. Please wait ...")

        # --- Set filename ---
        if len(saved_files) and self._out_entities is None:
            self.saved_files = list(saved_files)
        desc = bn.bids_label(save_prefix) or self.bids_desc
        save_dir = Path(self.work_dir) / 'RTP'
        savefname = save_dir / bn.bids_name(
            self.out_entities(), "bold", ".nii.gz", desc=desc)
        if not save_dir.is_dir():
            os.makedirs(save_dir)

        # --- Save ---
        start_idx = max(self._proc_start_idx, 0)
        img_data = saved_data[:, :, :, start_idx:self._saved_last_idx + 1]
        out_img = nib.Nifti1Image(img_data, affine)
        TR = self._out_meta.get('RepetitionTime')
        if TR:
            out_img.header.set_xyzt_units('mm', 'sec')
            zooms = out_img.header.get_zooms()
            out_img.header.set_zooms(tuple(zooms[:3]) + (float(TR),))
        out_img.to_filename(savefname)

        # JSON sidecar
        meta = {
            "Description": f"Real-time processed by RTPSpy "
            f"({self.__class__.__name__}).",
            "NumberOfVolumesDiscardedByUser": int(start_idx),
        }
        meta.update(self._out_meta)
        meta["RTPSpyModule"] = self.__class__.__name__
        bn.write_json(savefname.parent / (bn.name_stem(savefname) + ".json"),
                      meta)

        # --- Delete the single-volume files saved with save_temp ---
        for ff in self._temp_saved_files:
            if Path(ff).is_file():
                self._logger.info(f'Delete {ff}')
                Path(ff).unlink()

        self.saved_filename = savefname
        self._logger.info(f"Done. Processed data is saved as {savefname}")
        if self.main_win is not None:
            self._notifier().process_events()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def err_popup(self, errmsg):
        """
        Report an error to the user through the notifier: a message box when
        the module is in the GUI (main_win is set), otherwise a log message
        (the default notifier).
        """
        if self.main_win is not None:
            notifier = self._notifier()
        else:
            notifier = default_notifier()
        notifier.error(str(errmsg), title=self.__class__.__name__)


# %% Helpers for get_params and save_parameters ==============================
def _is_unsaved_type(val):
    """True if val is an object that is never saved as a parameter."""
    if isinstance(val, (RTP, serial.Serial, threading.Thread)):
        return True

    # A QThread (only if Qt is used; Qt is not imported here)
    qtc = sys.modules.get('PyQt6.QtCore')
    return qtc is not None and isinstance(val, qtc.QThread)


def _is_picklable(val):
    try:
        pickle.dumps(val)
        return True
    except Exception:
        return False


# %% save_parameters ==========================================================
def save_parameters(objs, fname='RTPSpy_params.pkl'):
    """
    Save the parameters of the RTP modules to a pickle file.

    Parameters
    ----------
    objs : dict
        {module name: RTP object}.
    fname : str or Path, optional
        Output file. The default is 'RTPSpy_params.pkl'.
    """
    props = dict()
    for k in objs.keys():
        if k == 'RETROTS':
            continue

        props[k] = dict()
        if objs[k] is None or not hasattr(objs[k], 'get_params'):
            continue

        # get_params can be overridden, so filter its output again
        for var_name, var_val in objs[k].get_params().items():
            if var_name in ('main_win', '_std_out', '_err_out') or \
                    'ui_' in var_name or _is_unsaved_type(var_val):
                continue

            if _is_picklable(var_val):
                props[k][var_name] = var_val

    with open(fname, 'wb') as fd:
        pickle.dump(props, fd)


# %% load_parameters ==========================================================
# Old parameter names saved by earlier versions: {old name: current name}
RENAMED_PARAMS = {'aseg_oirg': 'aseg_orig'}


def load_parameters(objs, fname='RTPSpy_params.pkl'):
    """
    Load the parameters saved by save_parameters and set them to the RTP
    modules with set_param. Only the attributes that the objects have are
    set.

    Returns
    -------
    dict, False, or None
        The loaded parameters; False if fname is not found; None if loading
        failed.
    """
    if not Path(fname).is_file():
        return False

    try:
        sys.stdout.write(f"Load parameters from {Path(fname).absolute()}\n")
        with open(fname, 'rb') as fd:
            props = pickle.load(fd)

        for mod in props.keys():
            if mod in objs:
                obj = objs[mod]
                for var_name, val in props[mod].items():
                    var_name = RENAMED_PARAMS.get(var_name, var_name)
                    if not hasattr(obj, var_name) or var_name is None:
                        continue
                    try:
                        obj.set_param(var_name, val)
                    except TypeError:
                        sys.stderr.write(
                            "Error setting "
                            f"{obj.__class__.__name__}.{var_name} = {val}")

        return props

    except Exception:
        print(traceback.format_exc())


# %% plot_pause ===============================================================
def plot_pause(interval):
    """
    Pause function for matplotlib to update a plot in the background
    without raising the window
    https://stackoverflow.com/questions/45729092/make-interactive-matplotlib-window-not-pop-to-front-on-each-update-windows-7/45734500#45734500
    """

    import matplotlib
    import matplotlib.pyplot as plt

    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()

            canvas.start_event_loop(interval)


# %% RingBuffer ===============================================================
class RingBuffer:
    """Ring buffer in a shared memory Array"""

    def __init__(self, max_size):
        self.cur = Value('i', 0)
        self.max = max_size
        self.data = Array('d', np.ones(max_size)*np.nan)
        self._lock = Lock()

    def append(self, x):
        """Append an element, overwriting the oldest one."""
        with self._lock:
            self.data[self.cur.value] = x
            self.cur.value = (self.cur.value+1) % self.max

    def get(self):
        """Return the elements in order from the oldest."""
        return np.concatenate([self.data[self.cur.value:],
                               self.data[:self.cur.value]])


# %% make_design_matrix =======================================================
def make_design_matrix(stim_times, scan_len, TR, out_f):
    """
    Make a design matrix with AFNI 3dDeconvolve and save it as a CSV file.

    Parameters
    ----------
    stim_times : list of str
        Each element is 'times;basis;name', where times and basis are given
        to the -stim_times option of 3dDeconvolve.
    scan_len : float
        Scan length (seconds).
    TR : float
        Repetition time (seconds).
    out_f : str or Path
        Output CSV file.
    """
    NT = int(scan_len//TR)
    cmd = f"3dDeconvolve -x1D_stop -nodata {NT} {TR}"
    cmd += " -local_times -x1D stdout:"
    cmd += f" -num_stimts {len(stim_times)}"
    names = []
    for ii, optstr in enumerate(stim_times, start=1):
        times, opt, name = optstr.split(';')
        names.append(name)
        cmd += f" -stim_times {ii} '{times}' '{opt}'"
        cmd += f" -stim_label {ii} {name}"

    pr = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE)
    out = pr.stdout.decode()

    # The matrix is between the '# >' and '# <' lines
    ma = re.search(r'\# \>\n(.+)\n# \<', out, re.MULTILINE | re.DOTALL)
    mtx = ma.groups()[0]
    mtx = np.array([[float(v) for v in ll.split()] for ll in mtx.split('\n')])
    mtx = mtx[:, -len(stim_times):]

    desMtx = pd.DataFrame(mtx, columns=names)
    desMtx.to_csv(out_f, index=False)


# %% _list_proc_argv ==========================================================
def _list_proc_argv():
    """
    List (pid, argv) of the running processes, except this one. argv is
    read from /proc/<pid>/cmdline (NUL-separated). Returns an empty list
    if /proc is not available.
    """
    procs = []
    for cmdline_f in Path('/proc').glob('[0-9]*/cmdline'):
        pid = int(cmdline_f.parent.name)
        if pid == os.getpid():
            continue
        try:
            argv = cmdline_f.read_bytes().decode(errors='replace')
        except OSError:
            continue  # The process has ended or is not accessible
        argv = argv.rstrip('\0').split('\0')
        if len(argv[0]):
            procs.append((pid, argv))

    return procs


# %% boot_afni ================================================================
def boot_afni(main_win=None, boot_dir='./', rt=True, TRUSTHOST=None):
    """
    Start AFNI in boot_dir. If an AFNI process for boot_dir is already
    running, the user is asked whether to kill it.

    Parameters
    ----------
    main_win : QMainWindow, optional
        The AFNI window is placed below this window.
    boot_dir : str or Path, optional
        Directory to open with AFNI. The default is './'.
    rt : bool, optional
        Start AFNI in real-time mode. The default is True.
    TRUSTHOST : str, optional
        AFNI_TRUSTHOST for the real-time mode. 'ask' shows an input dialog
        (only with main_win). If None, or 'ask' without main_win, the subnet
        of this host is used.

    The dialogs are shown by the notifier (rtpspy.notifier). Without a GUI,
    the input dialog is canceled and an existing AFNI process is kept
    without booting a new one.
    """
    notifier = get_notifier()
    if rt:
        # Set TRUSTHOST
        ip = socket.gethostbyname(socket.gethostname())
        if TRUSTHOST is None or (TRUSTHOST == 'ask' and main_win is None):
            TRUSTHOST = '.'.join(ip.split('.')[:-1])
        elif TRUSTHOST == 'ask':
            TRUSTHOST, okflag = notifier.get_text(
                'AFNI_TRUSTHOST', '.'.join(ip.split('.')[:-1]),
                parent=main_win)
            if not okflag:
                return

    # Check for a running AFNI process for boot_dir, identified by its
    # argument vector (the program is afni and the last argument is
    # boot_dir), not by a substring of the command line.
    for pid, argv in _list_proc_argv():
        if Path(argv[0]).name != 'afni' or argv[-1] != str(boot_dir):
            continue
        cmdl = shlex.join(argv)

        # Warning dialog
        ret = notifier.question(
            "Kill the existing AFNI process?", "Existing AFNI process",
            informative=f"pid={pid}: {cmdl}",
            buttons=("Yes", "Ignore", "Cancel"), default="Cancel",
            icon="warning")
        if ret == "Yes":
            os.kill(pid, 9)
        elif ret == "Cancel":
            return

    # Start AFNI in boot_dir, below main_win
    xpos = 0
    if main_win is not None:
        ypos = main_win.frameGeometry().height()+25  # 25: title bar height
    else:
        ypos = 0

    cmd = 'afni'
    if rt:
        cmd += f" -rt -DAFNI_TRUSTHOST={TRUSTHOST} -DAFNI_REALTIME_WRITEMODE=1"
    cmd += " -yesplugouts -DAFNI_IMSAVE_WARNINGS=NO"
    cmd += f" -com 'OPEN_WINDOW A geom=+{xpos}+{ypos}'"
    cmd += f" {boot_dir}"
    subprocess.call(shlex.split(cmd), cwd=boot_dir)


# %% excepthook ===============================================================
def excepthook(exc_type, exc_value, exc_tb):
    """Exception hook that also appends the traceback to rtpspy.error."""
    with open('rtpspy.error', 'a') as fd:
        traceback.print_exception(exc_type, exc_value, exc_tb, file=fd)

    traceback.print_exception(exc_type, exc_value, exc_tb)


# %% Names moved to rtpspy.ui =================================================
# The Qt dialogs and widgets are in rtpspy.ui.widgets. They are still
# importable from here (Qt is imported on first access).
_MOVED_TO_UI = ('DlgProgressBar', 'MatplotlibWindow')


def __getattr__(name):
    if name in _MOVED_TO_UI:
        from .ui import widgets
        return getattr(widgets, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
