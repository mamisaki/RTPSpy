#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time recording of the scanner TTL and physiological signals (cardiac and
respiration).
@author: mmisaki@libr.net

Model classes: NumatoGPIORecording, GEPhysioRecording, DummyRecording
View class: PlotTTLPhysio
Controller class: RtPhysio
"""

# %% import ===================================================================
from pathlib import Path
import os
import time
import traceback
from multiprocessing import Process, Lock, Queue, Pipe
from queue import Full, Empty
import re
import logging
import argparse
import warnings
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox
from tempfile import NamedTemporaryFile
from collections import deque
import json
import gc
import struct
import threading
import sys
import subprocess

import numpy as np
import pandas as pd
import serial
from serial import SerialException
try:
    from termios import error as TermiosError
except ImportError:  # Not POSIX
    TermiosError = OSError
from serial.tools.list_ports import comports
from scipy.interpolate import interp1d
from scipy.signal import lfilter, firwin, butter, sosfiltfilt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib as mpl

try:
    from rpc_socket_server import RPCSocketServer, RPCSocketCom, pack_data
    from window_layout import default_geometry, screen_area
    from alarm_sound import AlarmSound
    import gui_theme
    from gui_theme import COLORS
except ImportError:
    from .rpc_socket_server import RPCSocketServer, RPCSocketCom, pack_data
    from .window_layout import default_geometry, screen_area
    from .alarm_sound import AlarmSound
    from . import gui_theme
    from .gui_theme import COLORS

mpl.rcParams["font.size"] = 8


# %% Constants ================================================================
DEFAULT_SERIAL_TIMEOUT = 0.001
DEFAULT_SERIAL_BAUDRATE = 19200
DEFAULT_BUFFER_SIZE = 3600  # seconds
DEFAULT_SAMPLE_FREQ = 100  # Hz
PLOT_UPDATE_INTERVAL = 1.0 / 60  # 60 FPS
ADC_MAX_VALUE = 1024
YAXIS_ADJUST_RANGE = 25
MIN_YAXIS_RANGE = 50

# Errors of a serial device that is disconnected or reset: pyserial raises
# SerialException, but also OSError (e.g., in_waiting) and termios.error
# (e.g., reset_input_buffer).
SERIAL_ERRORS = (SerialException, OSError, TermiosError)

# Consecutive errors before the port is reopened, and the longest wait
# between reopen attempts (s)
N_ERR_REOPEN = 3
MAX_REOPEN_WAIT = 5.0

# Read timeout of the Numato poll loop (s). A response takes 0.4-0.9 ms
# (longest seen 1.2 ms; measured on five boards, 2026-11-09). After the
# timeout the command is sent again, so this is also the longest gap a lost
# response adds between TTL reads; it must stay well below the 10 ms TTL
# pulse.
NUMATO_POLL_TIMEOUT = 0.003


# %% create_temp_file =========================================================
def create_temp_file(prefix, dir_path="/dev/shm", delete=False):
    """Create a temporary file in dir_path (/dev/shm, i.e., in memory) if
    it is writable, otherwise in the default temporary directory.
    """
    if Path(dir_path).is_dir() and os.access(dir_path, os.W_OK):
        return NamedTemporaryFile(
            mode="w+b", prefix=prefix, dir=dir_path, delete=delete
        )
    else:
        return NamedTemporaryFile(prefix=prefix, delete=delete)


# %% physio_window_title ======================================================
def physio_window_title(device):
    """Title of the physio window with the recording device."""
    if device is None:
        return "Physio signals (no device)"
    return f"Physio signals ({device})"


# %% SharedMemoryRingBuffer ===================================================
class SharedMemoryRingBuffer:
    """Ring buffer implemented on a memory-mapped NumPy array for sharing data
    across processes.
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(
        self,
        length,
        data_file=None,
        cpos_file=None,
        initial_value=None,
        read_only=False,
    ):
        """Initialize SharedMemoryRingBuffer.

        Without data_file and cpos_file, new mmap files are created in
        /dev/shm if available, otherwise in the temporary directory.

        Parameters
        ----------
        length : int
            Buffer length.
        data_file : Path or str, optional
            Existing data mmap file to attach to (made by another process).
        cpos_file : Path or str, optional
            Existing current position mmap file to attach to.
        initial_value : float, optional
            Value to initialize the buffer data with. The default (None)
            keeps the file contents (zeros for new files).
        read_only : bool, optional
            Open the memory maps read-only.
        """
        self._logger = logging.getLogger("SharedMemoryRingBuffer")
        self.length = int(length)
        self.data_mmap_file = data_file
        self.cpos_mmap_file = cpos_file

        self.pid = None
        self._data = None
        self._cpos = None
        self._data_mmap_fd = None
        self._cpos_mmap_fd = None
        self._creator = False

        try:
            if data_file is not None or cpos_file is not None:
                # Attach to the buffer of another process. If its files are
                # missing, raise: a new empty buffer made here would be read
                # as if it were the recorder's (e.g., no physio data for the
                # respiration correction, without an error).
                self._load_existing_files(Path(data_file), Path(cpos_file))
            else:
                self._create_new_files()

            self._setup_memory_maps(
                initial_value=initial_value, read_only=read_only
            )
            self.pid = os.getpid()

        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            self._logger.error(errstr)

            # Ensure object is in a valid state even if initialization fails
            if self._data is None:
                self._data = np.array([initial_value] * self.length)
            if self._cpos is None:
                self._cpos = np.array([0], dtype=np.int64)
            raise e

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _load_existing_files(self, data_file, cpos_file):
        """Load existing mmap files."""
        if not Path(data_file).is_file():
            raise FileNotFoundError(f"Not found data_file: {data_file}")
        if not Path(cpos_file).is_file():
            raise FileNotFoundError(f"Not found cpos_file: {cpos_file}")

        self._data_mmap_fd = open(data_file, "r+b")
        self._cpos_mmap_fd = open(cpos_file, "r+b")
        self.data_mmap_file = Path(data_file)
        self.cpos_mmap_file = Path(cpos_file)
        self._creator = False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _create_new_files(self):
        """Create new temporary mmap files."""
        pid = os.getpid()
        self._data_mmap_fd = create_temp_file(f"rtmri_physio_{pid}_rbuffer_")
        self._cpos_mmap_fd = create_temp_file(
            f"rtmri_physio_{pid}_rbuffer_cpos_"
        )

        # Allocate space for data and position
        data_size = self.length * np.dtype(float).itemsize
        self._data_mmap_fd.write(b"\x00" * data_size)
        self._data_mmap_fd.flush()
        self.data_mmap_file = Path(self._data_mmap_fd.name)

        pos_size = np.dtype(np.int64).itemsize
        self._cpos_mmap_fd.write(b"\x00" * pos_size)
        self._cpos_mmap_fd.flush()
        self.cpos_mmap_file = Path(self._cpos_mmap_fd.name)

        self._creator = True

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _setup_memory_maps(self, initial_value=None, read_only=False):
        """Setup numpy memory maps for data and position."""
        mode = "r" if read_only else "r+"
        self._data = np.memmap(
            self._data_mmap_fd, dtype=float, mode=mode, shape=(self.length,)
        )
        self._cpos = np.memmap(
            self._cpos_mmap_fd, dtype=np.int64, mode=mode, shape=(1,)
        )

        if initial_value is not None and not read_only:
            self._data[:] = initial_value
            self._data.flush()
            self._cpos[0] = 0
            self._cpos.flush()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def append(self, x):
        """Append an element"""
        try:
            # Check if attributes exist
            if self._cpos is None or self._data is None:
                self._logger.error("The ring buffer is not initialized.")
                return

            cpos = self._cpos[0]
            self._data[cpos] = x
            self._cpos[0] = (cpos + 1) % self.length
            self._data.flush()
            self._cpos.flush()

        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            self._logger.error("Failed to append to ring buffer:\n" + errstr)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get(self):
        """Return all elements, oldest first."""
        try:
            # Check if attributes exist
            if self._cpos is None or self._data is None:
                self._logger.error("The ring buffer is not initialized.")
                return np.array([])

            cpos = self._cpos[0]
            if cpos == 0:
                return self._data.copy()
            else:
                data = self._data
                return np.concatenate([data[cpos:], data[:cpos]])

        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            self._logger.error("Failed to get ring buffer data:\n" + errstr)
            return np.array([])

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_tail(self, n):
        """Return the last n elements, in the order of append. Cheaper than
        get() when only recent data are needed (e.g., for the plot)."""
        try:
            if self._cpos is None or self._data is None:
                self._logger.error("The ring buffer is not initialized.")
                return np.array([])

            n = int(min(n, self.length))
            cpos = int(self._cpos[0])
            if n <= cpos:
                return self._data[cpos - n:cpos].copy()
            return np.concatenate(
                [self._data[self.length - (n - cpos):], self._data[:cpos]]
            )

        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            self._logger.error("Failed to get ring buffer data:\n" + errstr)
            return np.array([])

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def reset(self, initial_value=np.nan):
        """Reset the ring buffer: fill the data and reset the position.

        Parameters
        ----------
        initial_value : float, optional
            The value to fill the buffer with. Default is np.nan.
        """
        try:
            # Check if attributes exist
            if self._cpos is None or self._data is None:
                self._logger.error("The ring buffer is not initialized.")
                return

            # Reset position to start
            self._cpos[0] = 0

            # Fill buffer with initial value
            self._data[:] = initial_value

            # Flush changes to disk
            self._data.flush()
            self._cpos.flush()

        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            self._logger.error("Failed to reset ring buffer:\n" + errstr)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_property(self):
        """Arguments to attach to this buffer from another process
        (SharedMemoryRingBuffer(**kwds))."""
        kwds = {
            "length": self.length,
            "data_file": self.data_mmap_file,
            "cpos_file": self.cpos_mmap_file,
        }
        return kwds

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def validate(self, pid, read_only=False):
        """Return a buffer usable in process pid: self if it was made in
        pid, otherwise a new instance attached to the same files."""
        try:
            if pid == self.pid:
                return self

            self._logger.debug("Accessing shared memory on another process.")

            length = self.length
            data_file = self.data_mmap_file
            cpos_file = self.cpos_mmap_file
            new_access = SharedMemoryRingBuffer(
                length,
                data_file=data_file,
                cpos_file=cpos_file,
                read_only=read_only,
            )
            new_access._creator = False
            return new_access

        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            self._logger.error("Failed to validate shared memory:\n" + errstr)
            return None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def clear_buffer_files(self):
        """Delete the mmap files, if this instance created them."""
        try:
            if hasattr(self, "_creator") and self._creator:
                if (
                    hasattr(self, "data_mmap_file")
                    and self.data_mmap_file.is_file()
                ):
                    self.data_mmap_file.unlink()
                    self._logger.debug(
                        f"Deleted temporary file: {self.data_mmap_file}"
                    )

                if (
                    hasattr(self, "cpos_mmap_file")
                    and self.cpos_mmap_file.is_file()
                ):
                    self.cpos_mmap_file.unlink()
                    self._logger.debug(
                        f"Deleted temporary file: {self.cpos_mmap_file}"
                    )

                # Close file descriptors
                if (
                    hasattr(self, "_data_mmap_fd")
                    and self._data_mmap_fd is not None
                ):
                    self._data_mmap_fd.close()
                if (
                    hasattr(self, "_cpos_mmap_fd")
                    and self._cpos_mmap_fd is not None
                ):
                    self._cpos_mmap_fd.close()

        except Exception:
            # Called at cleanup; do not raise
            pass

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        if self._data is not None:
            del self._data
        if self._cpos is not None:
            del self._cpos
        if self._data_mmap_fd is not None:
            self._data_mmap_fd.close()
        if self._cpos_mmap_fd is not None:
            self._cpos_mmap_fd.close()


# %% fir_numtaps ==============================================================
def fir_numtaps(fs, sec=1.0):
    """Number of FIR taps for a filter of about `sec` seconds (odd number).

    The filters were designed with 41 taps at 40 Hz. Scaling the number of
    taps with the sampling frequency keeps the same filter length in seconds,
    and so the same frequency response, at any sampling frequency.
    """
    n = int(round(sec * float(fs)))
    return n + 1 if n % 2 == 0 else n


# %% Physio signal check ======================================================
# Signal check before saving (_save_physio_data), with thresholds per
# recording device. For each signal, the amplitude (std of the signal
# low-pass filtered at 3 Hz) and the periodicity (respiratory_periodicity,
# cardiac_periodicity) are both checked:
#   ok:        amplitude >= *_ok    and periodicity >= *_ac
#   weak:      amplitude >= *_floor and periodicity >= *_ac (a real but weak
#              signal, e.g., a loose belt or sensor)
#   irregular: amplitude >= *_ok    and periodicity <  *_ac (e.g., movement,
#              a clipped respiration signal, or a noisy cardiac input)
#   none:      otherwise
# The data are saved unless both signals are "none". For a subject, a
# warning dialog is shown when a signal is not "ok".
# The GE values were set from several thousand recorded runs of the GE
# physio output. The default (Numato, 10-bit ADC) keeps the former
# respiration check (amplitude >= 1). The respiration periodicity is not
# used for it, as it was low in a substantial part of valid recordings. The
# cardiac check uses the noise floor of a flat cardiac input.
SIGNAL_CHECK = {
    "GE": {"resp_ok": 30.0, "resp_floor": 1.0, "resp_ac": 0.35,
           "card_ok": 60.0, "card_floor": 20.0, "card_ac": 0.4},
    "default": {"resp_ok": 1.0, "resp_floor": 1.0, "resp_ac": 0.0,
                "card_ok": 5.0, "card_floor": 5.0, "card_ac": 0.4},
}


def lowpass_std(x, fs, cutoff=3.0):
    """Std of the signal low-pass filtered at `cutoff` Hz (zero phase),
    without the filter edges. 0 for a signal shorter than 3 filter
    lengths."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    numtaps = fir_numtaps(fs)
    if len(x) < 3 * numtaps:
        return 0.0
    b = firwin(numtaps=numtaps, cutoff=cutoff, window="hamming",
               pass_zero="lowpass", fs=fs)
    y = lfilter(b, 1, x)
    y = np.flipud(lfilter(b, 1, np.flipud(y)))
    return float(np.nanstd(y[numtaps:-numtaps]))


def cardiac_periodicity(x, fs, win_sec=10.0):
    """Periodicity of a cardiac signal: the median, over win_sec windows, of
    the autocorrelation peak at lags of 0.33-1.7 s (35-180 beats/min) of
    the signal band-passed at 0.6-3 Hz. About 0.8 for a pulse or ECG
    signal, about 0.2 for noise."""
    x = np.asarray(x, dtype=float)
    if len(x) < 4 * fs or np.all(np.isnan(x)):
        return 0.0
    x = np.nan_to_num(x - np.nanmean(x))
    sos = butter(2, [0.6, 3.0], btype="band", fs=fs, output="sos")
    y = sosfiltfilt(sos, x)
    w = int(min(win_sec * fs, len(y)))
    lo, hi = int(0.33 * fs), int(1.7 * fs)
    acs = []
    for s0 in range(0, len(y) - w + 1, w):
        seg = y[s0:s0 + w] - np.mean(y[s0:s0 + w])
        den = float(np.dot(seg, seg))
        if den <= 0:
            acs.append(0.0)
            continue
        ac = np.correlate(seg, seg, "full")[w - 1:] / den
        acs.append(float(np.max(ac[lo:min(hi, w)])))
    return float(np.median(acs)) if acs else 0.0


def respiratory_periodicity(x, fs, win_sec=20.0):
    """Periodicity of a respiration signal: the median, over win_sec
    windows, of the autocorrelation peak at lags of 1.5-12 s (5-40
    breaths/min) of the signal band-passed at 0.05-1 Hz. 0.4-0.85 for a
    breathing signal (also a weak one), <= 0.33 without a belt; low also
    for a clipped or moving signal. 20-s windows keep it high when the
    breathing rate changes slowly."""
    x = np.asarray(x, dtype=float)
    if len(x) < 15 * fs or np.all(np.isnan(x)):
        return 0.0
    x = np.nan_to_num(x - np.nanmean(x))
    sos = butter(2, [0.05, 1.0], btype="band", fs=fs, output="sos")
    y = sosfiltfilt(sos, x)
    w = int(min(win_sec * fs, len(y)))
    lo, hi = int(1.5 * fs), int(12 * fs)
    acs = []
    for s0 in range(0, len(y) - w + 1, w):
        seg = y[s0:s0 + w] - np.mean(y[s0:s0 + w])
        den = float(np.dot(seg, seg))
        if den <= 0:
            acs.append(0.0)
            continue
        ac = np.correlate(seg, seg, "full")[w - 1:] / den
        acs.append(float(np.max(ac[lo:min(hi, w)])))
    return float(np.median(acs)) if acs else 0.0


def check_physio_signals(card, resp, fs, device=None):
    """Signal check before saving. Return a dict with 'respiratory' and
    'cardiac' ("ok", "weak", "irregular" or "none") and the values."""
    th = SIGNAL_CHECK.get(device, SIGNAL_CHECK["default"])
    resp_std = lowpass_std(resp, fs)
    card_std = lowpass_std(card, fs)
    resp_ac = respiratory_periodicity(resp, fs)
    card_ac = cardiac_periodicity(card, fs)

    def state(amp, ac, ok, floor, ac_th):
        if amp >= ok:
            return "ok" if ac >= ac_th else "irregular"
        if amp >= floor and ac >= ac_th:
            return "weak"
        return "none"

    resp_st = state(resp_std, resp_ac, th["resp_ok"], th["resp_floor"],
                    th["resp_ac"])
    card_st = state(card_std, card_ac, th["card_ok"], th["card_floor"],
                    th["card_ac"])
    return {
        "respiratory": resp_st,
        "cardiac": card_st,
        "resp_std": round(resp_std, 2),
        "resp_periodicity": round(resp_ac, 3),
        "card_std": round(card_std, 2),
        "card_periodicity": round(card_ac, 3),
        "thresholds": dict(th),
    }


# %% load_bids_physio =========================================================
def load_bids_physio(tsv_file):
    """Read a BIDS physiological recording.

    Parameters
    ----------
    tsv_file : str or Path
        *_physio.tsv or *_physio.tsv.gz. The JSON sidecar with the same name
        (*_physio.json) must exist.

    Returns
    -------
    dict
        "SamplingFrequency" (float), "StartTime" (float), "Columns" (list),
        "cardiac" and "respiratory" (numpy arrays; zeros if the column is
        missing), and "json" (the sidecar contents).
        Columns are found by name, case-insensitively. At least one of
        "cardiac" and "respiratory" must exist.
    """
    tsv_file = Path(tsv_file)
    name = tsv_file.name
    if name.endswith(".tsv.gz"):
        stem = name[: -len(".tsv.gz")]
    elif name.endswith(".tsv"):
        stem = name[: -len(".tsv")]
    else:
        raise ValueError(f"{tsv_file} is not a .tsv or .tsv.gz file.")
    json_file = tsv_file.with_name(stem + ".json")
    if not json_file.is_file():
        raise FileNotFoundError(f"JSON sidecar {json_file} is not found.")

    with open(json_file, "r") as fid:
        meta = json.load(fid)
    for key in ("SamplingFrequency", "StartTime", "Columns"):
        if key not in meta:
            raise ValueError(f"{json_file} has no '{key}'.")

    columns = [str(c).lower() for c in meta["Columns"]]
    data = np.loadtxt(tsv_file, delimiter="\t", ndmin=2)
    if data.shape[1] != len(columns):
        raise ValueError(
            f"{tsv_file} has {data.shape[1]} columns, but 'Columns' in "
            f"{json_file.name} lists {len(columns)}: {meta['Columns']}"
        )
    if "cardiac" not in columns and "respiratory" not in columns:
        raise ValueError(
            f"Neither 'cardiac' nor 'respiratory' is in 'Columns' of "
            f"{json_file.name}: {meta['Columns']}"
        )

    out = {
        "SamplingFrequency": float(meta["SamplingFrequency"]),
        "StartTime": float(meta["StartTime"]),
        "Columns": list(meta["Columns"]),
        "json": meta,
    }
    for col in ("cardiac", "respiratory"):
        if col in columns:
            out[col] = data[:, columns.index(col)].astype(float)
        else:
            out[col] = np.zeros(len(data))
    return out


# %% find_scans ===============================================================
def find_scans(onsets, min_gap=3.0, min_pulses=3):
    """Split TTL onsets into scans at gaps.

    A gap longer than max(min_gap, 3 * median interval) separates scans.
    The TR is not needed. Groups with fewer than min_pulses TTLs (e.g., a
    stray pulse) are not returned.

    Returns
    -------
    list of (onset, last TTL time, number of TTLs)
    """
    on = np.sort(np.asarray(onsets, dtype=float))
    on = on[np.isfinite(on) & (on > 0)]
    if len(on) == 0:
        return []
    iv = np.diff(on)
    gap = max(min_gap, 3 * float(np.median(iv))) if len(iv) else min_gap
    starts = np.r_[0, np.where(iv > gap)[0] + 1]
    ends = np.r_[starts[1:], len(on)]
    scans = []
    for s0, e0 in zip(starts, ends):
        if e0 - s0 >= min_pulses:
            scans.append((float(on[s0]), float(on[e0 - 1]), int(e0 - s0)))
    return scans


# %% find_series_onset ========================================================
def find_series_onset(onsets, TR, block_gap=0.3, tol_TR=0.05, max_missed=3,
                      tol_int=0.3):
    """Find the first TTL of the current series by walking back from the
    latest TTL while the TTL pattern continues.

    Scanners send a TTL per volume or per slice. TTLs of a prescan or a
    previous scan can come shortly before the scan (e.g., GE prescans), so
    the first TTL after a gap > 1.5 TR can be the start of those TTLs, not of
    the scan. Walking back only while the pattern continues avoids this.
    Volumes before the steady state that send no TTL make a gap longer than
    TR before the first TTL, so the walk stops at the first TTL of the scan.
    A missed TTL in the scan also stops the walk (the onset is then late).

    Parameters
    ----------
    onsets : array
        TTL onset times (NaN is ignored).
    TR : float
        Repetition time (s) of the scan.

    Two patterns are handled:
    - Block: TTLs come in TR blocks separated by gaps > block_gap. Walk back
      while the previous block started one TR earlier (+- tol_TR * TR, plus
      max_missed slice intervals for missed first pulses).
    - Continuous: no gap in the last TR. Walk back while each interval is k
      times the median interval (k = 1-3, +- tol_int). A single pulse whose
      time is off (e.g., a delayed read) breaks two intervals; it is skipped
      if the interval over it is k times the median interval (k = 2-4).

    Returns
    -------
    float or None
        Onset time of the series. None if there is no TTL.
    """
    on = np.sort(np.asarray(onsets, dtype=float))
    on = on[~np.isnan(on)]
    if len(on) == 0:
        return None
    if len(on) == 1:
        return float(on[0])

    iv = np.diff(on)
    bstart = np.r_[0, np.where(iv > block_gap)[0] + 1]
    last_block = on[bstart[-1]:]
    slice_int = (
        float(np.median(np.diff(last_block))) if len(last_block) > 1 else 0.0
    )

    if on[-1] - on[bstart[-1]] < TR:
        # The last block is shorter than one TR: block pattern
        tol = tol_TR * TR + max_missed * slice_int
        k = len(bstart) - 1
        while k > 0 and \
                abs((on[bstart[k]] - on[bstart[k - 1]]) - TR) <= tol:
            k -= 1
        return float(on[bstart[k]])

    # Continuous pattern: walk back over intervals
    m = float(np.median(iv[-min(len(iv), 20):]))

    def fits(d, kmin, kmax):
        k = round(d / m)
        return kmin <= k <= kmax and abs(d - k * m) <= tol_int * m

    j = len(on) - 1
    while j > 0:
        if fits(on[j] - on[j - 1], 1, 3):
            j -= 1
        elif j > 1 and fits(on[j] - on[j - 2], 2, 4):
            j -= 2  # on[j - 1] is displaced
        else:
            break
    return float(on[j])


# %% _put_drop_oldest =========================================================
def _put_drop_oldest(que, item):
    """Put item in the queue. If the queue is full, drop the oldest item."""
    try:
        que.put_nowait(item)
    except Full:
        try:
            # Discard the oldest. A short wait, because a multiprocessing
            # queue can report Empty while its items are being flushed.
            que.get(timeout=0.01)
        except Empty:
            pass
        # Then put the new item (it was dropped too before)
        try:
            que.put_nowait(item)
        except Full:
            pass


# %% NumatoGPIORecording ======================================================
class NumatoGPIORecording:
    """
    Receive signals from the USB GPIO device, Numato Lab 8 Channel USB GPIO:
    https://numato.com/product/8-channel-usb-gpio-module-with-analog-inputs/
    IO0/DIO0: the scanner TTL signal.
    IO1/ADC1: the cardiac signal.
    IO2/ADC2: the respiration signal.
    The device is recognized as 'CDC RS-232 Emulation Demo' or
    'Numato Lab 8 Channel USB GPIO M'.
    """

    SUPPORT_DEVICES = [
        "CDC RS-232 Emulation Demo",
        "Numato Lab 8 Channel USB GPIO",
    ]
    TTL_CMD = b"gpio read 0\r"
    CARD_CMD = b"adc read 1\r"
    RESP_CMD = b"adc read 2\r"
    # Response: b"gpio read 0\n\r1\n\r>" or b"adc read 1\n\r512\n\r>"
    RESP_PAT = re.compile(rb"(?:gpio|adc) read (\d)\n\r(\d+)\n\r")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(
        self,
        ttl_onset_que=None,
        ttl_offset_que=None,
        physio_que=None,
        sample_freq=DEFAULT_SAMPLE_FREQ,
    ):
        """Initialize

        Parameters
        ----------
        ttl_onset_que, ttl_offset_que, physio_que : multiprocessing.Queue
           Queues for sharing the recorded data with the controller class.
        sample_freq : float
            Recording sampling frequency (Hz).

        The first available port with a device in SUPPORT_DEVICES is used.
        """
        self._logger = logging.getLogger("NumatoGPIORecording")

        # Set parameters
        self._ttl_onset_que = ttl_onset_que
        self._ttl_offset_que = ttl_offset_que
        self._physio_que = physio_que
        self._sample_freq = sample_freq
        self._queue_lock = Lock()

        # Get available serial ports
        self.dict_sig_sport = {}
        self.update_port_list()

        # Use the first available device
        if len(self.dict_sig_sport):
            self.sig_sport = list(self.dict_sig_sport.keys())[0]
        else:
            self.sig_sport = None
        self._sig_ser = None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ getter and setter methods +++
    @property
    def sig_sport(self):
        return self._sig_sport

    @sig_sport.setter
    def sig_sport(self, dev):
        if dev is not None:
            self.update_port_list()
            if dev not in self.dict_sig_sport:
                self._logger.error(f"{dev} is not available.")
                dev = None
        self._sig_sport = dev

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def update_port_list(self):
        """Get and sort available serial ports for supported devices."""
        self.dict_sig_sport = {}
        for pt in comports():
            if self._is_supported_device(pt.description):
                self.dict_sig_sport[pt.device] = pt.description

        # Sort by device name
        self.dict_sig_sport = dict(sorted(self.dict_sig_sport.items()))

        # Validate current device
        if hasattr(self, "_sig_sport") and self._sig_sport is not None:
            self.sig_sport = self._sig_sport

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _is_supported_device(self, description):
        """Check if device description matches any supported patterns."""
        return any(
            re.match(pattern, description) is not None
            for pattern in NumatoGPIORecording.SUPPORT_DEVICES
        )

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def is_device_available():
        """Check if a Numato device is available, without making an
        instance."""
        for pt in comports():
            if any(
                re.match(pattern, pt.description) is not None
                for pattern in NumatoGPIORecording.SUPPORT_DEVICES
            ):
                return True
        return False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def open_sig_port(self):
        """Open the serial port of the Numato device."""
        if self._sig_sport is None:
            self._logger.warning("No Numato GPIO device was found.")
            return False

        self._logger.debug(
            f"Attempting to open serial port: {self._sig_sport}")
        self._close_existing_port()

        if self._sig_sport not in self.dict_sig_sport:
            self._logger.error(f"{self._sig_sport} is not available.")
            return False

        try:
            # exclusive=True locks the port (flock), so the port cannot be
            # opened twice by processes that also open it with
            # exclusive=True, e.g., two RT-MRI instances. A process that
            # opened the port without the lock is not detected.
            self._sig_ser = serial.Serial(
                self._sig_sport,
                DEFAULT_SERIAL_BAUDRATE,
                timeout=DEFAULT_SERIAL_TIMEOUT,
                exclusive=True,
            )
            self._sig_ser.flushOutput()
            self._sig_ser.write(b"gpio clear 0\r")
            self._logger.info(f"Opened signal port {self._sig_sport}")
            return True

        except serial.serialutil.SerialException as e:
            if "exclusively lock" in str(e):
                self._logger.error(
                    f"Serial port {self._sig_sport} is already in use "
                    "by another process."
                )
            else:
                errstr = str(e) + "\n" + traceback.format_exc()
                self._logger.error(
                    f"Failed to open {self._sig_sport}: {errstr}"
                )

        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            self._logger.error(f"Failed to open {self._sig_sport}: {errstr}")

        self._sig_ser = None
        return False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _reopen_sig_port(self, is_stop):
        """Reopen the Numato port after device errors (e.g., the USB cable
        was pulled or the hub was reset). The device can come back on another
        /dev/ttyACM*, so the port list is searched again. Tries until it
        succeeds or is_stop() returns True.

        Returns the new serial port, or None if stopped.
        """
        try:
            if self._sig_ser is not None:
                self._sig_ser.close()
        except Exception:
            pass
        self._sig_ser = None

        wait = 0.5
        while not is_stop():
            ports = {
                pt.device: pt.description for pt in comports()
                if self._is_supported_device(pt.description)
            }
            if ports:
                self.dict_sig_sport = dict(sorted(ports.items()))
                if self._sig_sport not in ports:
                    self._sig_sport = sorted(ports)[0]
                if self.open_sig_port():
                    return self._sig_ser
            t_end = time.time() + wait
            while time.time() < t_end:
                if is_stop():
                    return None
                time.sleep(0.1)
            wait = min(wait * 2, MAX_REOPEN_WAIT)
        return None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _close_existing_port(self):
        """Close the serial port if it is open."""
        if self._sig_ser is not None and self._sig_ser.is_open:
            self._logger.debug(f"Closing serial port: {self._sig_sport}")
            self._sig_ser.close()
            self._sig_ser = None
            time.sleep(1)  # Give the device time to release the port
            self._logger.debug(f"Serial port closed: {self._sig_sport}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_config(self):
        """Return the current device configuration."""
        self.update_port_list()
        if self.dict_sig_sport and self.sig_sport:
            port_info = self.dict_sig_sport[self.sig_sport]
            dio_port = f"{self.sig_sport}:{port_info}"
        else:
            dio_port = "None"

        return {
            "IO port": dio_port,
            "IO port list": self.dict_sig_sport,
        }

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_config(self, conf):
        for lab, val in conf.items():
            if lab == "USB port":
                if val != "None":
                    port = val.split(":")[0]
                    self.sig_sport = port

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def read_signal_loop(self, cmd_pipe=None):
        """Read TTL (IO0) and physio (ADC1, ADC2) until QUIT is received."""
        if (
            self._ttl_onset_que is None
            or self._ttl_offset_que is None
            or self._physio_que is None
        ):
            self._logger.error("Recording queues are not set.")
            return

        if not self.open_sig_port():
            return

        self._queue_lock.acquire()
        self._logger.debug("Start recording in read_signal_loop.")

        def is_quit():
            if cmd_pipe is not None and cmd_pipe.poll():
                cmd = cmd_pipe.recv()
                self._logger.debug(f"Receive {cmd} in read_signal_loop.")
                if cmd == "QUIT":
                    return True
            return False

        self.poll_loop(is_quit, read_adc=True)

        if cmd_pipe is not None:
            cmd_pipe.send("END")
        self._queue_lock.release()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def poll_loop(self, is_stop, read_adc=True):
        """Polling loop of the Numato device.

        A command is sent as soon as the response to the previous one (ending
        with the prompt '>') arrives, without waiting for a read timeout. The
        Numato device answers a command in 0.4-0.9 ms, and sending several
        commands ahead does not increase the rate. So IO0 (TTL) is read about
        every 0.5-0.85 ms, except when ADC1 (cardiac) and ADC2 (respiration)
        are read at self._sample_freq. If no response arrives within
        NUMATO_POLL_TIMEOUT, the TTL command is sent again.

        Parameters
        ----------
        is_stop : callable
            Called at every command. The loop ends when it returns True.
        read_adc : bool
            Read ADC1 and ADC2 as physio signals. If False, only TTL is read.

        The time stamp of a TTL sample is the midpoint between sending the
        command and receiving the response. The time stamp of a physio sample
        is the midpoint between sending the ADC1 command and receiving the
        ADC2 response.
        """
        ser = self._sig_ser
        ser.timeout = NUMATO_POLL_TIMEOUT
        physio_rec_interval = 1.0 / float(self._sample_freq)
        next_rec = time.time() + physio_rec_interval

        ttl_state = 0
        card = np.nan
        t_card_send = None
        buf = b""
        cmd = None  # Command waiting for the response
        t_send = None
        n_err = 0  # Consecutive device errors
        while not is_stop():
            try:
                if cmd is None:
                    # (Re)start the command cycle
                    ser.reset_input_buffer()
                    buf = b""
                    cmd = NumatoGPIORecording.TTL_CMD
                    ser.write(cmd)
                    t_send = time.time()

                data = ser.read(max(ser.in_waiting, 1))
                t_recv = time.time()
                if not data:
                    # No response within the timeout. Send again.
                    cmd = None
                    continue

                buf += data
                # A late response to an earlier command (sent again after a
                # timeout) is skipped, and the response to this command is
                # waited for. Taking it as this command's response would
                # shift every later response by one command, e.g., the
                # cardiac command would get the TTL response and every
                # physio sample would be NaN. A late response to the same
                # command (TTL after TTL) cannot be told apart and is used;
                # it is at most NUMATO_POLL_TIMEOUT old.
                resp = None
                while b">" in buf:
                    resp, buf = buf.split(b">", 1)
                    ma = NumatoGPIORecording.RESP_PAT.search(resp)
                    if ma is None or cmd.startswith(
                            resp[ma.start():ma.start(1) + 1]):
                        break
                    resp = None
                if resp is None:
                    continue
                resp_cmd, resp_t_send = cmd, t_send

                # Send the next command before processing the response
                if resp_cmd == NumatoGPIORecording.CARD_CMD:
                    cmd = NumatoGPIORecording.RESP_CMD
                elif read_adc and t_recv >= next_rec:
                    cmd = NumatoGPIORecording.CARD_CMD
                    next_rec += physio_rec_interval
                    if next_rec < t_recv:
                        # Too late. Restart the schedule from now.
                        next_rec = t_recv + physio_rec_interval
                else:
                    cmd = NumatoGPIORecording.TTL_CMD
                ser.write(cmd)
                t_send = time.time()

            except SERIAL_ERRORS as e:
                cmd = None
                n_err += 1
                if n_err < N_ERR_REOPEN:
                    self._logger.debug(f"Error reading signal: {e}")
                    time.sleep(0.01)
                    continue
                # The device is gone or reset: reopen the port
                self._logger.warning(
                    f"Numato device error ({e}). TTLs"
                    + (" and physio signals" if read_adc else "")
                    + " are not recorded until it is reconnected."
                )
                ser = self._reopen_sig_port(is_stop)
                if ser is None:
                    break  # Stopped while reconnecting
                ser.timeout = NUMATO_POLL_TIMEOUT
                self._logger.warning(
                    f"Numato device reconnected on {self._sig_sport}."
                )
                n_err = 0
                next_rec = time.time() + physio_rec_interval
                continue

            n_err = 0
            # ma is None for a broken response. A response to another command
            # was skipped above, so it is never read as this one (e.g.,
            # "adc read 1 ... 512" as TTL 0, a false TTL offset and onset).
            ma = NumatoGPIORecording.RESP_PAT.search(resp)
            if resp_cmd == NumatoGPIORecording.TTL_CMD:
                if ma is None:
                    continue  # Keep the TTL state
                ttl = int(ma.group(2) == b"1")
                tstamp_ttl = (resp_t_send + t_recv) / 2
                if ttl == 1 and ttl_state == 0:
                    _put_drop_oldest(self._ttl_onset_que, tstamp_ttl)
                elif ttl == 0 and ttl_state == 1:
                    _put_drop_oldest(self._ttl_offset_que, tstamp_ttl)
                ttl_state = ttl

            elif resp_cmd == NumatoGPIORecording.CARD_CMD:
                card = float(ma.group(2)) if ma else np.nan
                t_card_send = resp_t_send

            elif resp_cmd == NumatoGPIORecording.RESP_CMD:
                resp_val = float(ma.group(2)) if ma else np.nan
                tstamp_physio = (t_card_send + t_recv) / 2
                _put_drop_oldest(
                    self._physio_que, (tstamp_physio, card, resp_val)
                )

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        try:
            self._logger.debug("Deleting NumatoGPIORecording instance.")
            if hasattr(self, "_sig_ser") and self._sig_ser:
                self._sig_ser.close()
                self._sig_ser = None
        except Exception:
            pass  # Called at cleanup; do not raise


# %% _SequenceClock ===========================================================
class _SequenceClock:
    """Time stamps of GE physio packets from their sequence numbers.

    GE sends a packet every 5 ms by its own clock, and each packet has a
    16-bit sequence number. The arrival time of a packet, estimated from the
    read time and the number of packets still waiting in the buffer, has an
    error of about +-2.5 ms. A line, time = a + b * packet count, is fitted to
    the arrival times of the last window_sec seconds, and the time stamp is
    taken from the line. This removes the packet-to-packet jitter and follows
    the rate difference between the GE clock and the PC clock (30-60 ppm,
    about 0.1-0.2 s per hour).
    Not corrected: a constant latency (inside GE, serial transfer), and a
    slow offset of up to about +-2.5 ms. The arrival estimate drifts slowly
    within one packet period (5 ms) and then jumps back by one period, so the
    true offset is known only within one period. A long window averages this.

    Packets arriving more than reset_err seconds off the line (e.g., after a
    stall of this process) are stamped from the line and not used for the
    fit. If this continues for refit_sec, the fit is reset. The fit is also
    reset at once when the sequence number jumps ahead of the arrival time
    by more than reset_err, e.g., when GE restarts the counter.
    """

    def __init__(self, period, window_sec=120.0, refit_sec=1.0, min_sec=2.0,
                 reset_err=0.05):
        self.period = period
        self._maxlen = int(window_sec / period)
        self._refit_n = int(refit_sec / period)
        self._min_n = int(min_sec / period)
        self._reset_err = reset_err
        self.reset()

    def reset(self):
        self._prev_seq = None
        self._prev_t = None
        self._count = 0
        self._cnt = deque(maxlen=self._maxlen)
        self._t = deque(maxlen=self._maxlen)
        self._fit = None
        self._since_fit = 0
        self._n_out = 0

    def stamp(self, seq, t_arrival):
        """Return (time stamp, number of lost packets before this one).
        The time stamp is None for a duplicate packet."""
        lost = 0
        if self._prev_seq is not None:
            d = (seq - self._prev_seq) % 65536
            if d == 0:
                return None, 0
            if d * self.period - (t_arrival - self._prev_t) > \
                    self._reset_err:
                # The counter jumped ahead of the time (e.g., GE restarted
                # the counter). Lost packets do not cause this, because
                # their time also passes.
                self.reset()
                d = 1
            else:
                lost = d - 1
            self._count += d
        self._prev_seq = seq
        self._prev_t = t_arrival

        if self._fit is not None:
            pred = self._fit[0] + self._fit[1] * self._count
            if abs(t_arrival - pred) > self._reset_err:
                self._n_out += 1
                if self._n_out < self._refit_n:
                    return pred, lost
                # The line does not fit any more. Start again.
                self.reset()
                self._prev_seq = seq
                self._prev_t = t_arrival
            else:
                self._n_out = 0

        self._cnt.append(self._count)
        self._t.append(t_arrival)
        self._since_fit += 1
        if len(self._cnt) >= self._min_n and \
                (self._fit is None or self._since_fit >= self._refit_n):
            c = np.array(self._cnt, dtype=float)
            t = np.array(self._t)
            b, a = np.polyfit(c - c[0], t - t[0], 1)
            self._fit = (t[0] + a - b * c[0], b)
            self._since_fit = 0

        if self._fit is None:
            return t_arrival, lost
        return self._fit[0] + self._fit[1] * self._count, lost


# %% GEPhysioRecording ========================================================
class GEPhysioRecording:
    """
    Receive the GE physio signals from a serial port, and the TTL signal from
    the Numato GPIO device (IO0/DIO0, as NumatoGPIORecording).

    The GE MRI system sends physio data from its serial port (115200 baud) as
    12-byte packets at 200 Hz:
        struct '<Hhhhhh' = (sequence number, -, -, respiration, cardiac,
                            checksum)
    The checksum is the sum of the first 10 bytes.
    A moving average of MOVING_AVERAGE packets is taken at every packet, and
    every DECIMATE-th value is recorded, so the recording frequency is
    200 / 2 = 100 Hz. The time stamp of a value is the center of the averaged
    packets. Packet times are taken from the sequence numbers
    (_SequenceClock).

    The packet format was taken from rt_physio_ge.py, which is kept as a
    record of the original implementation.
    """

    PHYSIO_BAUDRATE = 115200
    PHYSIO_PACKET_FREQ = 200  # Hz
    MOVING_AVERAGE = 3
    DECIMATE = 2
    SAMPLE_FREQ = PHYSIO_PACKET_FREQ / DECIMATE  # 100 Hz
    PACKET_SIZE = 12
    PACKET_FORMAT = "<Hhhhhh"
    SIGNAL_RANGE = (-32768, 32767)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(
        self,
        ttl_onset_que=None,
        ttl_offset_que=None,
        physio_que=None,
        physio_sport=None,
    ):
        """Initialize

        Parameters
        ----------
        ttl_onset_que, ttl_offset_que, physio_que : multiprocessing.Queue
           Queues for sharing the recorded data with the controller class.
        physio_sport : str, optional
            Serial port of the GE physio output.
            The default is None, in which case the first non-Numato port that
            sends valid GE packets is used.
        """
        self._logger = logging.getLogger("GEPhysioRecording")

        self._ttl_onset_que = ttl_onset_que
        self._ttl_offset_que = ttl_offset_que
        self._physio_que = physio_que
        self._queue_lock = Lock()

        if physio_sport is None:
            physio_sport = GEPhysioRecording.find_physio_port()
            if physio_sport is None:
                ports = GEPhysioRecording.physio_port_candidates(
                    with_description_only=True
                )
                if len(ports):
                    physio_sport = ports[0]
                    self._logger.warning(
                        "No GE physio packets were found on any serial port. "
                        f"Using {physio_sport}."
                    )
        self.physio_sport = physio_sport
        self._physio_ser = None

        # TTL is read with the Numato device
        self._ttl_rec = NumatoGPIORecording(
            ttl_onset_que=ttl_onset_que,
            ttl_offset_que=ttl_offset_que,
        )

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def physio_port_candidates(with_description_only=False):
        """Serial ports that are not Numato GPIO devices.

        Ports with a device description (e.g., USB-serial adapters) come
        first. Legacy ports without hardware info (description 'n/a', e.g.,
        /dev/ttyS*) come last, or are excluded if with_description_only.
        """
        ports = []
        for pt in comports():
            if any(
                re.match(pattern, pt.description) is not None
                for pattern in NumatoGPIORecording.SUPPORT_DEVICES
            ):
                continue
            no_desc = pt.description in ("", "n/a")
            if with_description_only and no_desc:
                continue
            ports.append((no_desc, pt.device))
        return [dev for _, dev in sorted(ports)]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def is_packet_good(packet):
        """True if packet has the packet size and a valid checksum."""
        if len(packet) != GEPhysioRecording.PACKET_SIZE:
            return False
        chksum = struct.unpack("<h", bytes(packet[10:12]))[0]
        return chksum == sum(packet[:10])

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def probe_physio_port(port, probe_sec=0.5, n_good=3):
        """Check whether the port sends GE physio packets.

        Returns True if n_good consecutive valid packets with consecutive
        sequence numbers are received within probe_sec seconds.
        """
        try:
            # exclusive=True: a port used by another RT-MRI process is not
            # touched (open fails).
            ser = serial.Serial(
                port,
                GEPhysioRecording.PHYSIO_BAUDRATE,
                timeout=0.05,
                exclusive=True,
            )
        except Exception:
            return False

        psize = GEPhysioRecording.PACKET_SIZE
        buf = bytearray()
        n_seq = 0
        prev_seqn = None
        try:
            ser.reset_input_buffer()
            end_t = time.time() + probe_sec
            while time.time() < end_t:
                buf += ser.read(psize)
                while len(buf) >= psize:
                    if not GEPhysioRecording.is_packet_good(buf[:psize]):
                        del buf[0]
                        n_seq = 0
                        prev_seqn = None
                        continue
                    seqn = struct.unpack("<H", bytes(buf[:2]))[0]
                    del buf[:psize]
                    if prev_seqn is not None and \
                            seqn == (prev_seqn + 1) % 65536:
                        n_seq += 1
                    else:
                        n_seq = 1
                    prev_seqn = seqn
                    if n_seq >= n_good:
                        return True
        except Exception:
            return False
        finally:
            ser.close()

        return False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def find_physio_port():
        """Return the first port that sends GE physio packets, or None."""
        for port in GEPhysioRecording.physio_port_candidates():
            if GEPhysioRecording.probe_physio_port(port):
                return port
        return None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def is_device_available(probe=True):
        """Check if the GE physio port is available.

        Parameters
        ----------
        probe : bool
            If True, a port must send valid GE packets. If False, any
            non-Numato serial port is accepted.
        """
        if probe:
            return GEPhysioRecording.find_physio_port() is not None
        else:
            return (
                len(
                    GEPhysioRecording.physio_port_candidates(
                        with_description_only=True
                    )
                )
                > 0
            )

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _reopen_physio_port(self, is_stop):
        """Reopen the GE physio port after device errors. If the port is
        gone (e.g., a USB serial adapter came back under another name), the
        port that sends GE packets is searched. Tries until it succeeds or
        is_stop() returns True. Returns True when reopened."""
        try:
            if self._physio_ser is not None:
                self._physio_ser.close()
        except Exception:
            pass
        self._physio_ser = None

        wait = 0.5
        while not is_stop():
            if self.physio_sport is None or \
                    not Path(self.physio_sport).exists():
                port = GEPhysioRecording.find_physio_port()
                if port is not None:
                    self.physio_sport = port
            if self.physio_sport is not None and \
                    Path(self.physio_sport).exists() and \
                    self.open_physio_port():
                return True
            t_end = time.time() + wait
            while time.time() < t_end:
                if is_stop():
                    return False
                time.sleep(0.1)
            wait = min(wait * 2, MAX_REOPEN_WAIT)
        return False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def open_physio_port(self):
        """Open the serial port of the GE physio output."""
        if self.physio_sport is None:
            self._logger.error("No serial port was found for GE physio.")
            return False

        try:
            # exclusive=True locks the port (flock). See
            # NumatoGPIORecording.open_sig_port.
            self._physio_ser = serial.Serial(
                self.physio_sport,
                GEPhysioRecording.PHYSIO_BAUDRATE,
                timeout=0.01,
                exclusive=True,
            )
            self._physio_ser.reset_input_buffer()
            self._logger.info(f"Opened GE physio port {self.physio_sport}")
            return True

        except Exception as e:
            if "exclusively lock" in str(e):
                self._logger.error(
                    f"Serial port {self.physio_sport} is already in use "
                    "by another process."
                )
            else:
                errstr = str(e) + "\n" + traceback.format_exc()
                self._logger.error(
                    f"Failed to open {self.physio_sport}: {errstr}"
                )

        self._physio_ser = None
        return False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_config(self):
        return {
            "GE physio port": str(self.physio_sport),
            "IO port": self._ttl_rec.get_config()["IO port"],
        }

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_config(self, conf):
        self._ttl_rec.set_config(conf)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def read_signal_loop(self, cmd_pipe=None):
        """Read GE physio packets, and TTL in a thread, until QUIT is
        received."""
        if (
            self._ttl_onset_que is None
            or self._ttl_offset_que is None
            or self._physio_que is None
        ):
            self._logger.error("Recording queues are not set.")
            return

        if not self.open_physio_port():
            return

        self._queue_lock.acquire()
        self._logger.debug("Start recording in read_signal_loop.")

        # TTL is read by the Numato device in a thread
        stop_event = threading.Event()
        ttl_thread = None
        if self._ttl_rec.open_sig_port():
            ttl_thread = threading.Thread(
                target=self._ttl_rec.poll_loop,
                args=(stop_event.is_set,),
                kwargs={"read_adc": False},
                daemon=True,
            )
            ttl_thread.start()
        else:
            self._logger.warning(
                "No Numato GPIO device for TTL. Only physio is recorded."
            )

        psize = GEPhysioRecording.PACKET_SIZE
        pck_intv = 1.0 / GEPhysioRecording.PHYSIO_PACKET_FREQ
        n_ave = GEPhysioRecording.MOVING_AVERAGE
        clock = _SequenceClock(pck_intv)
        ave_buf = deque(maxlen=n_ave)  # (time stamp, card, resp)
        n_since_out = 0
        buf = bytearray()
        n_skip_bytes = 0
        n_lost = 0
        n_err = 0  # Consecutive device errors
        quit_received = False

        def is_quit():
            nonlocal quit_received
            if cmd_pipe is not None and cmd_pipe.poll():
                cmd = cmd_pipe.recv()
                self._logger.debug(f"Receive {cmd} in read_signal_loop.")
                if cmd == "QUIT":
                    quit_received = True
            return quit_received

        while not is_quit():
            try:
                data = self._physio_ser.read(
                    max(self._physio_ser.in_waiting, 1)
                )
            except SERIAL_ERRORS as e:
                n_err += 1
                if n_err < N_ERR_REOPEN:
                    self._logger.debug(f"Error reading GE physio: {e}")
                    time.sleep(0.1)
                    continue
                self._logger.warning(
                    f"GE physio port error ({e}). Physio signals are not "
                    "recorded until the port is available again."
                )
                if not self._reopen_physio_port(is_quit):
                    break  # Stopped while reconnecting
                self._logger.warning(
                    f"GE physio port reopened ({self.physio_sport}).")
                n_err = 0
                buf = bytearray()
                clock.reset()
                ave_buf.clear()
                continue
            n_err = 0
            tnow = time.time()
            if not data:
                continue
            buf += data

            while len(buf) >= psize:
                if not GEPhysioRecording.is_packet_good(buf[:psize]):
                    # Skip one byte to find the packet boundary
                    del buf[0]
                    n_skip_bytes += 1
                    continue

                seqn, _, _, resp, card, _ = struct.unpack(
                    GEPhysioRecording.PACKET_FORMAT, bytes(buf[:psize])
                )
                del buf[:psize]

                # Arrival time of this packet, estimated from the packets
                # still in buf, then corrected by the sequence clock.
                t_arrival = tnow - (len(buf) // psize) * pck_intv
                try:
                    tstamp, lost = clock.stamp(seqn, t_arrival)
                except Exception:
                    # Do not stop the recording for an error in the clock
                    self._logger.error(
                        "GE physio clock error. Resetting the clock.\n"
                        + traceback.format_exc()
                    )
                    clock.reset()
                    continue
                if tstamp is None:
                    continue  # Duplicate packet
                n_lost += lost

                ave_buf.append((tstamp, card, resp))
                n_since_out += 1
                if len(ave_buf) == n_ave and \
                        n_since_out >= GEPhysioRecording.DECIMATE:
                    vals = np.array(ave_buf)
                    _put_drop_oldest(
                        self._physio_que,
                        (vals[:, 0].mean(), vals[:, 1].mean(),
                         vals[:, 2].mean()),
                    )
                    n_since_out = 0

            if n_skip_bytes > 0 or n_lost > 0:
                self._logger.debug(
                    f"GE physio: {n_skip_bytes} bytes skipped to synchronize,"
                    f" {n_lost} packets lost."
                )
                n_skip_bytes = 0
                n_lost = 0

        # --- end loop ---
        stop_event.set()
        if ttl_thread is not None:
            ttl_thread.join(1)
        try:
            self._physio_ser.close()
        except Exception:
            pass
        self._physio_ser = None
        if cmd_pipe is not None:
            cmd_pipe.send("END")
        self._queue_lock.release()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        try:
            if hasattr(self, "_physio_ser") and self._physio_ser:
                self._physio_ser.close()
                self._physio_ser = None
        except Exception:
            pass


# %% DummyRecording ===========================================================
class DummyRecording:
    """Dummy recording device that plays signals from files. TTL pulses
    are given by the PULSE command."""

    def __init__(
        self,
        ttl_onset_que=None,
        ttl_offset_que=None,
        physio_que=None,
        sim_card_f=None,
        sim_resp_f=None,
        sample_freq=40,
        sim_bids_f=None,
    ):
        """sim_bids_f: BIDS *_physio.tsv[.gz] file. If given, it is used
        instead of sim_card_f and sim_resp_f, and the sampling frequency is
        taken from its JSON sidecar."""
        self._logger = logging.getLogger("DummyRecording")

        # Set parameters
        self._ttl_onset_que = ttl_onset_que
        self._ttl_offset_que = ttl_offset_que
        self._physio_que = physio_que
        self._sample_freq = sample_freq
        self._sim_card = None
        self._sim_resp = None
        self._queue_lock = Lock()

        if sim_bids_f is not None:
            self.set_sim_bids(sim_bids_f)
        else:
            self.set_sim_data(sample_freq, sim_card_f, sim_resp_f)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_sim_bids(self, sim_bids_f):
        """Set the dummy signals from a BIDS physio file."""
        try:
            bids = load_bids_physio(sim_bids_f)
        except Exception as e:
            self._logger.error(f"Error reading {sim_bids_f}: {e}")
            self.set_sim_data(self._sample_freq)
            return
        self._sample_freq = bids["SamplingFrequency"]
        self._sim_card = bids["cardiac"]
        self._sim_resp = bids["respiratory"]
        self._sim_data_len = len(self._sim_card)
        self._sim_data_pos = 0
        self._logger.info(
            f"Dummy signals from {Path(sim_bids_f).name}: "
            f"{self._sim_data_len} samples at {self._sample_freq} Hz"
        )

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_sim_data(self, sample_freq, sim_card_f=None, sim_resp_f=None):
        """Set the dummy signals from text files (one value per line). A
        missing signal is zeros; without both, the signals are constant."""
        sim_data_len = np.inf

        if sim_card_f is not None:
            sim_card_f = Path(sim_card_f)
            if not sim_card_f.is_file():
                self._logger.error(
                    "File for the cardiac dummy signal not found: "
                    f"{sim_card_f}"
                )
            else:
                try:
                    self._sim_card = np.loadtxt(sim_card_f)
                    sim_data_len = min(sim_data_len, len(self._sim_card))
                except Exception as e:
                    errstr = str(e) + "\n" + traceback.format_exc()
                    self._logger.error(f"Error reading {sim_card_f}: {errstr}")

        if sim_resp_f is not None:
            sim_resp_f = Path(sim_resp_f)
            if not sim_resp_f.is_file():
                self._logger.error(
                    "File for the respiration dummy signal not found: "
                    f"{sim_resp_f}"
                )
            else:
                try:
                    self._sim_resp = np.loadtxt(sim_resp_f)
                    sim_data_len = min(sim_data_len, len(self._sim_resp))
                except Exception as e:
                    errstr = str(e) + "\n" + traceback.format_exc()
                    self._logger.error(f"Error reading {sim_resp_f}: {errstr}")

        if not np.isinf(sim_data_len):
            self._sim_data_len = sim_data_len
            if self._sim_card is None:
                self._sim_card = np.zeros(self._sim_data_len)
            if self._sim_resp is None:
                self._sim_resp = np.zeros(self._sim_data_len)
        else:
            self._sim_card = np.ones(1)
            self._sim_resp = np.ones(1)
            self._sim_data_len = 1

        self._sim_data_pos = 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def read_signal_loop(self, cmd_pipe=None):
        """Put the dummy signals at the sampling frequency, and TTL pulses
        on PULSE commands, until QUIT is received."""
        if (
            self._ttl_onset_que is None
            or self._ttl_offset_que is None
            or self._physio_que is None
        ):
            self._logger.error("Recording queues are not set.")
            return

        self._queue_lock.acquire()

        self._logger.debug("Start recording in read_signal_loop.")

        physio_rec_interval = 1.0 / self._sample_freq
        # Recent read times, to start each read earlier by their mean
        rec_delays = deque(maxlen=10)
        rec_delay = 0
        next_rec = time.time() + physio_rec_interval
        st_physio_read = 0
        tstamp_physio = None
        while True:
            if time.time() >= next_rec - rec_delay:
                st_physio_read = time.time()
                if self._sim_card is not None:
                    card = self._sim_card[self._sim_data_pos]
                else:
                    card = 1

                if self._sim_resp is not None:
                    resp = self._sim_resp[self._sim_data_pos]
                else:
                    resp = 1

                tstamp_physio = time.time()
                _put_drop_oldest(self._physio_que,
                                 (tstamp_physio, card, resp))

                self._sim_data_pos += 1
                self._sim_data_pos %= self._sim_data_len

                rec_delays.append(time.time() - st_physio_read)
                rec_delay = np.mean(rec_delays) if rec_delays else 0.0
                next_rec += physio_rec_interval

            if cmd_pipe is not None and cmd_pipe.poll():
                cmd = cmd_pipe.recv()
                self._logger.debug(f"Receive {cmd} in read_signal_loop.")
                if cmd == "QUIT":
                    cmd_pipe.send("END")
                    break
                elif cmd == "PULSE" or (
                    type(cmd) is tuple and cmd[0] == "PULSE"
                ):
                    # Add TTL pulse. ("PULSE", t): pulse at time t
                    t_on = cmd[1] if type(cmd) is tuple else time.time()
                    _put_drop_oldest(self._ttl_onset_que, t_on)

                    if type(cmd) is not tuple:
                        time.sleep(0.001)
                    _put_drop_oldest(
                        self._ttl_offset_que,
                        t_on + 0.001 if type(cmd) is tuple else time.time())

            time.sleep(0.0005)

        self._queue_lock.release()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_config(self, config):
        self._logger.debug(f"Set config: {config}")
        self._config = config


# %% PlotTTLPhysio ============================================================
class PlotTTLPhysio:
    """View class: plot window of the TTL and physio signals. It runs in its
    own process (run) and reads the signals from the shared ring buffers.
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(
        self,
        controller,
        geometry=None,
        signal_freq=DEFAULT_SAMPLE_FREQ,
        plot_len_sec=10,
        buf_len_sec=3600,
        disable_close=False,
        signal_range=(0, ADC_MAX_VALUE),
    ):
        self._logger = logging.getLogger("PlotTTLPhysio")
        self.controller = controller

        # --- Initialize parameters ---
        self.signal_freq = signal_freq
        self.signal_range = signal_range  # (min, max) of the signal values
        self._plot_len_sec = plot_len_sec
        self._buf_len_sec = buf_len_sec
        self._is_scanning = False
        self._cmd_pipe = None
        # Redraw 6 times per second: smooth enough for the physio signals,
        # with less CPU than 10 times
        self._timer_interval_ms = int(1000 / 6)
        self._rbuf_lock = None
        self._rbuf = None

        # --- Plot window ---
        self._plt_root = tk.Tk()
        gui_theme.apply_tk(self._plt_root)
        gui_theme.apply_mpl()
        self._disable_close = disable_close

        self._plt_root.title(
            physio_window_title(self.controller._recorder_type))
        if geometry is None:
            geometry = default_geometry("physio")
        self.set_position(geometry)

        # Set the margins in inches
        self._left_margin_inch = 0.42
        self._right_margin_inch = 0.4
        self._top_margin_inch = 0.1
        self._bottom_margin_inch = 0.38

        # Initialize the plot (packed after the button below)
        plot_widget = self.init_plot()
        self.reset_plot()

        self.config_win = None

        # Dump button
        dump_dur = str(timedelta(seconds=self._buf_len_sec)).split(".")[0]
        dump_button = tk.Button(
            self._plt_root,
            text=f"dump ({dump_dur})",
            command=self.dump_data,
            font=("Helvetica", 10),
        )
        # Pack the button before the plot, so that it keeps its space and
        # the plot takes the rest.
        dump_button.pack(side=tk.BOTTOM, anchor=tk.SE)
        plot_widget.pack(side=tk.TOP, fill="both", expand=True)

        self._resize_debounce_id = None
        self._plt_root.bind("<Configure>", self.update_plot_size)
        self.update_plot_size(None)

        # Connect WM_DELETE_WINDOW event to self.on_closing
        self._plt_root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_position(self, geometry):
        self._plt_root.geometry(geometry)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def init_plot(self):
        self._plot_fig = Figure(figsize=(6, 4.2))
        self._canvas = FigureCanvasTkAgg(self._plot_fig, master=self._plt_root)
        gui_theme.fix_mpl_canvas(self._canvas)
        self._ax_ttl, self._ax_card, self._ax_card_filtered, self._ax_resp = (
            self._plot_fig.subplots(4, 1)
        )

        self._plot_fig.subplots_adjust(
            left=0.15, bottom=0.1, right=0.98, top=0.95, hspace=0.35
        )

        return self._canvas.get_tk_widget()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def reset_plot(self):
        self._buf_size = int(np.round(self._plot_len_sec * self.signal_freq))
        sig_xi = np.arange(self._buf_size) * 1.0 / self.signal_freq

        # Set TTL axis
        self._ax_ttl.clear()
        self._ax_ttl.set_ylabel("Scanner\nPulse", fontsize=9)
        zeros_data = np.zeros(self._buf_size)
        self._ln_ttl = self._ax_ttl.plot(
            sig_xi, zeros_data, "-", color=COLORS["green"]
        )
        self._ax_ttl.set_xlim(sig_xi[0], sig_xi[-1])
        self._ax_ttl.set_ylim((-0.1, 1.1))
        self._ax_ttl.set_yticks((0, 1))
        self._ax_ttl.yaxis.set_ticks_position("right")

        # Set card axis
        self._ax_card.clear()
        self._ax_card.set_ylabel("Cardiogram", fontsize=9)
        self._ln_card = self._ax_card.plot(
            sig_xi, np.zeros(self._buf_size), "-", color=COLORS["cyan"]
        )
        self._ax_card.set_xlim(sig_xi[0], sig_xi[-1])
        self._ax_card.yaxis.set_ticks_position("right")

        # Set filtered card axis
        self._ax_card_filtered.clear()
        self._ax_card_filtered.set_ylabel("Cardiogram\n(filtered)", fontsize=9)
        self._ln_card_filtered = self._ax_card_filtered.plot(
            sig_xi, np.zeros(self._buf_size), "-", color=COLORS["purple"]
        )
        self._ax_card_filtered.set_xlim(sig_xi[0], sig_xi[-1])
        self._ax_card_filtered.yaxis.set_ticks_position("right")

        # Set Resp axis
        self._ax_resp.clear()
        self._ax_resp.set_ylabel("Respiration", fontsize=9)
        self._ln_resp = self._ax_resp.plot(
            sig_xi, np.zeros(self._buf_size), "-", color=COLORS["cyan"]
        )
        self._ax_resp.set_xlim(sig_xi[0], sig_xi[-1])
        self._ax_resp.yaxis.set_ticks_position("right")
        self._ax_resp.set_xlabel("second", fontsize=9)
        for ax in (self._ax_ttl, self._ax_card, self._ax_card_filtered,
                   self._ax_resp):
            ax.tick_params(labelsize=8)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run(self, cmd_pipe, rbuf_lock, rbuf, dump_pipe=None,
            reply_pipe=None):
        """Run the plot window (the target of the plot process)."""
        self._plt_root.deiconify()
        self._cmd_pipe = cmd_pipe
        # Replies to GET_GEOMETRY/GET_WINDOW_STATE. A separate pipe, so that
        # the recorder's mainloop, which reads cmd_pipe, never takes them.
        self._reply_pipe = reply_pipe if reply_pipe is not None else cmd_pipe
        # Dump requests and results (see dump_data)
        self._dump_pipe = dump_pipe
        self._dump_win = None
        self._rbuf_lock = rbuf_lock
        self._rbuf = rbuf
        self._running = True

        # Attach to the ring buffers in this process
        pid = os.getpid()
        for lab, rb in self._rbuf.items():
            self._rbuf[lab] = rb.validate(pid, read_only=True)

        # Start the update timer. The QUIT command ends the mainloop.
        self._plt_root.after(self._timer_interval_ms, self._update)

        try:
            self._plt_root.mainloop()
        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            self._logger.error(f"Error in mainloop: {errstr}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def stop_updates(self):
        """Stop the plot updates."""
        self._running = False
        # Cancel any pending after callbacks
        if hasattr(self, "_update_after_id") and self._update_after_id:
            self._plt_root.after_cancel(self._update_after_id)
            self._update_after_id = None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def start_updates(self):
        """Restart the plot updates."""
        self._running = True
        self._update()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_plot_signals(self, plot_len_sec):
        """Return the signals of the last plot_len_sec seconds for the plot,
        or None if there are no data.
        """
        if self._rbuf is None or self._rbuf_lock is None:
            return None

        # Read only the recent part of the signal buffers (with a margin for
        # samples out of order). Reading and sorting the whole buffers
        # (buf_len_sec) every frame made the plot loop use all the CPU, and
        # the window did not respond to clicks (e.g., the dump button).
        n_tail = int(plot_len_sec * self.signal_freq * 2) + 100
        try:
            with self._rbuf_lock:
                # TTLs are sampled with the signals, so there are no more
                # TTLs than samples in the tail
                ttl_onsets = self._rbuf["ttl_onsets"].get_tail(n_tail)
                ttl_offsets = self._rbuf["ttl_offsets"].get_tail(n_tail)
                card = self._rbuf["card"].get_tail(n_tail)
                resp = self._rbuf["resp"].get_tail(n_tail)
                tstamp = self._rbuf["tstamp"].get_tail(n_tail)
        except Exception as e:
            self._logger.debug(f"Error getting plot signals: {e}")
            return None

        # Remove nan values
        ttl_onsets = ttl_onsets[~np.isnan(ttl_onsets)]
        ttl_offsets = ttl_offsets[~np.isnan(ttl_offsets)]

        # Filter physio data by valid timestamps
        valid_mask = ~np.isnan(tstamp)
        card = card[valid_mask]
        resp = resp[valid_mask]
        tstamp = tstamp[valid_mask]

        if len(tstamp) == 0:
            return None

        # Sort by timestamp
        sidx = np.argsort(tstamp)
        card = card[sidx]
        resp = resp[sidx]
        tstamp = tstamp[sidx]

        ttl_onsets = np.sort(ttl_onsets)
        ttl_offsets = np.sort(ttl_offsets)

        # The latest plot_len_sec seconds, padded with NaN at the start
        data_len = int(plot_len_sec * self.signal_freq)
        data = {}
        for k, dd in [("tstamp", tstamp), ("card", card), ("resp", resp)]:
            if len(dd) >= data_len:
                data[k] = dd[-data_len:]
            else:
                data[k] = np.ones(data_len) * np.nan
                if len(dd):
                    data[k][-len(dd):] = dd

        if len(data["tstamp"]) == 0:
            return None

        # Filter TTL events to plot window
        t0 = np.nanmax(data["tstamp"]) - plot_len_sec
        data["ttl_onsets"] = ttl_onsets[ttl_onsets >= t0]
        data["ttl_offsets"] = ttl_offsets[ttl_offsets >= t0]

        # Determine initial TTL state (at t0) for the plot window
        if np.any(ttl_onsets < t0):
            last_onset = np.max(ttl_onsets[ttl_onsets < t0])
        else:
            last_onset = 0

        if np.any(ttl_offsets < t0):
            last_offset = np.max(ttl_offsets[ttl_offsets < t0])
        else:
            last_offset = 0

        if last_onset == 0 and last_offset == 0:
            data["ttl_init_state"] = 0
        elif last_onset > last_offset:
            data["ttl_init_state"] = 1
        else:
            data["ttl_init_state"] = 0

        return data

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _update(self):
        """Timer callback: handle a command from the controller and redraw
        the plot."""
        smin, smax = self.signal_range  # y range limits of the signals
        if not self._running:
            return

        st = time.time()

        # Check command on pipe
        if self._cmd_pipe and self._cmd_pipe.poll():
            msg = self._cmd_pipe.recv()
            if msg == "QUIT":
                self._running = False
                self._plt_root.quit()  # Exit the mainloop
                return
            elif msg == "STOP_UPDATES":
                self.stop_updates()
                return
            elif msg == "START_UPDATES":
                self.start_updates()
                return
            elif msg == "SHOW":
                self.show()
            elif msg == "HIDE":
                self.hide()
            elif isinstance(msg, tuple) and msg[0] == "SET_TITLE":
                self._plt_root.title(physio_window_title(msg[1]))
            elif isinstance(msg, tuple) and msg[0] == "RESIZE":
                self.set_position(msg[1])
            elif isinstance(msg, tuple) and msg[0] == "TTL_ALARM":
                self.show_ttl_alarm(msg[1])
            elif isinstance(msg, tuple) and msg[0] == "PHYSIO_ALARM":
                self.show_physio_alarm(msg[1])
            elif msg == "SCAN_ON":
                self._is_scanning = True
            elif msg == "SCAN_OFF":
                self._is_scanning = False
            elif msg == "GET_GEOMETRY":
                geometry = self._plt_root.geometry()
                self._reply_pipe.send(geometry)
            elif msg == "GET_WINDOW_STATE":
                self._reply_pipe.send(self._plt_root.wm_state())
            elif isinstance(msg, tuple) and msg[0] == "RESET":
                config = msg[1]
                for key, value in config.items():
                    if hasattr(self, key):
                        if key == "signal_freq":
                            value = float(value)
                        setattr(self, key, value)
                        self._logger.debug(f"Reset {key} to {value}")
                self.reset_plot()

        try:
            # No update while the window is hidden
            if self._plt_root.wm_state() in ("iconic", "withdrawn"):
                # Set next timer event
                if self._running:
                    self._update_after_id = self._plt_root.after(
                        int(self._timer_interval_ms), self._update
                    )
                return
        except Exception:
            if self._running:
                self._update_after_id = self._plt_root.after(
                    int(self._timer_interval_ms), self._update
                )
            return

        # --- Get data --------------------------------------------------------
        plt_data = self.get_plot_signals(self._plot_len_sec + 1)
        if plt_data is None:
            if self._running:
                self._update_after_id = self._plt_root.after(
                    int(self._timer_interval_ms), self._update
                )
            return

        ttl_init_state = plt_data["ttl_init_state"]
        ttl_onsets = plt_data["ttl_onsets"]
        ttl_offsets = plt_data["ttl_offsets"]
        card = plt_data["card"]
        resp = plt_data["resp"]
        tstamp = plt_data["tstamp"]

        # Clean invalid data
        card[tstamp == 0] = 0
        resp[tstamp == 0] = 0

        zero_t = time.time() - np.max(self._ln_ttl[0].get_xdata())
        tstamp = tstamp - zero_t
        ttl_onsets = ttl_onsets - zero_t
        ttl_offsets = ttl_offsets - zero_t
        plt_xt = self._ln_ttl[0].get_xdata()

        # Extend xt (time points) for interpolation
        xt_interval = np.mean(np.diff(plt_xt))
        l_xt_extend = np.arange(-100 * xt_interval, 0, xt_interval) + plt_xt[0]
        r_xt_extend = np.arange(
            plt_xt[-1] + xt_interval, tstamp[-1] + xt_interval, xt_interval
        )
        xt_interval_ex = np.concatenate([l_xt_extend, plt_xt, r_xt_extend])
        # The plotted points within the extended points (a slice; testing
        # each point for membership in plt_xt was O(N^2))
        xt_ex_mask = slice(len(l_xt_extend), len(l_xt_extend) + len(plt_xt))

        # --- Resample in regular interval ------------------------------------
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                f = interp1d(tstamp, card, bounds_error=False)
                card_ex = f(xt_interval_ex)
                card = card_ex[xt_ex_mask]
                f = interp1d(tstamp, resp, bounds_error=False)
                resp_ex = f(xt_interval_ex)
                resp = resp_ex[xt_ex_mask]

        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            self._logger.error(errstr)

        # --- Plot ------------------------------------------------------------
        # region: TTL
        ttl = np.zeros_like(card)
        ttl_onset_plt = []
        ttl_offset_plt = []
        on_off_plt = np.array([])
        change_state = ""

        if len(ttl_onsets):
            ttl_onset_plt = ttl_onsets[
                (ttl_onsets >= plt_xt[0]) & (ttl_onsets <= plt_xt[-1])
            ]
            ttl_onset_plt = ttl_onset_plt - plt_xt[0]
            on_off_plt = np.concatenate((on_off_plt, ttl_onset_plt))

        if len(ttl_offsets):
            ttl_offset_plt = ttl_offsets[
                (ttl_offsets >= plt_xt[0]) & (ttl_offsets <= plt_xt[-1])
            ]
            ttl_offset_plt = ttl_offset_plt - plt_xt[0]
            on_off_plt = np.concatenate((on_off_plt, ttl_offset_plt))

        if len(on_off_plt):
            on_off_state = []
            if len(ttl_onset_plt):
                on_off_state += ["+"] * len(ttl_onset_plt)
            if len(ttl_offset_plt):
                on_off_state += ["-"] * len(ttl_offset_plt)
            on_off_state = np.array(on_off_state, dtype="<U1")

            sidx = np.argsort(on_off_plt).ravel()
            on_off_time = on_off_plt[sidx]
            on_off_state = on_off_state[sidx]

            on_off_idx = np.array(
                [int(np.round(ons * self.signal_freq)) for ons in on_off_time],
                dtype=int,
            )
            on_off_idx = on_off_idx[
                (on_off_idx >= 0) & (on_off_idx < len(ttl))]

            last_idx = 0
            for ii, change_idx in enumerate(on_off_idx):
                change_idx = max(last_idx + 1, change_idx)
                change_state = on_off_state[ii]
                if change_state == "-":
                    ttl[last_idx:change_idx] = 1
                    ttl[change_idx:] = 0
                elif change_state == "+":
                    ttl[last_idx:change_idx] = 0
                    ttl[change_idx:] = 1
                last_idx = change_idx
            ttl[last_idx:] = int(change_state == "+")
        else:
            # No TTL change in the plot range - use initial state
            ttl = np.ones_like(card) * ttl_init_state

        self._ln_ttl[0].set_ydata(ttl)
        # endregion

        # region: Card
        # Set ylim from the latest 3 seconds (adjust_period samples)
        adjust_period = int(np.round(3 * self.signal_freq))

        self._ln_card[0].set_ydata(card)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Adjust ylim
            ymin = max(
                smin, np.floor(np.nanmin(card[-adjust_period:]) / 25) * 25
            )
            ymax = min(
                smax, np.ceil(np.nanmax(card[-adjust_period:]) / 25) * 25
            )
            if ymax - ymin < 50:
                if ymin + 50 < smax:
                    ymax = ymin + 50
                else:
                    ymin = ymax - 50
        if (
            not np.isnan(ymin)
            and not np.isinf(ymin)
            and not np.isnan(ymax)
            and not np.isinf(ymax)
        ):
            self._ax_card.set_ylim((ymin, ymax))
        # endregion

        # region: Card filtered
        b = firwin(
            numtaps=fir_numtaps(self.signal_freq),
            cutoff=3,
            window="hamming",
            pass_zero="lowpass",
            fs=self.signal_freq,
        )
        card_ex_filtered = lfilter(b, 1, card_ex, axis=0)
        card_ex_filtered = np.flipud(card_ex_filtered)
        card_ex_filtered = lfilter(b, 1, card_ex_filtered)
        card_ex_filtered = np.flipud(card_ex_filtered)
        card_filtered = card_ex_filtered[xt_ex_mask]
        self._ln_card_filtered[0].set_ydata(card_filtered)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Adjust ylim
            min_card_filtered = np.nanmin(card_filtered[-adjust_period:])
            ymin = max(smin, np.floor(min_card_filtered / 25) * 25)
            max_card_filtered = np.nanmax(card_filtered[-adjust_period:])
            ymax = min(smax, np.ceil(max_card_filtered / 25) * 25)
            if ymax - ymin < 50:
                if ymin + 50 < smax:
                    ymax = ymin + 50
                else:
                    ymin = ymax - 50
        if (
            not np.isnan(ymin)
            and not np.isinf(ymin)
            and not np.isnan(ymax)
            and not np.isinf(ymax)
        ):
            self._ax_card_filtered.set_ylim((ymin, ymax))
        # endregion

        # region: Resp
        self._ln_resp[0].set_ydata(resp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ymin = max(smin, np.floor(np.nanmin(resp) / 25) * 25)
            ymax = min(smax, np.ceil(np.nanmax(resp) / 25) * 25)
            if ymax - ymin < 50:
                if ymin + 50 < smax:
                    ymax = ymin + 50
                else:
                    ymin = ymax - 50
        if (
            not np.isnan(ymin)
            and not np.isinf(ymin)
            and not np.isnan(ymax)
            and not np.isinf(ymax)
        ):
            self._ax_resp.set_ylim((ymin, ymax))

        if self._is_scanning:
            self._ln_card[0].set_color(COLORS["magenta"])
            self._ln_resp[0].set_color(COLORS["yellow"])
        else:
            self._ln_card[0].set_color(COLORS["cyan"])
            self._ln_resp[0].set_color(COLORS["cyan"])

        self._canvas.draw()
        self._plt_root.update_idletasks()
        # endregion

        # Set next timer event
        et = (time.time() - st) * 1000
        # A negative delay makes Tk call _update at once, which redraws
        # continuously (100% CPU) when drawing takes longer than the interval.
        # Keep at least 30 ms idle, so that clicks are handled.
        after_ms = int(max(30, self._timer_interval_ms - et))
        if self._running:
            self._update_after_id = self._plt_root.after(
                after_ms, self._update)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def config(self):
        if self._cmd_pipe is None:
            return

        self._cmd_pipe.send("GET_CONFIG")
        conf = self._cmd_pipe.recv()
        conf["Plot length (sec)"] = self._plot_len_sec

        # Open a config dialog
        if self.config_win is not None:
            if self.config_win.winfo_exists():
                return
            else:
                self.config_win.destroy()

        self.config_win = tk.Toplevel(self._plt_root)
        self.config_win.title("Recording configurations")

        # Create widgets
        labs = {}
        widgets = {}
        for lab, val in conf.items():
            if "port list" not in lab:
                labs[lab] = tk.Label(
                    self.config_win, text=lab, font=("Helvetica", 11)
                )
                if "port" not in lab:
                    widgets[lab] = tk.Entry(
                        self.config_win,
                        width=10,
                        justify=tk.RIGHT,
                        font=("Helvetica", 11),
                    )
                    widgets[lab].insert(0, val)
                else:
                    port_dict = conf[lab + " list"]
                    if len(port_dict) == 0:
                        continue
                    combo_list = [f"{k}:{v}" for k, v in port_dict.items()]
                    widgets[lab] = ttk.Combobox(
                        self.config_win,
                        values=combo_list,
                        font=("Helvetica", 11),
                        width=30,
                    )
                    widgets[lab].set(val)

        cancelButton = tk.Button(
            self.config_win,
            text="Cancel",
            font=("Helvetica", 11),
            command=self.cancel_config,
        )
        setButton = tk.Button(
            self.config_win,
            text="Set",
            font=("Helvetica", 11),
            command=lambda: self.set_config(widgets),
        )

        # Place widgets
        for ii, (lab, lab_wdgt) in enumerate(labs.items()):
            lab_wdgt.grid(row=ii, column=0, sticky=tk.W + tk.E)
            if lab not in widgets:
                continue
            widgets[lab].grid(
                row=ii, column=1, columnspan=2, sticky=tk.W + tk.E
            )

        cancelButton.grid(row=ii + 1, column=1, sticky=tk.W + tk.E)
        setButton.grid(row=ii + 1, column=2, sticky=tk.W + tk.E)

        # Adjust layout
        col_count, row_count = self.config_win.grid_size()
        for row in range(row_count):
            self.config_win.grid_rowconfigure(row, minsize=32)

        # Move window under the plt_win
        cfg_win_x = self._plt_root.winfo_x()
        cfg_win_y = self._plt_root.winfo_y() + self._plt_root.winfo_height()
        self.config_win.geometry(f"+{cfg_win_x}+{cfg_win_y}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_config(self, widgets):
        conf = {}
        for lab, wdgt in widgets.items():
            conf[lab] = wdgt.get()

        if "Plot length (sec)" in conf:
            self._plot_len_sec = float(conf["Plot length (sec)"])

        if self._cmd_pipe is not None:
            self._cmd_pipe.send("SET_CONFIG")
            self._cmd_pipe.send(conf)

        self.reset_plot()
        self.update_plot_size(None)

        self.config_win.destroy()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def cancel_config(self):
        self.config_win.destroy()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _buffer_times(self):
        """Return (the oldest time stamp in the buffer, True if the buffer is
        full, list of scans in the buffer (see find_scans)).

        When the buffer is full, the oldest data are overwritten as new data
        come, so the oldest time moves on until the data are saved.
        """
        if self._rbuf is None or self._rbuf_lock is None:
            return None, False, []
        with self._rbuf_lock:
            tstamp = self._rbuf["tstamp"].get()
            ttl_onsets = self._rbuf["ttl_onsets"].get()
        valid = np.isfinite(tstamp) & (tstamp > 0)
        is_full = bool(valid.all())
        tstamp = tstamp[valid]
        if len(tstamp) == 0:
            return None, False, []
        oldest = float(np.min(tstamp))
        ttl_onsets = ttl_onsets[np.isfinite(ttl_onsets)]
        scans = find_scans(ttl_onsets[ttl_onsets >= oldest])
        return oldest, is_full, scans

    def dump_data(self):
        """Dump button: choose the start time and confirm, then the
        recorder process saves the data (progress bar while saving), and
        the saved period is shown."""
        if self._dump_pipe is None:
            return
        if not hasattr(self, "_dump_req_id"):
            self._dump_req_id = 0
        if self._dump_win is not None and self._dump_win.winfo_exists():
            self._dump_win.lift()
            return

        oldest, is_full, scans = self._buffer_times()
        if oldest is None:
            messagebox.showinfo(
                "Dump", "No physio data in the buffer.",
                parent=self._plt_root,
            )
            return

        def tstr(t):
            return datetime.fromtimestamp(t).strftime("%H:%M:%S")

        def dstr(sec):
            return str(timedelta(seconds=int(round(sec))))

        # Start time choices: the oldest data and each scan onset. When the
        # buffer is full, the oldest data move on, so the start is the
        # oldest data at the time of saving (onset None), and the time shown
        # is a guide.
        if is_full:
            choices = [(
                f"Oldest available data (about {tstr(oldest)})", None
            )]
        else:
            choices = [
                (f"Oldest data in the buffer ({tstr(oldest)})", oldest)
            ]
        for ii, (ons, last, n) in enumerate(scans):
            choices.append((
                f"Scan {ii + 1}: {tstr(ons)} ({n} TTLs, {dstr(last - ons)})",
                ons,
            ))

        win = tk.Toplevel(self._plt_root)
        self._dump_win = win
        win.title("Dump physio data")
        win.transient(self._plt_root)
        win.resizable(False, False)
        font = ("Helvetica", 11)
        frm = tk.Frame(win, padx=12, pady=10)
        frm.pack(fill="both", expand=True)

        tk.Label(
            frm, text="Save the buffered physio data from:", font=font
        ).pack(anchor="w")
        choice_var = tk.StringVar(value=choices[0][0])
        combo = ttk.Combobox(
            frm, textvariable=choice_var, state="readonly", font=font,
            values=[c[0] for c in choices], width=42,
        )
        combo.pack(anchor="w", pady=(4, 4))
        tk.Label(
            frm, text="to the latest data, in the folder\n"
            f"{self.controller.dump_dir}", font=font, justify="left",
        ).pack(anchor="w")

        btn_frm = tk.Frame(frm)
        btn_frm.pack(fill="x", pady=(10, 0))

        def cancel():
            win.destroy()

        def save():
            onset = dict(choices)[choice_var.get()]
            self._logger.info(
                "Dump physio data from "
                + ("the oldest available data" if onset is None
                   else datetime.fromtimestamp(onset).isoformat())
            )
            # Each request has an id, so that a late result of an earlier
            # request (after its timeout) is not shown as this one's
            self._dump_req_id += 1
            req_id = self._dump_req_id
            self._dump_pipe.send(("DUMP", onset, req_id))

            # Progress view
            for w in frm.winfo_children():
                w.destroy()
            tk.Label(
                frm, text="Saving the physio data ...", font=font
            ).pack(anchor="w")
            pbar = ttk.Progressbar(frm, mode="indeterminate", length=320)
            pbar.pack(fill="x", pady=8)
            pbar.start(15)
            win.protocol("WM_DELETE_WINDOW", lambda: None)
            self._wait_dump_result(win, frm, pbar, time.time(), req_id)

        tk.Button(btn_frm, text="Cancel", command=cancel, font=font,
                  width=8).pack(side=tk.RIGHT, padx=(6, 0))
        tk.Button(btn_frm, text="Save", command=save, font=font,
                  width=8).pack(side=tk.RIGHT)

        # Put the dialog over the plot window
        win.update_idletasks()
        x = self._plt_root.winfo_rootx() + 20
        y = self._plt_root.winfo_rooty() + 40
        win.geometry(f"+{x}+{y}")
        win.grab_set()

    def _wait_dump_result(self, win, frm, pbar, start_t, req_id,
                          timeout=600):
        """Poll the result of dump request req_id from the recorder
        process. Results of earlier requests (arriving after their timeout)
        are logged and skipped."""
        if not win.winfo_exists():
            return
        result = None
        while result is None and self._dump_pipe.poll():
            msg = self._dump_pipe.recv()
            if len(msg) >= 3 and msg[2] == req_id:
                result = msg
            else:
                self._logger.info(
                    f"Late result of an earlier dump request: {msg[:2]}")
        if result is None and time.time() - start_t > timeout:
            result = ("DUMP_FAILED", "No response from the recorder.")
        if result is None:
            win.after(200, self._wait_dump_result, win, frm, pbar, start_t,
                      req_id)
            return

        pbar.stop()
        for w in frm.winfo_children():
            w.destroy()
        font = ("Helvetica", 11)
        if result[0] == "DUMP_DONE":
            info = result[1]
            t0 = datetime.fromtimestamp(info["onset"])
            t1 = datetime.fromtimestamp(info["offset"])
            dur = timedelta(seconds=int(round(info["offset"] - info["onset"])))
            msg = (
                f"Saved {dur} (h:mm:ss) of physio data\n"
                f"from {t0.strftime('%Y-%m-%d %H:%M:%S')}"
                f" to {t1.strftime('%H:%M:%S')}.\n\n"
                f"File: {info['path']}"
            )
            fg = COLORS["fg"]
        else:
            msg = f"Failed to save the physio data.\n{result[1]}"
            fg = COLORS["red"]
        tk.Label(frm, text=msg, font=font, justify="left", fg=fg,
                 wraplength=520).pack(anchor="w")
        tk.Button(frm, text="OK", command=win.destroy, font=font,
                  width=8).pack(anchor="e", pady=(10, 0))
        win.protocol("WM_DELETE_WINDOW", win.destroy)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Alarm sound interval while the dialog is open
    TTL_ALARM_INTERVAL_MS = 3000

    def show_ttl_alarm(self, message):
        """Open the missing TTL warning dialog. The alarm sounds every
        TTL_ALARM_INTERVAL_MS until the dialog is closed."""
        win = getattr(self, "_ttl_alarm_win", None)
        if win is not None and win.winfo_exists():
            win.lift()
            return
        if getattr(self, "_alarm", None) is None:
            self._alarm = AlarmSound()

        c = gui_theme.COLORS
        win = tk.Toplevel(self._plt_root)
        self._ttl_alarm_win = win
        win.title("TTL missing")
        win.attributes("-topmost", True)
        win.configure(highlightthickness=3, highlightbackground=c["red"],
                      highlightcolor=c["red"])
        frm = tk.Frame(win, padx=20, pady=16)
        frm.pack(fill="both", expand=True)
        tk.Label(frm, text="⚠  NO TTL FROM THE SCANNER", fg=c["red"],
                 font=("Helvetica", 16, "bold")).pack(anchor="w")
        tk.Label(frm, text=message, font=("Helvetica", 12), justify="left",
                 wraplength=520).pack(anchor="w", pady=(10, 14))
        tk.Button(frm, text="OK", font=("Helvetica", 12), width=10,
                  command=win.destroy).pack(anchor="e")
        win.protocol("WM_DELETE_WINDOW", win.destroy)

        # Center on the primary screen
        win.update_idletasks()
        x0, y0, sw, sh = screen_area()
        w, h = win.winfo_reqwidth(), win.winfo_reqheight()
        win.geometry(f"+{x0 + (sw - w) // 2}+{y0 + (sh - h) // 3}")

        def beep():
            if win.winfo_exists():
                self._alarm.play(tk_widget=win)
                win.after(self.TTL_ALARM_INTERVAL_MS, beep)

        beep()

    def show_physio_alarm(self, message):
        """Open the physio signal warning dialog (one alarm
        sound)."""
        win = getattr(self, "_physio_alarm_win", None)
        if win is not None and win.winfo_exists():
            win.destroy()
        if getattr(self, "_alarm", None) is None:
            self._alarm = AlarmSound()

        c = gui_theme.COLORS
        win = tk.Toplevel(self._plt_root)
        self._physio_alarm_win = win
        win.title("Physio signal problem")
        win.attributes("-topmost", True)
        win.configure(highlightthickness=3, highlightbackground=c["red"],
                      highlightcolor=c["red"])
        frm = tk.Frame(win, padx=20, pady=16)
        frm.pack(fill="both", expand=True)
        tk.Label(frm, text="⚠  PHYSIO SIGNAL PROBLEM", fg=c["red"],
                 font=("Helvetica", 16, "bold")).pack(anchor="w")
        tk.Label(frm, text=message, font=("Helvetica", 12), justify="left",
                 wraplength=520).pack(anchor="w", pady=(10, 14))
        tk.Button(frm, text="OK", font=("Helvetica", 12), width=10,
                  command=win.destroy).pack(anchor="e")
        win.protocol("WM_DELETE_WINDOW", win.destroy)

        win.update_idletasks()
        x0, y0, sw, sh = screen_area()
        w, h = win.winfo_reqwidth(), win.winfo_reqheight()
        win.geometry(f"+{x0 + (sw - w) // 2}+{y0 + (sh - h) // 3}")
        self._alarm.play(tk_widget=win)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def show(self):
        # Show window
        while self._plt_root.wm_state() in ("iconic", "withdrawn"):
            self._plt_root.deiconify()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def hide(self):
        # Hide window
        while self._plt_root.wm_state() not in ("iconic", "withdrawn"):
            self._plt_root.withdraw()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def update_plot_size(self, event):
        if self._resize_debounce_id:
            self._plt_root.after_cancel(self._resize_debounce_id)
        self._resize_debounce_id = self._plt_root.after(
            1, self._handle_resize, event
        )

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _handle_resize(self, event):
        try:
            # Get the new height of the window
            if event is None:
                new_height = self._plt_root.winfo_height()
            else:
                new_height = event.height

            # Update the size of the Matplotlib canvas
            plot_height = new_height - 1
            self._canvas.get_tk_widget().config(height=plot_height)

            # Get the figure size in inches
            fig_width_inch, fig_height_inch = self._plot_fig.get_size_inches()

            # Calculate the normalized margin values
            left = self._left_margin_inch / fig_width_inch
            right = 1 - (self._right_margin_inch / fig_width_inch)
            top = 1 - (self._top_margin_inch / fig_height_inch)
            bottom = self._bottom_margin_inch / fig_height_inch

            # Adjust the subplots
            self._plot_fig.subplots_adjust(
                left=left, right=right, top=top, bottom=bottom
            )

            # When the Tk scaling is not 96 dpi, matplotlib draws the figure
            # slightly smaller than the canvas, and the strip outside it
            # keeps the image drawn at the earlier size. Clear the image;
            # the timer update draws the figure again.
            try:
                self._canvas._tkphoto.blank()
            except Exception:
                pass

        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            self._logger.error(errstr)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def on_closing(self):
        if self._disable_close:
            if self.controller.rt_mri_main_com.rpc_ping():
                self.controller.rt_mri_main_com.call_rt_proc("HIDE_PHYSIO")
            else:
                self.hide()

            return

        self._plt_root.destroy()
        if self._cmd_pipe:
            self._cmd_pipe.send("QUIT")


# %% _display_available =======================================================
def _display_available(timeout=5.0):
    """Check that a Tk window can be opened, in a subprocess with a timeout:
    tk.Tk() blocks indefinitely when the X display (e.g., a stalled X11
    forwarding) does not respond."""
    try:
        ret = subprocess.run(
            [sys.executable, "-c", "import tkinter; tkinter.Tk().destroy()"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=timeout)
        return ret.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


# %% RtPhysio =================================================================
class RtPhysio:
    """
    Controller class: records the TTL and physiological (cardiac and
    respiration) signals, and serves them via the RPC socket server and the
    shared ring buffers.
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(
        self,
        device=None,
        sample_freq=DEFAULT_SAMPLE_FREQ,
        buf_len_sec=3600,
        save_ttl=True,
        rpc_socket_name="RtTTLPhysioSocketServer",
        rt_mri_main_address_name=["localhost", None, "RtMriMainSocketServer"],
        config_path=Path.home() / ".RTPSpy" / "rtmri_config.json",
        geometry=None,
        disable_close=False,
        **kwargs,
    ):
        """Start the RPC socket server and the recording.

        Parameters
        ----------
        device : str, optional
            Recording device: "Numato", "GE", "Dummy" or "NULL". The default
            (None) selects the first available one of Numato, GE and Dummy.
        sample_freq : float, optional
            Sampling frequency (Hz) of the Numato and Dummy devices. The
            default is DEFAULT_SAMPLE_FREQ. GE is always recorded at
            GEPhysioRecording.SAMPLE_FREQ.
        buf_len_sec : float, optional
            Length (s) of the recording buffer. The default is 3600 s.
        save_ttl : bool, optional
            Save the TTL onset and offset times. The default is True.
        rpc_socket_name : str, optional
            Name of the RPC socket server.
        rt_mri_main_address_name : list, optional
            RtMRIMain RPC address and socket name.
        config_path : Path or str, optional
            Config file in which the RPC servers record their ports.
        geometry : str, optional
            Plot window geometry (WxH+X+Y). The default (None) fits it to
            the screen.
        disable_close : bool, optional
            Disable the close button. The default is False.
        plot_len_sec : float, optional (in kwargs)
            Length (s) of the plotted signals. The default is 10.
        """
        self._logger = logging.getLogger("RtPhysio")

        # --- Initialize parameters ---
        self.buf_len_sec = buf_len_sec
        self.sample_freq = sample_freq
        self._numato_sample_freq = sample_freq
        self.save_ttl = save_ttl
        self.sim_card_f = None
        self.sim_resp_f = None
        self.sim_bids_f = None  # BIDS physio file for the Dummy device
        # (min, max) of the Dummy signals, for the plot range
        self._sim_signal_range = None

        config_path = Path(config_path)
        self.rt_mri_main_com = RPCSocketCom(
            rt_mri_main_address_name, config_path
        )
        self.geometry = geometry
        self.disable_close = disable_close

        self.plot_len_sec = kwargs.get("plot_len_sec", 10)

        # Set state variables
        self.wait_ttl_on = False  # Waiting for the TTL of the scan start
        self._plot = None  # View class of signal plot
        self._recorder_type = None  # Signal recorder type

        # Queues to receive the recorded data from the recorder process
        self._ttl_onset_que = Queue(maxsize=1000)
        self._ttl_offset_que = Queue(maxsize=1000)
        self._physio_que = Queue(maxsize=1000)

        self._rec_proc = None  # Signal recording process
        self._rec_proc_pipe = None

        # Clear shared memory buffers left by processes that have ended.
        # The files of a running process (e.g., another instance) are kept:
        # deleting them broke its readers (the motion monitor).
        dir_path = Path("/dev/shm")
        for rm_f in dir_path.glob("rtmri_physio_*"):
            ma = re.match(r"rtmri_physio_(\d+)_", rm_f.name)
            if ma is None:
                continue
            try:
                os.kill(int(ma.group(1)), 0)
                continue  # The process is running
            except ProcessLookupError:
                pass  # Ended
            except PermissionError:
                continue  # Running, of another user
            try:
                rm_f.unlink()
            except Exception:
                pass

        # Scan onset, shared with other processes via an mmap file
        self._scan_onset = SharedMemoryRingBuffer(1, initial_value=-1.0)
        self.inScan = False  # In a scan (START_SCAN to the scan end)
        self.fname_fmt = "./physio.tsv"
        # Files written by the automatic save (scan end detected from TTL),
        # keyed by fname_fmt. A later SAVE_PHYSIO_DATA for the same file
        # name replaces them, so that one scan has one set of files.
        self._auto_saved = {}
        # save_physio_data runs in the RPC thread (SAVE_PHYSIO_DATA) and in
        # the mainloop thread (automatic save at the TTL end, dump). One at
        # a time, so that a save that replaces the automatic save finds it.
        self._save_lock = threading.RLock()
        self._scan_start_time = None  # time.time() of START_SCAN
        self._no_ttl_warned = False
        # fMRI series in progress (START_SCAN to END_SCAN from the DICOM
        # monitor), and its volumes: (acquisition time or None, arrival
        # time, TR) from FMRI_VOLUME
        self._series_active = False
        self._fmri_vols = deque(maxlen=10)
        # Physio signal states already warned {(session, state)}
        self._physio_warned = set()
        self._ttl_alarm_done = False
        # A series (any type) started: (time, study UID, series number,
        # description), checked by _check_no_ttl_at_all. Studies already
        # warned of.
        self._mri_series = None
        self._no_ttl_warned_studies = set()

        # Names of the ring buffers shared with other processes
        self._rbuf_names = [
            "ttl_onsets",
            "ttl_offsets",
            "card",
            "resp",
            "tstamp",
        ]
        self._rbuf_lock = Lock()

        # --- Start RPC socket server ---
        self._rpc_pipe, cmd_pipe = Pipe()
        self.socket_srv = RPCSocketServer(
            config_path,
            self.RPC_handler,
            handler_kwargs={"cmd_pipe": cmd_pipe},
            socket_name=rpc_socket_name,
        )

        # --- Create dump directory ---
        self.dump_dir = Path.cwd() / "dump"
        self.dump_dir.mkdir(exist_ok=True)
        self._logger.info(f"Dump directory: {self.dump_dir}")

        # Ring buffers are created even if no device is available, so that
        # the plot and the RPC commands work without a device.
        self._init_ring_buffers()

        # Set device and start recording
        self.set_device(device)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _init_ring_buffers(self):
        """Make the ring buffers shared with other processes."""
        buf_len = self.buf_len_sec * self.sample_freq
        # Make all buffers first and then set self._rbuf at once, so that
        # GET_RBUF in the RPC thread never returns a partial set.
        rbuf = {}
        for label in self._rbuf_names:
            if label == "tstamp":
                initial_value = 0.0
            else:
                initial_value = np.nan
            rbuf[label] = SharedMemoryRingBuffer(
                buf_len, initial_value=initial_value
            )
        self._rbuf = rbuf

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_device(self, device=None):
        """Change the recording device and restart the recording.

        device None selects the first available one of Numato, GE and
        Dummy. "NULL" stops using a device.
        """
        if device is not None and device not in (
            "Numato",
            "GE",
            "Dummy",
            "NULL",
        ):
            self._logger.error(f"Device {device} is not supported.")
            return

        # --- Set recorder device ---
        if device is None:
            try_devices = ["Numato", "GE", "Dummy"]
        else:
            try_devices = [device]

        self._logger.debug(f"Attempting to set device: {try_devices}")

        recorder_type = None
        for dev in try_devices:
            self._logger.debug(f"Trying device: {dev}")
            try:
                if dev == "Numato":
                    if not NumatoGPIORecording.is_device_available():
                        self._logger.debug("Numato device not available.")
                        continue
                    else:
                        recorder_type = dev
                        break

                elif dev == "GE":
                    # In auto selection, the port must send GE packets.
                    if not GEPhysioRecording.is_device_available(
                        probe=device is None
                    ):
                        self._logger.debug("GE physio port not available.")
                        continue
                    else:
                        recorder_type = dev
                        break

                elif dev == "Dummy":
                    if (
                        self.sim_bids_f is not None
                        and self.sim_bids_f.is_file()
                    ) or (
                        self.sim_card_f is not None
                        and self.sim_card_f.is_file()
                        and self.sim_resp_f is not None
                        and self.sim_resp_f.is_file()
                    ):
                        recorder_type = dev
                        break

                elif dev == "NULL":
                    recorder_type = None
                    break

            except Exception as e:
                errstr = str(e) + "\n" + traceback.format_exc()
                self._logger.error(
                    f"Error initializing device {dev}: {errstr}")

        if recorder_type is None and device != "NULL":
            self._logger.warning(
                "No suitable device was found. The NULL device is used."
            )
            return

        if recorder_type == self._recorder_type:
            self._logger.debug(f"Device {recorder_type} is already in use.")
            return
        else:
            self._recorder_type = recorder_type

        # Recording frequency depends on the device
        if self._recorder_type == "GE":
            self.sample_freq = GEPhysioRecording.SAMPLE_FREQ
        elif self._recorder_type == "Numato":
            self.sample_freq = self._numato_sample_freq
        self._logger.info(
            f"Recording device: {self._recorder_type}, "
            f"{self.sample_freq} Hz"
        )
        # The plot process can be made before the device is chosen
        self._plot_send(("SET_TITLE", self._recorder_type))

        if self._recorder_type is not None:
            self._logger.debug(
                f"Stopping current recorder: {self._recorder_type}"
            )
            try:
                self.stop_recording()
                time.sleep(0.5)  # Let the device be released
            except Exception as e:
                errstr = str(e) + "\n" + traceback.format_exc()
                self._logger.error(
                    f"Error stopping current recorder: {errstr}")

        # Start the new recorder process. The plot continues with the data
        # of the new device.
        self._logger.debug(f"Starting recorder: {self._recorder_type}")
        try:
            self.start_recording()
        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            self._logger.error(
                f"Error starting recorder {self._recorder_type}: {errstr}"
            )

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _create_recorder(self, device_type):
        """Return the recorder (model class) instance of device_type."""
        if device_type == "Numato":
            return NumatoGPIORecording(
                ttl_onset_que=self._ttl_onset_que,
                ttl_offset_que=self._ttl_offset_que,
                physio_que=self._physio_que,
                sample_freq=self.sample_freq,
            )
        elif device_type == "GE":
            return GEPhysioRecording(
                ttl_onset_que=self._ttl_onset_que,
                ttl_offset_que=self._ttl_offset_que,
                physio_que=self._physio_que,
            )
        elif device_type == "Dummy":
            return DummyRecording(
                ttl_onset_que=self._ttl_onset_que,
                ttl_offset_que=self._ttl_offset_que,
                physio_que=self._physio_que,
                sim_card_f=self.sim_card_f,
                sim_resp_f=self.sim_resp_f,
                sample_freq=self.sample_freq,
                sim_bids_f=self.sim_bids_f,
            )
        elif device_type == "NULL":
            return None
        else:
            self._logger.error(f"Unknown device type: {device_type}")
            return None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # getter, setter
    @property
    def signal_range(self):
        """(min, max) of the signal values of the device, for the plot."""
        if self._recorder_type == "GE":
            return GEPhysioRecording.SIGNAL_RANGE
        elif (
            self._recorder_type == "Dummy"
            and self._sim_signal_range is not None
        ):
            return self._sim_signal_range
        else:
            return (0, ADC_MAX_VALUE)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_sim_signal_range(self, *signals):
        """Set the plot range from the Dummy signals. Recorded signals (e.g.,
        GE int16 values) can be outside the Numato ADC range."""
        vals = [np.asarray(v, dtype=float).ravel() for v in signals
                if v is not None]
        vals = np.concatenate(vals) if vals else np.array([])
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0 or vals.min() == vals.max():
            self._sim_signal_range = None
        else:
            self._sim_signal_range = (
                float(np.floor(vals.min())), float(np.ceil(vals.max()))
            )

    @property
    def scan_onset(self):
        return self._scan_onset.get()[0]

    @scan_onset.setter
    def scan_onset(self, onset):
        self._scan_onset.append(onset)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def start_recording(self):
        """Start the recording loop in a separate process.
        To change the recording device, use set_device, which calls
        start_recording.
        """
        if not hasattr(self, "_rbuf") or self._rbuf is None:
            self._init_ring_buffers()

        self._rec_proc_pipe, cmd_pipe = Pipe()
        self._rec_proc = Process(target=self._run_recording, args=(cmd_pipe,))

        self._rec_proc.start()

        # Reset the plot for the sampling frequency and range of the device
        if hasattr(self, "_plot_proc_pipe") and self._plot_proc_pipe:
            config = {
                "signal_freq": self.sample_freq,
                "signal_range": self.signal_range,
            }
            self._logger.debug(f"Sending plot config: {config}")
            self._plot_send(("RESET", config))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def is_recording(self):
        return self._rec_proc is not None and self._rec_proc.is_alive()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def stop_recording(self):
        if not self.is_recording():
            return

        self._rec_proc_pipe.send("QUIT")
        self._rec_proc.join(3)
        self._rec_proc.terminate()

        # Close the pipe, so that it is not read after the process ended
        if hasattr(self, "_rec_proc_pipe") and self._rec_proc_pipe:
            try:
                self._rec_proc_pipe.close()
            except Exception:
                pass
            self._rec_proc_pipe = None

        del self._rec_proc
        self._rec_proc = None

        gc.collect()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def open_plot(self):
        """Start the plot window process."""
        if self._plot is None:
            self._plot = PlotTTLPhysio(
                self,
                geometry=self.geometry,
                signal_freq=self.sample_freq,
                plot_len_sec=self.plot_len_sec,
                disable_close=self.disable_close,
                signal_range=self.signal_range,
            )

        self._plot_proc_pipe, cmd_pipe = Pipe()
        # Dump requests from the dump button and their results
        self._dump_pipe, dump_pipe = Pipe()
        # Replies of the plot to GET_GEOMETRY/GET_WINDOW_STATE (see
        # _plot_request)
        self._plot_reply_pipe, reply_pipe = Pipe(duplex=False)
        # The RPC thread and the mainloop thread both send to the plot.
        self._plot_send_lock = threading.Lock()
        self._plot_request_lock = threading.Lock()
        self._plot_proc = Process(
            target=self._plot.run,
            args=(cmd_pipe, self._rbuf_lock, self._rbuf, dump_pipe,
                  reply_pipe),
        )
        self._plot_proc.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _plot_send(self, msg):
        """Send a command to the plot process (thread-safe). Returns False
        if there is no plot or the pipe is broken."""
        pipe = getattr(self, "_plot_proc_pipe", None)
        if not pipe:
            return False
        try:
            with self._plot_send_lock:
                pipe.send(msg)
            return True
        except (OSError, EOFError, ValueError) as e:
            self._logger.error(f"Failed to send {msg!r} to the plot: {e}")
            return False

    def _plot_request(self, cmd, timeout=2.0):
        """Send cmd and return the plot's reply, or None if there is no
        reply within timeout (e.g., the plot is not running yet)."""
        reply = getattr(self, "_plot_reply_pipe", None)
        if reply is None:
            return None
        with self._plot_request_lock:
            try:
                while reply.poll():  # Drop a late reply to an earlier call
                    reply.recv()
                if not self._plot_send(cmd):
                    return None
                if reply.poll(timeout):
                    return reply.recv()
            except (OSError, EOFError) as e:
                self._logger.error(f"No reply from the plot to {cmd}: {e}")
        return None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def stop_plot(self):
        if not hasattr(self, "_plot_proc") or self._plot_proc is None:
            return

        if self._plot_proc.is_alive():
            # Ask the plot to quit, then terminate or kill it if it does not
            try:
                self._plot_send("QUIT")
                self._plot_proc.join(timeout=1.0)
            except (BrokenPipeError, EOFError):
                pass  # The process may have ended already

            if self._plot_proc.is_alive():
                self._plot_proc.terminate()
                self._plot_proc.join(timeout=1.0)

            if self._plot_proc.is_alive():
                self._plot_proc.kill()
                self._plot_proc.join(timeout=0.5)

        if hasattr(self, "_plot_proc_pipe") and self._plot_proc_pipe:
            try:
                self._plot_proc_pipe.close()
            except Exception:
                pass
            self._plot_proc_pipe = None

        if hasattr(self, "_plot_proc"):
            del self._plot_proc
            self._plot_proc = None

        # open_plot makes a new plot
        self._plot = None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def close_plot(self):
        if not hasattr(self, "_plot") or self._plot is None:
            return

        self._plot.on_closing()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_config(self):
        return None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_config(self, conf):
        return None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _run_recording(self, cmd_pipe):
        """Recording process: start the reader of the device in another
        process, and move the data from its queues to the ring buffers."""
        self.wait_ttl_on = False
        self.scan_onset = 0
        device_type = self._recorder_type

        recorder = self._create_recorder(device_type)
        if recorder is None:
            self._logger.info("No recorder is available.")
            return

        # Attach to the ring buffers in this process
        pid = os.getpid()
        for lab, rbuf in self._rbuf.items():
            self._rbuf[lab] = rbuf.validate(pid)

        # Start the reading process. It sends the data via the queues.
        _recorder_pipe, cmd_pipe_recorder = Pipe()
        _read_proc = Process(
            target=recorder.read_signal_loop,
            kwargs={"cmd_pipe": cmd_pipe_recorder},
        )
        _read_proc.start()

        # Queue reading loop
        while True:
            if cmd_pipe.poll():
                cmd = cmd_pipe.recv()
                if cmd == "QUIT":
                    self._logger.debug(
                        "Receive QUIT in _run_recording. Break recording loop."
                    )
                    break
                elif cmd == "PULSE" or (
                    type(cmd) is tuple and cmd[0] == "PULSE"
                ):
                    _recorder_pipe.send(cmd)
                elif cmd == "WAIT_TTL_ON":
                    self.wait_ttl_on = True
                elif cmd == "WAIT_TTL_OFF":
                    self.wait_ttl_on = False

            if not self._ttl_onset_que.empty():
                while not self._ttl_onset_que.empty():
                    ttl_onsets = self._ttl_onset_que.get()
                    if self.wait_ttl_on:
                        self.scan_onset = ttl_onsets
                        self.wait_ttl_on = False
                    with self._rbuf_lock:
                        self._rbuf["ttl_onsets"].append(ttl_onsets)

            if not self._ttl_offset_que.empty():
                while not self._ttl_offset_que.empty():
                    ttl_offsets = self._ttl_offset_que.get()
                    with self._rbuf_lock:
                        self._rbuf["ttl_offsets"].append(ttl_offsets)

            if not self._physio_que.empty():
                while not self._physio_que.empty():
                    tstamp, card, resp = self._physio_que.get()
                    with self._rbuf_lock:
                        self._rbuf["card"].append(card)
                        self._rbuf["resp"].append(resp)
                        self._rbuf["tstamp"].append(tstamp)

            time.sleep(0.1 / self.sample_freq)

        # --- end loop ---

        # Stop the reading process
        cmd_pipe.send("END_RECORDING")
        _recorder_pipe.send("QUIT")
        if _recorder_pipe.poll(timeout=3):
            _recorder_pipe.recv()
        _read_proc.join(1)
        _read_proc.terminate()

        del recorder

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def dump(self):
        """Return the recorded signals (valid samples sorted by time and
        sorted TTL times), or None if not recording or no sample."""
        if not self.is_recording():
            return None

        # Attach to the ring buffers in this process
        pid = os.getpid()
        for lab, rbuf in self._rbuf.items():
            self._rbuf[lab] = rbuf.validate(pid)

        with self._rbuf_lock:
            ttl_onsets = self._rbuf["ttl_onsets"].get().copy()
            ttl_offsets = self._rbuf["ttl_offsets"].get().copy()
            tstamp = self._rbuf["tstamp"].get().copy()
            card = self._rbuf["card"].get().copy()
            resp = self._rbuf["resp"].get().copy()

        # Remove nan
        ttl_onsets = ttl_onsets[~np.isnan(ttl_onsets)]
        ttl_offsets = ttl_offsets[~np.isnan(ttl_offsets)]

        tmask = (
            ~np.isnan(tstamp) & ~np.isnan(card) &
            ~np.isnan(resp) & (tstamp > 0)
        )
        card = card[tmask]
        resp = resp[tmask]
        tstamp = tstamp[tmask]
        if len(tstamp) == 0:
            return None

        # Sort by time stamp
        sidx = np.argsort(tstamp)
        card = card[sidx]
        resp = resp[sidx]
        tstamp = tstamp[sidx]
        ttl_onsets = np.sort(ttl_onsets)
        ttl_offsets = np.sort(ttl_offsets)

        data = {
            "ttl_onsets": ttl_onsets,
            "ttl_offsets": ttl_offsets,
            "card": card,
            "resp": resp,
            "tstamp": tstamp,
        }

        return data

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_ttl_onsets(self, num=None):
        """Return the last num TTL onsets (all if None), or None if not
        recording."""
        if not self.is_recording():
            return None

        # Attach to the ring buffer in this process
        pid = os.getpid()
        self._rbuf["ttl_onsets"] = self._rbuf["ttl_onsets"].validate(pid)
        ttl_onsets = self._rbuf["ttl_onsets"].get().copy()

        # Remove nan
        ttl_onsets = ttl_onsets[~np.isnan(ttl_onsets)]
        if num is not None:
            ttl_onsets = ttl_onsets[-num:]

        return ttl_onsets

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _warn_physio_signal(self, fname_fmt, sig):
        """Show a warning dialog when the respiration or the cardiac signal
        is weak or missing in a scan of a subject. Once per session for the
        same state of the signals."""
        state = (sig["respiratory"], sig["cardiac"])
        if state == ("ok", "ok"):
            return
        name = Path(fname_fmt).name
        ma = re.search(r"_ses-([^_]+)", name)
        key = (ma.group(1) if ma else name, state)
        if key in self._physio_warned:
            return
        self._physio_warned.add(key)
        run = re.sub(r"_physio\.tsv$", "", name)
        text = {
            ("respiratory", "none"): "RESPIRATION: not detected. Check the "
            "respiration belt.",
            ("respiratory", "weak"): "RESPIRATION: weak. The belt may be "
            "loose or out of place; check its position and tightness.",
            ("cardiac", "none"): "CARDIAC: not detected. Check the pulse "
            "oximeter (PPG) or the ECG leads.",
            ("cardiac", "weak"): "CARDIAC: weak. The pulse sensor may be "
            "loose or poorly placed.",
            ("respiratory", "irregular"): "RESPIRATION: irregular (no "
            "regular breathing cycle). Check for movement, and that the "
            "belt is not too tight (clipped signal).",
            ("cardiac", "irregular"): "CARDIAC: irregular (no regular "
            "pulse; noise). Check the pulse sensor contact or the ECG "
            "leads.",
        }
        lines = [text[(k, sig[k])] for k in ("respiratory", "cardiac")
                 if sig[k] != "ok"]
        if state == ("none", "none"):
            lines.append("The physio data of this scan were NOT saved.")
        msg = ("\n".join(lines)
               + f"\n\nScan: {run}\n(respiration std {sig['resp_std']}, "
               f"periodicity {sig['resp_periodicity']}; cardiac std "
               f"{sig['card_std']}, periodicity {sig['card_periodicity']})"
               "\n\nThis warning is not shown again in this session for "
               "the same state.")
        self._logger.warning("Physio signal: " + " ".join(lines))
        self._plot_send(("PHYSIO_ALARM", msg))

    def save_physio_data(self, *args, **kwargs):
        """Save the physio data of a scan (see _save_physio_data). Saves
        are serialized."""
        with self._save_lock:
            return self._save_physio_data(*args, **kwargs)

    def _save_physio_data(
        self,
        onset=None,
        len_sec=None,
        fname_fmt="./physio.tsv",
        resample_regular_interval=True,
        nosignal_nosave=True,
        auto=False,
        whole_buffer=False,
        est_onset=None,
        est_tol=None,
        is_phantom=None,
    ):
        """Save the physio data of a scan.

        auto : bool
            True for the automatic save at the scan end detected from TTL.
            A later save with the same fname_fmt and auto=False (e.g.,
            SAVE_PHYSIO_DATA from rt_dicom_monitor) overwrites the files of
            the automatic save instead of adding new ones.
        whole_buffer : bool
            True to save the whole buffer, ignoring the scan onset (the dump
            button). Otherwise, the data are cut at the last scan onset.
        est_onset, est_tol : float
            Scan onset estimated from the DICOM times by the DICOM monitor
            (the scan end minus the series duration), and its tolerance (s).
            It is used when the onset from the TTLs is not set or differs
            from it by more than est_tol (e.g., no TTL in this scan, and the
            onset of an earlier scan remains). The onset used is recorded
            in the log and in the JSON (ScanOnsetSource).
        is_phantom : bool or None
            From the DICOM monitor. False (a subject): a warning dialog is
            shown when the respiration or the cardiac signal is missing.
            None (e.g., the automatic save): no dialog.
        """
        scan_onset = 0 if whole_buffer else self.scan_onset
        onset_source = "TTL"
        if not whole_buffer and est_onset is not None:
            tol = est_tol if est_tol is not None else 10.0
            if scan_onset <= 0 or abs(scan_onset - est_onset) > tol:
                ttl_str = (datetime.fromtimestamp(scan_onset).isoformat()
                           if scan_onset > 0 else "none")
                self._logger.warning(
                    f"The scan onset was not found from the TTLs (TTL onset: "
                    f"{ttl_str}). Using the onset estimated from the DICOM "
                    f"times: {datetime.fromtimestamp(est_onset).isoformat()}"
                )
                scan_onset = est_onset
                onset_source = "DICOM time (estimated)"
        elif scan_onset <= 0:
            onset_source = None

        # region: Get data ----------------------------------------------------
        wait_end = None  # Limit of waiting for the data until the scan end
        while True:
            data = self.dump()
            if data is None:
                return

            tstamp = data["tstamp"]
            if len(tstamp) == 0:
                self._logger.warning("No valid data found.")
                return

            if onset is None:
                onset = tstamp[0]
                if scan_onset > 0 and scan_onset > onset:
                    # Save from 1 second before the scan onset
                    onset = scan_onset - 1

            # The onset (also an explicit onset=0) cannot be earlier than
            # the buffer length before the latest data
            buffer_limited_onset = tstamp[-1] - self.buf_len_sec
            if onset < buffer_limited_onset:
                onset = buffer_limited_onset

            scan_end = None
            if len_sec is not None:
                if scan_onset > 0 and scan_onset > onset:
                    scan_end = scan_onset + len_sec
                    # Save until 1 second after the scan end
                    offset = scan_end + 1
                else:
                    offset = onset + len_sec
                if tstamp[-1] < offset + 2:
                    # Wait for the data until the scan end. If no data come
                    # (e.g., the device stopped), save what is there after
                    # a limit, rather than waiting forever (this runs in the
                    # RPC thread and would block all later calls).
                    if wait_end is None:
                        wait_end = (time.time()
                                    + max(0.0, offset + 2 - tstamp[-1])
                                    + 10.0)
                    if time.time() < wait_end:
                        time.sleep(1)
                        continue
                    self._logger.warning(
                        "Physio data did not reach the scan end "
                        f"({datetime.fromtimestamp(offset).isoformat()}). "
                        "Saving the data recorded until "
                        f"{datetime.fromtimestamp(tstamp[-1]).isoformat()}."
                    )
                    offset = tstamp[-1]
            else:
                offset = tstamp[-1]
                if scan_onset > 0:
                    # Scan end is not given (e.g., saved when TTL stopped).
                    # Use the last TTL of the scan plus the median interval.
                    scan_ttl = data["ttl_onsets"]
                    scan_ttl = np.sort(
                        scan_ttl[scan_ttl >= scan_onset]
                    )
                    if len(scan_ttl) > 1:
                        scan_end = scan_ttl[-1] + np.median(
                            np.diff(scan_ttl[-10:])
                        )
                        offset = min(offset, scan_end + 1)

            if scan_end is not None:
                # The margin after the scan ends just before the first TTL
                # of the next scan, if that comes within 1 second.
                next_ttl = data["ttl_onsets"]
                next_ttl = next_ttl[
                    (next_ttl >= scan_end) & (next_ttl < offset)
                ]
                if len(next_ttl):
                    offset = float(np.min(next_ttl))
                    self._logger.info(
                        "The next scan starts "
                        f"{offset - scan_end:.3f} s after the end of this "
                        "scan. The data are saved until just before it."
                    )

            break

        self._logger.info(
            "Saving physio data"
            f" from {datetime.fromtimestamp(onset).isoformat()}"
            f" for {timedelta(seconds=offset - onset)}"
        )

        # Warn of implausible time stamps
        current_time = time.time()
        if onset < current_time - 86400 * 365 * 10:  # More than 10 years ago
            self._logger.warning(
                f"Onset timestamp {onset} seems too old "
                f"(current time: {current_time})"
            )
        if offset < current_time - 86400 * 365 * 10:  # More than 10 years ago
            self._logger.warning(
                f"Offset timestamp {offset} seems too old "
                f"(current time: {current_time})"
            )
        # endregion

        # region: Select TTL onsets
        # TTLs after the buffer limit, from the scan onset (or from onset if
        # the scan onset is not set) to offset
        ttl_onsets = data["ttl_onsets"]
        ttl_onsets = ttl_onsets[ttl_onsets > buffer_limited_onset]
        if scan_onset > 0:
            ttl_onsets = ttl_onsets[
                (ttl_onsets >= scan_onset) & (ttl_onsets < offset)
            ]
        else:
            ttl_onsets = ttl_onsets[
                (ttl_onsets >= onset) & (ttl_onsets < offset)
            ]
        # endregion

        save_path = Path(fname_fmt).resolve().parent
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)

        save_key = str(Path(fname_fmt).resolve())
        replace = {} if auto else self._auto_saved.pop(save_key, {})
        written = {}

        if self.save_ttl:
            # region: Save TTL onsets -----------------------------------------
            if len(ttl_onsets) > 0:
                ttl_onset_df = pd.DataFrame(columns=("DateTime",))
                ttl_onset_df["DateTime"] = [
                    datetime.fromtimestamp(ons).isoformat()
                    for ons in ttl_onsets
                ]
                if scan_onset > 0:
                    ttl_onset_df["TimefromScanOnset"] = (
                        ttl_onsets - scan_onset
                    )

                ttl_onset_stem = Path(fname_fmt).stem.replace(
                    "physio", "PulseOnset"
                )
                ttl_onset_fname = self._output_path(
                    save_path / (ttl_onset_stem + ".tsv"), "onset", replace
                )
                ttl_onset_df.to_csv(ttl_onset_fname, sep="\t")
                written["onset"] = ttl_onset_fname
            # endregion

            # region: Save TTL offsets ----------------------------------------
            if len(data["ttl_offsets"]):
                # Selected as the TTL onsets
                ttl_offsets = data["ttl_offsets"]
                ttl_offsets = ttl_offsets[ttl_offsets > buffer_limited_onset]
                if scan_onset > 0:
                    ttl_offsets = ttl_offsets[
                        (ttl_offsets >= scan_onset)
                        & (ttl_offsets < offset)
                    ]
                else:
                    ttl_offsets = ttl_offsets[
                        (ttl_offsets >= onset) & (ttl_offsets < offset)
                    ]
                ttl_offset_df = pd.DataFrame(columns=("DateTime",))
                ttl_offset_df["DateTime"] = [
                    datetime.fromtimestamp(ons).isoformat()
                    for ons in ttl_offsets
                ]
                if scan_onset > 0:
                    ttl_offset_df["TimefromScanOnset"] = (
                        ttl_offsets - scan_onset
                    )

                ttl_offset_stem = Path(fname_fmt).stem.replace(
                    "physio", "PulseOffset"
                )
                ttl_offset_fname = self._output_path(
                    save_path / (ttl_offset_stem + ".tsv"), "offset", replace
                )
                ttl_offset_df.to_csv(ttl_offset_fname, sep="\t")
                written["offset"] = ttl_offset_fname
            # endregion

        # region: Clean and resample physio data ------------------------------
        dataMask = (
            (tstamp >= (onset - 5.0))
            & (tstamp <= (offset + 5.0))
            & np.logical_not(np.isnan(tstamp))
        )
        if dataMask.sum() == 0:
            return

        save_card = data["card"][dataMask]
        save_resp = data["resp"][dataMask]
        tstamp = tstamp[dataMask]

        save_card = self._clean_resamp(save_card, tstamp)
        save_resp = self._clean_resamp(save_resp, tstamp)

        if resample_regular_interval:
            try:
                if not np.isfinite(onset) or not np.isfinite(offset):
                    self._logger.error(
                        f"Invalid onset ({onset}) or offset ({offset}) values"
                    )
                    return None

                duration = offset - onset
                if duration <= 0:
                    self._logger.error(
                        f"Invalid duration: offset ({offset}) <= "
                        f"onset ({onset})"
                    )
                    return None

                # Resample card and resp at the sampling interval
                intv = 1.0 / self.sample_freq
                ti = np.arange(onset, offset, intv)

                f = interp1d(tstamp, save_card, bounds_error=False)
                save_card = f(ti)
                f = interp1d(tstamp, save_resp, bounds_error=False)
                save_resp = f(ti)

                tstamp = ti

            except Exception as e:
                errstr = str(e) + "\n" + traceback.format_exc()
                self._logger.error(errstr)
        else:
            tmask = (tstamp >= onset) & (tstamp < offset)
            save_card = data["card"][tmask]
            save_resp = data["resp"][tmask]
            tstamp = tstamp[tmask]

        # endregion

        sig = None
        if nosignal_nosave:
            # The data are saved unless both signals are "none"
            # (check_physio_signals, thresholds per device).
            sig = check_physio_signals(save_card, save_resp,
                                       self.sample_freq, self._recorder_type)
            self._logger.info(
                f"Physio signal check: respiration {sig['respiratory']} "
                f"(std {sig['resp_std']}, periodicity "
                f"{sig['resp_periodicity']}), cardiac {sig['cardiac']} (std "
                f"{sig['card_std']}, periodicity {sig['card_periodicity']}),"
                f" thresholds {sig['thresholds']}")
            if is_phantom is False and (sig["respiratory"] != "ok"
                                        or sig["cardiac"] != "ok"):
                self._warn_physio_signal(fname_fmt, sig)
            if sig["respiratory"] == "none" and sig["cardiac"] == "none":
                self._logger.error(
                    "Physio data not saved: neither the respiration nor the "
                    "cardiac signal was detected.")
                return

        # region: Save physio data --------------------------------------------
        physio_df = pd.DataFrame()
        physio_df["Time"] = tstamp[: len(save_card)]
        physio_df["cardiac"] = save_card
        physio_df["respiratory"] = save_resp
        physio_df.index.name = "Time"

        # Add trigger: 1 at the time stamp closest to each TTL onset.
        # tstamp is sorted, so the closest one is found by searchsorted
        # (the same as argmin(|tstamp - onset|), but fast for long data).
        trigger = np.zeros(len(physio_df), dtype=int)
        if len(ttl_onsets) and len(tstamp):
            if len(tstamp) == 1:
                ttl_idx = np.zeros(len(ttl_onsets), dtype=int)
            else:
                ttl_idx = np.clip(
                    np.searchsorted(tstamp, ttl_onsets), 1, len(tstamp) - 1
                )
                left = np.abs(ttl_onsets - tstamp[ttl_idx - 1])
                right = np.abs(tstamp[ttl_idx] - ttl_onsets)
                ttl_idx = np.where(left <= right, ttl_idx - 1, ttl_idx)
            ttl_idx = ttl_idx[ttl_idx < len(physio_df)]
            trigger[ttl_idx] = 1
        physio_df["trigger"] = trigger

        tsv_path = self._output_path(
            Path(fname_fmt).with_suffix(".tsv"), "physio", replace
        )
        json_fname = tsv_path.with_suffix(".json")

        physio_df.to_csv(tsv_path, sep="\t", index=False, header=False)
        metadata = {
            "SamplingFrequency": self.sample_freq,
            "StartTime": 0,
            "Columns": list(physio_df.columns),
            # Numato digitizes the analog outputs of the BIOPAC amplifiers
            # (PhysioRecording.md)
            "Manufacturer": {
                "GE": "GE", "Dummy": "Dummy", "Numato": "BIOPAC"
            }.get(self._recorder_type, "n/a"),
        }
        if sig is not None:
            metadata["SignalCheck"] = {
                k: sig[k] for k in ("respiratory", "cardiac", "resp_std",
                                    "resp_periodicity", "card_std",
                                    "card_periodicity")}
        if scan_onset > 0:
            metadata["StartTime"] = onset - scan_onset
            if not whole_buffer and onset_source is not None:
                metadata["ScanOnsetSource"] = onset_source
                metadata["ScanOnset"] = datetime.fromtimestamp(
                    scan_onset).isoformat()

        with open(json_fname, "w") as f:
            json.dump(metadata, f)

        written["physio"] = tsv_path
        self._logger.info(f"Physio data saved to {tsv_path.resolve()}")

        if auto:
            self._auto_saved[save_key] = written
        elif replace:
            # Remove files of the automatic save that were not rewritten
            for kind, path in replace.items():
                if kind in written:
                    continue
                stale = [path]
                if kind == "physio":
                    stale.append(path.with_suffix(".json"))
                for f in stale:
                    if f.is_file():
                        f.unlink()
            self._logger.info(
                "Replaced the automatically saved files of this scan: "
                f"{', '.join(str(p.name) for p in replace.values())}"
            )
        # endregion

        return {"onset": onset, "offset": offset, "path": tsv_path}

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def _output_path(path, kind, replace):
        """Path to write the `kind` output to.

        If the automatic save of the same scan wrote this kind of file, that
        file is overwritten. Otherwise an existing file is not overwritten,
        and _1, _2, ... is appended to the file name.
        """
        if kind in replace:
            return Path(replace[kind])
        path = Path(path)
        stem, ii = path.stem, 0
        while path.is_file():
            ii += 1
            path = path.with_name(f"{stem}_{ii}{path.suffix}")
        return path

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _clean_resamp(self, v, tstamp=None):
        """Fill NaN in v (in place): NaN at the ends with the nearest valid
        value, and others by linear interpolation over tstamp."""
        idx_bad = np.argwhere(np.isnan(v)).ravel()
        if len(idx_bad) == 0:
            return v

        idx_good = np.setdiff1d(np.arange(0, len(v)), idx_bad)
        if idx_good[0] > 0:
            v[: idx_good[0]] = v[idx_good[0]]
        if idx_good[-1] < len(v) - 1:
            v[idx_good[-1]:] = v[idx_good[-1]]

        idx_bad = np.argwhere(np.isnan(v)).ravel()
        if len(idx_bad):
            x = np.arange(0, len(v))
            idx_good = np.setdiff1d(x, idx_bad)
            if tstamp is None:
                tstamp = x
            v[idx_bad] = interp1d(
                tstamp[idx_good], v[idx_good], bounds_error=None
            )(tstamp[idx_bad])

        return v

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _restart_dummy(self):
        """Start the Dummy device with the current simulation files.
        It is restarted if it is already running, so that new files are
        used."""
        if self._recorder_type == "Dummy":
            self.stop_recording()
            self._recorder_type = None
        self.set_device("Dummy")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def send_dummy_pulse(self, tstamp=None):
        """Send a TTL pulse to the Dummy device.
        tstamp: time of the pulse (time.time()). None: the time it arrives.
        A simulator sends the planned times so that the RPC delay does not
        change the TTL timing."""
        if self._recorder_type == "Dummy" and self.is_recording():
            self._logger.debug("Sending dummy pulse")
            if tstamp is None:
                self._rec_proc_pipe.send("PULSE")
            else:
                self._rec_proc_pipe.send(("PULSE", float(tstamp)))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_scan_onset_bkwd(self, TR=None):
        """Set the scan onset by going back from the latest TTL.

        With TR, the walk back follows the TTL pattern (find_series_onset).
        Without TR, the onset is the first TTL after the last interval
        longer than 1.5 times the shortest interval.
        """
        data = self.dump()
        if data is None:
            return

        ttl_onsets = data["ttl_onsets"]
        ttl_onsets = ttl_onsets[~np.isnan(ttl_onsets)]
        if len(ttl_onsets) == 0:
            return

        if TR is not None:
            # Walk back from the latest TTL while the TTL pattern of the
            # scan continues (TTL per volume or per slice; see
            # find_series_onset).
            ser_onset = find_series_onset(ttl_onsets, TR)
            self._logger.info(
                "Scan onset from the TTL pattern: "
                f"{datetime.fromtimestamp(ser_onset).isoformat()}"
            )
            self.scan_onset = ser_onset
            return

        elif len(ttl_onsets) == 1:
            ser_onset = ttl_onsets[0]

        else:
            ttl_interval = np.diff(ttl_onsets)
            interval_thresh = np.nanmin(ttl_interval) * 1.5
            long_intervals = np.argwhere(
                ttl_interval > interval_thresh).ravel()
            if len(long_intervals) == 0:
                ser_onset = ttl_onsets[0]
            else:
                ser_onset = ttl_onsets[long_intervals[-1] + 1]

        self.scan_onset = ser_onset

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def RPC_handler(self, call, cmd_pipe=None):
        """
        Handle an RPC call. A returned str or bytes is sent as is, and other
        objects are pickled (pack_data) by the server.
        Note: This runs in the RPC server thread. Operations that must run
        in the main thread are requested via cmd_pipe.
        """

        if call == "ping":
            return pack_data("pong")

        elif call == "WAIT_TTL_ON":
            if hasattr(self, "_rec_proc_pipe") and self._rec_proc_pipe:
                self._rec_proc_pipe.send("WAIT_TTL_ON")

        elif call == "CANCEL_WAIT_TTL":
            if hasattr(self, "_rec_proc_pipe") and self._rec_proc_pipe:
                self._rec_proc_pipe.send("WAIT_TTL_OFF")

        elif call == "TTL_PULSE":
            if self._recorder_type == "Dummy":
                self.send_dummy_pulse()

        elif call == "START_SCAN":
            self._scan_start_time = time.time()
            self._no_ttl_warned = False
            self._series_active = True
            self._fmri_vols.clear()
            self._ttl_alarm_done = False
            self.inScan = True
            self._plot_send("SCAN_ON")

        elif call == "END_SCAN":
            self._series_active = False
            self.inScan = False
            self._plot_send("SCAN_OFF")

        elif call == "SHOW":
            self._plot_send("SHOW")

        elif call == "HIDE":
            self._plot_send("HIDE")

        elif call == "GET_GEOMETRY":
            return self._plot_request("GET_GEOMETRY")

        elif call == "GET_WINDOW_STATE":
            state = self._plot_request("GET_WINDOW_STATE")
            return state if state is not None else "None"

        elif call == "GET_SCAN_ONSET":
            if hasattr(self, "_scan_onset") and self._scan_onset is not None:
                kwds = self._scan_onset.get_property()
                return kwds
            else:
                return None

        elif call == "GET_RBUF":
            if hasattr(self, "_rbuf") and self._rbuf is not None:
                send_kwds = {}
                for lab, rb in self._rbuf.items():
                    send_kwds[lab] = rb.get_property()
                return send_kwds
            else:
                return None

        elif call == "QUIT":
            # Let mainloop call end(); shutting down the socket server from
            # its own handler thread would deadlock.
            if cmd_pipe is not None:
                cmd_pipe.send("QUIT")

        elif type(call) is tuple:  # Call with arguments
            try:
                if call[0] == "SAVE_PHYSIO_DATA":
                    # (onset, len_sec, fname_fmt[, est_onset, est_tol,
                    #  is_phantom])
                    onset, len_sec, prefix = call[1:4]
                    est_onset = call[4] if len(call) > 4 else None
                    est_tol = call[5] if len(call) > 5 else None
                    is_phantom = call[6] if len(call) > 6 else None
                    self.save_physio_data(
                        onset, len_sec, prefix, est_onset=est_onset,
                        est_tol=est_tol, is_phantom=is_phantom)

                elif call[0] == "FMRI_VOLUME":
                    # (acquisition time or None, arrival time, TR)
                    self._fmri_vols.append(tuple(call[1:4]))

                elif call[0] == "MRI_SERIES":
                    # (study UID, series number, description)
                    self._mri_series = (time.time(),) + tuple(call[1:4])

                elif call[0] == "TTL_PULSE":
                    # call[1]: time of the pulse (time.time())
                    if self._recorder_type == "Dummy":
                        self.send_dummy_pulse(call[1])

                elif call[0] == "SET_FNAME_FMT":
                    self.fname_fmt = call[1]

                elif call[0] == "SET_SCAN_START_BACKWARD":
                    TR = call[1]
                    self.set_scan_onset_bkwd(TR)

                elif call[0] == "SET_GEOMETRY":
                    geometry = call[1]
                    self._plot_send(("RESIZE", geometry))

                elif call[0] == "SET_CONFIG":
                    conf = call[1]
                    self.set_config(conf)

                elif call[0] == "SET_CARD_F":
                    self.sim_card_f = Path(call[1])

                elif call[0] == "SET_RESP_F":
                    self.sim_resp_f = Path(call[1])

                elif call[0] == "SET_SAMPLE_FREQ":
                    self.sample_freq = float(call[1])

                elif call[0] == "SET_REC_DEV":
                    self.set_device(call[1])

                elif call[0] == "SET_PARAMS":
                    params = call[1]
                    for par, val in params.items():
                        if hasattr(self, par):
                            setattr(self, par, val)

                            if par == "wait_ttl_on":
                                if (
                                    hasattr(self, "_rec_proc_pipe") and
                                    self._rec_proc_pipe
                                ):
                                    if self.wait_ttl_on:
                                        self._rec_proc_pipe.send(
                                            "WAIT_TTL_ON")
                                    else:
                                        self._rec_proc_pipe.send(
                                            "WAIT_TTL_OFF")

                elif call[0] == "GET_PARAMS":
                    params = call[1]
                    ret_params = {}
                    for par in params:
                        if hasattr(self, par):
                            ret_params[par] = getattr(self, par)
                        else:
                            ret_params[par] = None
                    return ret_params

                elif call[0] == "START_DUMMY_FEEDING_WITH_FILES":
                    self.sim_card_f = Path(call[1][0])
                    self.sim_resp_f = Path(call[1][1])
                    self.sim_bids_f = None
                    self.sample_freq = float(call[1][2])
                    sigs = []
                    for f in (self.sim_card_f, self.sim_resp_f):
                        try:
                            sigs.append(np.loadtxt(f))
                        except Exception:
                            pass  # DummyRecording reports the error
                    self._set_sim_signal_range(*sigs)
                    self._restart_dummy()

                elif call[0] == "START_DUMMY_FEEDING_WITH_BIDS":
                    # call[1]: BIDS *_physio.tsv[.gz] file
                    bids = load_bids_physio(call[1])  # Raises if invalid
                    self.sim_bids_f = Path(call[1])
                    self.sim_card_f = None
                    self.sim_resp_f = None
                    self.sample_freq = bids["SamplingFrequency"]
                    self._set_sim_signal_range(
                        bids["cardiac"], bids["respiratory"])
                    self._restart_dummy()

            except Exception as e:
                errstr = str(e) + "\n" + traceback.format_exc()
                self._logger.error(f"Error in RPC call {call[0]}: {errstr}")
                return f"Error: Failed to load physio files - {errstr}"

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def mainloop(self):
        """Main thread loop: handle the messages from the recording, plot
        and RPC threads, and check the TTLs (scan end, missing TTL) every
        second."""
        if _display_available():
            try:
                self.open_plot()
            except tk.TclError as e:
                self._logger.warning(f"Run without the plot window: {e}")
                self._plot = None
                self._plot_proc_pipe = None
        else:
            # No display (e.g., a closed or stalled X forwarding): keep
            # recording and serving RPC without the plot window.
            self._logger.warning(
                "Run without the plot window: the display is not available.")
            self._plot = None
            self._plot_proc_pipe = None

        while True:
            msg = None
            # The recording process ends when the device is changed or reset
            # (stop_recording in the RPC thread, which also closes this
            # pipe). That must not end this loop: ending it ended the
            # physio process, and the launcher then stopped all of RT-MRI.
            rec_pipe = self._rec_proc_pipe
            if rec_pipe is not None:
                try:
                    if rec_pipe.poll():
                        msg = rec_pipe.recv()
                        if msg == "END_RECORDING":
                            self._logger.debug("The recording process ended.")
                except (OSError, EOFError, ValueError):
                    pass  # Closed by stop_recording

            pipe_available = (
                self._plot_proc_pipe is not None and
                self._plot_proc_pipe.poll()
            )
            if pipe_available:
                msg = self._plot_proc_pipe.recv()
                self._logger.debug(f"Received message from plot pipe: {msg}")
                if msg == "QUIT":
                    self.end()
                    break
                elif msg == "GET_CONFIG":
                    conf = self.get_config()
                    self._plot_send(conf)

                elif msg == "SET_CONFIG":
                    conf = self._plot_proc_pipe.recv()
                    self.set_config(conf)

                elif msg == "DUMP":
                    # Old dump request (the dump dialog uses _dump_pipe)
                    self._handle_dump_request(("DUMP", None))

            dump_pipe = getattr(self, "_dump_pipe", None)
            if dump_pipe is not None and dump_pipe.poll():
                self._handle_dump_request(dump_pipe.recv())

            if self._rpc_pipe is not None and self._rpc_pipe.poll():
                msg = self._rpc_pipe.recv()
                if msg == "QUIT":
                    self.end()
                    break

            if self.inScan:
                try:
                    self._check_scan_end()
                except Exception:
                    # Do not stop the mainloop (saving and dump)
                    self._logger.error(
                        "Error in the scan end check:\n"
                        + traceback.format_exc()
                    )

            if self._mri_series is not None:
                try:
                    self._check_no_ttl_at_all()
                except Exception:
                    self._logger.error(
                        "Error in the TTL check:\n" + traceback.format_exc()
                    )

            if self._series_active:
                try:
                    self._check_ttl_missing()
                except Exception:
                    self._logger.error(
                        "Error in the TTL check:\n" + traceback.format_exc()
                    )

            time.sleep(1)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    NO_TTL_CHECK_DELAY = 3.0  # s after the series start

    def _check_no_ttl_at_all(self):
        """Warn when MRI images of a series (any type, e.g., a localizer)
        come but no TTL has been recorded at all (none in the buffer), or
        no TTL device is recording. This points to a connection problem,
        which should be fixed before fMRI. Once per study."""
        t_series, study, ser_nr, desc = self._mri_series
        if time.time() - t_series < self.NO_TTL_CHECK_DELAY:
            return  # Let a TTL that comes with the images arrive
        self._mri_series = None
        if study in self._no_ttl_warned_studies:
            return

        onsets = self.get_ttl_onsets(1)
        if onsets is not None and len(onsets) > 0:
            return  # TTLs are recorded

        self._no_ttl_warned_studies.add(study)
        if onsets is None:
            reason = ("No TTL recording device is connected "
                      f"({self._recorder_type}).")
        else:
            buf_min = self.buf_len_sec / 60
            reason = (f"No TTL has been recorded in the last "
                      f"{buf_min:g} min.")
        msg = (
            f"MRI images are arriving (series {ser_nr}: {desc}), but no "
            f"TTL (scanner pulse) is recorded. {reason}\n\n"
            "The TTL cable or the Numato board may not be connected. "
            "Check them before starting fMRI."
        )
        self._logger.error(
            f"No TTL was recorded while series {ser_nr} ({desc}) was "
            f"being scanned. {reason}")
        self._plot_send(("TTL_ALARM", msg))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _check_ttl_missing(self):
        """Warn when fMRI volumes are coming but TTLs are not (e.g., the TTL
        cable is off or the Numato board is not connected).

        The last two volumes must have been acquired (acquisition time from
        the DICOM monitor) well after the last TTL. Without the acquisition
        time, the arrival time is used with a larger margin for the
        reconstruction delay. A dialog with a repeating alarm sound is
        shown once per series (see PlotTTLPhysio.show_ttl_alarm).
        """
        if self._ttl_alarm_done or len(self._fmri_vols) < 2:
            return
        vols = list(self._fmri_vols)[-2:]
        if time.time() - vols[-1][1] > 60:
            return  # Volumes have stopped (e.g., aborted series)

        onsets = self.get_ttl_onsets(1)
        recording = onsets is not None
        last_ttl = float(onsets[-1]) if recording and len(onsets) else 0.0
        for acq_t, t_arrival, TR in vols:
            TR = float(TR) if TR else 2.0
            margin = max(2 * TR, 3.0) + 2.0
            if acq_t is not None:
                t_vol = acq_t
            else:
                t_vol = t_arrival
                margin += 30.0  # Unknown reconstruction delay
            if t_vol - last_ttl <= margin:
                return  # A TTL came around this volume

        self._ttl_alarm_done = True
        if not recording:
            reason = ("No TTL recording device is connected "
                      f"({self._recorder_type}).")
        elif last_ttl > 0:
            ago = time.time() - last_ttl
            reason = f"The last TTL was {ago:.0f} s ago."
        else:
            reason = "No TTL has been recorded."
        msg = (
            "fMRI images are arriving, but no TTL (scanner pulse) is "
            f"recorded. {reason}\n\n"
            "Check the TTL cable and the Numato board. Without TTLs, the "
            "scan onset of the physio data is estimated from the image "
            "times."
        )
        self._logger.error("No TTL during the fMRI series. " + reason)
        self._plot_send(("TTL_ALARM", msg))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _check_scan_end(self):
        """Detect the scan end by the TTLs stopping, and save the data.

        Only TTLs of this scan count: the TTLs must still be coming at
        START_SCAN (sent when the first image of the series arrives, after
        the first TTLs). Otherwise, the last TTLs of the previous scan would
        end this scan at once and save the previous data under its name.
        """
        ttl_onsets = self.get_ttl_onsets(10)
        if ttl_onsets is None or len(ttl_onsets) < 3:
            return  # Not recording, or too few TTLs
        tr = np.median(np.diff(ttl_onsets))
        silent_limit = max(2 * tr, 2.5)
        start_t = self._scan_start_time
        if start_t is not None and ttl_onsets[-1] < start_t - silent_limit:
            # No TTL of this scan (yet)
            if not self._no_ttl_warned and \
                    time.time() - start_t > 2 * silent_limit:
                self._logger.warning(
                    "No TTL during this scan, so the scan end cannot be "
                    "detected from the TTLs."
                )
                self._no_ttl_warned = True
            return
        silent_time = time.time() - ttl_onsets[-1]
        if silent_time > silent_limit:
            self._logger.debug("Detected scan end due to silent TTL.")
            self.inScan = False
            self._plot_send("SCAN_OFF")
            self.save_physio_data(None, None, self.fname_fmt, auto=True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _handle_dump_request(self, msg):
        """("DUMP", onset) from the dump dialog: save the buffer from onset
        (None: the oldest available data) to the latest data, and send the
        result back."""
        if not (isinstance(msg, tuple) and msg[0] == "DUMP"):
            return
        onset = msg[1]
        req_id = msg[2] if len(msg) > 2 else None
        try:
            tstr = datetime.now().strftime("%Y%m%dT%H%M%S")
            fname_fmt = str(self.dump_dir / f"dump_{tstr}_physio.tsv")
            self._logger.info(f"Saving dump files to: {self.dump_dir}")
            info = self.save_physio_data(
                onset=onset, fname_fmt=fname_fmt, nosignal_nosave=False,
                whole_buffer=True,
            )
            if info is None:
                result = ("DUMP_FAILED", "No data to save (see the log).")
            else:
                info["path"] = str(info["path"])
                result = ("DUMP_DONE", info)
        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            self._logger.error(errstr)
            result = ("DUMP_FAILED", str(e))
        try:
            self._dump_pipe.send(result + (req_id,))
        except Exception:
            pass

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end(self):
        """Stop the recording, the plot and the RPC server, and delete the
        ring buffer files."""
        self.stop_recording()
        # The plot window runs in its own process. Stop it via the pipe;
        # touching its Tk root from this process breaks the X connection.
        self.stop_plot()
        self.socket_srv.shutdown()
        if hasattr(self, "_rbuf") and self._rbuf is not None:
            for lab, rbuf in self._rbuf.items():
                rbuf.clear_buffer_files()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        self.end()


# %% main =====================================================================
if __name__ == "__main__":
    LOG_FILE = (
        Path(__file__).resolve().parents[1]
        / "log"
        / f"{Path(__file__).resolve().stem}.log"
    )
    parser = argparse.ArgumentParser(description="RT physio")
    parser.add_argument(
        "--device",
        default=None,
        choices=["Numato", "GE", "Dummy"],
        help="Recording device. Numato: TTL and physio from the Numato GPIO "
        "device. GE: physio from the GE serial port and TTL from the Numato "
        "GPIO device. The default is automatic selection (Numato, GE).",
    )
    parser.add_argument(
        "--sample_freq",
        type=float,
        default=DEFAULT_SAMPLE_FREQ,
        help="Sampling frequency (Hz)",
    )
    parser.add_argument(
        "--buf_len_sec", type=float, default=3600,
        help="Recording buffer length (s)"
    )
    parser.add_argument("--log_file", default=LOG_FILE, help="Log file path")
    parser.add_argument(
        "--rpc_socket_name", default="RtTTLPhysioSocketServer",
        help="RPC socket server name"
    )
    parser.add_argument(
        "--config_path",
        default=str(Path.home() / ".RTPSpy" / "rtmri_config.json"),
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--rt_mri_main_address_name",
        default="localhost:RtMriMainSocketServer",
        help="RtMRIMain RPC address and socket name",
    )
    parser.add_argument(
        "--geometry", default=None,
        help="Plot window geometry (WxH+X+Y). Default: fit to the screen"
    )
    parser.add_argument(
        "--disable_close", action="store_true",
        help="Disable the close button"
    )
    parser.add_argument(
        "--save_ttl",
        action="store_true",
        help="Save the TTL onset and offset times (off by default when "
        "run from the command line)",
    )
    parser.add_argument("--debug", action="store_true",
                        help="Log DEBUG messages to the console")

    args = parser.parse_args()
    device = args.device
    sample_freq = args.sample_freq
    buf_len_sec = args.buf_len_sec
    log_file = args.log_file
    rpc_socket_name = args.rpc_socket_name
    config_path = args.config_path
    rt_mri_main_address_name = [
        args.rt_mri_main_address_name.split(":")[0],
        None,
        args.rt_mri_main_address_name.split(":")[1],
    ]
    geometry = args.geometry
    disable_close = args.disable_close
    save_ttl = args.save_ttl
    debug = args.debug

    if debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format=(
                "%(asctime)s.%(msecs)03d,[%(levelname)s],%(name)s,%(message)s"
            ),
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            filename=log_file,
            filemode="a",
            format=(
                "%(asctime)s.%(msecs)03d,[%(levelname)s],%(name)s,%(message)s"
            ),
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    kwarg = {
        "device": device,
        "buf_len_sec": buf_len_sec,
        "sample_freq": sample_freq,
        "rpc_socket_name": rpc_socket_name,
        "config_path": config_path,
        "rt_mri_main_address_name": rt_mri_main_address_name,
        "geometry": geometry,
        "disable_close": disable_close,
        "save_ttl": save_ttl,
    }

    rt_physio = RtPhysio(**kwarg)

    # Run mainloop
    rt_physio.mainloop()
