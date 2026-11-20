#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RT-MRI Simulator (PyQt6)

A GUI tool to simulate the real-time data of an MRI scan for testing RTPSpy,
without the scanner:

- DICOM: the files of a recorded session are copied, series by series, into
  the output (watch) directory at the time each image was made on the
  scanner (ContentTime) plus the transfer delay.
- TTL: a pulse is sent to the rt_physio process (RPC) at the acquisition
  time of each volume. The planned time is sent with the pulse, so that the
  RPC delay does not change the recorded TTL time.
- Physio: the dummy recorder of rt_physio is fed with cardiac and
  respiration signals, from two 1D text files or from a BIDS
  *_physio.tsv[.gz] file (with its JSON sidecar). rt_physio plays them in a
  loop.
- 'T' key: on a PC without the TTL device, the scan start is triggered by
  the 'T' key in RTPSpy. The simulator can press the key (xdotool) at the
  scan start or at every volume, in the window whose title matches a
  pattern (default: the RTPSpy main window).

This is the PyQt6 version of rtmri_simulator.py (Tk), which was ported from
RT-MRI simulation/rtmri_simulator.py. The series cache in each session
directory (.rtmri_series_cache.csv) is shared with the Tk version.

Author: mmisaki@laureateinstitute.org
"""

# %% import libraries =========================================================
import sys
import json
import time
import queue
import shutil
import logging
import argparse
import threading
import subprocess
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pydicom
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal

try:
    from rtpspy.dicom_reader import DicomReader
    from rtpspy.rpc_socket_server import RPCSocketCom
    from rtpspy import gui_theme
except ImportError:
    from dicom_reader import DicomReader
    from rpc_socket_server import RPCSocketCom
    import gui_theme

APP_DIR = Path(__file__).resolve().parent
CACHE_NAME = ".rtmri_series_cache.csv"  # Same as the Tk version


# %% DICOM time helpers =======================================================
# The same functions as in rtmri_simulator.py (Tk version).
def dicom_time_to_sec(value):
    """Convert a DICOM time to seconds of the day.

    value: TM (HHMMSS.ffffff) or DT (YYYYMMDDHHMMSS.ffffff) string, or the
    number pandas reads from the cache CSV (e.g., 93359.826 for TM
    "093359.826"). A TM must not be used as a number directly: 093400.05 -
    093359.826 is 40.22, not 0.22 s.
    """
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        if not np.isfinite(v):
            return np.nan
        if v >= 1e6:  # DT read as a number: drop the date
            v = v % 1e6
        h = int(v // 10000)
        m = int(v // 100) % 100
        return h * 3600 + m * 60 + (v - h * 10000 - m * 100)
    txt = str(value).strip()
    if not txt:
        return np.nan
    if len(txt.split(".")[0]) > 6:  # DT: drop the date
        txt = txt[8:]
    try:
        h = int(txt[0:2])
        m = int(txt[2:4]) if len(txt) >= 4 else 0
        sec = float(txt[4:]) if len(txt) > 4 else 0.0
    except ValueError:
        return np.nan
    return h * 3600 + m * 60 + sec


def content_sec(content_time):
    """Seconds of the day of the ContentTime values (a pandas Series),
    continued over midnight within a series."""
    sec = content_time.map(dicom_time_to_sec).astype(float)
    if np.nanmax(sec) - np.nanmin(sec) > 43200:  # Over midnight
        sec = sec.where(sec >= 43200, sec + 86400)
    return sec


def read_acq_sec(dicom_file):
    """Acquisition time of a DICOM file in seconds of the day (NaN if the
    file has no acquisition time)."""
    try:
        dcm = pydicom.dcmread(
            dicom_file, stop_before_pixels=True,
            specific_tags=["AcquisitionDateTime", "AcquisitionTime"],
        )
    except Exception:
        return np.nan
    for tag in ("AcquisitionDateTime", "AcquisitionTime"):
        if tag in dcm and dcm.get(tag):
            return dicom_time_to_sec(dcm.get(tag))
    return np.nan


def series_timing(series_df, transfer_delay=0.0):
    """
    Times (s from the simulation start) to copy each file and to send each
    TTL, as on the scanner:

    - TTL: the acquisition time of each volume (AcquisitionNumber).
    - File: its image creation time (ContentTime) + transfer_delay, the time
      to transfer the file to the export directory.

    The first file therefore comes TR + reconstruction + transfer after the
    first TTL. Without acquisition times (an old cache), a TTL is sent one TR
    before the first file of each volume (no reconstruction delay).

    series_df needs the columns ContentSec, AcqSec, AcquisitionNumber, and
    ImageType, in the order of the files to copy.
    """
    file_sec = series_df["ContentSec"].values.astype(float)
    acq_nrs = series_df["AcquisitionNumber"].values
    acq_sec = series_df["AcqSec"].values.astype(float)

    # Over midnight
    if np.nanmax(file_sec) - np.nanmin(file_sec) > 43200:
        file_sec = np.where(file_sec < 43200, file_sec + 86400, file_sec)
        acq_sec = np.where(acq_sec < 43200, acq_sec + 86400, acq_sec)

    first_file, ttl = {}, {}
    for nr, a_sec, f_sec in zip(acq_nrs, acq_sec, file_sec):
        if nr not in first_file:
            first_file[nr] = f_sec
            ttl[nr] = a_sec
    acq_list = sorted(first_file, key=lambda nr: first_file[nr])
    ff = np.array([first_file[nr] for nr in acq_list])
    ttl_sec = np.array([ttl[nr] for nr in acq_list])
    if np.isnan(ttl_sec).any():
        is_fmri = (
            "FMRI" in str(series_df.ImageType.values[0]) and len(ff) > 1
        )
        TR = float(np.median(np.diff(ff))) if is_fmri else 0.0
        ttl_sec = np.where(np.isnan(ttl_sec), ff - TR, ttl_sec)
    ttl_sec = np.sort(ttl_sec)

    file_sec = file_sec + transfer_delay
    t_ref = min(ttl_sec[0], np.nanmin(file_sec))
    return file_sec - t_ref, ttl_sec - t_ref


# %% Session search ===========================================================
def _study_description(dicom_files):
    """StudyDescription of the first readable file, or 'Unknown'."""
    reader = DicomReader()
    for dcm_file in dicom_files:
        try:
            info = reader.read_dicom_info(dcm_file, timeout=0.1)
            if info and info.get("StudyDescription"):
                return str(info["StudyDescription"])
        except Exception:
            continue
    return "Unknown"


def _is_series_dir(d):
    """S001, Ser001, etc."""
    return (
        (d.name.startswith("S") and d.name[1:].isdigit())
        or (d.name.startswith("Ser") and d.name[3:].isdigit())
    )


def find_sessions(base_dir, max_depth=3):
    """Find the session directories under base_dir (the same rules as the Tk
    version).

    A session directory has DICOM files (*.dcm) directly or in a 'dicom'
    subdirectory, or has series subdirectories (S001, Ser001, ...) and no
    other subdirectories.

    Returns a list of dicts with 'name' (path relative to base_dir), 'path',
    'files', and 'study_description', sorted by name.
    """
    base_dir = Path(base_dir)
    sessions = []

    def _add(cur, dicom_files):
        name = (cur.name if cur == base_dir
                else str(cur.relative_to(base_dir)))
        sessions.append({
            "name": name,
            "path": str(cur),
            "files": len(dicom_files),
            "study_description": _study_description(dicom_files[:3]),
        })

    def _search(cur, depth):
        if depth > max_depth or not cur.is_dir():
            return
        dicom_subdir = cur / "dicom"
        if any(cur.glob("*.dcm")) or (
            dicom_subdir.is_dir() and any(dicom_subdir.glob("**/*.dcm"))
        ):
            _add(cur, list(cur.glob("**/*.dcm")))
            return

        subdirs = [d for d in cur.iterdir() if d.is_dir()]
        series_dirs = [d for d in subdirs if _is_series_dir(d)]
        if series_dirs:
            dicom_files = list(cur.glob("**/*.dcm"))
            if dicom_files:
                # A directory with series and other subdirectories is likely
                # a container of sessions; search deeper.
                is_container = (
                    len(subdirs) > len(series_dirs) and depth < max_depth
                )
                if not is_container:
                    _add(cur, dicom_files)
                    return

        for d in subdirs:
            if not d.name.startswith("."):
                _search(d, depth + 1)

    try:
        _search(base_dir, 0)
    except PermissionError:
        pass

    unique = {s["path"]: s for s in sessions}
    return sorted(unique.values(), key=lambda s: s["name"])


# %% Series helpers ===========================================================
def phase_partners(dicom_series):
    """Map a series number to its phase series number.

    The phase images of a series are in a series with the description
    '<description>_Pha' (Siemens). They are copied with the magnitude images,
    so a phase series is not simulated by itself. When the description is
    repeated (e.g., several runs), the phase series is the first one with a
    larger series number than the magnitude series.
    """
    first = (
        dicom_series.groupby("SeriesNumber")["SeriesDescription"]
        .first().astype(str)
    )
    partners = {}
    used = set()
    for ser_nr, desc in first.items():
        if desc.endswith("_Pha"):
            continue
        cands = [
            nr for nr, d in first.items()
            if d == f"{desc}_Pha" and nr > ser_nr and nr not in used
        ]
        if cands:
            partners[ser_nr] = min(cands)
            used.add(min(cands))
    return partners


# %% Signals from the worker threads ==========================================
class _Signals(QtCore.QObject):
    """Signals emitted from the worker threads. Qt queues them to the main
    thread, where the widgets are updated."""

    log = pyqtSignal(str)
    error = pyqtSignal(str)

    sessions_loaded = pyqtSignal(object)  # list of session dicts
    series_progress = pyqtSignal(int, int, str)  # current, total, message
    series_loaded = pyqtSignal(object, object)  # session path, DataFrame

    sim_series = pyqtSignal(int, int)  # series number, number of files
    sim_file = pyqtSignal(int, int, str)  # image nr, total, file name
    sim_wait = pyqtSignal(float)  # seconds to the next series
    sim_finished = pyqtSignal(int, int, int)  # series nr, image nr, total

    ttl_sent = pyqtSignal(int)  # total number of TTLs sent
    key_sent = pyqtSignal(int)  # total number of 'T' keys pressed


# %% RTMRISimulatorQt =========================================================
class RTMRISimulatorQt(QtWidgets.QMainWindow):
    """RT-MRI simulator window"""

    # GUI settings saved at closing and restored at start
    CONFIG_FILE = Path.home() / ".RTPSpy" / "rtmri_simulator_qt_config.json"

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(
        self,
        rpc_physio_address_name=None,
        rpc_config_path=Path.home() / ".RTPSpy" / "rtmri_config.json",
    ):
        super().__init__()
        self.setWindowTitle("RT-MRI Simulator")
        icon = APP_DIR / "RT-MRI_icon.png"
        if icon.is_file():
            self.setWindowIcon(QtGui.QIcon(str(icon)))

        self._logger = logging.getLogger("RTMRISimulatorQt")
        if rpc_physio_address_name is None:
            rpc_physio_address_name = [
                "localhost", None, "RtTTLPhysioSocketServer"]
        self.rpc_physio_com = RPCSocketCom(
            list(rpc_physio_address_name), rpc_config_path)

        # State
        self.config = {
            "src_data_dir": str(APP_DIR / "simulation_data"),
            "simulation_output_dir": str(APP_DIR / "RTMRI_simulation"),
            "auto_advance": False,
            "series_interval": 10.0,
            # Time (s) to transfer an image file from the scanner to the
            # export directory (measured on the Siemens RT PC: 0.1-0.25 s)
            "transfer_delay": 0.2,
            "send_ttl": True,
            # 'T' key press with xdotool: off, 'start' (the first volume of
            # a series) or 'every' (every volume), to the window whose
            # title matches t_key_window (regular expression; empty: the
            # focused window)
            "t_key": False,
            "t_key_mode": "start",
            "t_key_window": "^RTPSpy$",
            "physio_source": "1D",  # '1D' or 'BIDS'
            "card_file": "",
            "resp_file": "",
            "physio_freq": 100,
            "bids_physio_file": "",
            "current_session_path": "",
        }
        self.current_session_path = None
        self.dicom_series = None
        self.partners = {}
        self.current_series_nr = None
        self.current_image_nr = 0

        self.simulation_running = False
        self._stop_event = threading.Event()
        self._stop_reset = False
        self._select_after_load = None
        self._scan_cancel = threading.Event()
        self._series_load_id = 0
        self._sim_thread = None

        # TTL sender thread (the RPC call must not delay the file copies)
        self._ttl_queue = queue.Queue()
        self._ttl_count = 0
        self._ttl_thread = threading.Thread(
            target=self._ttl_sender, daemon=True)

        # 'T' key sender thread (xdotool)
        self._xdotool = shutil.which("xdotool")
        self._key_queue = queue.Queue()
        self._key_count = 0
        self._key_thread = threading.Thread(
            target=self._key_sender, daemon=True)

        self.sig = _Signals()
        self.sig.log.connect(self.log_message)
        self.sig.error.connect(self.err_message)
        self.sig.sessions_loaded.connect(self._on_sessions_loaded)
        self.sig.series_progress.connect(self._on_series_progress)
        self.sig.series_loaded.connect(self._on_series_loaded)
        self.sig.sim_series.connect(self._on_sim_series)
        self.sig.sim_file.connect(self._on_sim_file)
        self.sig.sim_wait.connect(self._on_sim_wait)
        self.sig.sim_finished.connect(self._on_sim_finished)
        self.sig.ttl_sent.connect(self._show_counts)
        self.sig.key_sent.connect(self._show_counts)

        self._build_gui()
        self.load_config()
        self._apply_config_to_widgets()
        self._ttl_thread.start()
        self._key_thread.start()

        self.resize(1100, 900)
        self.load_dicom_sessions(
            select_session=self.config.get("current_session_path"))

    # =========================================================================
    # region: GUI
    # =========================================================================
    def _build_gui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)

        vsplit = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        vbox.addWidget(vsplit)

        top = QtWidgets.QWidget()
        hbox = QtWidgets.QHBoxLayout(top)
        hbox.setContentsMargins(0, 0, 0, 0)
        self.left_panel = self._build_session_panel()
        hbox.addWidget(self.left_panel, stretch=3)
        hbox.addWidget(self._build_simulation_panel(), stretch=2)
        vsplit.addWidget(top)

        bottom = QtWidgets.QWidget()
        bbox = QtWidgets.QVBoxLayout(bottom)
        bbox.setContentsMargins(0, 0, 0, 0)
        bbox.addWidget(self._build_physio_panel())
        bbox.addWidget(self._build_log_panel())
        vsplit.addWidget(bottom)
        vsplit.setStretchFactor(0, 3)
        vsplit.setStretchFactor(1, 2)

        self.statusBar().showMessage("Ready")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def _path_edit():
        """Read-only line edit for a path (the text can be copied)"""
        ed = QtWidgets.QLineEdit()
        ed.setReadOnly(True)
        return ed

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _build_session_panel(self):
        panel = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)

        # --- DICOM sessions ---
        grp = QtWidgets.QGroupBox("DICOM Sessions")
        g = QtWidgets.QVBoxLayout(grp)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Source Root:"))
        self.src_dir_edit = self._path_edit()
        row.addWidget(self.src_dir_edit, stretch=1)
        self.src_refresh_btn = QtWidgets.QPushButton("Refresh")
        self.src_refresh_btn.clicked.connect(
            lambda: self.load_dicom_sessions())
        row.addWidget(self.src_refresh_btn)
        self.src_set_btn = QtWidgets.QPushButton("Set")
        self.src_set_btn.clicked.connect(self.set_src_dir)
        row.addWidget(self.src_set_btn)
        g.addLayout(row)

        self.session_tree = QtWidgets.QTreeWidget()
        self.session_tree.setHeaderLabels(
            ["Name", "Files", "Study Description"])
        self.session_tree.setRootIsDecorated(False)
        self.session_tree.setAlternatingRowColors(True)
        self.session_tree.header().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.session_tree.itemSelectionChanged.connect(
            self.on_session_select)
        g.addWidget(self.session_tree)
        vbox.addWidget(grp, stretch=1)

        # --- Series ---
        grp = QtWidgets.QGroupBox("Series Information")
        g = QtWidgets.QVBoxLayout(grp)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Current Session:"))
        self.current_session_lbl = QtWidgets.QLabel("No session selected")
        f = self.current_session_lbl.font()
        f.setItalic(True)
        self.current_session_lbl.setFont(f)
        row.addWidget(self.current_session_lbl, stretch=1)
        self.rescan_btn = QtWidgets.QPushButton("Rescan")
        self.rescan_btn.setToolTip(
            "Read the DICOM headers again (ignore the cache)")
        self.rescan_btn.clicked.connect(self.handle_rescan_click)
        row.addWidget(self.rescan_btn)
        g.addLayout(row)

        self.series_loading_lbl = QtWidgets.QLabel("")
        self.series_loading_bar = QtWidgets.QProgressBar()
        self.series_loading_lbl.hide()
        self.series_loading_bar.hide()
        g.addWidget(self.series_loading_lbl)
        g.addWidget(self.series_loading_bar)

        self.series_tree = QtWidgets.QTreeWidget()
        self.series_tree.setHeaderLabels(
            ["Series", "Files", "Image Type", "Description"])
        self.series_tree.setRootIsDecorated(False)
        self.series_tree.setAlternatingRowColors(True)
        hdr = self.series_tree.header()
        for col in (0, 1):
            hdr.setSectionResizeMode(
                col, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.series_tree.setColumnWidth(2, 220)
        self.series_tree.itemSelectionChanged.connect(self.on_series_select)
        g.addWidget(self.series_tree)
        vbox.addWidget(grp, stretch=3)

        return panel

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _build_simulation_panel(self):
        grp = QtWidgets.QGroupBox("Simulation Control")
        vbox = QtWidgets.QVBoxLayout(grp)

        # --- Output directory ---
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Output:"))
        self.out_dir_edit = self._path_edit()
        row.addWidget(self.out_dir_edit, stretch=1)
        self.out_set_btn = QtWidgets.QPushButton("Set")
        self.out_set_btn.clicked.connect(self.set_output_directory)
        row.addWidget(self.out_set_btn)
        self.out_clear_btn = QtWidgets.QPushButton("Clear")
        self.out_clear_btn.setToolTip(
            "Delete all files in the output directory")
        self.out_clear_btn.clicked.connect(self.clear_output_directory)
        row.addWidget(self.out_clear_btn)
        vbox.addLayout(row)

        # --- Parameters ---
        form = QtWidgets.QFormLayout()
        self.auto_advance_chk = QtWidgets.QCheckBox(
            "Auto-advance to the next series")
        form.addRow(self.auto_advance_chk)

        self.series_interval_spin = QtWidgets.QDoubleSpinBox()
        self.series_interval_spin.setRange(0, 3600)
        self.series_interval_spin.setDecimals(1)
        self.series_interval_spin.setSuffix(" s")
        form.addRow("Series interval:", self.series_interval_spin)

        self.transfer_delay_spin = QtWidgets.QDoubleSpinBox()
        self.transfer_delay_spin.setRange(0, 60)
        self.transfer_delay_spin.setDecimals(2)
        self.transfer_delay_spin.setSingleStep(0.05)
        self.transfer_delay_spin.setSuffix(" s")
        self.transfer_delay_spin.setToolTip(
            "Time to transfer an image file from the scanner to the export "
            "directory, added to the image creation time")
        form.addRow("Transfer delay:", self.transfer_delay_spin)

        self.send_ttl_chk = QtWidgets.QCheckBox(
            "Send TTL at each volume acquisition")
        form.addRow(self.send_ttl_chk)

        krow = QtWidgets.QHBoxLayout()
        self.t_key_chk = QtWidgets.QCheckBox("Press 'T' key")
        self.t_key_chk.setToolTip(
            "Press the 'T' key with xdotool, to trigger the scan start "
            "without a TTL device")
        krow.addWidget(self.t_key_chk)
        self.t_key_mode_cmb = QtWidgets.QComboBox()
        self.t_key_mode_cmb.addItem("at scan start", "start")
        self.t_key_mode_cmb.addItem("at every volume", "every")
        krow.addWidget(self.t_key_mode_cmb)
        krow.addStretch(1)
        form.addRow(krow)
        self.t_key_window_edit = QtWidgets.QLineEdit()
        self.t_key_window_edit.setToolTip(
            "Title of the window to which the key is sent (regular "
            "expression). The window is activated before the key press. "
            "Empty: the window with the focus.")
        form.addRow("Key window:", self.t_key_window_edit)
        if self._xdotool is None:
            for w in (self.t_key_chk, self.t_key_mode_cmb,
                      self.t_key_window_edit):
                w.setEnabled(False)
                w.setToolTip("xdotool is not installed")
        vbox.addLayout(form)

        # --- Current status ---
        st = QtWidgets.QGroupBox("Current Status")
        sv = QtWidgets.QVBoxLayout(st)
        self.current_series_lbl = QtWidgets.QLabel("No series selected")
        self.current_series_lbl.setWordWrap(True)
        sv.addWidget(self.current_series_lbl)
        self.series_progress_lbl = QtWidgets.QLabel("Series Progress:")
        sv.addWidget(self.series_progress_lbl)
        self.series_progress_bar = QtWidgets.QProgressBar()
        self.series_progress_bar.setFormat("%v / %m")
        sv.addWidget(self.series_progress_bar)
        self.ttl_count_lbl = QtWidgets.QLabel("TTL sent: 0, 'T' key: 0")
        sv.addWidget(self.ttl_count_lbl)
        vbox.addWidget(st)

        # --- Control buttons ---
        self.start_btn = QtWidgets.QPushButton("Start Simulation")
        self.start_btn.setStyleSheet(
            gui_theme.qt_button_style("start", "padding: 6px;"))
        self.start_btn.clicked.connect(lambda: self.start_simulation())
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.pause_btn.clicked.connect(self.pause_simulation)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setStyleSheet(
            gui_theme.qt_button_style("stop", "padding: 6px;"))
        self.stop_btn.setToolTip(
            "Stop and go back to the start of the series")
        self.stop_btn.clicked.connect(self.stop_simulation)
        self.step_btn = QtWidgets.QPushButton("Step Forward")
        self.step_btn.setToolTip(
            "Copy the next file now (with the TTLs due before it)")
        self.step_btn.clicked.connect(lambda: self.start_simulation(
            resume=True, one_step=True))
        for b in (self.start_btn, self.pause_btn, self.stop_btn,
                  self.step_btn):
            vbox.addWidget(b)
        vbox.addStretch(1)

        self._set_control_state("idle")
        return grp

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _build_physio_panel(self):
        grp = QtWidgets.QGroupBox("TTL and Physiological Signals (rt_physio)")
        hbox = QtWidgets.QHBoxLayout(grp)

        # --- Signal source ---
        src = QtWidgets.QGridLayout()
        self.physio_1d_radio = QtWidgets.QRadioButton("1D files")
        self.physio_bids_radio = QtWidgets.QRadioButton("BIDS physio")
        self.physio_1d_radio.toggled.connect(self._update_physio_widgets)
        src.addWidget(self.physio_1d_radio, 0, 0)

        src.addWidget(QtWidgets.QLabel("Cardiac:"), 1, 0,
                      Qt.AlignmentFlag.AlignRight)
        self.card_file_edit = self._path_edit()
        src.addWidget(self.card_file_edit, 1, 1)
        self.card_set_btn = QtWidgets.QPushButton("Set")
        self.card_set_btn.clicked.connect(
            lambda: self._select_file(self.card_file_edit, "cardiac"))
        src.addWidget(self.card_set_btn, 1, 2)

        src.addWidget(QtWidgets.QLabel("Respiration:"), 2, 0,
                      Qt.AlignmentFlag.AlignRight)
        self.resp_file_edit = self._path_edit()
        src.addWidget(self.resp_file_edit, 2, 1)
        self.resp_set_btn = QtWidgets.QPushButton("Set")
        self.resp_set_btn.clicked.connect(
            lambda: self._select_file(self.resp_file_edit, "respiration"))
        src.addWidget(self.resp_set_btn, 2, 2)

        frow = QtWidgets.QHBoxLayout()
        frow.addWidget(QtWidgets.QLabel("Sampling freq.:"))
        self.physio_freq_spin = QtWidgets.QSpinBox()
        self.physio_freq_spin.setRange(1, 1000)
        self.physio_freq_spin.setSuffix(" Hz")
        frow.addWidget(self.physio_freq_spin)
        frow.addStretch(1)
        src.addLayout(frow, 3, 1)

        src.addWidget(self.physio_bids_radio, 4, 0)
        self.bids_file_edit = self._path_edit()
        self.bids_file_edit.setToolTip(
            "*_physio.tsv[.gz] with a JSON sidecar (SamplingFrequency, "
            "StartTime, Columns with 'cardiac' and/or 'respiratory')")
        src.addWidget(self.bids_file_edit, 4, 1)
        self.bids_set_btn = QtWidgets.QPushButton("Set")
        self.bids_set_btn.clicked.connect(
            lambda: self._select_file(self.bids_file_edit, "bids"))
        src.addWidget(self.bids_set_btn, 4, 2)
        src.setColumnStretch(1, 1)
        hbox.addLayout(src, stretch=1)

        # --- Buttons ---
        btns = QtWidgets.QGridLayout()
        self.start_physio_btn = QtWidgets.QPushButton("Start Physio")
        self.start_physio_btn.setToolTip(
            "Feed the dummy recorder of rt_physio with the signals")
        self.start_physio_btn.clicked.connect(self.start_physio_feeding)
        btns.addWidget(self.start_physio_btn, 0, 0)
        self.reset_physio_btn = QtWidgets.QPushButton("Stop Physio")
        self.reset_physio_btn.setToolTip(
            "Reset the recording device of rt_physio")
        self.reset_physio_btn.clicked.connect(self.reset_physio)
        btns.addWidget(self.reset_physio_btn, 0, 1)
        self.check_rpc_btn = QtWidgets.QPushButton("Check RPC")
        self.check_rpc_btn.clicked.connect(lambda: self.ping_physio_rpc())
        btns.addWidget(self.check_rpc_btn, 1, 0)
        self.send_pulse_btn = QtWidgets.QPushButton("Send TTL")
        self.send_pulse_btn.clicked.connect(self.send_manual_pulse)
        btns.addWidget(self.send_pulse_btn, 1, 1)
        self.press_t_btn = QtWidgets.QPushButton("Press 'T'")
        self.press_t_btn.setToolTip(
            "Press the 'T' key in the key window (xdotool)")
        self.press_t_btn.clicked.connect(self.press_manual_key)
        self.press_t_btn.setEnabled(self._xdotool is not None)
        btns.addWidget(self.press_t_btn, 2, 1)
        self.rpc_status_lbl = QtWidgets.QLabel("RPC: not checked")
        btns.addWidget(self.rpc_status_lbl, 3, 0, 1, 2)
        hbox.addLayout(btns)

        return grp

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _build_log_panel(self):
        grp = QtWidgets.QGroupBox("Log")
        v = QtWidgets.QVBoxLayout(grp)
        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(5000)
        v.addWidget(self.log_text)
        return grp

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _update_physio_widgets(self):
        use_1d = self.physio_1d_radio.isChecked()
        for w in (self.card_file_edit, self.card_set_btn,
                  self.resp_file_edit, self.resp_set_btn,
                  self.physio_freq_spin):
            w.setEnabled(use_1d)
        for w in (self.bids_file_edit, self.bids_set_btn):
            w.setEnabled(not use_1d)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_control_state(self, state):
        """Enable the controls for the state: 'idle' (no series), 'ready'
        (a series is selected), 'running', or 'paused'."""
        running = state == "running"
        self.start_btn.setEnabled(state in ("ready", "paused"))
        self.step_btn.setEnabled(state in ("ready", "paused"))
        self.pause_btn.setEnabled(state in ("running", "paused"))
        self.pause_btn.setText("Resume" if state == "paused" else "Pause")
        self.stop_btn.setEnabled(state in ("running", "paused"))
        for w in (self.out_set_btn, self.out_clear_btn,
                  self.transfer_delay_spin, self.send_ttl_chk):
            w.setEnabled(not running)
        if self._xdotool is not None and hasattr(self, "t_key_window_edit"):
            for w in (self.t_key_chk, self.t_key_mode_cmb,
                      self.t_key_window_edit):
                w.setEnabled(not running)
        if hasattr(self, "left_panel"):
            self.left_panel.setEnabled(not running)

    # endregion

    # =========================================================================
    # region: Logging
    # =========================================================================
    def log_message(self, message):
        """Add a message to the log and the status bar (main thread)"""
        if threading.current_thread() is not threading.main_thread():
            self.sig.log.emit(str(message))
            return
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{ts}] {message}")
        self.statusBar().showMessage(str(message).split("\n")[0])
        self._logger.info(message)

    def err_message(self, message, popup=True):
        """Add an error message (red) to the log and show a popup"""
        if threading.current_thread() is not threading.main_thread():
            self.sig.error.emit(str(message))
            return
        ts = datetime.now().strftime("%H:%M:%S")
        color = gui_theme.COLORS["red"]
        html = (str(message).replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace("\n", "<br>"))
        self.log_text.appendHtml(
            f'<span style="color:{color}">[{ts}] {html}</span>')
        self.statusBar().showMessage(str(message).split("\n")[0])
        self._logger.error(message)
        if popup:
            QtWidgets.QMessageBox.critical(
                self, "Error", str(message).split("\nTraceback")[0])

    # endregion

    # =========================================================================
    # region: Sessions and series
    # =========================================================================
    def set_src_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select the source data directory",
            self.src_dir_edit.text())
        if d:
            self.src_dir_edit.setText(d)
            self.log_message(f"Source data directory: {d}")
            self.load_dicom_sessions()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def load_dicom_sessions(self, select_session=None):
        """List the sessions in the source directory (worker thread)"""
        src = Path(self.src_dir_edit.text())
        self.session_tree.clear()
        self.series_tree.clear()
        self.dicom_series = None
        self.current_session_path = None
        self.current_session_lbl.setText("No session selected")
        self.current_series_lbl.setText("No series selected")
        self._set_control_state("idle")
        if not src.is_dir():
            self.log_message(f"Source directory {src} does not exist")
            return

        self.log_message(f"Searching DICOM sessions in {src} ...")
        self.src_refresh_btn.setEnabled(False)
        self._select_after_load = select_session

        def _work():
            try:
                sessions = find_sessions(src)
            except Exception as e:
                self.sig.error.emit(
                    f"Error searching sessions: {e}\n"
                    f"{traceback.format_exc()}")
                sessions = []
            self.sig.sessions_loaded.emit(sessions)

        threading.Thread(target=_work, daemon=True).start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _on_sessions_loaded(self, sessions):
        self.src_refresh_btn.setEnabled(True)
        select = self._select_after_load
        self._select_after_load = None
        item_to_select = None
        for s in sessions:
            it = QtWidgets.QTreeWidgetItem(
                [s["name"], str(s["files"]), s["study_description"]])
            it.setData(0, Qt.ItemDataRole.UserRole, s["path"])
            it.setTextAlignment(1, Qt.AlignmentFlag.AlignCenter)
            self.session_tree.addTopLevelItem(it)
            if select and Path(select) == Path(s["path"]):
                item_to_select = it
        self.log_message(
            f"Found {len(sessions)} DICOM sessions in "
            f"{self.src_dir_edit.text()}")
        if item_to_select is not None:
            self.session_tree.setCurrentItem(item_to_select)
            self.session_tree.scrollToItem(item_to_select)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def on_session_select(self):
        items = self.session_tree.selectedItems()
        if not items or self.simulation_running:
            return
        it = items[0]
        session_path = Path(it.data(0, Qt.ItemDataRole.UserRole))
        desc = it.text(2)
        label = it.text(0)
        if desc and desc != "Unknown":
            label += f" ({desc})"
        self.current_session_lbl.setText(label)
        self.log_message(f"Selected session: {session_path}")
        self._load_series(session_path, rescan=False)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def handle_rescan_click(self):
        if self.rescan_btn.text() == "Cancel":
            self._scan_cancel.set()
            self.log_message("Series scan cancelled")
            return
        if self.current_session_path is None:
            items = self.session_tree.selectedItems()
            if not items:
                self.err_message("No session selected")
                return
            self.current_session_path = Path(
                items[0].data(0, Qt.ItemDataRole.UserRole))
        self._load_series(self.current_session_path, rescan=True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _load_series(self, session_path, rescan=False):
        """Read the series information of a session (worker thread). The
        result is cached in the session directory."""
        self._scan_cancel.set()  # Cancel a running scan
        self._series_load_id += 1
        load_id = self._series_load_id
        cancel = threading.Event()
        self._scan_cancel = cancel

        self.current_session_path = session_path
        self.dicom_series = None
        self.series_tree.clear()
        self.current_series_lbl.setText("No series selected")
        self._set_control_state("idle")
        self.series_loading_lbl.setText("Checking cache...")
        self.series_loading_lbl.show()
        self.series_loading_bar.setValue(0)
        self.series_loading_bar.show()
        self.rescan_btn.setText("Cancel")

        def _work():
            df = None
            try:
                cache = session_path / CACHE_NAME
                if not rescan and cache.is_file():
                    try:
                        df = pd.read_csv(cache)
                        self.sig.log.emit("Loaded series info from cache")
                    except Exception as e:
                        self.sig.log.emit(f"Error reading the cache: {e}")
                        df = None
                if df is None:
                    df = self._scan_series(session_path, cancel, load_id)
                    if df is not None and not cancel.is_set():
                        try:
                            df.to_csv(cache, index=False)
                        except Exception as e:
                            self.sig.log.emit(
                                f"Could not write the cache {cache}: {e}")
            except Exception as e:
                self.sig.error.emit(
                    f"Error loading series info: {e}\n"
                    f"{traceback.format_exc()}")
                df = None
            if cancel.is_set():
                df = None
            self.sig.series_loaded.emit((load_id, session_path), df)

        threading.Thread(target=_work, daemon=True).start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _scan_series(self, session_path, cancel, load_id):
        """Read the DICOM information of all files in a session (worker
        thread). Returns a DataFrame with a row per file sorted by
        ContentTime, or None."""
        files = sorted(session_path.glob("**/*.dcm"))
        if not files:
            self.sig.log.emit("No DICOM files found in the session")
            return None
        n = len(files)
        reader = DicomReader()
        rows = []
        for i, f in enumerate(files):
            if cancel.is_set():
                return None
            try:
                info = reader.read_dicom_info(f, timeout=1)
                if info:
                    info["FilePath"] = str(f)
                    info["AcqSec"] = read_acq_sec(f)
                    rows.append(info)
            except Exception as e:
                self.sig.log.emit(f"Error reading {f}: {e}")
            if i % 20 == 0 or i == n - 1:
                self.sig.series_progress.emit(
                    i + 1, n, f"Reading DICOM headers ({i + 1}/{n})")
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df.sort_values(
            "ContentTime", inplace=True, kind="stable",
            key=lambda t: t.map(dicom_time_to_sec))
        df.reset_index(drop=True, inplace=True)
        self.sig.log.emit(
            f"Read {df.SeriesNumber.nunique()} series, {len(df)} files")
        return df

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _on_series_progress(self, cur, total, msg):
        self.series_loading_bar.setMaximum(total)
        self.series_loading_bar.setValue(cur)
        self.series_loading_lbl.setText(msg)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _on_series_loaded(self, key, df):
        load_id, session_path = key
        if load_id != self._series_load_id:
            return  # An outdated scan
        self.series_loading_lbl.hide()
        self.series_loading_bar.hide()
        self.rescan_btn.setText("Rescan")
        if df is None or len(df) == 0:
            self.dicom_series = None
            return

        for col in ("SeriesDescription", "ImageType"):
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("").astype(str)
        if "AcqSec" not in df.columns:
            df["AcqSec"] = np.nan
            self.log_message(
                "The cache has no acquisition times. TTLs are sent one TR "
                "before each volume's image. Rescan the session to simulate "
                "the acquisition timing.")
        self.dicom_series = df
        self.partners = phase_partners(df)
        phase_series = set(self.partners.values())

        for ser in sorted(df.SeriesNumber.unique()):
            sdf = df[df.SeriesNumber == ser]
            r = sdf.iloc[0]
            it = QtWidgets.QTreeWidgetItem([
                f"Series {ser}", str(len(sdf)), str(r.get("ImageType", "")),
                str(r.get("SeriesDescription", "")),
            ])
            it.setData(0, Qt.ItemDataRole.UserRole, int(ser))
            it.setTextAlignment(1, Qt.AlignmentFlag.AlignCenter)
            if ser in phase_series:
                # Copied with its magnitude series
                it.setForeground(0, QtGui.QBrush(QtGui.QColor(
                    gui_theme.COLORS["dim"])))
                it.setToolTip(0, "Phase images: copied with the magnitude "
                                 "series")
            self.series_tree.addTopLevelItem(it)
        self.log_message(
            f"Series information loaded: {df.SeriesNumber.nunique()} series,"
            f" {len(df)} files")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _series_nr_list(self):
        """Series numbers in the tree order, without phase series"""
        phase_series = set(self.partners.values())
        out = []
        for i in range(self.series_tree.topLevelItemCount()):
            nr = self.series_tree.topLevelItem(i).data(
                0, Qt.ItemDataRole.UserRole)
            if nr not in phase_series:
                out.append(nr)
        return out

    def _series_item(self, ser_nr):
        for i in range(self.series_tree.topLevelItemCount()):
            it = self.series_tree.topLevelItem(i)
            if it.data(0, Qt.ItemDataRole.UserRole) == ser_nr:
                return it
        return None

    def _n_files(self, ser_nr):
        return int((self.dicom_series.SeriesNumber == ser_nr).sum())

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def on_series_select(self):
        if self.simulation_running:
            return
        items = self.series_tree.selectedItems()
        if not items or self.dicom_series is None:
            self.current_series_lbl.setText("No series selected")
            self._set_control_state("idle")
            return
        ser_nr = items[0].data(0, Qt.ItemDataRole.UserRole)
        self._set_current_series(ser_nr)
        self._set_control_state("ready")

    def _set_current_series(self, ser_nr, image_nr=0):
        self.current_series_nr = ser_nr
        self.current_image_nr = image_nr
        it = self._series_item(ser_nr)
        n = self._n_files(ser_nr)
        desc = it.text(3) if it is not None else ""
        self.current_series_lbl.setText(
            f"Series {ser_nr}: {desc} ({n} files)")
        self.series_progress_bar.setMaximum(max(n, 1))
        self.series_progress_bar.setValue(image_nr)
        self.series_progress_lbl.setText("Series Progress:")

    # endregion

    # =========================================================================
    # region: Output directory
    # =========================================================================
    def set_output_directory(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select the output (watch) directory",
            self.out_dir_edit.text())
        if d:
            self.out_dir_edit.setText(d)
            self.log_message(f"Output directory: {d}")

    def clear_output_directory(self):
        """Delete everything in the output directory (after confirmation)"""
        out = Path(self.out_dir_edit.text())
        if not out.is_dir():
            return
        items = list(out.iterdir())
        if not items:
            self.log_message("The output directory is empty")
            return
        ans = QtWidgets.QMessageBox.question(
            self, "Clear output",
            f"Delete all {len(items)} items in\n{out} ?")
        if ans != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        for p in items:
            try:
                if p.is_dir() and not p.is_symlink():
                    shutil.rmtree(p)
                else:
                    p.unlink()
            except Exception as e:
                self.err_message(f"Could not delete {p}: {e}", popup=False)
        self.log_message(f"Cleared the output directory {out}")

    # endregion

    # =========================================================================
    # region: Simulation
    # =========================================================================
    def start_simulation(self, resume=False, one_step=False):
        """Start (or resume) the simulation from the current series"""
        if self.simulation_running:
            return
        if self.dicom_series is None or self.current_series_nr is None:
            self.err_message("Select a session and a series")
            return
        out_root = Path(self.out_dir_edit.text())
        try:
            out_root.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.err_message(f"Cannot create the output directory: {e}")
            return
        if not resume:
            self.current_image_nr = 0
            self.series_progress_bar.setValue(0)

        send_ttl = self.send_ttl_chk.isChecked()
        if send_ttl and not one_step:
            # Warn once, but run anyway (e.g., to test the DICOM side only)
            if not self.rpc_physio_com.rpc_ping(timeout=0.5):
                self.rpc_status_lbl.setText("RPC: no response")
                self.log_message(
                    "rt_physio does not respond: the TTLs are not received")

        series_list = self._series_nr_list()
        if self.current_series_nr not in series_list:
            # A phase series selected: simulate it by itself
            series_list = [self.current_series_nr]
            start_idx = 0
        else:
            start_idx = series_list.index(self.current_series_nr)

        params = {
            "series_list": series_list,
            "start_idx": start_idx,
            "start_image": self.current_image_nr,
            "one_step": one_step,
            "auto_advance": self.auto_advance_chk.isChecked(),
            "interval": float(self.series_interval_spin.value()),
            "delay": float(self.transfer_delay_spin.value()),
            "out_base": out_root / self.current_session_path.name,
            "send_ttl": send_ttl,
            # 'T' key: None, 'start', or 'every'
            "t_key": (self.t_key_mode_cmb.currentData()
                      if self.t_key_chk.isChecked() and self._xdotool
                      else None),
            "t_key_window": self.t_key_window_edit.text().strip(),
        }
        self._stop_event = threading.Event()
        self._stop_reset = False
        self.simulation_running = True
        self._set_control_state("running")
        if not one_step:
            self.log_message(
                ("Resume" if resume and self.current_image_nr > 0
                 else "Start") +
                f" simulation of series {self.current_series_nr}")
        self._sim_thread = threading.Thread(
            target=self._run_simulation, args=(params, self._stop_event),
            daemon=True)
        self._sim_thread.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def pause_simulation(self):
        if self.simulation_running:
            self._stop_event.set()
            self.log_message("Pausing simulation...")
        else:
            self.start_simulation(resume=True)

    def stop_simulation(self):
        self._stop_reset = True
        if self.simulation_running:
            self._stop_event.set()
            self.log_message("Stopping simulation...")
        else:
            # Paused: go back to the start of the series
            self._stop_reset = False
            self._set_current_series(self.current_series_nr, 0)
            self._set_control_state("ready")
            self.log_message("Simulation stopped")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _wait_until(self, t0, t_target, stop, ttl_times, n_ttl, wall0,
                    p, series_start):
        """Wait until t_target (s from t0, perf_counter), sending the TTLs
        (and 'T' keys) due in the meantime. Returns the updated TTL index,
        or None if stopped.

        series_start: the TTLs are from the start of the series (not a
        resume), so that the first one is the scan start."""
        while True:
            if stop.is_set():
                return None
            now = time.perf_counter() - t0
            while n_ttl < len(ttl_times) and ttl_times[n_ttl] <= now:
                self._emit_ttl(p, n_ttl, wall0 + ttl_times[n_ttl],
                               series_start)
                n_ttl += 1
            if now >= t_target:
                return n_ttl
            rest = t_target - now
            if n_ttl < len(ttl_times):
                rest = min(rest, ttl_times[n_ttl] - now)
            time.sleep(max(0.0005, min(rest, 0.005)))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _run_simulation(self, p, stop):
        """Simulation loop (worker thread). The widgets are updated through
        the signals."""
        ser_nr = p["series_list"][p["start_idx"]]
        image_nr = p["start_image"]
        total = 0
        try:
            idx = p["start_idx"]
            while idx < len(p["series_list"]):
                ser_nr = p["series_list"][idx]
                sdf = self.dicom_series[
                    self.dicom_series.SeriesNumber == ser_nr].copy()
                sdf["ContentSec"] = content_sec(sdf["ContentTime"])
                sdf.sort_values("ContentSec", inplace=True, kind="stable")
                sdf.reset_index(drop=True, inplace=True)
                total = len(sdf)
                self.sig.sim_series.emit(int(ser_nr), total)

                todo = sdf.iloc[image_nr:]
                if len(todo) == 0:
                    break

                # Phase images copied with the magnitude images
                pha = None
                if ser_nr in self.partners:
                    pha = self.dicom_series[
                        self.dicom_series.SeriesNumber ==
                        self.partners[ser_nr]]
                    pha = dict(zip(pha.InstanceNumber, pha.FilePath))

                file_t, ttl_t = series_timing(todo, p["delay"])
                if p["one_step"]:
                    # Copy the next file now, with the TTLs due before it
                    ttl_t = ttl_t[ttl_t <= file_t[0]] - file_t[0]
                    file_t = file_t - file_t[0]

                rel = Path(todo.FilePath.values[0]).relative_to(
                    self.current_session_path)
                out_dir = p["out_base"] / rel.parent
                out_dir.mkdir(parents=True, exist_ok=True)

                t0 = time.perf_counter()
                wall0 = time.time()
                n_ttl = 0
                series_start = image_nr == 0
                for ii, row in enumerate(todo.itertuples()):
                    n_ttl = self._wait_until(
                        t0, file_t[ii], stop, ttl_t, n_ttl, wall0, p,
                        series_start)
                    if n_ttl is None:
                        break
                    src = Path(row.FilePath)
                    shutil.copy(src, out_dir / src.name)
                    if pha is not None and row.InstanceNumber in pha:
                        psrc = Path(pha[row.InstanceNumber])
                        shutil.copy(psrc, out_dir / psrc.name)
                    image_nr += 1
                    self.sig.sim_file.emit(image_nr, total, src.name)
                    if p["one_step"]:
                        break

                if stop.is_set() or p["one_step"] or image_nr < total:
                    break

                # TTLs after the last file (none in a normal series)
                for i in range(n_ttl, len(ttl_t)):
                    self._emit_ttl(p, i, wall0 + ttl_t[i], series_start)

                self.sig.log.emit(f"Completed series {ser_nr}")
                if not p["auto_advance"] or idx + 1 >= len(p["series_list"]):
                    break

                # Wait for the series interval
                self.sig.sim_wait.emit(p["interval"])
                if stop.wait(p["interval"]):
                    break
                idx += 1
                image_nr = 0

        except Exception as e:
            self.sig.error.emit(
                f"Simulation error: {e}\n{traceback.format_exc()}")
        finally:
            self.sig.sim_finished.emit(int(ser_nr), image_nr, total)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _on_sim_series(self, ser_nr, total):
        it = self._series_item(ser_nr)
        if it is not None:
            self.series_tree.blockSignals(True)
            self.series_tree.setCurrentItem(it)
            self.series_tree.scrollToItem(it)
            self.series_tree.blockSignals(False)
        if ser_nr != self.current_series_nr:
            self._set_current_series(ser_nr, 0)
            self.log_message(f"Start series {ser_nr}")

    def _on_sim_file(self, image_nr, total, name):
        self.current_image_nr = image_nr
        self.series_progress_bar.setMaximum(total)
        self.series_progress_bar.setValue(image_nr)
        self.log_message(f"Copy file {image_nr}/{total}: {name}")

    def _on_sim_wait(self, sec):
        self.log_message(f"Waiting {sec:g} s for the next series")

    def _on_sim_finished(self, ser_nr, image_nr, total):
        self.simulation_running = False
        self.current_series_nr = ser_nr
        self.current_image_nr = image_nr
        if self._stop_reset:
            self._stop_reset = False
            self._set_current_series(ser_nr, 0)
            self._set_control_state("ready")
            self.log_message("Simulation stopped")
        elif 0 < image_nr < total:
            self._set_control_state("paused")
            if self._stop_event.is_set():
                self.log_message(
                    f"Simulation paused at file {image_nr}/{total}")
        else:
            self._set_control_state("ready")
            if image_nr >= total and total > 0:
                # Select the next series
                lst = self._series_nr_list()
                if ser_nr in lst and lst.index(ser_nr) + 1 < len(lst):
                    nxt = lst[lst.index(ser_nr) + 1]
                    it = self._series_item(nxt)
                    self.series_tree.setCurrentItem(it)
                    self.series_tree.scrollToItem(it)
                else:
                    self.log_message("The last series is done")
                    self._set_current_series(ser_nr, 0)

    # endregion

    # =========================================================================
    # region: TTL and physio (RPC to rt_physio)
    # =========================================================================
    def _ttl_sender(self):
        """Send the queued TTLs to rt_physio (worker thread)"""
        while True:
            tstamp = self._ttl_queue.get()
            if tstamp is None:
                return
            try:
                self.rpc_physio_com.call_rt_proc(
                    ("TTL_PULSE", float(tstamp)), pkl=True, timeout=0.5)
            except Exception as e:
                self.sig.log.emit(f"Error sending a TTL: {e}")
            self._ttl_count += 1
            self.sig.ttl_sent.emit(self._ttl_count)

    def send_manual_pulse(self):
        self._ttl_queue.put(time.time())
        self.log_message("Send TTL")

    def _show_counts(self, *args):
        self.ttl_count_lbl.setText(
            f"TTL sent: {self._ttl_count}, 'T' key: {self._key_count}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _emit_ttl(self, p, ttl_idx, tstamp, series_start):
        """Send a TTL of the simulation (worker thread): to rt_physio, and
        as a 'T' key press by the mode."""
        if p["send_ttl"]:
            # The planned time, so that the RPC delay does not shift the
            # TTL in the recording
            self._ttl_queue.put(tstamp)
        mode = p.get("t_key")
        if mode == "every" or (
            mode == "start" and series_start and ttl_idx == 0
        ):
            self._key_queue.put(p.get("t_key_window", ""))

    def press_manual_key(self):
        self._key_queue.put(self.t_key_window_edit.text().strip())
        self.log_message("Press 'T'")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _key_sender(self):
        """Press the queued 'T' keys with xdotool (worker thread). The item
        is the title pattern of the window to activate first ("" for the
        focused window)."""
        win_cache = {}
        while True:
            pattern = self._key_queue.get()
            if pattern is None:
                return
            try:
                if pattern:
                    win = win_cache.get(pattern)
                    if win is None or not self._window_exists(win):
                        win = self._find_window(pattern)
                        win_cache[pattern] = win
                    if win is None:
                        self.sig.log.emit(
                            f"'T' key not sent: no window titled "
                            f"'{pattern}'")
                        continue
                    # Activate the window (raise and focus). Without a
                    # window manager that supports it, only focus it.
                    ret = subprocess.run(
                        [self._xdotool, "windowactivate", "--sync", win],
                        capture_output=True, text=True, timeout=5)
                    if ret.returncode != 0:
                        subprocess.run(
                            [self._xdotool, "windowfocus", "--sync", win],
                            capture_output=True, text=True, timeout=5)
                ret = subprocess.run(
                    [self._xdotool, "key", "--clearmodifiers", "t"],
                    capture_output=True, text=True, timeout=5)
                if ret.returncode != 0:
                    self.sig.log.emit(
                        f"xdotool failed: {ret.stderr.strip()}")
                    continue
            except Exception as e:
                self.sig.log.emit(f"Error pressing the 'T' key: {e}")
                continue
            self._key_count += 1
            self.sig.key_sent.emit(self._key_count)

    def _find_window(self, pattern):
        """ID of the first visible window whose title matches pattern."""
        ret = subprocess.run(
            [self._xdotool, "search", "--onlyvisible", "--name", pattern],
            capture_output=True, text=True, timeout=5)
        ids = ret.stdout.split()
        return ids[0] if ids else None

    def _window_exists(self, win):
        ret = subprocess.run(
            [self._xdotool, "getwindowname", win],
            capture_output=True, text=True, timeout=5)
        return ret.returncode == 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ping_physio_rpc(self, popup=True):
        ok = self.rpc_physio_com.rpc_ping()
        if ok:
            port = self.rpc_physio_com.address_name[1]
            self.rpc_status_lbl.setText(f"RPC: connected (port {port})")
            self.log_message("rt_physio responds")
            if popup:
                QtWidgets.QMessageBox.information(
                    self, "rt_physio RPC", "rt_physio responds.")
        else:
            self.rpc_status_lbl.setText("RPC: no response")
            self.err_message(
                "rt_physio does not respond (is RTPSpy or RT-MRI with "
                "rt_physio running?)", popup=popup)
        return ok

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _select_file(self, edit, kind):
        start = edit.text() or str(self.current_session_path or APP_DIR)
        if kind == "bids":
            filt = "BIDS physio (*_physio.tsv *_physio.tsv.gz);;All (*)"
        else:
            filt = "1D files (*.1D *.txt);;All (*)"
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Select the {kind} file", start, filt)
        if f:
            edit.setText(f)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def start_physio_feeding(self):
        """Start the dummy recorder of rt_physio with the selected signals"""
        if self.physio_1d_radio.isChecked():
            card = self.card_file_edit.text()
            resp = self.resp_file_edit.text()
            if not card or not resp:
                self.err_message("Select the cardiac and respiration files")
                return
            for f in (card, resp):
                if not Path(f).is_file():
                    self.err_message(f"{f} is not found")
                    return
            command = (
                "START_DUMMY_FEEDING_WITH_FILES",
                (str(Path(card).resolve()), str(Path(resp).resolve()),
                 self.physio_freq_spin.value()),
            )
            what = f"{Path(card).name}, {Path(resp).name}"
        else:
            bids = self.bids_file_edit.text()
            if not bids or not Path(bids).is_file():
                self.err_message("Select a BIDS *_physio.tsv[.gz] file")
                return
            command = ("START_DUMMY_FEEDING_WITH_BIDS",
                       str(Path(bids).resolve()))
            what = Path(bids).name

        if not self.ping_physio_rpc(popup=False):
            QtWidgets.QMessageBox.critical(
                self, "Error", "rt_physio does not respond.")
            return
        try:
            ret = self.rpc_physio_com.call_rt_proc(
                command, pkl=True, get_return=True, quiet=True)
        except Exception as e:
            self.err_message(
                f"Error starting the physio feeding: {e}\n"
                f"{traceback.format_exc()}")
            return
        if isinstance(ret, bytes):
            ret = ret.decode("utf-8", errors="replace")
        if isinstance(ret, str) and ret.startswith("Error"):
            self.err_message(f"rt_physio: {ret}")
            return
        self.log_message(f"Physio feeding started: {what}")

    def reset_physio(self):
        try:
            # "NULL": no device. (None selects a device automatically,
            # which would choose the Dummy device again.)
            self.rpc_physio_com.call_rt_proc(
                ("SET_REC_DEV", "NULL"), pkl=True)
            self.log_message("Physio feeding stopped")
        except Exception as e:
            self.err_message(f"Error stopping the physio feeding: {e}")

    # endregion

    # =========================================================================
    # region: Config
    # =========================================================================
    def load_config(self):
        try:
            if self.CONFIG_FILE.is_file():
                with open(self.CONFIG_FILE) as fid:
                    saved = json.load(fid)
                self.config.update(
                    {k: v for k, v in saved.items() if k in self.config})
        except Exception as e:
            self.log_message(f"Error reading {self.CONFIG_FILE}: {e}")

    def _apply_config_to_widgets(self):
        c = self.config
        self.src_dir_edit.setText(str(c["src_data_dir"]))
        self.out_dir_edit.setText(str(c["simulation_output_dir"]))
        self.auto_advance_chk.setChecked(bool(c["auto_advance"]))
        self.series_interval_spin.setValue(float(c["series_interval"]))
        self.transfer_delay_spin.setValue(float(c["transfer_delay"]))
        self.send_ttl_chk.setChecked(bool(c["send_ttl"]))
        self.t_key_chk.setChecked(bool(c["t_key"]) and bool(self._xdotool))
        idx = self.t_key_mode_cmb.findData(c["t_key_mode"])
        self.t_key_mode_cmb.setCurrentIndex(max(idx, 0))
        self.t_key_window_edit.setText(str(c["t_key_window"]))
        self.card_file_edit.setText(c["card_file"])
        self.resp_file_edit.setText(c["resp_file"])
        self.physio_freq_spin.setValue(int(c["physio_freq"]))
        self.bids_file_edit.setText(c["bids_physio_file"])
        if c["physio_source"] == "BIDS":
            self.physio_bids_radio.setChecked(True)
        else:
            self.physio_1d_radio.setChecked(True)
        self._update_physio_widgets()

    def save_config(self):
        c = self.config
        c["src_data_dir"] = self.src_dir_edit.text()
        c["simulation_output_dir"] = self.out_dir_edit.text()
        c["auto_advance"] = self.auto_advance_chk.isChecked()
        c["series_interval"] = self.series_interval_spin.value()
        c["transfer_delay"] = self.transfer_delay_spin.value()
        c["send_ttl"] = self.send_ttl_chk.isChecked()
        c["t_key"] = self.t_key_chk.isChecked()
        c["t_key_mode"] = self.t_key_mode_cmb.currentData()
        c["t_key_window"] = self.t_key_window_edit.text()
        c["card_file"] = self.card_file_edit.text()
        c["resp_file"] = self.resp_file_edit.text()
        c["physio_freq"] = self.physio_freq_spin.value()
        c["bids_physio_file"] = self.bids_file_edit.text()
        c["physio_source"] = (
            "BIDS" if self.physio_bids_radio.isChecked() else "1D")
        c["current_session_path"] = (
            str(self.current_session_path)
            if self.current_session_path else "")
        try:
            self.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self.CONFIG_FILE, "w") as fid:
                json.dump(c, fid, indent=2)
        except Exception as e:
            self._logger.error(f"Error saving {self.CONFIG_FILE}: {e}")

    # endregion

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def closeEvent(self, event):
        if self.simulation_running:
            ans = QtWidgets.QMessageBox.question(
                self, "Quit", "The simulation is running. Stop and quit?")
            if ans != QtWidgets.QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._stop_event.set()
            if self._sim_thread is not None:
                self._sim_thread.join(timeout=2)
        self._scan_cancel.set()
        self._ttl_queue.put(None)
        self._key_queue.put(None)
        self.save_config()
        event.accept()


# %% main =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="RT-MRI simulator: replay DICOM, TTL, and physio")
    parser.add_argument(
        "--physio_host", default="localhost",
        help="Host of the rt_physio process (default: localhost)")
    parser.add_argument(
        "--rpc_config", default=str(
            Path.home() / ".RTPSpy" / "rtmri_config.json"),
        help="Config file with the RPC port and token of rt_physio")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    app = QtWidgets.QApplication(sys.argv)
    gui_theme.apply_qt(app)
    win = RTMRISimulatorQt(
        rpc_physio_address_name=[
            args.physio_host, None, "RtTTLPhysioSocketServer"],
        rpc_config_path=Path(args.rpc_config))
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
