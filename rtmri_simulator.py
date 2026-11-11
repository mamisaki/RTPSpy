#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RT-MRI Simulator

A GUI tool to simulate real-time MRI data export for testing RTPSpy:
- Select a DICOM session directory and its series.
- Copy the DICOM files of a series to the output directory at the original
  acquisition timing, sending a TTL pulse at each volume.
- Select cardiogram and respiration files to feed the dummy recorder of the
  rt_physio process via RPC.

Author: mmisaki@laureateinstitute.org
"""

# %% import libraries =========================================================
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import time
import shutil
import logging
from pathlib import Path
from datetime import datetime
import json
import traceback

import numpy as np
import pandas as pd
import pydicom

try:
    from dicom_reader import DicomReader
    from rpc_socket_server import RPCSocketCom
except ImportError:
    from rtpspy.dicom_reader import DicomReader
    from rtpspy.rpc_socket_server import RPCSocketCom


# %% dicom_time_to_sec ========================================================
def dicom_time_to_sec(value):
    """Convert a DICOM time to seconds of the day.

    value: TM (HHMMSS.ffffff) or DT (YYYYMMDDHHMMSS.ffffff) string, or the
    number pandas reads from the cache CSV (e.g., 93359.826 for TM
    "093359.826"). A TM must not be used as a number directly: 093400.05 -
    093359.826 is 40.22, not 0.22 s.
    (The same function as in RT-MRI simulation/rtmri_simulator.py.)
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
    file has no acquisition time).
    (The same function as in RT-MRI simulation/rtmri_simulator.py.)"""
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
    TTL, as on the scanner (from RT-MRI simulation/rtmri_simulator.py, with
    the transfer delay added):

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


# %% RTMRISimulator ===========================================================
class RTMRISimulator:
    """GUI-based RT-MRI simulator with series-by-series control"""

    # GUI settings saved at closing and restored at start
    CONFIG_FILE = Path.home() / ".RTPSpy" / "rtmri_simulator_config.json"

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(
        self,
        rpc_physio_address_name=["localhost", None, "RtTTLPhysioSocketServer"],
        config_path=Path.home() / ".RTPSpy" / "rtmri_config.json"
    ):
        self.root = tk.Tk()
        self.root.title("RT-MRI Simulator")
        self.root.geometry("900x850")

        # Default fonts
        self.root.option_add("*Font", ("Arial", 10))
        self.root.option_add("*Label.Font", ("Arial", 10))
        self.root.option_add("*Entry.Font", ("Arial", 10))
        self.root.option_add("*Listbox.Font", ("Arial", 10))
        self.root.option_add("*Text.Font", ("Arial", 10))
        self.root.option_add("*Checkbutton.Font", ("Arial", 10))

        # Configure ttk styles for themed widgets
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 9, "bold"))
        style.configure("TCheckbutton", font=("Arial", 10))
        style.configure("TLabelframe.Label", font=("Arial", 10))

        # Initialize variables
        self.current_session_path = None
        self.dicom_series = None
        self.current_series_nr = 0

        self.selected_card_file = None
        self.selected_resp_file = None
        self.rpc_physio_com = RPCSocketCom(
            rpc_physio_address_name, config_path
        )

        # Initialize status variables
        self.current_image_nr = 0
        self.simulation_running = False
        self.simulation_thread = None
        self.scanning_cancelled = False
        self._load_series_lock = threading.Lock()

        # Initialize logger
        self.logger = logging.getLogger("RTMRISimulator")

        # Default directories
        src_data_dir = Path(__file__).parent / "simulation_data"
        if not src_data_dir.exists():
            src_data_dir.mkdir(parents=True)
        simulation_output_dir = Path(__file__).parent / "RTMRI_simulation"
        if not simulation_output_dir.exists():
            simulation_output_dir.mkdir(parents=True)

        self.config = {
            "src_data_dir": src_data_dir,
            "simulation_output_dir": simulation_output_dir,
            "auto_advance": False,
            "series_interval": 10,
            # Time (s) to transfer an image file from the scanner to the
            # export directory (measured on the Siemens RT PC: 0.1-0.25 s)
            "transfer_delay": 0.2,
        }

        self.setup_gui()
        self.load_dicom_sessions()

        # Restore the saved settings
        self.load_config()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def setup_gui(self):
        """Create the main GUI"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Session/series panel (left) and simulation control panel (right)
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))

        # Physio and log panels at the bottom with full width
        physio_panel = ttk.Frame(main_frame)
        physio_panel.pack(fill=tk.BOTH, expand=False, pady=(5, 0))

        log_panel = ttk.Frame(main_frame)
        log_panel.pack(fill=tk.BOTH, expand=False, pady=(5, 0))

        # Setup panels
        self.setup_session_panel(left_panel)
        self.setup_simulation_panel(right_panel)
        self.setup_ttl_physio_panel(physio_panel)
        self.setup_log_panel(log_panel)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def setup_session_panel(self, parent):
        """Set up the DICOM session and series selection panels"""
        # region: DICOM Sessions ----------------------------------------------
        session_frame = ttk.LabelFrame(parent, text="DICOM Sessions")
        session_frame.pack(fill=tk.BOTH, pady=(0, 5))

        # Source data path
        src_path_frame = ttk.Frame(session_frame)
        src_path_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(src_path_frame, text="Source Root:").pack(side=tk.LEFT)

        # Current source data path (read-only, scrollable)
        self.src_dir_var = tk.StringVar(value=str(self.config["src_data_dir"]))
        self._readonly_entry(src_path_frame, self.src_dir_var, width=40).pack(
            side=tk.LEFT, padx=2)

        ttk.Button(
            src_path_frame,
            text="Refresh",
            command=self.load_dicom_sessions,
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            src_path_frame,
            text="Set",
            command=self.set_src_dir,
        ).pack(side=tk.LEFT, padx=2)

        # Session list
        session_list_frame = ttk.Frame(session_frame)
        session_list_frame.pack(fill=tk.BOTH, padx=5, pady=5)

        # Treeview for sessions
        columns = ("Name", "Files", "Study Description")
        self.session_tree = ttk.Treeview(
            session_list_frame, columns=columns, show="headings", height=4
        )

        for col in columns:
            self.session_tree.heading(col, text=col)
            if col == "Name":
                self.session_tree.column(col, width=40)
            elif col == "Files":
                self.session_tree.column(col, width=50, anchor=tk.CENTER)

        session_scrollbar = ttk.Scrollbar(
            session_list_frame,
            orient=tk.VERTICAL,
            command=self.session_tree.yview,
        )
        self.session_tree.configure(yscrollcommand=session_scrollbar.set)

        self.session_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        session_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind selection event
        self.session_tree.bind("<<TreeviewSelect>>", self.on_session_select)

        # endregion: DICOM Sessions -------------------------------------------

        # region: Series information ------------------------------------------
        series_frame = ttk.LabelFrame(parent, text="Series Information")
        series_frame.pack(fill=tk.BOTH, pady=5)

        # Progress bar for series loading, and the Rescan button
        self.series_progress_frame = ttk.Frame(series_frame)
        self.series_progress_frame.pack(fill=tk.X, padx=5, pady=2)

        # Progress elements (left side)
        progress_elements_frame = ttk.Frame(self.series_progress_frame)
        progress_elements_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.series_loading_label = ttk.Label(
            progress_elements_frame, text="Loading series..."
        )
        self.series_loading_progress = ttk.Progressbar(
            progress_elements_frame, mode="determinate"
        )

        # Rescan button (right side)
        self.rescan_btn = ttk.Button(
            self.series_progress_frame,
            text="Rescan",
            command=self.handle_rescan_click,
            width=8,
        )
        self.rescan_btn.pack(side=tk.RIGHT, padx=(5, 0))

        # The progress elements are shown only while loading
        self.series_loading_label.pack_forget()
        self.series_loading_progress.pack_forget()

        # Current session display
        session_info_frame = ttk.Frame(series_frame)
        session_info_frame.pack(fill=tk.X, padx=5, pady=(2, 5))

        ttk.Label(session_info_frame, text="Current Session:").pack(
            side=tk.LEFT
        )
        self.current_session_var = tk.StringVar(value="No session selected")
        current_session_label = ttk.Label(
            session_info_frame,
            textvariable=self.current_session_var,
            font=("Arial", 10, "italic"),
        )
        current_session_label.pack(side=tk.LEFT, padx=(5, 0))

        # Series treeview
        series_columns = ("Series", "Files", "Image Type", "Description")
        self.series_tree = ttk.Treeview(
            series_frame, columns=series_columns, show="headings", height=10
        )

        for col in series_columns:
            self.series_tree.heading(col, text=col)
            if col == "Series":
                self.series_tree.column(col, width=10)
            elif col == "Files":
                self.series_tree.column(col, width=4, anchor=tk.CENTER)
            elif col == "Image Type":
                self.series_tree.column(col, width=160)

        series_scrollbar = ttk.Scrollbar(
            series_frame, orient=tk.VERTICAL, command=self.series_tree.yview
        )
        self.series_tree.configure(yscrollcommand=series_scrollbar.set)

        self.series_tree.pack(
            side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5
        )
        series_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Update the simulation panel at series selection
        self.series_tree.bind("<<TreeviewSelect>>", self.on_series_select)
        # endregion: Series information ---------------------------------------

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def setup_simulation_panel(self, parent):
        """Set up the simulation control panel"""
        sim_frame = ttk.LabelFrame(parent, text="Simulation Control")
        sim_frame.pack(fill=tk.BOTH, pady=(0, 5))

        # region: Output path -------------------------------------------------
        self.outpath_frame = ttk.Frame(sim_frame)
        self.outpath_frame.pack(fill=tk.X, padx=5, pady=5)

        # Top row: label, path, and Set button
        top_row = ttk.Frame(self.outpath_frame)
        top_row.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(top_row, text="Output:").pack(side=tk.LEFT, padx=2)

        self.output_dir_var = tk.StringVar(
            value=str(self.config["simulation_output_dir"])
        )
        self._readonly_entry(top_row, self.output_dir_var).pack(
            side=tk.LEFT, padx=2)

        ttk.Button(
            top_row,
            text="Set",
            command=self.set_output_directory,
        ).pack(side=tk.LEFT, padx=2)

        # Clear
        ttk.Button(
            self.outpath_frame,
            text="Clear",
            command=self.clear_output_directory,
        ).pack(side=tk.BOTTOM, padx=2)

        # endregion: Output path ----------------------------------------------

        # region: Simulation control ------------------------------------------
        # Parameters
        params_frame = ttk.Frame(sim_frame)
        params_frame.pack(fill=tk.X, padx=5, pady=5)

        # Auto-advance setting
        aa_frame = ttk.Frame(params_frame)
        aa_frame.pack(fill=tk.X, padx=5, pady=5)

        self.auto_advance_var = tk.BooleanVar(
            value=self.config["auto_advance"]
        )
        ttk.Checkbutton(
            aa_frame, text="Auto-advance", variable=self.auto_advance_var
        ).pack(side=tk.LEFT, padx=5)

        # Interval (s) between series in auto-advance
        self.series_interval_var = tk.IntVar(
            value=self.config["series_interval"])
        ttk.Label(aa_frame, text="Series Interval:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(
            aa_frame,
            textvariable=self.series_interval_var,
            width=5
        ).pack(side=tk.LEFT, padx=5)

        # Transfer delay (s) of the image files (see series_timing)
        self.transfer_delay_var = tk.DoubleVar(
            value=self.config["transfer_delay"])
        ttk.Label(aa_frame, text="Transfer delay (s):").pack(
            side=tk.LEFT, padx=5)
        ttk.Entry(
            aa_frame,
            textvariable=self.transfer_delay_var,
            width=5
        ).pack(side=tk.LEFT, padx=5)

        # Current series display
        current_frame = ttk.LabelFrame(sim_frame, text="Current Status")
        current_frame.pack(fill=tk.X, padx=5, pady=5)

        self.current_series_var = tk.StringVar(value="No series selected")
        status_label = ttk.Label(
            current_frame,
            textvariable=self.current_series_var,
            font=("Arial", 10),
            wraplength=250,
        )
        status_label.pack(padx=5, pady=5)

        # Progress bar
        self.series_progress_label = ttk.Label(
            current_frame, text="Series Progress:"
        )
        self.series_progress_label.pack(anchor=tk.W, padx=5)
        self.series_progress = ttk.Progressbar(
            current_frame, mode="determinate"
        )
        self.series_progress.pack(fill=tk.X, padx=5, pady=2)

        # Control buttons
        control_frame = ttk.LabelFrame(sim_frame, text="Controls")
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Main control buttons
        main_btn_frame = ttk.Frame(control_frame)
        main_btn_frame.pack(fill=tk.X, padx=5, pady=5)

        self.start_btn = ttk.Button(
            main_btn_frame,
            text="Start Simulation",
            command=self.start_simulation,
            state=tk.DISABLED,  # Initially disabled
        )
        self.start_btn.pack(fill=tk.X, pady=2)

        self.pause_btn = ttk.Button(
            main_btn_frame,
            text="Pause",
            command=self.pause_simulation,
            state=tk.DISABLED,
        )
        self.pause_btn.pack(fill=tk.X, pady=2)

        self.step_btn = ttk.Button(
            main_btn_frame,
            text="Step Forward",
            command=self.step_simulation,
            state=tk.DISABLED,  # Initially disabled
        )
        self.step_btn.pack(fill=tk.X, pady=2)
        # endregion: Simulation control ---------------------------------------

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def setup_ttl_physio_panel(self, parent):
        """Set up the physiological data panel"""
        physio_frame = ttk.LabelFrame(parent, text="Physiological Data")
        physio_frame.pack(fill=tk.X, pady=5)

        # region: File Selection ----------------------------------------------
        file_col = ttk.Frame(physio_frame)
        file_col.pack(side=tk.LEFT, padx=5, pady=5)

        # Cardiogram row
        cardiogram_row = ttk.Frame(file_col)
        cardiogram_row.pack(fill=tk.X, padx=5, pady=5)
        tk.Label(cardiogram_row, text="Cardiogram File:").pack(side=tk.LEFT)

        self.card_file_var = tk.StringVar()
        self._readonly_entry(cardiogram_row, self.card_file_var,
                             width=40).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            cardiogram_row,
            text="Set",
            command=self.set_cardiogram_file,
        ).pack(side=tk.LEFT, padx=2)

        # Respiration row
        respiration_row = ttk.Frame(file_col)
        respiration_row.pack(fill=tk.X, padx=5, pady=5)
        tk.Label(respiration_row, text="Respiration File:").pack(side=tk.LEFT)

        self.resp_file_var = tk.StringVar()
        self._readonly_entry(respiration_row, self.resp_file_var,
                             width=40).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            respiration_row,
            text="Set",
            command=self.set_respiration_file,
        ).pack(side=tk.LEFT, padx=2)

        # An empty frame kept as a spacer (10 px) in the layout
        ttk.Frame(physio_frame).pack(side=tk.LEFT, padx=5, pady=5)
        # endregion: File Selection -------------------------------------------

        # region: Frequency setting column ------------------------------------
        frequency_col = ttk.Frame(physio_frame)
        frequency_col.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Label(frequency_col, text="Freq. (Hz)").pack(anchor=tk.W, pady=2)

        self.physio_freq_var = tk.StringVar(value="100")
        freq_spinbox = ttk.Spinbox(
            frequency_col,
            from_=1,
            to=1000,
            increment=1,
            width=4,
            textvariable=self.physio_freq_var,
        )
        freq_spinbox.pack(anchor=tk.W, pady=2)
        # endregion: Frequency setting ----------------------------------------

        # region: Command buttons ---------------------------------------------
        button_col = ttk.Frame(physio_frame)
        button_col.pack(side=tk.LEFT, padx=5, pady=5)
        button_col.grid_columnconfigure(0, weight=1)
        button_col.grid_columnconfigure(1, weight=1)

        # Start physio feeding
        self.start_physio_btn = ttk.Button(
            button_col,
            text="Set Physio",
            command=self.start_physio_feeding,
        )
        self.start_physio_btn.grid(
            row=0, column=0, padx=5, pady=5, sticky="ew")

        # Stop physio feeding
        self.reset_physio_btn = ttk.Button(
            button_col,
            text="Reset Physio",
            command=self.reset_physio,
        )
        self.reset_physio_btn.grid(
            row=0, column=1, padx=5, pady=5, sticky="ew")

        # Check the RPC connection
        ttk.Button(
            button_col,
            text="Check RPC",
            command=self.ping_physio_rpc,
        ).grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        # Send a TTL pulse
        self.send_pulse_btn = ttk.Button(
            button_col,
            text="Send Pulse",
            command=self.send_pulse,
        )
        self.send_pulse_btn.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        # endregion: Command buttons ------------------------------------------

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def setup_log_panel(self, parent):
        """Set up the log panel at the bottom"""
        log_frame = ttk.LabelFrame(parent, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=10, wrap=tk.WORD
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _readonly_entry(self, parent, textvariable, width=None):
        """Make a read-only entry whose text can be selected and copied"""
        kwargs = {} if width is None else {"width": width}
        entry = tk.Entry(
            parent,
            textvariable=textvariable,
            font=("Arial", 10),
            state="readonly",
            readonlybackground="white",
            **kwargs,
        )
        entry.configure(exportselection=True)
        return entry

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _find_sessions_recursive(
        self, current_dir, base_dir, sessions, max_depth=3, current_depth=0
    ):
        """Recursively find session directories containing DICOM files"""
        if current_depth > max_depth:
            return

        if not current_dir.exists() or not current_dir.is_dir():
            return

        # A session directory has DICOM files directly or in a "dicom"
        # subdirectory.
        has_direct_dicoms = any(current_dir.glob("*.dcm"))
        has_dicom_subdir = False
        dicom_subdir = current_dir / "dicom"
        if dicom_subdir.exists() and dicom_subdir.is_dir():
            has_dicom_subdir = any(dicom_subdir.glob("**/*.dcm"))

        if has_direct_dicoms or has_dicom_subdir:
            all_dicom_files = list(current_dir.glob("**/*.dcm"))

            if current_dir == base_dir:
                session_name = current_dir.name
            else:
                session_name = str(current_dir.relative_to(base_dir))

            study_description = self._get_study_description(
                all_dicom_files[:3])

            sessions.append(
                {
                    "name": session_name,
                    "path": str(current_dir),
                    "files": len(all_dicom_files),
                    "study_description": study_description,
                }
            )
            return  # Don't go deeper if we found DICOM files here

        # Check subdirectories for series-like structure (S001, Ser001, etc.)
        series_dirs = [
            d
            for d in current_dir.iterdir()
            if d.is_dir()
            and (
                (d.name.startswith("S") and d.name[1:].isdigit())
                or (d.name.startswith("Ser") and d.name[3:].isdigit())
            )
        ]

        if series_dirs:
            # Check if this directory has DICOM files anywhere
            all_dicom_files = list(current_dir.glob("**/*.dcm"))
            if all_dicom_files:
                # A directory with both series directories and other
                # subdirectories is likely a container of sessions; explore
                # deeper instead of treating it as a session.
                all_subdirs = [d for d in current_dir.iterdir() if d.is_dir()]
                non_series_dirs = [
                    d for d in all_subdirs if d not in series_dirs
                ]
                is_container = (
                    len(non_series_dirs) > 0 and current_depth < max_depth
                )
                if not is_container:
                    # A session directory with series subdirectories
                    if current_dir == base_dir:
                        session_name = current_dir.name
                    else:
                        session_name = str(current_dir.relative_to(base_dir))

                    study_desc = self._get_study_description(
                        all_dicom_files[:3]
                    )

                    sessions.append(
                        {
                            "name": session_name,
                            "path": str(current_dir),
                            "files": len(all_dicom_files),
                            "study_description": study_desc,
                        }
                    )
                    return  # Don't go deeper

        # Recursively check subdirectories for sessions
        for subdir in current_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("."):
                self._find_sessions_recursive(
                    subdir, base_dir, sessions, max_depth, current_depth + 1
                )

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _get_study_description(self, dicom_files):
        """Get study description from DICOM files"""
        study_description = "Unknown"
        try:
            reader = DicomReader()
            for dcm_file in dicom_files:
                try:
                    info = reader.read_dicom_info(dcm_file, timeout=0.1)
                    if info and "StudyDescription" in info:
                        study_description = info["StudyDescription"]
                        break
                except Exception:
                    continue
        except Exception:
            pass
        return study_description

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def on_session_select(self, event=None, session=None):
        """Handle session selection"""
        if self.simulation_running:
            return

        if session:
            # Select the tree item of the session path without triggering
            # the selection event.
            self.session_tree.unbind("<<TreeviewSelect>>")
            src_data_dir = Path(self.config["src_data_dir"])
            for item_id in self.session_tree.get_children():
                session_name = self.session_tree.item(item_id)["values"][0]
                if Path(session) == src_data_dir / session_name:
                    self.session_tree.selection_set(item_id)
                    self.session_tree.see(item_id)
                    break
            self.session_tree.bind("<<TreeviewSelect>>",
                                   self.on_session_select)
            return

        selection = self.session_tree.selection()
        if not selection:
            self.current_session_var.set("No session selected")
            self.start_btn.config(state=tk.DISABLED)
            self.step_btn.config(state=tk.DISABLED)
            return

        item = self.session_tree.item(selection[0])
        session_name = item["values"][0]
        study_description = item["values"][2]

        # Reconstruct the session path from the session name
        src_data_dir = Path(self.config["src_data_dir"])
        if "/" in session_name:
            # The name is a path relative to src_data_dir
            # (e.g., "subject/session")
            session_path = src_data_dir / session_name
        else:
            # Look in src_data_dir and its subdirectories
            possible_paths = [
                src_data_dir / session_name,
            ]

            for subdir in src_data_dir.glob("*"):
                if subdir.is_dir():
                    possible_paths.append(subdir / session_name)
                    # Check study directories within subdirectories
                    for study_dir in subdir.glob("*"):
                        if (
                            study_dir.is_dir()
                            and study_dir.name == session_name
                        ):
                            possible_paths.append(study_dir)

            # Take the first one that has DICOM files
            session_path = None
            for path in possible_paths:
                if path.exists() and any(path.glob("**/*.dcm")):
                    session_path = path
                    break

            if not session_path:
                self.err_message(
                    f"Could not find session directory for: {session_name}"
                )
                return

        self.current_session_path = session_path

        if study_description and study_description != "Unknown":
            session_display = f"{session_name} ({study_description})"
        else:
            session_display = session_name
        self.current_session_var.set(session_display)

        self.log_message(f"Selected session: {session_path}")

        # Load the series information in a worker thread
        loading_thread = threading.Thread(
            target=self._load_series_info_threaded,
            args=(session_path,),
            daemon=True,
        )
        loading_thread.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def on_series_select(self, event):
        """Handle series selection and update simulation controls"""
        if self.simulation_running:
            return

        selection = self.series_tree.selection()
        if not selection:
            self.current_series_var.set("No series selected")
            self.start_btn.config(state=tk.DISABLED)
            self.step_btn.config(state=tk.DISABLED)
            return

        item = self.series_tree.item(selection[0])
        series_name = item["values"][0]  # Series column
        series_files = item["values"][1]  # Files column
        series_description = item["values"][-1]  # Description column

        series_status = (
            f"{series_name}: {series_description} ({series_files} files)"
        )
        self.current_series_var.set(series_status)

        self.current_series_nr = int(series_name.split()[-1])
        self.current_image_nr = 0
        self._reset_series_progress()

        self.start_btn.config(state=tk.NORMAL)
        self.step_btn.config(state=tk.NORMAL)
        self.pause_btn.config(text="Pause")
        self.pause_btn.config(state=tk.DISABLED)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _reset_series_progress(self):
        """Reset the series progress bar for self.current_series_nr"""
        num_files = (
            self.dicom_series.SeriesNumber == self.current_series_nr
        ).sum()
        self.series_progress.config({"maximum": num_files})
        self.series_progress.config({"value": 0})
        self.series_progress_label.config(
            text=f"Series Progress: 0/{num_files}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def show_series_loading_progress(self, show=True):
        """Show or hide the series loading progress bar"""
        if show:
            self.series_loading_label.pack(fill=tk.X, padx=5, pady=2)
            self.series_loading_progress.pack(fill=tk.X, padx=5, pady=2)
            self.rescan_btn.config(text="Cancel")
        else:
            self.series_loading_label.pack_forget()
            self.series_loading_progress.pack_forget()
            self.rescan_btn.config(text="Rescan")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def handle_rescan_click(self):
        """Handle Rescan/Cancel button click"""
        if self.simulation_running:
            return

        if self.rescan_btn.config("text")[4] == "Cancel":
            # Cancel current scan
            self.scanning_cancelled = True
            self.log_message("Series scan cancelled by user")
            self.show_series_loading_progress(False)
        else:
            # Start new scan
            self.refresh_series_info()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _get_cache_file_path(self, session_path):
        """Get the path for the cached series info file"""
        return session_path / ".rtmri_series_cache.csv"

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _populate_series_tree(self):
        """Populate the series tree from self.dicom_series"""
        if self.dicom_series is not None and len(self.dicom_series) > 0:
            ser_nums = sorted(self.dicom_series.SeriesNumber.unique())
            for ser in ser_nums:
                ser_rows = self.dicom_series[
                    self.dicom_series.SeriesNumber == ser
                ]
                ser_row = ser_rows.iloc[0].squeeze()
                series_data = (
                    f"Series {ser}",
                    len(ser_rows),
                    ser_row.get("ImageType", ""),
                    ser_row.get("SeriesDescription", ""),
                )
                self.series_tree.insert("", "end", values=series_data)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def refresh_series_info(self):
        """Refresh series information by rescanning"""
        if self.simulation_running:
            return

        if not self.current_session_path:
            self.err_message("No session selected")
            return

        self.log_message("Rescanning series information...")
        thread = threading.Thread(
            target=self._load_series_info_threaded,
            args=(self.current_session_path, True),
        )
        thread.daemon = True
        thread.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _load_series_info_threaded(self, session_path, rescan=False):
        """Load the series information with progress (run in a worker
        thread). The result is cached in the session directory."""

        self._load_series_lock.acquire()

        self.current_session_path = session_path
        self.scanning_cancelled = False

        # Tk widgets must be updated in the main thread, via root.after().
        self.root.after(0, self.show_series_loading_progress, True)
        self.root.after(
            0, self.series_loading_label.config, {"text": "Checking cache..."}
        )

        try:
            self.root.after(0, self._clear_series_tree)

            if self.scanning_cancelled:
                self.dicom_series = None
                self._load_series_lock.release()
                return

            series_df = None
            cache_file = self._get_cache_file_path(session_path)

            if not rescan and cache_file.exists():
                try:
                    series_df = pd.read_csv(cache_file)
                    self.log_message("Loaded series info from cache")
                except Exception as e:
                    errstr = str(e) + "\n" + traceback.format_exc()
                    self.log_message(f"Error loading cache: {errstr}")
                    series_df = None

            if series_df is None:
                # No cache; scan all files
                self.root.after(
                    0,
                    self.series_loading_label.config,
                    {"text": "Finding DICOM files..."},
                )

                dicom_files = list(session_path.glob("**/*.dcm"))

                if self.scanning_cancelled:
                    self.dicom_series = None
                    self._load_series_lock.release()
                    return

                if not dicom_files:
                    self.root.after(
                        0, self.log_message, "No DICOM files found in session"
                    )
                    self.root.after(
                        0, self.show_series_loading_progress, False
                    )
                    self._load_series_lock.release()
                    return

                total_files = len(dicom_files)
                progress_config = {"maximum": total_files}
                self.root.after(
                    0, self.series_loading_progress.config, progress_config
                )
                self.root.after(
                    0, self.series_loading_progress.config, {"value": 0}
                )
                self.root.after(
                    0,
                    self.series_loading_label.config,
                    {"text": f"Processing {total_files} DICOM files..."},
                )

                series_df = self.load_series_info(
                    session_path,
                    progress_callback=self._update_series_progress
                )
                if series_df is not None:
                    series_df.to_csv(cache_file, index=False)

            if series_df is not None and not self.scanning_cancelled:
                if "AcqSec" not in series_df.columns:
                    series_df["AcqSec"] = np.nan
                    self.log_message(
                        "The cache has no acquisition times. TTLs are sent "
                        "one TR before each volume's image. Rescan the "
                        "session to simulate the acquisition timing."
                    )
                self.dicom_series = series_df
                self.root.after(0, self._populate_series_tree)
                self.log_message("Series information loaded successfully")
            else:
                self.dicom_series = None

            self.root.after(0, self.show_series_loading_progress, False)

        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            if not self.scanning_cancelled:
                self.root.after(
                    0, self.err_message, f"Error loading series info: {errstr}"
                )
        finally:
            self.root.after(0, self.show_series_loading_progress, False)

        self._load_series_lock.release()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _clear_series_tree(self):
        """Clear the series tree (must be called from the main thread)"""
        for item in self.series_tree.get_children():
            self.series_tree.delete(item)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _update_series_progress(self, current, message=""):
        """Update series loading progress (called from worker thread)"""
        self.root.after(
            0, self.series_loading_progress.config, {"value": current}
        )
        if message:
            self.root.after(
                0, self.series_loading_label.config, {"text": message}
            )

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def load_series_info(self, session_path, progress_callback=None):
        """Read the DICOM information of all files in the session.

        Returns a DataFrame with a row per file (sorted by ContentTime), or
        None if no file is found, the scan is canceled, or an error occurs.
        progress_callback(n, message) is called after each file; when it is
        given (worker thread), log messages are passed to the main thread.
        """
        if not progress_callback:
            self.log_message("Loading series information...")

        try:
            dicom_files = list(session_path.glob("**/*.dcm"))
            if not dicom_files:
                if not progress_callback:
                    self.log_message("No DICOM files found in session")
                return

            series_dfs = []
            reader = DicomReader()
            for i, dcm_file in enumerate(dicom_files):
                if progress_callback and self.scanning_cancelled:
                    return

                try:
                    info = reader.read_dicom_info(dcm_file, timeout=1)
                    if info:
                        info["FilePath"] = dcm_file
                        info["AcqSec"] = read_acq_sec(dcm_file)
                        series_dfs.append(pd.DataFrame([info]))
                except Exception as e:
                    errstr = str(e) + "\n" + traceback.format_exc()
                    if not progress_callback:
                        self.log_message(f"Error reading {dcm_file}: {errstr}")

                if progress_callback:
                    progress_callback(
                        (i + 1),
                        f"Reading ... ({i + 1}/{len(dicom_files)})",
                    )

            series_df = pd.concat(series_dfs, ignore_index=True)
            series_df.sort_values(
                "ContentTime", inplace=True,
                key=lambda t: t.map(dicom_time_to_sec))

            message = (
                f"Loaded {len(series_df.SeriesNumber.unique())} series "
                f"with {len(series_df)} total files"
            )

            if progress_callback:
                self.root.after(0, self.log_message, message)
            else:
                self.log_message(message)

            return series_df

        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            error_msg = f"Error loading series info: {errstr}"
            if progress_callback:
                self.root.after(0, self.log_message, error_msg)
            else:
                self.log_message(error_msg)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_src_dir(self, directory=None):
        """Set the source data root directory and refresh the session
        list"""
        if self.simulation_running:
            return

        if directory is None:
            directory = filedialog.askdirectory(
                title="Select Test Source Data Directory",
                initialdir=self.config["src_data_dir"],
            )

        if directory:
            self.config["src_data_dir"] = Path(directory)
            self.src_dir_var.set(str(directory))
            self.log_message(
                f"Source data directory changed to: {directory}"
            )

            # Clear the current session and series
            self.current_session_path = None
            self.current_session_var.set("No session selected")
            self._clear_series_tree()
            self.dicom_series = []

            self.load_dicom_sessions()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def load_dicom_sessions(self):
        """List the DICOM sessions in the source data directory"""
        self.log_message("Loading DICOM sessions...")

        self.disable_session_treeview()
        try:
            for item in self.session_tree.get_children():
                self.session_tree.delete(item)

            src_data_dir = Path(self.config["src_data_dir"])
            sessions = []

            self._find_sessions_recursive(
                src_data_dir, src_data_dir, sessions, max_depth=3
            )

            # Remove duplicates and sort sessions
            unique_sessions = {}
            for session in sessions:
                unique_sessions[session["path"]] = session
            sessions = list(unique_sessions.values())
            sessions.sort(key=lambda x: x["name"])

            # Add to treeview
            for session in sessions:
                self.session_tree.insert(
                    "",
                    "end",
                    values=(
                        session["name"],
                        session["files"],
                        session["study_description"],
                    ),
                )

            self.dicom_series = []
            self._clear_series_tree()

            self.log_message(
                f"Loaded {len(sessions)} DICOM sessions from {src_data_dir}"
            )

        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            self.err_message(f"Error loading DICOM sessions: {errstr}")

        finally:
            self.enable_session_treeviews()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_output_directory(self, directory=None):
        """Set the output directory of the simulated DICOM export"""
        if self.simulation_running:
            return

        if directory is None:
            directory = filedialog.askdirectory(
                title="Select Output Directory",
                initialdir=self.config["simulation_output_dir"],
            )
        if directory:
            self.config["simulation_output_dir"] = Path(directory)
            self.output_dir_var.set(str(directory))
            self.log_message(f"Output directory changed to: {directory}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def clear_output_directory(self):
        """Delete all files in the output directory (after confirmation)"""
        if self.simulation_running:
            return

        if (
            self.config["simulation_output_dir"] is not None and
            Path(self.config["simulation_output_dir"]).exists()
        ):
            if messagebox.askyesno(
                "Confirm Clear",
                ("Are you sure you want to clear all files "
                 "in the output directory: "
                 f"{self.config['simulation_output_dir']}?")
            ):
                for item in (
                    Path(self.config["simulation_output_dir"]).iterdir()
                ):
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)

                self.log_message("Output directory cleared")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_cardiogram_file(self, filename=None):
        """Set the cardiogram file"""
        if filename is None:
            filename = filedialog.askopenfilename(
                title="Select Cardiogram File",
                initialdir=self.current_session_path,
                filetypes=[("1D files", "*.1D"), ("All files", "*.*")],
            )
        if filename:
            self.selected_card_file = Path(filename)
            self.card_file_var.set(filename)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_respiration_file(self, filename=None):
        """Set the respiration file"""
        if filename is None:
            filename = filedialog.askopenfilename(
                title="Select Respiration File",
                initialdir=self.current_session_path,
                filetypes=[("1D files", "*.1D"), ("All files", "*.*")],
            )
        if filename:
            self.selected_resp_file = Path(filename)
            self.resp_file_var.set(filename)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def validate_simulation_setup(self):
        """Check the session and series, and create the output directory"""
        if not self.current_session_path:
            messagebox.showerror("Error", "Please select a DICOM session")
            return False

        if self.dicom_series is None or len(self.dicom_series) == 0:
            messagebox.showerror(
                "Error", "No DICOM series found in selected session"
            )
            return False

        try:
            output_dir = Path(self.output_dir_var.get())
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            messagebox.showerror(
                "Error", f"Cannot create output directory: {errstr}"
            )
            return False

        return True

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def start_simulation(self, resume=False):
        """Start (or resume) the simulation of the selected series"""
        if not self.validate_simulation_setup():
            return

        if not self._set_selected_series_nr():
            return

        if not resume:
            self.current_image_nr = 0

        # Update UI
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL, text="Pause")
        self.step_btn.config(state=tk.DISABLED)
        self.disable_treeviews()
        self._set_outpath_buttons_state(tk.DISABLED)

        self.simulation_running = True
        self.simulation_thread = threading.Thread(
            target=self.run_simulation, daemon=True
        )
        self.simulation_thread.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def step_simulation(self):
        """Copy the next image of the selected series"""
        if not self.validate_simulation_setup():
            return

        if not self._set_selected_series_nr():
            return

        self.simulation_running = True
        step_thread = threading.Thread(
            target=self.run_simulation,
            args=(True,),  # one_step=True
            daemon=True,
        )
        step_thread.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_selected_series_nr(self):
        """Set self.current_series_nr to the series selected in the tree.
        Returns False if no series is selected."""
        selection = self.series_tree.selection()
        if not selection:
            return False

        series_name = self.series_tree.item(selection[0])["values"][0]
        self.current_series_nr = int(series_name.split()[1])
        return True

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_outpath_buttons_state(self, state):
        """Set the state of the buttons in the output path frame"""
        for widget in self.outpath_frame.winfo_children():
            if isinstance(widget, ttk.Frame):
                for nested_widget in widget.winfo_children():
                    if isinstance(nested_widget, ttk.Button):
                        nested_widget.config(state=state)
            elif isinstance(widget, ttk.Button):
                widget.config(state=state)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run_simulation(self, one_step=False):
        """Simulation loop (run in a worker thread).

        Copies the files of the current series at their acquisition timing.
        With auto-advance, continues to the next series after the series
        interval. With one_step, copies only one file.
        """
        try:
            output_base_dir = (
                Path(self.output_dir_var.get()) /
                self.current_session_path.name
            )

            # Loop over the series while auto-advancing
            while (
                self.current_series_nr <= self.dicom_series.SeriesNumber.max()
            ):
                series_df = self.dicom_series[
                    self.dicom_series.SeriesNumber == self.current_series_nr
                ].copy()

                # Times in seconds (ContentTime is HHMMSS.ffffff, which is
                # not a number of seconds)
                series_df["ContentSec"] = content_sec(
                    series_df["ContentTime"])
                series_df.sort_values("ContentSec", inplace=True)
                series_df.reset_index(drop=True, inplace=True)
                total_files = len(series_df)

                # Resume from the current image
                if self.current_image_nr > 0:
                    series_df = series_df.iloc[self.current_image_nr:]

                # TTLs at the acquisition times, files at their image
                # creation times + the transfer delay (see series_timing)
                try:
                    transfer_delay = float(self.transfer_delay_var.get())
                except (tk.TclError, ValueError):
                    transfer_delay = 0.0
                file_t, ttl_t = series_timing(series_df, transfer_delay)

                rel_path = Path(series_df["FilePath"].values[0]).relative_to(
                    self.current_session_path
                )

                series_output_dir = output_base_dir / rel_path.parent
                if not series_output_dir.exists():
                    series_output_dir.mkdir(parents=True, exist_ok=True)

                # Update series progress bar
                self.root.after(
                    0,
                    self.series_progress.config,
                    {"value": self.current_image_nr},
                )

                # Phase images ('<description>_Pha' series) are copied with
                # the magnitude images.
                ser_desc = series_df.iloc[0].SeriesDescription
                phase_ser = f"{ser_desc}_Pha"
                if np.any(
                    phase_ser in self.dicom_series.SeriesDescription.unique()
                ):
                    pha_series_df = self.dicom_series[
                        self.dicom_series.SeriesDescription == phase_ser
                    ].copy()
                else:
                    pha_series_df = None

                # region: Simulate file creation in a series ------------------
                # TTLs on time, images late by the acquisition (TR), the
                # reconstruction, and the transfer.
                if not series_output_dir.is_dir():
                    series_output_dir.mkdir(parents=True, exist_ok=True)
                st_time = time.time()
                n_ttl = 0
                for ii, (_, row) in enumerate(series_df.iterrows()):
                    dicom_file = Path(row["FilePath"])
                    dest_file = series_output_dir / dicom_file.name
                    if pha_series_df is not None:
                        pha_file = pha_series_df[
                            pha_series_df.InstanceNumber == row.InstanceNumber
                        ]["FilePath"]
                        pha_file = Path(pha_file.values[0])
                        dest_pha_file = series_output_dir / pha_file.name
                    else:
                        dest_pha_file = None

                    # Send the TTLs due before this file, then wait for the
                    # file time
                    while self.simulation_running:
                        now = time.time() - st_time
                        while n_ttl < len(ttl_t) and ttl_t[n_ttl] <= now:
                            self.send_pulse()
                            n_ttl += 1
                        if now >= file_t[ii]:
                            break
                        time.sleep(0.005)

                    if not self.simulation_running:
                        break

                    shutil.copy(dicom_file, dest_file)
                    self.current_image_nr = self.current_image_nr + 1

                    self.root.after(
                        0,
                        self.series_progress.config,
                        {"value": self.current_image_nr},
                    )
                    self.root.after(
                        0,
                        self.series_progress_label.config,
                        {
                            "text":
                                ("Series Progress: "
                                 f"{self.current_image_nr}/{total_files}")
                        }
                    )
                    self.root.after(
                        0,
                        self.log_message,
                        (
                            f"Copy file {self.current_image_nr}/"
                            f"{len(series_df)}: "
                            f"{dicom_file.name}"
                        ),
                    )

                    if dest_pha_file is not None:
                        shutil.copy(pha_file, dest_pha_file)
                        self.root.after(
                            0,
                            self.log_message,
                            (
                                f"Copy file {self.current_image_nr}/"
                                f"{len(series_df)}: "
                                f"{pha_file.name}"
                            ),
                        )

                    if self.current_image_nr < len(series_df) and one_step:
                        self.root.after(0, self.simulation_finished)
                        return

                if not self.simulation_running:
                    break

                self.root.after(
                    0,
                    self.log_message,
                    f"Completed series {self.current_series_nr}",
                )

                if self.auto_advance_var.get():
                    # Select the next series in the tree
                    current_selection = self.series_tree.selection()
                    if current_selection:
                        current_item = current_selection[0]
                        next_item = self.series_tree.next(current_item)
                        if next_item:
                            self.root.after(
                                0, self.series_tree.selection_set, next_item
                            )
                            self.root.after(
                                0, self.series_tree.focus, next_item
                            )
                            item = self.series_tree.item(next_item)
                            series_name = item["values"][0]
                            series_files = item["values"][1]
                            series_description = item["values"][-1]

                            series_number = int(series_name.split()[1])
                            self.current_series_nr = series_number

                            series_status = (
                                f"{series_name}: {series_description} "
                                f"({series_files} files)"
                            )
                            self.root.after(
                                0, self.current_series_var.set, series_status
                            )

                            self.current_image_nr = 0
                            self.root.after(0, self._reset_series_progress)

                            # Wait for the series interval
                            series_interval = float(
                                self.series_interval_var.get())
                            self.root.after(
                                0,
                                self.log_message,
                                ("Waiting for next series for "
                                 f"{series_interval} s"),
                            )

                            wait_start = time.perf_counter()
                            while (
                                time.perf_counter() - wait_start <
                                series_interval
                            ):
                                if not self.simulation_running:
                                    break
                                time.sleep(0.1)
                        else:
                            # The last series is done
                            self.root.after(
                                0, self.series_tree.selection_clear
                            )
                            self.current_series_nr = 0
                            break
                else:
                    break

            if not self.simulation_running:
                self.root.after(
                    0, self.log_message, "Simulation stopped by user"
                )

        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            self.root.after(0, self.err_message, f"Simulation error: {errstr}")

        finally:
            self.root.after(0, self.simulation_finished)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def pause_simulation(self):
        """Pause/resume simulation"""
        if self.pause_btn.cget("text") == "Resume":
            self.pause_btn.config(text="Pause")
            self.start_simulation(resume=True)
            self.log_message("Simulation resumed")
        else:
            self.simulation_running = False
            self.pause_btn.config(text="Resume")
            self.log_message("Simulation paused")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def stop_simulation(self):
        """Stop simulation"""
        self.simulation_running = False
        self.log_message("Stopping simulation...")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def simulation_finished(self):
        """Restore the controls when the simulation ends or pauses"""
        self.start_btn.config(state=tk.NORMAL)
        self.step_btn.config(state=tk.NORMAL)
        self.enable_treeviews()
        self._set_outpath_buttons_state(tk.NORMAL)

        self.simulation_running = False

        num_imgs = (
            self.dicom_series.SeriesNumber == self.current_series_nr
        ).sum()
        if self.current_image_nr < num_imgs:
            # Paused in the middle of the series
            self.pause_btn.config(state=tk.NORMAL, text="Resume")
        else:
            # Series completed
            self.pause_btn.config(state=tk.DISABLED, text="Pause")

            # Select the next series
            current_selection = self.series_tree.selection()
            current_item = current_selection[0]
            next_item = self.series_tree.next(current_item)
            if next_item:
                self.root.after(
                    0, self.series_tree.selection_set, next_item
                )
                self.root.after(0, self.series_tree.focus, next_item)
                self.root.after(0, self.on_series_select, None)
                item = self.series_tree.item(next_item)
                series_name = item["values"][0]
                series_number = int(series_name.split()[1])
                self.current_series_nr = series_number
                self.current_image_nr = 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ping_physio_rpc(self, popup=True):
        """Ping the rt_physio process to check that it responds"""
        if self.rpc_physio_com.rpc_ping():
            self.log_message("Physio process ping successful")
            if popup:
                messagebox.showinfo(
                    "Physio RPC Status",
                    "Physio process ping successful",
                )
            return True
        else:
            self.err_message("Physio process ping failed")  # Shows a popup
            return False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def start_physio_feeding(self):
        """Start the dummy physio feeding with the selected files"""
        if not self.ping_physio_rpc(popup=False):
            return

        # Send selected physio files to rt_physio for simulation
        card_file = (
            str(self.selected_card_file.resolve())
            if self.selected_card_file
            else ""
        )
        resp_file = (
            str(self.selected_resp_file.resolve())
            if self.selected_resp_file
            else ""
        )

        if not card_file or not resp_file:
            self.err_message("No physio files selected.")  # Shows a popup
            return

        try:
            command = (
                "START_DUMMY_FEEDING_WITH_FILES",
                (card_file, resp_file, self.physio_freq_var.get()),
            )
            if self.rpc_physio_com.rpc_ping():
                self.rpc_physio_com.call_rt_proc(
                    command, pkl=True)
                self.log_message("Start physio feeding")
            else:
                self.err_message("Failed to start physio feeding")

        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            self.err_message(
                f"Error communicating with physio process: {errstr}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def reset_physio(self):
        """Stop the physio feeding by resetting the recording device"""
        self.log_message("Stop physio feeding")
        try:
            self.rpc_physio_com.call_rt_proc(
                ("SET_REC_DEV", None),
                pkl=True,
            )
        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            self.err_message(f"Error stopping physio feeding: {errstr}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def send_pulse(self):
        """Send a TTL pulse to the rt_physio process"""
        self.log_message("Send pulse")
        try:
            self.rpc_physio_com.call_rt_proc("TTL_PULSE")
        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            self.err_message(f"Error sending pulse: {errstr}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def log_message(self, message):
        """Add a message to the log and the status bar"""
        if threading.current_thread() is not threading.main_thread():
            # Tk widgets must be updated in the main thread
            self.root.after(0, self.log_message, message)
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"

        # Before the log panel is made, print the message.
        if hasattr(self, 'log_text'):
            self.log_text.insert(tk.END, formatted_message)
            self.log_text.see(tk.END)
            self.status_var.set(message)
        else:
            print(formatted_message)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def err_message(self, message):
        """Add an error message (red) to the log and show a popup"""
        if threading.current_thread() is not threading.main_thread():
            # Tk widgets must be updated in the main thread
            self.root.after(0, self.err_message, message)
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"

        # The log text box stays editable (state NORMAL) so that the
        # following messages are also shown.
        self.log_text.tag_configure("error", foreground="red")
        self.log_text.insert(tk.END, formatted_message, "error")

        self.status_var.set(message)
        messagebox.showerror("Error", message)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def load_config(self):
        """Load the settings from the config file and apply them"""
        config_file = self.CONFIG_FILE

        try:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    saved_config = json.load(f)

                # Update the config, converting paths back to Path
                for key, value in saved_config.items():
                    if key in [
                        "src_data_dir",
                        "simulation_output_dir",
                        "current_session_path",
                        "selected_card_file",
                        "selected_resp_file"
                    ]:
                        self.config[key] = Path(value)
                    else:
                        self.config[key] = value

                self.log_message(f"Configuration loaded from {config_file}")
            else:
                self.log_message(
                    "Configuration file not found, using defaults"
                )

            # Set the widgets from the config
            if (
                "src_data_dir" in self.config and
                Path(self.config["src_data_dir"]).exists()
            ):
                self.set_src_dir(directory=self.config["src_data_dir"])

            if (
                "simulation_output_dir" in self.config and
                Path(self.config["simulation_output_dir"]).exists()
            ):
                self.set_output_directory(
                    directory=str(self.config["simulation_output_dir"])
                )
            if (
                "selected_card_file" in self.config and
                Path(self.config["selected_card_file"]).exists()
            ):
                self.set_cardiogram_file(
                    filename=str(self.config["selected_card_file"])
                )
            if (
                "selected_resp_file" in self.config and
                Path(self.config["selected_resp_file"]).exists()
            ):
                self.set_respiration_file(
                    filename=str(self.config["selected_resp_file"])
                )
            if (
                "current_session_path" in self.config and
                Path(self.config["current_session_path"]).exists()
            ):
                self.on_session_select(
                    session=self.config["current_session_path"]
                )

            if "auto_advance" in self.config:
                self.auto_advance_var.set(self.config["auto_advance"])

            if "series_interval" in self.config:
                self.series_interval_var.set(self.config["series_interval"])
            if "transfer_delay" in self.config:
                self.transfer_delay_var.set(self.config["transfer_delay"])

        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            # Continue with the default settings
            self.log_message(f"Error loading configuration: {errstr}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save_config(self):
        """Save the current settings to the config file"""
        config_file = self.CONFIG_FILE

        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)

            # Update the config with the current GUI values
            self.config["src_data_dir"] = Path(self.src_dir_var.get())
            self.config["simulation_output_dir"] = Path(
                self.output_dir_var.get())
            self.config["auto_advance"] = self.auto_advance_var.get()
            self.config["series_interval"] = self.series_interval_var.get()
            self.config["transfer_delay"] = self.transfer_delay_var.get()

            if self.current_session_path:
                self.config["current_session_path"] = self.current_session_path

            if self.selected_card_file:
                self.config["selected_card_file"] = self.selected_card_file

            if self.selected_resp_file:
                self.config["selected_resp_file"] = self.selected_resp_file

            # Path is not JSON serializable
            serializable_config = {}
            for key, value in self.config.items():
                if isinstance(value, Path):
                    serializable_config[key] = str(value)
                else:
                    serializable_config[key] = value

            with open(config_file, 'w') as f:
                json.dump(serializable_config, f, indent=2)

            self.log_message(f"Configuration saved to {config_file}")

        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            self.err_message(f"Error saving configuration: {errstr}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def disable_treeviews(self):
        """Disable both session and series trees, and the related controls"""
        self._block_tree_input(self.series_tree)
        self.disable_session_treeview()
        self.rescan_btn.config(state=tk.DISABLED)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def disable_session_treeview(self):
        """Disable the session tree and the buttons in its frame"""
        self._block_tree_input(self.session_tree)
        self._set_buttons_state(self.session_tree.master.master, tk.DISABLED)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def enable_treeviews(self):
        """Re-enable both session and series trees, and the related
        controls"""
        self._unblock_tree_input(self.series_tree)
        self.session_tree.bind("<<TreeviewSelect>>", self.on_session_select)
        self.rescan_btn.config(state=tk.NORMAL)
        self.enable_session_treeviews()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def enable_session_treeviews(self):
        """Re-enable the session tree and the buttons in its frame"""
        self._unblock_tree_input(self.session_tree)
        self.series_tree.bind("<<TreeviewSelect>>", self.on_series_select)
        self._set_buttons_state(self.session_tree.master.master, tk.NORMAL)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    _BLOCKED_EVENTS = ("<Button-1>", "<Button-3>", "<Double-Button-1>",
                       "<Key>")

    def _block_tree_input(self, tree):
        """Block mouse and key input on a treeview"""
        for ev in self._BLOCKED_EVENTS:
            tree.bind(ev, lambda e: "break")

    def _unblock_tree_input(self, tree):
        """Remove the input blocking set by _block_tree_input()"""
        for ev in self._BLOCKED_EVENTS:
            tree.unbind(ev)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_buttons_state(self, frame, state):
        """Recursively set the state of all buttons in a frame"""
        try:
            for widget in frame.winfo_children():
                if isinstance(widget, (ttk.Button, tk.Button)):
                    widget.config(state=state)
                elif isinstance(widget, (ttk.Frame, tk.Frame, ttk.LabelFrame)):
                    self._set_buttons_state(widget, state)
        except Exception:
            pass

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run(self):
        """Run the GUI"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def on_closing(self):
        """Save the settings and close the window"""
        if self.simulation_running:
            if messagebox.askokcancel(
                "Quit", "Simulation is running. Stop and quit?"
            ):
                self.stop_simulation()
            else:
                return

        self.save_config()
        self.root.destroy()


# %% __main__ =================================================================
if __name__ == "__main__":
    app = RTMRISimulator()
    app.run()
