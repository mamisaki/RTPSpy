#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RT-MRI Simulator

A comprehensive GUI-based simulation tool for the RT-MRI system that allows:
- Interactive selection of DICOM session directories
- Selection of cardiogram and respiration recording files
- Series-by-series simulation with real-time controls
- Integration with the RT-MRI physio monitoring system

Based on rt_physio.py and rtmri_simulator.py

Author: mmisaki@laureateinstitute.org
"""

# %% import libraries =========================================================
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import time
import shutil
import pickle
import logging
import socket
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd

from rtpspy.dicom_reader import DicomReader
import traceback
from rtpspy.rpc_socket_server import (
    get_port_by_name,
    rpc_send_data,
    rpc_recv_data,
)


# %% RTMRISimulator ===========================================================
class RTMRISimulator:
    """GUI-based RT-MRI simulator with series-by-series control"""

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("RT-MRI Simulator")
        self.root.geometry("900x850")

        # Configure default fonts for better readability
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
        self.physio_rpc_port = None

        # Initialize status variables
        self.current_image_nr = 0
        self.simulation_running = False
        self.simulation_thread = None
        self.scanning_cancelled = False

        # Initialize logger
        self.logger = logging.getLogger("RTMRISimulator")

        # Configuration
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
        }

        # Restore config
        self.load_config()

        self.setup_gui()
        self.load_dicom_sessions()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def setup_gui(self):
        """Create the main GUI interface"""
        # Create main frame with scrollbar
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create main content area and log area
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # Left panel for session selection
        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Right panel for simulation controls (narrower)
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))

        # Bottom physio panel with full width
        physio_panel = ttk.Frame(main_frame)
        physio_panel.pack(fill=tk.BOTH, expand=False, pady=(5, 0))

        # Bottom log panel with full width
        log_panel = ttk.Frame(main_frame)
        log_panel.pack(fill=tk.BOTH, expand=False, pady=(5, 0))

        # Setup panels
        self.setup_session_panel(left_panel)
        self.setup_simulation_panel(right_panel)
        self.setup_physio_panel(physio_panel)
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
        """Setup the DICOM session and series selection panels"""
        # region: DICOM Sessions ----------------------------------------------
        session_frame = ttk.LabelFrame(parent, text="DICOM Sessions")
        session_frame.pack(fill=tk.BOTH, pady=(0, 5))

        # Source data path
        src_path_frame = ttk.Frame(session_frame)
        src_path_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(src_path_frame, text="Source Root:").pack(side=tk.LEFT)

        # Current source data path (scrollable)
        self.src_dir_var = tk.StringVar(value=str(self.config["src_data_dir"]))
        src_path_entry = tk.Entry(
            src_path_frame,
            textvariable=self.src_dir_var,
            font=("Arial", 10),
            state="readonly",
            readonlybackground="white",
            width=40,
        )
        src_path_entry.configure(exportselection=True)
        src_path_entry.pack(side=tk.LEFT, padx=2)

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

        # Progress bar for series loading with Rescan button
        self.series_progress_frame = ttk.Frame(series_frame)
        self.series_progress_frame.pack(fill=tk.X, padx=5, pady=2)

        # Create a frame for progress elements (left side)
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

        # Initially hide the progress bar elements
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

        # Bind series selection event to update Simulation panel
        self.series_tree.bind("<<TreeviewSelect>>", self.on_series_select)
        # endregion: Series information ---------------------------------------

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def setup_simulation_panel(self, parent):
        """Setup the simulation control panel"""
        sim_frame = ttk.LabelFrame(parent, text="Simulation Control")
        sim_frame.pack(fill=tk.BOTH, pady=(0, 5))

        # region: Output path -------------------------------------------------
        self.outpath_frame = ttk.Frame(sim_frame)
        self.outpath_frame.pack(fill=tk.X, padx=5, pady=5)

        # Top row for label, entry, and set button
        top_row = ttk.Frame(self.outpath_frame)
        top_row.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(top_row, text="Output:").pack(side=tk.LEFT, padx=2)

        # Show output path
        self.output_dir_var = tk.StringVar(
            value=str(self.config["simulation_output_dir"])
        )
        output_path_entry = tk.Entry(
            top_row,
            textvariable=self.output_dir_var,
            font=("Arial", 10),
            state="readonly",
            readonlybackground="white",
        )
        output_path_entry.configure(exportselection=True)
        output_path_entry.pack(side=tk.LEFT, padx=2)

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

        # Auto advance setting
        aa_frame = ttk.Frame(params_frame)
        aa_frame.pack(fill=tk.X, padx=5, pady=5)

        self.auto_advance_var = tk.BooleanVar(
            value=self.config["auto_advance"]
        )
        ttk.Checkbutton(
            aa_frame, text="Auto-advance", variable=self.auto_advance_var
        ).pack(side=tk.LEFT, padx=5)

        # Series interval box
        self.series_interval_var = tk.IntVar(
            value=self.config["series_interval"])
        ttk.Label(aa_frame, text="Series Interval:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(
            aa_frame,
            textvariable=self.series_interval_var,
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

        # Progress bars
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
    def setup_physio_panel(self, parent):
        """Setup the physiological data selection panel"""
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
        card_path_entry = tk.Entry(
            cardiogram_row,
            textvariable=self.card_file_var,
            font=("Arial", 10),
            state="readonly",
            readonlybackground="white",
            width=40,
        )
        card_path_entry.configure(exportselection=True)
        card_path_entry.pack(side=tk.LEFT, padx=2)

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
        resp_path_entry = tk.Entry(
            respiration_row,
            textvariable=self.resp_file_var,
            font=("Arial", 10),
            state="readonly",
            readonlybackground="white",
            width=40,
        )
        resp_path_entry.configure(exportselection=True)
        resp_path_entry.pack(side=tk.LEFT, padx=2)

        ttk.Button(
            respiration_row,
            text="Set",
            command=self.set_respiration_file,
        ).pack(side=tk.LEFT, padx=2)

        button_col = ttk.Frame(physio_frame)
        button_col.pack(side=tk.LEFT, padx=5, pady=5)
        # endregion: File Selection -------------------------------------------

        # region: Frequency setting column ------------------------------------
        frequency_col = ttk.Frame(physio_frame)
        frequency_col.pack(side=tk.LEFT, padx=5, pady=5)

        # Frequency setting
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

        # Start Physio feeding button
        self.start_physio_btn = ttk.Button(
            button_col,
            text="Send Physio",
            command=self.start_physio_feeding,
        )
        self.start_physio_btn.pack(side=tk.LEFT, padx=5)

        # Stop Physio feeding button
        self.stop_physio_btn = ttk.Button(
            button_col,
            text="Stop Physio",
            command=self.stop_physio_feeding,
            state=tk.DISABLED,
        )
        self.stop_physio_btn.pack(side=tk.LEFT, padx=5)

        # Check RPC connection button
        ttk.Button(
            button_col,
            text="Check RPC",
            command=self.ping_physio_rpc,
        ).pack(side=tk.LEFT, padx=5)
        # endregion: Command buttons ------------------------------------------

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def setup_log_panel(self, parent):
        """Setup the log panel at the bottom"""
        log_frame = ttk.LabelFrame(parent, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=10, wrap=tk.WORD
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _find_sessions_recursive(
        self, current_dir, base_dir, sessions, max_depth, current_depth=0
    ):
        """Recursively find session directories containing DICOM files"""
        if current_depth > max_depth:
            return

        if not current_dir.exists() or not current_dir.is_dir():
            return

        # Check if current directory contains DICOM files directly
        # (quick check)
        has_direct_dicoms = any(current_dir.glob("*.dcm"))

        # Also check if it has a "dicom" subdirectory with DICOM files
        has_dicom_subdir = False
        dicom_subdir = current_dir / "dicom"
        if dicom_subdir.exists() and dicom_subdir.is_dir():
            has_dicom_subdir = any(dicom_subdir.glob("**/*.dcm"))

        # If this directory has direct DICOM files OR a dicom subdir with files
        if has_direct_dicoms or has_dicom_subdir:
            # Count all DICOM files (only when we know it's a session)
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
                # Special check: If this directory has both series dirs
                # AND other non-series subdirectories, it's likely a container
                all_subdirs = [d for d in current_dir.iterdir() if d.is_dir()]
                non_series_dirs = [
                    d for d in all_subdirs if d not in series_dirs
                ]

                # If we have non-series directories that might be sessions,
                # don't treat this as a session - explore deeper instead
                if len(non_series_dirs) > 0 and current_depth < max_depth:
                    # This looks like a container directory, continue exploring
                    pass
                else:
                    # This looks like session dir with series subdirectories
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
            # Try to read from first few DICOM files
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
    def on_session_select(self, event=None):
        """Handle session selection"""
        if self.simulation_running:
            return

        selection = self.session_tree.selection()
        if not selection:
            self.current_session_var.set("No session selected")
            # Disable simulation buttons when no session selected
            self.start_btn.config(state=tk.DISABLED)
            self.step_btn.config(state=tk.DISABLED)
            return

        item = self.session_tree.item(selection[0])
        session_name = item["values"][0]
        study_description = item["values"][2]

        # Find the session path by searching through the loaded sessions
        # We need to reconstruct the path from the session name
        src_data_dir = Path(self.config["src_data_dir"])

        # Try different path constructions based on session name format
        if "/" in session_name:
            # Session name includes relative path
            # (e.g., "Patientname_19800101/E26580742")
            session_path = src_data_dir / session_name
        else:
            # Simple session name, look in common locations
            possible_paths = [
                src_data_dir / session_name,
            ]

            # Also check all directories for the session
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

            # Find the first existing path
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

        # Update current session display
        if study_description and study_description != "Unknown":
            session_display = f"{session_name} ({study_description})"
        else:
            session_display = session_name
        self.current_session_var.set(session_display)

        self.log_message(f"Selected session: {session_path}")

        # Load series information in a separate thread
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
            # No series selected, disable buttons and clear status
            self.current_series_var.set("No series selected")
            self.start_btn.config(state=tk.DISABLED)
            self.step_btn.config(state=tk.DISABLED)
            return

        item = self.series_tree.item(selection[0])
        series_name = item["values"][0]  # Series column
        series_files = item["values"][1]  # Files column
        series_description = item["values"][-1]  # Description column

        # Update simulation control current status
        series_status = (
            f"{series_name}: {series_description} ({series_files} files)"
        )
        self.current_series_var.set(series_status)

        self.current_series_nr = int(series_name.split()[-1])
        # Reset image number when series selection changes
        self.current_image_nr = 0

        # Reset series progress bar
        num_files = (
            self.dicom_series.SeriesNumber == self.current_series_nr
        ).sum()
        self.series_progress.config({"maximum": num_files})
        self.series_progress.config({"value": 0})
        self.series_progress_label.config(
            text=f"Series Progress: 0/{num_files}")

        # Enable simulation buttons when a series is selected
        self.start_btn.config(state=tk.NORMAL)
        self.step_btn.config(state=tk.NORMAL)
        self.pause_btn.config(text="Pause")
        self.pause_btn.config(state=tk.DISABLED)

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
        return session_path / ".rtmri_series_cache.pkl"

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _save_series_info_cache(self, session_path, series_data):
        """Save series information to cache file"""
        try:
            cache_file = self._get_cache_file_path(session_path)

            # Convert Path objects to strings for serialization
            cache_data = {"timestamp": time.time(), "series": []}

            for series in series_data:
                series_cache = {
                    "number": series["number"],
                    "description": series["description"],
                    "file_count": len(series["files"]),
                    "size": series["size"],
                    "files": [str(f) for f in series["files"]],
                    "tr": series.get("tr"),
                    "image_type": series.get("image_type"),
                }
                cache_data["series"].append(series_cache)

            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)

            self.log_message(f"Cached series info to {cache_file.name}")

        except Exception as e:
            self.err_message(f"Error saving cache: {e}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _populate_series_tree(self):
        """Populate series tree from loaded cache data"""
        if (
            hasattr(self, "dicom_series")
            and self.dicom_series is not None
            and len(self.dicom_series) > 0
        ):
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

        if (
            not hasattr(self, "current_session_path")
            or not self.current_session_path
        ):
            self.err_message("No session selected")
            return

        # Start fresh scan
        self.log_message("Rescanning series information...")
        thread = threading.Thread(
            target=self._load_series_info_threaded,
            args=(self.current_session_path, True),
        )
        thread.daemon = True
        thread.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _load_series_info_threaded(self, session_path, rescan=False):
        """Load series information in a separate thread with progress"""
        # Store current session path for refresh functionality
        self.current_session_path = session_path

        # Reset cancellation flag
        self.scanning_cancelled = False

        # Show progress bar on main thread
        self.root.after(0, self.show_series_loading_progress, True)
        self.root.after(
            0, self.series_loading_label.config, {"text": "Checking cache..."}
        )

        try:
            # Clear existing series on main thread
            self.root.after(0, self._clear_series_tree)

            # Check for cancellation
            if self.scanning_cancelled:
                self.dicom_series = None
                return

            series_df = None
            cache_file = self._get_cache_file_path(session_path)

            # Try to load cache
            if not rescan and cache_file.exists():
                try:
                    series_df = pd.read_csv(cache_file)
                    self.log_message("Loaded series info from cache")
                except Exception as e:
                    self.log_message(f"Error loading cache: {e}")
                    series_df = None

            if series_df is None:
                # If no cache, proceed with full scan
                self.root.after(
                    0,
                    self.series_loading_label.config,
                    {"text": "Finding DICOM files..."},
                )

                # Find all DICOM files
                dicom_files = list(session_path.glob("**/*.dcm"))

                # Check for cancellation
                if self.scanning_cancelled:
                    self.dicom_series = None
                    return

                if not dicom_files:
                    self.root.after(
                        0, self.log_message, "No DICOM files found in session"
                    )
                    self.root.after(
                        0, self.show_series_loading_progress, False
                    )
                    return

                # Update progress bar maximum
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

                # Process files with progress updates
                series_df = self.load_series_info(
                    session_path,
                    progress_callback=self._update_series_progress
                )
                if series_df is not None:
                    series_df.to_csv(cache_file, index=False)

            if series_df is not None and not self.scanning_cancelled:
                self.dicom_series = series_df
                # Schedule tree population on main thread
                self.root.after(0, self._populate_series_tree)
                self.log_message("Series information loaded successfully")
            else:
                # Ensure series data is cleared if cancelled
                self.dicom_series = None

            self.root.after(0, self.show_series_loading_progress, False)

        except Exception as e:
            if not self.scanning_cancelled:
                self.root.after(
                    0, self.err_message, f"Error loading series info: {e}"
                )
        finally:
            # Hide progress bar
            self.root.after(0, self.show_series_loading_progress, False)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _clear_series_tree(self):
        """Clear the series tree (must be called from main thread)"""
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
        """Load DICOM series information for selected session"""
        if not progress_callback:
            self.log_message("Loading series information...")

        try:
            # Find all DICOM files
            dicom_files = list(session_path.glob("**/*.dcm"))

            if not dicom_files:
                if not progress_callback:
                    self.log_message("No DICOM files found in session")
                return

            # Group by series if DicomReader is available
            series_dfs = []

            reader = DicomReader()

            # Scan sampled files to identify series
            for i, dcm_file in enumerate(dicom_files):
                # Check for cancellation
                if progress_callback and self.scanning_cancelled:
                    return

                dcm_file = dicom_files[i]
                try:
                    info = reader.read_dicom_info(dcm_file, timeout=0.2)
                    info["FilePath"] = dcm_file
                    series_dfs.append(pd.DataFrame([info]))
                except Exception as e:
                    if not progress_callback:
                        self.log_message(f"Error reading {dcm_file}: {e}")

                # Update progress for initial scan
                if progress_callback:
                    progress_callback(
                        (i + 1),
                        f"Reading ... ({i + 1}/{len(dicom_files)})",
                    )

            series_df = pd.concat(series_dfs, ignore_index=True)
            series_df.sort_values("ContentTime", inplace=True)

            # Final progress update
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
            error_msg = f"Error loading series info: {e}"
            if progress_callback:
                self.root.after(0, self.log_message, error_msg)
            else:
                self.log_message(error_msg)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_src_dir(self):
        """Set test data root directory and refresh sessions list"""
        if self.simulation_running:
            return

        directory = filedialog.askdirectory(
            title="Select Test Source Data Directory",
            initialdir=self.config["src_data_dir"],
        )
        if directory:
            # Update test data directory configuration
            self.config["src_data_dir"] = Path(directory)
            self.src_dir_var.set(str(directory))
            self.log_message(
                f"Test source data directory changed to: {directory}"
            )

            # Clear current session selection
            self.current_session_path = None
            self.current_session_var.set("No session selected")

            # Clear series information
            for item in self.series_tree.get_children():
                self.series_tree.delete(item)
            self.dicom_series = []

            # Refresh DICOM sessions list with new test data directory
            self.load_dicom_sessions()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def load_dicom_sessions(self):
        """Load available DICOM sessions"""
        self.log_message("Loading DICOM sessions...")

        # Disable session tree during loading
        self.disable_session_treeview()

        try:
            # Clear existing items
            for item in self.session_tree.get_children():
                self.session_tree.delete(item)

            src_data_dir = Path(self.config["src_data_dir"])
            sessions = []

            # Strategy: Look for DICOM sessions in various directory structures
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

            # Clear series information
            self.dicom_series = []
            for item in self.series_tree.get_children():
                self.series_tree.delete(item)

            self.log_message(
                f"Loaded {len(sessions)} DICOM sessions from {src_data_dir}"
            )

        except Exception as e:
            self.err_message(f"Error loading DICOM sessions: {e}")

        finally:
            # Always re-enable session tree when finished
            self.enable_session_treeviews()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_output_directory(self):
        """Set the output directory for simulated feeding"""
        if self.simulation_running:
            return

        directory = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.config["simulation_output_dir"],
        )
        if directory:
            # Update output directory configuration
            self.config["simulation_output_dir"] = Path(directory)
            self.output_dir_var.set(str(directory))
            self.log_message(f"Output directory changed to: {directory}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def clear_output_directory(self):
        """Clear the output directory for simulated feeding"""
        if self.simulation_running:
            return

        if (
            self.config["simulation_output_dir"] is not None and
            Path(self.config["simulation_output_dir"]).exists()
        ):
            # Confirm before clearing
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
    def set_cardiogram_file(self):
        """Set Cardiogram file"""
        filename = filedialog.askopenfilename(
            title="Select Cardiogram File",
            initialdir=self.current_session_path,
            filetypes=[("1D files", "*.1D"), ("All files", "*.*")],
        )
        if filename:
            self.selected_card_file = Path(filename)
            self.card_file_var.set(filename)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_respiration_file(self):
        """Set Respiration file"""
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
        """Validate simulation setup"""
        if not self.current_session_path:
            messagebox.showerror("Error", "Please select a DICOM session")
            return False

        if (
            not hasattr(self, "dicom_series")
            or self.dicom_series is None
            or len(self.dicom_series) == 0
        ):
            messagebox.showerror(
                "Error", "No DICOM series found in selected session"
            )
            return False

        # Create output directory
        try:
            output_dir = Path(self.output_dir_var.get())
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showerror(
                "Error", f"Cannot create output directory: {e}"
            )
            return False

        return True

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def start_simulation(self, resume=False):
        """Start the simulation"""
        if not self.validate_simulation_setup():
            return

        # Get currently selected series from the treeview
        selection = self.series_tree.selection()
        if selection:
            # Extract series number from the selected item
            item = self.series_tree.item(selection[0])
            series_name = item["values"][0]
            series_number = int(series_name.split()[1])
            self.current_series_nr = series_number
        else:
            return

        if not resume:
            self.current_image_nr = 0

        # Update UI
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL, text="Pause")
        self.step_btn.config(state=tk.DISABLED)
        self.disable_treeviews()

        # Disable buttons in outpath_frame
        for widget in self.outpath_frame.winfo_children():
            if isinstance(widget, ttk.Frame):
                # Check nested frames for buttons
                for nested_widget in widget.winfo_children():
                    if isinstance(nested_widget, ttk.Button):
                        nested_widget.config(state=tk.DISABLED)
            elif isinstance(widget, ttk.Button):
                widget.config(state=tk.DISABLED)

        self.simulation_running = True

        # Start simulation thread
        self.simulation_thread = threading.Thread(
            target=self.run_simulation, daemon=True
        )
        self.simulation_thread.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def step_simulation(self):
        """Step through simulation one image at a time"""
        if not self.validate_simulation_setup():
            return

        # Get currently selected series from the treeview
        selection = self.series_tree.selection()
        if selection:
            # Extract series number from the selected item
            item = self.series_tree.item(selection[0])
            series_name = item["values"][0]
            series_number = int(series_name.split()[1])
            self.current_series_nr = series_number
        else:
            return

        # Start step simulation thread
        self.simulation_running = True
        step_thread = threading.Thread(
            target=self.run_simulation,
            args=(True,),  # one_step=True
            daemon=True,
        )
        step_thread.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run_simulation(self, one_step=False):
        """Main simulation loop"""
        try:
            output_base_dir = (
                Path(self.output_dir_var.get()) /
                self.current_session_path.name
            )

            # Loop until session end if auto-advancing
            while (
                self.current_series_nr <= self.dicom_series.SeriesNumber.max()
            ):
                # Extract series data
                series_df = self.dicom_series[
                    self.dicom_series.SeriesNumber == self.current_series_nr
                ].copy()
                series_df["ContentTime"] = series_df["ContentTime"].astype(
                    float
                )
                series_df.sort_values("ContentTime", inplace=True)
                series_df.reset_index(drop=True, inplace=True)
                total_files = len(series_df)

                if self.current_image_nr > 0:
                    series_df = series_df.iloc[self.current_image_nr:]

                if (
                    "FMRI" in series_df.ImageType.values[0] and
                    len(series_df) > 1
                ):
                    TR = np.median(np.diff(series_df["ContentTime"].values))
                    timings = np.linspace(
                        0, TR * (len(series_df) - 1), len(series_df)
                    )
                else:
                    timings = (
                        series_df["ContentTime"].values -
                        series_df["ContentTime"].values[0]
                    )

                rel_path = Path(series_df["FilePath"].values[0]).relative_to(
                    self.current_session_path
                )

                # Create series output directory
                series_output_dir = output_base_dir / rel_path.parent
                if not series_output_dir.exists():
                    series_output_dir.mkdir(parents=True, exist_ok=True)

                # Update series progress bar
                self.root.after(
                    0,
                    self.series_progress.config,
                    {"value": self.current_image_nr},
                )

                # region: Simulate files in series ----------------------------
                st_time = time.time()
                for ii, dicom_file in enumerate(series_df["FilePath"]):
                    dicom_file = Path(dicom_file)
                    while time.time() - st_time < timings[ii]:
                        if not self.simulation_running:
                            break
                        # Wait for content timing
                        time.sleep(0.01)

                    if not self.simulation_running:
                        break

                    # Copy file
                    dest_file = series_output_dir / dicom_file.name
                    shutil.copy2(dicom_file, dest_file)
                    self.current_image_nr = self.current_image_nr + 1

                    # Update progress
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
                    # Log progress
                    self.root.after(
                        0,
                        self.log_message,
                        (
                            f"Copy file {self.current_image_nr}/"
                            f"{len(series_df)}: "
                            f"{dicom_file.name}"
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
                    # Move to the next series in the tree selection
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

                            # Update simulation control current status
                            series_status = (
                                f"{series_name}: {series_description} "
                                f"({series_files} files)"
                            )
                            self.current_series_var.set(series_status)

                            # Reset image number when series selection changes
                            self.current_image_nr = 0

                            # Reset series progress bar
                            num_files = (
                                self.dicom_series.SeriesNumber ==
                                self.current_series_nr
                            ).sum()
                            self.series_progress.config({"maximum": num_files})
                            self.series_progress.config({"value": 0})
                            self.series_progress_label.config(
                                text=f"Series Progress: 0/{num_files}")

                            # Wait for series interval
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
                            # Clear self.series_tree.focus
                            self.series_tree.selection_clear(
                                self.series_tree.focus()
                            )
                            self.current_series_nr = 0
                            break
                else:
                    break

            # Simulation completed
            if not self.simulation_running:
                self.root.after(
                    0, self.log_message, "Simulation stopped by user"
                )

        except Exception as e:
            error_traceback = traceback.format_exc()
            self.root.after(0, self.log_message,
                            f"Traceback:\n{error_traceback}")
            self.root.after(0, self.err_message, f"Simulation error: {e}")

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
        """Handle simulation completion"""
        self.start_btn.config(state=tk.NORMAL)
        self.step_btn.config(state=tk.NORMAL)
        self.enable_treeviews()

        # Enable buttons in outpath_frame
        for widget in self.outpath_frame.winfo_children():
            if isinstance(widget, ttk.Frame):
                # Check nested frames for buttons
                for nested_widget in widget.winfo_children():
                    if isinstance(nested_widget, ttk.Button):
                        nested_widget.config(state=tk.NORMAL)
            elif isinstance(widget, ttk.Button):
                widget.config(state=tk.NORMAL)

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

            # Move the selection next
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
    def ping_physio_rpc(self):
        """Ping the rt_physio.py process to check if it's responsive"""
        response = self.send_rpc_command("ping")
        if response == "pong":
            self.log_message(f"Physio process ping successful: {response}")
            messagebox.showinfo(
                "Physio RPC Status",
                f"Physio process ping successful: {response}",
            )
        else:
            self.err_message(f"Physio process ping failed: {response}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def send_rpc_command(
        self, command, socket_name="RtpTTLPhysioSocketServer", host=None
    ):
        if self.physio_rpc_port is None:
            port, errmsg = get_port_by_name(socket_name, host=host)
            if port is None:
                self.err_message(f"Error getting port: {errmsg}")
                return None
            self.physio_rpc_port = port

        if host is None:
            host = "localhost"
        address = (host, self.physio_rpc_port)

        """Send RPC command to rt_physio.py process"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)  # 5 second timeout
            sock.connect(address)

            # Send the formatted message
            if isinstance(command, tuple):
                rpc_send_data(sock, command, pkl=True)
            else:
                rpc_send_data(sock, command)

            # Receive response (simple string response expected)
            response = rpc_recv_data(sock)

            sock.close()
            return response

        except ConnectionError as e:
            self.err_message(f"RPC connection error: {e}")
            self.physio_rpc_port = None
            return None

        except Exception as e:
            self.err_message(f"RPC communication error: {e}")
            return None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def start_physio_feeding(self):
        """Start physiological feeding with selected files"""

        # First try to connect to existing physiological recording process
        try:
            # Test connection with a simple ping
            response = self.send_rpc_command("ping")
            if response is None:
                raise Exception("No response from existing process")

        except Exception:
            self.err_message(
                "Failed to connect to existing physiological recording process"
            )
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
            self.err_message("No physio files selected.")
            return

        try:
            # Send RPC command to start feeding with selected files
            command = (
                "START_DUMMY_FEEDING_WITH_FILES",
                (card_file, resp_file, self.physio_freq_var.get()),
            )
            response = self.send_rpc_command(command)

            # Handle the case where rt_physio doesn't respond (known issue)
            if response is None:
                # Fallback: assume command was successful but no response
                self.start_physio_btn.config(state=tk.DISABLED)
                self.stop_physio_btn.config(state=tk.NORMAL)
                self.log_message("Start physio feeding")
            elif response and "Error" not in response:
                self.start_physio_btn.config(state=tk.DISABLED)
                self.stop_physio_btn.config(state=tk.NORMAL)
                self.log_message(f"Physio feeding started: {response}")
            else:
                self.err_message(f"Failed to start physio feeding: {response}")

        except Exception as e:
            self.err_message(f"Error communicating with physio process: {e}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def stop_physio_feeding(self):
        """Stop physiological feeding"""
        # Stop physio feeding via RPC
        self.log_message("Stop physio feeding")
        try:
            response = self.send_rpc_command(("SET_REC_DEV", None))
            if response == "Error":
                self.err_message("Failed to stop physio feeding")
        except Exception as e:
            self.err_message(f"Error stopping physio feeding: {e}")

        self.start_physio_btn.config(state=tk.NORMAL)
        self.stop_physio_btn.config(state=tk.DISABLED)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def log_message(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"

        # Update log text
        if hasattr(self, 'log_text'):
            self.log_text.insert(tk.END, formatted_message)
            self.log_text.see(tk.END)
            # Update status
            self.status_var.set(message)
        else:
            print(formatted_message)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def err_message(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"

        # Update log text
        self.log_text.config(state=tk.NORMAL)
        self.log_text.tag_configure("error", foreground="red")
        self.log_text.insert(tk.END, formatted_message, "error")
        self.log_text.config(state=tk.DISABLED)

        # Update status
        self.status_var.set(message)

        # Popup error
        messagebox.showerror("Error", message)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def load_config(self):
        """Load configuration from file"""
        config_file = Path.home() / ".RTPSpy" / "rtmri_simulator_config.json"

        try:
            if config_file.exists():
                import json
                with open(config_file, 'r') as f:
                    saved_config = json.load(f)

                # Update config with saved values, converting string paths
                # back to Path objects
                for key, value in saved_config.items():
                    if key in ["src_data_dir", "simulation_output_dir"]:
                        # Convert string paths back to Path objects
                        self.config[key] = Path(value)
                    else:
                        self.config[key] = value

                self.log_message(f"Configuration loaded from {config_file}")

        except Exception as e:
            self.log_message(f"Error loading configuration: {e}")
            # Continue with default config values

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save_config(self):
        """Save current configuration to file"""
        config_file = Path.home() / ".RTPSpy" / "rtmri_simulator_config.json"

        try:
            # Create config directory if it doesn't exist
            config_file.parent.mkdir(parents=True, exist_ok=True)

            # Update config with current GUI values before saving
            self.config["src_data_dir"] = Path(self.src_dir_var.get())
            self.config["simulation_output_dir"] = Path(
                self.output_dir_var.get())
            self.config["auto_advance"] = self.auto_advance_var.get()
            self.config["series_interval"] = self.series_interval_var.get()

            # Convert Path objects to strings for JSON serialization
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
            self.err_message(f"Error saving configuration: {e}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def disable_treeviews(self):
        """Disable both session and series trees, and all related controls"""
        # Block all user interactions on series tree
        self.series_tree.bind('<Button-1>', lambda e: 'break')
        self.series_tree.bind('<Button-3>', lambda e: 'break')
        self.series_tree.bind('<Double-Button-1>', lambda e: 'break')
        self.series_tree.bind('<Key>', lambda e: 'break')

        # Block all user interactions on session tree
        self.disable_session_treeview()

        # Disable rescan button in series frame
        self.rescan_btn.config(state=tk.DISABLED)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def disable_session_treeview(self):
        """Disable session tree, and all related controls"""
        # Block all user interactions on session tree
        self.session_tree.bind('<Button-1>', lambda e: 'break')
        self.session_tree.bind('<Button-3>', lambda e: 'break')
        self.session_tree.bind('<Double-Button-1>', lambda e: 'break')
        self.session_tree.bind('<Key>', lambda e: 'break')

        # Disable all buttons in session frame (recursively)
        self._disable_buttons_in_frame(self.session_tree.master.master)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def enable_treeviews(self):
        """Re-enable both session and series trees, and all related controls"""
        # Remove the blocking bindings from series tree
        self.series_tree.unbind("<Button-1>")
        self.series_tree.unbind("<Button-3>")
        self.series_tree.unbind("<Double-Button-1>")
        self.series_tree.unbind("<Key>")

        # Restore the original selection events
        self.session_tree.bind("<<TreeviewSelect>>", self.on_session_select)

        # Re-enable rescan button in series frame
        self.rescan_btn.config(state=tk.NORMAL)

        # Re-enable session tree
        self.enable_session_treeviews()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def enable_session_treeviews(self):
        # Remove the blocking bindings from session tree
        self.session_tree.unbind("<Button-1>")
        self.session_tree.unbind("<Button-3>")
        self.session_tree.unbind("<Double-Button-1>")
        self.session_tree.unbind("<Key>")

        # Restore the original selection events
        self.series_tree.bind("<<TreeviewSelect>>", self.on_series_select)

        # Re-enable all buttons in session frame (recursively)
        self._enable_buttons_in_frame(self.session_tree.master.master)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _disable_buttons_in_frame(self, frame):
        """Recursively disable all buttons in a frame"""
        try:
            for widget in frame.winfo_children():
                if isinstance(widget, (ttk.Button, tk.Button)):
                    widget.config(state=tk.DISABLED)
                elif isinstance(widget, (ttk.Frame, tk.Frame, ttk.LabelFrame)):
                    # Recursively check child frames
                    self._disable_buttons_in_frame(widget)
        except Exception:
            pass

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _enable_buttons_in_frame(self, frame):
        """Recursively enable all buttons in a frame"""
        try:
            for widget in frame.winfo_children():
                if isinstance(widget, (ttk.Button, tk.Button)):
                    widget.config(state=tk.NORMAL)
                elif isinstance(widget, (ttk.Frame, tk.Frame, ttk.LabelFrame)):
                    # Recursively check child frames
                    self._enable_buttons_in_frame(widget)
        except Exception:
            pass

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run(self):
        """Run the GUI"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def on_closing(self):
        """Handle window closing"""
        if self.simulation_running:
            if messagebox.askokcancel(
                "Quit", "Simulation is running. Stop and quit?"
            ):
                self.stop_simulation()
            else:
                return

        self.save_config()
        self.root.destroy()


# %% ==========================================================================
if __name__ == "__main__":
    app = RTMRISimulator()
    app.run()
