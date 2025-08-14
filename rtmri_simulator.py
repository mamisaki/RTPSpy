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
import re
import random
import logging
import socket
from pathlib import Path
from datetime import datetime

from rtpspy.dicom_reader import DicomReader
from rtpspy.rpc_socket_server import (
    get_port_by_name, rpc_send_data, rpc_recv_data
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
        self.dicom_series = []
        self.current_series_index = 0

        self.selected_card_file = None
        self.selected_resp_file = None
        self.rpc_port = None

        # Initialize status variables
        self.simulation_running = False
        self.simulation_thread = None
        self.scanning_cancelled = False

        # Initialize logger
        self.logger = logging.getLogger("RTMRISimulator")

        # Configuration
        self.config = {
            "src_data_dir": Path(__file__).parent / "simulation_data",
            "simulation_output_dir": (
                Path(__file__).parent / "RTMRI_simulation"),
            "tr_seconds": 2.0,
            "auto_advance": False,
            "copy_delay_ms": 100,
            "physio_rpc_port": None,
            "ttl_rpc_port": None,
        }

        self.setup_gui()
        self.load_initial_data()

    # Setup the GUI components ++++++++++++++++++++++++++++++++++++++++++++++++
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
        ttk.Label(src_path_frame, text="Source Root:").pack(
            side=tk.LEFT
        )

        # Current source data path (scrollable)
        self.src_dir_var = tk.StringVar(
            value=str(self.config["src_data_dir"])
        )
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
                self.session_tree.column(col, width=200)
            elif col == "Study Description":
                self.session_tree.column(col, width=150)
            else:
                self.session_tree.column(col, width=80)

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
        series_columns = ("Series", "Description", "Files")
        self.series_tree = ttk.Treeview(
            series_frame, columns=series_columns, show="headings", height=10
        )

        for col in series_columns:
            self.series_tree.heading(col, text=col)
            if col == "Description":
                self.series_tree.column(col, width=150)
            else:
                self.series_tree.column(col, width=80)

        series_scrollbar = ttk.Scrollbar(
            series_frame, orient=tk.VERTICAL, command=self.series_tree.yview
        )
        self.series_tree.configure(yscrollcommand=series_scrollbar.set)

        self.series_tree.pack(
            side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5
        )
        series_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind series selection event to update TR state
        self.series_tree.bind("<<TreeviewSelect>>", self.on_series_select)
        # endregion: Series information ---------------------------------------

    # Setup the simulation control panel ++++++++++++++++++++++++++++++++++++++
    def setup_simulation_panel(self, parent):
        """Setup the simulation control panel"""
        sim_frame = ttk.LabelFrame(parent, text="Simulation Control")
        sim_frame.pack(fill=tk.BOTH, pady=(0, 5))

        # region: Output path -------------------------------------------------
        outpath_frame = ttk.Frame(sim_frame)
        outpath_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(outpath_frame, text="Output:").pack(side=tk.LEFT, padx=2)

        # Show output path
        self.output_dir_var = tk.StringVar(
            value=str(self.config["simulation_output_dir"])
        )
        output_path_entry = tk.Entry(
            outpath_frame,
            textvariable=self.output_dir_var,
            font=("Arial", 10),
            state="readonly",
            readonlybackground="white",
        )
        output_path_entry.configure(exportselection=True)
        output_path_entry.pack(side=tk.LEFT, padx=2)

        ttk.Button(
            outpath_frame,
            text="Set",
            command=self.set_output_directory,
        ).pack(side=tk.LEFT, padx=2)

        # endregion: Output path ----------------------------------------------

        # Parameters
        params_frame = ttk.Frame(sim_frame)
        params_frame.pack(fill=tk.X, padx=5, pady=5)

        # TR setting
        ttk.Label(params_frame, text="TR (sec):").grid(
            row=0, column=0, sticky=tk.W
        )
        self.tr_var = tk.StringVar(value="")  # Start empty
        self.tr_spinbox = ttk.Spinbox(
            params_frame,
            from_=0.1,
            to=10.0,
            increment=0.1,
            width=8,
            textvariable=self.tr_var,
            state=tk.DISABLED,  # Start disabled
        )
        self.tr_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5)

        # Auto advance setting
        auto_advance_value = self.config["auto_advance"]
        self.auto_advance_var = tk.BooleanVar(value=auto_advance_value)
        ttk.Checkbutton(
            params_frame, text="Auto-advance", variable=self.auto_advance_var
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5)

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
        ttk.Label(current_frame, text="Series Progress:").pack(
            anchor=tk.W, padx=5
        )
        self.series_progress = ttk.Progressbar(
            current_frame, mode="determinate"
        )
        self.series_progress.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(current_frame, text="Overall Progress:").pack(
            anchor=tk.W, padx=5, pady=(5, 0)
        )
        self.overall_progress = ttk.Progressbar(
            current_frame, mode="determinate"
        )
        self.overall_progress.pack(fill=tk.X, padx=5, pady=2)

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
        )
        self.start_btn.pack(fill=tk.X, pady=2)

        self.pause_btn = ttk.Button(
            main_btn_frame,
            text="Pause",
            command=self.pause_simulation,
            state=tk.DISABLED,
        )
        self.pause_btn.pack(fill=tk.X, pady=2)

        self.stop_btn = ttk.Button(
            main_btn_frame,
            text="Stop",
            command=self.stop_simulation,
            state=tk.DISABLED,
        )
        self.stop_btn.pack(fill=tk.X, pady=2)

        self.next_series_btn = ttk.Button(
            main_btn_frame,
            text="Next Series",
            command=self.next_series,
            state=tk.DISABLED,
        )
        self.next_series_btn.pack(fill=tk.X, pady=2)

    # Setup the physiological data selection panel +++++++++++++++++++++++++++
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
            text="Start Physio",
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

    # Setup the log panel +++++++++++++++++++++++++++++++++++++++++++++++++++++ 
    def setup_log_panel(self, parent):
        """Setup the log panel at the bottom"""
        log_frame = ttk.LabelFrame(parent, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=10, wrap=tk.WORD
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def load_initial_data(self):
        """Load initial data"""
        self.load_dicom_sessions()
        # self.load_physio_files()
        # self.log_initial_port_selections()

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
                all_dicom_files[:3]
            )

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
    def on_session_select(self, event):
        """Handle session selection"""
        selection = self.session_tree.selection()
        if not selection:
            self.current_session_var.set("No session selected")
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

        self.selected_session_path = session_path
        self.current_session_path = session_path  # Keep both in sync

        # Update current session display
        if study_description and study_description != "Unknown":
            session_display = f"{session_name} ({study_description})"
        else:
            session_display = session_name
        self.current_session_var.set(session_display)

        self.log_message(f"Selected session: {session_path}")

        # Load series information in a separate thread
        loading_thread = threading.Thread(
            target=self.load_series_info_threaded,
            args=(session_path,),
            daemon=True,
        )
        loading_thread.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def on_series_select(self, event):
        """Handle series selection and update simulation controls"""
        selection = self.series_tree.selection()
        if not selection:
            # No series selected, disable TR and clear status
            self.tr_spinbox.config(state=tk.DISABLED)
            self.tr_var.set("")
            self.current_series_var.set("No series selected")
            return

        item = self.series_tree.item(selection[0])
        series_name = item["values"][0]  # Series column
        series_description = item["values"][1]  # Description column
        series_files = item["values"][2]  # Files column

        # Update simulation control current status
        series_status = (
            f"{series_name}: {series_description} ({series_files} files)"
        )
        self.current_series_var.set(series_status)

        # Find the corresponding series data to check if functional
        selected_series = None
        for series in self.dicom_series:
            if f"Series {series['number']}" == series_name:
                selected_series = series
                break

        # Check if this is a functional series
        if selected_series:
            image_type = selected_series.get("image_type")
            is_functional_series = self.is_functional_series(
                series_description, image_type
            )
            tr_value = selected_series.get("tr")
        else:
            is_functional_series = self.is_functional_series(
                series_description
            )
            tr_value = None

        if is_functional_series:
            # Enable TR for functional series
            self.tr_spinbox.config(state=tk.NORMAL)

            # Set TR from DICOM if available, otherwise use default
            if tr_value:
                self.tr_var.set(f"{tr_value:.3f}")
                self.log_message(
                    f"Set TR to {tr_value:.3f}s from DICOM metadata"
                )
            elif not self.tr_var.get():  # Set default if empty
                self.tr_var.set(str(self.config["tr_seconds"]))
        else:
            # Disable TR for non-functional series
            self.tr_spinbox.config(state=tk.DISABLED)
            self.tr_var.set("")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def is_functional_series(self, series_description, image_type=None):
        """Check if series is functional based on ImageType.
        Uses same logic as rt_dicom_monitor.py
        """
        # Primary check: DICOM ImageType field
        # (exact same logic as rt_dicom_monitor.py)
        if image_type and "FMRI" in image_type:
            return True

        # Secondary check: Series description keywords (fallback only)
        functional_keywords = ["fmri", "functional", "bold", "ep2d", "func"]

        description_lower = series_description.lower()
        return any(
            keyword in description_lower for keyword in functional_keywords
        )

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
        if self.rescan_btn.config("text")[4] == "Cancel":
            # Cancel current scan
            self.scanning_cancelled = True
            self.log_message("Series scan cancelled by user")
            self.show_series_loading_progress(False)
        else:
            # Start new scan
            self.refresh_series_info()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_cache_file_path(self, session_path):
        """Get the path for the cached series info file"""
        return session_path / ".rtmri_series_cache.pkl"

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save_series_info_cache(self, session_path, series_data):
        """Save series information to cache file"""
        try:
            cache_file = self.get_cache_file_path(session_path)

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
    def load_series_info_cache(self, session_path):
        """Load series information from cache if available and recent"""
        try:
            cache_file = self.get_cache_file_path(session_path)

            if not cache_file.exists():
                self.log_message("No cache file found")
                return None

            # Check if cache is recent (within last 7 days)
            cache_age = time.time() - cache_file.stat().st_mtime
            cache_age_hours = cache_age / 3600

            if cache_age > 604800:  # 7 days
                cache_age_days = cache_age / 86400
                self.log_message(
                    f"Cache is too old ({cache_age_days:.1f} days), "
                    "will rescan"
                )
                return None

            self.log_message(
                f"Found cache file ({cache_age_hours:.1f} hours old), "
                "loading..."
            )

            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)

            # Convert back to our format
            self.dicom_series = []
            for series_cache in cache_data["series"]:
                # Convert file paths back to Path objects
                files = [Path(f) for f in series_cache["files"]]

                # Verify files still exist
                existing_files = [f for f in files if f.exists()]
                if len(existing_files) != len(files):
                    self.log_message(
                        "Some cached files no longer exist, will rescan"
                    )
                    return None

                self.dicom_series.append(
                    {
                        "number": series_cache["number"],
                        "description": series_cache["description"],
                        "files": existing_files,
                        "size": series_cache["size"],
                        "tr": series_cache.get("tr"),
                        "image_type": series_cache.get("image_type"),
                        "is_functional": self.is_functional_series(
                            series_cache["description"],
                            series_cache.get("image_type"),
                        ),
                    }
                )

            # Populate the tree view (will be scheduled on main thread)
            # Tree population moved to _populate_series_tree_from_cache()

            message = f"Loaded {len(self.dicom_series)} series from cache"
            self.log_message(message)
            return True

        except Exception as e:
            self.err_message(f"Error loading cache: {e}")
            return None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _populate_series_tree_from_cache(self):
        """Populate series tree from loaded cache data"""
        for series in self.dicom_series:
            series_data = (
                f"Series {series['number']}",
                series["description"],
                len(series["files"]),
            )
            self.series_tree.insert("", "end", values=series_data)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def refresh_series_info(self):
        """Refresh series information by rescanning"""
        if (
            not hasattr(self, "current_session_path")
            or not self.current_session_path
        ):
            self.err_message("No session selected")
            return

        # Delete cache file to force rescan
        try:
            cache_file = self.get_cache_file_path(self.current_session_path)
            if cache_file.exists():
                cache_file.unlink()
                self.log_message("Cleared series cache")
        except Exception as e:
            self.err_message(f"Error clearing cache: {e}")

        # Start fresh scan
        self.log_message("Rescanning series information...")
        thread = threading.Thread(
            target=self.load_series_info_threaded,
            args=(self.current_session_path,),
        )
        thread.daemon = True
        thread.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def load_series_info_threaded(self, session_path):
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
                return

            # Try to load from cache first
            cached = self.load_series_info_cache(session_path)
            if cached:
                # Schedule tree population on main thread
                self.root.after(0, self._populate_series_tree_from_cache)
                self.root.after(0, self.show_series_loading_progress, False)
                return

            # Check for cancellation before full scan
            if self.scanning_cancelled:
                return

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
                return

            if not dicom_files:
                self.root.after(
                    0, self.log_message, "No DICOM files found in session"
                )
                self.root.after(0, self.show_series_loading_progress, False)
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
            self.load_series_info(
                session_path, progress_callback=self._update_series_progress
            )

            # Save to cache after successful scan (if not cancelled)
            if self.dicom_series and not self.scanning_cancelled:
                self.save_series_info_cache(session_path, self.dicom_series)

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
        self.dicom_series = []

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _update_series_progress(self, current, total, message=""):
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
            series_dict = {}

            reader = DicomReader()

            if progress_callback:
                progress_callback(0, len(dicom_files), "Identifying series...")

            # Strategy: Sample files more strategically to identify all series
            # Take samples from beginning, middle, and end of file list
            sample_size = min(100, len(dicom_files))  # Increased sample size
            sample_indices = set()

            # Add files from beginning
            sample_indices.update(range(min(30, len(dicom_files))))

            # Add files from middle
            mid_point = len(dicom_files) // 2
            sample_indices.update(
                range(
                    max(0, mid_point - 15),
                    min(len(dicom_files), mid_point + 15),
                )
            )

            # Add files from end
            sample_indices.update(
                range(max(0, len(dicom_files) - 30), len(dicom_files))
            )

            # IMPORTANT: Ensure we include files from each series pattern
            # Look for files with different series numbers in filename patterns
            series_pattern_files = {}
            for i, dcm_file in enumerate(dicom_files):
                # Extract series number from filename pattern:
                # 001_SERIES_INSTANCE
                parts = dcm_file.name.split("_")
                if len(parts) >= 3 and parts[1].isdigit():
                    series_from_filename = int(parts[1])
                    if series_from_filename not in series_pattern_files:
                        series_pattern_files[series_from_filename] = i
                        sample_indices.add(i)

            # Add random samples if we haven't reached our target
            remaining_needed = sample_size - len(sample_indices)
            if remaining_needed > 0:
                available_indices = (
                    set(range(len(dicom_files))) - sample_indices
                )
                additional_samples = random.sample(
                    list(available_indices),
                    min(remaining_needed, len(available_indices)),
                )
                sample_indices.update(additional_samples)

            # Sort sample indices for orderly processing
            sample_indices = sorted(list(sample_indices))

            # Scan sampled files to identify series
            for i, file_idx in enumerate(sample_indices):
                # Check for cancellation
                if progress_callback and self.scanning_cancelled:
                    return

                dcm_file = dicom_files[file_idx]
                try:
                    info = reader.read_dicom_info(dcm_file, timeout=0.2)
                    if info:
                        series_num = info.get("SeriesNumber", 1)

                        # Try multiple fields for series description
                        series_desc = (
                            info.get("SeriesDescription")
                            or info.get("Series De")
                            or info.get("Protocol Name")
                            or info.get("Protocol")
                            or info.get("Series Information")
                            or f"Series {series_num}"
                        )

                        if series_num not in series_dict:
                            series_dict[series_num] = {
                                "description": series_desc,
                                "files": [],
                                "size": 0,
                                "tr": None,
                                "image_type": None,
                            }
                            # Extract TR and ImageType if available
                            if "TR" in info:
                                # TR is in milliseconds in DICOM,
                                # convert to seconds
                                tr_ms = info["TR"]
                                tr_seconds = tr_ms / 1000.0
                                series_dict[series_num]["tr"] = tr_seconds
                                tr_msg = f"Series {series_num}: "
                                tr_msg += f"TR = {tr_seconds:.3f}s"
                                self.log_message(tr_msg)
                            if "ImageType" in info:
                                series_dict[series_num]["image_type"] = info[
                                    "ImageType"
                                ]
                                img_type = info['ImageType']
                                img_msg = f"Series {series_num}: "
                                img_msg += f"ImageType = {img_type}"
                                self.log_message(img_msg)
                        # Update description if we found a better one
                        elif series_dict[series_num]["description"].startswith(
                            "Series "
                        ) and not series_desc.startswith("Series "):
                            series_dict[series_num]["description"] = (
                                series_desc
                            )
                            # Also update TR and ImageType if missing
                            if (
                                series_dict[series_num]["tr"] is None
                                and "TR" in info
                            ):
                                # TR is in milliseconds in DICOM,
                                # convert to seconds
                                tr_ms = info["TR"]
                                tr_seconds = tr_ms / 1000.0
                                series_dict[series_num]["tr"] = tr_seconds
                                tr_msg = f"Series {series_num}: "
                                tr_msg += f"TR = {tr_seconds:.3f}s"
                                self.log_message(tr_msg)
                            if (
                                series_dict[series_num]["image_type"] is None
                                and "ImageType" in info
                            ):
                                series_dict[series_num]["image_type"] = info[
                                    "ImageType"
                                ]
                                img_type = info['ImageType']
                                img_msg = f"Series {series_num}: "
                                img_msg += f"ImageType = {img_type}"
                                self.log_message(img_msg)

                except Exception as e:
                    if not progress_callback:
                        self.log_message(f"Error reading {dcm_file}: {e}")
                    continue

                # Update progress for initial scan
                if progress_callback:
                    # During sampling, show progress as fraction of total work
                    # Sampling is roughly 30% of total, file counting is 70%
                    sampling_fraction = 0.3
                    current_sampling_progress = (i + 1) / len(sample_indices)
                    total_progress = int(
                        current_sampling_progress * sampling_fraction
                        * len(dicom_files)
                    )
                    progress_callback(
                        total_progress,
                        len(dicom_files),
                        f"Identifying... ({i + 1}/{len(sample_indices)})",
                    )

            # After sampling phase, start file counting at 30% progress
            sampling_fraction = 0.3
            base_progress = int(sampling_fraction * len(dicom_files))
            
            if progress_callback:
                progress_callback(
                    base_progress,
                    len(dicom_files),
                    "Counting all files...",
                )

            # Now assign ALL files to series based on what we learned
            for i, dcm_file in enumerate(dicom_files):
                # Check for cancellation
                if progress_callback and self.scanning_cancelled:
                    return

                # Try to get series number from this file (quick attempt)
                series_num = None
                try:
                    info = reader.read_dicom_info(dcm_file, timeout=0.05)
                    if info:
                        series_num = info.get("SeriesNumber", 1)
                        # If we get a good description, update it
                        # Try multiple fields for series description
                        series_desc = (
                            info.get("SeriesDescription")
                            or info.get("Protocol Name")
                            or info.get("Protocol")
                            or info.get("Series De")
                            or info.get("Series Information")
                            or f"Series {series_num}"
                        )
                        if (
                            series_num in series_dict
                            and series_dict[series_num][
                                "description"
                            ].startswith("Series ")
                            and not series_desc.startswith("Series ")
                        ):
                            series_dict[series_num]["description"] = (
                                series_desc
                            )
                except Exception:
                    # If quick read fails, try to infer from filename
                    series_num = self._infer_series_from_filename(dcm_file)

                # Add file to the appropriate series
                if series_num in series_dict:
                    series_dict[series_num]["files"].append(dcm_file)
                    series_dict[series_num]["size"] += dcm_file.stat().st_size
                else:
                    # This shouldn't happen often with better sampling
                    series_dict[series_num] = {
                        "description": f"Series {series_num}",
                        "files": [dcm_file],
                        "size": dcm_file.stat().st_size,
                        "tr": None,  # Initialize TR field
                        "image_type": None,  # Initialize ImageType field
                    }

                # Update progress for full file counting
                if progress_callback and (i + 1) % 20 == 0:
                    # Progress = base_progress + current file processing
                    file_counting_fraction = 0.7
                    file_progress = int(
                        (i + 1) / len(dicom_files)
                        * file_counting_fraction * len(dicom_files)
                    )
                    current_progress = base_progress + file_progress
                    progress_callback(
                        current_progress,
                        len(dicom_files),
                        f"Counting files... ({i + 1}/{len(dicom_files)})",
                    )

            # Final progress update
            if progress_callback:
                progress_callback(
                    len(dicom_files), len(dicom_files), "Organizing series..."
                )

            # Convert to list and sort
            self.dicom_series = []
            for series_num, info in sorted(series_dict.items()):
                self.dicom_series.append(
                    {
                        "number": series_num,
                        "description": info["description"],
                        "files": info["files"],
                        "size": info["size"],
                        "tr": info.get("tr"),
                        "image_type": info.get("image_type"),
                        "is_functional": self.is_functional_series(
                            info["description"], info.get("image_type")
                        ),
                    }
                )

                # Add to treeview (schedule on main thread if using callback)
                series_data = (
                    f"Series {series_num}",
                    info["description"],
                    len(info["files"]),
                )

                if progress_callback:
                    # Create wrapper function with proper closure
                    def add_series_item(data=series_data):
                        self.series_tree.insert("", "end", values=data)

                    self.root.after(0, add_series_item)
                else:
                    self.series_tree.insert("", "end", values=series_data)

            message = (
                f"Loaded {len(series_dict)} series "
                f"with {len(dicom_files)} total files"
            )

            if progress_callback:
                self.root.after(0, self.log_message, message)
            else:
                self.log_message(message)

        except Exception as e:
            error_msg = f"Error loading series info: {e}"
            if progress_callback:
                self.root.after(0, self.log_message, error_msg)
            else:
                self.log_message(error_msg)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _infer_series_from_filename(self, dcm_file):
        """Try to infer series number from filename patterns"""
        # Common DICOM filename patterns:
        # IM-0001-0001.dcm, I.001.001.dcm, etc.
        filename = dcm_file.name.upper()

        # Try to extract series number from various patterns

        # Pattern 1: IM-XXXX-YYYY where XXXX might be series
        match = re.search(r"IM-(\d+)-\d+", filename)
        if match:
            return int(match.group(1))

        # Pattern 2: I.XXX.YYY where XXX might be series
        match = re.search(r"I\.(\d+)\.\d+", filename)
        if match:
            return int(match.group(1))

        # Pattern 3: SeriesXXX or Series_XXX
        match = re.search(r"SERIES[_-]?(\d+)", filename)
        if match:
            return int(match.group(1))

        # Pattern 4: Look for any 3-4 digit number that might be series
        matches = re.findall(r"\d{3,4}", filename)
        if matches:
            # Take the first one as series number
            return int(matches[0])

        # Default fallback
        return 1

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_src_dir(self):
        """Set test data root directory and refresh sessions list"""
        directory = filedialog.askdirectory(
            title="Select Test Source Data Directory",
            initialdir=self.config["src_data_dir"],
        )
        if directory:
            # Update test data directory configuration
            self.config["src_data_dir"] = Path(directory)
            self.src_dir_var.set(str(directory))
            self.log_message(
                f"Test source data directory changed to: {directory}")

            # Clear current session selection
            self.selected_session_path = None
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

        self.log_message(
            f"Loaded {len(sessions)} DICOM sessions from {src_data_dir}"
        )

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_output_directory(self):
        """Set the output directory for simulated feeding"""
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
    def start_simulation(self):
        """Start the simulation"""
        if not self.validate_simulation_setup():
            return

        self.simulation_running = True
        self.current_series_index = 0

        # Update UI
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        self.next_series_btn.config(state=tk.NORMAL)

        # Update progress bars
        self.overall_progress.config(maximum=len(self.dicom_series))
        self.overall_progress["value"] = 0

        self.log_message("Starting simulation...")

        # Start simulation thread
        self.simulation_thread = threading.Thread(
            target=self.run_simulation, daemon=True
        )
        self.simulation_thread.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def validate_simulation_setup(self):
        """Validate simulation setup"""
        if not self.selected_session_path:
            messagebox.showerror("Error", "Please select a DICOM session")
            return False

        if not self.dicom_series:
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
    def run_simulation(self):
        """Main simulation loop"""
        try:
            output_base_dir = (
                Path(self.output_dir_var.get())
                / self.selected_session_path.name
            )

            for series_index, series_info in enumerate(self.dicom_series):
                if not self.simulation_running:
                    break

                self.current_series_index = series_index

                # Update current series display
                series_text = (
                    f"Series {series_info['number']}: "
                    f"{series_info['description']} "
                    f"({len(series_info['files'])} files)"
                )
                self.root.after(0, self.current_series_var.set, series_text)

                # Create series output directory
                series_output_dir = (
                    output_base_dir / f"series_{series_info['number']:03d}"
                )
                series_output_dir.mkdir(parents=True, exist_ok=True)

                # Update series progress bar
                self.root.after(
                    0,
                    self.series_progress.config,
                    {"maximum": len(series_info["files"])},
                )
                self.root.after(0, self.series_progress.config, {"value": 0})

                self.root.after(
                    0,
                    self.log_message,
                    f"Starting series {series_info['number']}",
                )

                # Simulate files in series
                try:
                    tr_value = self.tr_var.get()
                    tr_seconds = float(tr_value) if tr_value else 2.0
                except ValueError:
                    tr_seconds = 2.0  # Default fallback

                for file_index, dicom_file in enumerate(series_info["files"]):
                    if not self.simulation_running:
                        break

                    # Copy file
                    dest_file = series_output_dir / dicom_file.name
                    shutil.copy2(dicom_file, dest_file)

                    # Update progress
                    self.root.after(
                        0,
                        self.series_progress.config,
                        {"value": file_index + 1},
                    )

                    # Log progress (every 10 files or last file)
                    if (file_index + 1) % 10 == 0 or file_index == len(
                        series_info["files"]
                    ) - 1:
                        self.root.after(
                            0,
                            self.log_message,
                            (
                                f"  Volume {file_index + 1}/"
                                f"{len(series_info['files'])}: "
                                f"{dicom_file.name}"
                            ),
                        )

                    # Wait for TR
                    if (
                        file_index < len(series_info["files"]) - 1
                    ):  # Don't wait after last file
                        time.sleep(tr_seconds)

                # Update overall progress
                overall_value = {"value": series_index + 1}
                self.root.after(
                    0, self.overall_progress.config, overall_value
                )

                self.root.after(
                    0,
                    self.log_message,
                    f"Completed series {series_info['number']}",
                )

                # If not auto-advance, wait for user input
                if (
                    not self.auto_advance_var.get()
                    and series_index < len(self.dicom_series) - 1
                ):
                    self.root.after(
                        0,
                        self.log_message,
                        "Waiting for next series command...",
                    )
                    while (
                        self.simulation_running
                        and self.current_series_index == series_index
                    ):
                        time.sleep(0.1)

            # Simulation completed
            if self.simulation_running:
                self.root.after(
                    0, self.log_message, "Simulation completed successfully!"
                )
            else:
                self.root.after(
                    0, self.log_message, "Simulation stopped by user"
                )

        except Exception as e:
            self.root.after(0, self.err_message, f"Simulation error: {e}")
        finally:
            self.root.after(0, self.simulation_finished)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def pause_simulation(self):
        """Pause/resume simulation"""
        if self.simulation_running:
            self.simulation_running = False
            self.pause_btn.config(text="Resume")
            self.log_message("Simulation paused")
        else:
            self.simulation_running = True
            self.pause_btn.config(text="Pause")
            self.log_message("Simulation resumed")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def stop_simulation(self):
        """Stop simulation"""
        self.simulation_running = False
        self.log_message("Stopping simulation...")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def next_series(self):
        """Advance to next series"""
        if self.current_series_index < len(self.dicom_series) - 1:
            self.current_series_index += 1
            self.log_message(
                f"Advancing to series {self.current_series_index + 1}"
            )

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def simulation_finished(self):
        """Handle simulation completion"""
        self.simulation_running = False

        # Update UI
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED, text="Pause")
        self.stop_btn.config(state=tk.DISABLED)
        self.next_series_btn.config(state=tk.DISABLED)

        self.status_var.set("Simulation completed")
        self.series_progress.config(value=0)
        self.overall_progress.config(value=0)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ping_physio_rpc(self):
        """Ping the rt_physio.py process to check if it's responsive"""
        response = self.send_rpc_command("ping")
        if response == "pong":
            self.log_message(f"Physio process ping successful: {response}")
            messagebox.showinfo("Physio RPC Status",
                                f"Physio process ping successful: {response}")
        else:
            self.err_message(f"Physio process ping failed: {response}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def send_rpc_command(
        self, command, socket_name="RtpTTLPhysioSocketServer", host=None
    ):
        if self.rpc_port is None:
            port, errmsg = get_port_by_name(socket_name, host=host)
            if port is None:
                self.err_message(f"Error getting port: {errmsg}")
                return None
            self.rpc_port = port

        if host is None:
            host = "localhost"
        address = (host, self.rpc_port)

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
            self.rpc_port = None
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
                (card_file, resp_file, self.physio_freq_var.get())
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
        self.log_text.insert(tk.END, formatted_message)
        self.log_text.see(tk.END)

        # Update status
        self.status_var.set(message)

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

        self.root.destroy()


# %% ==========================================================================
if __name__ == "__main__":
    app = RTMRISimulator()
    app.run()
