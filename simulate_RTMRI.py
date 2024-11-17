#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% import ===================================================================
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import threading
import time
import re
from pathlib import Path
from datetime import datetime
import platform
import shutil
import json

import numpy as np
import pandas as pd
import pydicom

from rtpspy.rtp_ttl_physio import call_rt_physio


# %% RTMRISimulator ===========================================================
class RTMRISimulator():

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self):
        # Create the main application window
        self.root_win = tk.Tk()
        self.root_win.title("RTMRI simulation")
        if platform.system() == 'Darwin':
            # Bind Cmd+Q for macOS
            self.root_win.bind('<Command-q>', self.quit_application)
        else:
            # Bind Ctrl+Q for other operating systems
            self.root_win.bind('<Control-q>', self.quit_application)

        self.export_dir = None
        self.image_src = None
        self.image_file_pat = r".+\.dcm"
        self.file_list = None
        self.run_mode = tk.StringVar(value="Series")
        self.card_file = None
        self.resp_file = None

        self.font = ("TkDefaultFont", 14)

        self._copy_thread = None
        self._serEnd_event = threading.Event()

        self.create_widget()
        self.load_properties()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def create_widget(self):
        row_i = 0

        # --- RTMRI export diectory ---
        RTMRIexport_button = tk.Button(
            self.root_win, text="Set export path", font=self.font,
            command=self.set_export_dir
            )
        RTMRIexport_button.grid(row=row_i, column=0, padx=10, pady=5)

        self.RTMRIexport_entry = tk.Entry(
            self.root_win, width=30, font=self.font
            )
        self.RTMRIexport_entry.grid(row=row_i, column=1, padx=10, pady=5,
                                    sticky='ew')
        self.RTMRIexport_entry.config(state="readonly")
        row_i += 1

        # --- Image source ---
        imageFilePat_label = tk.Label(
            self.root_win, text="File pattern:", font=self.font
            )
        imageFilePat_label.grid(row=row_i, column=0, padx=10, pady=5)

        self.imageFilePat_entry = tk.Entry(
            self.root_win, width=30, font=self.font
            )
        self.imageFilePat_entry.insert(0, self.image_file_pat)
        self.imageFilePat_entry.bind(
            "<Return>", self.check_imageFilePat)
        self.imageFilePat_entry.grid(
            row=row_i, column=1, padx=10, pady=5, sticky='ew')
        row_i += 1

        imageSrc_button = tk.Button(
            self.root_win, text="Set image source", font=self.font,
            command=self.set_image_src
            )
        imageSrc_button.grid(row=row_i, column=0, padx=10, pady=5)

        self.imageSrc_entry = tk.Entry(
            self.root_win, width=30, font=self.font
            )
        self.imageSrc_entry.grid(row=row_i, column=1, padx=10, pady=5,
                                 sticky='ew')
        self.imageSrc_entry.config(state="readonly")
        row_i += 1

        # --- Listbox to display series ---
        # Create a dropdown list of image folders
        imageFolder_label = tk.Label(
            self.root_win, text="Select data folder", font=self.font)
        imageFolder_label.grid(
            row=row_i, column=0, padx=10, pady=5, sticky='e')

        self.imageFolder_dropdown = ttk.Combobox(
            self.root_win, values=[], width=30, font=self.font,
            )
        self.imageFolder_dropdown.bind(
            "<<ComboboxSelected>>", self.show_series_list
            )
        self.imageFolder_dropdown.grid(
            row=row_i, column=1, padx=10, pady=5, sticky='ew')
        row_i += 1

        self.series_label = tk.Label(
            self.root_win, text="Series list: Doubleclick a series to start",
            font=self.font)
        self.series_label.grid(
            row=row_i, column=0, columnspan=3, padx=10, pady=0, sticky='sw')
        row_i += 1

        self.series_listbox = tk.Listbox(
            self.root_win, width=50, height=10, font=self.font)
        self.series_listbox.grid(
            row=row_i, column=0, columnspan=10, padx=10, pady=5,
            sticky='ew')
        row_i += 1

        # Bind double-click event on the listbox
        self.series_listbox.bind("<Double-1>", self.run_series)

        # --- Mode select buttons ---
        # Create radio buttons to select a mode
        runMode_frame = tk.Frame(self.root_win)
        runMode_frame.grid(
            row=row_i, column=0, columnspan=2, padx=10, pady=5, sticky='e')
        row_i += 1

        runMode_label = tk.Label(
            runMode_frame, text="Run Mode:", font=self.font)
        runMode_label.grid(row=0, column=0, padx=10, pady=5)

        self.seriesMode_radio = tk.Radiobutton(
            runMode_frame, text="Series", variable=self.run_mode,
            font=self.font, value="Series"
            )
        self.seriesMode_radio.grid(
            row=0, column=1, padx=10, pady=5, sticky="ew")

        self.seriesMode_radio = tk.Radiobutton(
            runMode_frame, text="Session", variable=self.run_mode,
            font=self.font, value="Session"
            )
        self.seriesMode_radio.grid(
            row=0, column=2, padx=10, pady=5, sticky="ew")

        self.seriesMode_radio = tk.Radiobutton(
            runMode_frame, text="File", variable=self.run_mode,
            font=self.font, value="File"
            )
        self.seriesMode_radio.grid(
            row=0, column=3, padx=10, pady=5, sticky="ew")

        # Create a button to cancel the ongoing process
        self.stop_button = tk.Button(
            runMode_frame, text="Stop", font=self.font,
            command=self.cancel_ongoing_process)
        self.stop_button.grid(row=0, column=4, padx=10, pady=5, sticky="ew")

        # --- Physio files ---
        physio_separator = ttk.Separator(self.root_win, orient="horizontal")
        physio_separator.grid(row=row_i, column=0, columnspan=2, sticky="ew",
                              pady=10)
        row_i += 1

        card_button = tk.Button(
            self.root_win, text="Set Cardiac file", font=self.font,
            command=self.set_card_file
            )
        card_button.grid(row=row_i, column=0, padx=10, pady=5)

        self.card_entry = tk.Entry(
            self.root_win, width=30, font=self.font
            )
        self.card_entry.grid(row=row_i, column=1, padx=10, pady=5,
                             sticky='ew')
        self.card_entry.config(state="readonly")
        row_i += 1

        resp_button = tk.Button(
            self.root_win, text="Set Respiration file", font=self.font,
            command=self.set_resp_file
            )
        resp_button.grid(row=row_i, column=0, padx=10, pady=5)

        self.resp_entry = tk.Entry(
            self.root_win, width=30, font=self.font
            )
        self.resp_entry.grid(row=row_i, column=1, padx=10, pady=5,
                             sticky='ew')
        self.resp_entry.config(state="readonly")
        row_i += 1

        physioCtrl_frame = tk.Frame(self.root_win)
        physioCtrl_frame.grid(
            row=row_i, column=0, columnspan=2, padx=10, pady=5, sticky='e')
        row_i += 1

        physioStart_button = tk.Button(
            physioCtrl_frame, text="Start physio", font=self.font,
            command=self.start_physio
            )
        physioStart_button.grid(
            row=0, column=0, padx=10, pady=5, sticky="ew")

        physioStop_button = tk.Button(
            physioCtrl_frame, text="Stop physio", font=self.font,
            command=self.stop_physio
            )
        physioStop_button.grid(
            row=0, column=2, padx=10, pady=5, sticky="ew")

        physio_separator2 = ttk.Separator(self.root_win, orient="horizontal")
        physio_separator2.grid(row=row_i, column=0, columnspan=2, sticky="ew",
                               pady=10)
        row_i += 1

        # --- Log field ---
        self.log_display = scrolledtext.ScrolledText(
            self.root_win, height=10, width=50,
            font=self.font)
        self.log_display.grid(
            row=row_i, column=0, columnspan=2, padx=10, pady=5,
            sticky="nsew")
        self.log_display.tag_configure("ERROR", foreground="red")

        self.root_win.grid_rowconfigure(row_i, weight=1)
        self.root_win.grid_columnconfigure(1, weight=1)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def log(self, message):
        self.log_display.configure(state='normal')
        log_lines = message.split('\n')
        for ii, log_line in enumerate(log_lines):
            add_line = log_line + '\n'
            if ii == 0:
                add_line = \
                    f"[{datetime.isoformat(datetime.now())}]:" + add_line

            if 'ERROR' in log_line:
                self.log_display.insert(tk.END, add_line, "ERROR")
            else:
                self.log_display.insert(tk.END, add_line)

        self.log_display.configure(state='disabled')
        self.log_display.yview(tk.END)
        self.log_display.update()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_export_dir(self, dir_path=None):
        if dir_path is None:
            dir_path = filedialog.askdirectory(
                title="Select RTMRI export directory"
            )
            if dir_path is None:
                return
        else:
            if not Path(dir_path).is_dir():
                self.log(f"[ERROR] Not found {dir_path}")
                return

        self.RTMRIexport_entry.config(state="normal")
        self.RTMRIexport_entry.delete(0, tk.END)
        self.RTMRIexport_entry.insert(0, dir_path)
        self.RTMRIexport_entry.config(state="readonly")
        self.export_dir = dir_path
        self.log(f"Set export directory {self.export_dir}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def check_imageFilePat(self, event):
        try:
            file_pat = self.imageFilePat_entry.get()
            re.compile(file_pat)
            self.image_file_pat = file_pat
        except Exception:
            errmsg = "Error in the file pattern"
            errmsg += f"{file_pat} is not a valid regular expression."
            messagebox.showerror(errmsg)
            self.log(f"[ERROR]{errmsg}")

            self.imageFilePat_entry.delete(0, tk.END)
            self.imageFilePat_entry.insert(0, self.image_file_pat)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_image_src(self, dir_path=None):
        if dir_path is None:
            dir_path = filedialog.askdirectory(
                title="Select DICOM source directory",
            )
            if dir_path is None:
                return
        else:
            if not Path(dir_path).is_dir():
                self.log(f"[ERROR]Not found {dir_path}")
                return

        # Set imageSrc_entry
        self.imageSrc_entry.config(state="normal")
        self.imageSrc_entry.delete(0, tk.END)
        self.imageSrc_entry.insert(0, dir_path)
        self.image_src = Path(dir_path)
        self.imageSrc_entry.config(state="readonly")
        self.log(f"Set image source root {self.image_src}")

        # --- Open progress bar dialog ---
        # Create a new Toplevel window for the progress dialog
        progress_dialog = tk.Toplevel(self.root_win)
        progress_dialog.title("Reading directories")
        self.log("Reading directories ...")

        # Make the dialog modal
        # Set to be on top of the main window
        progress_dialog.transient(self.root_win)
        progress_dialog.grab_set()  # Block interactions with other windows

        # Create a Progressbar widget inside the progress dialog
        progress_bar = ttk.Progressbar(
            progress_dialog, orient=tk.HORIZONTAL, length=300,
            mode='indeterminate')
        progress_bar.grid(row=0, column=0, padx=20, pady=20)

        progress_text = tk.Label(
            progress_dialog, text="Searching ...", font=self.font
            )
        progress_text.grid(row=1, column=0, padx=20, pady=20)

        # Find files
        progress_bar.start(10)
        self.image_files = self._find_files(
            self.image_src, progress_dialog=progress_dialog,
            progress_text=progress_text)
        progress_bar.stop()

        # Format dataframe
        self.image_files['Path'] = self.image_files['Path'].astype(str)
        self.image_files['File'] = self.image_files['File'].astype(str)
        self.image_files['Study'] = self.image_files['Study'].astype(str)
        self.image_files['Series'] = self.image_files['Series'].astype(float)
        self.image_files['Desc'] = self.image_files['Desc'].astype(str)
        self.image_files['Atime'] = \
            self.image_files['Atime'].astype('datetime64[ns]')
        self.image_files['Ctime'] = \
            self.image_files['Ctime'].astype('datetime64[ns]')
        self.image_files['Ftime'] = self.image_files['Ftime'].astype(float)

        # Set imageFolder_dropdown
        img_folders = ['Select image folder ...']
        for pp in self.image_files.Path.unique():
            folder_path = str(Path(pp).relative_to(self.image_src))
            if folder_path == '.':
                folder_path += f'[{Path(pp).name}]'
            img_folders.append(folder_path)
        self.imageFolder_dropdown['values'] = img_folders
        self.imageFolder_dropdown.set(img_folders[0])
        self.log("Found directories\n" + '\n'.join(img_folders[1:]))

        progress_dialog.destroy()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _find_files(self, root_dir, image_files=None, progress_dialog=None,
                    progress_text=None):
        image_file_pat = self.image_file_pat

        if image_files is None:
            # Initialize the list
            image_files = pd.DataFrame(
                columns=('Path', 'File', 'Study', 'Series', 'Desc', 'Atime',
                         'Ctime', 'Ftime')
                )

        for fp in root_dir.glob('*'):
            if progress_dialog is not None:
                progress_dialog.update()

            if fp.is_dir():
                image_files = self._find_files(
                    fp, image_files=image_files,
                    progress_dialog=progress_dialog,
                    progress_text=progress_text)
            else:
                if progress_text is not None:
                    progress_text.config(text=fp.parent.name)
                if re.search(image_file_pat, fp.name) is None:
                    continue

                add_row = pd.Series(
                    index=image_files.columns).astype(object)
                add_row['Path'] = str(root_dir)
                add_row['File'] = fp.name
                add_row['Ftime'] = fp.stat().st_ctime
                add_row = pd.DataFrame([add_row])
                if len(image_files) == 0:
                    image_files = add_row
                else:
                    image_files = pd.concat(
                        [image_files, add_row], axis=0, ignore_index=True)

        image_files.reset_index(drop=True, inplace=True)
        image_files = image_files.sort_values(['Path', 'File'])

        return image_files

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def show_series_list(self, event):
        folder_path = self.imageFolder_dropdown.get().split('[')[0]
        image_folder = str(
            Path(self.image_src) / folder_path)
        try:
            assert Path(image_folder).is_dir()
        except Exception:
            self.log(
                f"[ERROR]{self.imageFolder_dropdown.get()} is not a directory")
            return

        self.log(f"Select folder {image_folder}")

        # --- Open progress bar dialog ---
        # Create a new Toplevel window for the progress dialog
        progress_dialog = tk.Toplevel(self.root_win)
        progress_dialog.title("Reading files")

        # Make the dialog modal
        # Set to be on top of the main window
        progress_dialog.transient(self.root_win)
        progress_dialog.grab_set()  # Block interactions with other windows

        # Create a Progressbar widget inside the progress dialog
        progress_bar = ttk.Progressbar(
            progress_dialog, orient=tk.HORIZONTAL, length=300,
            mode='determinate')
        progress_bar.grid(row=0, column=0, padx=20, pady=20)

        # Center the progress dialog in the screen
        self.root_win.update_idletasks()
        progress_dialog.geometry(
            f"+{self.root_win.winfo_x() + 100}+{self.root_win.winfo_y() + 100}"
            )

        # --- Get file info ---
        file_list = self.image_files[self.image_files.Path == image_folder]
        file_list = file_list.sort_values('File')

        progress_bar['maximum'] = len(file_list)
        excld_idx = []
        for ii, (idx, row) in enumerate(file_list.iterrows()):
            self.root_win.update_idletasks()
            progress_dialog.update()
            progress_bar['value'] = ii

            if pd.notna(row.Series) and pd.notna(row.Atime):
                continue

            img_file = Path(row.Path) / row.File
            try:
                dcm = pydicom.dcmread(img_file)

                if hasattr(dcm, 'StudyDescription'):
                    file_list.loc[idx, 'Study'] = dcm.StudyDescription
                    self.image_files.loc[idx, 'Study'] = dcm.StudyDescription

                if hasattr(dcm, 'SeriesNumber'):
                    file_list.loc[idx, 'Series'] = dcm.SeriesNumber
                    self.image_files.loc[idx, 'Series'] = dcm.SeriesNumber

                if hasattr(dcm, 'SeriesDescription'):
                    file_list.loc[idx, 'Desc'] = dcm.SeriesDescription
                    self.image_files.loc[idx, 'Desc'] = dcm.SeriesDescription

                if hasattr(dcm, 'StudyDate') and \
                        hasattr(dcm, 'AcquisitionTime'):
                    date = dcm.StudyDate
                    atime = date + dcm.AcquisitionTime
                    at = datetime.strptime(atime, '%Y%m%d%H%M%S.%f')
                    file_list.loc[idx, 'Atime'] = \
                        pd.to_datetime(at, errors='coerce')
                    self.image_files.loc[idx, 'Atime'] = \
                        pd.to_datetime(at, errors='coerce')

                elif hasattr(dcm, 'AcquisitionDateTime'):
                    atime = dcm.AcquisitionDateTime
                    date = atime[:8]
                    at = datetime.strptime(atime, '%Y%m%d%H%M%S.%f')
                    file_list.loc[idx, 'Atime'] = \
                        pd.to_datetime(at, errors='coerce')
                    self.image_files.loc[idx, 'Atime'] = \
                        pd.to_datetime(at, errors='coerce')
                else:
                    date = None

                if hasattr(dcm, 'ContentTime') and date is not None:
                    ctime = date + dcm.ContentTime
                    ct = datetime.strptime(ctime, '%Y%m%d%H%M%S.%f')
                    file_list.loc[idx, 'Ctime'] = \
                        pd.to_datetime(ct, errors='coerce')
                    self.image_files.loc[idx, 'Ctime'] = \
                        pd.to_datetime(ct, errors='coerce')

            except Exception:
                self.log(f"[ERROR]Could not read {img_file} as dicom.")
                excld_idx.append(idx)
                continue

        # --- Get series list ---
        file_list = file_list.drop(excld_idx)
        file_list = file_list[pd.notna(file_list.Atime)]
        self.file_list = file_list.sort_values('Atime').reset_index(drop=True)

        # Find series divisions
        ser_div = np.argwhere(np.diff(self.file_list.Series) > 0).ravel()
        td = [d / np.timedelta64(1, 's')
              for d in np.diff(self.file_list.Atime)]
        ser_dt = np.array(td)[ser_div]
        ser_dvi = ser_div[ser_dt > 3] + 1
        ser_dvi = np.concatenate([ser_dvi, [len(self.file_list)]])
        if len(ser_dvi) == 0:
            ser_dvi = [len(self.file_list)]

        series_list = []
        sstart = 0
        ri = 0
        self.file_list['Run_idx'] = -1
        for ri, send in enumerate(ser_dvi):
            sel_idx = np.arange(sstart, send, dtype=int)
            self.file_list.loc[sel_idx, 'Run_idx'] = ri
            ser_nums = np.unique(self.file_list.iloc[sel_idx, :].Series)
            if np.any(pd.notna(ser_nums)):
                ser_nums = ser_nums[pd.notna(ser_nums)]
            ser_label = f'Ser {ser_nums.astype(int)} (N={len(sel_idx)})'
            for sn in ser_nums:
                ser_label += f"; {self.file_list.Desc[sel_idx[0]]}"
            series_list.append(ser_label)
            sstart = send

        # Refresh series_listbox
        self.series_listbox.delete(0, tk.END)
        for item in series_list:
            self.series_listbox.insert(tk.END, item)

        self.log(
            f'{self.imageFolder_dropdown.get()} includes' +
            f' {len(series_list)} series runs.')

        progress_dialog.destroy()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run_series(self, event):
        if self.export_dir is None:
            messagebox.showerror("Error", "Export path is not set!")
            return
        try:
            if not Path(self.export_dir).is_dir():
                messagebox.showerror(
                    "Error", f"Export path {self.export_dir} does not exist!")
                return
        except Exception:
            messagebox.showerror(
                    "Error", f"Export path {self.export_dir} does not exist!")
            return

        sel_row = self.series_listbox.curselection()[0]
        # item_content = self.series_listbox.get(sel_row)
        run_files = self.file_list[self.file_list['Run_idx'] == sel_row]
        run_mode = self.run_mode.get()

        dst_dir = Path(self.export_dir) / Path(run_files.Path.unique()[0]).name

        # Clear existing files
        for ff in run_files.File:
            dst_f = dst_dir / ff
            if dst_f.is_file():
                dst_f.unlink()

        # Confirm another copy_thread is not running
        if self._copy_thread is not None:
            self._serEnd_event.set()
            self._copy_thread.join(timeout=3)

        # Start copy file thread
        self._copy_thread = threading.Thread(
            target=self._copy_files, args=(dst_dir, run_files, run_mode))
        self._serEnd_event.clear()
        self._copy_thread.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _copy_files(self, dst_dir, run_files, run_mode):
        sel_row = self.series_listbox.curselection()[0]
        item_content = self.series_listbox.get(sel_row)
        self.log(f'Start {item_content}')

        # Prepare the destination directory
        if not dst_dir.is_dir():
            os.makedirs(dst_dir)

        copy_files = run_files.reset_index(drop=True)
        if run_mode in ('Series', 'Session'):
            cp_times = copy_files.Atime.values
            cp_times = [tt / np.timedelta64(1, 's')
                        for tt in (cp_times - cp_times[0])]
            if len(cp_times) > 1:
                TR = np.diff(cp_times).mean()
                cp_times = cp_times + [cp_times[-1] + TR]
            else:
                cp_times = cp_times + [1]
            ser_start_t = time.time()

        for idx, row in copy_files.iterrows():
            if self._serEnd_event.is_set():
                self.log('Cancel series')
                break

            src_f = Path(row.Path) / row.File
            dst_f = dst_dir / row.File
            cp_time = cp_times[idx+1]
            if run_mode in ('Series', 'Session'):
                while time.time()-ser_start_t < cp_time:
                    time.sleep(0.001)
            elif run_mode == 'File':
                pass
                # TODO: Wait for a button press

            shutil.copy(src_f, dst_f)
            self.log(f"Copy {row.File}")
            self.root_win.update_idletasks()

        # End series
        run_row = copy_files.Run_idx.unique()[0]
        self.series_listbox.select_set(run_row)
        self.series_listbox.activate(run_row)

        self.log(f'End {item_content}')
        self._serEnd_event.set()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def cancel_ongoing_process(self):
        if not self._serEnd_event.is_set():
            self._serEnd_event.set()
            if self._copy_thread is not None:
                self._copy_thread.join(timeout=3)
            self._copy_thread = None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run(self):
        self.root_win.mainloop()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_card_file(self, card_file=None):
        if card_file is None:
            card_file = filedialog.askopenfilename(
                title="Select a caridiac recording file",
            )
            if card_file is None:
                return
        elif card_file == 'None':
            return
        else:
            if not Path(card_file).is_file():
                self.log(f"[ERROR] Not found {card_file}")
                return

        self.card_entry.config(state="normal")
        self.card_entry.delete(0, tk.END)
        self.card_entry.insert(0, card_file)
        self.card_entry.config(state="readonly")
        self.card_file = card_file
        self.log(f"Set cardiac file {self.card_file}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_resp_file(self, resp_file=None):
        if resp_file is None:
            resp_file = filedialog.askopenfilename(
                title="Select a respiration recording file",
            )
            if resp_file is None:
                return
        elif resp_file == 'None':
            return
        else:
            if not Path(resp_file).is_file():
                self.log(f"[ERROR] Not found {resp_file}")
                return

        self.resp_entry.config(state="normal")
        self.resp_entry.delete(0, tk.END)
        self.resp_entry.insert(0, resp_file)
        self.resp_entry.config(state="readonly")
        self.resp_file = resp_file
        self.log(f"Set cardiac file {self.resp_file}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def start_physio(self):
        if not call_rt_physio('ping'):
            self.log('[ERROR]Cannot connect to RtpTTLPhysio')
            return

        call_rt_physio(('SET_REC_DEV', 'None'))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def stop_physio(self):
        pass

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def quit_application(self, event=None):
        self.root_win.quit()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save_properties(self):
        # Save configurations
        properties = {}
        for kk in ('export_dir', 'image_src', 'image_file_pat', 'run_mode',
                   'card_file', 'resp_file'):
            val = getattr(self, kk)
            if hasattr(val, 'get'):
                val = val.get()
            val = str(val)
            properties[kk] = val

        prop_file = Path.home() / '.RTPSpy' / 'simulate_RTMRI.json'
        if not prop_file.parent.is_dir():
            prop_file.parent.mkdir()
        with open(prop_file, 'w') as fid:
            json.dump(properties, fid, indent=4)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def load_properties(self):
        prop_file = Path.home() / '.RTPSpy' / 'simulate_RTMRI.json'
        if not prop_file.is_file():
            return

        try:
            # Load configurations
            with open(prop_file, 'r') as fid:
                properties = json.load(fid)

            for kk, val in properties.items():
                if val is None:
                    continue
                if kk == 'export_dir':
                    self.set_export_dir(val)

                elif kk == 'image_src':
                    self.set_image_src(val)

                elif kk == 'card_file':
                    self.set_card_file(val)

                elif kk == 'resp_file':
                    self.set_resp_file(val)

        except Exception:
            return


# %% __main__ =================================================================
if __name__ == '__main__':
    rtmei_sim = RTMRISimulator()
    rtmei_sim.run()
    rtmei_sim.save_properties()
