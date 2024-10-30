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

import numpy as np
import pandas as pd
import pydicom


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

        self.font = ("TkDefaultFont", 14)

        self._copy_thread = None
        self._serEnd_event = threading.Event()

        self.create_widget()

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

        # --- Create a listbox to display series ---
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
            self.root_win, text="Series list: Doubleclick an item to start",
            font=self.font)
        self.series_label.grid(
            row=row_i, column=0, columnspan=3, padx=10, pady=0, sticky='sw')
        row_i += 1

        self.series_listbox = tk.Listbox(
            self.root_win, width=50, height=15, font=self.font)
        self.series_listbox.grid(
            row=row_i, column=0, columnspan=10, padx=10, pady=5,
            sticky='ew')
        row_i += 1

        # Bind double-click event on the listbox
        self.series_listbox.bind("<Double-1>", self.run_series)

        # --- Create mode select buttons ---
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

        # # --- Create a log field ---
        self.log_display = scrolledtext.ScrolledText(
            self.root_win, height=10, width=50,
            font=self.font)
        self.log_display.grid(
            row=row_i, column=0, columnspan=2, padx=10, pady=5,
            sticky="nsew")
        self.log_display.tag_configure("error", foreground="red")

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
                self.log_display.insert(tk.END, add_line, "error")
            else:
                self.log_display.insert(tk.END, add_line)

        self.log_display.configure(state='disabled')
        self.log_display.yview(tk.END)
        self.log_display.update()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_export_dir(self):
        dir_path = filedialog.askdirectory()
        if dir_path is None:
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
            self.log(f"ERROR: {errmsg}")

            self.imageFilePat_entry.delete(0, tk.END)
            self.imageFilePat_entry.insert(0, self.image_file_pat)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_image_src(self):
        dir_path = filedialog.askdirectory()
        if dir_path is None:
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
        img_folders += [str(Path(pp).relative_to(self.image_src))
                        for pp in self.image_files.Path.unique()]
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

        image_folder = str(
            Path(self.image_src) / self.imageFolder_dropdown.get())
        try:
            assert Path(image_folder).is_dir()
        except Exception:
            self.log(
                f"ERROR:{self.imageFolder_dropdown.get()} is not a directory")
            return

        self.log(f"Select folder {self.imageFolder_dropdown.get()}")

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
        progress_bar['maximum'] = len(file_list)
        for idx, row in file_list.iterrows():
            self.root_win.update_idletasks()
            progress_dialog.update()
            progress_bar['value'] = idx

            if pd.notna(row.Series) and pd.notna(row.Atime):
                continue

            img_file = Path(row.Path) / row.File
            try:
                dcm = pydicom.dcmread(img_file)

                if hasattr(dcm, 'StudyDescription'):
                    self.image_files.loc[idx, 'Study'] = dcm.StudyDescription

                if hasattr(dcm, 'SeriesNumber'):
                    self.image_files.loc[idx, 'Series'] = dcm.SeriesNumber

                if hasattr(dcm, 'SeriesDescription'):
                    self.image_files.loc[idx, 'Desc'] = dcm.SeriesDescription

                if hasattr(dcm, 'StudyDate') and \
                        hasattr(dcm, 'AcquisitionTime'):
                    date = dcm.StudyDate
                    atime = date + dcm.AcquisitionTime
                    at = datetime.strptime(atime, '%Y%m%d%H%M%S.%f')
                    self.image_files.loc[idx, 'Atime'] = \
                        pd.to_datetime(at, errors='coerce')

                elif hasattr(dcm, 'AcquisitionDateTime'):
                    atime = dcm.AcquisitionDateTime
                    date = atime[:8]
                    at = datetime.strptime(atime, '%Y%m%d%H%M%S.%f')
                    self.image_files.loc[idx, 'Atime'] = \
                        pd.to_datetime(at, errors='coerce')
                else:
                    date = None

                if hasattr(dcm, 'ContentTime') and date is not None:
                    ctime = date + dcm.ContentTime
                    ct = datetime.strptime(ctime, '%Y%m%d%H%M%S.%f')
                    self.image_files.loc[idx, 'Ctime'] = \
                        pd.to_datetime(ct, errors='coerce')

            except Exception:
                continue

        # --- Get series list ---
        file_list = self.image_files[self.image_files.Path == image_folder]
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
            ser_label = f'Ser {ser_nums.astype(int)} (N={len(sel_idx)})'
            for sn in ser_nums:
                ser_label += f"; {self.file_list.Desc[sel_idx[0]]}"
            series_list.append(ser_label)
            sstart = send

        # Refresh series_listbox
        self.series_listbox.config(height=len(series_list))
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
            while self._copy_thread.is_alive():
                time.sleep(0.1)

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
                # Wait for a button press

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
        self._serEnd_event.set()
        # if self._copy_thread is not None:
        #     while self._copy_thread.is_alive():
        #         self.root_win.update()
        #         time.sleep(0.1)
        #     self._copy_thread = None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run(self):
        self.root_win.mainloop()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def quit_application(self, event=None):
        self.root_win.quit()

# def on_item_double_click(event):
#     # Get the selected item from the listbox
#     selected_item_index = listbox.curselection()

#     if selected_item_index:
#         selected_item = listbox.get(selected_item_index)

#         # Show a confirmation dialog
#         response = messagebox.askyesno("Confirmation", f"Do you want to start a process with '{selected_item}' in mode {mode.get()}?")

#         if response:
#             # Start the process in a new thread to simulate a long-running task
#             global cancel_process
#             cancel_process = False
#             threading.Thread(target=start_process, args=(selected_item, mode.get())).start()
#         else:
#             messagebox.showinfo("Cancelled", "Process not started.")

# def start_process(item, selected_mode):
#     global cancel_process
#     # Example long-running process simulation
#     for i in range(10):
#         if cancel_process:
#             messagebox.showinfo("Process Cancelled", f"Process for '{item}' in {selected_mode} was cancelled.")
#             return
#         print(f"Processing '{item}' in {selected_mode}... Step {i+1}/10")
#         time.sleep(1)  # Simulate a time-consuming task with sleep
#     messagebox.showinfo("Process Completed", f"Process completed for '{item}' in {selected_mode}")


# %% __main__ =================================================================
if __name__ == '__main__':
    rtmei_sim = RTMRISimulator()
    rtmei_sim.run()
