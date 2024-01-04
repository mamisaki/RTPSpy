#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time physiological signal recording application.

Model and controller classes:
    RingBuffer:
        Ring buffer using share memory.
    GEPhysioRecording:
        Recording cardiogram and respiration signals from a GE scanner's
        serial port.
    RtPhysioRecoder:
        Interface for recording cardiogram and respiration signals using
        GEPhysioRecording as the backend.
    NumatoGPIORecoding
        REcording TTL signal from Numato Lab 8 Channel USB GPIO (
        https://numato.com/product/8-channel-usb-gpio-module-with-analog-inputs/
        ).
    RtTTLRecoder:
        Interface for recording TTL signal using NumatoGPIORecoding
        as the backend.

View classes:
    RtViewPhysio:
        Real time display of cardiogram and respiration signals
    RtViewTTL:
        Real time display of the TTL signal
    ConfigView:
        Parameter configuration panel
Controller class:
    RTPhysioController:
        Model-View controller

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import time
import sys
import traceback
from multiprocessing import Lock, Pipe, Process, shared_memory
import re
import tkinter as tk
from tkinter import ttk
from datetime import datetime, timedelta
import logging
import argparse
import warnings
import socket

import numpy as np
import serial
from serial.tools.list_ports import comports
from scipy import interpolate

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib as mpl

try:
    from .rpc_socket_server import (
        RPCSocketServer, rpc_send_data, rpc_recv_data, pack_data)
except Exception:
    from rtpspy.rpc_socket_server import (
        RPCSocketServer, rpc_send_data, rpc_recv_data, pack_data)

mpl.rcParams['font.size'] = 8


# %% call_rt_physio ===========================================================
def call_rt_physio(rtp_physio_address, data, pkl=False, get_return=False,
                   logger=None):
    """
    Parameters:
        data:
            Sending data
        pkl bool, optional):
            To pack the data in pickle. Defaults to False.
        get_return (bool, optional):
            Falg to receive a return. Defaults to False.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.connect(rtp_physio_address)
    except ConnectionRefusedError:
        time.sleep(1)
        if data == 'ping':
            return False
        return

    if data == 'ping':
        return True

    if not rpc_send_data(sock, data, pkl=pkl):
        errmsg = f'Failed to send {data}'
        if logger:
            logger.error(errmsg)
        else:
            sys.stderr.write(errmsg)
        return

    if get_return:
        data = rpc_recv_data(sock)
        if data is None:
            errmsg = f'Failed to recieve response to {data}'
            if logger:
                logger.error(errmsg)
            else:
                sys.stderr.write(errmsg)

        return data


# %% RingBuffer ===============================================================
class RingBuffer:
    """ Shared memory array ring buffer """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, shm_name, max_size, dtype=float, initialize=np.nan):
        """_summary_
        Args:
            max_size (int): buffer size (number of elements)
        """
        self.shm_name = shm_name
        self.max_size = int(max_size)
        self.dtype = dtype
        self._cpos = 0

        if not self._init_shared_memory(initialize):
            return None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _init_shared_memory(self, initialize):
        # Create
        size = self.max_size * np.dtype(self.dtype).itemsize
        try:
            shm = shared_memory.SharedMemory(
                name=self.shm_name, create=True, size=size)
        except FileExistsError:
            shm = shared_memory.SharedMemory(name=self.shm_name)
            shm.close()
            shm.unlink()
            shm = shared_memory.SharedMemory(
                name=self.shm_name, create=True, size=size)

        data = np.ndarray(self.max_size, dtype=self.dtype, buffer=shm.buf)
        data[:] = initialize

        shm.close()

        # self._data = np.ones(self.max_size, dtype=self.dtype) * initialize

        return True

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def append(self, x):
        """ Append an element """
        try:
            shm = shared_memory.SharedMemory(name=self.shm_name)
            data = np.ndarray(self.max_size, dtype=float, buffer=shm.buf)
            data[self._cpos] = x
            shm.close()
            # self._data[self._cpos] = x
            self._cpos = (self._cpos+1) % self.max_size

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errstr = ''.join(
                traceback.format_exception(exc_type, exc_obj, exc_tb))
            sys.stderr.write(errstr)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get(self):
        """ return list of elements in correct order """
        try:
            shm = shared_memory.SharedMemory(name=self.shm_name)
            data = np.ndarray(
                self.max_size, dtype=float, buffer=shm.buf).copy()
            shm.close()
            # data = self._data
            return np.concatenate([data[self._cpos:], data[:self._cpos]])

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errstr = ''.join(
                traceback.format_exception(exc_type, exc_obj, exc_tb))
            sys.stderr.write(errstr)
            return None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        try:
            shm = shared_memory.SharedMemory(name=self.shm_name)
            shm.close()
            shm.unlink()
            pass
        except FileNotFoundError:
            pass


# %% TTLPhysioPlot ============================================================
class TTLPhysioPlot():
    """ View class for dispaying real-time TTL and physio recordings
    """
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, recorder, plt_root, as_root=False,
                 geometry='600x600+1024+0', plot_len_sec=10,
                 disable_close=False):
        self.logger = logging.getLogger('TTLPhysioPlot')

        self._plt_root = plt_root
        if as_root:
            self._plt_win = plt_root
        else:
            self._plt_win = tk.Toplevel(self._plt_root)
        self._plt_win.title('Physio signals')
        self.set_position(geometry)

        self.plot_len_sec = plot_len_sec
        self.recorder = recorder
        self._disable_close = disable_close

        # Set the margins in inches
        self.left_margin_inch = 0.3
        self.right_margin_inch = 0.4
        self.top_margin_inch = 0.1
        self.bottom_margin_inch = 0.38

        # initialize plot
        plot_widget = self.init_plot()
        plot_widget.pack(side=tk.TOP, fill='both', expand=True)
        self.set_plot()

        # config button
        self.config_button = tk.Button(self._plt_win, text='config',
                                       command=self.config,
                                       font=("Arial", 10))
        self.config_button.pack(side=tk.LEFT, anchor=tk.SE)

        # dump button
        dump_dur = str(
            timedelta(seconds=self.recorder.buf_len_sec)).split('.')[0]
        dump_button = tk.Button(self._plt_win, text=f"dump ({dump_dur})",
                                command=self.dump, font=("Arial", 10))
        dump_button.pack(side=tk.RIGHT, anchor=tk.SW)

        self._resize_debounce_id = None
        self._plt_win.bind('<Configure>', self.update_plot_size)
        self.update_plot_size(None)

        # Connect WM_DELETE_WINDOW event to self.on_closing
        self._plt_win.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.config_win = None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_position(self, geometry):
        self._plt_win.geometry(geometry)
        # RuntimeError: main thread is not in main loop

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def init_plot(self):
        self.plot_fig = Figure(figsize=(6, 4.2))
        self.canvas = FigureCanvasTkAgg(self.plot_fig, master=self._plt_win)
        self._ax_ttl, self._ax_card, self._ax_resp, = \
            self.plot_fig.subplots(3, 1)

        self.plot_fig.subplots_adjust(
            left=0.15, bottom=0.1, right=0.98, top=0.95, hspace=0.35)

        return self.canvas.get_tk_widget()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_plot(self):
        signal_freq = self.recorder.sample_freq
        buf_size = int(np.round(self.plot_len_sec * signal_freq))
        sig_xi = np.arange(buf_size) * 1.0/signal_freq

        # Set TTL axis
        self._ax_ttl.clear()
        self._ax_ttl.set_ylabel('TTL')
        self._ln_ttl = self._ax_ttl.plot(
            sig_xi, np.zeros(buf_size), 'k-')
        self._ax_ttl.set_xlim(sig_xi[0], sig_xi[-1])
        self._ax_ttl.set_ylim((-0.1, 1.1))
        self._ax_ttl.set_yticks((0, 1))
        self._ax_ttl.yaxis.set_ticks_position("right")

        # Set card axis
        self._ax_card.clear()
        self._ax_card.set_ylabel('Cardiogram')
        self._ln_card = self._ax_card.plot(
            sig_xi, np.zeros(buf_size), 'k-')
        self._ax_card.set_xlim(sig_xi[0], sig_xi[-1])
        self._ax_card.yaxis.set_ticks_position("right")

        # Set Resp axis
        self._ax_resp.clear()
        self._ax_resp.set_ylabel('Respiration')
        self._ln_resp = self._ax_resp.plot(
            sig_xi, np.zeros(buf_size), 'k-')
        self._ax_resp.set_xlim(sig_xi[0], sig_xi[-1])
        self._ax_resp.yaxis.set_ticks_position("right")
        self._ax_resp.set_xlabel('second')

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def update(self, ttl=None, resp=None, card=None):
        try:
            # If the window is hide, no update is done.
            if self._plt_win.wm_state() in ('iconic', 'withdrawn'):
                return
        except Exception:
            return

        # Get signals
        plt_data = self.recorder.get_plot_signals(self.plot_len_sec)
        if plt_data is None:
            return

        ttl = plt_data['ttl']
        card = plt_data['card']
        resp = plt_data['resp']
        tstamp = plt_data['tstamp']

        ttl[tstamp == 0] = 0
        card[tstamp == 0] = 0
        resp[tstamp == 0] = 0

        zero_t = time.time() - self._ln_ttl[0].get_xdata()[-1]
        plt_xt = zero_t + self._ln_ttl[0].get_xdata()

        # Resample
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                f = interpolate.interp1d(tstamp, ttl, bounds_error=False)
                ttl = f(plt_xt)
                ttl = (ttl > 0.5).astype(int)

                f = interpolate.interp1d(tstamp, card, bounds_error=False)
                card = f(plt_xt)
                f = interpolate.interp1d(tstamp, resp, bounds_error=False)
                resp = f(plt_xt)

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errstr = ''.join(
                traceback.format_exception(exc_type, exc_obj, exc_tb))
            self.logger.error(errstr)

        # Plot
        # TTL
        self._ln_ttl[0].set_ydata(ttl)

        # Card
        self._ln_card[0].set_ydata(card)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ymin = max(0, np.floor(np.nanmin(card) / 50) * 50)
            ymax = min(1024, np.ceil(np.nanmax(card) / 50) * 50)
        self._ax_card.set_ylim((ymin, ymax))

        # Resp
        self._ln_resp[0].set_ydata(resp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ymin = max(0, np.floor(np.nanmin(resp) / 50) * 50)
            ymax = min(1024, np.ceil(np.nanmax(resp) / 50) * 50)
        self._ax_resp.set_ylim((ymin, ymax))

        if self.recorder.is_scanning:
            self._ln_card[0].set_color('r')
            self._ln_resp[0].set_color('b')
        else:
            self._ln_card[0].set_color('k')
            self._ln_resp[0].set_color('k')

        self.canvas.draw()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def config(self):
        # Get configurations
        conf = self.recorder.get_config()
        conf['Plot length (sec)'] = self.plot_len_sec

        # Open a config dialog
        if self.config_win is not None:
            if self.config_win.winfo_exists():
                return
            else:
                self.config_win.destroy()

        self.config_win = tk.Toplevel(self._plt_root)
        self.config_win.title('Recording configurations')

        # Create widgets
        labs = {}
        widgets = {}
        for lab, val in conf.items():
            if 'port list' not in lab:
                labs[lab] = tk.Label(self.config_win, text=lab,
                                     font=("Arial", 11))
                if 'port' not in lab:
                    widgets[lab] = \
                        tk.Entry(self.config_win, width=10, justify=tk.RIGHT,
                                 font=("Arial", 11))
                    widgets[lab].insert(0, val)
                else:
                    port_dict = conf[lab + ' list']
                    if len(port_dict) == 0:
                        continue
                    combo_list = [f"{k}:{v}" for k, v in port_dict.items()]
                    widgets[lab] = \
                        ttk.Combobox(self.config_win, values=combo_list,
                                     font=("Arial", 11), width=30)
                    widgets[lab].set(val)

        cancelButton = tk.Button(self.config_win, text='Cancel',
                                 font=("Arial", 11),
                                 command=self.cancel_config)
        setButton = tk.Button(self.config_win, text='Set',
                              font=("Arial", 11),
                              command=lambda: self.set_config(widgets))

        # Place widgets
        for ii, (lab, lab_wdgt) in enumerate(labs.items()):
            lab_wdgt.grid(row=ii, column=0, sticky=tk.W+tk.E)
            if lab not in widgets:
                continue
            widgets[lab].grid(row=ii, column=1, columnspan=2, sticky=tk.W+tk.E)

        cancelButton.grid(row=ii+1, column=1, sticky=tk.W+tk.E)
        setButton.grid(row=ii+1, column=2, sticky=tk.W+tk.E)

        # Adjust layout
        col_count, row_count = self.config_win.grid_size()
        for row in range(row_count):
            self.config_win.grid_rowconfigure(row, minsize=32)

        # Move window under the plt_win
        cfg_win_x = self._plt_win.winfo_x()
        cfg_win_y = self._plt_win.winfo_y() + self._plt_win.winfo_height()
        self.config_win.geometry(f"+{cfg_win_x}+{cfg_win_y}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_config(self, widgets):
        conf = {}
        for lab, wdgt in widgets.items():
            conf[lab] = wdgt.get()

        self.recorder.set_config(conf)
        if 'Plot length (sec)' in conf:
            self.plot_len_sec = float(conf['Plot length (sec)'])

        self.set_plot()
        self.update_plot_size(None)

        self.config_win.destroy()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def cancel_config(self):
        self.config_win.destroy()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def dump(self, prefix='rtPhysTTLDump'):
        # Dump all data in the buffers to files
        data = self.recorder.dump()

        if len(data):
            now_str = datetime.now().strftime("%Y%m%dT%H%M%S")
            for name, v in data.items():
                if name == 'scan_onset':
                    continue
                f = Path(f"{prefix}_{name}_{now_str}.npy")
                np.save(f, v)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def show(self):
        # Show window
        while self._plt_win.wm_state() in ('iconic', 'withdrawn'):
            self._plt_win.deiconify()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def hide(self):
        # Hide window
        while self._plt_win.wm_state() not in ('iconic', 'withdrawn'):
            self._plt_win.withdraw()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def update_plot_size(self, event):
        if self._resize_debounce_id:
            self._plt_root.after_cancel(self._resize_debounce_id)
        self._resize_debounce_id = self._plt_root.after(
            1, self._handle_resize, event)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _handle_resize(self, event):
        try:
            # Get the new height of the window
            if event is None:
                new_height = self._plt_win.winfo_height()
            else:
                new_height = event.height

            # Update the size of the Matplotlib canvas
            plot_height = new_height - 1
            self.canvas.get_tk_widget().config(height=plot_height)

            # Get the figure size in inches
            fig_width_inch, fig_height_inch = \
                self.plot_fig.get_size_inches()

            # Calculate the normalized margin values
            left = self.left_margin_inch / fig_width_inch
            right = 1 - (self.right_margin_inch / fig_width_inch)
            top = 1 - (self.top_margin_inch / fig_height_inch)
            bottom = self.bottom_margin_inch / fig_height_inch

            # Adjust the subplots
            self.plot_fig.subplots_adjust(
                left=left, right=right, top=top, bottom=bottom)

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errstr = ''.join(
                traceback.format_exception(exc_type, exc_obj, exc_tb))
            self.logger.error(errstr)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def on_closing(self):
        if self._disable_close:
            return

        self.recorder.close()
        self._plt_win.destroy()


# %% NumatoGPIORecoding class ========================================
class NumatoGPIORecoding():
    """
    Receiving signal in USB GPIO device, Numato Lab 8 Channel USB GPIO (
    https://numato.com/product/8-channel-usb-gpio-module-with-analog-inputs/
    ).
    Read IO0/DIO0 to receive the scan start TTL signal.
    Read IO1/ADC1 to receive cardiogram signal.
    Read IO2/ADC2 to receive respiration signal.
    The device is recognized as 'CDC RS-232 Emulation Demo' or
    'Numato Lab 8 Channel USB GPIO M'
    """
    SUPPORT_DEVICES = ['CDC RS-232 Emulation Demo',
                       'Numato Lab 8 Channel USB GPIO']

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, rbuf_names, sport, sample_freq=500,
                 buf_len_sec=1800, verb=True, debug=False, sim_data=None):
        """ Initialize real-time physio recordign class
        Set parameter values and list of serial ports.

        Parameters
        ----------
        rbuf_names : dict
            RingBuffer names.
        sport : str, optional
            Serial port name connected to the Numato Lab 8 Channel USB GPIO.
            This must be one of the items listed in SUPPORT_DEVS.
        buf_len_sec : float, optional
            Length (seconds) of signal recording buffer. The default is 1800s.
        verb : bool, optional
            Verbose flag to print log message. The default is True.
        """
        self.logger = logging.getLogger('GPIORecorder')

        # Set parameters
        self._rbuf_names = rbuf_names
        self.sample_freq = sample_freq
        self.buf_len = int(buf_len_sec * self.sample_freq)
        self.verb = verb

        # Get available serial ports
        self.dict_sig_sport = {}
        self.update_port_list()

        if sport is not None:
            self.sig_sport = sport
        else:
            # Set the serial device to the first available one.
            if len(self.dict_sig_sport):
                self.sig_sport = list(self.dict_sig_sport.keys())[0]
            else:
                self.sig_sport = None
        self._sig_ser = None

        # Initialize data buffers
        self.rbuf_lock = Lock()
        self.rec_proc = None  # recording process
        self._rbuf = {}
        with self.rbuf_lock:
            for buf in self._rbuf_names:
                self._rbuf[buf] = RingBuffer(self._rbuf_names[buf],
                                             self.buf_len)

        self._debug = debug
        if debug:
            self._sim_card, self._sim_resp = sim_data
            self._sim_data_pos = 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ getter and setter methods +++
    @property
    def sig_sport(self):
        return self._sig_sport

    @sig_sport.setter
    def sig_sport(self, dev):
        if dev is not None:
            if dev not in self.dict_sig_sport:
                self.logger.error(f"{dev} is not available.")
                dev = None
        self._sig_sport = dev

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def update_port_list(self):
        # Get serial port list
        self.dict_sig_sport = {}
        for pt in comports():
            if np.any([re.match(devpat, pt.description) is not None
                       for devpat in
                       NumatoGPIORecoding.SUPPORT_DEVICES]
                      ):
                self.dict_sig_sport[pt.device] = pt.description

        # Sort self.dict_sig_sport by key
        self.dict_sig_sport = {
            k: self.dict_sig_sport[k]
            for k in sorted(list(self.dict_sig_sport.keys()))}

        # Check the availability of the current device
        if hasattr(self, '_sig_sport') and self._sig_sport is not None:
            self.sig_sport = self._sig_sport

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def open_sig_port(self):

        # Check the current port and close if it opens
        if self._sig_ser is not None and self._sig_ser.is_open:
            self._sig_ser.close()
            self._sig_ser = None
            time.sleep(1)  # Be sure to close

        try:
            self._sig_ser = serial.Serial(self._sig_sport, 19200,
                                          timeout=0.001)
            self._sig_ser.flushOutput()
            self._sig_ser.write(b"gpio clear 0\r")

        except serial.serialutil.SerialException:
            if self.verb:
                self.logger.error(f"Failed open {self._sig_sport}")
            self._sig_ser = None
            return False

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errstr = ''.join(
                traceback.format_exception(exc_type, exc_obj, exc_tb))
            self.logger.error(f"Failed to open {self._sig_sport}: {errstr}")
            self._sig_ser = None
            return False

        if self.verb:
            self.logger.info(f"Open signal port {self._sig_sport}")

        return True

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_config(self):
        self.update_port_list()
        if len(self.dict_sig_sport):
            dio_port = \
                f"{self.sig_sport}:{self.dict_sig_sport[self.sig_sport]}"
        else:
            dio_port = 'None'
        conf = {
            'IO port': dio_port,
            'IO port list': self.dict_sig_sport,
            'Sampling freq (Hz)': self.sample_freq,
        }
        return conf

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_config(self, conf):
        for lab, val in conf.items():
            if lab == 'Sampling frequency (Hz)':
                self.sample_freq = float(val)
            elif lab == 'USB port':
                if val != 'None':
                    port = val.split(':')[0]
                    self.sig_sport = port

        if self.is_recording():
            self.start_recording(restart=True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def start_recording(self, restart=False):
        if self.rec_proc is not None and self.rec_proc.is_alive():
            if not restart:
                return
            else:
                self.stop_recording()

        # (Re-)open serial port
        if self._sig_sport is None:
            self.logger.error("IO port is not set")
            return

        # Start recording process
        self.rec_pipe, cmd_pipe = Pipe()
        # self.run(cmd_pipe)
        self.rec_proc = Process(target=self.run, args=(cmd_pipe,))
        # self.rec_proc = Thread(target=self.run, args=(cmd_pipe,))
        self.rec_proc.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def is_recording(self):
        return self.rec_proc is not None and self.rec_proc.is_alive()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def stop_recording(self):
        # Close the recording process
        if self.is_recording():
            self.rec_pipe.send('QUIT')
            self.rec_proc.join()
            del self.rec_proc
        self.rec_proc = None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _read_sig_port(self):
        if self._sig_ser is None:
            return 0

        try:
            self._sig_ser.reset_output_buffer()
            self._sig_ser.reset_input_buffer()
            # TTL
            self._sig_ser.write(b"gpio read 0\r")
            resp0 = self._sig_ser.read(1024)
            self._sig_ser.write(b"adc read 1\r")
            resp1 = self._sig_ser.read(25)
            self._sig_ser.write(b"adc read 2\r")
            resp2 = self._sig_ser.read(25)
            tstamp = time.time()
        except Exception:
            return 0

        # TTL
        ma = re.search(r'gpio read 0\n\r(\d)\n', resp0.decode())
        if ma:
            sig = ma.groups()[0]
            ttl = int(sig == '1')
        else:
            ttl = 0

        if self._wait_ttl_on and ttl:
            self.scan_onset(tstamp)
            self._wait_ttl_on = False

        if self._debug:
            card = self._sim_card[self._sim_data_pos]
            resp = self._sim_resp[self._sim_data_pos]
            self._sim_data_pos = (self._sim_data_pos + 1) % len(self._sim_card)
        else:
            # Card
            try:
                card = float(resp1.decode().split('\n\r')[1])
            except Exception:
                card = np.nan

            # Resp
            try:
                resp = float(resp2.decode().split('\n\r')[1])
            except Exception:
                resp = np.nan

        with self.rbuf_lock:
            self._rbuf['ttl'].append(ttl)
            self._rbuf['card'].append(card)
            self._rbuf['resp'].append(resp)
            self._rbuf['tstamp'].append(tstamp)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def scan_onset(self, onset=None):
        try:
            shm = shared_memory.SharedMemory(name='scan_onset')
        except Exception:
            return None

        if onset is None:
            onset = np.ndarray((1,), dtype=np.dtype(float), buffer=shm.buf)[0]
            shm.close()
            return onset
        else:
            shm_array = np.ndarray((1,), dtype=np.dtype(float), buffer=shm.buf)
            shm_array[0] = onset
            shm.close()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run(self, cmd_pipe=None):
        if not self.open_sig_port():
            return

        self._wait_ttl_on = False
        self.scan_onset(0)

        # +++ chk_cmd +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        def chk_cmd(cmd_pipe):
            # listen command
            if cmd_pipe is not None and cmd_pipe.poll():
                cmd = cmd_pipe.recv()
                if cmd == 'QUIT':
                    if self._sig_ser is not None and self._sig_ser.is_open:
                        self._sig_ser.close()
                        return -1
                elif cmd == 'WAIT_TTL_ON':
                    self.scan_onset(0)
                    self._wait_ttl_on = True

                elif cmd == 'CANCEL_WAIT_TTL':
                    self._wait_ttl_on = False

            return 0

        # Start the recording loop
        rec_interval = 1.0 / self.sample_freq
        rec_delay = 0
        next_rec = time.time() + rec_interval

        while chk_cmd(cmd_pipe) == 0:
            if time.time() >= next_rec-rec_delay:
                st = time.time()
                self._read_sig_port()
                ct = time.time()
                rec_delay = time.time() - st

                while next_rec-rec_delay < ct:
                    next_rec += rec_interval

        # Close serial port
        if self._sig_ser is not None and self._sig_ser.is_open:
            self._sig_ser.close()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def wait_ttl(self, state):
        if state:
            self.rec_pipe.send('WAIT_TTL_ON')
        else:
            self.rec_pipe.send('WAIT_TTL_OFF')

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_rbuf_prop(self):
        rbuf_prop = {}
        for k, rbuf in self._rbuf.items():
            rbuf_prop[k] = {'shm_name': rbuf.shm_name,
                            'max_size': rbuf.max_size,
                            'dtype': rbuf.dtype}

        return rbuf_prop


# %% ==========================================================================
class RtSignalRecorder():
    """
    Recording signals from GPIO.
    """
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, cmd_pipe=None, buf_len_sec=1800, sport=None,
                 sample_freq=500, debug=False, sim_data=None):
        """ Initialize real-time signal recording class
        Set parameter values and list of serial ports.

        Parameters
        ----------
        cmd_pipe : multiprocessing.Pipe object, optional
            Pipe for communication with a parent process.
        buf_len_sec : float, optional
            Length (seconds) of signal recording buffer. The default is 1800s.
        sample_freq : float, optional
            Frequency (Hz) of raw signal data. The default is 500.
        sport : str
            Serial port name. The default is None.
        verb : bool, optional
            Verbose flag to print log message. The default is True.
        """
        self.logger = logging.getLogger('RtGPIORecorder')
        self.cmd_pipe = cmd_pipe
        self.buf_len_sec = buf_len_sec
        self.sample_freq = sample_freq
        self.is_scanning = False
        self.work_dir = Path.cwd()
        self.plot = None
        self._rbuf_names = {'ttl': 'ttl',
                            'card': 'card',
                            'resp': 'resp',
                            'tstamp': 'tstamp'}

        # Create scan_onset SharedMemory
        try:
            shm = shared_memory.SharedMemory(
                name='scan_onset', create=True, size=np.dtype(float).itemsize)
        except FileExistsError:
            shm = shared_memory.SharedMemory(name='scan_onset')
            shm.close()
            shm.unlink()
            shm = shared_memory.SharedMemory(
                name='scan_onset', create=True, size=np.dtype(float).itemsize)

        scan_onset = np.ndarray((1,), dtype=float, buffer=shm.buf)
        scan_onset[0] = 0
        shm.close()

        # Create recorder
        self._recoder = NumatoGPIORecoding(
            self._rbuf_names, sport, sample_freq, buf_len_sec,
            debug=debug, sim_data=sim_data)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_config(self):
        return self._recoder.get_config()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_config(self, conf):
        self._recoder.set_config(conf)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def start_recording(self, restart=False):
        self._recoder.start_recording(restart)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def is_recording(self):
        return self._recoder.is_recording()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def stop_recording(self):
        self._recoder.stop_recording()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_rbuf_prop(self):
        return self._recoder.get_rbuf_prop()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save_physio_data(self, onset=None, len_sec=None, fname_fmt='./{}.1D'):
        # Get data
        data = self.dump()
        tstamp = data['tstamp']

        if onset is None:
            shm = shared_memory.SharedMemory(name='scan_onset')
            onset = np.ndarray((1,), dtype=np.dtype(float), buffer=shm.buf)[0]
            shm.close()

        if len_sec is not None:
            offset = onset + len_sec
        else:
            offset = tstamp[-1]

        dataMask = (tstamp >= onset) & (tstamp <= offset) & \
            np.logical_not(np.isnan(tstamp))
        save_card = data['card'][dataMask]
        save_resp = data['resp'][dataMask]
        tstamp = tstamp[dataMask]

        # Resample
        ti = np.arange(onset, offset, 1.0/self.sample_freq)
        f = interpolate.interp1d(tstamp, save_card, bounds_error=False)
        save_card = f(ti)
        f = interpolate.interp1d(tstamp, save_resp, bounds_error=False)
        save_resp = f(ti)

        # Set filename
        card_fname = Path(str(fname_fmt).format(f'Card_{self.sample_freq}Hz'))
        resp_fname = Path(str(fname_fmt).format(f'Resp_{self.sample_freq}Hz'))
        if resp_fname.is_file():
            # Add a number suffux to the filename
            prefix0 = Path(fname_fmt)
            ii = 0
            while resp_fname.is_file():
                ii += 1
                prefix = prefix0.parent / (prefix0.stem + f"_{ii}" +
                                           prefix0.suffix)
                card_fname = Path(str(prefix).format(
                    f'Card_{self.sample_freq}Hz'))
                resp_fname = Path(str(prefix).format(
                    f'Resp_{self.sample_freq}Hz'))

        # Save
        np.savetxt(resp_fname, np.reshape(save_resp, [-1, 1]), '%.2f')
        np.savetxt(card_fname, np.reshape(save_card, [-1, 1]), '%.2f')

        self.logger.info(f"Save physio data in {card_fname} and {resp_fname}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def dump(self):
        if not self.is_recording():
            # No recording
            return None

        # Get data
        with self._recoder.rbuf_lock:
            buf_len = self._recoder.buf_len
            shm = shared_memory.SharedMemory(name='tstamp')
            timestamp = np.ndarray(buf_len, dtype=float, buffer=shm.buf).copy()
            shm.close()
            shm = shared_memory.SharedMemory(name='ttl')
            ttl = np.ndarray(buf_len, dtype=float, buffer=shm.buf).copy()
            shm.close()
            shm = shared_memory.SharedMemory(name='card')
            card = np.ndarray(buf_len, dtype=float, buffer=shm.buf).copy()
            shm.close()
            shm = shared_memory.SharedMemory(name='resp')
            resp = np.ndarray(buf_len, dtype=float, buffer=shm.buf).copy()
            shm.close()
            # timestamp = self._recoder._rbuf['tstamp']._data.copy()
            # ttl = self._recoder._rbuf['ttl']._data.copy()
            # card = self._recoder._rbuf['card']._data.copy()
            # resp = self._recoder._rbuf['resp']._data.copy()

        ttl = ttl[~np.isnan(timestamp)]
        card = card[~np.isnan(timestamp)]
        resp = resp[~np.isnan(timestamp)]
        timestamp = timestamp[~np.isnan(timestamp)]

        sidx = np.argsort(timestamp).ravel()
        ttl = ttl[sidx]
        card = card[sidx]
        resp = resp[sidx]
        timestamp = timestamp[sidx]

        data = {'ttl': ttl, 'card': card, 'resp': resp, 'tstamp': timestamp}
        return data

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_plot_signals(self, plot_len_sec):
        data = self.dump()
        if data is not None:
            data_len = int(plot_len_sec*self.sample_freq)
            for k, dd in data.items():
                if k == 'sacn_onset':
                    continue
                if len(dd) >= data_len:
                    data[k] = dd[-data_len:]
                else:
                    data[k] = np.ones(data_len) * np.nan
                    if len(dd):
                        data[k][-len(dd):] = dd

        return data

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_scan_onset_bkwd(self, TR=None):
        # Read data
        data = self.dump()
        if data is None:
            return

        ttl = data['ttl']
        tstamp = data['tstamp']
        ttl_ons = tstamp[1:][np.diff(ttl) > 0]

        if len(ttl_ons) == 0:
            return

        elif len(ttl_ons) == 1:
            ser_onset = ttl_ons[0]
            interval_thresh = 0.0

        else:
            ttl_interval = np.diff(ttl_ons)
            if TR is not None:
                interval_thresh = TR*1.5
            else:
                interval_thresh = np.nanmin(ttl_interval) * 1.5

            long_intervals = np.argwhere(
                ttl_interval > interval_thresh).ravel()
            if len(long_intervals) == 0:
                ser_onset = ttl_ons[0]
            else:
                ser_onset = ttl_ons[long_intervals[-1]+1]

        shm = shared_memory.SharedMemory(name='scan_onset')
        sacn_onset = np.ndarray((1,), dtype=np.dtype(float), buffer=shm.buf)
        if np.abs(ser_onset - sacn_onset[0]) > interval_thresh:
            sacn_onset[0] = ser_onset

        shm.close()

    # # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # def get_retrots(self, PhysFS, TR, tshift, NVol):
    #     shm = shared_memory.SharedMemory(name='scan_onset')
    #     sacn_onset = np.ndarray((1,), dtype=np.dtype(float), buffer=shm.buf)[0]
    #     if sacn_onset == 0.0:
    #         return (None, None)

    #     data = self.dump()
    #     tstamp = data['tstamp'] - sacn_onset
    #     card = data['card']
    #     resp = data['resp']

    #     # Resample
    #     xt = np.arange(0, tstamp[-1], 1.0/PhysFS)
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore", category=RuntimeWarning)
    #         f = interpolate.interp1d(tstamp, card, bounds_error=False)
    #         Card = f(xt)
    #         f = interpolate.interp1d(tstamp, resp, bounds_error=False)
    #         Resp = f(xt)

    #     n = int(NVol * TR * PhysFS)
    #     Card = Card[:n]
    #     Resp = Resp[:n]

    #     retroTSReg = self._rtp_retrots.do_proc(Resp, Card, TR, PhysFS, tshift)

    #     return retroTSReg

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def RPC_handler(self, call):
        """
        To return data other than str, pack_data() should be used.
        """
        if call == 'GET_RECORDING_PARMAS':
            return pack_data((self.sample_freq, self._recoder.buf_len))

        elif call == 'WAIT_TTL_ON':
            self._recoder.wait_ttl(True)

        elif call == 'CANCEL_WAIT_TTL':
            self._recoder.wait_ttl(False)

        elif call == 'START_SCAN':
            self.is_scanning = True

        elif call == 'END_SCAN':
            self.is_scanning = False

        elif call == 'GET_RBUF_PROP':
            rbuf_prop = self.get_rbuf_prop()
            return pack_data(rbuf_prop)

        elif type(call) is tuple:  # Call with arguments
            if call[0] == 'SAVE_PHYSIO_DATA':
                onset, len_sec, prefix = call[1:]
                self.save_physio_data(onset, len_sec, prefix)

            elif call[0] == 'SET_SCAN_START_BACKWARD':
                TR = call[1]
                self.set_scan_onset_bkwd(TR)

            elif call[0] == 'SET_GEOMETRY':
                print(call)
                if self.plot is not None:
                    geometry = call[1]
                    self.plot.set_position(geometry)

            elif call[0] == 'SET_CONFIG':
                conf = call[1]
                self.set_config(conf)

            # elif call[0] == 'GET_RETROTS':
            #     args = call[1:]
            #     return pack_data(self.get_retrots(*args))

        elif call == 'QUIT':
            self.close()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def close(self):
        self.stop_recording()
        try:
            shm = shared_memory.SharedMemory(name='scan_onset')
            shm.close()
            shm.unlink()
        except Exception:
            pass

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        self.close()


# %% main =====================================================================
if __name__ == '__main__':

    # Parse arguments
    LOG_FILE = f'{Path(__file__).stem}.log'
    parser = argparse.ArgumentParser(description='RT physio')
    parser.add_argument('--log_file', default=LOG_FILE,
                        help='Log file path')
    parser.add_argument('--rpc_port', default=63212,
                        help='RPC socket server port')
    parser.add_argument('--geometry', default='450x450+910+0',
                        help='Plot window position')
    parser.add_argument('--disable_close', action='store_true',
                        help='Disable close button')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    log_file = args.log_file
    rpc_port = args.rpc_port
    geometry = args.geometry
    disable_close = args.disable_close
    debug = args.debug

    # Logger
    logging.basicConfig(level=logging.INFO,
                        filename=log_file, filemode='a',
                        format='%(name)s - %(levelname)s - %(message)s')

    # recorder
    if debug:
        card_f = Path('/data/rt/S20231229091434/Card_500Hz_ser-2.1D')
        resp_f = Path('/data/rt/S20231229091434/Resp_500Hz_ser-2.1D')
        resp = np.loadtxt(resp_f)
        card = np.loadtxt(card_f)
        recorder = RtSignalRecorder(sample_freq=500, debug=True,
                                    sim_data=(card, resp))
    else:
        recorder = RtSignalRecorder()

    # TKinter root window
    plt_root = tk.Tk()

    # Open plot
    signal_plot = TTLPhysioPlot(
        recorder, plt_root, as_root=True, geometry=geometry,
        disable_close=disable_close)
    recorder.plot = signal_plot

    # Start recorder
    recorder.start_recording()

    # Start RPC socket server
    socekt_srv = RPCSocketServer(rpc_port, recorder.RPC_handler,
                                 socket_name='RtPhysioSocketServer')

    # DEBUG
    # st = time.time()
    # TR = 2
    # set_scan_onset = True
    # last_tr = 0
    # while True:
    #     time.sleep(1/60)
    #     signal_plot.update()
    #     plt_root.update()
    #     try:
    #         assert plt_root.winfo_exists()
    #     except Exception:
    #         break

    #     if time.time() - st > 20:
    #         if set_scan_onset:
    #             recorder.set_scan_onset_bkwd()
    #             set_scan_onset = False

    #         if time.time() - last_tr > TR:
    #             NVol = int((time.time() - st) // TR)
    #             retroTSReg = recorder.get_retrots(50, TR, 0, NVol)
    #             last_tr = time.time()
    #             print(retroTSReg)

    #         # retroTSReg = call_rt_physio(
    #         #     ('localhost', rpc_port),
    #         #     ('GET_RETROTS', 50, 2, 0),
    #         #     pkl=True, get_return=True)

    while True:
        time.sleep(1/60)
        signal_plot.update()
        plt_root.update()
        try:
            assert plt_root.winfo_exists()
        except Exception:
            break
