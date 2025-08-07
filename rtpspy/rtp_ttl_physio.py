#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTL and Physiological Signal Recorder Class for RTP

Model Classes:
    NumatoGPIORecording: Handles recording from a Numato GPIO device.
    DummyRecording: Simulates recording using pre-recorded files.
View Class:
    TTLPhysioPlot: Displays plots of TTL, cardiac, and respiration signals.
Controller Class:
    RtpTTLPhysio: The main controller class that interfaces with all
    operations.
"""

# %% import ===================================================================
from pathlib import Path
import os
import time
import sys
import traceback
import multiprocessing as mp
from multiprocessing import Process, Lock, Queue, Pipe
import re
import logging
import argparse
import warnings
import socket
from datetime import datetime
import json
import platform
import threading
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import serial
from serial.tools.list_ports import comports
from scipy.interpolate import interp1d
from scipy.signal import lfilter, firwin
import matplotlib as mpl

from PyQt5 import QtWidgets

try:
    from .rpc_socket_server import (
        RPCSocketServer,
        rpc_send_data,
        rpc_recv_data,
        pack_data,
    )
    from .rtp_common import RTP, MatplotlibWindow
    from .rtp_retrots import RtpRetroTS
except Exception:
    from rtpspy.rpc_socket_server import (
        RPCSocketServer,
        rpc_send_data,
        rpc_recv_data,
        pack_data,
    )
    from rtpspy.rtp_common import RTP, MatplotlibWindow
    from rtpspy.rtp_retrots import RtpRetroTS

mpl.rcParams["font.size"] = 8


# %% Constants ================================================================
DEFAULT_SERIAL_TIMEOUT = 0.001
DEFAULT_SERIAL_BAUDRATE = 19200
DEFAULT_BUFFER_SIZE = 3600  # seconds
DEFAULT_SAMPLE_FREQ = 50  # Hz
PLOT_UPDATE_INTERVAL = 1.0 / 60  # 60 FPS
ADC_MAX_VALUE = 1024
YAXIS_ADJUST_RANGE = 25
MIN_YAXIS_RANGE = 50


# %% Helper functions =========================================================
def log_exception(logger, msg="Exception occurred"):
    """Helper function to log exceptions with traceback."""
    exc_type, exc_obj, exc_tb = sys.exc_info()
    errstr = "".join(traceback.format_exception(exc_type, exc_obj, exc_tb))
    logger.error(f"{msg}: {errstr}")


def create_temp_file(prefix, dir_path="/dev/shm", delete=False):
    """Helper function to create temporary files."""
    if Path(dir_path).is_dir() and os.access(dir_path, os.W_OK):
        return NamedTemporaryFile(
            mode="w+b", prefix=prefix, dir=dir_path, delete=delete
        )
    else:
        return NamedTemporaryFile(prefix=prefix, delete=delete)


# %% call_RtpTTLPhysio ========================================================
def call_RtpTTLPhysio(data, pkl=False, get_return=False, logger=None):
    """
    RPC interface for an RtpTTLPhysio instance

    Parameters:
        data:
            Sending data
        pkl (bool, optional):
            To pack the data in pickle. Defaults to False.
        get_return (bool, optional):
            Flag to receive a return. Defaults to False.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        config_f = Path.home() / ".RTPSpy" / "rtpspy"
        with open(config_f, "r", encoding="utf-8") as fid:
            rtpspy_config = json.load(fid)
        port = rtpspy_config["RtpTTLPhysioSocketServer_port"]
        sock.connect(("localhost", port))
    except ConnectionRefusedError:
        time.sleep(1)
        if data == "ping":
            return False
        return

    if data == "ping":
        return True

    if not rpc_send_data(sock, data, pkl=pkl, logger=logger):
        errmsg = f"Failed to send {data}"
        if logger:
            logger.error(errmsg)
        else:
            sys.stderr.write(errmsg)
        return

    if get_return:
        data = rpc_recv_data(sock, logger=logger)
        if data is None:
            errmsg = f"Failed to receive response to {data}"
            if logger:
                logger.error(errmsg)
            else:
                sys.stderr.write(errmsg)

        return data


# %% SharedMemoryRingBuffer ===================================================
class SharedMemoryRingBuffer:
    """Ring buffer implemented on a memory-mapped NumPy array for sharing data
    across processes.
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(
        self, length, data_file=None, cpos_file=None, initial_value=np.nan
    ):
        """Initialize SharedMemoryRingBuffer.

        Creates an mmap file with the given 'name' in /dev/shm if available;
        otherwise, it is created in temporary directory.

        Parameters
        ----------
        length : int
            Buffer length
        data_file : Path or str, optional
            Path to a data mmap file.
        cpos_file : Path or str, optional
            Path to a current position mmap file.
        initial_value : float, optional
            The value used to initialize the buffer data. The default is nan.
        """
        self._logger = logging.getLogger("SharedMemoryRingBuffer")
        self.length = int(length)
        self._data = None
        self._cpos = None
        self._data_mmap_fd = None
        self._cpos_mmap_fd = None
        self._creator = False

        try:
            if (
                data_file is not None
                and data_file.exists()
                and cpos_file is not None
                and cpos_file.exists()
            ):
                self._load_existing_files(data_file, cpos_file)
            else:
                self._create_new_files()

            self._setup_memory_maps(initial_value)
            self.pid = os.getpid()

        except Exception as e:
            log_exception(
                self._logger, "Failed to initialize SharedMemoryRingBuffer"
            )
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
        self._data_mmap_fd = create_temp_file(f"rtpspy_{pid}_rbuffer_")
        self._cpos_mmap_fd = create_temp_file(f"rtpspy_{pid}_rbuffer_cpos_")

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
    def _setup_memory_maps(self, initial_value):
        """Setup numpy memory maps for data and position."""
        self._data = np.memmap(
            self._data_mmap_fd, dtype=float, mode="w+", shape=(self.length,)
        )
        self._cpos = np.memmap(
            self._cpos_mmap_fd, dtype=np.int64, mode="w+", shape=(1,)
        )

        if initial_value is not None:
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
                self._logger.error("Ring buffer not properly initialized")
                return

            cpos = self._cpos[0]
            self._data[cpos] = x
            self._cpos[0] = (cpos + 1) % self.length
            self._data.flush()
            self._cpos.flush()

        except Exception:
            log_exception(self._logger, "Failed to append to ring buffer")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get(self):
        """Return list of elements"""
        try:
            # Check if attributes exist
            if self._cpos is None or self._data is None:
                self._logger.error("Ring buffer not properly initialized")
                return np.array([])

            cpos = self._cpos[0]
            if cpos == 0:
                return self._data.copy()
            else:
                data = self._data
                return np.concatenate([data[cpos:], data[:cpos]])

        except Exception:
            log_exception(self._logger, "Failed to get ring buffer data")
            return np.array([])

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        try:
            if hasattr(self, "_creator") and self._creator:
                if (
                    hasattr(self, "data_mmap_file")
                    and self.data_mmap_file.is_file()
                ):
                    self.data_mmap_file.unlink()
                    self._logger.debug(
                        f"Delete temporary file: {self.data_mmap_file}"
                    )

                if (
                    hasattr(self, "cpos_mmap_file")
                    and self.cpos_mmap_file.is_file()
                ):
                    self.cpos_mmap_file.unlink()
                    self._logger.debug(
                        f"Delete temporary file: {self.cpos_mmap_file}"
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
            # Don't raise exceptions in __del__
            pass


# %% NumatoGPIORecoding class =================================================
class NumatoGPIORecording:
    """
    Receiving signals from USB GPIO device, Numato Lab 8 Channel USB GPIO (
    https://numato.com/product/8-channel-usb-gpio-module-with-analog-inputs/
    ).
    Read IO0/DIO0 to receive the scan start TTL signal.
    Read IO1/ADC1 to receive cardiogram signal.
    Read IO2/ADC2 to receive respiration signal.
    The device is recognized as 'CDC RS-232 Emulation Demo' or
    'Numato Lab 8 Channel USB GPIO M'
    """

    SUPPORT_DEVICES = [
        "CDC RS-232 Emulation Demo",
        "Numato Lab 8 Channel USB GPIO",
    ]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(
        self,
        ttl_onset_que,
        ttl_offset_que,
        physio_que,
        sport,
        rec_sample_freq=DEFAULT_SAMPLE_FREQ,
    ):
        """Initialize

        Parameters
        ----------
        ttl_onset_que, ttl_offset_que, physio_que : multiprocessing.Queue
           Queues for sharing the recorded data with the controller class.
        sport : str, optional
            Serial port name.
            The default is None, in which case the first available port with a
            device from SUPPORT_DEVICES is used.
        rec_sample_freq : float
            Recording sampling frequency (Hz).
        """
        self._logger = logging.getLogger("NumatoGPIORecoding")

        # Set parameters
        self._ttl_onset_que = ttl_onset_que
        self._ttl_offset_que = ttl_offset_que
        self._physio_que = physio_que
        self._sample_freq = rec_sample_freq

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

    def _is_supported_device(self, description):
        """Check if device description matches any supported patterns."""
        return any(
            re.match(pattern, description) is not None
            for pattern in NumatoGPIORecording.SUPPORT_DEVICES
        )

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def open_sig_port(self):
        """Open serial port for signal communication."""
        if self._sig_sport is None:
            self._logger.warning("There is no Numato GPIO device.")
            return False

        self._close_existing_port()

        try:
            self._sig_ser = serial.Serial(
                self._sig_sport,
                DEFAULT_SERIAL_BAUDRATE,
                timeout=DEFAULT_SERIAL_TIMEOUT,
            )
            self._sig_ser.flushOutput()
            self._sig_ser.write(b"gpio clear 0\r")
            self._logger.info(f"Open signal port {self._sig_sport}")
            return True

        except serial.serialutil.SerialException:
            self._logger.error(f"Failed to open {self._sig_sport}")
        except Exception:
            log_exception(self._logger, f"Failed to open {self._sig_sport}")

        self._sig_ser = None
        return False

    def _close_existing_port(self):
        """Close existing serial port if open."""
        if self._sig_ser is not None and self._sig_ser.is_open:
            self._sig_ser.close()
            self._sig_ser = None
            time.sleep(1)  # Ensure port is closed

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_config(self):
        """Get current device configuration."""
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
        if not self.open_sig_port():
            return

        self._logger.debug("Start recording in read_signal_loop.")

        ttl_state = 0
        rec_interval = 1.0 / self._sample_freq
        rec_delay = 0
        next_rec = time.time() + rec_interval
        st_physio_read = 0
        tstamp_physio = None
        while True:
            # Read TTL
            self._sig_ser.reset_output_buffer()
            self._sig_ser.reset_input_buffer()
            self._sig_ser.write(b"gpio read 0\r")
            port0 = self._sig_ser.read(1024)
            tstamp_ttl = time.time()

            if time.time() >= next_rec - rec_delay:
                st_physio_read = time.time()
                self._sig_ser.reset_output_buffer()
                self._sig_ser.reset_input_buffer()
                # Card
                self._sig_ser.write(b"adc read 1\r")
                port1 = self._sig_ser.read(25)
                # Resp
                self._sig_ser.write(b"adc read 2\r")
                port2 = self._sig_ser.read(25)
                tstamp_physio = time.time()
            else:
                tstamp_physio = None
                port1 = None
                port2 = None

            ma = re.search(r"gpio read 0\n\r(\d)\n", port0.decode())
            if ma:
                sig = ma.groups()[0]
                ttl = int(sig == "1")
            else:
                ttl = 0

            if ttl_state == 0 and ttl == 1:
                self._ttl_onset_que.put(tstamp_ttl)
            elif ttl_state == 1 and ttl == 0:
                self._ttl_offset_que.put(tstamp_ttl)
            ttl_state = ttl

            if tstamp_physio is not None:
                # Card
                try:
                    card = float(port1.decode().split("\n\r")[1])
                except Exception:
                    card = np.nan

                # Resp
                try:
                    resp = float(port2.decode().split("\n\r")[1])
                except Exception:
                    resp = np.nan

                self._physio_que.put((tstamp_physio, card, resp))

                ct = time.time()
                rec_delay = time.time() - st_physio_read

                while next_rec - rec_delay < ct:
                    next_rec += rec_interval

            if cmd_pipe is not None and cmd_pipe.poll():
                cmd = cmd_pipe.recv()
                self._logger.debug(f"Receive {cmd} in read_signal_loop.")
                if cmd == "QUIT":
                    cmd_pipe.send("END")
                    break

            time.sleep(0.001)


# %% DummyRecording ===========================================================
class DummyRecording:
    """Dummy class for physio recording"""

    def __init__(
        self,
        ttl_onset_que,
        ttl_offset_que,
        physio_que,
        sim_card_f=None,
        sim_resp_f=None,
        sample_freq=40,
    ):
        self._logger = logging.getLogger("DummyRecording")

        # Set parameters
        self._ttl_onset_que = ttl_onset_que
        self._ttl_offset_que = ttl_offset_que
        self._physio_que = physio_que
        self._sample_freq = sample_freq
        self._sim_card = None
        self._sim_resp = None
        sim_data_len = np.inf

        if sim_card_f is not None:
            sim_card_f = Path(sim_card_f)
            if not sim_card_f.is_file():
                self._logger.error(
                    f"Not found {sim_card_f} for cardiac dummy signal."
                )
            else:
                try:
                    self._sim_card = np.loadtxt(sim_card_f)
                    sim_data_len = len(self._sim_card)
                except Exception:
                    self._logger.error(f"Error reading {sim_card_f}")

        if sim_resp_f is not None:
            sim_resp_f = Path(sim_resp_f)
            if not sim_resp_f.is_file():
                self._logger.error(
                    f"Not found {sim_resp_f} for respiration dummy signal."
                )
            else:
                try:
                    self._sim_resp = np.loadtxt(sim_resp_f)
                    sim_data_len = min(sim_data_len, len(self._sim_resp))
                except Exception:
                    self._logger.error(f"Error reading {sim_resp_f}")

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
    def set_sim_data(self, sim_card_f, sim_resp_f, sample_freq):
        if not sim_card_f.is_file():
            self._logger.error(
                f"Not found {sim_card_f} for cardiac dummy signal."
            )
            return
        else:
            try:
                self._sim_card = np.loadtxt(sim_card_f)
                sim_data_len = len(self._sim_card)
            except Exception:
                self._logger.error(f"Error reading {sim_card_f}")
                return

        if not sim_resp_f.is_file():
            self._logger.error(
                f"Not found {sim_resp_f} for respiration dummy signal."
            )
            return
        else:
            try:
                self._sim_resp = np.loadtxt(sim_resp_f)
                sim_data_len = min(sim_data_len, len(self._sim_resp))
            except Exception:
                self._logger.error(f"Error reading {sim_resp_f}")
                return

        self._sim_data_len = sim_data_len
        self._sample_freq = sample_freq
        self._sim_data_pos = 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def read_signal_loop(self, cmd_pipe=None):
        self._logger.debug("Start recording in read_signal_loop.")

        ttl_state = 0
        rec_interval = 1.0 / self._sample_freq
        rec_delay = 0
        next_rec = time.time() + rec_interval
        st_physio_read = 0
        tstamp_physio = None
        while True:
            tstamp_ttl = time.time()
            if time.time() >= next_rec - rec_delay:
                st_physio_read = time.time()
                tstamp_physio = time.time()
            else:
                tstamp_physio = None

            ttl = 0
            if ttl_state == 0 and ttl == 1:
                self._ttl_onset_que.put(tstamp_ttl)
            elif ttl_state == 1 and ttl == 0:
                self._ttl_offset_que.put(tstamp_ttl)
            ttl_state = ttl

            if tstamp_physio is not None:
                if self._sim_card is not None:
                    card = self._sim_card[self._sim_data_pos]
                else:
                    card = 1

                if self._sim_resp is not None:
                    resp = self._sim_resp[self._sim_data_pos]
                else:
                    resp = 1

                self._sim_data_pos = (
                    self._sim_data_pos + 1
                ) % self._sim_data_len
                self._physio_que.put((tstamp_physio, card, resp))
                ct = time.time()
                rec_delay = time.time() - st_physio_read

                while next_rec - rec_delay < ct:
                    next_rec += rec_interval

            if cmd_pipe is not None and cmd_pipe.poll():
                cmd = cmd_pipe.recv()
                self._logger.debug(f"Receive {cmd} in read_signal_loop.")
                if cmd == "QUIT":
                    cmd_pipe.send("END")
                    break

            time.sleep(0.01)


# %% TTLPhysioPlot ============================================================
class TTLPhysioPlot:
    """View class for displaying TTL and physio recording signals"""

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, recorder):
        super().__init__()

        self._logger = logging.getLogger("TTLPhysioPlot")
        self.recorder = recorder

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def open(self, plot_geometry, plot_len_sec=10, disable_close=False):
        self._plot_len_sec = plot_len_sec
        self.disable_close = disable_close
        self.is_scanning = False
        self._cancel = False

        # Initialize figure
        plt_winname = f"Physio signals ({self.recorder._device})"
        self.plt_win = MatplotlibWindow()
        self.plt_win.setWindowTitle(plt_winname)
        self.plt_win.setGeometry(*plot_geometry)
        self.init_plot()

        # show window
        self.plt_win.show()

        # Plot process
        self._plt_proc_pipe, cmd_pipe = Pipe()
        self._pltTh = threading.Thread(
            target=self._run, args=(cmd_pipe,), daemon=True
        )
        self._pltTh.start()

        # os_name = platform.system()
        # if os_name == "Linux":
        #     self._plt_proc = Process(
        #         target=self._run, args=(cmd_pipe, plot_geometry)
        #     )
        # elif os_name == "Darwin":
        #     ctx = mp.get_context("fork")
        #     self._plt_proc = ctx.Process(
        #         target=self._run, args=(cmd_pipe, self._rbuf_lock)
        #     )
        # else:
        #     assert False

        # self._plt_proc.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def init_plot(self):
        # Set axis
        self._ax_ttl, self._ax_card, self._ax_card_filtered, self._ax_resp = (
            self.plt_win.canvas.figure.subplots(4, 1)
        )
        self.plt_win.canvas.figure.subplots_adjust(
            left=0.05, bottom=0.1, right=0.91, top=0.98, hspace=0.35
        )

        signal_freq = self.recorder.sample_freq
        buf_size = int(np.round(self._plot_len_sec * signal_freq))
        sig_xi = np.arange(buf_size) * 1.0 / signal_freq

        # Set TTL axis
        self._ax_ttl.clear()
        self._ax_ttl.set_ylabel("TTL")
        self._ln_ttl = self._ax_ttl.plot(sig_xi, np.zeros(buf_size), "k-")
        self._ax_ttl.set_xlim(sig_xi[0], sig_xi[-1])
        self._ax_ttl.set_ylim((-0.1, 1.1))
        self._ax_ttl.set_yticks((0, 1))
        self._ax_ttl.yaxis.set_ticks_position("right")

        # Set card axis
        self._ax_card.clear()
        self._ax_card.set_ylabel("Cardiogram")
        self._ln_card = self._ax_card.plot(sig_xi, np.zeros(buf_size), "k-")
        self._ax_card.set_xlim(sig_xi[0], sig_xi[-1])
        self._ax_card.yaxis.set_ticks_position("right")

        # Set filtered card axis
        self._ax_card_filtered.clear()
        self._ax_card_filtered.set_ylabel("Cardiogram (filtered)")
        self._ln_card_filtered = self._ax_card_filtered.plot(
            sig_xi, np.zeros(buf_size), "k-"
        )
        self._ax_card_filtered.set_xlim(sig_xi[0], sig_xi[-1])
        self._ax_card_filtered.yaxis.set_ticks_position("right")

        # Set Resp axis
        self._ax_resp.clear()
        self._ax_resp.set_ylabel("Respiration")
        self._ln_resp = self._ax_resp.plot(sig_xi, np.zeros(buf_size), "k-")
        self._ax_resp.set_xlim(sig_xi[0], sig_xi[-1])
        self._ax_resp.yaxis.set_ticks_position("right")
        self._ax_resp.set_xlabel("second")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _run(self, cmd_pipe):
        signal_freq = self.recorder.sample_freq

        while self.plt_win.isVisible() and not self._cancel:
            time.sleep(0.001)
            if cmd_pipe.poll():
                cmd = cmd_pipe.recv()
                if cmd == "QUIT":
                    self._logger.debug("Receive QUIT in _run.")
                    break

            try:
                # Get signals
                plt_data = self.recorder.get_plot_signals(
                    self._plot_len_sec + 1
                )
                if plt_data is None:
                    continue

                ttl_init_state = plt_data["ttl_init_state"]
                ttl_onsets = plt_data["ttl_onsets"]
                ttl_offsets = plt_data["ttl_offsets"]
                card = plt_data["card"]
                resp = plt_data["resp"]
                tstamp = plt_data["tstamp"]

                card[tstamp == 0] = 0
                resp[tstamp == 0] = 0
                zero_t = time.time() - np.max(self._ln_ttl[0].get_xdata())
                tstamp = tstamp - zero_t
                ttl_onsets = ttl_onsets - zero_t
                ttl_offsets = ttl_offsets - zero_t
                plt_xt = self._ln_ttl[0].get_xdata()

                # Extend xt (time points) for interpolation
                xt_interval = np.mean(np.diff(plt_xt))
                l_xt_extend = (
                    np.arange(-100 * xt_interval, 0, xt_interval) + plt_xt[0]
                )
                r_xt_extend = np.arange(
                    plt_xt[-1] + xt_interval,
                    tstamp[-1] + xt_interval,
                    xt_interval,
                )
                xt_interval_ex = np.concatenate(
                    [l_xt_extend, plt_xt, r_xt_extend]
                )
                xt_ex_mask = [t in plt_xt for t in xt_interval_ex]

                # --- Resample in regular interval ----------------------------
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore",
                                              category=RuntimeWarning)
                        f = interp1d(tstamp, card, bounds_error=False)
                        card_ex = f(xt_interval_ex)
                        card = card_ex[xt_ex_mask]

                        f = interp1d(tstamp, resp, bounds_error=False)
                        resp_ex = f(xt_interval_ex)
                        resp = resp_ex[xt_ex_mask]

                except Exception:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    errstr = "".join(
                        traceback.format_exception(exc_type, exc_obj, exc_tb)
                    )
                    self._logger.error(errstr)

                # --- Plot ----------------------------------------------------
                # TTL
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
                        (ttl_offsets >= plt_xt[0]) &
                        (ttl_offsets <= plt_xt[-1])
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
                        [
                            int(np.round(ons * signal_freq))
                            for ons in on_off_time
                        ],
                        dtype=int,
                    )
                    on_off_idx = on_off_idx[
                        (on_off_idx >= 0) & (on_off_idx < len(ttl))
                    ]

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
                    ttl = np.ones_like(card) * ttl_init_state

                self._ln_ttl[0].set_ydata(ttl)

                # Adjust crad/resp ylim for the latest adjust_period seconds
                adjust_period = int(np.round(3 * self.recorder.sample_freq))

                # Card
                self._ln_card[0].set_ydata(card)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    # Adjust ylim
                    ymin = np.floor(np.nanmin(card[-adjust_period:]) / 25) * 25
                    ymax = np.ceil(np.nanmax(card[-adjust_period:]) / 25) * 25
                    if ymax - ymin < 50:
                        if ymin + 50 < 1024:
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

                # Card filtered
                b = firwin(
                    numtaps=41,
                    cutoff=3,
                    window="hamming",
                    pass_zero="lowpass",
                    fs=self.recorder.sample_freq,
                )
                card_ex_filtered = lfilter(b, 1, card_ex)
                card_ex_filtered = np.flipud(card_ex_filtered)
                card_ex_filtered = lfilter(b, 1, card_ex_filtered)
                card_ex_filtered = np.flipud(card_ex_filtered)
                card_filtered = card_ex_filtered[xt_ex_mask]
                self._ln_card_filtered[0].set_ydata(card_filtered)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    # Adjust ylim
                    ymin = (
                        np.floor(np.nanmin(card_filtered[-adjust_period:]
                                           ) / 25) * 25
                    )
                    ymax = (
                        np.ceil(np.nanmax(card_filtered[-adjust_period:]) / 25)
                        * 25
                    )
                    if ymax - ymin < 50:
                        if ymin + 50 < 1024:
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

                # Resp
                self._ln_resp[0].set_ydata(resp)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    # Adjust ylim
                    ymin = np.floor(np.nanmin(resp) / 25) * 25
                    ymax = np.ceil(np.nanmax(resp) / 25) * 25
                    if ymax - ymin < 50:
                        if ymin + 50 < 1024:
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

                if self.is_scanning:
                    self._ln_card[0].set_color("r")
                    self._ln_resp[0].set_color("b")
                else:
                    self._ln_card[0].set_color("k")
                    self._ln_resp[0].set_color("k")

                self.plt_win.canvas.draw()
                time.sleep(1 / 60)

            except Exception:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                errmsg = "".join(
                    traceback.format_exception(exc_type, exc_obj, exc_tb)
                )
                self._logger.error(errmsg)

        self.end_thread()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end_thread(self):
        self.plt_win.close()
        del self.plt_win

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def close(self):
        if not hasattr(self, "_pltTh") or not self._pltTh.is_alive():
            return

        self._plt_proc_pipe.send("QUIT")
        self._pltTh.join(2)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_position(self, size=None, pos=None):
        if size is not None:
            self.plt_win.resize(size)

        if pos is not None:
            self.plt_win.move(pos)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def closeEvent(self, event):
        if self.disable_close:
            event.ignore()
        else:
            self.close()


# %% RtpTTLPhysio =============================================================
class RtpTTLPhysio(RTP):
    """
    Interface for external signals, including TTL and physiological signals.
    The recording class runs in a separate process, and the data are shared
    with the process of this class via Queue.
    The recording loop in _run_recording also runs in a separate process and
    the data are saved in an mmap file in /dev/shm if available; otherwise,
    they are stored in $HOME/.RTPSpy.

    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(
        self,
        device=None,
        sample_freq=DEFAULT_SAMPLE_FREQ,
        buf_len_sec=3600,
        sport=None,
        sim_card_f=None,
        sim_resp_f=None,
        save_ttl=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        del self.work_dir  # This is set by RTP base class but not used here.

        # Set parameters
        self.buf_len_sec = buf_len_sec
        self.save_ttl = save_ttl
        self.sample_freq = sample_freq
        self.sim_card_f = sim_card_f
        self.sim_resp_f = sim_resp_f

        # Set state variables
        self.wait_ttl_on = False  # Waiting for TTL to signal scan start.
        self._plot = None  # View class of signal plot
        self._recorder = None  # Signal recorder class

        # Queues to retrieve recorded data from a recorder process
        self._ttl_onset_que = Queue()
        self._ttl_offset_que = Queue()
        self._physio_que = Queue()

        # Initializing recording process variables
        self._rec_proc = None  # Signal recording process
        self._rec_proc_pipe = None

        # Scan onset mmap file for sharing among multiple processes
        self._scan_onset = SharedMemoryRingBuffer(1, initial_value=-1.0)

        # Prepare data buffer files for sharing among multiple processes
        # Buffer names for sharing data across processes
        self._rbuf_names = [
            "ttl_onsets",
            "ttl_offsets",
            "card",
            "resp",
            "tstamp",
        ]
        self._rbuf_lock = Lock()

        # Start RPC socket server
        self.socket_srv = RPCSocketServer(
            self.RPC_handler, socket_name="RtpTTLPhysioSocketServer"
        )

        self._retrots = RtpRetroTS()

        # Set device and start recording
        self.set_device(device, sport=sport)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _init_ring_buffers(self):
        """Initialize ring buffers for sharing data among processes."""
        # Delete existing buffers if they exist
        if hasattr(self, "_rbuf"):
            for rbuf in self._rbuf.values():
                try:
                    del rbuf
                except Exception:
                    pass

        buf_len = self.buf_len_sec * self.sample_freq
        self._rbuf = {}
        for label in self._rbuf_names:
            if label == "tstamp":
                initial_value = 0.0
            else:
                initial_value = np.nan
            self._rbuf[label] = SharedMemoryRingBuffer(
                buf_len, initial_value=initial_value
            )

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_device(self, device, sport=None):
        """Change or reset recording device."""
        if device is not None and device not in (
            "Numato",
            "GE",
            "Dummy",
            "NULL",
        ):
            self._logger.error(f"Device {device} is not supported.")
            return

        if self._recorder is not None:
            self.stop_recording()

        # --- Set recorder device ---
        if device is None:
            try_devices = ["Numato", "GE", "Dummy"]
        else:
            try_devices = [device]

        for dev in try_devices:
            if dev == "Numato":
                self._recorder = NumatoGPIORecording(
                    self._ttl_onset_que,
                    self._ttl_offset_que,
                    self._physio_que,
                    sport,
                    rec_sample_freq=self.sample_freq,
                )
                if self._recorder._sig_sport is None:
                    self._recorder = None

            elif dev == "GE":
                self._recorder = None

            elif dev == "Dummy":
                if (
                    self.sim_card_f is not None
                    and Path(self.sim_card_f).is_file()
                    and self.sim_resp_f is not None
                    and Path(self.sim_resp_f).is_file()
                ):
                    self._recorder = DummyRecording(
                        self._ttl_onset_que,
                        self._ttl_offset_que,
                        self._physio_que,
                        sim_card_f=self.sim_card_f,
                        sim_resp_f=self.sim_resp_f,
                        sample_freq=self.sample_freq,
                    )

            elif dev == "NULL":
                self._recorder = None

            if self._recorder is not None:
                self._device = dev
                break

        if self._recorder is None:
            self._device = "NULL"
        else:
            self.start_recording()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # getter, setter
    @property
    def scan_onset(self):
        return self._scan_onset.get()[0]

    @scan_onset.setter
    def scan_onset(self, onset):
        self._scan_onset.append(onset)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def start_recording(self):
        """Start recording loop in a separate process
        If the recording device has been changed, use set_device instead of
        start_recording. The set_device function internally calls
        start_recording.
        """
        self._init_ring_buffers()

        # Empty _physio_ques before starting a new recording process
        while not self._physio_que.empty():
            try:
                self._physio_que.get_nowait()
            except Exception:
                break

        self._rec_proc_pipe, cmd_pipe = Pipe()
        os_name = platform.system()
        if os_name == "Linux":
            self._rec_proc = Process(
                target=self._run_recording, args=(cmd_pipe, self._rbuf_lock)
            )
        elif os_name == "Darwin":
            ctx = mp.get_context("fork")
            self._rec_proc = ctx.Process(
                target=self._run_recording, args=(cmd_pipe, self._rbuf_lock)
            )
        else:
            assert False

        self._rec_proc.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def is_recording(self):
        return self._rec_proc is not None and self._rec_proc.is_alive()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def stop_recording(self, close_plot=True):
        if close_plot:
            self.close_plot()

        if not self.is_recording():
            return

        self._rec_proc_pipe.send("QUIT")
        if self._rec_proc_pipe.poll(timeout=3):
            self._rec_proc_pipe.recv()
            self._rec_proc.join(3)
        else:
            self._rec_proc.terminate()
        del self._rec_proc
        self._rec_proc = None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def open_plot(
        self,
        main_win=None,
        win_shape=(450, 450),
        plot_len_sec=10,
        disable_close=False,
    ):
        if self._plot is None:
            self._plot = TTLPhysioPlot(self)

        if main_win is not None:
            main_geom = main_win.geometry()
            x = main_geom.x() + main_geom.width() + 10
            y = main_geom.y() - 26
        else:
            x = 0
            y = 0
        plot_geometry = (x, y, win_shape[0], win_shape[1])

        self._plot.open(
            plot_geometry,
            plot_len_sec=plot_len_sec,
            disable_close=disable_close,
        )

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def close_plot(self):
        if not hasattr(self, "_plot") or self._plot is None:
            return

        self._plot.close()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_config(self):
        return self._recorder.get_config()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_config(self, conf):
        self._recorder.set_config(conf)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _run_recording(self, cmd_pipe, rbuf_lock):
        self.wait_ttl_on = False
        self.scan_onset = 0

        # Check if the self._rbuf are on the same process
        cpid = os.getpid()
        for lab, rbuf in self._rbuf.items():
            if cpid != rbuf.pid:
                # Get access to rbuf on another process.
                length = rbuf.length
                data_file = rbuf.data_mmap_file
                cpos_file = rbuf.cpos_mmap_file
                rbuf_rev = SharedMemoryRingBuffer(
                    length, data_file=data_file, cpos_file=cpos_file
                )
                self._rbuf[lab] = rbuf_rev

        # Start reading process. The data is shared by queues.
        _recoder_pipe, cmd_pipe_recorder = Pipe()
        _read_proc = Process(
            target=self._recorder.read_signal_loop,
            kwargs={"cmd_pipe": cmd_pipe_recorder},
        )
        _read_proc.start()

        # Queue reading loop
        while True:
            if cmd_pipe.poll():
                cmd = cmd_pipe.recv()
                if cmd == "QUIT":
                    self._logger.debug("Receive QUIT in _run_recording.")
                    _recoder_pipe.send("QUIT")
                    if _recoder_pipe.poll(timeout=3):
                        _recoder_pipe.recv()

                    break

            if not self._ttl_onset_que.empty():
                while not self._ttl_onset_que.empty():
                    ttl_onsets = self._ttl_onset_que.get()
                    if self.wait_ttl_on:
                        self.scan_onset = ttl_onsets
                        self.wait_ttl_on = False
                    with rbuf_lock:
                        self._rbuf["ttl_onsets"].append(ttl_onsets)

            if not self._ttl_offset_que.empty():
                while not self._ttl_offset_que.empty():
                    ttl_offsets = self._ttl_offset_que.get()
                    with rbuf_lock:
                        self._rbuf["ttl_offsets"].append(ttl_offsets)

            if not self._physio_que.empty():
                while not self._physio_que.empty():
                    tstamp, card, resp = self._physio_que.get()
                    with rbuf_lock:
                        self._rbuf["card"].append(card)
                        self._rbuf["resp"].append(resp)
                        self._rbuf["tstamp"].append(tstamp)

            time.sleep(0.5 / self.sample_freq)

        # --- end loop ---
        # Stop recording process
        cmd_pipe.send("END")
        _read_proc.kill()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def dump(self):
        if not self.is_recording():
            # No recording
            return None

        # Check if the self._rbuf are on the same process
        cpid = os.getpid()
        for lab, rbuf in self._rbuf.items():
            if cpid != rbuf.pid:
                # Reset access to rbuf on another process.
                length = rbuf.length
                data_file = rbuf.data_mmap_file
                cpos_file = rbuf.cpos_mmap_file
                rbuf_rev = SharedMemoryRingBuffer(
                    length, data_file=data_file, cpos_file=cpos_file
                )
                self._rbuf[lab] = rbuf_rev

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

        # Show actual sampling frequency
        self._logger.debug(
            "Actual physio sampling rate: "
            f"{1 / np.mean(np.diff(tstamp)):.2f} Hz"
        )

        data = {
            "ttl_onsets": ttl_onsets,
            "ttl_offsets": ttl_offsets,
            "card": card,
            "resp": resp,
            "tstamp": tstamp,
        }

        return data

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save_physio_data(
        self,
        onset=None,
        len_sec=None,
        fname_fmt="./{}.1D",
        resample_regular_interval=True,
    ):
        # Get data
        data = self.dump()
        if data is None:
            return
        tstamp = data["tstamp"]

        if onset is None:
            onset = tstamp[0]
            if self.scan_onset > 0 and self.scan_onset > onset:
                onset = self.scan_onset

        if len_sec is not None:
            offset = onset + len_sec
        else:
            offset = tstamp[-1]

        if self.save_ttl:
            # --- TTL ---
            ttl_onsets = data["ttl_onsets"]
            ttl_onsets = ttl_onsets[
                (ttl_onsets >= onset) & (ttl_onsets < offset)
            ]
            ttl_onsets = ttl_onsets[ttl_onsets < offset]
            ttl_onset_df = pd.DataFrame(
                columns=("DateTime", "TimefromScanOnset")
            )
            ttl_onset_df["DateTime"] = [
                datetime.fromtimestamp(ons).isoformat() for ons in ttl_onsets
            ]
            ttl_onset_df["TimefromScanOnset"] = ttl_onsets - onset
            ttl_onset_fname = Path(str(fname_fmt).format("TTLonset"))
            ttl_onset_fname = ttl_onset_fname.parent / (
                ttl_onset_fname.stem + ".csv"
            )
            ii = 0
            while ttl_onset_fname.is_file():
                ii += 1
                ttl_onset_fname = ttl_onset_fname.parent / (
                    ttl_onset_fname.stem + f"_{ii}" + ttl_onset_fname.suffix
                )
            ttl_onset_df.to_csv(ttl_onset_fname)

            ttl_offsets = data["ttl_offsets"]
            ttl_offsets = ttl_offsets[
                (ttl_offsets >= onset) & (ttl_offsets < offset)
            ]
            ttl_offsets = ttl_offsets[ttl_offsets < offset]
            ttl_offset_df = pd.DataFrame(
                columns=("DateTime", "TimefromScanOnset")
            )
            ttl_offset_df["DateTime"] = [
                datetime.fromtimestamp(ons).isoformat() for ons in ttl_offsets
            ]
            ttl_offset_df["TimefromScanOnset"] = ttl_offsets - onset
            ttl_offset_fname = Path(str(fname_fmt).format("TTLoffset"))
            ttl_offset_fname = ttl_offset_fname.parent / (
                ttl_offset_fname.stem + ".csv"
            )
            ii = 0
            while ttl_offset_fname.is_file():
                ii += 1
                ttl_offset_fname = ttl_offset_fname.parent / (
                    ttl_offset_fname.stem + f"_{ii}" + ttl_offset_fname.suffix
                )
            ttl_offset_df.to_csv(ttl_offset_fname)

            self._logger.info(
                f"Save TTL data in {ttl_onset_fname} and {ttl_offset_fname}"
            )

        # --- Physio data ---
        # As the data is resampled, 2 s outside the scan period is included.
        dataMask = (
            (tstamp >= onset - 2.0)
            & (tstamp <= offset + 2.0)
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
            # Resample
            try:
                ti = np.arange(onset, offset, 1.0 / self.sample_freq)
                f = interp1d(tstamp, save_card, bounds_error=False)
                save_card = f(ti)
                f = interp1d(tstamp, save_resp, bounds_error=False)
                save_resp = f(ti)
            except Exception:
                self._logger.error(
                    f"Failed in resampling for tstamp = {tstamp}"
                )

        # Set filename
        card_fname = Path(str(fname_fmt).format(f"Card_{self.sample_freq}Hz"))
        resp_fname = Path(str(fname_fmt).format(f"Resp_{self.sample_freq}Hz"))
        if resp_fname.is_file():
            # Add a number suffux to the filename
            prefix0 = Path(fname_fmt)
            ii = 0
            while resp_fname.is_file():
                ii += 1
                prefix = prefix0.parent / (
                    prefix0.stem + f"_{ii}" + prefix0.suffix
                )
                card_fname = Path(
                    str(prefix).format(f"Card_{self.sample_freq}Hz")
                )
                resp_fname = Path(
                    str(prefix).format(f"Resp_{self.sample_freq}Hz")
                )

        # Save
        np.savetxt(resp_fname, np.reshape(save_resp, [-1, 1]), "%.2f")
        np.savetxt(card_fname, np.reshape(save_card, [-1, 1]), "%.2f")

        self._logger.info(f"Save physio data in {card_fname} and {resp_fname}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_plot_signals(self, plot_len_sec):
        data = self.dump()
        if data is None:
            return None

        data_len = int(plot_len_sec * self.sample_freq)
        for k in ("tstamp", "card", "resp"):
            dd = data[k]
            if len(dd) >= data_len:
                data[k] = dd[-data_len:]
            else:
                data[k] = np.ones(data_len) * np.nan
                if len(dd):
                    data[k][-len(dd):] = dd

        t0 = np.nanmax(data["tstamp"]) - plot_len_sec
        data["ttl_onsets"] = data["ttl_onsets"][data["ttl_onsets"] >= t0]
        data["ttl_offsets"] = data["ttl_offsets"][data["ttl_offsets"] >= t0]

        onset = data["ttl_onsets"][data["ttl_onsets"] < data["tstamp"][0]]
        offset = data["ttl_offsets"][data["ttl_offsets"] < data["tstamp"][0]]
        if len(onset):
            if len(offset):
                if onset[-1] > offset[-1]:
                    data["ttl_init_state"] = 1
                else:
                    data["ttl_init_state"] = 0
            else:
                data["ttl_init_state"] = 1
        else:
            data["ttl_init_state"] = 0

        return data

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _clean_resamp(self, v, tstamp=None):
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
    def set_scan_onset_bkwd(self, TR=None):
        # Read data
        data = self.dump()
        if data is None:
            return

        ttl_onsets = data["ttl_onsets"]
        ttl_onsets = ttl_onsets[~np.isnan(ttl_onsets)]
        if len(ttl_onsets) == 0:
            return

        elif len(ttl_onsets) == 1:
            ser_onset = ttl_onsets[0]
            interval_thresh = 0.0

        else:
            ttl_interval = np.diff(ttl_onsets)
            if TR is not None:
                interval_thresh = TR * 1.5
            else:
                interval_thresh = np.nanmin(ttl_interval) * 1.5

            long_intervals = np.argwhere(
                ttl_interval > interval_thresh).ravel()
            if len(long_intervals) == 0:
                ser_onset = ttl_onsets[0]
            else:
                ser_onset = ttl_onsets[long_intervals[-1] + 1]

        self.scan_onset = ser_onset

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_retrots(
        self,
        TR,
        Nvol=np.inf,
        tshift=0,
        resample_phys_fs=DEFAULT_SAMPLE_FREQ,
        timeout=2,
    ):
        onset = self.scan_onset
        if onset == 0:
            return None

        data = self.dump()
        if data is None:
            return None

        tstamp = data["tstamp"] - onset

        if np.isinf(Nvol):
            Nvol = int(np.nanmax(tstamp) // TR)
        else:
            st = time.time()
            while (
                int(np.nanmax(tstamp) // TR) < Nvol
                and time.time() - st < timeout
            ):
                self._logger.debug(
                    f"Received data for {np.nanmax(tstamp) / TR}/{Nvol}"
                )
                # Wait until Nvol samples
                time.sleep(0.001)
                data = self.dump()
                tstamp = data["tstamp"] - onset

            if int(np.nanmax(tstamp) // TR) < Nvol:
                # ERROR: timeout
                self._logger.error(
                    "Not received enough data to make RETROICOR regressors"
                    f" for {timeout} s."
                )
                return None

        dataMask = (tstamp >= -TR) & ~np.isnan(tstamp)
        dataMask &= ~np.isnan(data["resp"])
        dataMask &= ~np.isnan(data["card"])
        resp = data["resp"][dataMask]
        card = data["card"][dataMask]
        tstamp = tstamp[dataMask]

        # Resample
        if resample_phys_fs is None or resample_phys_fs > self.sample_freq:
            physFS = self.sample_freq
        else:
            physFS = resample_phys_fs

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            res_t = np.arange(0, Nvol * TR + 1.0, 1.0 / physFS)
            resp_res_f = interp1d(tstamp, resp, bounds_error=False)
            Resp = resp_res_f(res_t)
            Resp = Resp[~np.isnan(Resp)]

            card_res_f = interp1d(tstamp, card, bounds_error=False)
            Card = card_res_f(res_t)
            Card = Card[~np.isnan(Card)]

        retroTSReg = self._retrots.RetroTs(
            Resp, Card, TR, physFS, tshift, Nvol)

        return retroTSReg

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def RPC_handler(self, call):
        """
        To return data other than str, pack_data() should be used.
        """
        self._logger.debug("Process RPC %s", call)

        if call == "GET_RECORDING_PARMAS":
            return pack_data(self.sample_freq)

        elif call == "WAIT_TTL_ON":
            self.wait_ttl_on = True

        elif call == "CANCEL_WAIT_TTL":
            self.wait_ttl_on = False

        elif call == "START_SCAN":
            if self._plot:
                self._plot.is_scanning = True

        elif call == "END_SCAN":
            if self._plot:
                self._plot.is_scanning = False

        elif type(call) is tuple:  # Call with arguments
            if call[0] == "SAVE_PHYSIO_DATA":
                onset, len_sec, prefix = call[1:]
                self.save_physio_data(onset, len_sec, prefix)

            elif call[0] == "SET_SCAN_START_BACKWARD":
                TR = call[1]
                self.set_scan_onset_bkwd(TR)

            elif call[0] == "SET_GEOMETRY":
                if self._plot is not None:
                    geometry = call[1]
                    self._plot.set_position(geometry)

            elif call[0] == "SET_CONFIG":
                conf = call[1]
                self.set_config(conf)

            elif call[0] == "SET_REC_DEV":
                self.set_device(call[1])

        elif call == "QUIT":
            self.end()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end(self):
        self.stop_recording()
        self.socket_srv.shutdown()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        self.end()

        for rbuf in self._rbuf.values():
            data_file = rbuf.data_mmap_file
            for rmf in data_file.parent.glob(
                f"rtpspy_{os.getpid()}_rbuffer_*"
            ):
                rmf.unlink()
                self._logger.debug(f"Remove temporary file {rmf}.")

            cpos_file = rbuf.cpos_mmap_file
            for rmf in cpos_file.parent.glob(f"rtpspy_{os.getpid()}_cpos_*"):
                rmf.unlink()
                self._logger.debug(f"Remove temporary file {rmf}.")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_param(self, attr, val, reset_fn=None, echo=False):
        reset_device = False

        self._logger.debug(f"set_param: {attr} = {val}")

        if attr == "device":
            if self._device != val:
                if self._rec_proc is not None:
                    reset_device = True

        elif attr == "sample_freq":
            setattr(self, attr, val)
            if self._rec_proc is not None:
                reset_device = True

        elif attr == "buf_len_sec":
            setattr(self, attr, val)
            if self._rec_proc is not None:
                reset_device = True

        elif attr == "save_ttl":
            setattr(self, attr, val)

        elif attr == "sport":
            setattr(self, attr, val)
            if self._rec_proc is not None:
                reset_device = True

        elif attr == "sim_card_f":
            setattr(self, attr, val)
            if self._device == "Dummy" and self._rec_proc is not None:
                reset_device = True

        elif attr == "sim_resp_f":
            setattr(self, attr, val)
            if self._device == "Dummy" and self._rec_proc is not None:
                reset_device = True

        if reset_device:
            self.set_device(self._device)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ui_set_param(self):
        """
        When reset_fn is None, set_param is considered to be called from
        load_parameters function.
        """
        return
        ui_rows = []
        self.ui_objs = []

        # # enabled
        # self.ui_enabled_rdb = QtWidgets.QRadioButton("Enable")
        # self.ui_enabled_rdb.setChecked(self.enabled)
        # self.ui_enabled_rdb.toggled.connect(
        #         lambda checked: self.set_param('enabled', checked,
        #                                        self.ui_enabled_rdb.setChecked))
        # ui_rows.append((self.ui_enabled_rdb, None))

        # # mask_file
        # var_lb = QtWidgets.QLabel("Mask :")
        # self.ui_mask_cmbBx = QtWidgets.QComboBox()
        # self.ui_mask_cmbBx.addItems(['external file',
        #                              'initial volume of internal run'])
        # self.ui_mask_cmbBx.activated.connect(
        #         lambda idx:
        #         self.set_param('mask_file',
        #                        self.ui_mask_cmbBx.currentText(),
        #                        self.ui_mask_cmbBx.setCurrentIndex))

        # self.ui_mask_lnEd = QtWidgets.QLineEdit()
        # self.ui_mask_lnEd.setReadOnly(True)
        # self.ui_mask_lnEd.setStyleSheet(
        #     'border: 0px none;')
        # self.ui_objs.extend([var_lb, self.ui_mask_cmbBx,
        #                      self.ui_mask_lnEd])

        # if self.mask_file == 0:
        #     self.ui_mask_cmbBx.setCurrentIndex(1)
        #     self.ui_mask_lnEd.setText('zero-out initial received volume')
        # else:
        #     self.ui_mask_cmbBx.setCurrentIndex(0)
        #     self.ui_mask_lnEd.setText(str(self.mask_file))

        # mask_hLayout = QtWidgets.QHBoxLayout()
        # mask_hLayout.addWidget(self.ui_mask_cmbBx)
        # mask_hLayout.addWidget(self.ui_mask_lnEd)
        # ui_rows.append((var_lb, mask_hLayout))

        # # wait_num
        # var_lb = QtWidgets.QLabel("Wait REGRESS until (volumes) :")
        # self.ui_waitNum_cmbBx = QtWidgets.QComboBox()
        # self.ui_waitNum_cmbBx.addItems(['number of regressors', 'set value'])
        # self.ui_waitNum_cmbBx.activated.connect(
        #         lambda idx:
        #         self.set_param('wait_num',
        #                        self.ui_waitNum_cmbBx.currentText(),
        #                        self.ui_waitNum_cmbBx.setCurrentIndex))

        # self.ui_waitNum_lb = QtWidgets.QLabel()
        # regNum = self.get_reg_num()
        # self.ui_waitNum_lb.setText(
        #         f'Wait REGRESS until receiving {regNum} volumes')
        # self.ui_objs.extend([var_lb, self.ui_waitNum_cmbBx,
        #                      self.ui_waitNum_lb])

        # wait_num_hLayout = QtWidgets.QHBoxLayout()
        # wait_num_hLayout.addWidget(self.ui_waitNum_cmbBx)
        # wait_num_hLayout.addWidget(self.ui_waitNum_lb)
        # ui_rows.append((var_lb, wait_num_hLayout))

        # # max_scan_length
        # var_lb = QtWidgets.QLabel("Maximum scan length :")
        # self.ui_maxLen_spBx = QtWidgets.QSpinBox()
        # self.ui_maxLen_spBx.setMinimum(1)
        # self.ui_maxLen_spBx.setMaximum(9999)
        # self.ui_maxLen_spBx.setValue(self.max_scan_length)
        # self.ui_maxLen_spBx.editingFinished.connect(
        #         lambda: self.set_param('max_scan_length',
        #                                self.ui_maxLen_spBx.value(),
        #                                self.ui_maxLen_spBx.setValue))
        # ui_rows.append((var_lb, self.ui_maxLen_spBx))
        # self.ui_objs.extend([var_lb, self.ui_maxLen_spBx])

        # # max_poly_order
        # var_lb = QtWidgets.QLabel("Maximum polynomial order :\n"
        #                           "regressors for slow fluctuation")
        # self.ui_maxPoly_cmbBx = QtWidgets.QComboBox()
        # self.ui_maxPoly_cmbBx.addItems(['auto', 'set'])
        # self.ui_maxPoly_cmbBx.activated.connect(
        #         lambda idx:
        #         self.set_param('max_poly_order',
        #                        self.ui_maxPoly_cmbBx.currentText(),
        #                        self.ui_maxPoly_cmbBx.setCurrentIndex))

        # self.ui_maxPoly_lb = QtWidgets.QLabel()
        # self.ui_objs.extend([var_lb, self.ui_maxPoly_cmbBx,
        #                      self.ui_maxPoly_lb])
        # if np.isinf(self.max_poly_order):
        #     self.ui_maxPoly_cmbBx.setCurrentIndex(0)
        #     self.ui_maxPoly_lb.setText('Increase polynomial order ' +
        #                                'with the scan length')
        # else:
        #     self.ui_maxPoly_cmbBx.setCurrentIndex(1)
        #     self.ui_maxPoly_lb.setText('Increase polynomial order ' +
        #                                'with the scan length' +
        #                                f' up to {self.max_poly_order}')

        # maxPoly_hLayout = QtWidgets.QHBoxLayout()
        # maxPoly_hLayout.addWidget(self.ui_maxPoly_cmbBx)
        # maxPoly_hLayout.addWidget(self.ui_maxPoly_lb)
        # ui_rows.append((var_lb, maxPoly_hLayout))

        # # mot_reg
        # var_lb = QtWidgets.QLabel("Motion regressor :")
        # self.ui_motReg_cmbBx = QtWidgets.QComboBox()
        # self.ui_motReg_cmbBx.addItems(
        #         ['None', '6 motions (yaw, pitch, roll, dS, dL, dP)',
        #          '12 motions (6 motions and their temporal derivatives)',
        #          '6 motion derivatives'])
        # ci = {'None': 0, 'mot6': 1, 'mot12': 2, 'dmot6': 3}[self.mot_reg]
        # self.ui_motReg_cmbBx.setCurrentIndex(ci)
        # self.ui_motReg_cmbBx.currentIndexChanged.connect(
        #         lambda idx:
        #         self.set_param('mot_reg',
        #                        self.ui_motReg_cmbBx.currentText(),
        #                        self.ui_motReg_cmbBx.setCurrentIndex))
        # ui_rows.append((var_lb, self.ui_motReg_cmbBx))
        # self.ui_objs.extend([var_lb, self.ui_motReg_cmbBx])

        # # GS ROI regressor
        # self.ui_GS_reg_chb = QtWidgets.QCheckBox("Regress global signal :")
        # self.ui_GS_reg_chb.setChecked(self.GS_reg)
        # self.ui_GS_reg_chb.stateChanged.connect(
        #         lambda state: self.set_param('GS_reg', state > 0))

        # GSmask_hBLayout = QtWidgets.QHBoxLayout()
        # self.ui_GS_mask_lnEd = QtWidgets.QLineEdit()
        # self.ui_GS_mask_lnEd.setText(str(self.GS_mask))
        # self.ui_GS_mask_lnEd.setReadOnly(True)
        # self.ui_GS_mask_lnEd.setStyleSheet(
        #     'border: 0px none;')
        # GSmask_hBLayout.addWidget(self.ui_GS_mask_lnEd)

        # self.ui_GSmask_btn = QtWidgets.QPushButton('Set')
        # self.ui_GSmask_btn.clicked.connect(
        #         lambda: self.set_param(
        #                 'GS_mask',
        #                 Path(self.ui_GS_mask_lnEd.text()).parent,
        #                 self.ui_GS_mask_lnEd.setText))
        # GSmask_hBLayout.addWidget(self.ui_GSmask_btn)

        # self.ui_objs.extend([self.ui_GS_reg_chb, self.ui_GS_mask_lnEd,
        #                      self.ui_GSmask_btn])
        # ui_rows.append((self.ui_GS_reg_chb, GSmask_hBLayout))

        # # WM ROI regressor
        # self.ui_WM_reg_chb = QtWidgets.QCheckBox("Regress WM signal :")
        # self.ui_WM_reg_chb.setChecked(self.WM_reg)
        # self.ui_WM_reg_chb.stateChanged.connect(
        #         lambda state: self.set_param('WM_reg', state > 0))

        # WMmask_hBLayout = QtWidgets.QHBoxLayout()
        # self.ui_WM_mask_lnEd = QtWidgets.QLineEdit()
        # self.ui_WM_mask_lnEd.setText(str(self.WM_mask))
        # self.ui_WM_mask_lnEd.setReadOnly(True)
        # self.ui_WM_mask_lnEd.setStyleSheet(
        #     'border: 0px none;')
        # WMmask_hBLayout.addWidget(self.ui_WM_mask_lnEd)

        # self.ui_WMmask_btn = QtWidgets.QPushButton('Set')
        # self.ui_WMmask_btn.clicked.connect(
        #         lambda: self.set_param(
        #                 'WM_mask',
        #                 Path(self.ui_WM_mask_lnEd.text()).parent,
        #                 self.ui_WM_mask_lnEd.setText))
        # WMmask_hBLayout.addWidget(self.ui_WMmask_btn)

        # self.ui_objs.extend([self.ui_WM_reg_chb, self.ui_WM_mask_lnEd,
        #                      self.ui_WMmask_btn])
        # ui_rows.append((self.ui_WM_reg_chb, WMmask_hBLayout))

        # # Vent ROI regressor
        # self.ui_Vent_reg_chb = QtWidgets.QCheckBox("Regress Vent signal :")
        # self.ui_Vent_reg_chb.setChecked(self.Vent_reg)
        # self.ui_Vent_reg_chb.stateChanged.connect(
        #         lambda state: self.set_param('Vent_reg', state > 0))

        # Ventmask_hBLayout = QtWidgets.QHBoxLayout()

        # self.ui_Vent_mask_lnEd = QtWidgets.QLineEdit()
        # self.ui_Vent_mask_lnEd.setText(str(self.Vent_mask))
        # self.ui_Vent_mask_lnEd.setReadOnly(True)
        # self.ui_Vent_mask_lnEd.setStyleSheet(
        #     'border: 0px none;')
        # Ventmask_hBLayout.addWidget(self.ui_Vent_mask_lnEd)

        # self.ui_Ventmask_btn = QtWidgets.QPushButton('Set')
        # self.ui_Ventmask_btn.clicked.connect(
        #         lambda: self.set_param(
        #                 'Vent_mask',
        #                 Path(self.ui_Vent_mask_lnEd.text()).parent,
        #                 self.ui_Vent_mask_lnEd.setText))
        # Ventmask_hBLayout.addWidget(self.ui_Ventmask_btn)

        # self.ui_objs.extend([self.ui_Vent_reg_chb, self.ui_Vent_mask_lnEd,
        #                      self.ui_Ventmask_btn])
        # ui_rows.append((self.ui_Vent_reg_chb, Ventmask_hBLayout))

        # # phys_reg
        # var_lb = QtWidgets.QLabel("RICOR regressor :")
        # self.ui_physReg_cmbBx = QtWidgets.QComboBox()
        # self.ui_physReg_cmbBx.addItems(
        #         ['None', '8 RICOR (4 Resp and 4 Card)']
        #         )
        # ci = {'None': 0, 'RICOR8': 1, 'RVT5': 2,
        #       'RVT+RICOR13': 3}[self.phys_reg]
        # self.ui_physReg_cmbBx.setCurrentIndex(ci)
        # self.ui_physReg_cmbBx.currentIndexChanged.connect(
        #         lambda idx:
        #         self.set_param('phys_reg',
        #                        self.ui_physReg_cmbBx.currentText(),
        #                        self.ui_physReg_cmbBx.setCurrentIndex))
        # ui_rows.append((var_lb, self.ui_physReg_cmbBx))
        # self.ui_objs.extend([var_lb, self.ui_physReg_cmbBx])

        # # desMtx
        # var_lb = QtWidgets.QLabel("Design matrix :")

        # desMtx_hBLayout = QtWidgets.QHBoxLayout()
        # self.ui_loadDesMtx_btn = QtWidgets.QPushButton('Set')
        # self.ui_loadDesMtx_btn.clicked.connect(
        #         lambda: self.set_param('desMtx_f', 'set'))
        # desMtx_hBLayout.addWidget(self.ui_loadDesMtx_btn)

        # self.ui_showDesMtx_btn = QtWidgets.QPushButton()
        # self.ui_showDesMtx_btn.clicked.connect(
        #         lambda: self.set_param('showDesMtx'))
        # desMtx_hBLayout.addWidget(self.ui_showDesMtx_btn)

        # self.ui_objs.extend([var_lb, self.ui_loadDesMtx_btn,
        #                      self.ui_showDesMtx_btn])
        # ui_rows.append((var_lb, desMtx_hBLayout))
        # self.ui_showDesMtx_btn.setText('Show desing matrix')
        # if self.desMtx_read is None:
        #     self.ui_showDesMtx_btn.setEnabled(False)
        # else:
        #     self.ui_showDesMtx_btn.setEnabled(True)

        # # --- Checkbox row ------------------------------------------------
        # # Restrocpective process
        # self.ui_retroProc_chb = QtWidgets.QCheckBox("Retrospective process")
        # self.ui_retroProc_chb.setChecked(self.reg_retro_proc)
        # self.ui_retroProc_chb.stateChanged.connect(
        #         lambda state: setattr(self, 'reg_retro_proc', state > 0))
        # self.ui_objs.append(self.ui_retroProc_chb)

        # # Save
        # self.ui_saveProc_chb = QtWidgets.QCheckBox("Save processed image")
        # self.ui_saveProc_chb.setChecked(self.save_proc)
        # self.ui_saveProc_chb.stateChanged.connect(
        #         lambda state: setattr(self, 'save_proc', state > 0))
        # self.ui_objs.append(self.ui_saveProc_chb)

        # chb_hLayout = QtWidgets.QHBoxLayout()
        # chb_hLayout.addStretch()
        # chb_hLayout.addWidget(self.ui_saveProc_chb)
        # ui_rows.append((self.ui_retroProc_chb, chb_hLayout))

        return ui_rows

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_params(self):
        all_opts = super().get_params()
        incld_opts = ("device", "sample_freq", "buf_len_sec", "save_ttl")
        sel_opts = {}
        for k, v in all_opts.items():
            if k not in incld_opts:
                continue
            if isinstance(v, Path):
                v = str(v)
            sel_opts[k] = v

        return sel_opts


# %% main =====================================================================
if __name__ == "__main__":
    # Parse arguments
    LOG_FILE = f"{Path(__file__).stem}.log"
    parser = argparse.ArgumentParser(description="RTP physio")
    parser.add_argument("--device", default="Numato", help="Device type")
    parser.add_argument(
        "--sample_freq",
        default=DEFAULT_SAMPLE_FREQ,
        type=float,
        help="sampling frequency (Hz)",
    )
    parser.add_argument(
        "--card_file", help="Cardiac signal file for dummy device"
    )
    parser.add_argument(
        "--resp_file", help="Respiration signal file for dummy device"
    )
    parser.add_argument(
        "--win_shape", default="450x450", help="Plot window position"
    )
    parser.add_argument(
        "--disable_close", action="store_true", help="Disable close button"
    )
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    device = args.device
    card_file = args.card_file
    resp_file = args.resp_file
    sample_freq = args.sample_freq
    win_shape = args.win_shape
    win_shape = [int(v) for v in win_shape.split("x")]
    disable_close = args.disable_close
    debug = args.debug

    # Logger
    logging.basicConfig(
        level=logging.DEBUG,
        stream=sys.stdout,
        format="%(asctime)s.%(msecs)04d,%(name)s,%(message)s",
    )

    # Open app
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    if debug:
        test_dir = Path(__file__).absolute().parent.parent / "tests"
        card_file = test_dir / "ECG.1D"
        resp_file = test_dir / "Resp.1D"
        sample_freq = 40
        device = "Dummy"

    # Create RtpTTLPhysio instance
    rtp_ttl_physio = RtpTTLPhysio(
        device=device,
        sample_freq=sample_freq,
        sim_card_f=card_file,
        sim_resp_f=resp_file,
    )

    rtp_ttl_physio.open_plot()

    sys.exit(app.exec_())
