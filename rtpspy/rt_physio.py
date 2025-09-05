#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time physiological signal recording class.
@author: mmisaki@libr.net

Model class : NumatoGPIORecoding, DummyRecording
View class : PlotTTLPhysio
Controler class: RtPhysio
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
from tkinter import ttk
from tempfile import NamedTemporaryFile
import subprocess
from collections import deque
import json
import gc

import numpy as np
import pandas as pd
import serial
from serial import SerialException
from serial.tools.list_ports import comports
from scipy.interpolate import interp1d
from scipy.signal import lfilter, firwin
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib as mpl

try:
    from rpc_socket_server import RPCSocketServer, RPCSocketCom, pack_data
except ImportError:
    from .rpc_socket_server import RPCSocketServer, RPCSocketCom, pack_data


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


# %% create_temp_file =========================================================
def create_temp_file(prefix, dir_path="/dev/shm", delete=False):
    """Helper function to create temporary files. This should be outside of the
    SharedMemoryRingBuffer class.
    """
    if Path(dir_path).is_dir() and os.access(dir_path, os.W_OK):
        return NamedTemporaryFile(
            mode="w+b", prefix=prefix, dir=dir_path, delete=delete
        )
    else:
        return NamedTemporaryFile(prefix=prefix, delete=delete)


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
        self.data_mmap_file = data_file
        self.cpos_mmap_file = cpos_file

        self.pid = None
        self._data = None
        self._cpos = None
        self._data_mmap_fd = None
        self._cpos_mmap_fd = None
        self._creator = False
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
                self._logger.error("Ring buffer not properly initialized")
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

        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            self._logger.error("Failed to get ring buffer data:\n" + errstr)
            return np.array([])

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def reset(self, initial_value=np.nan):
        """Reset the ring buffer by clearing all data and resetting position.
        Parameters
        ----------
        initial_value : float, optional
            The value to fill the buffer with. Default is np.nan.
        """
        try:
            # Check if attributes exist
            if self._cpos is None or self._data is None:
                self._logger.error("Ring buffer not properly initialized")
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
        kwds = {
            "length": self.length,
            "data_file": self.data_mmap_file,
            "cpos_file": self.cpos_mmap_file,
        }
        return kwds

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def validate(self, pid, read_only=False):
        """Validate the access to the shared memory from another process."""
        try:
            if pid == self.pid:
                return self

            self._logger.debug("Accessing shared memory on another process.")

            # Get access to shared memory on another process.
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
        sport : str, optional
            Serial port name.
            The default is None, in which case the first available port with a
            device from SUPPORT_DEVICES is used.
        sample_freq : float
            Recording sampling frequency (Hz).
        """
        self._logger = logging.getLogger("NumatoGPIORecoding")

        # Set parameters
        self._ttl_onset_que = ttl_onset_que
        self._ttl_offset_que = ttl_offset_que
        self._physio_que = physio_que
        self._sample_freq = sample_freq
        self._queue_lock = Lock()

        # Get available serial ports
        self.dict_sig_sport = {}
        self.update_port_list()

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
        """Check if Numato device is available without creating instance."""
        for pt in comports():
            if any(
                re.match(pattern, pt.description) is not None
                for pattern in NumatoGPIORecording.SUPPORT_DEVICES
            ):
                return True
        return False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def open_sig_port(self):
        """Open serial port for signal communication."""
        if self._sig_sport is None:
            self._logger.warning("There is no Numato GPIO device.")
            return False

        self._logger.debug(
            f"Attempting to open serial port: {self._sig_sport}")
        self._close_existing_port()

        # Check if the serial port is accessed by other processes
        try:
            result = subprocess.run(
                ["lsof", self._sig_sport],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode == 0 and result.stdout:
                self._logger.debug(
                    f"Serial port {self._sig_sport} is already in use "
                    "by another process."
                )
                return False

        except FileNotFoundError:
            self._logger.warning(
                "`lsof` command not found. Skipping port access check."
            )
        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            self._logger.error(f"Error checking serial port access: {errstr}")

        if self._sig_sport not in self.dict_sig_sport:
            self._logger.error(f"{self._sig_sport} is not available.")
            return False

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

        except serial.serialutil.SerialException as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            self._logger.error(f"Failed to open {self._sig_sport}: {errstr}")

        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            self._logger.error(f"Failed to open {self._sig_sport}: {errstr}")

        self._sig_ser = None
        return False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _close_existing_port(self):
        """Close existing serial port if open."""
        if self._sig_ser is not None and self._sig_ser.is_open:
            self._logger.debug(f"Closing serial port: {self._sig_sport}")
            self._sig_ser.close()
            self._sig_ser = None
            time.sleep(1)  # Ensure port is closed
            self._logger.debug(f"Serial port closed: {self._sig_sport}")

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
        if (
            self._ttl_onset_que is None
            or self._ttl_offset_que is None
            or self._physio_que is None
        ):
            self._logger.error("Recording queues are not set.")
            return

        if not self.open_sig_port():
            return

        self._logger.debug("Start recording in read_signal_loop.")

        ttl_state = 0
        physio_rec_interval = 1.0 / self._sample_freq
        rec_delays = deque(maxlen=10)  # Stack to store delays, size = 10
        rec_delay = 0
        next_rec = time.time() + physio_rec_interval
        st_physio_read = 0
        tstamp_physio = None
        tstamp_physio0 = None
        while True:
            # Read TTL
            self._sig_ser.reset_output_buffer()
            self._sig_ser.reset_input_buffer()
            self._sig_ser.write(b"gpio read 0\r")

            try:
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
                    rec_delays.append(time.time() - st_physio_read)
                else:
                    tstamp_physio = None
                    port1 = None
                    port2 = None
            except SerialException as e:
                self._logger.debug(f"Error reading signal: {e}")
                pass

            ma = re.search(r"gpio read 0\n\r(\d)\n", port0.decode())
            if ma:
                sig = ma.groups()[0]
                ttl = int(sig == "1")
            else:
                ttl = 0

            if ttl != ttl_state:
                if ttl == 1 and ttl_state == 0:
                    try:
                        self._ttl_onset_que.put_nowait(tstamp_ttl)
                    except Full:
                        try:
                            self._ttl_onset_que.get_nowait()  # discard oldest
                            self._ttl_onset_que.put_nowait(tstamp_ttl)
                        except Empty:
                            pass
                    # self._logger.debug(f"TTL Onset: {tstamp_ttl}")
                    # if self._logger.handlers:
                    #     self._logger.handlers[0].flush()

                elif ttl == 0 and ttl_state == 1:
                    try:
                        self._ttl_offset_que.put_nowait(tstamp_ttl)
                    except Full:
                        try:
                            self._ttl_offset_que.get_nowait()  # discard oldest
                            self._ttl_offset_que.put_nowait(tstamp_ttl)
                        except Empty:
                            pass
                    # self._logger.debug(f"TTL Offset: {tstamp_ttl}")
                    # if self._logger.handlers:
                    #     self._logger.handlers[0].flush()
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

                try:
                    self._physio_que.put_nowait((tstamp_physio, card, resp))
                except Full:
                    try:
                        self._physio_que.get_nowait()  # discard oldest
                        self._physio_que.put_nowait(
                            (tstamp_physio, card, resp))
                    except Empty:
                        pass
                    except Full:
                        pass

                if tstamp_physio0 is not None:
                    td = tstamp_physio - tstamp_physio0
                    if td > 2.5 / self._sample_freq:
                        self._logger.warning(
                            f"Large time gap detected in physio data: "
                            f"{td:.3f} sec"
                        )
                tstamp_physio0 = tstamp_physio

                rec_delay = np.mean(rec_delays) if rec_delays else 0.0
                next_rec += physio_rec_interval

            if cmd_pipe is not None and cmd_pipe.poll(timeout=0):
                cmd = cmd_pipe.recv()
                self._logger.debug(f"Receive {cmd} in read_signal_loop.")
                if cmd == "QUIT":
                    cmd_pipe.send("END")
                    break

            time.sleep(0.0005)
        self._queue_lock.release()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        try:
            self._logger.debug("Deleting NumatoGPIORecording instance.")
            # Avoid potential X server conflicts during cleanup
            if hasattr(self, "_sig_ser") and self._sig_ser:
                self._sig_ser.close()
                self._sig_ser = None
        except Exception:
            # Ignore all cleanup errors to prevent X server conflicts
            pass


# %% DummyRecording ===========================================================
class DummyRecording:
    """Dummy class for physio recording"""

    def __init__(
        self,
        ttl_onset_que=None,
        ttl_offset_que=None,
        physio_que=None,
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
        self._queue_lock = Lock()

        self.set_sim_data(sample_freq, sim_card_f, sim_resp_f)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_sim_data(self, sample_freq, sim_card_f=None, sim_resp_f=None):
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
                    sim_data_len = min(sim_data_len, len(self._sim_card))
                except Exception as e:
                    errstr = str(e) + "\n" + traceback.format_exc()
                    self._logger.error(f"Error reading {sim_card_f}: {errstr}")

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
        if (
            self._ttl_onset_que is None
            or self._ttl_offset_que is None
            or self._physio_que is None
        ):
            self._logger.error("Recording queues are not set.")
            return

        self._logger.debug("Start recording in read_signal_loop.")

        physio_rec_interval = 1.0 / self._sample_freq
        rec_delays = deque(maxlen=10)  # Stack to store delays, size = 10
        rec_delay = 0
        next_rec = time.time() + physio_rec_interval
        st_physio_read = 0
        tstamp_physio = None
        tstamp_physio0 = None
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

                try:
                    self._physio_que.put_nowait((tstamp_physio, card, resp))
                except Full:
                    try:
                        self._physio_que.get_nowait()  # discard oldest
                        self._physio_que.put_nowait(
                            (tstamp_physio, card, resp))
                    except Empty:
                        pass
                    except Full:
                        pass

                if tstamp_physio0 is not None:
                    td = tstamp_physio - tstamp_physio0
                    if td > 2.5 / self._sample_freq:
                        self._logger.warning(
                            f"Large time gap detected in physio data: "
                            f"{td:.3f} sec"
                        )
                tstamp_physio0 = tstamp_physio

                self._sim_data_pos += 1
                self._sim_data_pos %= self._sim_data_len

                # self._logger.debug(
                #     f"tstamp_physio={tstamp_physio} card={card}, resp={resp}"
                #     f" at {self._sim_data_pos}"
                # )

                rec_delays.append(time.time() - st_physio_read)
                rec_delay = np.mean(rec_delays) if rec_delays else 0.0
                next_rec += physio_rec_interval

            if cmd_pipe is not None and cmd_pipe.poll(timeout=0):
                cmd = cmd_pipe.recv()
                self._logger.debug(f"Receive {cmd} in read_signal_loop.")
                if cmd == "QUIT":
                    cmd_pipe.send("END")
                    break
                elif cmd == "PULSE":
                    # Add TTL pulse
                    try:
                        self._ttl_onset_que.put_nowait(time.time())
                    except Full:
                        try:
                            self._ttl_onset_que.get_nowait()  # discard oldest
                            self._ttl_onset_que.put_nowait(time.time())
                        except Empty:
                            pass

                    time.sleep(0.0005)  # pulse width

                    try:
                        self._ttl_offset_que.put(time.time())
                    except Full:
                        try:
                            self._ttl_offset_que.get()  # discard oldest
                        except Empty:
                            pass

            time.sleep(0.0005)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_config(self, config):
        self._logger.debug(f"Set config: {config}")
        self._config = config


# %% PlotTTLPhysio ============================================================
class PlotTTLPhysio:
    """View class for displaying TTL and physio recording signals"""

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(
        self,
        controller,
        geometry="610x570+1025+0",
        signal_freq=DEFAULT_SAMPLE_FREQ,
        plot_len_sec=10,
        buf_len_sec=3600,
        disable_close=False,
    ):
        self._logger = logging.getLogger("PlotTTLPhysio")
        self.controller = controller

        # --- Initialize parameters ---
        self.signal_freq = signal_freq
        self._plot_len_sec = plot_len_sec
        self._buf_len_sec = buf_len_sec
        self._is_scanning = False
        self._cmd_pipe = None
        self._timer_interval_ms = int(1000 / 10)
        self._rbuf_lock = None
        self._rbuf = None

        # --- Plot window ---
        self._plt_root = tk.Tk()
        self._disable_close = disable_close

        self._plt_root.title(
            f"Physio signals ({self.controller._recorder_type})"
        )
        self.set_position(geometry)

        # Set the margins in inches
        self._left_margin_inch = 0.3
        self._right_margin_inch = 0.4
        self._top_margin_inch = 0.1
        self._bottom_margin_inch = 0.38

        # initialize plot
        plot_widget = self.init_plot()
        plot_widget.pack(side=tk.TOP, fill="both", expand=True)
        self.reset_plot()

        # config button
        # self.config_button = tk.Button(self._plt_root, text='config',
        #                                command=self.config,
        #                                font=("Helvetica", 10))
        # self.config_button.pack(side=tk.LEFT, anchor=tk.SE)
        self.config_win = None

        # dump button
        dump_dur = str(timedelta(seconds=self._buf_len_sec)).split(".")[0]
        dump_button = tk.Button(
            self._plt_root,
            text=f"dump ({dump_dur})",
            command=self.dump_data,
            font=("Helvetica", 10),
        )
        dump_button.pack(side=tk.RIGHT, anchor=tk.SW)

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
        self._ax_ttl.set_ylabel("Scanner Pulse")
        zeros_data = np.zeros(self._buf_size)
        self._ln_ttl = self._ax_ttl.plot(sig_xi, zeros_data, "k-")
        self._ax_ttl.set_xlim(sig_xi[0], sig_xi[-1])
        self._ax_ttl.set_ylim((-0.1, 1.1))
        self._ax_ttl.set_yticks((0, 1))
        self._ax_ttl.yaxis.set_ticks_position("right")

        # Set card axis
        self._ax_card.clear()
        self._ax_card.set_ylabel("Cardiogram")
        self._ln_card = self._ax_card.plot(
            sig_xi, np.zeros(self._buf_size), "k-"
        )
        self._ax_card.set_xlim(sig_xi[0], sig_xi[-1])
        self._ax_card.yaxis.set_ticks_position("right")

        # Set filtered card axis
        self._ax_card_filtered.clear()
        self._ax_card_filtered.set_ylabel("Cardiogram(flitered)")
        self._ln_card_flitered = self._ax_card_filtered.plot(
            sig_xi, np.zeros(self._buf_size), "k-"
        )
        self._ax_card_filtered.set_xlim(sig_xi[0], sig_xi[-1])
        self._ax_card_filtered.yaxis.set_ticks_position("right")

        # Set Resp axis
        self._ax_resp.clear()
        self._ax_resp.set_ylabel("Respiration")
        self._ln_resp = self._ax_resp.plot(
            sig_xi, np.zeros(self._buf_size), "k-"
        )
        self._ax_resp.set_xlim(sig_xi[0], sig_xi[-1])
        self._ax_resp.yaxis.set_ticks_position("right")
        self._ax_resp.set_xlabel("second")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run(self, cmd_pipe, rbuf_lock, rbuf):
        self._plt_root.deiconify()
        self._cmd_pipe = cmd_pipe
        self._rbuf_lock = rbuf_lock
        self._rbuf = rbuf
        self._running = True

        # Check if the self._rbuf are on the same process
        pid = os.getpid()
        for lab, rb in self._rbuf.items():
            self._rbuf[lab] = rb.validate(pid, read_only=True)

        # Start the update loop using traditional after() calls
        self._plt_root.after(self._timer_interval_ms, self._update)

        # Use traditional mainloop but with a way to exit gracefully
        try:
            self._plt_root.mainloop()
        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            self._logger.error(f"Error in mainloop: {errstr}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def stop_updates(self):
        """Stop the update process"""
        self._running = False
        # Cancel any pending after callbacks
        if hasattr(self, "_update_after_id") and self._update_after_id:
            self._plt_root.after_cancel(self._update_after_id)
            self._update_after_id = None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def start_updates(self):
        """Restart the update process"""
        self._running = True
        self._update()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_plot_signals(self, plot_len_sec):
        """Get signals formatted for plotting.
        This method mimics the functionality from rtp_ttl_physio.py
        to provide a consistent interface for the PlotTTLPhysio class.
        """
        if self._rbuf is None or self._rbuf_lock is None:
            return None

        try:
            with self._rbuf_lock:
                ttl_onsets = self._rbuf["ttl_onsets"].get().copy()
                ttl_offsets = self._rbuf["ttl_offsets"].get().copy()
                card = self._rbuf["card"].get().copy()
                resp = self._rbuf["resp"].get().copy()
                tstamp = self._rbuf["tstamp"].get().copy()
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

        # Prepare data for the specified plot length
        data_len = int(plot_len_sec * self.signal_freq)

        # Get latest data within plot window
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
        """Update timer callback"""
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
            elif msg == "RESIZE":
                geometry = self._cmd_pipe.recv()
                self.set_position(geometry)
            elif msg == "SCAN_ON":
                self._is_scanning = True
            elif msg == "SCAN_OFF":
                self._is_scanning = False
            elif msg == "GET_GEOMETRY":
                geometry = self._plt_root.geometry()
                self._cmd_pipe.send(geometry)
            elif msg == "GET_WINDOW_STATE":
                self._cmd_pipe.send(self._plt_root.wm_state())
            elif msg == "RESET":
                config = self._cmd_pipe.recv()
                for key, value in config.items():
                    if hasattr(self, key):
                        if key == "signal_freq":
                            value = float(value)
                        setattr(self, key, value)
                        self._logger.debug(f"Reset {key} to {value}")
                self.reset_plot()

        try:
            # If the window is hide, no update is done.
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
        xt_ex_mask = [t in plt_xt for t in xt_interval_ex]

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
        # Adjust ylim for the latest adjust_period seconds
        adjust_period = int(np.round(3 * self.signal_freq))

        self._ln_card[0].set_ydata(card)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Adjust ylim
            ymin = max(0, np.floor(np.nanmin(card[-adjust_period:]) / 25) * 25)
            ymax = min(
                1024, np.ceil(np.nanmax(card[-adjust_period:]) / 25) * 25
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
            self._ax_card.set_ylim((ymin, ymax))
        # endregion

        # region: Card filtered
        b = firwin(
            numtaps=41,
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
        self._ln_card_flitered[0].set_ydata(card_filtered)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Adjust ylim
            min_card_filtered = np.nanmin(card_filtered[-adjust_period:])
            ymin = max(0, np.floor(min_card_filtered / 25) * 25)
            max_card_filtered = np.nanmax(card_filtered[-adjust_period:])
            ymax = min(1024, np.ceil(max_card_filtered / 25) * 25)
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
        # endregion

        # region: Resp
        self._ln_resp[0].set_ydata(resp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ymin = max(0, np.floor(np.nanmin(resp) / 25) * 25)
            ymax = min(1024, np.ceil(np.nanmax(resp) / 25) * 25)
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

        if self._is_scanning:
            self._ln_card[0].set_color("r")
            self._ln_resp[0].set_color("b")
        else:
            self._ln_card[0].set_color("k")
            self._ln_resp[0].set_color("k")

        self._canvas.draw()
        self._plt_root.update_idletasks()
        # endregion

        # Set next timer event
        et = (time.time() - st) * 1000
        self._plt_root.after(int(self._timer_interval_ms - et), self._update)

        # Card
        self._ln_card[0].set_ydata(card)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Adjust ylim
            ymin = max(0, np.floor(np.nanmin(card[-adjust_period:]) / 25) * 25)
            ymax = min(
                1024, np.ceil(np.nanmax(card[-adjust_period:]) / 25) * 25
            )
            if ymax - ymin < 50:
                if ymin + 50 < 1024:
                    ymax = ymin + 50
                else:
                    ymin = ymax - 50
        if not np.isnan(ymin) and not np.isnan(ymax):
            self._ax_card.set_ylim((ymin, ymax))

        # Card filtered
        b = firwin(
            numtaps=41,
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
        self._ln_card_flitered[0].set_ydata(card_filtered)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Adjust ylim
            min_card_filtered = np.nanmin(card_filtered[-adjust_period:])
            ymin = max(0, np.floor(min_card_filtered / 25) * 25)
            max_card_filtered = np.nanmax(card_filtered[-adjust_period:])
            ymax = min(1024, np.ceil(max_card_filtered / 25) * 25)
            if ymax - ymin < 50:
                if ymin + 50 < 1024:
                    ymax = ymin + 50
                else:
                    ymin = ymax - 50
        if not np.isnan(ymin) and not np.isnan(ymax):
            self._ax_card_filtered.set_ylim((ymin, ymax))

        # Resp
        self._ln_resp[0].set_ydata(resp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ymin = max(0, np.floor(np.nanmin(resp) / 25) * 25)
            ymax = min(1024, np.ceil(np.nanmax(resp) / 25) * 25)
            if ymax - ymin < 50:
                if ymin + 50 < 1024:
                    ymax = ymin + 50
                else:
                    ymin = ymax - 50
        if not np.isnan(ymin) and not np.isnan(ymax):
            self._ax_resp.set_ylim((ymin, ymax))

        if self._is_scanning:
            self._ln_card[0].set_color("r")
            self._ln_resp[0].set_color("b")
        else:
            self._ln_card[0].set_color("k")
            self._ln_resp[0].set_color("k")

        self._canvas.draw()
        self._plt_root.update_idletasks()

        # Set next timer event
        et = (time.time() - st) * 1000
        after_ms = int(max(10, self._timer_interval_ms - et))
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

        if self._cmd_pipe is None:
            self._cmd_pipe.send("SET_CONFIG")
            self._cmd_pipe.send(conf)

        self.reset_plot()
        self.update_plot_size(None)

        self.config_win.destroy()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def cancel_config(self):
        self.config_win.destroy()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def dump_data(self):
        if hasattr(self, "_cmd_pipe") and self._cmd_pipe:
            self._logger.info("Dumping data...")
            if self._logger.handlers:
                self._logger.handlers[0].flush()
            self._cmd_pipe.send("DUMP")

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


# %% ==========================================================================
class RtPhysio:
    """
    TTL and physiological (cardiogram and respiration) signal recording
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
        geometry="610x570+1025+0",
        disable_close=False,
        **kwargs,
    ):
        """Initialize signal recording class
        Set parameter values and list of serial ports.

        Parameters
        ----------
        buf_len_sec : float, optional
            Length (seconds) of signal recording buffer. The default is 3600s.
        sample_freq : float, optional
            Frequency (Hz) of raw signal data. The default is
            DEFAULT_SAMPLE_FREQ.
        rpc_port : int, optional
            RPC socket server port. The default is 63212.
        rt_mri_main_address_name : list, optional
            RtMRIMain RPC address and socket name.
        geometry : str, optional
            Plot window position. The default is "610x570+1025+0".
        disable_close : bool, optional
            Disable close button. The default is False.
        save_ttl : bool, optional
            Save TTL onsets and offsets times. The default is False.
        debug : bool, optional
            Enable debug mode with simulation data. The default is False.
        sim_data : tuple, optional
            Simulation data (card, resp) for debug mode.
        """
        self._logger = logging.getLogger("RtPhysio")

        # --- Initialize parameters ---
        self.buf_len_sec = buf_len_sec
        self.sample_freq = sample_freq
        self.save_ttl = save_ttl
        self.sim_card_f = None
        self.sim_resp_f = None

        config_path = Path(config_path)
        self.rt_mri_main_com = RPCSocketCom(
            rt_mri_main_address_name, config_path
        )
        self.geometry = geometry
        self.disable_close = disable_close

        # Default plot length if not specified
        self.plot_len_sec = kwargs.get("plot_len_sec", 10)

        # Set state variables
        self.wait_ttl_on = False  # Waiting for TTL to signal scan start.
        self._plot = None  # View class of signal plot
        self._recorder_type = None  # Signal recorder type

        # Queues to retrieve recorded data from a recorder process
        self._ttl_onset_que = Queue(maxsize=512)
        self._ttl_offset_que = Queue(maxsize=512)
        self._physio_que = Queue(maxsize=512)

        # Initializing recording process variables
        self._rec_proc = None  # Signal recording process
        self._rec_proc_pipe = None

        # Clear shared memory buffers
        dir_path = Path("/dev/shm")
        for rm_f in dir_path.glob("rtmri_physio_*"):
            try:
                rm_f.unlink()
            except Exception:
                pass

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

        # Set device and start recording
        self.set_device(device)

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
    def set_device(self, device=None):
        """Change or reset recording device."""
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
                    recorder_type = dev
                    break

                elif dev == "Dummy":
                    if (
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
                "No suitable device found. Defaulting to NULL."
            )
            return

        if recorder_type == self._recorder_type:
            self._logger.debug(f"Device {recorder_type} is already in use.")
            return
        else:
            self._recorder_type = recorder_type

        if self._recorder_type is not None:
            self._logger.debug(
                f"Stopping current recorder: {self._recorder_type}"
            )
            try:
                # Stop recording process
                self.stop_recording()
                # Brief pause for cleanup
                time.sleep(0.5)
            except Exception as e:
                errstr = str(e) + "\n" + traceback.format_exc()
                self._logger.error(
                    f"Error stopping current recorder: {errstr}")

        # Start new recorder process
        self._logger.debug(f"Starting recorder: {self._recorder_type}")
        try:
            # Brief pause before starting new recorder
            self.start_recording()
            # Plot continues running with new device data
        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            self._logger.error(
                f"Error starting recorder {self._recorder_type}: {errstr}"
            )

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _create_recorder(self, device_type):
        if device_type == "Numato":
            return NumatoGPIORecording(
                ttl_onset_que=self._ttl_onset_que,
                ttl_offset_que=self._ttl_offset_que,
                physio_que=self._physio_que,
                sample_freq=self.sample_freq,
            )
        elif device_type == "GE":
            return None
        elif device_type == "Dummy":
            return DummyRecording(
                ttl_onset_que=self._ttl_onset_que,
                ttl_offset_que=self._ttl_offset_que,
                physio_que=self._physio_que,
                sim_card_f=self.sim_card_f,
                sim_resp_f=self.sim_resp_f,
                sample_freq=self.sample_freq,
            )
        elif device_type == "NULL":
            return None
        else:
            self._logger.error(f"Unknown device type: {device_type}")
            return None

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
        # Initialize ring buffers if they don't exist, or reset existing ones
        if not hasattr(self, "_rbuf") or self._rbuf is None:
            self._init_ring_buffers()

        # Empty _physio_ques before starting a new recording process
        # while not self._physio_que.empty():
        #     try:
        #         self._physio_que.get_nowait()
        #     except Exception:
        #         break

        gc.disable()

        self._rec_proc_pipe, cmd_pipe = Pipe()
        self._rec_proc = Process(target=self._run_recording, args=(cmd_pipe,))

        self._rec_proc.start()

        # update plot
        if hasattr(self, "_plot_proc_pipe") and self._plot_proc_pipe:
            config = {
                "signal_freq": self.sample_freq,
            }
            self._logger.debug(f"Sending plot config: {config}")
            self._plot_proc_pipe.send("RESET")
            self._plot_proc_pipe.send(config)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def is_recording(self):
        return (
            hasattr(self, "_rec_proc") and
            self._rec_proc is not None and
            self._rec_proc.is_alive()
        )

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def stop_recording(self):
        if not self.is_recording():
            return

        self._rec_proc_pipe.send("QUIT")
        self._rec_proc.join(3)
        self._rec_proc.terminate()

        # Clean up pipe to prevent EOFError
        if hasattr(self, "_rec_proc_pipe") and self._rec_proc_pipe:
            try:
                self._rec_proc_pipe.close()
            except Exception:
                pass
            self._rec_proc_pipe = None

        del self._rec_proc
        self._rec_proc = None
        gc.enable()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def open_plot(self):
        if self._plot is None:
            self._plot = PlotTTLPhysio(
                self,
                geometry=self.geometry,
                signal_freq=self.sample_freq,
                plot_len_sec=self.plot_len_sec,
                disable_close=self.disable_close,
            )

        self._plot_proc_pipe, cmd_pipe = Pipe()
        self._plot_proc = Process(
            target=self._plot.run,
            args=(cmd_pipe, self._rbuf_lock, self._rbuf),
        )
        self._plot_proc.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def stop_plot(self):
        if not hasattr(self, "_plot_proc") or self._plot_proc is None:
            return

        if self._plot_proc.is_alive():
            try:
                # Send stop message to exit gracefully
                self._plot_proc_pipe.send("QUIT")
                # Wait briefly for graceful exit
                self._plot_proc.join(timeout=1.0)
            except (BrokenPipeError, EOFError):
                # Pipe is already broken, process might be dead
                pass

            # If still alive, terminate forcefully
            if self._plot_proc.is_alive():
                self._plot_proc.terminate()
                self._plot_proc.join(timeout=1.0)

            # Final kill if needed
            if self._plot_proc.is_alive():
                self._plot_proc.kill()
                self._plot_proc.join(timeout=0.5)

        # Clean up pipe resources
        if hasattr(self, "_plot_proc_pipe") and self._plot_proc_pipe:
            try:
                self._plot_proc_pipe.close()
            except Exception:
                pass
            self._plot_proc_pipe = None

        # Clean up process reference
        if hasattr(self, "_plot_proc"):
            del self._plot_proc
            self._plot_proc = None

        # Mark plot object for recreation
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
        self.wait_ttl_on = False
        self.scan_onset = 0
        device_type = self._recorder_type

        # Open the device
        recorder = self._create_recorder(device_type)
        if recorder is None:
            self._logger.info("No recorder is available.")
            return

        # Check if the self._rbuf are on the same process
        pid = os.getpid()
        for lab, rbuf in self._rbuf.items():
            self._rbuf[lab] = rbuf.validate(pid)

        # Start the reading process.
        # Data is shared with the main process via queues.
        _recorder_pipe, cmd_pipe_recorder = Pipe()
        _read_proc = Process(
            target=recorder.read_signal_loop,
            kwargs={"cmd_pipe": cmd_pipe_recorder},
        )
        _read_proc.start()

        tstamp0 = None

        # Queue reading loop
        while True:
            if cmd_pipe.poll():
                cmd = cmd_pipe.recv()
                if cmd == "QUIT":
                    self._logger.debug(
                        "Receive QUIT in _run_recording. Break recording loop."
                    )
                    break
                elif cmd == "PULSE":
                    _recorder_pipe.send("PULSE")
                elif cmd == "WAIT_TTL_ON":
                    self.wait_ttl_on = True
                elif cmd == "WAIT_TTL_OFF":
                    self.wait_ttl_off = False

            if not self._ttl_onset_que.empty():
                while not self._ttl_onset_que.empty():
                    ttl_onsets = self._ttl_onset_que.get()
                    if self.wait_ttl_on:
                        self.scan_onset = ttl_onsets
                        self.wait_ttl_on = False
                    with self._rbuf_lock:
                        self._rbuf["ttl_onsets"].append(ttl_onsets)
                    # self._logger.debug(
                    #     f"TTL onsets: {ttl_onsets}")
                    # if self._logger.handlers:
                    #     self._logger.handlers[0].flush()

            if not self._ttl_offset_que.empty():
                while not self._ttl_offset_que.empty():
                    ttl_offsets = self._ttl_offset_que.get()
                    with self._rbuf_lock:
                        self._rbuf["ttl_offsets"].append(ttl_offsets)
                    # self._logger.debug(
                    #     f"TTL offsets: {ttl_offsets}")
                    # if self._logger.handlers:
                    #     self._logger.handlers[0].flush()

            drained = []
            while True:
                try:
                    drained.append(self._physio_que.get_nowait())
                except Empty:
                    break

                if tstamp0 is not None:
                    td = drained[-1][0] - tstamp0
                    if td > 2.5 / self.sample_freq:
                        self._logger.warning(
                            f"Large time gap detected in physio data: "
                            f"{td:.3f} sec"
                        )
                tstamp0 = drained[-1][0]

            if drained:
                with self._rbuf_lock:
                    for tstamp, card, resp in drained:
                        self._rbuf["card"].append(card)
                        self._rbuf["resp"].append(resp)
                        self._rbuf["tstamp"].append(tstamp)

            # time.sleep(0.1 / self.sample_freq)

        # --- end loop ---

        # Stop recording process
        cmd_pipe.send("END_RECORDING")
        _recorder_pipe.send("QUIT")
        if _recorder_pipe.poll(timeout=3):
            _recorder_pipe.recv()
        _read_proc.join(1)
        _read_proc.terminate()

        del recorder

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def dump(self):
        if not self.is_recording():
            # No recording
            return None

        # Check if the self._rbuf are on the same process
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

        # Show actual sampling frequency
        # self._logger.debug(
        #     "Actual physio sampling rate: "
        #     f"{1 / np.mean(np.diff(tstamp)):.2f} Hz"
        # )

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
        fname_fmt="./physio.tsv",
        resample_regular_interval=True,
        nosignal_nosave=True,
    ):
        # region: Get data ----------------------------------------------------
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
                if self.scan_onset > 0 and self.scan_onset > onset:
                    onset = self.scan_onset - 1
                    # Save from 1 second before the scan onset

            # Always ensure we don't exceed buffer capacity or create
            # unreasonable durations. This applies to all onset values,
            # including explicit onset=0
            max_duration = self.buf_len_sec  # Maximum buffer duration
            latest_time = tstamp[-1]
            # Calculate the earliest onset that fits within buffer size
            buffer_limited_onset = latest_time - max_duration

            # Apply buffer limit if onset is too old or would create
            # excessive duration
            if onset < buffer_limited_onset:  # Beyond buffer capacity
                # Note: Removed onset == 0 check since we now filter out
                # invalid TTL events at timestamp 0 at the source
                onset = buffer_limited_onset

            if len_sec is not None:
                if self.scan_onset > 0 and self.scan_onset > onset:
                    offset = self.scan_onset + len_sec + 1
                    # Save until 1 second after the scan offset
                else:
                    offset = onset + len_sec
                if tstamp[-1] < offset + 2:
                    time.sleep(1)
                    continue
            else:
                offset = tstamp[-1]

            break

        # Add debug logging for onset/offset values
        self._logger.info(
            "Save physio data"
            f" from {datetime.fromtimestamp(onset).isoformat()}"
            f" for {timedelta(seconds=offset - onset)}"
        )

        # Additional safety check for reasonable timestamps
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

        # region: Clean TTL onsets
        ttl_onsets = data["ttl_onsets"]

        # Filter out invalid TTL events at timestamp buffer_limited_onset
        ttl_onsets = ttl_onsets[ttl_onsets > buffer_limited_onset]

        # Filter TTL data based on scan onset if available
        if self.scan_onset > 0:
            # If scan onset is set, filter relative to scan start
            ttl_onsets = ttl_onsets[
                (ttl_onsets >= self.scan_onset) & (ttl_onsets < offset)
            ]
        else:
            # If scan onset is not set, use the original onset/offset range
            ttl_onsets = ttl_onsets[
                (ttl_onsets >= onset) & (ttl_onsets < offset)
            ]
        # endregion

        save_path = Path(fname_fmt).resolve().parent
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)

        if self.save_ttl:
            # region: Save TTL onsets -----------------------------------------
            if len(ttl_onsets) > 0:
                ttl_onset_df = pd.DataFrame(columns=("DateTime",))
                ttl_onset_df["DateTime"] = [
                    datetime.fromtimestamp(ons).isoformat()
                    for ons in ttl_onsets
                ]
                # Set TimefromScanOnset only if scan onset is properly set
                if self.scan_onset > 0:
                    ttl_onset_df["TimefromScanOnset"] = (
                        ttl_onsets - self.scan_onset
                    )

                # Set filename
                ttl_onset_stem = Path(fname_fmt).stem.replace(
                    "physio", "PulseOnset"
                )
                ttl_onset_fname = save_path / (ttl_onset_stem + ".tsv")
                # Rotate filename
                ii = 0
                while ttl_onset_fname.is_file():
                    ii += 1
                    ttl_onset_fname = (
                        save_path / ttl_onset_fname.stem + f"_{ii}.tsv"
                    )
                # Save DataFrame to TSV
                ttl_onset_df.to_csv(ttl_onset_fname, sep="\t")
            # endregion

            # region: Save TTL offsets ----------------------------------------
            if len(data["ttl_offsets"]):
                ttl_offsets = data["ttl_offsets"]

                # Filter out invalid TTL events at timestamp
                # buffer_limited_onset
                ttl_offsets = ttl_offsets[ttl_offsets > buffer_limited_onset]

                # Filter TTL offset data based on scan onset if available
                if self.scan_onset > 0:
                    # If scan onset is set, filter relative to scan start
                    ttl_offsets = ttl_offsets[
                        (ttl_offsets >= self.scan_onset)
                        & (ttl_offsets < offset)
                    ]
                else:
                    # If scan onset is not set, use the original onset/offset
                    # range
                    ttl_offsets = ttl_offsets[
                        (ttl_offsets >= onset) & (ttl_offsets < offset)
                    ]
                ttl_offset_df = pd.DataFrame(columns=("DateTime",))
                ttl_offset_df["DateTime"] = [
                    datetime.fromtimestamp(ons).isoformat()
                    for ons in ttl_offsets
                ]
                # Set TimefromScanOnset only if scan onset is properly set
                if self.scan_onset > 0:
                    ttl_offset_df["TimefromScanOnset"] = (
                        ttl_offsets - self.scan_onset
                    )

                # Set filename
                ttl_offset_stem = Path(fname_fmt).stem.replace(
                    "physio", "PulseOffset"
                )
                ttl_offset_fname = save_path / (ttl_offset_stem + ".tsv")
                # Rotate filename
                ii = 0
                while ttl_offset_fname.is_file():
                    ii += 1
                    ttl_offset_fname = (
                        save_path / ttl_offset_fname.stem + f"_{ii}.tsv"
                    )
                # Save DataFrame to TSV
                ttl_offset_df.to_csv(ttl_offset_fname, sep="\t")
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
            # Resample with safety checks
            try:
                # Add safety checks for onset/offset values
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

                # Resample card and resp at regular intervals, ti
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
            tmask = (tstamp >= onset) & (tstamp <= offset)
            save_card = data["card"][tmask]
            save_resp = data["resp"][tmask]
            tstamp = tstamp[tmask]

        # endregion

        if nosignal_nosave:
            # Evaluate the variance of the filtered signal to determine whether
            # the signal should be saved.
            # A low variance indicates that no meaningful recording has
            # occurred.
            # The Resp signal is used for this check as it typically exhibits
            # better quality than the Card signal during actual recordings.
            b = firwin(
                numtaps=41,
                cutoff=3,
                window="hamming",
                pass_zero="lowpass",
                fs=self.sample_freq,
            )
            resp_filtered = lfilter(b, 1, save_resp, axis=0)
            resp_filtered = np.flipud(resp_filtered)
            resp_filtered = lfilter(b, 1, resp_filtered)
            resp_filtered = np.flipud(resp_filtered)
            resp_std = np.nanstd(resp_filtered[40:-40])
            if resp_std < 1:
                # No saving
                msg = "Physio data not saved: "
                msg += "No significant signal changes detected."
                self._logger.error(msg)
                return

        # region: Save physio data --------------------------------------------
        # Format in dataframe
        physio_df = pd.DataFrame()
        physio_df["Time"] = tstamp[: len(save_card)]
        physio_df["cardiac"] = save_card
        physio_df["respiratory"] = save_resp
        physio_df.index.name = "Time"

        # Add trigger
        physio_df["trigger"] = 0
        for ons in ttl_onsets:
            # Find the closest timestamp to the TTL onset
            ttl_idx = np.argmin(np.abs(tstamp - ons))
            if ttl_idx < len(physio_df):
                physio_df.loc[physio_df.index[ttl_idx], "trigger"] = 1

        # Set filename
        tsv_path = Path(fname_fmt).with_suffix(".tsv")
        if tsv_path.is_file():
            # Rotate filename
            ii = 0
            while tsv_path.is_file():
                ii += 1
                tsv_path = save_path / (Path(fname_fmt).stem + f"_{ii}.tsv")
        json_fname = tsv_path.with_suffix(".json")

        # Save
        physio_df.to_csv(tsv_path, sep="\t", index=False, header=False)
        metadata = {
            "SamplingFrequency": self.sample_freq,
            "StartTime": 0,
            "Columns": list(physio_df.columns),
            "Manufacturer": "BIOPAC",
        }
        if self.scan_onset > 0:
            metadata["StartTime"] = onset - self.scan_onset

        with open(json_fname, "w") as f:
            json.dump(metadata, f)

        self._logger.info(
            f"Physio data has been saved to {tsv_path.resolve()}")
        # endregion

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
    def send_dummy_pulse(self):
        if self._recorder_type == "Dummy" and self.is_recording():
            self._logger.debug("Sending dummy pulse")
            self._rec_proc_pipe.send("PULSE")

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
    def RPC_handler(self, call, cmd_pipe=None):
        """
        To return data other than str, pack_data() should be used.
        Note: The RPC process is running in a separate thread, and any
        operations on tkinter must be done in the main thread. To execute
        tkinter operations, use cmd_pipe to send a command to the main thread.
        """

        if call == "ping":
            return pack_data("pong")

        elif call == "WAIT_TTL_ON":
            self._rec_proc_pipe.send("WAIT_TTL_ON")
            self.wait_ttl_on = True

        elif call == "CANCEL_WAIT_TTL":
            self._rec_proc_pipe.send("WAIT_TTL_OFF")
            self.wait_ttl_on = False

        elif call == "TTL_PULSE":
            if self._recorder_type == "Dummy":
                self.send_dummy_pulse()

        elif call == "START_SCAN":
            if hasattr(self, "_plot_proc_pipe") and self._plot_proc_pipe:
                self._plot_proc_pipe.send("SCAN_ON")

        elif call == "END_SCAN":
            if hasattr(self, "_plot_proc_pipe") and self._plot_proc_pipe:
                self._plot_proc_pipe.send("SCAN_OFF")

        elif call == "SHOW":
            if hasattr(self, "_plot_proc_pipe") and self._plot_proc_pipe:
                self._plot_proc_pipe.send("SHOW")

        elif call == "HIDE":
            if hasattr(self, "_plot_proc_pipe") and self._plot_proc_pipe:
                self._plot_proc_pipe.send("HIDE")

        elif call == "GET_GEOMETRY":
            if hasattr(self, "_plot_proc_pipe") and self._plot_proc_pipe:
                self._plot_proc_pipe.send("GET_GEOMETRY")
                geometry = self._plot_proc_pipe.recv()
                return geometry
            else:
                return None

        elif call == "GET_WINDOW_STATE":
            if hasattr(self, "_plot_proc_pipe") and self._plot_proc_pipe:
                self._plot_proc_pipe.send("GET_WINDOW_STATE")
                win_state = self._plot_proc_pipe.recv()
                return win_state
            else:
                return "None"

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
            self.close()

        elif type(call) is tuple:  # Call with arguments
            try:
                if call[0] == "SAVE_PHYSIO_DATA":
                    onset, len_sec, prefix = call[1:]
                    self.save_physio_data(onset, len_sec, prefix)

                elif call[0] == "SET_SCAN_START_BACKWARD":
                    TR = call[1]
                    self.set_scan_onset_bkwd(TR)

                elif call[0] == "SET_GEOMETRY":
                    geometry = call[1]
                    if (
                        hasattr(self, "_plot_proc_pipe")
                        and self._plot_proc_pipe
                    ):
                        self._plot_proc_pipe.send("RESIZE")
                        self._plot_proc_pipe.send(geometry)

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
                    self.sample_freq = float(call[1][2])
                    self.set_device("Dummy")
                    self.send_dummy_pulse()

                elif call[0] == "GET_RICOR":
                    tr, nvol = call[1:]
                    ricor = self.get_retrots(tr, nvol)
                    return ricor

            except Exception as e:
                errstr = str(e) + "\n" + traceback.format_exc()
                self._logger.error(f"Error loading physio files: {errstr}")
                return f"Error: Failed to load physio files - {errstr}"

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def mainloop(self):
        # Open plot
        self.open_plot()

        while True:
            msg = None
            if self._rec_proc_pipe is not None and self._rec_proc_pipe.poll():
                msg = self._rec_proc_pipe.recv()
                if msg == "END_RECORDING":
                    break

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
                    self._plot_proc_pipe.send(conf)

                elif msg == "SET_CONFIG":
                    conf = self._plot_proc_pipe.recv()
                    self.set_config(conf)

                elif msg == "DUMP":
                    tstr = datetime.now().strftime("%Y%m%dT%H%M%S")
                    dump_pattern = f"dump_{tstr}_physio.tsv"
                    dump_filename_fmt = str(self.dump_dir / dump_pattern)
                    self._logger.info(f"Saving dump files to: {self.dump_dir}")
                    self.save_physio_data(
                        fname_fmt=dump_filename_fmt, nosignal_nosave=False
                    )

            if self._rpc_pipe is not None and self._rpc_pipe.poll(timeout=0):
                msg = self._rpc_pipe.recv()
                if msg == "QUIT":
                    self.end()
                    break

            time.sleep(1)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end(self):
        self.stop_recording()
        self.close_plot()
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
        Path(__file__).resolve().parent.parent
        / "log"
        / f"{Path(__file__).resolve().stem}.log"
    )
    parser = argparse.ArgumentParser(description="RT physio")
    parser.add_argument(
        "--sample_freq",
        default=DEFAULT_SAMPLE_FREQ,
        help="sampling frequency (Hz)",
    )
    parser.add_argument(
        "--buf_len_sec", default=3600, help="recording buffer size (second)"
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
        "--geometry", default="610x570+1025+0", help="Plot window position"
    )
    parser.add_argument(
        "--disable_close", action="store_true", help="Disable close button"
    )
    parser.add_argument(
        "--save_ttl",
        action="store_true",
        help="Save TTL onsets and offsets times",
    )
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
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

    # rt_physio model class
    kwarg = {
        "buf_len_sec": buf_len_sec,
        "sample_freq": sample_freq,
        "rpc_socket_name": rpc_socket_name,
        "config_path": config_path,
        "rt_mri_main_address_name": rt_mri_main_address_name,
        "geometry": geometry,
        "disable_close": disable_close,
        "save_ttl": save_ttl,
    }

    if debug:
        card_f = Path("test_data/Physio/P000200/Card_100Hz_ser-12.1D")
        resp_f = Path("test_data/Physio/P000200/Resp_100Hz_ser-12.1D")
        card = np.loadtxt(card_f)
        resp = np.loadtxt(resp_f)
        kwarg["debug"] = debug
        kwarg["sim_data"] = (card, resp)

    rt_physio = RtPhysio(**kwarg)

    # Run mainloop
    rt_physio.mainloop()
