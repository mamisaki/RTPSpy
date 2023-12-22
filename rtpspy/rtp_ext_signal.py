#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 13:20:48 2018

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import re
from multiprocessing import Lock, Pipe, Process, shared_memory
import time
import sys
import traceback
import warnings

import serial
from serial.tools.list_ports import comports
import numpy as np
from PyQt5 import QtWidgets, QtCore
import matplotlib as mpl
from scipy import interpolate

from rtpspy.rtp_common import RTP, MatplotlibWindow

mpl.rcParams['font.size'] = 8


# %% RtpExtSignal class ======================================================
class RtpExtSignal(RTP):
    """
    External signal recording class

    + Supporting devices
    ++ CDC RS-232 Emulation Demo | Numato Lab 8 Channel USB GPIO M
        These are the same devices but can be recognized with different names.
        Numato Lab 8 Channel USB GPIO
        https://numato.com/product/8-channel-usb-gpio-module-with-analog-inputs/
        Read DIO 0 to recieve scan onset TTL signal.

    To use other serial devices, add the descipiotn retruned by
    serial.tools.list_ports in SUPPORT_DEVS class variable, and write support
    functions in init_onsig_port and read_onsig_port methods.
    """

    SUPPORT_DEVS = ['CDC RS-232 Emulation Demo',
                    'Numato Lab 8 Channel USB GPIO M']
    # Numato Lab 8 ... : Read DIO 0 to recieve scan onset TTL signal.
    #                    Read DIO 1 to read cardiogram signal.
    #                    Read DIO 1 to read respiration signal.

    def __init__(self, sig_port=None, sample_freq=500, plot_len_sec=10,
                 buf_len_sec=1800, rtp_retrots=None, verb=True):
        """
        Parameters
        ----------
        sig_port : str, optional
            USB seriel port name to monitor signals.
            <ust be one of the items listed in SUPPORT_DEVS.
            The default is None.
        sample_freq : float, optional
            Sampling frequency (Hz). The default is 500.
        plot_len_sec : float, optional
            Length (seconds) of signal plot. The default is 10.
        buf_len_sec : float, optional
            Length (seconds) of signal recording buffer.
            The default is 1800s (30m).
        rtp_retrots : RtpRetrots object, optional
            RtpRetrots object instance for making RetroTS regressor.
            The default is None.
        verb : bool, optional
            Verbose flag to print log message. The default is True.

        Internal properties
        self._sig_port_ser : serial.Serial object
            Serial port object
        """

        super().__init__()  # call __init__() in RTP class

        # --- Set parameters ---
        self.sig_port = None  # will be set later in self._init_onsig_port
        self.sample_freq = sample_freq
        self.plot_len_sec = plot_len_sec
        self.buf_len_sec = buf_len_sec
        self.verb = verb

        # --- Set available ports list ---
        self._dict_onsig_port = {}
        self._update_port_list()

        if len(self._dict_onsig_port) and \
                (sig_port is None or
                 sig_port not in self._dict_onsig_port.keys()):
            sig_port = list(self._dict_onsig_port.keys())[0]

        # --- Set self._sig_port_ser ---
        self._sig_port_ser = None
        self._init_onsig_port(sig_port)

        # --- Prepare shared memory ---
        self._rbuf_lock = Lock()
        self._ttl_shm_name = 'TTL'
        self._card_shm_name = 'Card'
        self._resp_shm_name = 'Resp'
        self._tstamp_shm_name = 'TimeStamp'
        self._init_shmem()

        # --- recording status ---
        self._rec_pipe = None
        self._wait_start = False
        self._scanning = False
        self.scan_onset = -1.0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ getter method +++
    @property
    def isOpen(self):
        if self.sig_port is not None and hasattr(self, '_sig_port_ser') and \
                self._sig_port_ser.is_open:
            return True
        else:
            return False

    @property
    def scanning(self):
        return self.is_scan_on()

    @property
    def not_available(self):
        self._update_port_list()
        return len(self._dict_onsig_port) == 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _update_port_list(self):
        # list serial (usb) ports
        for pt in comports():
            for desc in RtpExtSignal.SUPPORT_DEVS:
                if desc in pt.description:
                    if desc in self._dict_onsig_port:
                        num_dev = np.sum(
                            [desc in k for k in self._dict_onsig_port.keys()])
                        desc += f" {num_dev}"
                    self._dict_onsig_port[desc] = pt.device

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _init_onsig_port(self, sig_port):
        if sig_port is None:
            return

        # Delete string after ' : ('
        sig_port = re.sub(r' : \(.+', '', sig_port).rstrip()

        if 'CDC RS-232 Emulation Demo' in sig_port or \
                'Numato Lab 8 Channel USB GPIO M' in sig_port:
            # --- Numato Lab 8 Channel USB GPIO Module ---
            if self.sig_port == sig_port:
                # delete the serial.Serial object to reset
                try:
                    if self.is_recording():
                        self.stop_recording()
                    self._sig_port_ser.close()
                except Exception:
                    pass
                del self._sig_port_ser
                time.sleep(1)

            dev = self._dict_onsig_port[sig_port]
            try:
                self._sig_port_ser = serial.Serial(dev, 115200,
                                                   timeout=0.01)
                self._sig_port_ser.flushOutput()
                self._sig_port_ser.write(b"gpio clear 0\r")
                self.sig_port = sig_port

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                traceback.print_exception(exc_type, exc_obj, exc_tb)

                self.errmsg(e)
                errmsg = f"Failed to open {sig_port}"
                self.errmsg(errmsg)
                if hasattr(self, '_sig_port_ser'):
                    del self._sig_port_ser
                self.sig_port = None

            return

        else:
            self.errmsg(f"{sig_port} is not defined" +
                        " for receiving signals.\n")
            return

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _init_shmem(self):
        """Initialize shared memories"""
        self._close_shmem()
        self._buf_len = self.buf_len_sec * self.sample_freq
        shm_size = self._buf_len * np.dtype(float).itemsize
        with self._rbuf_lock:
            # TTL
            shm = shared_memory.SharedMemory(
                name=self._ttl_shm_name, create=True, size=shm_size)
            shm.close()
            self._ttl_rbuf = RtpExtSignal.RingBuffer(
                self._buf_len, self._ttl_shm_name, 0)
            # Card
            shm = shared_memory.SharedMemory(
                name=self._card_shm_name, create=True, size=shm_size)
            shm.close()
            self._card_rbuf = RtpExtSignal.RingBuffer(
                self._buf_len, self._card_shm_name, 0)
            # Resp
            shm = shared_memory.SharedMemory(
                name=self._resp_shm_name, create=True, size=shm_size)
            shm.close()
            self._resp_rbuf = RtpExtSignal.RingBuffer(
                self._buf_len, self._resp_shm_name, 0)
            # Time stamp
            shm = shared_memory.SharedMemory(
                name=self._tstamp_shm_name, create=True, size=shm_size)
            shm.close()
            self._timestamp_rbuf = RtpExtSignal.RingBuffer(
                self._buf_len, self._tstamp_shm_name, 0)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _close_shmem(self):
        with self._rbuf_lock:
            for shm_name in (self._ttl_shm_name, self._card_shm_name,
                             self._resp_shm_name, self._tstamp_shm_name):
                try:
                    # Delete shared_memory with shm_name if it exists
                    shm = shared_memory.SharedMemory(name=shm_name)
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass

    # /////////////////////////////////////////////////////////////////////////
    class RingBuffer:
        """ Ring buffer in shared memory Array """

        def __init__(self, max_size, shm_name, initialize=np.nan):
            """_summary_
            Args:
                max_size (int): buffer size (number of elements)
            """
            self._cur = 0
            self._max = int(max_size)
            self._shm_name = shm_name
            shm = shared_memory.SharedMemory(name=self._shm_name)
            data = np.ndarray(self._max, dtype=float, buffer=shm.buf)
            data[:] = initialize
            shm.close()

        def append(self, x):
            """ Append an element overwriting the oldest one. """
            shm = shared_memory.SharedMemory(name=self._shm_name)
            data = np.ndarray(self._max, dtype=float, buffer=shm.buf)
            data[self._cur] = x
            self._cur = (self._cur+1) % self._max

        def get(self):
            """ return list of elements in correct order """
            shm = shared_memory.SharedMemory(name=self._shm_name)
            data = np.ndarray(self._max, dtype=float, buffer=shm.buf).copy()
            return np.concatenate([data[self._cur:], data[:self._cur]])

        def __del__(self):
            self.shm.close()
            self.shm.unlink()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def start_recording(self):
        # Disable ui
        if hasattr(self, 'ui_objs'):
            for ui in self.ui_objs:
                ui.setEnabled(False)

        if hasattr(self, '_rec_proc') and self.rec_proc.isRunning():
            return

        self._rec_pipe, cmd_pipe = Pipe()
        self._rec_proc = Process(target=self.run, args=(cmd_pipe,))
        self._rec_proc.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def is_recording(self):
        return hasattr(self, '_rec_proc') and self._rec_proc.is_alive()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def stop_recording(self):
        if self.is_recording():
            self._rec_pipe.send('QUIT')
            self._rec_proc.terminate()
            del self._rec_proc
            del self._rec_pipe

        # Enable ui
        if hasattr(self, 'ui_objs'):
            for ui in self.ui_objs:
                ui.setEnabled(True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def open_signal_plot(self):
        if not self.is_recording():
            return

        self.thPltSignal = QtCore.QThread()
        self.pltSignal = RtpExtSignal.PlotSignal(self, main_win=self.main_win)
        self.pltSignal.moveToThread(self.thPltSignal)
        self.thPltSignal.started.connect(self.pltSignal.run)
        self.pltSignal.finished.connect(self.thPltSignal.quit)
        self.thPltSignal.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def close_signal_plot(self):
        if not hasattr(self, 'thPltSignal') or \
                not self.thPltSignal.isRunning():
            return

        self.pltSignal.abort = True

        self.thPltSignal.quit()
        self.thPltSignal.wait()
        del self.thPltSignal
        del self.pltSignal

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _read_signals(self, cmd_pipe):
        if self._sig_port_ser is None:
            return 0

        try:
            self._sig_port_ser.reset_output_buffer()
            self._sig_port_ser.reset_input_buffer()
            # TTL
            self._sig_port_ser.write(b"gpio read 0\r")
            resp0 = self._sig_port_ser.read(1024)
            self._sig_port_ser.write(b"adc read 1\r")
            resp1 = self._sig_port_ser.read(25)
            self._sig_port_ser.write(b"adc read 2\r")
            resp2 = self._sig_port_ser.read(25)
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
            cmd_pipe.send('TTL_ON')
            cmd_pipe.send(tstamp)
            self._wait_ttl_on = False

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

        with self._rbuf_lock:
            self._ttl_rbuf.append(ttl)
            self._card_rbuf.append(card)
            self._resp_rbuf.append(resp)
            self._timestamp_rbuf.append(tstamp)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run(self, cmd_pipe=None):
        if self._sig_port_ser is None:
            return 0

        self._wait_ttl_on = False

        # +++ chk_cmd +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        def chk_cmd(cmd_pipe):
            # listen command
            if cmd_pipe is not None and cmd_pipe.poll():
                cmd = cmd_pipe.recv()
                if cmd == 'QUIT':
                    return -1

                elif cmd == 'WAIT_TTL_ON':
                    self._wait_ttl_on = True

                elif cmd == 'CANCEL_WAIT_TTL':
                    self._wait_ttl_on = False

                elif cmd == 'START_SAVING':
                    onset = cmd_pipe.recv()
                    self.initiate_saving(onset)

                elif cmd == 'STOP_SAVING':
                    self.saving = False

                elif cmd == 'WRITE_DATA':
                    prefix, len_sec = cmd_pipe.recv()
                    self.write_data(prefix, len_sec)

            return 0

        # Start recording loop
        rec_interval = 1.0 / self.sample_freq
        rec_delay = 0
        next_rec = time.time() + rec_interval

        while chk_cmd(cmd_pipe) == 0:
            if time.time() >= next_rec-rec_delay:
                st = time.time()
                self._read_signals(cmd_pipe)
                rec_delay = time.time() - st

                ct = time.time()
                while next_rec-rec_delay < ct:
                    next_rec += rec_interval

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def reset(self):
        if self._verb:
            msg = "Reset scan status."
            self.logmsg(msg)

        self._wait_start = False
        self._scanning = False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def dump(self):
        if not self.is_recording():
            # No recording
            return None

        # Get data
        with self._rbuf_lock:
            buf_len = self._buf_len
            shm = shared_memory.SharedMemory(name=self._tstamp_shm_name)
            timestamp = np.ndarray(buf_len, dtype=float, buffer=shm.buf).copy()
            shm.close()

            shm = shared_memory.SharedMemory(name=self._ttl_shm_name)
            ttl = np.ndarray(buf_len, dtype=float, buffer=shm.buf).copy()
            shm.close()

            shm = shared_memory.SharedMemory(name=self._card_shm_name)
            card = np.ndarray(buf_len, dtype=float, buffer=shm.buf).copy()
            shm.close()

            shm = shared_memory.SharedMemory(name=self._resp_shm_name)
            resp = np.ndarray(buf_len, dtype=float, buffer=shm.buf).copy()
            shm.close()

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
    def save_data(self, prefix='./{}_ser', onset=None, len_sec=None):
        if onset is None and self.scan_onset < 0:
            errmsg = 'No onset information is available. '
            errmsg += 'Physio data was not saved.'
            self.errmsg(errmsg)

        if len_sec is not None:
            data_len = int(len_sec*self.sample_freq)
        else:
            data_len = None

        data = self.dump()
        card = data['card']
        resp = data['resp']
        tstamp = data['tstamp']

        # Find the onset index
        ons_index = np.argmin(np.abs(tstamp - onset)).ravel()

        if data_len:
            save_card = card[ons_index:ons_index+data_len]
            save_resp = resp[ons_index:ons_index+data_len]
        else:
            save_card = card[ons_index:]
            save_resp = resp[ons_index:]

        card_fname = prefix.format(f'Card_{self.sample_freq}Hz')
        resp_fname = prefix.format(f'Resp_{self.sample_freq}Hz')

        if Path(card_fname).is_file():
            # Add a number to the filename if the file exists.
            prefix0 = Path(prefix)
            ii = 1
            while Path(card_fname).is_file():
                prefix = prefix0.parent / (prefix0.stem + f"_{ii}" +
                                           prefix0.suffix)
                card_fname = str(prefix).format(f'Card_{self.sample_freq}Hz')
                resp_fname = str(prefix).format(f'Resp_{self.sample_freq}Hz')
                ii += 1

        np.savetxt(card_fname, np.reshape(save_card, [-1, 1]), '%.2f')
        np.savetxt(resp_fname, np.reshape(save_resp, [-1, 1]), '%.2f')

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def wait_scan_onset(self):
        if not self._scanning and not self._wait_start and self.is_recording():
            self._rec_pipe.send('WAIT_TTL_ON')
            self._wait_start = True

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def abort_waiting(self):
        if self._wait_start:
            self._rec_pipe.send('CANCEL_WAIT_TTL')
            self._wait_start = False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def is_scan_on(self):
        if not self._scanning and self._wait_start and self.is_recording():
            # Get TTL signal onset
            if self._rec_pipe.poll():
                resp = self._rec_pipe.recv()
                if 'TTL_ON' in resp:
                    self.scan_onset = self._rec_pipe.recv()
                    self._scanning = True
                    self._wait_start = False

        return self._scanning

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def manual_start(self):
        if not self._wait_start:
            return

        self.scan_onset = time.time()
        if self.verb:
            self.logmsg("Manual start")
        self._scanning = True

        self.abort_waiting()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end_reset(self):
        self._wait_start = False
        self._scanning = False
        self.scan_onset = -1.0

    # /////////////////////////////////////////////////////////////////////////
    class PlotSignal(QtCore.QObject):
        finished = QtCore.pyqtSignal()

        def __init__(self, root_th, main_win=None):
            super().__init__()

            self.root_th = root_th
            self.main_win = main_win
            self.abort = False

            # Initialize figure
            plt_winname = 'External signals'
            self.plt_win = MatplotlibWindow()
            self.plt_win.setWindowTitle(plt_winname)

            # set position
            if main_win is not None:
                main_geom = main_win.geometry()
                x = main_geom.x() + main_geom.width() + 10
                y = main_geom.y()
            else:
                x, y = (0, 0)
            self.plt_win.setGeometry(x, y, 350, 350)

            # Set axes
            self._ax_ttl, self._ax_card, self._ax_resp = \
                self.plt_win.canvas.figure.subplots(3, 1)

            self.plt_win.canvas.figure.subplots_adjust(
                    left=0.2, bottom=0.1, right=0.98, top=0.95, hspace=0.35)

            self.init_plot()

            # show window
            self.plt_win.show()

        # ---------------------------------------------------------------------
        def init_plot(self):
            signal_freq = self.root_th.sample_freq
            self.plot_buf_size = int(
                np.round(self.root_th.plot_len_sec * signal_freq))
            plt_xi = np.arange(self.plot_buf_size) * 1.0/signal_freq

            # TTL
            self._ax_ttl.clear()
            self._ax_ttl.set_ylabel('TTL')
            self._ln_ttl = self._ax_ttl.plot(
                plt_xi, np.zeros(self.plot_buf_size), 'k-')
            self._ax_ttl.set_xlim(plt_xi[0], plt_xi[-1])
            self._ax_ttl.set_ylim(-0.1, 1.1)
            self._ax_ttl.set_yticks([0, 1], labels=['0', '1'])

            # Cardiac
            self._ax_card.clear()
            self._ax_card.set_ylabel('Cardiac')
            self._ln_card = self._ax_card.plot(
                plt_xi, np.zeros(self.plot_buf_size), 'k-')
            self._ax_card.set_xlim(plt_xi[0], plt_xi[-1])
            self._ax_card.set_ylim(-1, 1025)

            # Resp
            self._ax_resp.clear()
            self._ax_resp.set_ylabel('Respiration')
            self._ln_resp = self._ax_resp.plot(
                plt_xi, np.zeros(self.plot_buf_size), 'k-')
            self._ax_resp.set_xlim(plt_xi[0], plt_xi[-1])
            self._ax_resp.set_ylim(-1, 1025)

            self._ax_resp.set_xlabel('seconds')

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        def run(self):
            while self.plt_win.isVisible() and not self.abort:
                try:
                    # Get plot data
                    data = self.root_th.dump()
                    if data is None:
                        continue

                    plt_data = {}
                    for k, dd in data.items():
                        if len(dd) >= self.plot_buf_size:
                            plt_data[k] = dd[-self.plot_buf_size:]
                        else:
                            plt_data[k] = np.ones(self.plot_buf_size) * np.nan
                            if len(dd):
                                plt_data[k][-len(dd):] = dd

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
                            warnings.simplefilter("ignore",
                                                  category=RuntimeWarning)
                            f = interpolate.interp1d(
                                tstamp, ttl, bounds_error=False,
                                fill_value="extrapolate")
                            ttl = f(plt_xt)
                            ttl = (ttl > 0.5).astype(int)

                            f = interpolate.interp1d(
                                tstamp, card, bounds_error=False,
                                fill_value="extrapolate")
                            card = f(plt_xt)
                            f = interpolate.interp1d(
                                tstamp, resp, bounds_error=False,
                                fill_value="extrapolate")
                            resp = f(plt_xt)

                    except Exception:
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        errstr = ''.join(
                            traceback.format_exception(
                                exc_type, exc_obj, exc_tb))
                        self.logger.error(errstr)

                    # Plot
                    # TTL
                    self._ln_ttl[0].set_ydata(ttl)

                    # Cardiac
                    self._ln_card[0].set_ydata(card)
                    if self.root_th._scanning:
                        self._ln_card[0].set_color('r')
                    else:
                        self._ln_card[0].set_color('k')
                    self._ax_card.relim()
                    if np.max(card) < 100:
                        self._ax_card.set_ylim([-1, 101])
                    self._ax_card.autoscale_view()

                    # Resp
                    self._ln_resp[0].set_ydata(resp)
                    if self.root_th._scanning:
                        self._ln_resp[0].set_color('b')
                    else:
                        self._ln_resp[0].set_color('k')
                    self._ax_resp.relim()
                    if np.max(resp) < 100:
                        self._ax_resp.set_ylim([-1, 101])
                    self._ax_resp.autoscale_view()

                    self.plt_win.canvas.draw()
                    self.plt_win.canvas.start_event_loop(0.005)

                except Exception:
                    self.init_plot()

                if self.main_win is not None and not self.main_win.isVisible():
                    break

            self.end_thread()

        # ---------------------------------------------------------------------
        def end_thread(self):
            if self.plt_win.isVisible():
                self.plt_win.close()

            self.finished.emit()

            if self.main_win is not None:
                if hasattr(self.main_win, 'chbShowExtSig'):
                    self.main_win.chbShowExtSig.setCheckState(0)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_param(self, attr, val=None, reset_fn=None, echo=False):
        """
        When reset_fn is None, set_param is considered to be called from
        load_parameters function.
        """

        # -- check value --
        if attr == 'sig_port' and reset_fn is None:
            if self.sig_port == val:
                return

            idx = self.ui_sigPort_cmbBx.findText(val, QtCore.Qt.MatchContains)
            if idx == -1:
                return

            if hasattr(self, 'ui_sigPort_cmbBx'):
                self.ui_sigPort_cmbBx.setCurrentIndex(idx)

            if val is not None:
                self._init_onsig_port(val)

            return

        elif attr == 'sample_freq':
            if reset_fn is None and hasattr(self, 'ui_sampFreq_dSpBx'):
                self.ui_sampFreq_dSpBx.setValue(val)

        elif attr == 'plot_len_sec' and reset_fn is None:
            if hasattr(self, 'ui_pltLen_dSpBx'):
                self.ui_pltLen_dSpBx.setValue(val)

        elif attr == 'buf_len_sec' and reset_fn is None:
            if hasattr(self, 'ui_bufLen_dSpBx'):
                self.ui_bufLen_dSpBx.setValue(val)

        elif attr == 'verb':
            if hasattr(self, 'ui_verb_chb'):
                self.ui_verb_chb.setChecked(val)

        elif reset_fn is None:
            # Ignore an unrecognized parameter
            if not hasattr(self, attr):
                self.errmsg(f"{attr} is unrecognized parameter.", no_pop=True)
                return

        # -- Set value --
        setattr(self, attr, val)
        if echo and self.verb:
            print("{}.".format(self.__class__.__name__) + attr, '=',
                  getattr(self, attr))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ui_set_param(self):

        ui_rows = []
        self.ui_objs = []

        # sig_port combobox
        var_lb = QtWidgets.QLabel("USB serial port to receive signals :")
        self.ui_sigPort_cmbBx = QtWidgets.QComboBox()
        devlist = sorted([f"{lab} : ({dev})"
                          for lab, dev in self._dict_onsig_port.items()])
        self.ui_sigPort_cmbBx.addItems(devlist)
        if len(devlist) and self.sig_port is not None:
            try:
                selIdx = list(self._dict_onsig_port).index(self.sig_port)
                self.ui_sigPort_cmbBx.setCurrentIndex(selIdx)
            except ValueError:
                pass

        self.ui_sigPort_cmbBx.activated.connect(
                lambda idx: self.set_param(
                        'sig_port', self.ui_sigPort_cmbBx.currentText(),
                        self.ui_sigPort_cmbBx.setCurrentIndex))
        ui_rows.append((var_lb, self.ui_sigPort_cmbBx))
        self.ui_objs.extend([var_lb, self.ui_sigPort_cmbBx])

        # update port list button
        self.ui_serPortUpdate_btn = QtWidgets.QPushButton('Update port list')
        self.ui_serPortUpdate_btn.clicked.connect(self._update_port_list)
        ui_rows.append((self.ui_serPortUpdate_btn,))
        self.ui_objs.extend([var_lb, self.ui_serPortUpdate_btn])

        # sample_freq
        var_lb = QtWidgets.QLabel("Sampling frequency :")
        self.ui_sampFreq_dSpBx = QtWidgets.QDoubleSpinBox()
        self.ui_sampFreq_dSpBx.setMinimum(1.0)
        self.ui_sampFreq_dSpBx.setMaximum(1000.0)
        self.ui_sampFreq_dSpBx.setSingleStep(10.0)
        self.ui_sampFreq_dSpBx.setDecimals(2)
        self.ui_sampFreq_dSpBx.setSuffix(" Hz")
        self.ui_sampFreq_dSpBx.setValue(self.sample_freq)
        self.ui_sampFreq_dSpBx.valueChanged.connect(
                lambda x: self.set_param('sample_freq', x,
                                         self.ui_sampFreq_dSpBx.setValue))
        ui_rows.append((var_lb, self.ui_sampFreq_dSpBx))
        self.ui_objs.extend([var_lb, self.ui_sampFreq_dSpBx])

        # plot_len_sec
        var_lb = QtWidgets.QLabel("Signal plot length :")
        self.ui_pltLen_dSpBx = QtWidgets.QDoubleSpinBox()
        self.ui_pltLen_dSpBx.setMinimum(1)
        self.ui_pltLen_dSpBx.setSingleStep(1)
        self.ui_pltLen_dSpBx.setDecimals(1)
        self.ui_pltLen_dSpBx.setSuffix(" seconds")
        self.ui_pltLen_dSpBx.setValue(self.plot_len_sec)
        self.ui_pltLen_dSpBx.valueChanged.connect(
                lambda x: self.set_param('plot_len_sec', x,
                                         self.ui_pltLen_dSpBx.setValue))
        ui_rows.append((var_lb, self.ui_pltLen_dSpBx))
        self.ui_objs.extend([var_lb, self.ui_pltLen_dSpBx])

        # buf_len_sec
        var_lb = QtWidgets.QLabel("Recording buffer size :")
        self.ui_bufLen_dSpBx = QtWidgets.QDoubleSpinBox()
        self.ui_bufLen_dSpBx.setMinimum(5)
        self.ui_bufLen_dSpBx.setMaximum(36000)
        self.ui_bufLen_dSpBx.setSingleStep(5)
        self.ui_bufLen_dSpBx.setDecimals(0)
        self.ui_bufLen_dSpBx.setSuffix(" seconds")
        self.ui_bufLen_dSpBx.setValue(self.buf_len_sec)
        self.ui_bufLen_dSpBx.valueChanged.connect(
                lambda x: self.set_param('buf_len_sec', x,
                                         self.ui_bufLen_dSpBx.setValue))
        ui_rows.append((var_lb, self.ui_bufLen_dSpBx))
        self.ui_objs.extend([var_lb, self.ui_bufLen_dSpBx])

        # manual start button
        self.ui_manualStart_btn = QtWidgets.QPushButton()
        self.ui_manualStart_btn.setText('Manual start')
        self.ui_manualStart_btn.setStyleSheet("background-color: rgb(255,0,0)")
        self.ui_manualStart_btn.clicked.connect(self.manual_start)
        ui_rows.append((None, self.ui_manualStart_btn))

        # --- Checkbox row ----------------------------------------------------
        # verb
        self.ui_verb_chb = QtWidgets.QCheckBox("Verbose logging")
        self.ui_verb_chb.setChecked(self.verb)
        self.ui_verb_chb.stateChanged.connect(
                lambda state: setattr(self, 'verb', state > 0))
        self.ui_objs.append(self.ui_verb_chb)

        chb_hLayout = QtWidgets.QHBoxLayout()
        chb_hLayout.addStretch()
        chb_hLayout.addWidget(self.ui_verb_chb)
        ui_rows.append((None, chb_hLayout))

        return ui_rows

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_params(self):
        all_opts = super().get_params()
        excld_opts = ('scan_onset', 'ignore_init')
        sel_opts = {}
        for k, v in all_opts.items():
            if k in excld_opts or k[0] == '_':
                continue
            sel_opts[k] = v

        return sel_opts

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        if self.is_recording():
            self.stop_recording()
        self.close_signal_plot()
        self.abort_waiting()
        self._close_shmem()
