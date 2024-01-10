#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time physiological signal recording class.

@author: mmisaki@libr.net
"""


# %% import ===================================================================
from pathlib import Path
import time
import sys
import traceback
from multiprocessing import Process, Lock, Queue
import re
import logging
import argparse
import warnings
import socket
from datetime import datetime

import numpy as np
import pandas as pd
import serial
from serial.tools.list_ports import comports
from scipy import interpolate
import matplotlib as mpl

from PyQt5 import QtCore, QtWidgets

try:
    from .rpc_socket_server import (
        RPCSocketServer, rpc_send_data, rpc_recv_data, pack_data)
    from .rtp_common import RTP, MatplotlibWindow
    from .rtp_retrots import RtpRetroTS
except Exception:
    from rtpspy.rpc_socket_server import (
        RPCSocketServer, rpc_send_data, rpc_recv_data, pack_data)
    from rtpspy.rtp_common import RTP, MatplotlibWindow
    from rtpspy.rtp_retrots import RtpRetroTS

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

    if not rpc_send_data(sock, data, pkl=pkl, logger=logger):
        errmsg = f'Failed to send {data}'
        if logger:
            logger.error(errmsg)
        else:
            sys.stderr.write(errmsg)
        return

    if get_return:
        data = rpc_recv_data(sock, logger=logger)
        if data is None:
            errmsg = f'Failed to recieve response to {data}'
            if logger:
                logger.error(errmsg)
            else:
                sys.stderr.write(errmsg)

        return data


# %% RingBuffer ===============================================================
class RingBuffer:
    """ Ring buffer """

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
        self._data = np.ones(self.max_size, dtype=self.dtype) * initialize

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def append(self, x):
        """ Append an element """
        try:
            self._data[self._cpos] = x
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
            if self._cpos == 0:
                return self._data
            else:
                data = self._data
                return np.concatenate([data[self._cpos:], data[:self._cpos]])

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errstr = ''.join(
                traceback.format_exception(exc_type, exc_obj, exc_tb))
            sys.stderr.write(errstr)
            return None


# %% ==========================================================================
class TTLPhysioPlot(QtCore.QObject):
    finished = QtCore.pyqtSignal()

    def __init__(self, recorder, main_win=None, win_shape=(600, 600),
                 plot_len_sec=10, disable_close=False):
        super().__init__()

        self.recorder = recorder
        self.main_win = main_win
        self.plot_len_sec = plot_len_sec
        self.disable_close = disable_close

        self._cancel = False

        # Initialize figure
        plt_winname = 'Physio signals'
        self.plt_win = MatplotlibWindow()
        self.plt_win.setWindowTitle(plt_winname)

        # set position
        if main_win is not None:
            main_geom = main_win.geometry()
            x = main_geom.x() + main_geom.width() + 10
            y = main_geom.y() - 26
        else:
            x, y = (0, 0)
        self.plt_win.setGeometry(x, y, win_shape[0], win_shape[1])
        self.init_plot()

        # show window
        self.plt_win.show()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def init_plot(self):
        # Set axis
        self._ax_ttl, self._ax_card, self._ax_resp = \
            self.plt_win.canvas.figure.subplots(3, 1)
        self.plt_win.canvas.figure.subplots_adjust(
            left=0.05, bottom=0.1, right=0.91, top=0.98, hspace=0.35)

        self.reset_plot()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def reset_plot(self):
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
    def show(self):
        self.plt_win.show()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run(self):
        while self.plt_win.isVisible() and not self._cancel:
            try:
                # Get signals
                plt_data = self.recorder.get_plot_signals(self.plot_len_sec)

                if plt_data is None:
                    return

                card = plt_data['card']
                resp = plt_data['resp']
                tstamp = plt_data['tstamp']

                card[tstamp == 0] = 0
                resp[tstamp == 0] = 0

                zero_t = time.time() - self._ln_ttl[0].get_xdata()[-1]
                plt_xt = zero_t + self._ln_ttl[0].get_xdata()

                # Resample
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore",
                                              category=RuntimeWarning)

                        f = interpolate.interp1d(tstamp, card,
                                                 bounds_error=False)
                        card = f(plt_xt)
                        f = interpolate.interp1d(tstamp, resp,
                                                 bounds_error=False)
                        resp = f(plt_xt)

                except Exception:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    errstr = ''.join(
                        traceback.format_exception(exc_type, exc_obj, exc_tb))
                    self._logger.error(errstr)

                # Plot
                # TTL
                ttl = np.zeros_like(card)
                ttl_onset = plt_data['ttl_onset']
                ttl_onset = ttl_onset[~np.isnan(ttl_onset)]
                if len(ttl_onset):
                    ttl_onset = ttl_onset[(ttl_onset >= plt_xt[0]) &
                                          (ttl_onset <= plt_xt[-1])]
                    ttl_onset = ttl_onset - plt_xt[0]
                    ttl_onset_idx = np.array([
                        int(np.round(ons * self.recorder.sample_freq))
                        for ons in ttl_onset], dtype=int)
                    ttl_onset_idx = ttl_onset_idx[(ttl_onset_idx >= 0) &
                                                  (ttl_onset_idx < len(ttl))]
                    ttl[ttl_onset_idx] = 1
                self._ln_ttl[0].set_ydata(ttl)

                # Card
                self._ln_card[0].set_ydata(card)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    ymin = max(0, np.floor(np.nanmin(card) / 50) * 50)
                    ymax = min(1024, np.ceil(np.nanmax(card) / 50) * 50)
                    if ymax - ymin < 50:
                        if ymin + 50 < 1024:
                            ymax = ymin + 50
                        else:
                            ymin = ymax - 50
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

                self.plt_win.canvas.draw()
                time.sleep(1/60)

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                errmsg = '{}, {}:{}'.format(
                    exc_type, exc_tb.tb_frame.f_code.co_filename,
                    exc_tb.tb_lineno)
                errmsg += ' ' + str(e)
                print("!!!Error:{}".format(errmsg))

        self.end_thread()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end_thread(self):
        if self.plt_win.isVisible():
            self.plt_win.close()

        self.finished.emit()

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
            event.accept()


# %% NumatoGPIORecoding class ========================================
class NumatoGPIORecoding(QtCore.QObject):
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
    finished = QtCore.pyqtSignal()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, root, rbuf_names, sport, sample_freq=100,
                 buf_len_sec=1800, debug=False, sim_data=None):
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
        """
        super().__init__()

        self.root = root
        self._logger = logging.getLogger('GPIORecorder')

        # Set parameters
        self._rbuf_names = rbuf_names
        self.sample_freq = sample_freq
        self.buf_len_sec = buf_len_sec
        self.buf_len = int(buf_len_sec * self.sample_freq)

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

        self._rec_proc = None  # TTL recording process
        self._ttl_onsets_que = Queue()
        self._physio_que = Queue()

        self._scan_onset = 0
        self._cancel = False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ getter and setter methods +++
    @property
    def sig_sport(self):
        return self._sig_sport

    @sig_sport.setter
    def sig_sport(self, dev):
        if dev is not None:
            if dev not in self.dict_sig_sport:
                self._logger.error(f"{dev} is not available.")
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
            self._logger.error(f"Failed open {self._sig_sport}")
            self._sig_ser = None
            return False

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errstr = ''.join(
                traceback.format_exception(exc_type, exc_obj, exc_tb))
            self._logger.error(f"Failed to open {self._sig_sport}: {errstr}")
            self._sig_ser = None
            return False

        self._logger.info(f"Open signal port {self._sig_sport}")

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

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _read_signal(self, ttl_que, physio_que, phys_fs):
        ttl_state = 0
        rec_interval = 1.0 / phys_fs
        rec_delay = 0
        next_rec = time.time() + rec_interval
        st_physio_read = 0
        tstamp_physio = None
        while True:
            # Read TTL
            self._sig_ser.reset_output_buffer()
            self._sig_ser.reset_input_buffer()
            self._sig_ser.write(b"gpio read 0\r")
            resp0 = self._sig_ser.read(1024)
            tstamp_ttl = time.time()

            if time.time() >= next_rec-rec_delay:
                st_physio_read = time.time()
                self._sig_ser.reset_output_buffer()
                self._sig_ser.reset_input_buffer()
                # Card
                self._sig_ser.write(b"adc read 1\r")
                resp1 = self._sig_ser.read(25)
                # Resp
                self._sig_ser.write(b"adc read 2\r")
                resp2 = self._sig_ser.read(25)
                tstamp_physio = time.time()
            else:
                tstamp_physio = None

            ma = re.search(r'gpio read 0\n\r(\d)\n', resp0.decode())
            if ma:
                sig = ma.groups()[0]
                ttl = int(sig == '1')
            else:
                ttl = 0

            if ttl_state == 0 and ttl == 1:
                ttl_que.put(tstamp_ttl)
            ttl_state = ttl

            if tstamp_physio is not None:
                # Card
                try:
                    card = float(resp1.decode().split('\n\r')[1])
                except Exception:
                    continue

                # Resp
                try:
                    resp = float(resp2.decode().split('\n\r')[1])
                except Exception:
                    continue

                physio_que.put((tstamp_physio, card, resp))

                ct = time.time()
                rec_delay = time.time() - st_physio_read

                while next_rec-rec_delay < ct:
                    next_rec += rec_interval
            time.sleep(0.001)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def scan_onset(self, onset=None):
        if onset is None:
            return self._scan_onset
        else:
            self._scan_onset = onset

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run(self):
        if not self.open_sig_port():
            return

        self._cancel = False
        self._wait_ttl_on = False
        self.scan_onset(0)

        # Start reading process
        self._rec_proc = Process(
            target=self._read_signal,
            args=(self._ttl_onsets_que, self._physio_que, self.sample_freq))
        self._rec_proc.start()

        # Que reading loop
        while not self._cancel:
            if not self._ttl_onsets_que.empty():
                while not self._ttl_onsets_que.empty():
                    ttl_onset = self._ttl_onsets_que.get()
                    if self._wait_ttl_on:
                        self.scan_onset(ttl_onset)
                        self._wait_ttl_on = False
                    with self.rbuf_lock:
                        self._rbuf['ttl_onset'].append(ttl_onset)

            if not self._physio_que.empty():
                while not self._physio_que.empty():
                    tstamp, card, resp = self._physio_que.get()
                    with self.rbuf_lock:
                        self._rbuf['card'].append(card)
                        self._rbuf['resp'].append(resp)
                        self._rbuf['tstamp'].append(tstamp)
            time.sleep(0.005)

        # Close serial port
        if self._sig_ser is not None and self._sig_ser.is_open:
            self._sig_ser.close()

        self._rec_proc.kill()
        self._rec_proc = None
        self.end_thread()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def stop(self):
        self._cancel = True

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def wait_ttl(self, state):
        if state:
            self._wait_ttl_on = True
            self.scan_onset(0)
        else:
            self._wait_ttl_on = False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end_thread(self):
        self.finished.emit()


# %% ==========================================================================
class RtpPhysio(RTP):
    """
    Recording signals
    """
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, buf_len_sec=1800, sport=None,
                 sample_freq=100, rpc_port=63212,
                 debug=False, sim_data=None, **kwargs):
        """ Initialize real-time signal recording class
        Set parameter values and list of serial ports.

        Parameters
        ----------
        buf_len_sec : float, optional
            Length (seconds) of signal recording buffer. The default is 1800s.
        sample_freq : float, optional
            Frequency (Hz) of raw signal data. The default is 500.
        sport : str
            Serial port name. The default is None.
        """
        super().__init__(**kwargs)
        del self.work_dir

        self._logger = logging.getLogger('RtpPhysio')
        self.buf_len_sec = buf_len_sec
        self.sample_freq = sample_freq
        self.is_scanning = False
        self.plot = None

        # Start RPC socket server
        self.socekt_srv = RPCSocketServer(rpc_port, self.RPC_handler,
                                          socket_name='RtpPhysioSocketServer')

        self._rbuf_names = {'ttl_onset': 'ttl_onset',
                            'card': 'card',
                            'resp': 'resp',
                            'tstamp': 'tstamp'}

        self._retrots = RtpRetroTS()

        # Create recorder
        self._recorder = NumatoGPIORecoding(
            self, self._rbuf_names, sport, sample_freq, buf_len_sec,
            debug=debug, sim_data=sim_data)

        # Start recording
        self.start_recording()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # getter, setter
    @property
    def scan_onset(self):
        return self._recorder.scan_onset()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_config(self):
        return self._recorder.get_config()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_config(self, conf):
        self._recorder.set_config(conf)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def start_recording(self, restart=False):
        self._recTh = QtCore.QThread()
        self._recorder.moveToThread(self._recTh)
        self._recTh.started.connect(self._recorder.run)
        self._recorder.finished.connect(self._recTh.quit)
        self._recTh.start()
        # self._recTh.setPriority(QtCore.QThread.TimeCriticalPriority)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def is_recording(self):
        return hasattr(self, '_recTh') and self._recTh.isRunning()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def stop_recording(self):
        self.close_plot()

        if self.is_recording():
            self._recorder.stop()
            if not self._recTh.wait(1):
                self._recorder.finished.emit()
                self._recTh.wait()

            del self._recTh

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def open_plot(self, main_win=None, win_shape=(450, 450), plot_len_sec=10,
                  disable_close=False):
        if self.plot is None:
            self.plot = TTLPhysioPlot(
                self, main_win, win_shape=win_shape,
                plot_len_sec=plot_len_sec, disable_close=disable_close)

        self.plot.show()
        self._pltTh = QtCore.QThread()
        self.plot.moveToThread(self._pltTh)
        self._pltTh.started.connect(self.plot.run)
        self.plot.finished.connect(self._pltTh.quit)
        self._pltTh.start()
        # self._pltTh.setPriority(QtCore.QThread.LowPriority)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def close_plot(self):
        if not hasattr(self, '_pltTh') or \
                not self._pltTh.isRunning():
            return

        self.plot._cancel = True
        self._pltTh.quit()
        self._pltTh.wait()

        del self.plot
        self.plot = None
        del self._pltTh

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save_physio_data(self, onset=None, len_sec=None, fname_fmt='./{}.1D'):
        # Get data
        data = self.dump()
        tstamp = data['tstamp']

        if onset is None:
            onset = self._recorder.scan_onset()

        if len_sec is not None:
            offset = onset + len_sec
        else:
            offset = tstamp[-1]

        # TTL
        ttl_onset = data['ttl_onset']
        ttl_onset = ttl_onset[~np.isnan(ttl_onset)]
        ttl_onset = ttl_onset[(ttl_onset >= onset) & (ttl_onset < offset)]
        ttl_onset = ttl_onset[ttl_onset < offset]
        ttl_df = pd.DataFrame(columns=('DateTime', 'TimefromScanOnset'))
        ttl_df['DateTime'] = [datetime.fromtimestamp(ons).isoformat()
                              for ons in ttl_onset]
        ttl_df['TimefromScanOnset'] = ttl_onset - onset
        ttl_fname = Path(str(fname_fmt).format('TTLonset'))
        ttl_fname = ttl_fname.parent / (ttl_fname.stem + '.csv')
        ii = 0
        while ttl_fname.is_file():
            ii += 1
            ttl_fname = ttl_fname.parent / (ttl_fname.stem + f"_{ii}" +
                                            ttl_fname.suffix)
        ttl_df.to_csv(ttl_fname)

        dataMask = (tstamp >= onset) & (tstamp <= offset) & \
            np.logical_not(np.isnan(tstamp))
        if dataMask.sum() == 0:
            return

        save_card = data['card'][dataMask]
        save_resp = data['resp'][dataMask]
        tstamp = tstamp[dataMask]

        # Resample
        try:
            ti = np.arange(onset, offset, 1.0/self.sample_freq)
            f = interpolate.interp1d(tstamp, save_card, bounds_error=False)
            save_card = f(ti)
            f = interpolate.interp1d(tstamp, save_resp, bounds_error=False)
            save_resp = f(ti)
        except Exception:
            print(f"tstamp = {tstamp}")

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

        self._logger.info(f"Save physio data in {card_fname} and {resp_fname}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def dump(self):
        if not self.is_recording():
            # No recording
            return None

        # Get data
        with self._recorder.rbuf_lock:
            tstamp = self._recorder._rbuf['tstamp'].get().copy()
            ttl_onset = self._recorder._rbuf['ttl_onset'].get().copy()
            card = self._recorder._rbuf['card'].get().copy()
            resp = self._recorder._rbuf['resp'].get().copy()

        # Remove nan
        ttl_onset = ttl_onset[~np.isnan(ttl_onset)]
        card = card[~np.isnan(tstamp)]
        resp = resp[~np.isnan(tstamp)]
        tstamp = tstamp[~np.isnan(tstamp)]
        if len(tstamp) == 0:
            return None

        # Sort by time stamp
        sidx = np.argsort(tstamp)
        card = card[sidx]
        resp = resp[sidx]
        tstamp = tstamp[sidx]

        data = {'ttl_onset': ttl_onset, 'card': card, 'resp': resp,
                'tstamp': tstamp}
        return data

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_plot_signals(self, plot_len_sec):
        data = self.dump()
        if data is None:
            return None

        data_len = int(plot_len_sec*self.sample_freq)
        for k in ('tstamp', 'card', 'resp'):
            dd = data[k]
            if len(dd) >= data_len:
                data[k] = dd[-data_len:]
            else:
                data[k] = np.ones(data_len) * np.nan
                if len(dd):
                    data[k][-len(dd):] = dd

        data['ttl_onset'] = data['ttl_onset'][
            data['ttl_onset'] >= data['tstamp'][0]]

        return data

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_scan_onset_bkwd(self, TR=None):
        # Read data
        data = self.dump()
        if data is None:
            return

        ttl_onset = data['ttl_onset']
        ttl_onset = ttl_onset[~np.isnan(ttl_onset)]
        if len(ttl_onset) == 0:
            return

        elif len(ttl_onset) == 1:
            ser_onset = ttl_onset[0]
            interval_thresh = 0.0

        else:
            ttl_interval = np.diff(ttl_onset)
            if TR is not None:
                interval_thresh = TR*1.5
            else:
                interval_thresh = np.nanmin(ttl_interval) * 1.5

            long_intervals = np.argwhere(
                ttl_interval > interval_thresh).ravel()
            if len(long_intervals) == 0:
                ser_onset = ttl_onset[0]
            else:
                ser_onset = ttl_onset[long_intervals[-1]+1]

        self._recorder.scan_onset(ser_onset)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_retrots(self, TR, Nvol=np.inf, tshift=0, reasmple_phys_fs=100,
                    timeout=2):
        onset = self._recorder.scan_onset()
        if onset == 0:
            return None

        data = self.dump()
        if data is None:
            return None

        tstamp = data['tstamp'] - onset

        if np.isinf(Nvol):
            Nvol = int(np.nanmax(tstamp) // TR)
        else:
            st = time.time()
            while int(np.nanmax(tstamp) // TR) < Nvol and \
                    time.time() - st < timeout:
                # Waint until Nvol samples
                time.sleep(0.001)
                data = self.dump()
                tstamp = data['tstamp'] - onset

            if int(np.nanmax(tstamp) // TR) < Nvol:
                # ERROR: timeout
                self._logger.error(
                    "Not received enough data to make RETROICOR regressors"
                    f" for {timeout} s.")
                return None

        dataMask = (tstamp >= -TR) & ~np.isnan(tstamp)
        dataMask &= ~np.isnan(data['resp'])
        dataMask &= ~np.isnan(data['card'])
        resp = data['resp'][dataMask]
        card = data['card'][dataMask]
        tstamp = tstamp[dataMask]

        # Resample
        if reasmple_phys_fs is None or reasmple_phys_fs > self.sample_freq:
            physFS = self.sample_freq
        else:
            physFS = reasmple_phys_fs

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            res_t = np.arange(0, Nvol*TR+1.0, 1.0/physFS)
            resp_res_f = interpolate.interp1d(tstamp, resp,
                                              bounds_error=False)
            Resp = resp_res_f(res_t)
            Resp = Resp[~np.isnan(Resp)]

            card_res_f = interpolate.interp1d(tstamp, card,
                                              bounds_error=False)
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
        if call == 'GET_RECORDING_PARMAS':
            return pack_data((self.sample_freq, self._recorder.buf_len))

        elif call == 'WAIT_TTL_ON':
            self._recorder.wait_ttl(True)

        elif call == 'CANCEL_WAIT_TTL':
            self._recorder.wait_ttl(False)

        elif call == 'START_SCAN':
            self.is_scanning = True

        elif call == 'END_SCAN':
            self.is_scanning = False

        elif type(call) is tuple:  # Call with arguments
            if call[0] == 'SAVE_PHYSIO_DATA':
                onset, len_sec, prefix = call[1:]
                self.save_physio_data(onset, len_sec, prefix)

            elif call[0] == 'SET_SCAN_START_BACKWARD':
                TR = call[1]
                self.set_scan_onset_bkwd(TR)

            elif call[0] == 'SET_GEOMETRY':
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
    def end(self):
        self.socekt_srv.shutdown()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        self.end()


# %% main =====================================================================
if __name__ == '__main__':

    # Parse arguments
    LOG_FILE = f'{Path(__file__).stem}.log'
    parser = argparse.ArgumentParser(description='RT physio')
    parser.add_argument('--sample_freq', default=100,
                        help='sampling frequency (Hz)')
    parser.add_argument('--log_file', default=LOG_FILE,
                        help='Log file path')
    parser.add_argument('--rpc_port', default=63212,
                        help='RPC socket server port')
    parser.add_argument('--win_shape', default='450x450',
                        help='Plot window position')
    parser.add_argument('--disable_close', action='store_true',
                        help='Disable close button')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    log_file = args.log_file
    rpc_port = args.rpc_port
    win_shape = args.win_shape
    win_shape = [int(v) for v in win_shape.split('x')]
    disable_close = args.disable_close
    debug = args.debug

    # Logger
    logging.basicConfig(level=logging.INFO,
                        filename=log_file, filemode='a',
                        format='%(name)s - %(levelname)s - %(message)s')

    app = QtWidgets.QApplication(sys.argv)

    # recorder
    if False:  # debug:
        card_f = Path('/data/rt/S20240104150322/Card_500Hz_ser-12.1D')
        resp_f = Path('/data/rt/S20240104150322/Resp_500Hz_ser-12.1D')
        resp = np.loadtxt(resp_f)
        card = np.loadtxt(card_f)
        rtp_physio = RtpPhysio(
            sample_freq=100, rpc_port=rpc_port,
            debug=True, sim_data=(card, resp))
    else:
        rtp_physio = RtpPhysio(
            sample_freq=100, rpc_port=rpc_port)

    # time.sleep(5)
    # rtp_physio._recorder.scan_onset(time.time())
    # time.sleep(30)

    # for nv in range(20, 30):
    #     st = time.time()
    #     rtp_physio.get_retrots(TR=1.5, Nvol=nv, reasmple_phys_fs=100)
    #     print(time.time()-st)
    #     time.sleep(1.5)

    rtp_physio.open_plot()

    sys.exit(app.exec_())

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
