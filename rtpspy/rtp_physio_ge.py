#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time physiological signal recording class.
@author: mmisaki@libr.net

Model class : GENumatoRecoding
View class : TTLPhysioPlot
Controler class: RtpPhysio
"""


# %% import ===================================================================
from pathlib import Path
import time
import sys
import traceback
from multiprocessing import Process, Lock, Queue, Pipe
import re
import logging
import argparse
import warnings
import socket
from datetime import datetime
import struct

import numpy as np
import pandas as pd
import serial
from serial.tools.list_ports import comports
from scipy.interpolate import interp1d
from scipy.signal import lfilter, firwin
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
def call_rt_physio(data, pkl=False, get_return=False, logger=None):
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
        config_f = Path.home() / '.config' / 'rtpspy'
        with open(config_f, "r") as fid:
            rtpspy_config = json.load(fid)
        port = rtpspy_config['RtpPhysioSocketServer_pot']
        sock.connect(('localhost', port))

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
    def __init__(self, data_buffer=None, max_size=None):
        """_summary_
        Args:
            max_size (int): buffer size (number of elements)
        """
        if data_buffer is None:
            assert max_size
            self._max_size = max_size
            self._data = np.array(max_size)
        else:
            self._data = data_buffer
            self._max_size = len(data_buffer)
        self._cpos = 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def append(self, x):
        """ Append an element """
        try:
            self._data[self._cpos] = x
            self._cpos = (self._cpos+1) % self._max_size
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


# %% GENumatoTTLRecoding class ================================================
class GENumatoRecoding():
    """
    Receiving GE physio signals from a serial port and TTL from USB device,
    Numato Lab 8 Channel USB GPIO (
    https://numato.com/product/8-channel-usb-gpio-module-with-analog-inputs/
    ). Read IO0/DIO0 to receive the  TTL signal.
    The device is recognized as 'CDC RS-232 Emulation Demo' or
    'Numato Lab 8 Channel USB GPIO M'
    """
    TTL_DEVICES = ['CDC RS-232 Emulation Demo',
                   'Numato Lab 8 Channel USB GPIO']

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, ttl_onset_que, ttl_offset_que, physio_que,
                 physio_sport=None, physio_sample_freq=200,
                 physio_samples_to_average=5, ttl_sport=None, debug=False,
                 sim_data=None):
        """ Initialize real-time physio recordign class
        Set parameter values and list of serial ports.

        Parameters
        ----------
        """
        self._logger = logging.getLogger('GENumatoRecoding')

        # Set parameters
        self._ttl_onset_que = ttl_onset_que
        self._ttl_offset_que = ttl_offset_que
        self._physio_que = physio_que

        self._physio_sample_freq = physio_sample_freq
        self._physio_samples_to_average = physio_samples_to_average

        # Get available serial ports
        self.dict_sig_sport = self.update_port_list()
        physio_devs = []
        ttl_devs = []
        for dev, desc in self.dict_sig_sport.items():
            if np.any([ttl_dev_name in desc
                       for ttl_dev_name in GENumatoRecoding.TTL_DEVICES]):
                ttl_devs.append(dev)
            else:
                physio_devs.append(dev)

        # Set physio_sport
        if physio_sport is not None and physio_sport in physio_devs:
            self.physio_sport = physio_sport
        else:
            # Search active physio port
            dev = self._search_physio_active_port(physio_devs)
            if dev is None:
                # Set the serial device to the first available one.
                if len(physio_devs):
                    dev = physio_devs[0]
            self.physio_sport = dev
        self._physio_ser = None  # Initialize serial object for physio

        # Set ttl_sport
        if ttl_sport is not None and ttl_sport in ttl_devs:
            self.ttl_sport = ttl_sport
        else:
            if len(ttl_devs):
                self.ttl_sport = ttl_devs[0]
            else:
                self.ttl_sport = None

        self._ttl_ser = None  # Initialize serial object for ttl

        # Set debug mode
        self._debug = debug
        if debug:
            self._sim_card, self._sim_resp = sim_data
            self._sim_data_len = min(len(self._sim_card), len(self._sim_resp))
            self._sim_data_pos = 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ getter and setter methods +++
    @property
    def physio_sport(self):
        return self._physio_sport

    @physio_sport.setter
    def physio_sport(self, dev):
        if dev is not None:
            if dev not in self.dict_sig_sport:
                self._logger.error(f"{dev} is not available.")
                dev = None
        self._physio_sport = dev

    @property
    def ttl_sport(self):
        return self._ttl_sport

    @ttl_sport.setter
    def ttl_sport(self, dev):
        if dev is not None:
            if dev not in self.dict_sig_sport:
                self._logger.error(f"{dev} is not available.")
                dev = None
            elif not np.any(
                [ttl_dev_name in self.dict_sig_sport[dev]
                 for ttl_dev_name in GENumatoRecoding.TTL_DEVICES]):
                self._logger.error(f"{dev} is not Numato 8 Channel USB GPIO.")
                dev = None
        self._ttl_sport = dev

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def update_port_list(self):
        # Get serial port list
        dict_sig_sport = {}
        for pt in comports():
            dict_sig_sport[pt.device] = pt.description

        # Sort dict_sig_sport by key
        dict_sig_sport = {
            k: dict_sig_sport[k]
            for k in sorted(list(dict_sig_sport.keys()))}

        return dict_sig_sport

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def open_physio_port(self):
        # Check the current port and close if it opens
        if self._physio_ser is not None and self._physio_ser.is_open:
            self._physio_ser.close()
            self._physio_ser = None
            time.sleep(1)  # Be sure to close

        try:
            self._physio_ser = serial.Serial(self._physio_sport, 115200,
                                             timeout=0.01)

        except serial.serialutil.SerialException:
            self._logger.error(f"Failed open {self._physio_sport}")
            self._physio_ser = None
            return False

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errstr = ''.join(
                traceback.format_exception(exc_type, exc_obj, exc_tb))
            self._logger.error(
                f"Failed to open: {self._physio_sport}:{errstr}")
            self._physio_ser = None
            return False

        self._logger.info(f"Open pyshio port {self._physio_sport}")

        return True

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def open_ttl_port(self):
        # Check the current port and close if it opens
        if self._ttl_ser is not None and self._ttl_ser.is_open:
            self._ttl_ser.close()
            self._ttl_ser = None
            time.sleep(1)  # Be sure to close

        try:
            self._ttl_ser = serial.Serial(self._ttl_sport, 115200,
                                          timeout=0.001)
            self._ttl_ser.flushOutput()
            self._ttl_ser.write(b"gpio clear 0\r")

        except serial.serialutil.SerialException:
            self._logger.error(f"Failed open {self._ttl_sport}")
            self._sig_ser = None
            return False

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errstr = ''.join(
                traceback.format_exception(exc_type, exc_obj, exc_tb))
            self._logger.error(f"Failed to open {self._ttl_sport}: {errstr}")
            self._sig_ser = None
            return False

        self._logger.info(f"Open signal port {self._ttl_sport}")

        return True

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def read_ttl_loop(self):
        if not self.open_ttl_port():
            return

        ttl_state = 0
        while True:
            # Read TTL
            self._ttl_ser.reset_output_buffer()
            self._ttl_ser.reset_input_buffer()
            self._ttl_ser.write(b"gpio read 0\r")
            resp0 = self._ttl_ser.read(1024)
            tstamp_ttl = time.time()

            ma = re.search(r'gpio read 0\n\r(\d)\n', resp0.decode())
            if ma:
                sig = ma.groups()[0]
                ttl = int(sig == '1')
            else:
                ttl = 0

            if ttl_state == 0 and ttl == 1:
                self._ttl_onset_que.put(tstamp_ttl)
            elif ttl_state == 1 and ttl == 0:
                self._ttl_offset_que.put(tstamp_ttl)
            ttl_state = ttl

            time.sleep(0.001)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _is_packet_good(self, packet):
        try:
            pack = np.frombuffer(packet, dtype=np.int16)
            chksum = np.array(sum(struct.unpack('B' * 10, packet[:10])),
                              dtype=np.int16)
            return pack[5] == chksum
        except Exception:
            return False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _sync_fd(self, packet):
        while not self._is_packet_good(packet) and self._physio_ser.is_open:
            # Shift one byte
            packet = packet[1:]
            try:
                packet += self._physio_ser.read(1)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                errstr = ''.join(
                    traceback.format_exception(exc_type, exc_obj, exc_tb))
                self._logger.error(f"_sync_fd failed: {errstr}")
                raise e
                # self._packet_buf = []
                # continue

        if self._is_packet_good(packet):
            return 0
        else:
            return -1

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def read_physio_loop(self):
        if not self.open_physio_port():
            return

        # --- Initial synchronization to receive the first packet ---
        if self._physio_ser.in_waiting != 0:
            _ = self._physio_ser.read(self._physio_ser.in_waiting)
        self._physio_ser.reset_input_buffer()

        while True:
            packet = self._physio_ser.read(12)
            if len(packet) < 12:
                n_rest = 12 - len(packet)
                while n_rest > 0 and self._physio_ser.is_open:
                    try:
                        packet += self._physio_ser.read(n_rest)
                        n_rest = 12 - len(packet)
                    except Exception:
                        continue
            if not self._physio_ser.is_open:
                # Serial port is closed or recording is cancelled.
                return
            if self._sync_fd(packet) != 0:
                continue
            break

        self._packet_buf = b''
        GEpacket = struct.unpack('Hhhhhh', packet)
        new_pck_seqn = GEpacket[0]
        if new_pck_seqn == 0:
            self.prev_pck_seqn = 65535
        else:
            self.prev_pck_seqn = new_pck_seqn - 1
        self._packet_buf += packet

        # --- Data recording loop ---
        resp_buf = np.empty(self._physio_samples_to_average)
        card_buf = np.empty(self._physio_samples_to_average)
        len_packet_chunk = 12 * self._physio_samples_to_average
        while True:
            if not self._physio_ser.is_open:
                self._logger.error("Physio serial port is closed.")
                break

            # Read physio_samples_to_average packets
            tstamps = []
            n_read = len_packet_chunk - len(self._packet_buf)
            try:
                self._packet_buf = self._physio_ser.read(n_read)
                tstamps.append(time.time())
            except Exception:
                self._packet_buf = b''

            if len(self._packet_buf) < len_packet_chunk:
                n_rest = len_packet_chunk - len(self._packet_buf)
                while n_rest > 0 and self._physio_ser.is_open:
                    try:
                        self._packet_buf += self._physio_ser.read(n_rest)
                        tstamps.append(time.time())
                    except Exception:
                        continue
                    n_rest = len_packet_chunk - len(self._packet_buf)

                if not self._physio_ser.is_open:
                    break

            # Process packets
            resp_buf[:] = np.nan
            card_buf[:] = np.nan
            for ii in range(self._physio_samples_to_average):
                packet = self._packet_buf[:12]
                self._packet_buf = self._packet_buf[12:]

                if not self._is_packet_good(packet):
                    self._sync_fd(packet)

                GEpacket = struct.unpack('Hhhhhh', packet)
                if ii == 0:
                    stidx = int(GEpacket[0])

                bidx = (int(GEpacket[0])-stidx)
                if bidx < 0:
                    bidx += 65536

                if bidx >= self._physio_samples_to_average:
                    break

                resp_buf[bidx] = float(GEpacket[3])
                card_buf[bidx] = float(GEpacket[4])

            # Average
            resp_ave = np.nanmean(resp_buf)
            card_ave = np.nanmean(card_buf)
            self._physio_que.put((np.mean(tstamps), card_ave, resp_ave))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _search_physio_active_port(self, physio_devs):
        active_dev = None
        for dev in physio_devs:
            try:
                ser = serial.Serial(dev, 115200, timeout=1)
            except serial.serialutil.SerialException:
                continue

            except Exception:
                continue

            try:
                if ser.in_waiting != 0:
                    _ = ser.read(ser.in_waiting)
                ser.reset_input_buffer()

                packet = ser.read(12)
                if len(packet) == 0:
                    ser.close()
                    continue

                if len(packet) < 12:
                    n_rest = 12 - len(packet)
                    while n_rest > 0:
                        try:
                            packet += ser.read(n_rest)
                            n_rest = 12 - len(packet)
                        except Exception:
                            ser.close()
                            continue
                if self._sync_fd(packet) != 0:
                    ser.close()
                    continue

            except Exception:
                ser.close()
                continue

            active_dev = dev
            ser.close()
            break

        return active_dev


# %% TTLPhysioPlot ============================================================
class TTLPhysioPlot(QtCore.QObject):
    """ View class for dispaying TTL and physio recording signals
    """
    finished = QtCore.pyqtSignal()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, recorder, main_win=None, win_shape=(600, 600),
                 plot_len_sec=10, disable_close=False):
        super().__init__()

        self.recorder = recorder
        self.main_win = main_win
        self.plot_len_sec = plot_len_sec
        self.disable_close = disable_close
        self.is_scanning = False

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
        self._ax_ttl, self._ax_card, self._ax_card_filtered, self._ax_resp = \
            self.plt_win.canvas.figure.subplots(4, 1)
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

        # Set filtered card axis
        self._ax_card_filtered.clear()
        self._ax_card_filtered.set_ylabel('Cardiogram(flitered)')
        self._ln_card_flitered = self._ax_card_filtered.plot(
            sig_xi, np.zeros(buf_size), 'k-')
        self._ax_card_filtered.set_xlim(sig_xi[0], sig_xi[-1])
        self._ax_card_filtered.yaxis.set_ticks_position("right")

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
        signal_freq = self.recorder.sample_freq

        while self.plt_win.isVisible() and not self._cancel:
            try:
                # Get signals
                plt_data = self.recorder.get_plot_signals(self.plot_len_sec+1)
                if plt_data is None:
                    continue

                ttl_init_state = plt_data['ttl_init_state']
                ttl_onset = plt_data['ttl_onset']
                ttl_offset = plt_data['ttl_offset']
                card = plt_data['card']
                resp = plt_data['resp']
                tstamp = plt_data['tstamp']

                card[tstamp == 0] = 0
                resp[tstamp == 0] = 0
                zero_t = time.time() - np.max(self._ln_ttl[0].get_xdata())
                tstamp = tstamp - zero_t
                ttl_onset = ttl_onset - zero_t
                ttl_offset = ttl_offset - zero_t
                plt_xt = self._ln_ttl[0].get_xdata()

                # Extend xt (time points) for interpolation
                xt_interval = np.mean(np.diff(plt_xt))
                l_xt_extend = np.arange(
                    -100*xt_interval, 0, xt_interval) + plt_xt[0]
                r_xt_extend = np.arange(
                    plt_xt[-1]+xt_interval, tstamp[-1]+xt_interval,
                    xt_interval)
                xt_interval_ex = np.concatenate([
                    l_xt_extend, plt_xt, r_xt_extend
                ])
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
                    errstr = ''.join(
                        traceback.format_exception(exc_type, exc_obj, exc_tb))
                    self._logger.error(errstr)

                # --- Plot ----------------------------------------------------
                # TTL
                ttl = np.zeros_like(card)
                ttl_onset_plt = []
                ttl_offset_plt = []
                on_off_plt = np.array([])
                change_state = ''
                if len(ttl_onset):
                    ttl_onset_plt = ttl_onset[(ttl_onset >= plt_xt[0]) &
                                              (ttl_onset <= plt_xt[-1])]
                    ttl_onset_plt = ttl_onset_plt - plt_xt[0]
                    on_off_plt = np.concatenate((on_off_plt, ttl_onset_plt))

                if len(ttl_offset):
                    ttl_offset_plt = ttl_offset[(ttl_offset >= plt_xt[0]) &
                                                (ttl_offset <= plt_xt[-1])]
                    ttl_offset_plt = ttl_offset_plt - plt_xt[0]
                    on_off_plt = np.concatenate((on_off_plt, ttl_offset_plt))

                if len(on_off_plt):
                    on_off_state = []
                    if len(ttl_onset_plt):
                        on_off_state += ['+'] * len(ttl_onset_plt)
                    if len(ttl_offset_plt):
                        on_off_state += ['-'] * len(ttl_offset_plt)
                    on_off_state = np.array(on_off_state, dtype='<U1')

                    sidx = np.argsort(on_off_plt).ravel()
                    on_off_time = on_off_plt[sidx]
                    on_off_state = on_off_state[sidx]

                    on_off_idx = np.array([
                        int(np.round(ons * signal_freq))
                        for ons in on_off_time], dtype=int)
                    on_off_idx = on_off_idx[(on_off_idx >= 0) &
                                            (on_off_idx < len(ttl))]

                    last_idx = 0
                    for ii, change_idx in enumerate(on_off_idx):
                        change_idx = max(last_idx+1, change_idx)
                        change_state = on_off_state[ii]
                        if change_state == '-':
                            ttl[last_idx:change_idx] = 1
                            ttl[change_idx:] = 0
                        elif change_state == '+':
                            ttl[last_idx:change_idx] = 0
                            ttl[change_idx:] = 1
                        last_idx = change_idx
                    ttl[last_idx:] = int(change_state == '+')
                else:
                    ttl = np.ones_like(card) * ttl_init_state

                self._ln_ttl[0].set_ydata(ttl)

                # Adjust crad/resp ylim for the latest adjust_period seconds
                adjust_period = 3 * self.recorder.sample_freq

                # Card
                self._ln_card[0].set_ydata(card)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    # Adjust ylim
                    ymin = max(0,
                               np.floor(
                                   np.nanmin(card[-adjust_period:]) / 25) * 25)
                    ymax = min(1024,
                               np.ceil(
                                   np.nanmax(card[-adjust_period:]) / 25) * 25)
                    if ymax - ymin < 50:
                        if ymin + 50 < 1024:
                            ymax = ymin + 50
                        else:
                            ymin = ymax - 50
                self._ax_card.set_ylim((ymin, ymax))

                # Card filtered
                b = firwin(numtaps=41, cutoff=3, window="hamming",
                           pass_zero='lowpass', fs=self.recorder.sample_freq)
                card_ex_filtered = lfilter(b, 1, card_ex)
                card_ex_filtered = np.flipud(card_ex_filtered)
                card_ex_filtered = lfilter(b, 1, card_ex_filtered)
                card_ex_filtered = np.flipud(card_ex_filtered)
                card_filtered = card_ex_filtered[xt_ex_mask]
                self._ln_card_flitered[0].set_ydata(card_filtered)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    # Adjust ylim
                    ymin = max(
                        0, np.floor(
                            np.nanmin(
                                card_filtered[-adjust_period:]) / 25) * 25)
                    ymax = min(
                        1024, np.ceil(
                            np.nanmax(
                                card_filtered[-adjust_period:]) / 25) * 25)
                    if ymax - ymin < 50:
                        if ymin + 50 < 1024:
                            ymax = ymin + 50
                        else:
                            ymin = ymax - 50
                self._ax_card_filtered.set_ylim((ymin, ymax))

                # Resp
                self._ln_resp[0].set_ydata(resp)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    # Adjust ylim
                    ymin = max(0, np.floor(np.nanmin(resp) / 25) * 25)
                    ymax = min(1024, np.ceil(np.nanmax(resp) / 25) * 25)
                    if ymax - ymin < 50:
                        if ymin + 50 < 1024:
                            ymax = ymin + 50
                        else:
                            ymin = ymax - 50
                self._ax_resp.set_ylim((ymin, ymax))

                if self.is_scanning:
                    self._ln_card[0].set_color('r')
                    self._ln_resp[0].set_color('b')
                else:
                    self._ln_card[0].set_color('k')
                    self._ln_resp[0].set_color('k')

                self.plt_win.canvas.draw()
                time.sleep(1/60)

            except Exception:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                errmsg = ''.join(
                    traceback.format_exception(exc_type, exc_obj, exc_tb))
                print(f"!!!Error:{errmsg}")

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


# %% ==========================================================================
class RtpPhysio(RTP):
    """
    Recording signals
    """
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, buf_len_sec=1800, sport=None,
                 sample_freq=100, save_ttl=True, debug=False, sim_data=None,
                 **kwargs):
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
        self.wait_ttl_on = False
        self.plot = None
        self.save_ttl = save_ttl

        # --- Initialize data array shared with plotting process ---
        mmap_f = Path('/dev/shm') / 'scan_onset'
        self._scan_onset = np.memmap(mmap_f, dtype=float, mode='w+',
                                     shape=(1,))
        self._scan_onset[:] = 0

        self._rbuf_names = ['ttl_onset', 'ttl_offset',
                            'card', 'resp', 'tstamp']
        self._rbuf_lock = Lock()
        self.buf_len = self.buf_len_sec * self.sample_freq

        self.data_mmap_files = {}
        for label in self._rbuf_names:
            self.data_mmap_files[label] = Path('/dev/shm') / label
        self._rbuf = self.init_data_array(create=True)

        # --- Queues to retrieve recorded data from a recorder process ---
        self._ttl_onset_que = Queue()
        self._ttl_offset_que = Queue()
        self._physio_que = Queue()

        # --- Create recorder ---
        self._recorder = GENumatoRecoding(
            self._ttl_onset_que, self._ttl_offset_que, self._physio_que,
            debug=debug, sim_data=sim_data)

        # Initializing recording process variables
        self._rec_proc = None  # Signal recording process
        self._rec_proc_pipe = None

        # Start RPC socket server
        self.socekt_srv = RPCSocketServer(self.RPC_handler,
                                          socket_name='RtpPhysioSocketServer')

        self._retrots = RtpRetroTS()

        # Start recording
        self.start_recording()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def init_data_array(self, create=False):
        if create:
            mode = 'w+'
        else:
            mode = 'r+'

        rbuf = {}
        for label, mmap_f in self.data_mmap_files.items():
            mmap_data = np.memmap(mmap_f, dtype=float, mode=mode,
                                  shape=(self.buf_len,))
            if create:
                mmap_data[:] = np.nan
                mmap_data.flush()
            rbuf[label] = RingBuffer(mmap_data)

        return rbuf

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # getter, setter
    @property
    def scan_onset(self):
        return self._scan_onset[0]

    @scan_onset.setter
    def scan_onset(self, onset):
        self._scan_onset[:] = onset

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def start_recording(self, restart=False):
        """ Start recording loop in a separate process """
        self._rec_proc_pipe, cmd_pipe = Pipe()
        self._rec_proc = Process(target=self._run_recording,
                                 args=(cmd_pipe, self._rbuf_lock))
        self._rec_proc.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def is_recording(self):
        return self._rec_proc is not None and self._rec_proc.is_alive()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def stop_recording(self):
        self.close_plot()

        if not self.is_recording():
            return

        self._rec_proc_pipe.send('QUIT')
        self._rec_proc.join(3)
        if self.is_recording():
            self._rec_proc.terminate()
        del self._rec_proc
        self._rec_proc = None

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
    def get_config(self):
        return self._recorder.get_config()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_config(self, conf):
        self._recorder.set_config(conf)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _run_recording(self, cmd_pipe, rbuf_lock):
        self.wait_ttl_on = False
        self.scan_onset = 0

        # Start reading process
        _read_ttl_proc = Process(target=self._recorder.read_ttl_loop)
        _read_ttl_proc.start()

        _read_physio_proc = Process(target=self._recorder.read_physio_loop)
        _read_physio_proc.start()

        # Que reading loop
        while True:
            if cmd_pipe.poll():
                cmd = cmd_pipe.recv()
                if cmd == 'QUIT':
                    break

            if not self._ttl_onset_que.empty():
                while not self._ttl_onset_que.empty():
                    ttl_onset = self._ttl_onset_que.get()
                    if self.wait_ttl_on:
                        self.scan_onset = ttl_onset
                        self.wait_ttl_on = False
                    with rbuf_lock:
                        self._rbuf['ttl_onset'].append(ttl_onset)

            if not self._ttl_offset_que.empty():
                while not self._ttl_offset_que.empty():
                    ttl_offset = self._ttl_offset_que.get()
                    with rbuf_lock:
                        self._rbuf['ttl_offset'].append(ttl_offset)

            if not self._physio_que.empty():
                while not self._physio_que.empty():
                    tstamp, card, resp = self._physio_que.get()
                    with rbuf_lock:
                        self._rbuf['card'].append(card)
                        self._rbuf['resp'].append(resp)
                        self._rbuf['tstamp'].append(tstamp)

            time.sleep(0.5/self.sample_freq)

        # --- end loop ---
        # Stop recording process
        _read_ttl_proc.kill()
        _read_physio_proc.kill()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def dump(self):
        if not self.is_recording():
            # No recording
            return None

        # Get data
        with self._rbuf_lock:
            ttl_onset = self._rbuf['ttl_onset'].get().copy()
            ttl_offset = self._rbuf['ttl_offset'].get().copy()
            tstamp = self._rbuf['tstamp'].get().copy()
            card = self._rbuf['card'].get().copy()
            resp = self._rbuf['resp'].get().copy()

        # Remove nan
        ttl_onset = ttl_onset[~np.isnan(ttl_onset)]
        ttl_offset = ttl_offset[~np.isnan(ttl_offset)]

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

        ttl_onset = np.sort(ttl_onset)
        ttl_offset = np.sort(ttl_offset)

        data = {'ttl_onset': ttl_onset, 'ttl_offset': ttl_offset,
                'card': card, 'resp': resp, 'tstamp': tstamp}
        return data

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save_physio_data(self, onset=None, len_sec=None, fname_fmt='./{}.1D',
                         resample_regular_interval=True):
        # Get data
        data = self.dump()
        if data is None:
            return
        tstamp = data['tstamp']

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
            ttl_onset = data['ttl_onset']
            ttl_onset = ttl_onset[(ttl_onset >= onset) & (ttl_onset < offset)]
            ttl_onset = ttl_onset[ttl_onset < offset]
            ttl_onset_df = pd.DataFrame(
                columns=('DateTime', 'TimefromScanOnset'))
            ttl_onset_df['DateTime'] = [datetime.fromtimestamp(ons).isoformat()
                                        for ons in ttl_onset]
            ttl_onset_df['TimefromScanOnset'] = ttl_onset - onset
            ttl_onset_fname = Path(str(fname_fmt).format('TTLonset'))
            ttl_onset_fname = ttl_onset_fname.parent / \
                (ttl_onset_fname.stem + '.csv')
            ii = 0
            while ttl_onset_fname.is_file():
                ii += 1
                ttl_onset_fname = ttl_onset_fname.parent / \
                    (ttl_onset_fname.stem + f"_{ii}" + ttl_onset_fname.suffix)
            ttl_onset_df.to_csv(ttl_onset_fname)

            ttl_offset = data['ttl_offset']
            ttl_offset = ttl_offset[(ttl_offset >= onset) &
                                    (ttl_offset < offset)]
            ttl_offset = ttl_offset[ttl_offset < offset]
            ttl_offset_df = pd.DataFrame(
                columns=('DateTime', 'TimefromScanOnset'))
            ttl_offset_df['DateTime'] = [
                datetime.fromtimestamp(ons).isoformat()
                for ons in ttl_offset]
            ttl_offset_df['TimefromScanOnset'] = ttl_offset - onset
            ttl_offset_fname = Path(str(fname_fmt).format('TTLoffset'))
            ttl_offset_fname = ttl_offset_fname.parent / \
                (ttl_offset_fname.stem + '.csv')
            ii = 0
            while ttl_offset_fname.is_file():
                ii += 1
                ttl_offset_fname = ttl_offset_fname.parent / \
                    (ttl_offset_fname.stem + f"_{ii}" +
                     ttl_offset_fname.suffix)
            ttl_offset_df.to_csv(ttl_offset_fname)

            self._logger.info(
                f"Save TTL data in {ttl_onset_fname} and {ttl_offset_fname}")

        # --- Physio data ---
        # As the data is resampled, 2 s outside the scan period is included.
        dataMask = (tstamp >= onset-2.0) & (tstamp <= offset+2.0) & \
            np.logical_not(np.isnan(tstamp))
        if dataMask.sum() == 0:
            return

        save_card = data['card'][dataMask]
        save_resp = data['resp'][dataMask]
        tstamp = tstamp[dataMask]

        save_card = self._clean_resamp(save_card, tstamp)
        save_resp = self._clean_resamp(save_resp, tstamp)

        if resample_regular_interval:
            # Resample
            try:
                ti = np.arange(onset, offset, 1.0/self.sample_freq)
                f = interp1d(tstamp, save_card, bounds_error=False)
                save_card = f(ti)
                f = interp1d(tstamp, save_resp, bounds_error=False)
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

        t0 = np.nanmax(data['tstamp']) - plot_len_sec
        data['ttl_onset'] = data['ttl_onset'][data['ttl_onset'] >= t0]
        data['ttl_offset'] = data['ttl_offset'][data['ttl_offset'] >= t0]

        onset = data['ttl_onset'][
            data['ttl_onset'] < data['tstamp'][0]]
        offset = data['ttl_offset'][
            data['ttl_offset'] < data['tstamp'][0]]
        if len(onset):
            if len(offset):
                if onset[-1] > offset[-1]:
                    data['ttl_init_state'] = 1
                else:
                    data['ttl_init_state'] = 0
            else:
                data['ttl_init_state'] = 1
        else:
            data['ttl_init_state'] = 0

        return data

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _clean_resamp(self, v, tstamp=None):
        idx_bad = np.argwhere(np.isnan(v)).ravel()
        if len(idx_bad) == 0:
            return v

        idx_good = np.setdiff1d(np.arange(0, len(v)), idx_bad)
        if idx_good[0] > 0:
            v[:idx_good[0]] = v[idx_good[0]]
        if idx_good[-1] < len(v)-1:
            v[idx_good[-1]:] = v[idx_good[-1]]

        idx_bad = np.argwhere(np.isnan(v)).ravel()
        if len(idx_bad):
            x = np.arange(0, len(v))
            idx_good = np.setdiff1d(x, idx_bad)
            if tstamp is None:
                tstamp = x
            v[idx_bad] = interp1d(
                tstamp[idx_good], v[idx_good], bounds_error=None)(
                    tstamp[idx_bad])

        return v

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

        self.scan_onset = ser_onset

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_retrots(self, TR, Nvol=np.inf, tshift=0, reasmple_phys_fs=100,
                    timeout=2):
        onset = self.scan_onset
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
        if call == 'GET_RECORDING_PARMAS':
            return pack_data((self.sample_freq, self._recorder.buf_len))

        elif call == 'WAIT_TTL_ON':
            self.wait_ttl_on = True

        elif call == 'CANCEL_WAIT_TTL':
            self.wait_ttl_on = False

        elif call == 'START_SCAN':
            if self.plot:
                self.plot.is_scanning = True

        elif call == 'END_SCAN':
            if self.plot:
                self.plot.is_scanning = False

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
    parser = argparse.ArgumentParser(description='RTP physio')
    parser.add_argument('--sample_freq', default=100,
                        help='sampling frequency (Hz)')
    parser.add_argument('--log_file', default=LOG_FILE,
                        help='Log file path')
    parser.add_argument('--win_shape', default='450x450',
                        help='Plot window position')
    parser.add_argument('--disable_close', action='store_true',
                        help='Disable close button')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    log_file = args.log_file
    win_shape = args.win_shape
    win_shape = [int(v) for v in win_shape.split('x')]
    disable_close = args.disable_close
    debug = args.debug

    # Logger
    (level=logging.INFO,
                        filename=log_file, filemode='a',
                        format='%(name)s - %(levelname)s - %(message)s')

    app = QtWidgets.QApplication(sys.argv)

    # recorder
    if debug:
        test_dir = Path(__file__).absolute().parent.parent / 'tests'
        card_f = test_dir / 'Card_100Hz_ser-12.1D'
        resp_f = test_dir / 'Resp_100Hz_ser-12.1D'
        resp = np.loadtxt(resp_f)
        card = np.loadtxt(card_f)
        rtp_ttl_physio = RtpPhysio(
            sample_freq=100, debug=True, sim_data=(card, resp))
    else:
        rtp_ttl_physio = RtpPhysio(
            sample_freq=100)

    rtp_ttl_physio.open_plot()

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
