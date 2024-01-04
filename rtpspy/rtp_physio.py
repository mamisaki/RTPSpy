#!/usr/bin/env ipython3
# -*- coding: utf-8 -*-
"""
@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import pickle
import struct
import time
import sys
import traceback

import serial
from serial.tools.list_ports import comports
import numpy as np

from PyQt5 import QtWidgets, QtCore
import matplotlib as mpl

from rtpspy.rtp_common import RTP, RingBuffer, MatplotlibWindow
from rtpspy.rtp_ext_signal import RtpExtSignal

mpl.rcParams['font.size'] = 8


# %% RTP_PHYSIO ===============================================================
class RTP_PHYSIO(RTP):
    """
    Physiological signal receiver for real-time processing
    """

    # These devices will be used for RtpExtSignal
    excl_ports = ['CDC RS-232 Emulation Demo',
                  'Numato Lab 8 Channel USB GPIO M']

    def __init__(self, scan_onset, ser_port=None, sample_freq=200,
                 samples_to_average=5, rtp_retrots=None, plot_len_sec=10,
                 excl_port_list=excl_ports, verb=True):
        """
        Parameters
        ----------
        scan_onset : RtpExtSignal object
            RtpExtSignal that send a recording start signal.
        ser_port : str, optional
            Serial port name. The default is None.
        sample_freq : float, optional
            Frequency (Hz) of raw signal data. The default is 200.
        samples_to_average : TYPE, optional
            Number of samples to average data for smoothing and reducing noise.
            Output frequency will be 'sample_freq/samples_to_average'.
            The default is 5.
        rtp_retrots : RtpRetroTS object, optional
            RtpRetroTS object instance for making RetroTS reggressor.
            The default is None.
        plot_len_sec : float, optional
            Length (seconds) of signal plot. The default is 10.
        verb : bool, optional
            Verbose flag to print log message. The default is True.

        Returns
        -------
        None.


        Internal properties
        -------------------
        self._ser_port : str
            Serial port name. self.ser_port provides getter and setter
            methods to read and set self._ser_port value. The setter method
            closes the current port (self._ser_port) if it opens and initialize
            the port with self.init_serial_port method.
        self._ser : serial.Serial object

        """

        super().__init__()  # call __init__() in RTP class

        # --- Set initial parameters ---
        self.not_available = False  # Flag of no available ports.
        self.scan_onset = scan_onset
        self.sample_freq = sample_freq
        self.samples_to_average = samples_to_average
        self.effective_sample_freq = sample_freq/samples_to_average
        self.rtp_retrots = rtp_retrots
        self.plot_len_sec = plot_len_sec
        self._verb = verb
        delattr(self, 'work_dir')

        # --- Set serial port ---
        # port list
        self.dict_ser_port = {}
        for pt in comports():
            if pt.description in excl_port_list:
                continue
            else:
                self.dict_ser_port[pt.device] = pt.description

        # psuedo port
        ptss = [pp for pp in sorted(list(Path('/dev/pts').glob('*')))
                if 'ptmx' not in str(pp)]
        for pts in ptss:
            if pts.is_char_device():
                pn = pts.name
                self.dict_ser_port[str(pts)] = f"psuedo serial port {pn}"

        self._ser_port = None
        self._ser = None
        if len(self.dict_ser_port):
            if ser_port is None or ser_port not in self.dict_ser_port.keys():
                # Select the first one in the list
                ser_port = list(self.dict_ser_port.keys())[0]

            verb = self._verb
            self._verb = False  # temporary off verb at __init__
            self.init_serial_port(ser_port)
            if self._ser_port is None:
                for ser_port in self.dict_ser_port.keys():
                    self.init_serial_port(ser_port)
                    if self._ser_port is not None:
                        break

            if self._ser_port is None:
                self.not_available = True

            self._verb = verb
        else:
            self.not_available = True

        # --- recording status ---
        self.wait_scan = False
        self.scanning = False

        # --- initialize data list ---
        self.resp_data = []
        self.ecg_data = []

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ getter and setter methods +++
    @property
    def ser_port(self):
        return self._ser_port

    @ser_port.setter
    def ser_port(self, ser_port):
        if ser_port is not None:
            self.init_serial_port(ser_port)
        else:
            # Reset
            if self._ser is not None and self._ser.is_open:
                self._ser.close()
                self._ser = None
                time.sleep(1)

            self._ser_port = None
            self._ser = None

    @property
    def isOpen(self):
        if self._ser_port is not None and self._ser is not None and \
                self._ser.is_open:
            return True
        else:
            return False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def update_port_list(self):

        # get port list
        self.dict_ser_port = {}
        for pt in comports():
            if pt.description in RTP_PHYSIO.excl_ports:
                continue
            else:
                self.dict_ser_port[pt.device] = pt.description

        # get psuedo port list
        ptss = [pp for pp in sorted(list(Path('/dev/pts').glob('*')))
                if 'ptmx' not in str(pp)]
        for pts in ptss:
            if pts.is_char_device():
                pn = pts.name
                self.dict_ser_port[str(pts)] = f"psuedo serial port {pn}"

        # Set the port if the current _ser_port is not valid
        if len(self.dict_ser_port):
            if self._ser_port is None \
                    or self._ser_port not in list(self.dict_ser_port.keys()):
                ser_port = sorted(list(self.dict_ser_port.keys()))[0]
                self.init_serial_port(ser_port)
        else:
            self.not_available = True

        # update the combobox
        devlist = ['{} ({})'.format(dev, desc)
                   for dev, desc in self.dict_ser_port.items()]
        if hasattr(self, 'ui_serPort_cmbBx'):
            self.ui_serPort_cmbBx.clear()
            self.ui_serPort_cmbBx.addItems(devlist)
            if not self.not_available:
                selIdx = np.argwhere(
                    [self._ser_port in lst for lst in devlist])
                self.ui_serPort_cmbBx.setCurrentIndex(selIdx.ravel()[0])

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def init_serial_port(self, ser_port, reset_failed=True):
        if '(' in ser_port:
            ser_port = ser_port.split('(')[0].rstrip()

        if ser_port not in self.dict_ser_port.keys():
            self.errmsg("No port {} exists".format(ser_port))
            return -1

        # Check the current port and close it it opens
        if self._ser is not None and self._ser.is_open:
            self._ser.close()
            self._ser = None
            time.sleep(1)

        try:
            self._ser = serial.Serial(ser_port, 115200, timeout=0.01)
            self._ser_port = ser_port

        except serial.serialutil.SerialException:
            if self._verb:
                self.errmsg("Cannot open {}".format(ser_port))

            if self._ser_port is not None and reset_failed:
                # Return to the previous one
                self.init_serial_port(self._ser_port, reset_failed=False)

            return -1

        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_obj, exc_tb)
            if self._ser_port is not None and reset_failed:
                # Return to the previous one
                self.init_serial_port(self._ser_port, reset_failed=False)

            return -1

        if self._verb:
            self.logmsg(f"Set signal port {self._ser_port}.")

        return 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    class PlotSignal(QtCore.QObject):
        finished = QtCore.pyqtSignal()

        def __init__(self, root, main_win=None):
            super().__init__()

            self.root = root
            self.main_win = main_win
            self.abort = False

            # Initialize figure
            plt_winname = 'Physio'
            self.plt_win = MatplotlibWindow()
            self.plt_win.setWindowTitle(plt_winname)

            # set position
            if main_win is not None:
                main_geom = main_win.geometry()
                x = main_geom.x() + main_geom.width() + 10
                y = main_geom.y() - 26
            else:
                x, y = (0, 0)
            self.plt_win.setGeometry(x, y, 350, 200)

            # Set axis
            self._ax_resp, self._ax_ecg = \
                self.plt_win.canvas.figure.subplots(2, 1)
            self.plt_win.canvas.figure.subplots_adjust(
                    left=0.2, bottom=0.2, right=0.98, top=0.95, hspace=0.35)

            self.init_plot()

            # show window
            self.plt_win.show()

        # ---------------------------------------------------------------------
        def init_plot(self):
            data_buf_size = int(np.round(
                    self.root.plot_len_sec * self.root.sample_freq /
                    self.root.samples_to_average))

            plt_xi = np.arange(data_buf_size) * \
                1.0/self.root.sample_freq*self.root.samples_to_average

            # Resp
            self._ax_resp.clear()
            self._ax_resp.set_ylabel('Respiration')
            self._ln_resp = self._ax_resp.plot(plt_xi, np.zeros(data_buf_size),
                                               'k-')
            self._ax_resp.set_xlim(plt_xi[0], plt_xi[-1])

            # ECG
            self._ax_ecg.clear()
            self._ax_ecg.set_ylabel('ECG')
            self._ln_ecg = self._ax_ecg.plot(plt_xi, np.zeros(data_buf_size),
                                             'k-')
            self._ax_ecg.set_xlim(plt_xi[0], plt_xi[-1])
            self._ax_ecg.set_xlabel('seconds')

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        def run(self):
            while self.plt_win.isVisible() and not self.abort:
                try:
                    # Plot Resp
                    resp = self.root.resp_rbuf.get()
                    self._ln_resp[0].set_ydata(resp)
                    if self.root.scanning:
                        self._ln_resp[0].set_color('b')
                    else:
                        self._ln_resp[0].set_color('k')
                    self._ax_resp.relim()
                    self._ax_resp.autoscale_view()

                    # Plot ECG
                    ecg = self.root.ecg_rbuf.get()
                    self._ln_ecg[0].set_ydata(ecg)
                    if self.root.scanning:
                        self._ln_ecg[0].set_color('r')
                    else:
                        self._ln_ecg[0].set_color('k')
                    self._ax_ecg.relim()
                    self._ax_ecg.autoscale_view()

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
    class Recording(QtCore.QObject):
        finished = QtCore.pyqtSignal()

        def __init__(self, root, main_win=None):
            super().__init__()
            self.root = root
            self.abort = False
            self._ser = self.root._ser

        # ---------------------------------------------------------------------
        def _is_scan_on(self):
            if self.root.scan_onset.is_scan_on():
                self.root.scanning = True
                self.root.wait_scan = False

        # ---------------------------------------------------------------------
        def _is_packet_good(self, packet):
            try:
                pack = np.frombuffer(packet, dtype=np.int16)
                chksum = np.array(sum(struct.unpack('B' * 10, packet[:10])),
                                  dtype=np.int16)
                return pack[5] == chksum
            except Exception:
                return False

        # ---------------------------------------------------------------------
        def _sync_fd(self, packet):
            while not self._is_packet_good(packet) and self._ser.is_open:
                # Shift one byte
                packet = packet[1:]
                try:
                    packet += self._ser.read(1)

                    if self.abort:
                        return
                except Exception as e:
                    raise e
                    self._packet_buf = []
                    continue

            if self._is_packet_good(packet):
                return 0
            else:
                return -1

        # ---------------------------------------------------------------------
        def run(self):

            # --- Initialize buffers and flag ---
            self._packet_buf = b''

            data_buf_size = int(np.round(
                    self.root.plot_len_sec * self.root.sample_freq /
                    self.root.samples_to_average))

            resp_plot_rbuf = RingBuffer(data_buf_size)
            ecg_plot_rbuf = RingBuffer(data_buf_size)

            self.root.resp_rbuf = resp_plot_rbuf
            self.root.ecg_rbuf = ecg_plot_rbuf

            # --- Initial synchronization to get the first packet ---
            self._ser.reset_input_buffer()

            while not self.abort:
                if self.root.wait_scan:
                    self._is_scan_on()

                packet = self._ser.read(12)
                if len(packet) < 12:
                    n_rest = 12 - len(packet)
                    while n_rest > 0 and self._ser.is_open and not self.abort:
                        try:
                            packet += self._ser.read(n_rest)
                            n_rest = 12 - len(packet)
                        except Exception:
                            continue

                if not self._ser.is_open or self.abort:
                    self.end_thread()
                    return

                if self._sync_fd(packet) != 0:
                    continue

                break

            if self.abort:
                self.end_thread()
                return

            GEpacket = struct.unpack('Hhhhhh', packet)
            new_pck_seqn = GEpacket[0]
            if new_pck_seqn == 0:
                self.prev_pck_seqn = 65535
            else:
                self.prev_pck_seqn = new_pck_seqn - 1

            self._packet_buf += packet

            # --- Data recording loop ---
            resp_buf = np.empty(self.root.samples_to_average)
            ecg_buf = np.empty(self.root.samples_to_average)
            len_packet_chunk = 12*self.root.samples_to_average
            while not self.abort:
                if not self._ser.is_open:
                    msg = "!!! Serial port {} is not open.".format(
                            self._ser_port)
                    self.root.errmsg(msg)
                    self.end_thread()
                    break

                # Check scan onset signal
                if self.root.wait_scan:
                    self._is_scan_on()

                # Read samples_to_average packets
                n_read = len_packet_chunk - len(self._packet_buf)
                try:
                    self._packet_buf = self._ser.read(n_read)
                except Exception:
                    self._packet_buf = b''

                if len(self._packet_buf) < len_packet_chunk:
                    # Check scan onset signal
                    if self.root.wait_scan:
                        self._is_scan_on()

                    n_rest = len_packet_chunk - len(self._packet_buf)
                    while n_rest > 0 and self._ser.is_open and not self.abort:
                        try:
                            self._packet_buf += self._ser.read(n_rest)
                        except Exception:
                            continue
                        n_rest = len_packet_chunk - len(self._packet_buf)

                    if not self._ser.is_open or self.abort:
                        self.end_thread()
                        return

                # Process packets
                resp_buf[:] = np.nan
                ecg_buf[:] = np.nan
                for ii in range(self.root.samples_to_average):
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

                    if bidx >= self.root.samples_to_average:
                        break

                    resp_buf[bidx] = float(GEpacket[3])
                    ecg_buf[bidx] = float(GEpacket[4])

                # Average
                resp_ave = np.nanmean(resp_buf)
                ecg_ave = np.nanmean(ecg_buf)

                # Save in ring buffer for plot
                resp_plot_rbuf.append(resp_ave)
                ecg_plot_rbuf.append(ecg_ave)

                # Put data in list
                if self.root.scanning:
                    self.root.resp_data.append(resp_ave)
                    self.root.ecg_data.append(ecg_ave)

            # -- end loop --
            self.end_thread()

        # ---------------------------------------------------------------------
        def end_thread(self):
            self.finished.emit()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def start_recording(self):
        # Disable ui to fix parameters
        if hasattr(self, 'ui_objs'):
            for ui in self.ui_objs:
                ui.setEnabled(False)

        if hasattr(self, 'thRec') and self.thRec.isRunning():
            return

        self.thRec = QtCore.QThread()
        self.rec = RTP_PHYSIO.Recording(self, main_win=self.main_win)
        self.rec.moveToThread(self.thRec)
        self.thRec.started.connect(self.rec.run)
        self.rec.finished.connect(self.thRec.quit)
        self.thRec.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def is_recording(self):
        return hasattr(self, 'thRec') and self.thRec.isRunning()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def stop_recording(self):
        self.close_signal_plot()

        if self.is_recording():
            self.rec.abort = True
            if not self.thRec.wait(1):
                self.rec.finished.emit()
                self.thRec.wait()

            del self.thRec

        if hasattr(self, 'rec'):
            del self.rec

        # Enable ui
        if hasattr(self, 'ui_objs'):
            for ui in self.ui_objs:
                ui.setEnabled(True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def open_signal_plot(self):
        if not self.is_recording():
            return

        self.thPltSignal = QtCore.QThread()
        self.pltSignal = RTP_PHYSIO.PlotSignal(self, main_win=self.main_win)
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

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_retrots(self, TR, Nvol=np.inf, tshift=0, timeout=2):
        if self.rtp_retrots is None:
            cname = self.__class__.__name__
            self.errmsg(f"No retrots object for {cname}.", no_pop=True)
            return

        if not self.isOpen:
            self.errmsg(f"Physhio signal port {self.ser_port} is not opened.",
                        no_pop=True)
            return None

        if np.isinf(Nvol):
            tlen_current = len(self.resp_data)
            Nvol = (tlen_current*self.samples_to_average/self.sample_freq)//TR
            Nvol = int(Nvol)
            if Nvol < 1:
                # ERROR: number of received signal is less than one TR.
                return None

        st = time.time()
        tlen_need = int(Nvol * TR * self.sample_freq/self.samples_to_average)
        while len(self.resp_data) < tlen_need and time.time()-st < timeout:
            time.sleep(self.samples_to_average/self.sample_freq)

        if len(self.resp_data) < tlen_need:
            # ERROR: timeout
            self.errmsg("Not received enough data to make RETROICOR regressors"
                        f" for {timeout} s.", no_pop=True)
            return None

        Resp = self.resp_data[:tlen_need]
        ECG = self.ecg_data[:tlen_need]

        PhysFS = self.sample_freq/self.samples_to_average

        retroTSReg = self.rtp_retrots.do_proc(Resp, ECG, TR, PhysFS, tshift)

        return retroTSReg[:Nvol, :]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def reset(self):
        if self._verb:
            msg = "Reset Resp/ECG data buffer and scan status."
            self.logmsg(msg)

        self.wait_scan = False
        self.scanning = False

        time.sleep(0.01)
        self.resp_data[:] = []
        self.ecg_data[:] = []

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save_data(self, prefix='./{}_scan', len_sec=None):
        if len_sec is not None:
            data_len = int(len_sec*self.effective_sample_freq)
        else:
            data_len = len(self.resp_data)

        save_resp = self.resp_data[:data_len]
        save_ecg = self.ecg_data[:data_len]

        resp_fname = prefix.format('Resp')
        ecg_fname = prefix.format('ECG')

        if Path(resp_fname).is_file():
            # Add a number to the filename if the file exists.
            prefix0 = Path(prefix)
            ii = 1
            while Path(resp_fname).is_file():
                prefix = prefix0.parent / (prefix0.stem + f"_{ii}" +
                                           prefix0.suffix)
                resp_fname = str(prefix).format('Resp')
                ecg_fname = str(prefix).format('ECG')
                ii += 1

        np.savetxt(resp_fname, np.reshape(save_resp, [-1, 1]), '%.2f')
        np.savetxt(ecg_fname, np.reshape(save_ecg, [-1, 1]), '%.2f')

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_param(self, attr, val=None, reset_fn=None, echo=False):
        """
        When reset_fn is None, set_param is considered to be called from
        load_parameters function.
        """

        # -- check value --
        if attr == 'enabled':
            if hasattr(self, 'ui_enabled_rdb'):
                self.ui_enabled_rdb.setChecked(val)

            if hasattr(self, 'ui_objs'):
                for ui in self.ui_objs:
                    ui.setEnabled(val)

        elif attr == 'ser_port':
            if self._ser_port == val:
                return

            if reset_fn is None and hasattr(self, 'ui_serPort_cmbBx'):
                idx = self.ui_serPort_cmbBx.findText(val,
                                                     QtCore.Qt.MatchContains)
                if idx == -1:
                    return

                self.ui_serPort_cmbBx.setCurrentIndex(idx)

        elif attr == '_ser_port':
            if val not in self.dict_ser_port.keys():
                return

            if hasattr(self, 'ui_serPort_cmbBx'):
                idx = self.ui_serPort_cmbBx.findText(val,
                                                     QtCore.Qt.MatchContains)
                if idx == -1:
                    return

                self.ui_serPort_cmbBx.setCurrentIndex(idx)

            self.ser_port = val

        elif attr == 'sample_freq' or attr == 'samples_to_average':
            setattr(self, attr, val)
            self.effective_sample_freq = \
                self.sample_freq/self.samples_to_average
            if hasattr(self, 'ui_effSampFreq_lb'):
                self.ui_effSampFreq_lb.setText(
                        str(self.effective_sample_freq) + ' Hz')
            if reset_fn is None:
                if attr == 'sample_freq':
                    if hasattr(self, 'ui_sampFreq_dSpBx'):
                        self.ui_sampFreq_dSpBx.setValue(val)
                elif attr == 'samples_to_average':
                    if hasattr(self, 'ui_sampAve_spBx'):
                        self.ui_sampAve_spBx.setValue(val)

            return

        elif attr == 'plot_len_sec' and reset_fn is None:
            if hasattr(self, 'ui_pltLen_dSpBx'):
                self.ui_pltLen_dSpBx.setValue(val)

        elif attr == '_verb':
            if hasattr(self, 'ui_verb_chb'):
                self.ui_verb_chb.setChecked(val)

        elif reset_fn is None:
            # Ignore an unrecognized parameter
            if not hasattr(self, attr):
                self.errmsg(f"{attr} is unrecognized parameter.", no_pop=True)
                return

        # -- Set value --
        setattr(self, attr, val)
        if echo:
            print("{}.".format(self.__class__.__name__) + attr, '=',
                  getattr(self, attr))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ui_set_param(self):

        ui_rows = []
        self.ui_objs = []

        # ser_port
        var_lb = QtWidgets.QLabel("Physiological signal serial port :")
        self.ui_serPort_cmbBx = QtWidgets.QComboBox()
        devlist = (['{} ({})'.format(dev, desc)
                    for dev, desc in self.dict_ser_port.items()])
        self.ui_serPort_cmbBx.addItems(devlist)
        if not self.not_available and self._ser_port is not None:
            selIdx = np.argwhere([self._ser_port in lst for lst in devlist])
            self.ui_serPort_cmbBx.setCurrentIndex(selIdx.ravel()[0])
            self.ui_serPort_cmbBx.activated.connect(
                    lambda idx: self.set_param(
                            'ser_port', self.ui_serPort_cmbBx.currentText(),
                            self.ui_serPort_cmbBx.setCurrentIndex))
        ui_rows.append((var_lb, self.ui_serPort_cmbBx))
        self.ui_objs.extend([var_lb, self.ui_serPort_cmbBx])

        # update port list
        self.ui_serPortUpdate_btn = QtWidgets.QPushButton(
                'Update port list')
        self.ui_serPortUpdate_btn.clicked.connect(self.update_port_list)
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

        # samples_to_average
        var_lb = QtWidgets.QLabel("Samples to average :")
        self.ui_sampAve_spBx = QtWidgets.QSpinBox()
        self.ui_sampAve_spBx.setMinimum(1)
        self.ui_sampAve_spBx.setValue(self.samples_to_average)
        self.ui_sampAve_spBx.valueChanged.connect(
                lambda x: self.set_param('samples_to_average', x,
                                         self.ui_sampAve_spBx.setValue))
        ui_rows.append((var_lb, self.ui_sampAve_spBx))
        self.ui_objs.extend([var_lb, self.ui_sampAve_spBx])

        # effective_sample_freq
        var_lb = QtWidgets.QLabel("Effective sampling frequency :")
        self.ui_effSampFreq_lb = QtWidgets.QLabel()
        self.ui_effSampFreq_lb.setText(str(self.effective_sample_freq) + ' Hz')
        ui_rows.append((var_lb, self.ui_effSampFreq_lb))
        self.ui_objs.extend([var_lb, self.ui_effSampFreq_lb])

        # plot_len_sec
        var_lb = QtWidgets.QLabel("Resp/ECG plot length :")
        self.ui_pltLen_dSpBx = QtWidgets.QDoubleSpinBox()
        self.ui_pltLen_dSpBx.setMinimum(0.5)
        self.ui_pltLen_dSpBx.setSingleStep(0.5)
        self.ui_pltLen_dSpBx.setDecimals(1)
        self.ui_pltLen_dSpBx.setSuffix(" seconds")
        self.ui_pltLen_dSpBx.setValue(self.plot_len_sec)
        self.ui_pltLen_dSpBx.valueChanged.connect(
                lambda x: self.set_param('plot_len_sec', x,
                                         self.ui_pltLen_dSpBx.setValue))
        ui_rows.append((var_lb, self.ui_pltLen_dSpBx))
        self.ui_objs.extend([var_lb, self.ui_pltLen_dSpBx])

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
        excld_opts = ('rtp_retrots', 'plot_len_sec', 'scanning', 'ecg_data',
                      'wait_scan', 'resp_data', 'dict_ser_port',
                      'not_available')
        sel_opts = {}
        for k, v in all_opts.items():
            if k in excld_opts:
                continue

            if k == '_ser_port' and (v is None or '/dev/pts' in v):
                continue

            if isinstance(v, Path):
                v = str(v)
            sel_opts[k] = v

        return sel_opts

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        self.stop_recording()
        self.close_signal_plot()

        if self._ser is not None and self._ser.is_open:
            self._ser.close()


# %% ==========================================================================
class RTP_PHYSIO_DUMMY(RTP):
    """ Dummy class of RTP_PHYSIO for simulation"""

    def __init__(self, ecg_f, resp_f, sample_freq, rtp_retrots, verb=True):
        """
        Options
        -------
        ecg_f: Path object or string
            ecg signal file
        resp_f: Path object or string
        sample_freq: float
            Frequency of signal in the files (Hz)
        rtp_retrots: RtpRetroTS object
            instance of RtpRetroTS for making RetroTS reggressor
        verb: bool
            verbose flag to print log message
        """

        super().__init__()  # call __init__() in RTP class

        # --- Set parameters ---
        self.sample_freq = sample_freq
        self.rtp_retrots = rtp_retrots
        self._verb = verb

        # --- Load data ---
        assert Path(ecg_f).is_file()
        assert Path(resp_f).is_file()
        self.ecg_data = np.loadtxt(ecg_f)
        self.resp_data = np.loadtxt(resp_f)

        # --- recording status ---
        self.wait_scan = False
        self.scanning = False

        self.not_available = False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_retrots(self, TR, Nvol=np.inf, tshift=0, timeout=None):
        if self.rtp_retrots is None:
            cname = self.__class__.__name__
            self.errmsg(f"No retrots object for {cname}.", no_pop=True)
            return None

        tlen_current = len(self.resp_data)
        max_vol = int((tlen_current/self.sample_freq)//TR)
        if np.isinf(Nvol):
            Nvol = max_vol

        tlen_need = int(Nvol * TR * self.sample_freq)
        while len(self.resp_data) < tlen_need:
            self.errmsg(f"Physio data is availabel up to {Nvol*TR} s")
            return

        Resp = self.resp_data[:tlen_need]
        ECG = self.ecg_data[:tlen_need]

        PhysFS = self.sample_freq
        retroTSReg = self.rtp_retrots.do_proc(Resp, ECG, TR, PhysFS, tshift)

        return retroTSReg[:Nvol, :]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_param(self, attr, val=None, echo=False, **kwargs):
        if attr == 'ecg_f':
            if not Path(val).is_file():
                sys.stderr.write("File {val} not found.\n")
                return

            self.ecg_data = np.loadtxt(val)
            return
        elif attr == 'resp_f':
            if not Path(val).is_file():
                sys.stderr.write("File {val} not found.\n")
                return

            self.resp_data = np.loadtxt(val)
            return
        else:
            # Ignore an unrecognized parameter
            if not hasattr(self, attr):
                self.errmsg(f"{attr} is unrecognized parameter.", no_pop=True)
                return

        setattr(self, attr, val)
        if echo and self._verb:
            print("{}.".format(self.__class__.__name__) + attr, '=',
                  getattr(self, attr))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_params(self):
        opts = dict()
        for var_name, var_val in self.__dict__.items():
            try:
                pickle.dumps(var_val)
                opts[var_name] = var_val
            except Exception:
                continue

        return opts

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Dummy functions for comaptibility with RTP_PHYSIO
    def start_recording(self):
        pass

    def stop_recording(self):
        pass

    def open_signal_plot(self):
        pass

    def save_data(self, *args, **kwargs):
        pass


# %% main =====================================================================
if __name__ == '__main__':
    #import IPython
    #shell = IPython.get_ipython()
    #shell.enable_matplotlib(gui='qt')

    #from rtp_retrots import RtpRetroTS
    #rtp_retrots = RtpRetroTS()

    # Standalone test
    scan_onset = RtpExtSignal()
    rtp_phys = RTP_PHYSIO(scan_onset)
    #rtp_phys.rtp_retrots = rtp_retrots

    multiproc = True
    if multiproc:
        # Start recording in a child process
        rtp_phys.start()
    else:
        # within the same process
        rtp_phys.run(rtp_phys.resp_data, rtp_phys.ecg_data,
                     rtp_phys._is_scanning, rtp_phys._scan_onset)

    """
    rtp_phys.cmd('SCAN_START')

    """
