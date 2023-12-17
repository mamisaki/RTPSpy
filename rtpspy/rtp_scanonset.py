#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 13:20:48 2018

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import re
import serial
from serial.tools.list_ports import comports
import socket
import numpy as np
import time
from PyQt5 import QtWidgets, QtCore
import matplotlib as mpl
import sys
import traceback

from rtpspy.rtp_common import RTP, RingBuffer, MatplotlibWindow

if len(list(Path('/dev').glob('parport*'))) > 0:
    import parallel

mpl.rcParams['font.size'] = 8


# %% RtpExtSignal class ======================================================
class RtpExtSignal(RTP):
    """
    Scan onset monitor class

    + Supporting devices
    ++ Unix domain socket
        Use socket file /tmp/rtp_uds_socket to recieve scan onset signal.
        If the content of the file is b'1', read_onsig_port retruns True.

    ++ Parallel port
        Read pin 11 (Busy) to recieve scan onset TTL signal.

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

    def __init__(self, onsig_port=None, verb=True):
        """
        Parameters
        ----------
        onsig_port : str, optional
            Port to monitor scan onset signal.
            'Unix domain socket': Unix domain socket at socket file,
                /tmp/rtp_uds_socket.
            'parport*': parallel port pin 11.
            one of the items listed in SUPPORT_DEVS.
            The default is None.
        verb : bool, optional
            Verbose flag to print log message. The default is True.

        Internal properties
        self._onsig_sock : socket.socket object
            Unix domain socket object
        self._pport : parallel.Parallel object
            Parallel port object
        self._onsig_port_ser : serial.Serial object
            Serial port object
        """

        super().__init__()  # call __init__() in RTP class

        # --- Set parameters ---
        self.verb = verb
        delattr(self, 'work_dir')

        # --- Set available ports list ---
        self.dict_onsig_port = {}

        # Search serial (usb) ports
        for pt in comports():
            for desc in RtpExtSignal.SUPPORT_DEVS:
                if desc in pt.description:
                    if desc in self.dict_onsig_port:
                        num_dev = np.sum(
                            [desc in k for k in self.dict_onsig_port.keys()])
                        desc += f" {num_dev}"
                    self.dict_onsig_port[desc] = pt.device

        # parallel port
        #for pp in Path('/dev').glob('parport*'):
        #    self.dict_onsig_port[pp.name] = str(pp)

        #  Unix domain socket
        uds_sock_file = '/tmp/rtp_uds_socket'
        self.dict_onsig_port['Unix domain socket'] = uds_sock_file

        if onsig_port is None or onsig_port not in self.dict_onsig_port.keys():
            onsig_port = list(self.dict_onsig_port.keys())[0]

        self.onsig_port = None
        self.init_onsig_port(onsig_port)

        # --- recording status ---
        self.wait_scan = False
        self.scanning = False
        self.scan_onset = -1.0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ getter methos +++
    @property
    def isOpen(self):
        if self.onsig_port is not None:
            return True
        else:
            return False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def init_onsig_port(self, onsig_port):
        # Delete string after ' : ('
        onsig_port = re.sub(r' : \(.+', '', onsig_port).rstrip()

        if onsig_port is None:
            return

        if onsig_port not in self.dict_onsig_port.keys():
            self.errmsg(f"No port {onsig_port}")
            return

        if onsig_port == 'Unix domain socket':
            # --- Unix domain socket ---
            if self.onsig_port == onsig_port:
                # delete the socket to reset.
                if self._onsig_sock is not None:
                    self._onsig_sock.close()

                del self._onsig_sock
                self._onsig_sock = None
                time.sleep(1)

            uds_sock_file = Path(self.dict_onsig_port[onsig_port])
            try:
                # Create a UDS file
                self._onsig_sock = socket.socket(socket.AF_UNIX,
                                                 socket.SOCK_DGRAM)
                self._onsig_sock.settimeout(0.01)
                if uds_sock_file.is_socket():
                    uds_sock_file.unlink()
                    time.sleep(1)

                self._onsig_sock.bind(str(uds_sock_file))
                self.onsig_port = onsig_port

            except Exception:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                traceback.print_exception(exc_type, exc_obj, exc_tb)

                del self._onsig_sock
                if uds_sock_file.is_socket():
                    uds_sock_file.unlink()
                self.onsig_port = None

            return

        elif 'parport' in onsig_port:
            # --- Parallel port ---
            if self.onsig_port == onsig_port:
                # delete the parallel.Parallel object to reset.
                del self._pport
                self._onsig_sock = None
                time.sleep(1)

            dev = self.dict_onsig_port[onsig_port]
            try:
                self._pport = parallel.Parallel(dev)
                self.onsig_port = onsig_port

            except Exception:
                """
                self.errmsg(e)
                errmsg = "'sudo modprobe ppdev parport_pc parprot' and "
                errmsg += "'sudo modprobe -r lp' might solve the problem"
                self.errmsg(errmsg)
                errmsg = "Scan onset signal cannot be received"
                errmsg += " at {}".format(onsig_port)
                self.errmsg(errmsg)
                """
                exc_type, exc_obj, exc_tb = sys.exc_info()
                traceback.print_exception(exc_type, exc_obj, exc_tb)

                del self._pport
                self.onsig_port = None

            return

        elif 'CDC RS-232 Emulation Demo' in onsig_port or \
                'Numato Lab 8 Channel USB GPIO M' in onsig_port:
            # --- Numato Lab 8 Channel USB GPIO Module ---
            if self.onsig_port == onsig_port:
                # delete the serial.Serial object to reset.
                del self._onsig_port_ser
                time.sleep(1)

            dev = self.dict_onsig_port[onsig_port]
            try:
                self._onsig_port_ser = serial.Serial(dev, 19200,
                                                     timeout=0.0005)
                self._onsig_port_ser.flushOutput()
                self._onsig_port_ser.write(b"gpio clear 0\r")
                self.onsig_port = onsig_port

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                traceback.print_exception(exc_type, exc_obj, exc_tb)

                self.errmsg(e)
                errmsg = "Failed to open {}".format(onsig_port)
                self.errmsg(errmsg)
                errmsg = "Scan onset signal cannot be received"
                errmsg += " at {}".format(onsig_port)
                self.errmsg(errmsg)
                del self._onsig_port_ser
                self.onsig_port = None

            return

        else:
            self.errmsg(f"{onsig_port} is not defined" +
                        " for receiving scan onset signal.\n")

            return

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def read_onsig_port(self):
        if self.onsig_port is None:
            return 0

        if self.onsig_port == 'Unix domain socket':
            try:
                sig = self._onsig_sock.recvfrom(1)[0]
                return int(sig == b'1')

            except socket.timeout:
                self.onsig_port = None
                return 0

        elif 'parport' in self.onsig_port:
            if not hasattr(self, '_pport') or self._pport is None:
                self.onsig_port = None
                return 0

            # scan pin 11 (Busy), which can be read with getInBusy()
            busy = False
            # wait a while (0.1 ms)
            st = time.time()
            while time.time()-st < 0.0001 and not busy:
                busy |= self._pport.getInBusy()

            return int(busy)

        elif 'CDC RS-232 Emulation Demo' in self.onsig_port or \
                'Numato Lab 8 Channel USB GPIO M' in self.onsig_port:
            if not hasattr(self, '_onsig_port_ser') or \
                    self._onsig_port_ser is None:
                self.onsig_port = None
                return 0

            self._onsig_port_ser.reset_output_buffer()
            self._onsig_port_ser.reset_input_buffer()
            self._onsig_port_ser.write(b"gpio read 0\r")
            resp = self._onsig_port_ser.read(1024)
            ma = re.search(r'gpio read 0\n\r(\d)\n', resp.decode())
            if ma:
                sig = ma.groups()[0]
                return int(sig == '1')
            else:
                return 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    class Monitoring(QtCore.QObject):
        finished = QtCore.pyqtSignal()

        # ---------------------------------------------------------------------
        def __init__(self, root, main_win=None):
            super().__init__()
            self.root = root
            self.abort = False

        # ---------------------------------------------------------------------
        def run(self):
            while not self.abort:
                try:
                    if self.root.read_onsig_port() == 1:
                        self.root.scan_onset = time.time()
                        if self.root._verb:
                            self.root.logmsg("Received scan onset signal.")
                        self.root.scanning = True
                        break

                except Exception as e:
                    print(e)
                    break

                time.sleep(0.0001)

            # -- end loop --
            self.finished.emit()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def wait_scan_onset(self):
        if hasattr(self, 'thMonitor') and self.thMonitor.isRunning():
            return

        self.scanning = False
        self.scan_onset = -1.0

        self.thMonitor = QtCore.QThread()
        self.monitor = RtpExtSignal.Monitoring(self, main_win=self.main_win)
        self.monitor.moveToThread(self.thMonitor)
        self.thMonitor.started.connect(self.monitor.run)
        self.monitor.finished.connect(self.thMonitor.quit)
        self.thMonitor.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def is_waiting(self):
        return hasattr(self, 'thMonitor') and self.thMonitor.isRunning()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def abort_waiting(self):
        if self.is_waiting():
            self.monitor.abort = True
            if not self.thMonitor.wait(1):
                # self.monitor.finished is not emitted with sone reason
                self.monitor.finished.emit()
                self.thMonitor.wait()

            del self.thMonitor
            del self.monitor

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def is_scan_on(self):
        return self.scanning

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def manual_start(self):
        if not self.is_waiting():
            return

        self.scan_onset = time.time()
        if self.verb:
            self.logmsg("Manual start")
        self.scanning = True

        self.abort_waiting()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end_reset(self):
        self.scanning = False
        self.scan_onset = -1.0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    class PlotOnsigPort(QtCore.QObject):
        finished = QtCore.pyqtSignal()

        def __init__(self, root, sample_freq=40, show_length_sec=5,
                     main_win=None):
            """
            Options
            -------
            sample_freq: float
                sampling frequency, Hz
            show_length_sec: float
                plot window length, seconds
            """
            super().__init__()

            # Set variables
            self.root = root
            self.sample_freq = sample_freq
            self.main_win = main_win
            self.abort = False

            self.sig_rbuf = RingBuffer(sample_freq * show_length_sec)

            # Initialize figure
            plt_winname = 'Monitor {}'.format(self.root.onsig_port)
            self.plt_win = MatplotlibWindow()
            self.plt_win.setWindowTitle(plt_winname)

            # set position
            if main_win is not None:
                main_geom = main_win.geometry()
                x = main_geom.x() + main_geom.width()
                y = main_geom.y()
            else:
                x, y = (0, 0)
            self.plt_win.setGeometry(x, y, 360, 180)

            # Set axis
            self.ax = self.plt_win.canvas.figure.subplots()
            xi = np.arange(0, show_length_sec, 1.0/sample_freq) + \
                1.0/sample_freq
            self.ln = self.ax.plot(xi, self.sig_rbuf.get())

            self.ax.set_xlim([0, show_length_sec])
            self.ax.set_xlabel('seconds')
            self.ax.set_ylim([-0.1, 1.1])
            self.ax.set_yticks([0, 1])
            self.ax.set_ylabel('TTL')
            self.ax.set_position([0.15, 0.25, 0.8, 0.7])

            # show window
            self.plt_win.show()

        # ---------------------------------------------------------------------
        def run(self):
            interval = 1.0/self.sample_freq
            nt = time.time()+interval
            while self.plt_win.isVisible() and not self.abort:
                while time.time() < nt:
                    time.sleep(interval/10)

                val = self.root.read_onsig_port()
                if val is None:
                    pass
                    # self.sig_rbuf.append(np.nan)
                else:
                    self.sig_rbuf.append(val)
                    self.ln[0].set_ydata(self.sig_rbuf.get())

                self.ax.figure.canvas.draw()

                nt += interval

                if self.main_win is not None and \
                        not self.main_win.isVisible():
                    break

            self.end_thread()

        def end_thread(self):
            if self.plt_win.isVisible():
                self.plt_win.close()
            self.finished.emit()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_param(self, attr, val=None, reset_fn=None, echo=False):
        """
        When reset_fn is None, set_param is considered to be called from
        load_parameters function.
        """

        # -- check value --
        if attr == 'onsig_port' and reset_fn is None:
            if self.onsig_port == val:
                return

            idx = self.ui_onSigPort_cmbBx.findText(val,
                                                   QtCore.Qt.MatchContains)
            if idx == -1:
                return

            if hasattr(self, 'thMonOnsigPort') and \
                    self.thMonOnsigPort.isRunning():
                self.monOnsigPort.abort = True
                self.thMonOnsigPort.wait()
                del self.thMonOnsigPort

            if hasattr(self, 'ui_onSigPort_cmbBx'):
                self.ui_onSigPort_cmbBx.setCurrentIndex(idx)

            if val is not None:
                self.init_onsig_port(val)

            return

        elif attr == 'monitor_onsig_port':
            if hasattr(self, 'thPltOnsigPort') and \
                    self.thPltOnsigPort.isRunning():
                return

            self.thPltOnsigPort = QtCore.QThread()
            self.pltOnsigPort = \
                RtpExtSignal.PlotOnsigPort(self, main_win=self.main_win)
            self.pltOnsigPort.moveToThread(self.thPltOnsigPort)
            self.thPltOnsigPort.started.connect(self.pltOnsigPort.run)
            self.pltOnsigPort.finished.connect(self.thPltOnsigPort.quit)
            self.thPltOnsigPort.start()
            return

        elif attr == 'plot_len_sec' and reset_fn is None:
            if hasattr(self, 'ui_pltLen_dSpBx'):
                self.ui_pltLen_dSpBx.setValue(val)

        elif isinstance(val, serial.Serial) or \
                isinstance(val, QtCore.QThread):
            return

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
        if echo and self._verb:
            print("{}.".format(self.__class__.__name__) + attr, '=',
                  getattr(self, attr))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ui_set_param(self):

        ui_rows = []
        self.ui_objs = []

        # get list of comports
        dev_list = []
        for pinfo in comports():
            dev_list.append(pinfo.device)

        # onsig_port
        var_lb = QtWidgets.QLabel("Port to receive a scan onset signal :")
        self.ui_onSigPort_cmbBx = QtWidgets.QComboBox()
        devlist = sorted([f"{lab} : ({dev})"
                          for lab, dev in self.dict_onsig_port.items()])
        self.ui_onSigPort_cmbBx.addItems(devlist)
        if len(devlist) and self.onsig_port is not None:
            try:
                selIdx = list(self.dict_onsig_port).index(self.onsig_port)
                self.ui_onSigPort_cmbBx.setCurrentIndex(selIdx)
            except ValueError:
                pass

        self.ui_onSigPort_cmbBx.activated.connect(
                lambda idx: self.set_param(
                        'onsig_port', self.ui_onSigPort_cmbBx.currentText(),
                        self.ui_onSigPort_cmbBx.setCurrentIndex))
        ui_rows.append((var_lb, self.ui_onSigPort_cmbBx))
        self.ui_objs.extend([var_lb, self.ui_onSigPort_cmbBx])

        # monitor onsig_port
        self.ui_monitorOnSigPort_btn = QtWidgets.QPushButton()
        self.ui_monitorOnSigPort_btn.setText('Show port status')
        self.ui_monitorOnSigPort_btn.clicked.connect(
                lambda: self.set_param('monitor_onsig_port'))
        ui_rows.append((None, self.ui_monitorOnSigPort_btn))
        self.ui_objs.append(self.ui_monitorOnSigPort_btn)

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
                lambda state: setattr(self, 'polling_interval', state > 0))
        self.ui_objs.append(self.ui_verb_chb)

        chb_hLayout = QtWidgets.QHBoxLayout()
        chb_hLayout.addStretch()
        chb_hLayout.addWidget(self.ui_verb_chb)
        ui_rows.append((None, chb_hLayout))

        return ui_rows

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_params(self):
        all_opts = super().get_params()
        excld_opts = ('scanning', 'scan_onset',
                      'wait_scan', 'dict_onsig_port')
        sel_opts = {}
        for k, v in all_opts.items():
            if k in excld_opts:
                continue
            sel_opts[k] = v

        return sel_opts

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        if hasattr(self, 'thMonOnsigPort') and \
                self.thMonOnsigPort.isRunning():
            self.monOnsigPort.abort = True
            self.thMonOnsigPort.wait()
            del self.thMonOnsigPort

        if Path(self.dict_onsig_port['Unix domain socket']).is_socket():
            Path(self.dict_onsig_port['Unix domain socket']).unlink()


# %% Dummy EXTSIG class for debug
class DUMMY_EXTSIG(RTP):
    def __init__(self):
        super().__init__()  # call __init__() in RTP class
        self.scanning = False

    def is_scan_on(self):
        return self.scanning
