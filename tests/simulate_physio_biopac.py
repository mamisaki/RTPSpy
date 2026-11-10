#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate the physio recording device (serial packets of ECG and respiration)
and the scan TTL signal.

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
import sys
import os
from pathlib import Path
import time
import struct
import socket
import threading
import argparse
from datetime import datetime
import platform

import numpy as np
import serial
from serial.tools.list_ports import comports

platform_name = platform.system()
if platform_name != 'Windows':
    import pty

if '__file__' not in locals():
    __file__ = 'simulate_physio.py'


# %% PhysioFeeder class =======================================================
class PhysioFeeder(threading.Thread):
    """
    Thread that sends ECG and respiration signals to a serial port in the
    packet format of the physio recording device.
    """

    def __init__(self, ecg_f, resp_f, recording_rate_ms=25,
                 samples_to_average=5, sport='/dev/ptmx', parent=None):
        """
        Parameters
        ----------
        ecg_f : str
            ECG filename.
        resp_f : str
            Respiration filename.
        recording_rate_ms : int
            Sampling interval (ms) of the ECG and respiration file data.
        samples_to_average : int
            Number of samples averaged to make the ECG and respiration file
            data. The raw signal frequency is therefore
            1000 * samples_to_average / recording_rate_ms Hz.
        sport : str
            Output serial port device name. The default is the pseudo
            serial port, /dev/ptmx.
        parent : object
            Its logmsg() method is used for logging, if any.
        """
        super().__init__()

        self.parent = parent

        self.read_data_files(ecg_f, resp_f)

        self._recording_rate_ms = recording_rate_ms
        self._samples_to_average = samples_to_average

        self.set_out_ports(sport)

        self._count = 0  # data counter
        self.cmd = ''

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def read_data_files(self, ecg_f, resp_f):
        """
        Read the ECG and respiration data files.

        Sets self._ecg and self._resp (int16 arrays) and self._siglen (data
        length).
        """
        ecg = []
        resp = []
        minlen = 0

        if Path(ecg_f).is_file():
            ecg = self._read_int16(ecg_f)
            minlen = len(ecg)

        if Path(resp_f).is_file():
            resp = self._read_int16(resp_f)
            minlen = min([minlen, len(resp)])

        # -- Make the ECG and respiration data the same length --
        if minlen > 0:
            adjusted = False
            if len(ecg) > minlen:
                ecg = ecg[:minlen]
                adjusted = True
            elif len(ecg) == 0:
                ecg = np.zeros(minlen, dtype=np.int16)

            if len(resp) > minlen:
                resp = resp[:minlen]
                adjusted = True
            elif len(resp) == 0:
                resp = np.zeros(minlen, dtype=np.int16)

            if adjusted:
                log_msg = f"Data length is adjusted to {minlen}"
                self._log(log_msg)

        self._ecg = ecg
        self._resp = resp
        self._siglen = minlen

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _read_int16(self, fname):
        """Read a text file of one value per line as an int16 array."""
        txt = open(fname).read()
        data = [int(round(float(v))) for v in txt.rstrip().split('\n')]
        data = np.array(data, dtype=np.int16)
        self._log(f"Read {len(data)} values from {fname}")
        return data

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_out_ports(self, sport):
        """Open the output serial port.
        For a pseudo terminal, sport is 'name:fd', and fd is written.
        """
        if 'ptmx' in sport:
            sport, fd = sport.split(':')
            self.sport_fd = int(fd)

        self._ser = serial.Serial(sport, 115200)
        self._ser.flushOutput()

        log_msg = f"Open serial port {sport}"
        self._log(log_msg)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def data_packet(self, seqn, ecg, resp, ecg2, ecg3):
        """Pack the values into a packet with a checksum."""
        seqn = seqn % (np.iinfo(np.uint16).max+1)
        byte_pack = struct.pack('H', np.array(seqn, dtype=np.uint16))
        byte_pack += struct.pack('h', ecg2)
        byte_pack += struct.pack('h', ecg3)
        byte_pack += struct.pack('h', resp)
        byte_pack += struct.pack('h', ecg)

        # Checksum: sum of the 10 bytes above
        chsum = np.array(sum(struct.unpack('B' * 10, byte_pack)),
                         dtype=np.int16)
        byte_pack += struct.pack('h', chsum)

        return byte_pack

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run(self):
        # Interval of the raw (not averaged) samples
        interval_sec = self._recording_rate_ms/self._samples_to_average/1000

        self._log("Start physio signal feeding")

        t = 0
        self._count = 0

        t0 = time.time()
        runSend = True
        while runSend:
            # Repeat _samples_to_average times to simulate raw data
            for rep in range(self._samples_to_average):
                t = t % 65536  # Packet number
                ni = self._count % self._siglen  # Loop over the data

                byte_pack = self.data_packet(t, self._ecg[ni],
                                             self._resp[ni], 0, 0)

                # Wait until the feed time
                feed_sec = (self._count * self._samples_to_average + rep)
                feed_sec *= interval_sec

                while (time.time() - t0) < feed_sec:
                    time.sleep(interval_sec/10000)

                if 'ptmx' in self._ser.name:
                    os.write(self.sport_fd, byte_pack)
                else:
                    self._ser.write(byte_pack)
                t += 1

                if self.cmd == 'STOP':
                    runSend = False
                    break

            self._count += 1

        log_msg = "Finish physio signal feeding."
        n_sent = (self._count + 1) * self._samples_to_average
        log_msg += f" {n_sent} points were sent."
        self._log(log_msg)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _log(self, msg):
        if self.parent and hasattr(self.parent, 'logmsg'):
            self.parent.logmsg(msg)
        else:
            print(msg)


# %% SendTTL class ============================================================
class SendTTL():
    """Send the scan TTL signal via a USB GPIO module or a UNIX domain
    socket (sig_port='0')."""

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, sig_port, sig_sock_file='/tmp/rtp_uds_socket',
                 logfd=sys.stdout):

        self._sig_port = sig_port
        self._logfd = logfd
        self._cname = self.__class__.__name__

        if sig_port is None:
            return

        if sig_port == '0':
            self._sig_sock_file = sig_sock_file

        # -- Set the signal port --
        if '/dev/ttyACM' in sig_port or '/dev/cu.usbmodem' in self._sig_port:
            # Numato 8 Channel USB GPIO Module
            self._sig_port_ser = serial.Serial(sig_port, 19200, timeout=0.1,
                                               write_timeout=0.1)
            self._sig_port_ser.flushOutput()
            self._sig_port_ser.write(b"gpio clear 0\r")

        elif sig_port == '0':
            # Use a UDS socket to send the trigger signal
            self.onsig_sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            if self._logfd:
                self._log(f"UDS socket {self._sig_sock_file} will be used"
                          " for sending the scan start signal")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def send_signal(self, duration=0.015):
        """Send a pulse of duration (s) to the port."""
        if self._sig_port is None:
            return

        if '/dev/ttyACM' in self._sig_port or \
                '/dev/cu.usbmodem' in self._sig_port:
            try:
                ont = time.time()
                self._sig_port_ser.reset_output_buffer()
                self._sig_port_ser.write(b"gpio set 0\r")
                self._sig_port_ser.flush()
                time.sleep(duration)
                self._sig_port_ser.write(b"gpio clear 0\r")
                self._sig_port_ser.flush()
                print("TTL on {}".format(
                    datetime.fromtimestamp(ont).strftime(
                        "%Y-%m-%dT%H:%M:%S.%f")))
                sys.stdout.flush()
            except serial.serialutil.SerialTimeoutException:
                pass

        elif self._sig_port == '0':
            if not os.path.exists(self._sig_sock_file):
                errmsg = f'No UDS file {self._sig_sock_file}\n'
                errmsg += 'The receiver must create the UDS socket file.'
                if self._logfd:
                    self._log(errmsg)
            else:
                self.onsig_sock.sendto(b'1', self._sig_sock_file)
                print("TTL on {}".format(
                    datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")))
                sys.stdout.flush()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def reset_state(self):
        """Set the GPIO output low."""
        if self._sig_port is None:
            return

        if '/dev/ttyACM' in self._sig_port or \
                '/dev/cu.usbmodem' in self._sig_port:
            try:
                self._sig_port_ser.reset_output_buffer()
                self._sig_port_ser.write(b"gpio clear 0\r")
                self._sig_port_ser.flush()
            except serial.serialutil.SerialTimeoutException:
                pass

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _log(self, msg):
        wmsg = "{} [{}] {}\n".format(time.ctime(), self._cname, msg)
        self._logfd.write(wmsg)


# %% SimPhysio class ==========================================================
class SimPhysio():
    """
    Physiological signal simulation: runs a PhysioFeeder thread.
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self):
        self.physio_kwargs = {}
        self.physio_feeder = None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_physio(self, ecg_src, resp_src, physio_port, recording_rate_ms=25,
                   samples_to_average=5):
        self.physio_kwargs = {
            'ecg_f': ecg_src,
            'resp_f': resp_src,
            'recording_rate_ms': recording_rate_ms,
            'samples_to_average': samples_to_average,
            'sport': physio_port,
            'parent': self}

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def feed_Physio(self, mode):
        """Start ('start') or stop ('stop') physio feeding."""
        if mode == 'start':
            if self.physio_feeder is not None \
                    and self.physio_feeder.is_alive():
                return  # Already running

            if not ('ecg_f' in self.physio_kwargs and
                    'resp_f' in self.physio_kwargs and
                    'sport' in self.physio_kwargs):
                sys.stderr.write("!!! Physio parameters have not been set.")
                return

            self.physio_feeder = PhysioFeeder(**self.physio_kwargs)
            self.physio_feeder.start()

        elif mode == 'stop':
            if self.physio_feeder is not None and \
                    self.physio_feeder.is_alive():
                self.physio_feeder.cmd = 'STOP'
                self.physio_feeder.join()
                del self.physio_feeder
                self.physio_feeder = None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        self.feed_Physio('stop')


# %% __main__ =================================================================
if __name__ == '__main__':

    TEST_DATA_ROOT = Path(__file__).resolve().parent / 'test_data' / 'Physio'

    # --- Parse arguments -----------------------------------------------------
    parser = argparse.ArgumentParser(
        description='Simulate physio signals')
    parser.add_argument('--test_data', default=TEST_DATA_ROOT,
                        help='Test data directory')
    args = parser.parse_args()
    test_data_dir = Path(args.test_data)

    recording_rate_ms = 25
    samples_to_average = 5

    # --- List the serial ports ----------------------------------------------
    ser_ports = {}
    for pt in comports():
        ser_ports[pt.device] = pt.description

    if platform_name != 'Windows':
        # Make a pseudo serial port to simulate the physio recording device
        master, slave = pty.openpty()
        s_name = os.ttyname(slave)
        try:
            m_name = os.ttyname(master)
        except Exception:
            m_name = 'Pseudo master'
        physio_port = f"{m_name}:{master}"
        ser_ports[physio_port] = f'Pseudo tty to {s_name}'

    # --- Select TTL port -----------------------------------------------------
    default_sport_ttl = '2'
    msg_txt = '\n' + '=' * 80 + '\n'
    msg_txt += "Select TTL port\n0)exit"
    for ii, (sport, lab) in enumerate(ser_ports.items()):
        msg_txt += f"\n{ii+1}){sport} ({lab})"
    msg_txt += f'\n[{default_sport_ttl}]: '
    while True:
        sport_ttl_i = input(msg_txt)
        if sport_ttl_i == '':
            sport_ttl_i = default_sport_ttl
        try:
            sport_ttl_i = int(sport_ttl_i)
            assert sport_ttl_i <= len(ser_ports)

        except Exception:
            continue
        break
    if sport_ttl_i == 0:
        sys.exit()
    sport_ttl = list(ser_ports.keys())[sport_ttl_i-1]

    # --- Simulation loop -----------------------------------------------------
    END_P_LOOP = False
    default_phydata_i = '1'
    while not END_P_LOOP:
        # -- Select the physio data -----------------------------------------
        physfs = [f.name.replace('ECG_', '')
                  for f in test_data_dir.glob('ECG*.1D')
                  if (test_data_dir / f.name.replace('ECG', 'Resp')
                      ).is_file()]

        msg_txt = '\n' + '=' * 80 + '\n'
        msg_txt += "Select data\n0)exit"
        for ii, physf in enumerate(physfs):
            msg_txt += f"\n{ii+1}){physf}"
        msg_txt += f'\n[{default_phydata_i}]: '
        while True:
            phydata_i = input(msg_txt)
            if phydata_i == '':
                phydata_i = default_phydata_i
            try:
                phydata_i = int(phydata_i)
                assert phydata_i <= len(physfs)
            except Exception:
                continue
            break
        if phydata_i == 0:
            sys.exit()
        physf = physfs[phydata_i-1]
        ecg_f = test_data_dir / ('ECG_' + physf)
        assert ecg_f.is_file()
        resp_f = test_data_dir / ('Resp_' + physf)
        assert resp_f.is_file()

        # --- Send a TTL at each return key ----------------------------------
        sim_ttl = SendTTL(sport_ttl)

        while True:
            try:
                ttl = input('Press return to send TTL ')
                if ttl == 'q':
                    break

                sim_ttl.send_signal()

            except KeyboardInterrupt:
                print("\nExit TTL loop.\n")
                break

        if ttl == 'q':
            sys.exit()
