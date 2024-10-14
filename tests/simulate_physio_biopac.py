#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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

    def __init__(self, ecg_f, resp_f, recording_rate_ms=25,
                 samples_to_average=5, sport='/dev/ptmx', parent=None):
        """
        Options
        -------
        ecg_f: string
            ECG filename
        resp_f: string
            Resp filename
        recording_rate_ms: int
            Sampling interval of ECG and Resp file data
        samples_to_average: int
            Number of samples averaged at making ECG and Resp file data.
            The actual signal frequency was
            1000 * samples_to_average/recording_rate_ms Hz
        sport: string
            output serial port device name (default is pseudo serial port
            /dev/ptmx)
        """
        super().__init__()

        self._cname = self.__class__.__name__
        self.parent = parent

        # -- Read data from file --
        self.read_data_files(ecg_f, resp_f)

        # -- set signal timings --
        self._recording_rate_ms = recording_rate_ms
        self._samples_to_average = samples_to_average

        # -- Set output port --
        self._sport = sport
        self.set_out_ports(sport)

        self._count = 0  # data counter
        self.cmd = ''

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def read_data_files(self, ecg_f, resp_f):
        """
        Read ECG, Resp data files

        Set variables
        -------------
        self._ecg: array
            ECG data vector
        self._resp: array
            Resp data vector
        self._siglen: int
            data length
        """

        # -- Initialize --
        ecg = []
        resp = []
        minlen = 0

        # -- Read data --
        # ECG
        if Path(ecg_f).is_file():
            # Read text
            ecg = open(ecg_f).read()
            # Split text and convert to int
            ecg = [int(round(float(v))) for v in ecg.rstrip().split('\n')]
            # Convert to short int array
            ecg = np.array(ecg, dtype=np.int16)
            minlen = len(ecg)
            log_msg = f"Read {len(ecg)} values from {ecg_f}"
            self._log(log_msg)

        # Respiration
        if Path(resp_f).is_file():
            # Read text
            resp = open(resp_f).read()
            # Split text and convert to int
            resp = [int(round(float(v))) for v in resp.rstrip().split('\n')]
            # Convert to short int array
            resp = np.array(resp, dtype=np.int16)
            minlen = min([minlen, len(resp)])
            log_msg = f"Read {len(resp)} values from {resp_f}"
            self._log(log_msg)

        # -- adjust signal length to equalize data length of ECG and Resp --
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

        # -- Save data as class variables --
        self._ecg = ecg
        self._resp = resp
        self._siglen = minlen

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_out_ports(self, sport):

        # -- Open a serial port --
        if 'ptmx' in sport:
            sport, fd = sport.split(':')
            self.sport_fd = int(fd)

        self._ser = serial.Serial(sport, 115200)
        self._ser.flushOutput()

        log_msg = f"Open serial port {sport}"
        self._log(log_msg)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def data_packet(self, seqn, ecg, resp, ecg2, ecg3):

        # packing data into a packet
        seqn = seqn % (np.iinfo(np.uint16).max+1)
        byte_pack = struct.pack('H', np.array(seqn, dtype=np.uint16))
        byte_pack += struct.pack('h', ecg2)
        byte_pack += struct.pack('h', ecg3)
        byte_pack += struct.pack('h', resp)
        byte_pack += struct.pack('h', ecg)

        # checksum
        chsum = np.array(sum(struct.unpack('B' * 10, byte_pack)),
                         dtype=np.int16)
        byte_pack += struct.pack('h', chsum)

        return byte_pack

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run(self):

        # -- Set signal interval --
        interval_sec = self._recording_rate_ms/self._samples_to_average/1000

        # Start loop for sending signal
        log_msg = "Start physio signal feeding"
        self._log(log_msg)

        t = 0
        self._count = 0

        t0 = time.time()  # start time
        runSend = True
        while runSend:
            # Repeat _samples_to_average times to simulate raw data
            for rep in range(self._samples_to_average):
                # packet number
                t = t % 65536
                ni = self._count % self._siglen

                # Prepare packet
                byte_pack = self.data_packet(t, self._ecg[ni],
                                             self._resp[ni], 0, 0)

                # wait for feed time
                feed_sec = (self._count * self._samples_to_average + rep)
                feed_sec *= interval_sec

                while (time.time() - t0) < feed_sec:
                    time.sleep(interval_sec/10000)

                # Send packet
                if 'ptmx' in self._ser.name:
                    os.write(self.sport_fd, byte_pack)
                else:
                    self._ser.write(byte_pack)
                t += 1  # increment packet number

                if len(self.cmd):
                    if self.cmd == 'STOP':
                        runSend = False
                        break

            self._count += 1  # increment data index

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


# %% SendTTL class ========================================================
class SendTTL():
    """ Scan timing TTL signal sender class """

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

        # -- Set onset signal sending port --
        if '/dev/ttyACM' in sig_port or '/dev/cu.usbmodem' in self._sig_port:
            # Numato 8 Channel USB GPIO Module
            self._sig_port_ser = serial.Serial(sig_port, 19200, timeout=0.1,
                                               write_timeout=0.1)
            self._sig_port_ser.flushOutput()
            self._sig_port_ser.write(b"gpio clear 0\r")

        elif sig_port == '0':
            # Use UDS socket to send trigger signal
            self.onsig_sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            try:
                if self._logfd:
                    log_msg = "UDS socket {} will be used".format(
                            self._sig_sock_file)
                    log_msg += " for sending scan start signal"
                    self._log(log_msg)

            except socket.error:
                if self._logfd:
                    log_msg = "Cannot connect UDS socket "
                    log_msg += "{}".format(self._sig_sock_file)
                    self._log(log_msg)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def send_singnal(self, duration=0.015):
        """ send pulse to the port """
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
                errmsg = 'No UDS file'
                errmsg += ' {}\n'.format(self._sig_sock_file)
                errmsg += 'UDS socket file must be prepared by the receiver.'
                if self._logfd:
                    self._log(errmsg)
            else:
                self.onsig_sock.sendto(b'1', self._sig_sock_file)
                print("TTL on {}".format(
                    datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")))
                sys.stdout.flush()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def reset_state(self):
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
    Physiological signal simulation
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self):
        self.physio_kwargs = {}
        self.phyio_feeder = None

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
        if mode == 'start':
            if self.phyio_feeder is not None \
                    and self.phyio_feeder.is_alive():
                # phyio_feeder is running
                return

            if not ('ecg_f' in self.physio_kwargs and
                    'resp_f' in self.physio_kwargs and
                    'sport' in self.physio_kwargs):
                sys.stderr.write("!!! Physio parameters have not been set.")
                return

            self.phyio_feeder = PhysioFeeder(**self.physio_kwargs)
            self.phyio_feeder.start()

        elif mode == 'stop':
            if self.phyio_feeder is not None and \
                    self.phyio_feeder.is_alive():
                self.phyio_feeder.cmd = 'STOP'
                self.phyio_feeder.join()
                del self.phyio_feeder
                self.phyio_feeder = None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        self.feed_Physio('stop')


# %% __main__ =================================================================
if __name__ == '__main__':

    TEST_DATA_ROOT = Path(__file__).resolve().parent / 'test_data' / 'Physio'

    # --- Parse arguments -----------------------------------------------------
    parser = argparse.ArgumentParser(
        description='Simulate physio signal')
    parser.add_argument('--test_data', default=TEST_DATA_ROOT,
                        help='test data ditectory')
    args = parser.parse_args()
    test_data_dir = Path(args.test_data)

    recording_rate_ms = 25
    samples_to_average = 5

    # --- Get list of serial port ---------------------------------------------
    ser_ports = {}
    for pt in comports():
        ser_ports[pt.device] = pt.description

    if platform_name != 'Windows':
        # Set psuedo serial port for simulating physio recording
        master, slave = pty.openpty()
        s_name = os.ttyname(slave)
        try:
            m_name = os.ttyname(master)
        except Exception:
            m_name = 'Psuedo master'
        physio_port = f"{m_name}:{master}"
        ser_ports[physio_port] = f'Psuedo tty to {os.ttyname(slave)}'

    # # --- Select Physio port ------------------------------------------------
    # default_sport_phys = '1'
    # msg_txt = '\n' + '=' * 80 + '\n'
    # msg_txt += "Select physio port\n0)exit"
    # for ii, (sport, lab) in enumerate(ser_ports.items()):
    #     msg_txt += f"\n{ii+1}){sport} ({lab})"
    # msg_txt += f'\n[{default_sport_phys}]: '
    # while True:
    #     sport_phyis_i = input(msg_txt)
    #     if sport_phyis_i == '':
    #         sport_phyis_i = default_sport_phys
    #     try:
    #         sport_phyis_i = int(sport_phyis_i)
    #         assert sport_phyis_i <= len(ser_ports)
    #     except Exception:
    #         continue
    #     break
    # if sport_phyis_i == '0':
    #     sys.exit()
    # sport_phyis = list(ser_ports.keys())[sport_phyis_i-1]

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
    if sport_ttl_i == '0':
        sys.exit()
    sport_ttl = list(ser_ports.keys())[sport_ttl_i-1]

    # --- Simulation loop -----------------------------------------------------
    END_P_LOOP = False
    default_phydata_i = '1'
    while not END_P_LOOP:
        # -- Set Physio data -------------------------------------------------
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
        if phydata_i == '0':
            continue
        physf = physfs[phydata_i-1]
        ecg_f = test_data_dir / ('ECG_' + physf)
        assert ecg_f.is_file()
        resp_f = test_data_dir / ('Resp_' + physf)
        assert resp_f.is_file()

        # --- Loop for TTL wait -----------------------------------------------
        sim_ttl = SendTTL(sport_ttl)

        ttl_interval = 0
        while True:
            try:
                ttl = input('Press return to send TTL ')
                if ttl == 'q':
                    break

                sim_ttl.send_singnal()

            except KeyboardInterrupt:
                print("\nExit TTL loop.\n")
                full_automatic = False
                break

        if ttl == 'q':
            sys.exit()
