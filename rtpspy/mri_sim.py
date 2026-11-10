#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time MRI simulation: write fMRI volumes (or DICOM files) to a directory
at the TR timing, and send physio signals to a serial port.

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
import shutil

import numpy as np
import nibabel as nib
import serial
import pydicom

try:
    from .rtp_common import RTP
except Exception:
    from rtpspy.rtp_common import RTP


# %% MRIFeeder class ==========================================================
class MRIFeeder(threading.Thread):
    """
    Thread that writes MRI data files to dst_dir at the acquisition timing.
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, mri_src, imgType, dst_dir, TR=0.0,
                 suffix='sim_fMRI_{num}.nii.gz', parent=None):
        super().__init__()

        self.mri_src = mri_src
        self.imgType = imgType
        self.dst_dir = dst_dir
        self.suffix = suffix
        self.parent = parent

        # Set MRI data parameters
        if self.mri_src.is_file():
            img = nib.load(mri_src)
            self._mri_feed_num = img.shape[-1]
            if TR == 0.0:
                header = img.header
                self.TR = header.get_zooms()[-1]
            else:
                self.TR = TR

            if self.suffix == '':
                if '.gz' in self.mri_src.suffixes:
                    stem = Path(self.mri_src.stem).stem
                else:
                    stem = self.mri_src.stem
                self.suffix = stem

        elif self.mri_src.is_dir():
            if self.imgType == 'GE DICOM':
                ff = list(self.mri_src.glob('i*'))
                self._mri_feed_num = len(ff)
                dcm = pydicom.dcmread(
                    ff[0], defer_size="1 KB", stop_before_pixels=False,
                    force=False)
                self.TR = float(dcm[(0x0018, 0x0080)].value) / 1000
                self.NSlices = dcm[(0x0043, 0x1079)].value
            elif self.imgType == 'Siemens Mosaic':
                ff = list(self.mri_src.glob('*.dcm'))
                self._mri_feed_num = len(ff)
                dcm = pydicom.dcmread(
                    ff[0], defer_size="1 KB", stop_before_pixels=False,
                    force=False)
                self.TR = float(dcm[(0x0018, 0x0080)].value) / 1000

        self.cmd = ''

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run(self):
        num = 0
        self.cmd = ''
        try:
            if self.imgType in ('AFNI BRIK', 'NIfTI'):
                img = nib.load(self.mri_src)
                fmri_data = np.asanyarray(img.dataobj)
                N_files = img.shape[-1]
                feed_delay = 0.2
                feed_interval = self.TR
                dst_dir = Path(self.dst_dir)

            elif self.imgType == 'GE DICOM':
                dicom_fs = np.array(list(self.mri_src.glob('i*')))
                sort_idx = np.argsort(
                    [int(ff.suffix[1:]) for ff in dicom_fs]).ravel()
                src_dicom_fs = dicom_fs[sort_idx]
                N_files = len(src_dicom_fs)
                feed_delay = 0.01
                feed_interval = self.TR / self.NSlices
                dst_dir = Path(self.dst_dir) / self.mri_src.name

            elif self.imgType == 'Siemens Mosaic':
                dicom_fs = np.array(list(self.mri_src.glob('*.dcm')))
                sort_idx = np.argsort(
                    [int(ff.stem) for ff in dicom_fs]).ravel()
                src_dicom_fs = dicom_fs[sort_idx]
                N_files = len(src_dicom_fs)
                feed_delay = 0.2
                feed_interval = self.TR
                dst_dir = Path(self.dst_dir) / self.mri_src.name

            st = time.time()
            while num < N_files:
                if time.time() >= st + (num+1)*feed_interval-feed_delay:
                    # Write a file
                    feed_st = time.time()

                    if not dst_dir.is_dir():
                        dst_dir.mkdir()

                    if self.imgType in ('AFNI BRIK', 'NIfTI'):
                        suffix = self.suffix.format(**{'num': num})
                        dst_f = f"{dst_dir / suffix}" + f"_nr_{num:06d}.nii"
                        fmri_img = nib.Nifti1Image(fmri_data[:, :, :, num],
                                                   affine=img.affine)
                        nib.save(fmri_img, dst_f)
                    elif self.imgType in ('GE DICOM', 'Siemens Mosaic'):
                        src_f = src_dicom_fs[num]
                        dst_f = dst_dir / src_f.name
                        shutil.copy(src_f, dst_f)

                    feed_delay = time.time() - feed_st
                    # GE DICOM is a file per slice; log once per volume.
                    if self.imgType != 'GE DICOM' or \
                            (num+1) % self.NSlices == 0:
                        self._log(f"Write {dst_f}")

                    num += 1
                    if num == self._mri_feed_num:
                        break

                if self.cmd == 'STOP':
                    self._log("+++ Stop MRI feeding")
                    break

                time.sleep(feed_interval / 100)

        finally:
            self._log(f"+++ Finish feeding: wrote {num} files.")
            self.cmd = 'FINISH'

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _log(self, msg):
        """Log with the parent's logger (an RTP object), if any."""
        if self.parent is not None and hasattr(self.parent, '_logger'):
            self.parent._logger.info(msg)
        else:
            print(msg)


# %% PhysioFeeder class =======================================================
class PhysioFeeder(threading.Thread):
    """
    Thread that sends ECG and respiration signals to a serial port in the
    packet format of the physio recording device.
    """

    def __init__(self, ecg_f, resp_f, recording_rate_ms=25,
                 samples_to_average=5, sport='/dev/ptmx', OnsetSig=None,
                 parent=None):
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
        OnsetSig : OnsetSignal object
            Sends the scan start signal when the feeding starts.
        parent : RTP object
            Its logger is used for logging, if any.
        """
        super().__init__()

        self.parent = parent

        self.read_data_files(ecg_f, resp_f)

        self._recording_rate_ms = recording_rate_ms
        self._samples_to_average = samples_to_average

        self.set_out_ports(sport)

        self.OnsetSig = OnsetSig

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

        # -- Adjust the signal lengths
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

        if self.OnsetSig is not None:
            self.OnsetSig.send_scan_start()

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
        """Log with the parent's logger (an RTP object), if any."""
        if self.parent is not None and hasattr(self.parent, '_logger'):
            self.parent._logger.info(msg)
        else:
            print(msg)


# %% OnsetSignal class ========================================================
class OnsetSignal():
    """Send the scan onset signal via a USB GPIO module or a UNIX domain
    socket (onsig_port='0')."""

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, onsig_port, onsig_sock_file='/tmp/rtp_uds_socket',
                 logfd=sys.stdout):

        self._onsig_port = onsig_port
        self._logfd = logfd
        self._cname = self.__class__.__name__

        if onsig_port == '0':
            self._onsig_sock_file = onsig_sock_file

        # -- Set onset signal sending port --
        if '/dev/ttyACM' in onsig_port:
            # Numato 8 Channel USB GPIO Module
            self._onsig_port_ser = serial.Serial(onsig_port, 19200,
                                                 timeout=0.1)
            self._onsig_port_ser.flushOutput()
            self._onsig_port_ser.write(b"gpio clear 0\r")

        elif onsig_port == '0':
            # Use a UDS socket to send the trigger signal
            self.onsig_sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            if self._logfd:
                self._log(f"UDS socket {self._onsig_sock_file} will be used"
                          " for sending the scan start signal")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def send_scan_start(self):
        """Send a pulse to the port."""
        if '/dev/ttyACM' in self._onsig_port:
            ont = time.time()
            self._onsig_port_ser.write(b"gpio set 0\r")
            time.sleep(0.1)
            self._onsig_port_ser.write(b"gpio clear 0\r")
            print("Scan onset {}".format(ont))
            sys.stdout.flush()

        elif self._onsig_port == '0':
            if not os.path.exists(self._onsig_sock_file):
                errmsg = f'No UDS file {self._onsig_sock_file}\n'
                errmsg += 'The receiver must create the UDS socket file.'
                if self._logfd:
                    self._log(errmsg)
            else:
                self.onsig_sock.sendto(b'1', self._onsig_sock_file)
                print("Scan onset {}".format(time.time()))
                sys.stdout.flush()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _log(self, msg):
        wmsg = "{} [{}] {}\n".format(time.ctime(), self._cname, msg)
        self._logfd.write(wmsg)


# %% rtMRISim class ===========================================================
class rtMRISim(RTP):
    """
    Real-time MRI simulation: runs MRIFeeder and PhysioFeeder threads.
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, mri_src='', dst_dir='', suffix='', onsig_port=None):
        super().__init__()

        self.mri_src = mri_src
        self.dst_dir = dst_dir
        self.suffix = suffix
        self.mri_feeder = None

        if onsig_port is not None:
            self.OnsetSig = OnsetSignal(onsig_port)
        else:
            self.OnsetSig = None

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
            'OnsetSig': self.OnsetSig,
            'parent': self}

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run_MRI(self, mode, imgType=None):
        """Start ('start') or stop ('stop') MRI feeding."""
        if mode == 'start':
            if self.mri_feeder is not None and self.mri_feeder.is_alive():
                return  # Already running

            if not Path(self.dst_dir).is_dir():
                self._logger.error(
                    f"Destination directory not found: {self.dst_dir}.")
                return

            self.mri_feeder = MRIFeeder(self.mri_src, imgType, self.dst_dir,
                                        suffix=self.suffix, parent=self)

            self.mri_feeder.start()

        elif mode == 'stop':
            if self.mri_feeder is not None and self.mri_feeder.is_alive():
                self.mri_feeder.cmd = 'STOP'
                self.mri_feeder.join()
                del self.mri_feeder
                self.mri_feeder = None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run_Physio(self, mode):
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
        self.run_MRI('stop')
        self.run_Physio('stop')


# %% __main__ =================================================================
if __name__ == '__main__':
    import pty

    test_dir = Path(__file__).resolve().parent.parent / 'tests'

    # MRI files
    mri_src = test_dir / 'func_epi.nii.gz'
    dst_dir = test_dir / 'watch'
    if not dst_dir.is_dir():
        dst_dir.mkdir()
    sim_suffix = 'sim_fMRI_{num:05d}.nii.gz'

    # Physio files
    ecg_src = test_dir / 'ECG.1D'
    resp_src = test_dir / 'Resp.1D'

    # Make a pseudo serial port to simulate the physio recording device
    master, slave = pty.openpty()
    s_name = os.ttyname(slave)
    try:
        m_name = os.ttyname(master)
    except Exception:
        m_name = 'Pseudo master'
    physio_port = f"{m_name}:{master}"
    print(s_name)

    recording_rate_ms = 25
    samples_to_average = 5

    mri_sim = rtMRISim(mri_src, dst_dir, sim_suffix)
    mri_sim.set_physio(ecg_src, resp_src, physio_port,
                       recording_rate_ms, samples_to_average)

    # Start physio feeding
    mri_sim.run_Physio('start')
    while mri_sim.physio_feeder is None or \
            not mri_sim.physio_feeder.is_alive():
        time.sleep(1)  # Wait for the physio feeding to start

    # Start MRI feeding
    mri_sim.run_MRI('start', imgType='NIfTI')
    while mri_sim.mri_feeder is None or \
            not mri_sim.mri_feeder.is_alive():
        time.sleep(1)  # Wait for the MRI feeding to start

    while mri_sim.mri_feeder.is_alive():
        time.sleep(1)

    mri_sim.run_Physio('stop')
