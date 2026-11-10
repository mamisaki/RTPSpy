#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTPSpy interface to the rt_physio process (rt_physio.py).

RtpTTLPhysio boots rt_physio as an independent process, which records the
scanner TTL and the physio signals (cardiac and respiration). Commands are
sent via the RPC socket, and the recorded signals are read directly from the
shared-memory ring buffers of rt_physio, e.g., to make the RETROICOR
regressors in real time (get_retrots).
"""

# %% import ===================================================================
from pathlib import Path
import os
import signal
import sys
import time
import traceback
import subprocess
import warnings

import numpy as np
from scipy.interpolate import interp1d

try:
    # Load modules from the same directory
    from .rtp_common import RTP
    from .rpc_socket_server import RPCSocketCom
    from .rtp_retrots import RtpRetroTS
    from .rt_physio import SharedMemoryRingBuffer

except Exception:
    # When this file is not loaded as a part of the package (debugging)
    from rtpspy.rtp_common import RTP
    from rtpspy.rpc_socket_server import RPCSocketCom
    from rtpspy.rtp_retrots import RtpRetroTS
    from rtpspy.rt_physio import SharedMemoryRingBuffer


# %% RtpTTLPhysio =============================================================
class RtpTTLPhysio(RTP):
    """
    Interface to the rt_physio process.
    """
    # Ring buffers shared by rt_physio
    RBUF_NAMES = ("ttl_onsets", "ttl_offsets", "card", "resp", "tstamp")

    # Seconds to wait for rt_physio to quit before it is terminated
    QUIT_TIMEOUT = 20

    # Parameters to save and log (get_params). They are the recording
    # parameters of rt_physio.
    PARAM_KEYS = ("device", "sample_freq", "buf_len_sec", "save_ttl")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(
        self,
        physio_log_file=None,
        rt_physio_address_name=["localhost", None, "RtTTLPhysioSocketServer"],
        config_path=Path.home() / ".RTPSpy" / "rtmri_config.json",
        **kwargs,
    ):
        """
        Parameters
        ----------
        physio_log_file : Path or str, optional
            Log file of rt_physio. The default is rt_physio.log in work_dir.
        rt_physio_address_name : list, optional
            [host, port, socket name] of the rt_physio RPC server. The port
            None is read from config_path.
        config_path : Path or str, optional
            Config file in which the RPC servers record their ports.
        **kwargs :
            Passed to RTP (e.g., work_dir).
        """
        super().__init__(**kwargs)

        self._retrots = RtpRetroTS()
        self._rt_physio_proc = None
        self.TTLPhysioCom = None
        self.sample_freq = None
        self._scan_onset = None  # Shared-memory buffer of the scan onset
        self._rbuf = {}  # Shared-memory ring buffers of the signals

        if physio_log_file is None:
            physio_log_file = Path(self.work_dir) / "rt_physio.log"
        self.physio_log_file = physio_log_file
        self.rt_physio_address_name = rt_physio_address_name
        self.config_path = config_path
        self.init_timeout = 10  # Seconds to wait for rt_physio to start

        self.init_rt_physio_access()

        self.scan_onset = 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def init_rt_physio_access(self):
        """Boot rt_physio and connect to its RPC server and buffers."""
        try:
            # Boot rt_physio as an independent process
            physio = Path(__file__).resolve().parent / "rt_physio.py"
            cmd = [
                sys.executable, str(physio),
                "--log_file", str(Path(self.physio_log_file).resolve()),
                "--rpc_socket_name", self.rt_physio_address_name[2],
                "--config_path", str(self.config_path),
            ]

            # rt_physio writes its log to physio_log_file. The output is
            # discarded; an unread PIPE would block the process when full.
            # A new session lets quit() stop rt_physio together with its
            # recording and plot subprocesses.
            self._rt_physio_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            if self._rt_physio_proc.poll() is not None:
                self._logger.error(
                    "rt_physio process terminated unexpectedly."
                )
                self._rt_physio_proc = None
                return

            self.TTLPhysioCom = RPCSocketCom(
                self.rt_physio_address_name, self.config_path
            )

            # Wait for the RPC server to start
            st = time.time()
            while (
                not self.TTLPhysioCom.rpc_ping() and
                (time.time() - st) < self.init_timeout
            ):
                time.sleep(0.1)

            # The ring buffers may not be made yet just after the start:
            # retry quietly until init_timeout, then warn if still not ready.
            while (
                not self.init_rbuf_access(quiet=True) and
                (time.time() - st) < self.init_timeout
            ):
                time.sleep(0.1)
            if not self._rbuf:
                self.init_rbuf_access()

        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            self._logger.error(f"Failed to connect to rt_physio: {errstr}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def init_rbuf_access(self, quiet=False):
        """Attach to the shared-memory buffers of rt_physio (the scan onset
        and the signal ring buffers) and get its sampling frequency.

        Parameters
        ----------
        quiet : bool
            Log a failure at DEBUG instead of WARNING (for a retried call).

        Returns
        -------
        bool
            True if the buffers are attached.
        """
        if self.TTLPhysioCom is None:
            return False

        log_fail = self._logger.debug if quiet else self._logger.warning
        if not self.TTLPhysioCom.rpc_ping():
            log_fail("rt_physio RPC server does not respond.")
            return False

        # Scan onset
        scan_onset = self.TTLPhysioCom.call_rt_proc(
            "GET_SCAN_ONSET", get_return=True, quiet=quiet
        )
        if scan_onset is None:
            log_fail(
                "Failed to get the scan onset from the rt_physio RPC server."
            )
            return False

        self._scan_onset = SharedMemoryRingBuffer(**scan_onset)

        # Signal ring buffers
        physio_rbufs = self.TTLPhysioCom.call_rt_proc(
            "GET_RBUF", get_return=True, quiet=quiet
        )
        if physio_rbufs is None or not set(self.RBUF_NAMES).issubset(
            physio_rbufs
        ):
            # Not ready yet (e.g., rt_physio is making the buffers). The
            # available property tries again later.
            log_fail(
                "Failed to get the ring buffers from the rt_physio RPC "
                "server."
            )
            return False

        self._rbuf = {
            lab: SharedMemoryRingBuffer(**rb)
            for lab, rb in physio_rbufs.items()
        }

        # Sampling frequency
        physio_params = self.TTLPhysioCom.call_rt_proc(
            ("GET_PARAMS", ("sample_freq",)), pkl=True, get_return=True
        )
        if physio_params is not None:
            self.sample_freq = physio_params.get("sample_freq", None)
        return True

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # getter, setter
    @property
    def available(self):
        """True if rt_physio is connected. The access to its buffers is
        retried here if it was not made at init (e.g., rt_physio was slow
        to start)."""
        if self.TTLPhysioCom is None:
            return False

        if self._scan_onset is None or len(self._rbuf) == 0:
            self.init_rbuf_access()

        return True

    @property
    def scan_onset(self):
        """Scan onset time (time.time()) shared with rt_physio. 0 if not
        set."""
        if self._scan_onset is None:
            self._logger.warning("Scan onset is not initialized.")
            return 0
        return self._scan_onset.get()[0]

    @scan_onset.setter
    def scan_onset(self, onset):
        if self._scan_onset is None and self.TTLPhysioCom is not None:
            self.init_rbuf_access()

        if self._scan_onset is not None:
            self._scan_onset.append(onset)
        elif onset != 0:
            self._logger.warning("Scan onset could not be sent to rt_physio.")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def quit(self):
        """Quit rt_physio. The process booted by this instance is stopped
        with its subprocesses, even if the QUIT call did not reach it."""
        if self.TTLPhysioCom is not None:
            try:
                self.TTLPhysioCom.call_rt_proc("QUIT")
            except Exception:
                pass
            self.TTLPhysioCom = None

        if self._rt_physio_proc is None:
            return

        proc = self._rt_physio_proc
        self._rt_physio_proc = None
        # rt_physio stops its recording and plot processes before exiting,
        # which can take several seconds.
        st = time.time()
        try:
            proc.wait(timeout=self.QUIT_TIMEOUT)
            self._logger.info(f"rt_physio quit in {time.time() - st:.1f} s.")
        except subprocess.TimeoutExpired:
            self._logger.warning(
                f"rt_physio did not quit in {self.QUIT_TIMEOUT} s. "
                "Terminating it."
            )

        # Stop the subprocesses left in the process group (the group id is
        # the pid, as start_new_session=True).
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(proc.pid, sig)
            except (ProcessLookupError, PermissionError):
                break
            try:
                proc.wait(timeout=3)
                time.sleep(0.5)  # Let subprocesses handle the signal
            except subprocess.TimeoutExpired:
                pass

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def show(self):
        """Show the physio plot window."""
        if self.available:
            self.TTLPhysioCom.call_rt_proc("SHOW")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def hide(self):
        """Hide the physio plot window."""
        if self.available:
            self.TTLPhysioCom.call_rt_proc("HIDE")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def move(self, geometry):
        """Set the geometry ('WxH+X+Y') of the physio plot window."""
        if self.available:
            self.TTLPhysioCom.call_rt_proc(("SET_GEOMETRY", geometry),
                                           pkl=True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def standby_scan(self):
        """Reset the scan onset and let rt_physio set it at the next TTL."""
        self.scan_onset = 0
        if self.available:
            params = {"scan_onset": 0, "wait_ttl_on": True}
            self.TTLPhysioCom.call_rt_proc(("SET_PARAMS", params), pkl=True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def release_standby_scan(self):
        """Stop waiting for the TTL of the scan start."""
        if self.available:
            self.TTLPhysioCom.call_rt_proc("CANCEL_WAIT_TTL")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def start_scan(self):
        if self.available:
            self.TTLPhysioCom.call_rt_proc("START_SCAN")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end_scan(self):
        if self.available:
            self.TTLPhysioCom.call_rt_proc("END_SCAN")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_physio_fname(self, fname_fmt):
        """Set the file name used when rt_physio saves the data at the
        scan end."""
        if self.available:
            self.TTLPhysioCom.call_rt_proc(("SET_FNAME_FMT", fname_fmt),
                                           pkl=True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save_physio_data(self, onset=None, series_duration=None,
                         fname_fmt=None):
        """Let rt_physio save the physio data of a scan."""
        if self.available:
            args = ("SAVE_PHYSIO_DATA", onset, series_duration, fname_fmt)
            self.TTLPhysioCom.call_rt_proc(args, pkl=True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def dump(self):
        """Return the recorded signals in the ring buffers.

        Returns
        -------
        dict or None
            "ttl_onsets", "ttl_offsets": sorted TTL times.
            "tstamp", "card", "resp": samples with a valid time stamp and
            valid values, sorted by time.
            None if rt_physio is not available or there is no sample.
        """
        if not self.available or len(self._rbuf) == 0:
            return None

        try:
            data = {lab: self._rbuf[lab].get().copy()
                    for lab in self.RBUF_NAMES}

            ttl_onsets = data["ttl_onsets"]
            ttl_offsets = data["ttl_offsets"]
            data["ttl_onsets"] = np.sort(ttl_onsets[~np.isnan(ttl_onsets)])
            data["ttl_offsets"] = np.sort(
                ttl_offsets[~np.isnan(ttl_offsets)])

            tstamp = data["tstamp"]
            valid = (
                ~np.isnan(tstamp)
                & ~np.isnan(data["card"])
                & ~np.isnan(data["resp"])
                & (tstamp > 0)
            )
            if not np.any(valid):
                return None

            sidx = np.argsort(tstamp[valid])
            for lab in ("tstamp", "card", "resp"):
                data[lab] = data[lab][valid][sidx]

            return data

        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            self._logger.error(f"Error dumping the ring buffers: {errstr}")
            return None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_retrots(self, tr, n_vol=np.inf, tshift=0, timeout=1.5):
        """Make the RETROICOR regressors from the scan onset.

        Parameters
        ----------
        tr : float
            Repetition time (s).
        n_vol : int, optional
            Number of volumes. The default (inf) uses all data recorded
            since the scan onset. Otherwise, it waits up to timeout seconds
            for the data of n_vol volumes.
        tshift : float, optional
            Slice timing shift passed to RtpRetroTS.RetroTs.
        timeout : float, optional
            Seconds to wait for the data of n_vol volumes.

        Returns
        -------
        array or None
            Regressors from RtpRetroTS.RetroTs. None if the scan onset is
            not set or the data are not enough.
        """
        if not self.available:
            return None

        onset = self.scan_onset
        if onset == 0:
            return None

        data = self.dump()
        if data is None:
            return None

        tstamp = data["tstamp"] - onset

        if np.isinf(n_vol):
            n_vol = int(np.nanmax(tstamp) // tr)
        else:
            # Wait for the data of n_vol volumes
            timeout_st = time.time()
            while (
                int(np.nanmax(tstamp) // tr) < n_vol
                and time.time() - timeout_st < timeout
            ):
                time.sleep(0.001)
                new_data = self.dump()
                if new_data is not None:
                    data = new_data
                    tstamp = data["tstamp"] - onset

            if int(np.nanmax(tstamp) // tr) < n_vol:
                self._logger.error(
                    "Not enough data to make the RETROICOR regressors"
                    f" within {timeout} s."
                )
                return None

        # dump() returns only samples without NaN, so only the time range
        # is selected here.
        mask = tstamp >= -tr
        resp = data["resp"][mask]
        card = data["card"][mask]
        tstamp = tstamp[mask]

        # Resample at the sampling frequency from the scan onset
        fs = self.sample_freq
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            res_t = np.arange(0, n_vol * tr + 1.0, 1.0 / fs)
            resp_res = interp1d(tstamp, resp, bounds_error=False)(res_t)
            resp_res = resp_res[~np.isnan(resp_res)]
            card_res = interp1d(tstamp, card, bounds_error=False)(res_t)
            card_res = card_res[~np.isnan(card_res)]

        return self._retrots.RetroTs(resp_res, card_res, tr, fs, tshift,
                                     n_vol)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        try:
            self.quit()
        except Exception:
            pass

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_params(self):
        """Return the recording parameters of rt_physio (PARAM_KEYS) to save
        and log. device is the recorder type in use. A value that cannot be
        read from rt_physio is None."""
        physio_keys = {"device": "_recorder_type",
                       "sample_freq": "sample_freq",
                       "buf_len_sec": "buf_len_sec", "save_ttl": "save_ttl"}
        opts = {k: getattr(self, k, None) for k in self.PARAM_KEYS}
        if self.TTLPhysioCom is not None:
            ret = self.TTLPhysioCom.call_rt_proc(
                ("GET_PARAMS", tuple(physio_keys[k] for k in self.PARAM_KEYS)),
                pkl=True, get_return=True)
            if isinstance(ret, dict):
                for k in self.PARAM_KEYS:
                    if ret.get(physio_keys[k]) is not None:
                        opts[k] = ret[physio_keys[k]]

        for k, v in opts.items():
            if isinstance(v, Path):
                opts[k] = str(v)

        return opts

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_param(self, attr, val, reset_fn=None, echo=False):
        """Set a parameter of PARAM_KEYS; others are ignored. The recording
        itself is set in rt_physio, so sample_freq is taken from rt_physio
        when it is connected (a loaded value must not change the sampling
        frequency used for the RETROICOR regressors)."""
        self._logger.debug(f"set_param: {attr} = {val}")
        if attr not in self.PARAM_KEYS:
            return

        if attr == "sample_freq" and self.TTLPhysioCom is not None:
            ret = self.TTLPhysioCom.call_rt_proc(
                ("GET_PARAMS", ("sample_freq",)), pkl=True, get_return=True)
            if isinstance(ret, dict) and ret.get("sample_freq") is not None:
                val = ret["sample_freq"]

        setattr(self, attr, val)
        if echo:
            print(f"{self.__class__.__name__}.{attr} = {getattr(self, attr)}")
