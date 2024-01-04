#!/usr/bin/env ipython3
# -*- coding: utf-8 -*-
"""
@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import numpy as np
import pickle
import sys

try:
    from .rtp_common import RTP
except Exception:
    from rtp_common import RTP


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
