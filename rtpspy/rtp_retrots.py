# -*- coding: utf-8 -*-
"""
RTP retrots for making online RETROICOR and RVT regressors

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
import numpy as np
import ctypes
import pickle
import sys
from pathlib import Path
import time

if sys.platform == 'linux':
    lib_name = 'librtp.so'
elif sys.platform == 'darwin':
    lib_name = 'librtp.dylib'

try:
    librtp_path = str(Path(__file__).absolute().parent / lib_name)
except Exception:
    librtp_path = f'./{lib_name}'


# %% RetroTSOpt class =========================================================
class RetroTSOpt(ctypes.Structure):
    """
    RetroTSOpt struct class defined in rtp_restrots.h

    def struct {
        float VolTR; // TR of MRI acquisition in second
        float PhysFS; // Sampling frequency of physiological signal data
        float tshift;  // slice timing offset. 0.0 is the first slice
        float zerophaseoffset;
        int RVTshifts[256];
        int RVTshiftslength;
        int RespCutoffFreq;
        int fcutoff;
        int AmpPhase;
        int CardCutoffFreq;
        char ResamKernel[256];
        int FIROrder;
        int as_windwidth;
        int as_percover;
        int as_fftwin;
        } RetroTSOpt;

    VolTR, PhysFS, and tshift (=0 in default) sholud be be set manually. Other
    fields are initialized and used inside the rtp_retrots funtion.
    """

    _fields_ = [
            ('VolTR', ctypes.c_float),  # Volume TR in seconds
            ('PhysFS', ctypes.c_float),  # Sampling frequency (Hz)
            ('tshift', ctypes.c_float),  # slice timing offset
            ('zerophaseoffset', ctypes.c_float),
            ('RVTshifts', ctypes.c_int*256),
            ('RVTshiftslength', ctypes.c_int),
            ('RespCutoffFreq', ctypes.c_int),
            ('fcutoff', ctypes.c_int),  # cut off frequency for filter
            ('AmpPhase', ctypes.c_int),
            ('CardCutoffFreq', ctypes.c_int),
            ('ResamKernel', ctypes.c_char*256),
            ('FIROrder', ctypes.c_int),
            ('as_windwidth', ctypes.c_int),
            ('as_percover', ctypes.c_int),
            ('as_fftwin', ctypes.c_int)
            ]

    def __init__(self, VolTR, PhysFS, tshift=0):
        self.VolTR = np.float32(VolTR)
        self.PhysFS = np.float32(PhysFS)
        self.tshift = np.float32(tshift)


# %% RtpRetrots class ========================================================
class RtpRetrots:
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self):
        librtp = ctypes.cdll.LoadLibrary(librtp_path)

        # -- Define librtp.rtp_retrots setups --
        self.rtp_retrots = librtp.rtp_retrots
        """ C definition
        int rtp_retrots(RetroTSOpt *rtsOpt, double *Resp, double *Card,
                        int len, *regOut);
        """

        self.rtp_retrots.argtypes = \
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
             ctypes.c_void_p]

        self._rtsOpt = None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def setup_RetroTSOpt(self, TR, PhysFS, tshift=0):
        self.TR = TR
        self.PhysFS = PhysFS
        self.tshift = tshift
        self._rtsOpt = RetroTSOpt(TR, PhysFS, tshift)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def do_proc(self, Resp, Card, TR, PhysFS, tshift=0):
        """
        RetroTS process function, which will be called from RTP_PHYSIO instance

        Options
        -------
        Resp : array
            Respiration signal data array
        Card : array
            Cardiac sighnal data array

        Retrun
        ------
        RetroTS regressor
        """

        # Get pointer ot Resp and Card data array
        Resp_arr = np.array(Resp, dtype=np.float64)
        Resp_ptr = Resp_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        Card_arr = np.array(Card, dtype=np.float64)
        Card_ptr = Card_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Set data length and prepare output array
        dlenR = len(Resp)
        dlenC = len(Card)
        dlen = min(dlenR, dlenC)

        outlen = int((dlen * 1.0/PhysFS) / TR)
        regOut = np.ndarray((outlen, 13), dtype=np.float32)
        regOut_ptr = regOut.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Set rtsOpt
        if self._rtsOpt is None:
            self.setup_RetroTSOpt(TR, PhysFS, tshift)
        else:
            self._rtsOpt.VolTR = TR
            self.PhysFS = PhysFS
            self.tshift = tshift

        self.rtp_retrots(ctypes.byref(self._rtsOpt), Resp_ptr, Card_ptr, dlen,
                         regOut_ptr)

        return regOut

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_param(self, attr, val=None, echo=False):
        # -- Set value --
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


# %% __main__ (test) ==========================================================
if __name__ == '__main__':

    import warnings
    from scipy import interpolate

    card_f = Path('/data/rt/S20231229091434/Card_500Hz_ser-2.1D')
    resp_f = Path('/data/rt/S20231229091434/Resp_500Hz_ser-2.1D')
    resp = np.loadtxt(resp_f)
    card = np.loadtxt(card_f)

    # Resample
    PhysFS = 100

    tstamp = np.arange(0, len(resp)/500, 1.0/500)
    xt = np.arange(0, tstamp[-1], 1.0/PhysFS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        f = interpolate.interp1d(tstamp, card, bounds_error=False)
        Card = f(xt)
        f = interpolate.interp1d(tstamp, resp, bounds_error=False)
        Resp = f(xt)

    TR = 2.0
    tshift = 0

    restrots = RtpRetrots()

    for n in range(20, 201):
        num_points = int(n * TR * PhysFS)
        reg = restrots.do_proc(Resp[:num_points], Card[:num_points], TR,
                               PhysFS)
