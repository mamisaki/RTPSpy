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

from scipy.signal import lfilter, firwin
from numpy.fft import fft, ifft
from scipy.interpolate import interp1d

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

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def peak_finder(self, v, phys_fs, zero_phase_offset=0.5,
                    resample_fs=(1 / 0.025),
                    frequency_cutoff=10,
                    fir_order=80,
                    interpolation_style="linear",
                    demo=0,
                    as_window_width=0,
                    as_percover=0,
                    as_fftwin=0,
                    sep_dups=0,
                    no_dups=True):

        # De-mean
        v = v - np.mean(v)

        # Filtering with frequency_cutoff
        nyquist_filter = phys_fs / 2.0
        w = frequency_cutoff / nyquist_filter
        # FIR filter of order 40
        b = firwin(numtaps=(fir_order + 1), cutoff=w, window="hamming")
        b = np.array(b)

        # Filter both ways to cancel phase shift
        v = lfilter(b, 1, v, axis=0)
        v = np.flipud(v)
        v = lfilter(b, 1, v)
        v = np.flipud(v)

        # analytic_signal
        fv = fft(v)
        # zero negative frequencies, double positive frequencies
        wind = np.zeros(v.shape)
        nv = len(v)
        if nv % 2 == 0:
            wind[0] = 1  # keep DC
            wind[(nv//2)] = 1
            wind[1:(nv//2)] = 2  # double pos. freq
        else:
            wind[0] = 1
            wind[1:(nv+1)//2] = 2
        x = ifft(fv * wind)

        # Find peaks
        nt = len(x)
        fsi = 1.0 / phys_fs
        t = np.arange(0, len(x)*fsi, fsi)
        img_chg = (x[:-1].imag * x[1:].imag)
        iz = np.nonzero(img_chg <= 0)[0]
        polall = -np.sign(img_chg)
        pk = (x[iz]).real
        pol = polall[iz]
        tiz = t[iz]

        # Polishing: Find a peak around the identified points
        window_width = 0.2  # Window for adjusting peak location in seconds
        nww = int(np.ceil((window_width / 2) * phys_fs))
        pkp = pk
        for ii in range(len(iz)):
            n0 = int(max(2, iz[ii] - nww))
            n1 = int(min(nt, iz[ii] + nww))
            wval = (x[n0:n1+1]).real
            if pol[ii] > 0:
                xx, ixx = np.max(wval), np.argmax(wval)
            else:
                xx, ixx = np.min(wval), np.argmin(wval)
            iz[ii] = n0 + ixx - 1
            pkp[ii] = xx
        tizp = iz

        ppp = np.nonzero(pol > 0)[0]
        p_trace = pkp[ppp]
        tp_trace = tizp[ppp]

        npp = np.nonzero(pol < 0)[0]
        n_trace = pkp[npp]
        tn_trace = tizp[npp]

        # remove duplicates
        okflag = (np.diff(tp_trace) > 0.3) & (np.diff(tn_trace) != 0)
        okflag = [True, *okflag]
        p_trace = p_trace[okflag]
        tp_trace = tp_trace[okflag]
        n_trace = n_trace[okflag]
        tn_trace = tn_trace[okflag]

        # Calculate the period
        prd = np.diff(tp_trace)
        p_trace_mid_prd = (p_trace[1:] + p_trace[:-1]) / 2
        t_mid_prd = (tp_trace[1:] + tp_trace[:-1]) / 2

        # Interpolate envelope
        step_interval = 1.0 / resample_fs
        step_size = int(np.max(t)/step_interval + 0.00001) + 1
        tR = np.arange(0, step_size*step_interval+0.00001, step_interval)

        p_trace_r = interp1d(tp_trace, p_trace, interpolation_style,
                             bounds_error=False)(tR)
        pre_idx = np.argwhere(tR < tp_trace[0])
        if len(pre_idx):
            p_trace_r[pre_idx] = p_trace_r[pre_idx[-1]+1]
        post_idx = np.argwhere(tR > tp_trace[-1])
        if len(post_idx):
            p_trace_r[post_idx] = p_trace_r[post_idx[0]-1]

        n_trace_r = interp1d(tn_trace, n_trace, interpolation_style,
                             bounds_error=False)(tR)
        pre_idx = np.argwhere(tR < tn_trace[0])
        if len(pre_idx):
            n_trace_r[pre_idx] = n_trace_r[pre_idx[-1]+1]
        post_idx = np.argwhere(tR > tn_trace[-1])
        if len(post_idx):
            n_trace_r[post_idx] = n_trace_r[post_idx[0]-1]

        prdR = interp1d(t_mid_prd, prd, interpolation_style,
                        bounds_error=False)(tR)
        pre_idx = np.argwhere(tR < t_mid_prd[0])
        if len(pre_idx):
            prdR[pre_idx] = prdR[pre_idx[-1]+1]
        post_idx = np.argwhere(tR > t_mid_prd[-1])
        if len(post_idx):
            prdR[post_idx] = prdR[post_idx[0]-1]

        return    

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def phase_estimator(self,)
    if amp_type == 0:
        # Calculate the phase of the trace, with the peak to be the start of the phase
        nptrc = len(phasee["tp_trace"])
        phasee["phase"] = -2 * ones(size(phasee["t"]))
        i = 0
        j = 0
        while i <= (nptrc - 2):
            while phasee["t"][j] < phasee["tp_trace"][i + 1]:
                if phasee["t"][j] >= phasee["tp_trace"][i]:
                    # Note: Using a constant 244 period for each interval
                    # causes slope discontinuity within a period.
                    # One should resample period[i] so that it is
                    # estimated at each time in phasee['t'][j],
                    # dunno if that makes much of a difference in the end however.
                    if j == 10975:
                        pass
                    phasee["phase"][j] = (
                        phasee["t"][j] - phasee["tp_trace"][i]
                    ) / phasee["prd"][i] + phasee["zero_phase_offset"]
                    if phasee["phase"][j] < 0:
                        phasee["phase"][j] = -phasee["phase"][j]
                    if phasee["phase"][j] > 1:
                        phasee["phase"][j] -= 1
                j += 1
            if i == 124:
                pass
            i += 1

        # Remove the points flagged as unset
        temp = nonzero(phasee["phase"] < -1)
        phasee["phase"][temp] = 0.0
        # Change phase to radians
        phasee["phase"] = phasee["phase"] * 2 * pi
    else:  # phase based on amplitude
        # at first scale to the max
        mxamp = max(phasee["p_trace"])
        phasee["phase_pol"] = []
        gR = z_scale(phasee["v"], 0, mxamp)  # Scale, per Glover 2000's paper
        bins = arange(0.01, 1.01, 0.01) * mxamp
        hb_value = my_hist(gR, bins)
        # hb_value = histogram(gR, bins)
        if phasee["show_graphs"] == 1:
            center = (bins[:-1] + bins[1:]) / 2
            plt.bar(center, hb_value[: len(hb_value) - 1])  # , align='center')
            plt.show()
        # find the polarity of each time point in v
        i = 0
        itp = 0
        inp = 0
        tp = phasee["tp_trace"][0]
        tn = phasee["tn_trace"][0]
        while (
            (i <= len(phasee["v"])) and (phasee["t"][i] < tp) and (phasee["t"][i] < tn)
        ):
            phasee["phase_pol"].append(0)
            i += 1
        if tp < tn:
            # Expiring phase (peak behind us)
            cpol = -1
            itp = 1
        else:
            # Inspiring phase (bottom behind us)
            cpol = 1
            inp = 1
        phasee["phase_pol"] = zeros(
            size(phasee["v"])
        )  # Not sure why you would replace the
        # list that you created 10 lines prior to this
        # Add a fake point to tptrace and tntrace to avoid ugly if statements
        phasee["tp_trace"] = append(phasee["tp_trace"], phasee["t"][-1])
        phasee["tn_trace"] = append(phasee["tn_trace"], phasee["t"][-1])
        while i < len(phasee["v"]):
            phasee["phase_pol"][i] = cpol
            if phasee["t"][i] == phasee["tp_trace"][itp]:
                cpol = -1
                itp = min((itp + 1), (len(phasee["tp_trace"]) - 1))
            elif phasee["t"][i] == phasee["tn_trace"][inp]:
                cpol = 1
                inp = min((inp + 1), (len(phasee["tn_trace"]) - 1))
            # cpol, inp, itp, i, R
            i += 1
        phasee["tp_trace"] = delete(phasee["tp_trace"], -1)
        phasee["tn_trace"] = delete(phasee["tn_trace"], -1)
        if phasee["show_graphs"] == 1:
            # clf
            plt.plot(phasee["t"], gR, "b")
            ipositive = nonzero(phasee["phase_pol"] > 0)
            ipositive = ipositive[0]
            ipositive_x = []
            for i in ipositive:
                ipositive_x.append(phasee["t"][i])
            ipositive_y = zeros(size(ipositive_x))
            ipositive_y.fill(0.55 * mxamp)
            plt.plot(ipositive_x, ipositive_y, "r.")
            inegative = nonzero(phasee["phase_pol"] < 0)
            inegative = inegative[0]
            inegative_x = []
            for i in inegative:
                inegative_x.append(phasee["t"][i])
            inegative_y = zeros(size(inegative_x))
            inegative_y.fill(0.45 * mxamp)
            plt.plot(inegative_x, inegative_y, "g.")
            plt.show()
        # Now that we have the polarity, without computing sign(dR/dt)
        #   as in Glover et al 2000, calculate the phase per eq. 3 of that paper
        # First the sum in the numerator
        for i, val in enumerate(gR):
            gR[i] = round(val / mxamp * 100) + 1
        gR = clip(gR, 0, 99)
        shb = sum(hb_value)
        hbsum = []
        hbsum.append(float(hb_value[0]) / shb)
        for i in range(1, 100):
            hbsum.append(hbsum[i - 1] + (float(hb_value[i]) / shb))
        for i in range(len(phasee["t"])):
            phasee["phase"].append(pi * hbsum[int(gR[i]) - 1] * phasee["phase_pol"][i])
        phasee["phase"] = array(phasee["phase"])

    # Time series time vector
    phasee["time_series_time"] = arange(
        0, (max(phasee["t"]) - 0.5 * phasee["volume_tr"]), phasee["volume_tr"]
    )
    # Python uses half open ranges, so we need to catch the case when the stop
    # is evenly divisible by the step and add one more to the time series in
    # order to match Matlab, which uses closed ranges  1 Jun 2017 [D Nielson]
    if (max(phasee["t"]) - 0.5 * phasee["volume_tr"]) % phasee["volume_tr"] == 0:
        phasee["time_series_time"] = append(
            phasee["time_series_time"],
            [phasee["time_series_time"][-1] + phasee["volume_tr"]],
        )
    phasee["phase_slice"] = zeros(
        (len(phasee["time_series_time"]), phasee["number_of_slices"])
    )
    phasee["phase_slice_reg"] = zeros(
        (len(phasee["time_series_time"]), 4, phasee["number_of_slices"])
    )
    for i_slice in range(phasee["number_of_slices"]):
        tslc = phasee["time_series_time"] + phasee["slice_offset"][i_slice]
        for i in range(len(phasee["time_series_time"])):
            imin = argmin(abs(tslc[i] - phasee["t"]))
            # mi = abs(tslc[i] - phasee['t']) # probably not needed
            phasee["phase_slice"][i, i_slice] = phasee["phase"][imin]
        # Make regressors for each slice
        phasee["phase_slice_reg"][:, 0, i_slice] = sin(
            phasee["phase_slice"][:, i_slice]
        )
        phasee["phase_slice_reg"][:, 1, i_slice] = cos(
            phasee["phase_slice"][:, i_slice]
        )
        phasee["phase_slice_reg"][:, 2, i_slice] = sin(
            2 * phasee["phase_slice"][:, i_slice]
        )
        phasee["phase_slice_reg"][:, 3, i_slice] = cos(
            2 * phasee["phase_slice"][:, i_slice]
        )

    if phasee["quiet"] == 0 and phasee["show_graphs"] == 1:
        print("--> Calculated phase")
        plt.subplot(413)
        a = divide(divide(phasee["phase"], 2), pi)
        plt.plot(phasee["t"], divide(divide(phasee["phase"], 2), pi), "m")
        if "phase_r" in phasee:
            plt.plot(phasee["tR"], divide(divide(phasee["phase_r"], 2), pi), "m-.")
        plt.subplot(414)
        plt.plot(
            phasee["time_series_time"],
            phasee["phase_slice"][:, 1],
            "ro",
            phasee["time_series_time"],
            phasee["phase_slice"][:, 2],
            "bo",
            phasee["time_series_time"],
            phasee["phase_slice"][:, 2],
            "b-",
        )
        plt.plot(phasee["t"], phasee["phase"], "k")
        # grid on
        # title it
        plt.title(phasee["v_name"])
        plt.show()
        # Need to implement this yet
        # if phasee['Demo']:
        # uiwait(msgbox('Press button to resume', 'Pausing', 'modal'))
    return phasee


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
