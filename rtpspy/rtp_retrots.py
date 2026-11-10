# -*- coding: utf-8 -*-
"""
RtpRetroTS: online RETROICOR regressors

At each volume, the phases are estimated from the data available so far.
The cardiac phase is linear between the cardiac peaks, and the respiration
phase is the histogram-equalized amplitude signed by its slope (Glover et
al., 2000). The processing is designed for real-time use:
- The cardiac peaks are detected in a band-pass filtered signal around the
  dominant cardiac frequency. The end of the data is mirrored before
  filtering so that a peak near the end can be detected, and the phase after
  the last peak is extended with the recent period.
- The respiration signal is not high-pass filtered, as a zero-phase
  high-pass filter distorts the end of the data.
- The signals are downsampled to <= 50 Hz, and all steps are vectorized.

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


# %% RtpRetroTS class ========================================================
class RtpRetroTS():
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, max_proc_freq=50.0,
                 card_freq_range=(0.67, 2.5), card_freq_win=60.0,
                 card_band=(0.5, 2.5), card_min_interval=0.6,
                 card_prominence=0.3, card_edge_pad=1.0, card_period_n=8,
                 resp_lowpass=1.0, resp_slope_lowpass=1.0, resp_clip=1.0,
                 resp_nbin=100):
        """
        Parameters
        ----------
        max_proc_freq : float
            The signals are downsampled to this frequency (Hz) or lower.
        card_freq_range : (float, float)
            Search range (Hz) of the dominant cardiac frequency, f0.
        card_freq_win : float
            Length (s) of the latest data used to estimate f0.
        card_band : (float, float)
            Cardiac band-pass range, relative to f0.
        card_min_interval : float
            Minimum interval between cardiac peaks, relative to 1/f0.
        card_prominence : float
            Minimum peak prominence, relative to the SD of the filtered
            signal.
        card_edge_pad : float
            Length of the mirrored data at the end, relative to 1/f0.
        card_period_n : int
            Number of periods whose median extends the phase before the
            first peak and after the last peak.
        resp_lowpass : float
            Low-pass frequency (Hz) of the respiration signal.
        resp_slope_lowpass : float
            Low-pass frequency (Hz) of the signal for the slope (phase sign).
        resp_clip : float
            Percentile clipped at both ends of the amplitude histogram.
        resp_nbin : int
            Number of histogram bins.
        """
        self.max_proc_freq = max_proc_freq
        self.card_freq_range = card_freq_range
        self.card_freq_win = card_freq_win
        self.card_band = card_band
        self.card_min_interval = card_min_interval
        self.card_prominence = card_prominence
        self.card_edge_pad = card_edge_pad
        self.card_period_n = card_period_n
        self.resp_lowpass = resp_lowpass
        self.resp_slope_lowpass = resp_slope_lowpass
        self.resp_clip = resp_clip
        self.resp_nbin = resp_nbin

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def RetroTs(self, resp, card, TR, physFS, tshift=0, Nvol=None):
        """
        Make RETROICOR regressors.

        Parameters
        ----------
        resp, card : array
            Respiration and cardiac signals from the scan onset. NaNs are
            interpolated.
        TR : float
            Repetition time (s).
        physFS : float
            Sampling frequency (Hz) of the signals.
        tshift : float
            Time (s) in a volume at which the phases are taken (e.g., the
            reference time of slice-timing correction).
        Nvol : int, optional
            Number of volumes. Default: all volumes covered by the data.

        Returns
        -------
        reg : array (Nvol x 8)
            sin(p), cos(p), sin(2p), cos(2p) of the respiration phase, and
            the same of the cardiac phase.
        """
        card_ph, resp_ph = self.phases(card, resp, physFS)
        if Nvol is None:
            Nvol = int(np.floor(len(card_ph) / physFS / TR))

        idx = np.round((np.arange(Nvol) * TR + tshift) * physFS).astype(int)
        idx = np.clip(idx, 0, len(card_ph) - 1)
        reg = []
        for ph in (resp_ph[idx], card_ph[idx]):
            reg.extend([np.sin(ph), np.cos(ph), np.sin(2 * ph),
                        np.cos(2 * ph)])

        return np.stack(reg, axis=1)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def phases(self, card, resp, fs):
        """Sample-wise cardiac and respiration phases (rad)."""
        n = len(card)
        k = max(1, int(np.ceil(fs / self.max_proc_freq)))
        fsd = fs / k
        c = self._downsample(self._fill_nan(card), k)
        r = self._downsample(self._fill_nan(resp), k)
        if len(c) < 3 * fsd:  # Too short to estimate the phases
            return np.zeros(n), np.zeros(n)

        card_ph = self.card_phase(c, fsd)
        resp_ph = self.resp_phase(r, fsd)
        idx = np.minimum(np.arange(n) // k, len(c) - 1)
        return card_ph[idx], resp_ph[idx]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def card_phase(self, x, fs):
        """Cardiac phase (0 to 2pi), linear between the detected peaks."""
        n = len(x)
        f0 = self._dominant_freq(x, fs, *self.card_freq_range,
                                 self.card_freq_win)

        # Mirror the end so that a peak near the end can be detected
        npad = min(n - 2, int(self.card_edge_pad / f0 * fs))
        if npad > 0:
            x = np.concatenate([x, x[-2:-npad - 2:-1]])

        xf = self._filter(x, fs, self.card_band[0] * f0,
                          self.card_band[1] * f0)
        dist = max(1, int(self.card_min_interval / f0 * fs))
        prom = self.card_prominence * np.std(xf[:n])
        peaks, _ = find_peaks(xf, distance=dist, prominence=prom)
        peaks = peaks[peaks < n].astype(float)
        if len(peaks) < 2:
            return np.zeros(n)

        # Phase is linear between the peaks. Before the first and after the
        # last peak, pseudo peaks are placed with the median period.
        prd = np.diff(peaks)
        p0 = np.median(prd[:self.card_period_n])
        p1 = np.median(prd[-self.card_period_n:])
        pre = peaks[0] - p0 * np.arange(int(np.ceil(peaks[0] / p0)) + 1,
                                        0, -1)
        post = peaks[-1] + p1 * np.arange(
            1, int(np.ceil((n - peaks[-1]) / p1)) + 2)
        pk = np.concatenate([pre, peaks, post])
        cyc = np.interp(np.arange(n), pk, np.arange(len(pk)))
        return 2 * np.pi * (cyc - np.floor(cyc))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def resp_phase(self, x, fs):
        """Respiration phase (-pi to pi) from the equalized amplitude."""
        xf = self._filter(x, fs, None, self.resp_lowpass)

        # Histogram-equalized amplitude (0 to 1)
        lo, hi = np.percentile(xf, [self.resp_clip, 100 - self.resp_clip])
        if hi <= lo:
            return np.zeros(len(x))
        edges = np.linspace(lo, hi, self.resp_nbin + 1)
        cnt, _ = np.histogram(np.clip(xf, lo, hi), edges)
        cum = np.concatenate([[0], np.cumsum(cnt)]) / max(cnt.sum(), 1)
        amp = np.interp(np.clip(xf, lo, hi), edges, cum)

        # Sign: positive while inhaling (rising), negative while exhaling
        slope = np.gradient(self._filter(x, fs, None,
                                         self.resp_slope_lowpass))
        return np.pi * amp * np.where(slope >= 0, 1.0, -1.0)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def _fill_nan(x):
        """Linearly interpolate non-finite values."""
        x = np.array(x, dtype=float)
        bad = ~np.isfinite(x)
        if bad.all():
            return np.zeros_like(x)
        if bad.any():
            i = np.arange(len(x))
            x[bad] = np.interp(i[bad], i[~bad], x[~bad])
        return x

    @staticmethod
    def _downsample(x, k):
        """Mean of every k samples."""
        if k <= 1:
            return x
        m = len(x) // k * k
        y = x[:m].reshape(-1, k).mean(axis=1)
        if m < len(x):
            y = np.append(y, x[m:].mean())
        return y

    @staticmethod
    def _dominant_freq(x, fs, fmin, fmax, win_sec=None):
        """Frequency of the spectral peak in [fmin, fmax] of the last
        win_sec seconds."""
        if win_sec is not None:
            x = x[-int(win_sec * fs):]
        x = x - x.mean()
        nfft = max(len(x), int(fs * 64))  # zero-pad for finer resolution
        X = np.abs(np.fft.rfft(x * np.hanning(len(x)), nfft))
        f = np.fft.rfftfreq(nfft, 1 / fs)
        m = (f >= fmin) & (f <= fmax)
        return f[m][np.argmax(X[m])]

    @staticmethod
    def _filter(x, fs, lo, hi, order=2):
        """Zero-phase Butterworth low-pass (lo=None) or band-pass filter."""
        hi = min(hi, 0.45 * fs)
        if lo is None or lo <= 0:
            sos = butter(order, hi, btype="lowpass", fs=fs, output="sos")
        else:
            sos = butter(order, [lo, hi], btype="bandpass", fs=fs,
                         output="sos")
        return sosfiltfilt(sos, x)


# %% SavedPhysio ==============================================================
class SavedPhysio():
    """
    RETROICOR regressors from saved physio signals, for simulations.

    Replaces RtpTTLPhysio as RtpRegress.rtp_physio. At each volume, only the
    samples recorded up to the end of that volume are used, as
    RtpTTLPhysio.get_retrots does during a scan, so the phases near the end
    are estimated without future samples.
    """

    def __init__(self, card, resp, sample_freq):
        self.card = np.asarray(card, dtype=float)
        self.resp = np.asarray(resp, dtype=float)
        self.sample_freq = float(sample_freq)
        self._retrots = RtpRetroTS()

    def get_retrots(self, tr, n_vol=np.inf, tshift=0, timeout=None):
        """Regressors of n_vol volumes from the scan onset (the start of the
        signals). None if the signals are not long enough."""
        n_all = min(len(self.card), len(self.resp))
        if np.isinf(n_vol):
            n_vol = int(n_all / self.sample_freq / tr)
        n_smp = int(round(n_vol * tr * self.sample_freq))
        if n_smp > n_all:
            return None  # Not recorded yet
        return self._retrots.RetroTs(self.resp[:n_smp], self.card[:n_smp],
                                     tr, self.sample_freq, tshift, n_vol)


# %% __main__ (test) ==========================================================
if __name__ == '__main__':
    from pathlib import Path

    test_dir = Path(__file__).resolve().parent.parent / 'tests'
    card = np.loadtxt(test_dir / 'ECG.1D')
    resp = np.loadtxt(test_dir / 'Resp.1D')
    rtp_retrots = RtpRetroTS()
    reg = rtp_retrots.RetroTs(resp, card, TR=2.0, physFS=40)
    print(reg.shape)
