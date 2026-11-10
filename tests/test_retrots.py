"""RtpRetroTS (RETROICOR regressors) on tests/ECG.1D and tests/Resp.1D
(40 Hz)."""
import time

import numpy as np
import pytest
from scipy.signal import welch

from rtpspy.rtp_retrots import RtpRetroTS

FS = 40.0
TR = 2.0


@pytest.fixture(scope="module")
def physio(test_dir):
    card = np.loadtxt(test_dir / "ECG.1D")
    resp = np.loadtxt(test_dir / "Resp.1D")
    return card, resp


def _peak_freq(x, fmin, fmax):
    f, p = welch(x - np.mean(x), fs=FS, nperseg=4096)
    m = (f > fmin) & (f < fmax)
    return f[m][np.argmax(p[m])]


def test_regressors_shape_and_range(physio):
    card, resp = physio
    reg = RtpRetroTS().RetroTs(resp, card, TR=TR, physFS=FS)
    nvol = int(np.floor(len(card) / FS / TR))
    assert reg.shape == (nvol, 8)
    assert np.all(np.isfinite(reg))
    assert np.abs(reg).max() <= 1.0 + 1e-12
    # sin^2 + cos^2 = 1 for each phase and harmonic
    for i in range(0, 8, 2):
        assert np.allclose(reg[:, i] ** 2 + reg[:, i + 1] ** 2, 1.0)
    assert reg.std(0).min() > 0.3  # the regressors vary


def test_phase_ranges_and_rates(physio):
    card, resp = physio
    card_ph, resp_ph = RtpRetroTS().phases(card, resp, FS)
    assert card_ph.shape == resp_ph.shape == card.shape
    assert card_ph.min() >= 0 and card_ph.max() < 2 * np.pi
    assert resp_ph.min() >= -np.pi and resp_ph.max() <= np.pi

    # Cardiac cycles (phase wraps) per minute: physiological and equal to
    # the dominant frequency of the ECG signal
    dur = len(card) / FS
    hr = (np.diff(card_ph) < -np.pi).sum() / dur * 60
    assert 40 < hr < 150
    assert abs(hr - _peak_freq(card, 0.6, 3.0) * 60) < 6
    # Respiration phase follows the respiration rate
    rr_sig = _peak_freq(resp, 0.05, 0.7)
    assert 0.05 < rr_sig < 0.7
    assert abs(_peak_freq(np.sin(resp_ph), 0.05, 0.7) - rr_sig) < 0.02


def test_nvol_tshift_and_nan(physio):
    card, resp = physio
    rts = RtpRetroTS()
    reg = rts.RetroTs(resp, card, TR=TR, physFS=FS)
    reg50 = rts.RetroTs(resp, card, TR=TR, physFS=FS, Nvol=50)
    assert reg50.shape == (50, 8) and np.allclose(reg50, reg[:50])
    reg_s = rts.RetroTs(resp, card, TR=TR, physFS=FS, tshift=1.0)
    assert reg_s.shape == reg.shape and not np.allclose(reg_s, reg)

    c = card.copy()
    r = resp.copy()
    c[1000:1010] = np.nan
    r[2000:2040] = np.nan
    reg_nan = rts.RetroTs(r, c, TR=TR, physFS=FS)
    assert np.all(np.isfinite(reg_nan))

    # Too short data: zero phases, no error
    card_ph, resp_ph = rts.phases(card[:20], resp[:20], FS)
    assert np.all(card_ph == 0) and np.all(resp_ph == 0)


def test_speed(physio):
    """Fast enough to be called at every volume online."""
    card, resp = physio
    rts = RtpRetroTS()
    rts.RetroTs(resp, card, TR=TR, physFS=FS)  # warm up
    st = time.perf_counter()
    for _ in range(5):
        rts.RetroTs(resp, card, TR=TR, physFS=FS)
    assert (time.perf_counter() - st) / 5 < 0.5
