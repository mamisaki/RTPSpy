"""RtpRegress on synthetic data: online GLM residuals vs offline numpy least
squares (including the ring buffer for scans longer than the buffer), the
SVD solver, online QC (FD, DVARS, tSNR) and FD censoring."""
import numpy as np
import nibabel as nib
import pandas as pd
import pytest
import torch

import rtpspy.rtp_regress as rr
from rtpspy.rtp_regress import RtpRegress, lstsq_SVDsolver

SHAPE = (16, 16, 8)
T = 120
TR = 2.0


# %% Helpers ==================================================================
class FakeVolreg:
    """Stand-in for RtpVolreg: provides the motion parameters."""

    def __init__(self, mot):
        self._motion = mot


class FakePhysio:
    """Stand-in for RtpTTLPhysio: provides the RETROICOR regressors.
    get_retrots returns None (no physio data) for the volumes in none_at."""

    def __init__(self, reg, none_at=()):
        self.reg = reg
        self.none_at = set(none_at)

    def get_retrots(self, tr, n_vol, tshift, timeout=1.0):
        if n_vol - 1 in self.none_at:
            return None
        return self.reg[:n_vol].copy()


class Recorder:
    """Downstream process that records the received volumes and QC."""

    def __init__(self):
        self.out = {}
        self.qc = {}
        self.next_proc = None

    def ready_proc(self):
        return True

    def do_proc(self, img, vol_idx=None, pre_proc_time=None, **kwargs):
        self.out[vol_idx] = np.asanyarray(img.dataobj).copy()
        if "qc" in kwargs:
            self.qc[vol_idx] = kwargs["qc"]

    def end_reset(self):
        return None


def make_data(rng, mot_sd=0.1, spikes=()):
    """Synthetic fMRI data (T x SHAPE), motion (T x 6, the last one constant)
    and RETROICOR-like regressors (T x 8). Each volume in spikes gets a
    one-volume displacement (high FD at s and s + 1)."""
    t = np.arange(T)
    mot = rng.normal(0, mot_sd, (T, 6)).cumsum(0).astype(np.float32)
    mot[:, 5] = 0.0  # constant regressor (std 0)
    for s in spikes:
        mot[s, 3] += 1.0
        mot[s, 0] += 0.5
    ph = np.stack([f(2 * np.pi * t * k * TR / p)
                   for p in (4.3, 1.13) for k in (1, 2)
                   for f in (np.sin, np.cos)], 1).astype(np.float32)
    nvox = np.prod(SHAPE)
    base = rng.uniform(500, 1500, nvox)
    sig = np.sin(2 * np.pi * t / 30.0)
    X = np.concatenate([mot, ph, np.linspace(-1, 1, T)[:, None]], 1)
    B = rng.normal(0, 5, (X.shape[1], nvox))
    Y = base[None, :] + X @ B + 3 * np.outer(sig, rng.normal(0, 1, nvox)) \
        + rng.normal(0, 2, (T, nvox))
    Y = Y.reshape(T, *SHAPE)
    Y[:, 0, 0, :] = 0.0  # out-of-mask voxels
    return Y, mot, ph


def new_rtp(work_dir, device, mot, ph, msl, retro, ign, censor=0.0,
            mot_reg="mot12", none_at=(), buf_bytes=None):
    kw = dict(TR=TR, mot_reg=mot_reg, volreg=FakeVolreg(mot),
              max_scan_length=msl, onGPU=(device == "cuda"),
              reg_retro_proc=retro, ignore_init=ign, wait_num=0,
              censor_FD=censor, ridge_c=0.0)
    if ph is not None:
        kw.update(phys_reg="RICOR8", rtp_physio=FakePhysio(ph, none_at))
    rtp = RtpRegress(**kw)
    rtp.next_proc = Recorder()
    rtp.save_proc = True
    rtp.work_dir = work_dir
    rtp.max_buffer_bytes = buf_bytes
    assert rtp.ready_proc()
    return rtp


def feed(rtp, Y, aff=np.eye(4)):
    """Send the volumes of Y to rtp.do_proc one by one."""
    for v in range(Y.shape[0]):
        img = nib.Nifti1Image(np.ascontiguousarray(Y[v]).astype(np.float32),
                              aff)
        img.set_filename(f"test_nr_{v+1:04d}.nii.gz")
        rtp.do_proc(img, v, None)


# %% Solver ===================================================================
def test_solver_zero_matrix_and_rank_deficient(device):
    A = torch.zeros(20, 5, device=device)
    B = torch.randn(20, 3, device=device)
    assert torch.all(lstsq_SVDsolver(A, B) == 0)

    g = torch.Generator().manual_seed(0)
    A = torch.randn(30, 4, generator=g, dtype=torch.float64).to(device)
    A = torch.cat([A, A[:, :1]], 1)  # rank deficient
    B = torch.randn(30, 3, generator=g, dtype=torch.float64).to(device)
    x = lstsq_SVDsolver(A, B)
    xr = torch.linalg.pinv(A) @ B
    assert float((x - xr).abs().max()) < 1e-8


def test_solver_zero_rows_equal_dropped_rows(device):
    g = torch.Generator().manual_seed(0)
    A = torch.randn(80, 12, generator=g).to(device)
    B = torch.randn(80, 500, generator=g).to(device)
    z = torch.zeros(80, dtype=torch.bool)
    z[[3, 10, 11, 50, 79]] = True
    z = z.to(device)
    A0 = A.clone()
    A0[z] = 0
    x1 = lstsq_SVDsolver(A0, B)
    x2 = lstsq_SVDsolver(A[~z], B[~z])
    assert torch.allclose(x1, x2, rtol=1e-4, atol=1e-5)


# %% Online GLM vs offline lstsq ==============================================
def offline_fit(Y, mot, ph, rtp, start, none_at, mot_reg, retro_proc, L):
    """Offline residuals with the same model (window of the latest L
    volumes). Returns ({vol: residual}, maskV)."""
    maskV = Y[start] != 0
    Yf = Y[:, maskV].astype(np.float64)
    t0 = start + rtp.wait_num  # first fit volume
    Ymean = Yf[start:t0 + 1].mean(0)
    mm = np.abs(Ymean) > 1e-6
    Ys = np.zeros_like(Yf)
    Ys[:, mm] = np.minimum(Yf[:, mm] / Ymean[mm] * 100, 200)
    dmot = np.zeros_like(mot, dtype=np.float64)
    dmot[start + 1:] = np.diff(mot[start:].astype(np.float64), axis=0)

    def fit(t):
        s0 = max(start, t - L + 1)
        n = t - s0 + 1
        cols = [mot[s0:t + 1].astype(np.float64)]
        if mot_reg == "mot12":
            cols.append(dmot[s0:t + 1])
        if t not in none_at:
            cols.append(ph[s0:t + 1].astype(np.float64))
        cols.append(rtp.poly_reg(n, TR).astype(np.float64))
        X = np.concatenate(cols, 1)
        Yp = Ys[s0:t + 1]
        beta = np.linalg.lstsq(X, Yp, rcond=None)[0]
        return Yp - X @ beta

    ref = {}
    R0 = fit(t0)
    if retro_proc:
        for ii, v in enumerate(range(start, t0 + 1)):
            ref[v] = R0[ii]
    else:
        ref[t0] = R0[-1]
    for t in range(t0 + 1, T):
        ref[t] = fit(t)[-1]
    return ref, maskV


# (ignore_init, retro_proc, mot_reg, none_at, max_scan_length, buffer_bytes)
GLM_CASES = {
    "mot6": (0, True, "mot6", [], None, None),
    "mot6_ign_noretro": (3, False, "mot6", [], None, None),
    # first-fit volume and later volumes without physio (wait_num is 23)
    "mot12_nophysio": (3, True, "mot12", [3 + 23, 40, 77], None, None),
    "mot12_nophysio_last": (0, False, "mot12", [23, 55, T - 1], None, None),
    # ring buffer: scan longer than max_scan_length / the memory budget
    "ring_msl60": (0, True, "mot6", [], 60, None),
    "ring_msl50_ign": (3, False, "mot12", [40, 77], 50, None),
    "ring_budget": (0, True, "mot12", [70], None,
                    40 * 4 * (16 * 16 * 8 + 40)),
}


@pytest.mark.parametrize("case", list(GLM_CASES))
def test_glm_matches_offline(case, device, tmp_path):
    ign, retro, mot_reg, none_at, msl, buf_bytes = GLM_CASES[case]
    Y, mot, ph = make_data(np.random.default_rng(0))
    rtp = new_rtp(tmp_path, device, mot, ph, msl or T, retro, ign,
                  mot_reg=mot_reg, none_at=none_at, buf_bytes=buf_bytes)
    feed(rtp, Y)

    L = rtp.YMtx.shape[0]
    if msl is not None or buf_bytes is not None:
        assert L < T  # the ring buffer is used
    ref, maskV = offline_fit(Y, mot, ph, rtp, ign, set(none_at), mot_reg,
                             retro, L)
    rec = rtp.next_proc
    assert set(rec.out) == set(ref)
    rstd = np.std(np.stack(list(ref.values())))
    maxerr = max(np.abs(rec.out[v][maskV] - r).max() for v, r in ref.items())
    assert maxerr < 1e-2 * rstd
    for v in ref:
        assert np.all(rec.out[v][~maskV] == 0)
    # Saved slots are indexed by the 0-based vol_idx, as the output
    sd = rtp.saved_data
    assert all(np.array_equal(sd[..., v], rec.out[v]) for v in ref)


# %% Online QC ================================================================
def fd_numpy(mot, start):
    fd = np.full(len(mot), np.nan)
    d = np.abs(np.diff(mot[start:].astype(np.float64), axis=0))
    fd[start] = 0.0
    fd[start + 1:] = d[:, 3:].sum(1) + 50 * np.deg2rad(d[:, :3]).sum(1)
    return fd


def dvars_tsnr_numpy(Yf, start):
    """Yf: T x vox (masked input)."""
    Yf = Yf.astype(np.float64)
    n = len(Yf)
    dv = np.full(n, np.nan)
    ts = np.full(n, np.nan)
    for t in range(start, n):
        if t > start:
            d = Yf[t] - Yf[t - 1]
            dv[t] = 100 * np.sqrt(np.mean(d ** 2)) / Yf[t].mean()
        if t - start + 1 >= 3:
            w = Yf[start:t + 1]
            m = w.mean(0)
            sd = w.std(0, ddof=1)
            v = np.sort(m[sd > 0] / sd[sd > 0])
            ts[t] = v[(len(v) - 1) // 2]  # lower median (as torch)
    return dv, ts


def test_qc_values_and_csv(device, tmp_path):
    rng = np.random.default_rng(3)
    Y, mot, ph = make_data(rng, mot_sd=0.05, spikes=(30, 70))
    ign = 2
    rtp = new_rtp(tmp_path, device, mot, ph, 60, True, ign)
    feed(rtp, Y)

    tab = rtp.get_qc_table()
    assert list(tab.columns) == list(rr.QC_COLS)
    assert np.array_equal(tab.vol_idx.values, np.arange(ign, T))
    fd = fd_numpy(mot, ign)
    dv, ts = dvars_tsnr_numpy(Y[:, Y[ign] != 0], ign)
    assert np.abs(tab.FD.values - fd[ign:]).max() < 1e-5
    assert np.nanmax(np.abs(tab.DVARS.values - dv[ign:]) / dv[ign:]) < 1e-4
    assert np.nanmax(np.abs(tab.tSNR.values - ts[ign:]) / ts[ign:]) < 1e-3
    assert tab.FD.values[0] == 0 and np.isnan(tab.DVARS.values[0])
    assert np.all(np.isnan(tab.tSNR.values[:2]))
    assert not np.any(np.isnan(tab.tSNR.values[2:]))

    # QC passed to next_proc (retro and online volumes) equals the table
    rec = rtp.next_proc
    assert len(rec.qc) > 0
    for v, q in rec.qc.items():
        row = tab[tab.vol_idx == v].iloc[0]
        for k in ("FD", "DVARS", "tSNR"):
            assert np.isclose(q[k], row[k], equal_nan=True)

    # BIDS TSV (with a JSON sidecar) saved at end_reset with the entities
    # of the processed image
    out = rtp.end_reset()
    qc_f = out["RtpRegress:QC"]
    img_f = out["RtpRegress:processed image"]
    assert img_f.name.endswith("_desc-regress_bold.nii.gz")
    assert qc_f.name == img_f.name.replace(
        "_desc-regress_bold.nii.gz", "_desc-regressqc_timeseries.tsv")
    assert qc_f.with_suffix(".json").is_file()
    assert img_f.parent.joinpath(
        img_f.name.replace(".nii.gz", ".json")).is_file()
    df = pd.read_csv(qc_f, sep="\t", na_values="n/a")
    assert list(df.columns) == list(rr.QC_COLS) and len(df) == T - ign
    assert np.allclose(df.FD.values, tab.FD.values, rtol=1e-5,
                       equal_nan=True)
    assert rtp.get_qc_table() is not None  # kept after end_reset


# %% FD censoring =============================================================
def _standardize(R, keep):
    m = R[keep].mean(0)
    s = R[keep].std(0, ddof=1)
    valid = s > 1e-6 * np.maximum(np.abs(m), 1.0)
    out = np.zeros_like(R)
    out[:, valid] = (R[:, valid] - m[valid]) / s[valid]
    return out


def offline_censor(Y, mot, ph, rtp, start, cens, L):
    """Offline fit without the censored rows.
    Returns (first fit volume, {vol: residual}, {t: beta}, maskV)."""
    maskV = Y[start] != 0
    Yf = Y[:, maskV].astype(np.float64)
    reg0 = 12 + (8 if ph is not None else 0)

    def pnum(n):
        return min(1 + int(n * TR / 150), rtp.max_poly_order) + 1

    t0 = None
    for t in range(start + rtp.wait_num, T):
        s0 = max(start, t - L + 1)
        n = t - s0 + 1
        if n - cens[s0:t + 1].sum() > reg0 + pnum(n):
            t0 = t
            break
    assert t0 is not None
    s0 = max(start, t0 - L + 1)
    keep0 = ~cens[s0:t0 + 1]
    Ymean = Yf[s0:t0 + 1][keep0].mean(0)
    mm = np.abs(Ymean) > 1e-6
    Ys = np.zeros_like(Yf)
    Ys[:, mm] = np.minimum(Yf[:, mm] / Ymean[mm] * 100, 200)
    dmot = np.zeros_like(mot, dtype=np.float64)
    dmot[start + 1:] = np.diff(mot[start:].astype(np.float64), axis=0)

    def fit(t):
        s0 = max(start, t - L + 1)
        n = t - s0 + 1
        keep = ~cens[s0:t + 1]
        cols = [_standardize(mot[s0:t + 1].astype(np.float64), keep),
                _standardize(dmot[s0:t + 1], keep)]
        if ph is not None:
            cols.append(ph[s0:t + 1].astype(np.float64))
        cols.append(rtp.poly_reg(n, TR).astype(np.float64))
        X = np.concatenate(cols, 1)
        Yp = Ys[s0:t + 1]
        beta = np.linalg.lstsq(X[keep], Yp[keep], rcond=None)[0]
        return beta, Yp - X @ beta

    ref, betas = {}, {}
    b, R0 = fit(t0)
    betas[t0] = b
    if rtp.reg_retro_proc:
        for ii, v in enumerate(range(max(start, t0 - L + 1), t0 + 1)):
            ref[v] = R0[ii]
    else:
        ref[t0] = R0[-1]
    for t in range(t0 + 1, T):
        b, R = fit(t)
        betas[t] = b
        ref[t] = R[-1]
    return t0, ref, betas, maskV


# (max_scan_length, retro, ignore_init, spikes, use_physio, min_first_fit)
CENSOR_CASES = {
    "no_ring": (T, True, 0, (30, 31, 64, 90), True, None),
    "ring": (60, True, 0, (30, 31, 64, 90), True, None),
    "ring_ign_noretro": (50, False, 3, (10, 55, 56, 100), True, None),
    "ring_nophysio": (50, True, 3, (10, 55, 56, 100), False, None),
    # many censored volumes at the start delay the first fit
    "early_censored": (T, True, 0, tuple(range(1, 40, 2)), True, 24),
    "early_censored_ring": (40, True, 2, tuple(range(4, 50, 2)), True, None),
}


@pytest.mark.parametrize("case", list(CENSOR_CASES))
def test_fd_censoring(case, device, tmp_path, monkeypatch):
    msl, retro, ign, spikes, use_ph, min_t0 = CENSOR_CASES[case]
    thr = 0.5
    Y, mot, ph = make_data(np.random.default_rng(1), mot_sd=0.02,
                           spikes=spikes)
    if not use_ph:
        ph = None
    cens = np.nan_to_num(fd_numpy(mot, ign), nan=0) > thr
    rtp = new_rtp(tmp_path, device, mot, ph, msl, retro, ign, censor=thr)

    calls = {}
    orig_solver = rr.lstsq_SVDsolver

    def spy(A, B, rcond=None):
        X = orig_solver(A, B, rcond)
        calls[rtp._vol_num - 1] = (A.clone(), B.clone(), X.clone())
        return X

    monkeypatch.setattr(rr, "lstsq_SVDsolver", spy)
    feed(rtp, Y)
    monkeypatch.undo()

    L = rtp.YMtx.shape[0]
    t0, ref, betas, maskV = offline_censor(Y, mot, ph, rtp, ign, cens, L)
    if min_t0 is not None:
        assert t0 >= min_t0
    rec = rtp.next_proc
    assert min(calls) == t0
    assert set(rec.out) == set(ref)

    # Censored flags in the QC table and passed downstream
    assert np.array_equal(rtp.qc["censored"][ign:T], cens[ign:T])
    for v, q in rec.qc.items():
        assert q["censored"] == bool(cens[v])

    # Residuals (including the censored volumes)
    rstd = np.std(np.stack(list(ref.values())))
    rerr = max(np.abs(rec.out[v][maskV] - r).max() for v, r in ref.items())
    assert rerr < 1e-2 * rstd

    for t, (A, B, X) in calls.items():
        Xo = betas[t]
        Xn = X.cpu().double().numpy()
        # Constant regressors: offline beta 0, online float32 beta ~1e-5
        zc = np.abs(Xo).max(1) < 1e-10
        assert np.abs(Xn[zc]).max(initial=0) < 1e-3
        scale = np.abs(Xo[~zc]).max(1, keepdims=True)
        assert (np.abs(Xn[~zc] - Xo[~zc]) / scale).max() < 2e-3
        # Censored rows are zero rows, equivalent to dropping them
        zr = A.abs().sum(1) == 0
        s0 = max(ign, t - L + 1)
        assert int(zr.sum()) == int(cens[s0:t + 1].sum())
        if zr.any():
            Xd = orig_solver(A[~zr], B[~zr])
            assert float((Xd - X).abs().max() / X.abs().max()) < 1e-3


# --- Ridge regularization of the early fits ----------------------------------
def test_ridge_solver_matches_closed_form(device):
    g = torch.Generator().manual_seed(0)
    A = torch.randn(30, 6, generator=g, dtype=torch.float64).to(device)
    B = torch.randn(30, 50, generator=g, dtype=torch.float64).to(device)
    pen = torch.tensor([0, 0, 1.5, 0.5, 2.0, 0], dtype=torch.float64,
                       device=device)
    x = lstsq_SVDsolver(A, B, pen=pen)
    ref = torch.linalg.solve(A.T @ A + torch.diag(pen ** 2), A.T @ B)
    assert torch.allclose(x, ref, atol=1e-8)
    # No penalty: plain least squares
    x0 = lstsq_SVDsolver(A, B, pen=torch.zeros(6, dtype=torch.float64,
                                               device=device))
    assert torch.allclose(x0, lstsq_SVDsolver(A, B), atol=1e-10)


def test_ridge_c_param():
    rtp = RtpRegress(TR=TR)
    assert rtp.ridge_c == 0.5
    rtp.set_param('ridge_c', 0)
    assert rtp.ridge_c == 0.0
    rtp.set_param('ridge_c', -1)
    assert rtp.ridge_c == 0.0
    assert rtp.get_params()['ridge_c'] == 0.0
