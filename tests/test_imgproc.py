"""Online image processing modules: RtpTshift, RtpVolreg and RtpSmooth
(volreg and smooth call librtp.so) on synthetic data and on
tests/func_epi.nii.gz. The tests marked afni compare them with the AFNI
programs 3dTshift, 3dvolreg and 3dBlurInMask."""
import json
import subprocess

import nibabel as nib
import numpy as np
import pytest
from scipy import ndimage

from rtpspy.rtp_smooth import RtpSmooth
from rtpspy.rtp_tshift import RtpTshift
from rtpspy.rtp_volreg import RtpVolreg

N_EPI = 20  # number of func_epi volumes used


class Recorder:
    """Downstream process that records the received volumes."""

    def __init__(self):
        self.out = {}
        self.next_proc = None

    def ready_proc(self):
        return True

    def do_proc(self, img, vol_idx=None, pre_proc_time=None, **kwargs):
        self.out[vol_idx] = np.asanyarray(img.dataobj).copy()

    def end_reset(self):
        return None


def run_volumes(proc, data, affine):
    """Feed the volumes of data (x, y, z, t) to proc."""
    for v in range(data.shape[-1]):
        img = nib.Nifti1Image(np.ascontiguousarray(data[..., v]), affine)
        img.set_filename(f"test_nr_{v:04d}.nii.gz")
        proc.do_proc(img, v, None)


@pytest.fixture(scope="module")
def epi(test_dir):
    img = nib.load(test_dir / "func_epi.nii.gz")
    data = np.asarray(img.dataobj[..., :N_EPI], dtype=np.float32)
    return img, data


# %% RtpTshift ================================================================
@pytest.mark.parametrize("method", ["cubic", "linear"])
def test_tshift_synthetic(method):
    """Slices sampled at their slice times are shifted to time 0."""
    nz, TR, T = 10, 2.0, 40
    st = np.arange(nz) * TR / nz
    st = st[np.r_[0:nz:2, 1:nz:2]]  # interleaved
    t = np.arange(T) * TR
    freq = 1 / 30.0
    data = np.zeros((4, 4, nz, T), np.float32)
    for z in range(nz):
        data[:, :, z, :] = 100 + 10 * np.sin(2 * np.pi * freq * (t + st[z]))
    truth = 100 + 10 * np.sin(2 * np.pi * freq * t)

    ts = RtpTshift(TR=TR, method=method, ref_time=0)
    ts.slice_timing = list(st)
    ts.slice_dim = 2
    rec = Recorder()
    ts.next_proc = rec
    assert ts.ready_proc()
    run_volumes(ts, data, np.diag([2.0, 2.0, 3.0, 1.0]))

    assert sorted(rec.out) == list(range(T))  # held volumes are output too
    err = max(np.abs(rec.out[v] - truth[v]).max() for v in range(3, T))
    raw = max(np.abs(data[..., v] - truth[v]).max() for v in range(3, T))
    assert err < 0.1 * raw


def test_tshift_set_from_sample(test_dir):
    ts = RtpTshift()
    ts.set_from_sample(test_dir / "func_epi.nii.gz")
    st = json.loads((test_dir / "func_epi.json").read_text())["SliceTiming"]
    assert ts.TR == 2.0
    assert np.allclose(ts.slice_timing, st)
    assert ts.slice_dim == 2


# %% RtpVolreg ================================================================
def test_volreg_known_shifts(epi):
    img, data = epi
    ref = data[..., 0]
    vr = RtpVolreg(ref_vol=0, regmode="cubic")  # first volume of the scan
    rec = Recorder()
    vr.next_proc = rec
    assert vr.ready_proc()

    # Translations in voxels; func_epi voxels are 1.875 x 1.875 x 3.1 mm
    vox = np.array(img.header.get_zooms()[:3])
    shifts = [(0, 0, 0), (1, 0, 0), (0, 1.5, 0), (0, 0, 0.5)]
    shifted = np.stack([ndimage.shift(ref, s, order=3) for s in shifts], -1)
    run_volumes(vr, shifted, img.affine)

    mot = vr._motion[:len(shifts)]  # roll, pitch, yaw, dS, dL, dP
    assert np.abs(mot[0]).max() < 1e-3
    for i, s in enumerate(shifts[1:], start=1):
        mm = np.abs(np.array(s) * vox).max()
        assert np.abs(mot[i, :3]).max() < 0.02  # degrees
        assert abs(np.abs(mot[i, 3:]).max() - mm) < 0.05 * mm + 0.02
        assert np.sort(np.abs(mot[i, 3:]))[1] < 0.02  # only one axis moved
    assert set(rec.out) == set(range(len(shifts)))
    # The aligned volume is close to the reference
    m = ref > np.percentile(ref, 60)
    for v in range(len(shifts)):
        assert np.corrcoef(rec.out[v][m], ref[m])[0, 1] > 0.99


# %% RtpSmooth ================================================================
def test_smooth_impulse_fwhm():
    sm = RtpSmooth(blur_fwhm=6.0)
    rec = Recorder()
    sm.next_proc = rec
    d = np.zeros((40, 40, 40), np.float32)
    d[20, 20, 20] = 1000.0
    sm.set_mask(np.ones_like(d))
    assert sm.ready_proc()
    run_volumes(sm, d[..., None], np.diag([2.0, 2.0, 2.0, 1.0]))
    o = rec.out[0]
    assert np.isclose(o.sum(), 1000.0, rtol=1e-4)
    x = (np.arange(40) - 20) * 2.0
    for ax in range(3):
        p = o.sum(axis=tuple(a for a in range(3) if a != ax))
        fwhm = 2 * np.sqrt(2 * np.log(2)) * np.sqrt((p * x ** 2).sum()
                                                    / p.sum())
        assert abs(fwhm - 6.0) < 0.1


def test_smooth_mask_from_first_volume(epi):
    img, data = epi
    vol = data[..., 0].copy()
    vol[:, :, :5] = 0  # out of the mask
    sm = RtpSmooth(blur_fwhm=6.0, mask_file=0)
    rec = Recorder()
    sm.next_proc = rec
    assert sm.ready_proc()
    run_volumes(sm, vol[..., None], img.affine)
    o = rec.out[0]
    assert np.all(o[vol == 0] == 0)
    assert np.all(np.isfinite(o))
    assert o[vol != 0].std() < vol[vol != 0].std()


# %% Chain on func_epi ========================================================
@pytest.mark.parametrize("n", [10, pytest.param(100, marks=pytest.mark.slow)])
def test_chain_tshift_volreg_smooth(test_dir, n, tmp_path):
    img = nib.load(test_dir / "func_epi.nii.gz")
    data = np.asarray(img.dataobj[..., :n], dtype=np.float32)
    ts = RtpTshift(method="cubic", ref_time=0, ignore_init=3)
    ts.set_from_sample(test_dir / "func_epi.nii.gz")
    vr = RtpVolreg(regmode="cubic")
    vr.set_ref_vol(str(test_dir / "func_epi.nii.gz") + "[0]")
    sm = RtpSmooth(blur_fwhm=6.0)
    rec = Recorder()
    for p in (ts, vr, sm):
        p.work_dir = tmp_path
        p.save_proc = True
    ts.next_proc = vr
    vr.next_proc = sm
    sm.next_proc = rec
    assert ts.ready_proc()

    run_volumes(ts, data, img.affine)
    assert sorted(rec.out) == list(range(3, n))
    assert all(np.all(np.isfinite(rec.out[v])) for v in rec.out)
    assert np.abs(vr._motion[3:n]).max() < 3.0  # mm or degrees

    out = ts.end_reset()
    saved = nib.load(sm.saved_filename)
    assert saved.shape == data.shape[:3] + (n - 3,)
    assert out is not None


# %% Comparison with AFNI =====================================================
@pytest.fixture(scope="module")
def epi_sub(epi, tmp_path_factory):
    img, data = epi
    d = tmp_path_factory.mktemp("afni")
    sub = d / "sub.nii.gz"
    nib.save(nib.Nifti1Image(data, img.affine, img.header), sub)
    return d, sub


@pytest.mark.afni
def test_tshift_vs_3dTshift(test_dir, epi, epi_sub):
    img, data = epi
    d, sub = epi_sub
    st = json.loads((test_dir / "func_epi.json").read_text())["SliceTiming"]
    np.savetxt(d / "st.1D", np.array(st)[None, :], fmt="%.6f")
    subprocess.run(["3dTshift", "-overwrite", "-cubic", "-tzero", "0",
                    "-TR", "2", "-tpattern", f"@{d / 'st.1D'}",
                    "-prefix", str(d / "ts.nii.gz"), str(sub)],
                   check=True, capture_output=True)
    A = nib.load(d / "ts.nii.gz").get_fdata()

    ts = RtpTshift(method="cubic", ref_time=0)
    ts.set_from_sample(test_dir / "func_epi.nii.gz")
    rec = Recorder()
    ts.next_proc = rec
    assert ts.ready_proc()
    run_volumes(ts, data, img.affine)

    m = data[..., 0] > np.percentile(data[..., 0], 60)
    for v in (5, 10, 15):
        a, r, raw = A[..., v][m], rec.out[v][m], data[..., v][m]
        assert np.corrcoef(a, r)[0, 1] > 0.9999
        # Online (no future volume) vs offline: much closer than raw data
        assert np.abs(a - r).mean() < 0.5 * np.abs(a - raw).mean()


@pytest.mark.afni
def test_volreg_vs_3dvolreg(epi, epi_sub):
    img, data = epi
    d, sub = epi_sub
    subprocess.run(["3dvolreg", "-overwrite", "-cubic", "-base", "0",
                    "-1Dfile", str(d / "mot.1D"),
                    "-prefix", str(d / "vr.nii.gz"), str(sub)],
                   check=True, capture_output=True)
    mA = np.loadtxt(d / "mot.1D")

    vr = RtpVolreg(regmode="cubic")
    vr.set_ref_vol(str(sub) + "[0]")
    rec = Recorder()
    vr.next_proc = rec
    assert vr.ready_proc()
    run_volumes(vr, data, img.affine)
    mR = vr._motion[:N_EPI]
    # RtpVolreg starts from the previous estimate, so its volume-to-volume
    # jitter is smaller than that of 3dvolreg (see rtp_volreg.py).
    for i in range(6):
        assert np.corrcoef(mA[:, i], mR[:, i])[0, 1] > 0.7
    assert np.abs(mA - mR).max() < 0.3


@pytest.mark.afni
def test_smooth_vs_3dBlurInMask(epi, epi_sub):
    img, data = epi
    d, _ = epi_sub
    vol = data[..., 0]
    mask = (vol > 0).astype(np.int16)
    nib.save(nib.Nifti1Image(vol, img.affine), d / "v0.nii.gz")
    nib.save(nib.Nifti1Image(mask, img.affine), d / "mask.nii.gz")
    subprocess.run(["3dBlurInMask", "-overwrite", "-FWHM", "6",
                    "-mask", str(d / "mask.nii.gz"),
                    "-prefix", str(d / "bl.nii.gz"), str(d / "v0.nii.gz")],
                   check=True, capture_output=True)
    B = nib.load(d / "bl.nii.gz").get_fdata()

    sm = RtpSmooth(blur_fwhm=6.0)
    sm.set_mask(mask)
    rec = Recorder()
    sm.next_proc = rec
    assert sm.ready_proc()
    run_volumes(sm, vol[..., None], img.affine)
    S = rec.out[0]
    m = mask > 0
    assert np.abs(B[m] - S[m]).max() < 1e-3 * np.abs(B[m]).max()
