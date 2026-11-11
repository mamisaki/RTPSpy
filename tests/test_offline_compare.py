"""rtp_offline_compare: comparison metrics, causal physio regressors, and
an end-to-end run on a cropped test run (online RTPSpy chain vs AFNI)."""
import shutil

import numpy as np
import nibabel as nib
import pandas as pd
import pytest

from rtpspy.rtp_offline_compare import (
    compare_series, CausalSavedPhysio, compare_online_offline)


def test_compare_series_self():
    """Identical series: correlation 1 and RMSE 0."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(50, 20))
    r, rmse = compare_series(a, a.copy())
    np.testing.assert_allclose(r, 1.0, atol=1e-12)
    np.testing.assert_allclose(rmse, 0.0, atol=1e-12)
    # A constant voxel gives NaN, not an error
    a[:, 0] = 1.0
    r, _ = compare_series(a, a)
    assert np.isnan(r[0]) and np.all(np.isfinite(r[1:]))


def test_causal_physio(test_dir):
    """The regressors of n volumes use only the samples up to n * TR, and
    None is returned for volumes beyond the recorded signals."""
    card = np.loadtxt(test_dir / "ECG.1D")
    resp = np.loadtxt(test_dir / "Resp.1D")
    ph = CausalSavedPhysio(card, resp, 40)
    reg = ph.get_retrots(2.0, 30)
    assert reg.shape == (30, 8)
    # Changing the samples after 30 TRs does not change the regressors
    card2 = card.copy()
    card2[30 * 2 * 40:] = 0
    reg2 = CausalSavedPhysio(card2, resp, 40).get_retrots(2.0, 30)
    np.testing.assert_array_equal(reg, reg2)
    assert ph.get_retrots(2.0, len(card) // 40) is None


@pytest.mark.slow
@pytest.mark.afni
def test_compare_online_offline(test_dir, tmp_path):
    """End-to-end run on the first 50 volumes of func_epi."""
    masks_dir = test_dir / "work" / "RTP"
    roi_f = masks_dir / "MNI152_2009_template_LAmy_alFunc.nii.gz"
    if not (masks_dir / "RTP_mask.nii.gz").is_file() or not roi_f.is_file():
        pytest.skip("Masks in tests/work/RTP are not available")

    img = nib.load(test_dir / "func_epi.nii.gz")
    func_f = tmp_path / "func_short.nii.gz"
    hdr = img.header.copy()
    hdr.extensions.clear()  # The AFNI extension has the original length
    nib.Nifti1Image(np.asanyarray(img.dataobj)[..., :50], img.affine,
                    hdr).to_filename(func_f)
    # Slice timing sidecar
    shutil.copy(test_dir / "func_epi.json", tmp_path / "func_short.json")

    out_dir = tmp_path / "out"
    files = compare_online_offline(
        func_f, out_dir, masks_dir=masks_dir, roi_f=roi_f,
        card_f=test_dir / "ECG.1D", resp_f=test_dir / "Resp.1D",
        physio_freq=40)

    for key in ("metrics", "figure", "report", "roi_series"):
        assert files[key].is_file()
    assert (out_dir / "offline" / "afni_commands.log").is_file()
    assert any((out_dir / "online" / "RTP").glob("*_desc-regress_bold.nii.gz"))

    metrics = pd.read_csv(files["metrics"])
    assert set(metrics.stage) >= {"VOLREG", "TSHIFT", "SMOOTH", "REGRESS",
                                  "REGRESS_causal"}
    assert np.all(np.isfinite(metrics.value))
    val = metrics.set_index(["stage", "metric"]).value
    assert val["ALL", "n_volumes_compared"] == 47
    # Preprocessing stages agree closely with AFNI
    for st in ("VOLREG", "TSHIFT", "SMOOTH"):
        assert val[st, "voxel_r_median"] > 0.9

    roi = pd.read_csv(files["roi_series"])
    assert len(roi) == 47
    assert np.all(np.isfinite(roi.drop(columns="vol_idx").values))
