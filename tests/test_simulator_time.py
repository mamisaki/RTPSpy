"""rtmri_simulator_qt.py: DICOM times (HHMMSS.ffffff) are converted to seconds
before the file timing is computed. Subtracting them as numbers gives 40 s
more at each minute (4040 s at each hour)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sim = pytest.importorskip("rtmri_simulator_qt")


def test_dicom_time_to_sec():
    f = sim.dicom_time_to_sec
    assert f("140659.807000") == pytest.approx(14 * 3600 + 6 * 60 + 59.807)
    # Across a minute: 1.504 s, not 41.504
    assert f("140701.311000") - f("140659.807000") == pytest.approx(1.504)
    # Across an hour
    assert f("100000.5") - f("095959.5") == pytest.approx(1.0)
    # Number read from the cache CSV (leading zero lost)
    assert f(93359.826) == pytest.approx(f("093359.826"))
    # DT (date and time)
    assert f("20240613140659.807") == pytest.approx(f("140659.807"))
    assert np.isnan(f(None)) and np.isnan(f("")) and np.isnan(f("abc"))


def test_content_sec_over_midnight():
    t = pd.Series(["235959.0", "000000.5", "000002.0"])
    sec = sim.content_sec(t).values
    assert np.diff(sec) == pytest.approx([1.5, 1.5])


def test_series_timing_of_test_session():
    """The fMRI series of the test session crosses minutes: the file
    intervals stay at about one TR."""
    pydicom = pytest.importorskip("pydicom")
    files = sorted((ROOT / "RTMRI_simulation" / "P000535" / "dicom")
                   .glob("003_000005_*"))
    if len(files) < 50:
        pytest.skip("Test session P000535 is not available")
    ct = pd.Series([str(pydicom.dcmread(f, stop_before_pixels=True)
                        .ContentTime) for f in files[:60]])
    sec = np.sort(sim.content_sec(ct).values)
    d = np.diff(sec)
    assert d.max() < 2.0 and np.median(d) == pytest.approx(1.5, abs=0.05)
    # As numbers, the intervals across a minute were about 41 s
    assert np.diff(np.sort(ct.astype(float).values)).max() > 40


def _series_df(n=40):
    pydicom = pytest.importorskip("pydicom")
    files = sorted((ROOT / "RTMRI_simulation" / "P000535" / "dicom")
                   .glob("003_000005_*"))[:n]
    if len(files) < n:
        pytest.skip("Test session P000535 is not available")
    rows = []
    for f in files:
        h = pydicom.dcmread(f, stop_before_pixels=True)
        rows.append({"ContentTime": str(h.ContentTime),
                     "AcqSec": sim.read_acq_sec(f),
                     "AcquisitionNumber": int(h.AcquisitionNumber),
                     "ImageType": str(list(h.ImageType))})
    df = pd.DataFrame(rows)
    df["ContentSec"] = sim.content_sec(df["ContentTime"])
    return df.sort_values("ContentSec").reset_index(drop=True)


def test_series_timing_acquisition_and_recon_delay():
    """TTLs at the acquisition times; files at ContentTime + transfer, so
    the first file comes TR + reconstruction + transfer after the first
    TTL."""
    df = _series_df()
    assert np.isfinite(df.AcqSec).all()
    file_t, ttl_t = sim.series_timing(df, transfer_delay=0.2)
    assert ttl_t[0] == pytest.approx(0.0)
    assert np.median(np.diff(ttl_t)) == pytest.approx(1.5, abs=0.01)
    # ContentTime - AcquisitionTime of this series: about 1.78 s (first
    # volume 2.01 s), i.e., TR 1.5 s + reconstruction
    delay = file_t - ttl_t[:len(file_t)]
    assert delay[0] == pytest.approx(2.014 + 0.2, abs=0.02)
    assert np.median(delay) == pytest.approx(1.78 + 0.2, abs=0.05)
    # Every file comes after its TTL
    assert (delay > 1.5).all()


def test_series_timing_without_acquisition_time():
    """An old cache without acquisition times: a TTL one TR before each
    volume's first file."""
    df = _series_df()
    df["AcqSec"] = np.nan
    file_t, ttl_t = sim.series_timing(df, transfer_delay=0.0)
    assert (file_t - ttl_t) == pytest.approx(np.full(len(df), 1.5),
                                             abs=0.05)


def test_ge_plan_without_temporal_position():
    """Older GE software writes no TemporalPositionIdentifier (0 in the
    header table): the volumes are taken from InstanceNumber, so that the
    files of volume k are sent after k x TR, volume by volume."""
    nt, nsl, TR = 3, 4, 2000.0
    inst = np.arange(1, nt * nsl + 1)
    df = pd.DataFrame({
        "FilePath": [f"/x/p1/e2/s3/i{i}.MRDC.{i}" for i in inst],
        "SeriesNumber": 3, "InstanceNumber": inst,
        "NumberOfTemporalPositions": nt, "TemporalPositionIdentifier": 0,
        "InStackPositionNumber": (inst - 1) % nsl + 1,
        "RepetitionTime": TR, "IsGE": True})
    plan, ttl = sim.ge_plan(df, recon_delay=0.5)
    assert ttl == pytest.approx([0.0, 2.0, 4.0])
    assert list(plan.InstanceNumber) == list(inst)
    vol = (inst - 1) // nsl + 1
    assert list(plan.TemporalPositionIdentifier) == list(vol)
    assert (plan.PlanSec.values > vol * TR / 1000).all()
    assert (plan.PlanSec.values < vol * TR / 1000 + 1.0).all()
