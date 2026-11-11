"""Per-volume latency (rtpspy.rtp_latency)."""
import json
import numpy as np
import pytest

from rtpspy.rtp_latency import VolumeLatency, COLUMNS

TR = 2.0
ONSET = 1000.0


def test_match_one_pulse_per_volume():
    lat = VolumeLatency()
    # Scanner clock slightly longer than the nominal TR, with jitter
    ttl = ONSET + np.arange(50) * 2.001 + np.random.default_rng(0).normal(
        0, 0.002, 50)
    ttl[0] = ONSET
    for vi in range(50):
        on, est = lat.match_onset(vi, ttl, ONSET, TR)
        assert not est
        assert on == pytest.approx(ttl[vi])


def test_match_slice_pulses():
    # One pulse per slice (40 slices): the first slice pulse of the volume
    lat = VolumeLatency()
    ttl = (ONSET + np.arange(30)[:, None] * TR
           + np.arange(40)[None, :] * TR / 40).ravel()
    for vi in range(30):
        on, est = lat.match_onset(vi, ttl, ONSET, TR)
        assert not est and on == pytest.approx(ONSET + vi * TR)


def test_no_ttl_is_estimated():
    lat = VolumeLatency()
    on, est = lat.match_onset(5, [], ONSET, TR)
    assert est and on == ONSET + 5 * TR
    # Pulses from a previous scan are not used
    on, est = lat.match_onset(5, [ONSET - 100.0], ONSET, TR)
    assert est


def test_record_and_csv(tmp_path):
    lat = VolumeLatency()
    lat.record(0, ONSET, False, ONSET + 1.5, ONSET + 1.6, True)
    lat.record(1, ONSET + 2, False, ONSET + 3.4, ONSET + 3.5, False)
    lat.record(1, ONSET + 2, False, ONSET + 9, ONSET + 9, False)  # ignored
    assert len(lat) == 2
    assert lat.data["acq_to_read"][1] == pytest.approx(1.4)
    assert lat.data["read_to_fb"][1] == pytest.approx(0.1)
    f = lat.save_tsv(tmp_path / "lat_timeseries.tsv")
    lines = f.read_text().splitlines()
    assert lines[0] == "\t".join(COLUMNS)
    assert lines[1].split("\t")[-1] == "1" and \
        lines[2].split("\t")[-1] == "0"
    side = json.loads((tmp_path / "lat_timeseries.json").read_text())
    assert set(side) == set(COLUMNS) and side["onset"]["Units"] == "s"
    lat.reset()
    assert len(lat) == 0 and lat.save_tsv(tmp_path / "x.tsv") is None
