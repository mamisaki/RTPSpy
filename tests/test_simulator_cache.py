"""rtmri_simulator_qt.py: the series cache of a session keeps the file paths
relative to the session folder, so that a session can be moved or copied to
another computer with its cache. Old caches with absolute paths are
re-rooted to the folder where the session is now."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sim = pytest.importorskip("rtmri_simulator_qt")

NAMES = ["s3/i1.MRDC.1", "s3/i2.MRDC.2", "s8/i3.MRDC.1"]


def _session(tmp_path, name="p1/e2"):
    ses = tmp_path / name
    for n in NAMES:
        (ses / n).parent.mkdir(parents=True, exist_ok=True)
        (ses / n).write_bytes(b"")
    return ses


def test_cache_relative_round_trip(tmp_path):
    ses = _session(tmp_path)
    df = pd.DataFrame({"FilePath": [str(ses / n) for n in NAMES],
                       "SeriesNumber": [3, 3, 8]})
    rel = sim.cache_paths_relative(df, ses)
    assert list(rel.FilePath) == NAMES
    back, old_root = sim.cache_paths_resolve(rel, ses)
    assert old_root is None
    assert list(back.FilePath) == [str(ses / n) for n in NAMES]
    assert list(back.SeriesNumber) == [3, 3, 8]


def test_cache_moved_session(tmp_path):
    """A cache made on another computer (absolute paths that do not exist
    here) is read from the folder where the session is now."""
    ses = _session(tmp_path, "copy/E1/p1/e2")
    old = "/mnt/storage/data/E1/p1/e2"
    df = pd.DataFrame({"FilePath": [f"{old}/{n}" for n in NAMES]})
    out, old_root = sim.cache_paths_resolve(df, ses)
    assert old_root == Path(old)
    assert list(out.FilePath) == [str(ses / n) for n in NAMES]
    assert all(Path(p).is_file() for p in out.FilePath)


def test_cache_existing_absolute_paths_kept(tmp_path):
    ses = _session(tmp_path)
    df = pd.DataFrame({"FilePath": [str(ses / n) for n in NAMES]})
    out, old_root = sim.cache_paths_resolve(df, ses)
    assert old_root is None
    assert list(out.FilePath) == list(df.FilePath)
