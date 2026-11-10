"""Package import, the bundled librtp library, and the packaging config."""
import ctypes
from pathlib import Path

import pytest

import rtpspy

REPO = Path(__file__).resolve().parents[1]


def test_import_public_classes():
    for name in ("RTP", "RtpWatch", "RtpTshift", "RtpVolreg", "RtpSmooth",
                 "RtpRegress", "RtpApp", "RtpGUI", "RtpTTLPhysio",
                 "RtpRetroTS", "RtpImgProc", "RPCSocketServer",
                 "RPCSocketCom", "DicomConverter", "DicomReader",
                 "SharedMemoryRingBuffer"):
        assert hasattr(rtpspy, name), name


def test_librtp_loads():
    from rtpspy.rtp_smooth import librtp_path
    assert Path(librtp_path).is_file()
    lib = ctypes.cdll.LoadLibrary(librtp_path)
    assert hasattr(lib, "rtp_smooth")


def test_pyproject_config():
    tomllib = pytest.importorskip("tomllib")
    pyproject = REPO / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("not run from the source tree")
    cfg = tomllib.loads(pyproject.read_text())
    proj = cfg["project"]
    assert proj["name"] == "RTPSpy"
    st = cfg["tool"]["setuptools"]
    # rtpspy/FastSurfer is not packaged
    assert st["packages"] == ["rtpspy", "rtpspy.ui"]
    for f in st["script-files"]:
        assert (REPO / f).is_file()
    markers = cfg["tool"]["pytest"]["ini_options"]["markers"]
    assert {m.split(":")[0] for m in markers} == {"gpu", "afni", "slow"}
