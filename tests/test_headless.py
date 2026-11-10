"""The processing library runs without PyQt5 and without a display.

Each test runs tests/_headless_run.py in a subprocess in which PyQt5 imports
are blocked (a sys.meta_path finder raising ImportError) and DISPLAY is
unset.
"""
import os
import sys
import json
import subprocess
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parent / "_headless_run.py"


def _run_headless(mode, work_dir, timeout=600):
    env = {k: v for k, v in os.environ.items()
           if k not in ("DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM")}
    pr = subprocess.run([sys.executable, str(HELPER), mode, str(work_dir)],
                        capture_output=True, text=True, timeout=timeout,
                        env=env)
    assert pr.returncode == 0, pr.stdout[-2000:] + pr.stderr[-4000:]
    return json.loads(pr.stdout.strip().splitlines()[-1])


def test_import_without_qt():
    code = (
        "import sys\n"
        "class B:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] == 'PyQt5':\n"
        "            raise ImportError(name)\n"
        "sys.meta_path.insert(0, B())\n"
        f"sys.path.insert(0, {str(HELPER.parents[1])!r})\n"
        "import rtpspy, rtpspy.rtp_sim, rtpspy.rtp_gui as g\n"
        "raise SystemExit(1)\n")
    # rtpspy.rtp_gui (the GUI) needs Qt; the processing modules do not.
    pr = subprocess.run([sys.executable, "-c", code], capture_output=True,
                        text=True, timeout=300)
    assert "ImportError" in pr.stderr and "PyQt5" in pr.stderr

    code = code.replace(", rtpspy.rtp_gui as g", "").replace(
        "raise SystemExit(1)",
        "from rtpspy import *\n"
        "assert not [m for m in sys.modules if m.startswith('PyQt5')]\n"
        "try:\n"
        "    rtpspy.RtpGUI\n"
        "except ImportError:\n"
        "    print('GUI needs Qt')\n")
    pr = subprocess.run([sys.executable, "-c", code], capture_output=True,
                        text=True, timeout=300)
    assert pr.returncode == 0, pr.stderr
    assert "GUI needs Qt" in pr.stdout


def test_chain_without_qt(tmp_path):
    res = _run_headless("chain", tmp_path)
    assert res["qt_modules"] == []
    assert res["RTP_setup"] == 0
    assert res["chain_head"] == "RtpWatch"
    # 40 volumes, TSHIFT ignores the first 2
    assert res["n_roi_signal"] == 38
    assert res["regress_shape"] == [64, 64, 40, 38]
    assert res["saved_exist"]
    for key in ("RtpWatch:processed image", "RtpVolreg:processed image",
                "RtpTshift:processed image", "RtpSmooth:processed image",
                "RtpRegress:processed image", "RTP parameters",
                "ROI signals"):
        assert key in res["saved"]
    assert res["physio_left"] == []


def test_make_masks_without_qt(tmp_path):
    """make_masks (with stub image processing steps) runs without Qt, also
    with progress_dlg=True (no dialog without a GUI)."""
    res = _run_headless("make_masks", tmp_path)
    assert res["qt_modules"] == []
    for key in ("plain", "progress_dlg"):
        assert res[key]["ret"] is None
        assert res[key]["stdout_restored"]
        assert res[key]["masks"] == {
            "RTP_mask": "RTP_mask.nii.gz", "GSR_mask": "GSR_mask.nii.gz",
            "WM_orig": "WM_alFunc.nii.gz", "Vent_orig": "Vent_alFunc.nii.gz",
            "aseg_orig": "aseg_alFunc.nii.gz",
            "alAnat": "anat_Brain_alFunc.nii.gz"}
    assert res["physio_left"] == []


def test_notifier_defaults(monkeypatch):
    """Without a GUI, errors go to the log and the requests are canceled."""
    import logging
    from rtpspy import notifier as nt

    n = nt.default_notifier()
    # (A Qt application of another test may be running in this process.)
    monkeypatch.setattr(nt, "_notifier", n)
    assert n.get_open_file("c", ".", "*") == ("", "")
    assert n.get_save_file("c", ".", "*") == ("", "")
    assert n.get_directory("c", ".") == ""
    assert n.get_int("t", "l", 3, 0, 5) == (3, False)
    assert n.get_text("l", "cmd") == ("cmd", False)
    assert n.question("q", buttons=("Yes", "No"), default="No") == "No"
    assert n.progress_dialog(title="x") is None

    from rtpspy.rtp_tshift import RtpTshift
    ts = RtpTshift()
    records = []

    class H(logging.Handler):
        def emit(self, record):
            records.append(record)

    h = H(level=logging.DEBUG)
    lg = logging.getLogger("RtpTshift")
    lg.addHandler(h)
    level = lg.level
    lg.setLevel(logging.DEBUG)
    try:
        ts.err_popup("test error")  # main_win is None: logged
    finally:
        lg.removeHandler(h)
        lg.setLevel(level)
    assert any("test error" in r.getMessage() for r in records)
    # A headless file selection is canceled
    assert ts.set_param("set_from_sample") == -1


@pytest.mark.parametrize("name", ["DlgProgressBar", "MatplotlibWindow"])
def test_moved_names(name):
    import rtpspy.rtp_common as rc
    import rtpspy.ui.widgets as w
    assert getattr(rc, name) is getattr(w, name)
