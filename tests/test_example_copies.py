"""The example presentation scripts use copies of rtpspy/rtp_serve.py
(they run in the PsychoPy environment without rtpspy). The copies must be
the same as the library, apart from the note line in the docstring."""
from pathlib import Path

import pytest

ROOT = Path(__file__).absolute().parents[1]
NOTE = "(a copy of rtpspy/rtp_serve.py so that this example runs without rtpspy)"


@pytest.mark.parametrize("app", ["ROI-NF", "LA-NF"])
def test_rtp_serve_copy(app):
    lib = (ROOT / "rtpspy" / "rtp_serve.py").read_text().split("\n")
    copy = (ROOT / "example" / app / "rtp_serve.py").read_text().split("\n")
    assert NOTE in copy
    copy.remove(NOTE)
    assert copy == lib, (
        f"example/{app}/rtp_serve.py differs from rtpspy/rtp_serve.py;"
        " copy the library file and add the note line")
