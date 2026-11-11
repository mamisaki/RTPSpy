"""Plot window of rt_physio (PlotTTLPhysio, PyQt6): drawing, geometry,
commands, the dump dialog and the alarm dialogs, in this process with fake
ring buffers (no recorder process)."""
import time
from types import SimpleNamespace

import numpy as np
import pytest

QtWidgets = pytest.importorskip("PyQt6.QtWidgets")


class _FakeRB:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)

    def get_tail(self, n):
        return self.data[-n:].copy()

    def get(self):
        return self.data.copy()


class _FakePipe:
    def __init__(self):
        self.sent, self.inbox = [], []

    def send(self, msg):
        self.sent.append(msg)

    def poll(self, timeout=0):
        return bool(self.inbox)

    def recv(self):
        return self.inbox.pop(0)


@pytest.fixture
def plot(tmp_path):
    from rtpspy.rt_physio import PlotTTLPhysio
    from PyQt6 import QtCore  # noqa: F401

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ctrl = SimpleNamespace(_recorder_type="Dummy", dump_dir=tmp_path,
                           rt_mri_main_com=None)
    p = PlotTTLPhysio(ctrl, geometry="440x430+50+60", signal_freq=100)
    now = time.time()
    t = now - 20 + np.arange(2000) / 100
    rbuf = {
        "tstamp": _FakeRB(t),
        "card": _FakeRB(500 + 100 * np.sin(2 * np.pi * 1.2 * t)),
        "resp": _FakeRB(500 + 200 * np.sin(2 * np.pi * 0.25 * t)),
        "ttl_onsets": _FakeRB(now - 20 + np.arange(0, 20, 2.0)),
        "ttl_offsets": _FakeRB(now - 20 + np.arange(0, 20, 2.0) + 0.01),
    }
    import threading
    p._rbuf, p._rbuf_lock = rbuf, threading.Lock()
    p._cmd_pipe, p._reply_pipe, p._dump_pipe = (
        _FakePipe(), _FakePipe(), _FakePipe())
    p._init_window()
    p._win.show()
    app.processEvents()
    yield p, app
    p._running = False
    for w in app.topLevelWidgets():
        w.close()
    app.processEvents()


def test_draw_geometry_commands(plot):
    p, app = plot
    assert p._draw()
    ttl = p._ln_ttl[0].get_ydata()
    assert ttl.max() == 1 and ttl.min() == 0  # TTL pulses drawn

    w, h, x, y = (int(v) for v in
                  p.get_geometry().replace("x", "+").split("+"))
    assert (w, h) == (440, 430)
    p._handle_command(("RESIZE", "400x380+70+80"))
    app.processEvents()
    assert p.get_geometry().startswith("400x380")
    p._handle_command("GET_GEOMETRY")
    assert p._reply_pipe.sent[-1] == p.get_geometry()

    p._handle_command("HIDE")
    p._handle_command("GET_WINDOW_STATE")
    assert p._reply_pipe.sent[-1] == "withdrawn"
    p._handle_command("SHOW")
    p._handle_command("GET_WINDOW_STATE")
    assert p._reply_pipe.sent[-1] == "normal"

    p._handle_command(("SET_TITLE", "Numato"))
    assert p._win.windowTitle() == "Physio signals (Numato)"
    p._handle_command(("RESET", {"_plot_len_sec": 5}))
    assert len(p._ln_ttl[0].get_xdata()) == 500


def test_dump_dialog(plot):
    p, app = plot
    p.dump_data()
    dlg = p._dump_win
    assert dlg.isVisible()
    combo = dlg.findChild(QtWidgets.QComboBox)
    assert combo.count() >= 2  # Oldest data and the scan
    save = [b for b in dlg.findChildren(QtWidgets.QPushButton)
            if b.text() == "Save"][0]
    save.click()
    assert p._dump_pipe.sent[-1][0] == "DUMP"
    req_id = p._dump_pipe.sent[-1][2]
    dlg.close()
    assert dlg.isVisible()  # Not closed while saving

    now = time.time()
    p._dump_pipe.inbox.append(("DUMP_DONE", {
        "onset": now - 10, "offset": now, "path": "/x/physio.tsv"}, req_id))
    t0 = time.time()
    while time.time() - t0 < 3 and dlg.closable is False:
        app.processEvents()
        time.sleep(0.02)
    labels = " ".join(lb.text() for lb in dlg.findChildren(QtWidgets.QLabel))
    assert "Saved 0:00:10" in labels and "/x/physio.tsv" in labels
    dlg.close()
    assert not dlg.isVisible()


def test_alarm_dialogs(plot, monkeypatch):
    p, app = plot
    played = []
    monkeypatch.setattr(p, "_play_alarm", lambda: played.append(1))
    p.show_ttl_alarm("No TTL for 10 s")
    dlg = p._ttl_alarm_win
    assert dlg.isVisible() and "TTL missing" == dlg.windowTitle()
    p.show_ttl_alarm("again")  # The open dialog is kept
    assert p._ttl_alarm_win is dlg
    p.show_physio_alarm("Flat cardiogram")
    assert p._physio_alarm_win.isVisible()
    assert len(played) >= 2
    dlg.close()
    app.processEvents()
    assert not p._is_open(p._ttl_alarm_win)


def test_close_sends_quit(plot):
    p, app = plot
    p._win.close()
    assert "QUIT" in p._cmd_pipe.sent


def test_close_hides_when_disabled(plot):
    """With disable_close (RTPSpy starts rt_physio so), the close button
    hides the window and the recording goes on."""
    p, app = plot
    sent = []
    p._disable_close = True
    p.controller.rt_mri_main_com = SimpleNamespace(
        rpc_ping=lambda: True, call_rt_proc=lambda msg: sent.append(msg))
    p._win.close()
    app.processEvents()
    assert p.window_state() == "withdrawn"
    assert "QUIT" not in p._cmd_pipe.sent
    assert sent == ["HIDE_PHYSIO"]  # RT-MRI main is told, if it runs
    p._handle_command("SHOW")
    assert p.window_state() == "normal"
