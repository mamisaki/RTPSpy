"""rt_physio: DummyRecording queues, and the rt_physio process driven via
RPC by RtpTTLPhysio (Dummy device fed with tests/ECG.1D and Resp.1D).

rt_physio is started with a unique socket name, a config file in a temporary
directory, no display (no plot window), and is always quit at the end."""
import os
import time
from multiprocessing import Pipe, Queue
from threading import Thread
from unittest import mock

import numpy as np
import pytest

from rtpspy.rt_physio import DummyRecording
from rtpspy.rtp_ttl_physio import RtpTTLPhysio


def _drain(q, wait=1.0):
    out = []
    st = time.time()
    while time.time() - st < wait:
        try:
            out.append(q.get(timeout=0.2))
        except Exception:
            break
    return out


def test_dummy_recording_queue_full_keeps_newest():
    on_q, off_q, ph_q = Queue(maxsize=2), Queue(maxsize=2), Queue(maxsize=3)
    rec = DummyRecording(on_q, off_q, ph_q, sample_freq=100)
    rec._sim_card = np.arange(10000.0)
    rec._sim_resp = np.arange(10000.0)
    rec._sim_data_len = 10000
    p_test, p_rec = Pipe()
    th = Thread(target=rec.read_signal_loop, kwargs={"cmd_pipe": p_rec},
                daemon=True)
    th.start()
    try:
        for t in (1.0, 2.0, 3.0):
            p_test.send(("PULSE", t))
            time.sleep(0.05)
        time.sleep(0.3)
    finally:
        p_test.send("QUIT")
        th.join(5)
    assert not th.is_alive()

    assert _drain(on_q) == [2.0, 3.0]
    assert np.allclose(_drain(off_q), [2.001, 3.001])
    cards = [c for _, c, _ in _drain(ph_q)]
    assert len(cards) == 3 and np.all(np.diff(cards) == 1)


# %% rt_physio process ========================================================
@pytest.fixture(scope="module")
def rt_physio(tmp_path_factory, test_dir):
    """RtpTTLPhysio with its own rt_physio process (Dummy device)."""
    work = tmp_path_factory.mktemp("rt_physio")
    name = f"TestRtPhysio_{os.getpid()}_{time.time_ns()}"
    env = {k: v for k, v in os.environ.items()
           if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    cwd = os.getcwd()
    os.chdir(work)  # rt_physio writes its dump folder in the cwd
    ph = None
    try:
        with mock.patch.dict(os.environ, env, clear=True):
            ph = RtpTTLPhysio(
                work_dir=work,
                rt_physio_address_name=["localhost", None, name],
                config_path=work / "rtmri_config.json")
    finally:
        os.chdir(cwd)

    try:
        assert ph.available, "rt_physio did not start"
        yield ph
    finally:
        proc = ph._rt_physio_proc
        ph.quit()
        if proc is not None:
            assert proc.poll() is not None, "rt_physio is still running"
            with pytest.raises(ProcessLookupError):
                os.killpg(proc.pid, 0)  # no process left in its group


@pytest.fixture(scope="module")
def dummy_feeding(rt_physio, test_dir):
    ph = rt_physio
    ph.TTLPhysioCom.call_rt_proc(
        ("START_DUMMY_FEEDING_WITH_FILES",
         (str(test_dir / "ECG.1D"), str(test_dir / "Resp.1D"), 40)),
        pkl=True)
    st = time.time()
    while time.time() - st < 10:
        ph.init_rbuf_access()  # re-read sample_freq
        if ph.get_params()["device"] == "Dummy" and ph.sample_freq == 40:
            break
        time.sleep(0.5)
    return ph


def test_get_params_before_device(rt_physio):
    pars = rt_physio.get_params()
    assert tuple(pars) == RtpTTLPhysio.PARAM_KEYS
    assert pars["device"] is None


def test_get_set_params_with_dummy(dummy_feeding):
    ph = dummy_feeding
    pars = ph.get_params()
    assert tuple(pars) == ("device", "sample_freq", "buf_len_sec",
                           "save_ttl")
    assert pars["device"] == "Dummy"
    assert pars["sample_freq"] == ph.sample_freq == 40
    assert pars["buf_len_sec"] == 3600
    assert isinstance(pars["save_ttl"], bool)

    # sample_freq follows rt_physio; keys other than PARAM_KEYS are ignored
    ph.set_param("sample_freq", 12345)
    assert ph.sample_freq == 40
    ph.set_param("work_dir", "/nonexistent")
    assert str(ph.work_dir) != "/nonexistent"


def _pulse_then_onset(ph, t):
    ph.TTLPhysioCom.call_rt_proc(("TTL_PULSE", t), pkl=True)
    st = time.time()
    while time.time() - st < 0.7:
        if ph.scan_onset != 0:
            break
        time.sleep(0.05)
    return ph.scan_onset


def test_wait_ttl_and_scan_onset(dummy_feeding):
    ph = dummy_feeding
    com = ph.TTLPhysioCom

    # WAIT_TTL_ON then a pulse sets the onset
    ph.scan_onset = 0
    com.call_rt_proc("WAIT_TTL_ON")
    t1 = time.time()
    assert _pulse_then_onset(ph, t1) == t1

    # Cancelled wait: the pulse leaves the onset 0
    ph.scan_onset = 0
    com.call_rt_proc("WAIT_TTL_ON")
    com.call_rt_proc("CANCEL_WAIT_TTL")
    assert _pulse_then_onset(ph, time.time()) == 0

    # Without a wait: onset unchanged
    ph.scan_onset = 0
    assert _pulse_then_onset(ph, time.time()) == 0

    # RtpTTLPhysio API: standby_scan then release
    ph.standby_scan()
    ph.release_standby_scan()
    assert _pulse_then_onset(ph, time.time()) == 0
    ph.standby_scan()
    t2 = time.time()
    assert _pulse_then_onset(ph, t2) == t2
    ph.scan_onset = 0
