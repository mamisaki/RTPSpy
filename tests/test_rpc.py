"""RPCSocketServer / RPCSocketCom (rpc_socket_server.py): local round trip
with the token, port lookup (local config and mocked ssh for a remote host),
returns without a port, and a server that failed to start."""
import json
import time
from pathlib import Path
from unittest import mock

import pytest

from rtpspy import rpc_socket_server as rs


@pytest.fixture
def rpc_server(tmp_path):
    received = []

    def handler(data):
        received.append(data)
        if data == "ECHO_STR":
            return "pong"
        if isinstance(data, tuple) and data[0] == "ECHO_OBJ":
            return {"echo": data[1]}
        return None

    cfg = tmp_path / "rtmri_config.json"
    name = f"TestRPC_{time.time_ns()}"
    srv = rs.RPCSocketServer(cfg, RPC_handler=handler, socket_name=name)
    yield srv, cfg, name, received
    srv.shutdown()


def test_server_records_port(rpc_server):
    srv, cfg, name, _ = rpc_server
    assert srv.port is not None
    assert json.loads(cfg.read_text())[f"{name}_port"] == srv.port
    assert cfg.with_suffix(".token").stat().st_mode & 0o077 == 0


def test_round_trip(rpc_server):
    srv, cfg, name, received = rpc_server
    com = rs.RPCSocketCom(["localhost", None, name], cfg)
    assert com.get_port() == srv.port
    assert com.rpc_ping()
    assert com.call_rt_proc("ECHO_STR", get_return=True) in ("pong", b"pong")
    obj = {"a": [1, 2, 3], "b": "x;y"}
    assert com.call_rt_proc(("ECHO_OBJ", obj), pkl=True,
                            get_return=True) == {"echo": obj}
    com.call_rt_proc("NO_RETURN")
    st = time.time()
    while "NO_RETURN" not in received and time.time() - st < 5:
        time.sleep(0.05)
    assert "NO_RETURN" in received


def test_message_without_token_is_dropped(rpc_server, tmp_path):
    srv, cfg, name, received = rpc_server
    # A client with another token (another config path) is rejected
    other_cfg = tmp_path / "other" / "rtmri_config.json"
    com = rs.RPCSocketCom(["localhost", srv.port, name], other_cfg)
    assert com.call_rt_proc("ECHO_STR", get_return=True) is None
    assert "ECHO_STR" not in received


def test_remote_port_via_mocked_ssh():
    calls = []

    def fake_run(cmd, **kw):
        return mock.Mock(returncode=0)

    def fake_check_output(cmd, **kw):
        calls.append(cmd)
        return json.dumps({"X_port": 1234}).encode()

    cfg = Path.home() / ".RTPSpy" / "rtmri_config.json"
    com = rs.RPCSocketCom(["remotehost", None, "X"], cfg)
    with mock.patch.object(rs.subprocess, "run", fake_run), \
            mock.patch.object(rs.subprocess, "check_output",
                              fake_check_output):
        assert com.get_port() == 1234
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[:2] == ["ssh", "remotehost"]
    # A path under the local home is read relative to the remote home
    assert cmd[-1] == "cat .RTPSpy/rtmri_config.json"


def test_remote_ssh_failure_returns_none():
    def fake_run(cmd, **kw):
        return mock.Mock(returncode=0)

    def fake_check_output(cmd, **kw):
        raise OSError("ssh failed")

    com = rs.RPCSocketCom(["remotehost", None, "X"],
                          Path("rtmri_config.json"))
    with mock.patch.object(rs.subprocess, "run", fake_run), \
            mock.patch.object(rs.subprocess, "check_output",
                              fake_check_output):
        assert com.get_port() is None


def test_no_port_returns(tmp_path):
    com = rs.RPCSocketCom(["localhost", None, "NoSuchSocket"],
                          tmp_path / "none.json")
    assert com.get_port() is None
    assert com.call_rt_proc("ping") == (False, None)
    assert com.call_rt_proc("X") is None
    assert com.call_rt_proc("X", get_return=True) is None
    assert com.rpc_ping() is False


def test_failed_server_start(tmp_path):
    with mock.patch.object(rs.socketserver, "TCPServer",
                           side_effect=OSError("x")):
        srv = rs.RPCSocketServer(tmp_path / "none.json")
    assert srv._server is None and srv.port is None
    srv.shutdown()  # must not raise
    del srv
