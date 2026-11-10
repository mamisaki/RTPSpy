"""Message framing of RTP_SERVE (rtp_serve.py): text commands separated by
';', pickled (PKL_<n>_) and compressed pickled (ZPKL_<n>_) objects, split
packets, malformed messages, NF signals and IsAlive."""
import pickle
import socket
import time
import zlib

import pytest

from rtpspy.rtp_serve import RTP_SERVE

TIMEOUT = 5


def pack(data, z=False):
    b = pickle.dumps(data)
    if z:
        b = zlib.compress(b)
        return f"ZPKL_{len(b)}_".encode() + b
    return f"PKL_{len(b)}_".encode() + b


@pytest.fixture
def server():
    srv = RTP_SERVE()
    assert srv.server is not None
    sock = socket.create_connection(srv.server.server_address,
                                    timeout=TIMEOUT)
    sock.settimeout(TIMEOUT)
    st = time.time()
    while not srv.connected and time.time() - st < TIMEOUT:
        time.sleep(0.02)
    yield srv, sock
    try:
        sock.close()
    except OSError:
        pass
    srv.server.kill = True
    srv.server.shutdown()
    srv.server.server_close()


def drain(srv, n, timeout=TIMEOUT):
    """Get n items from the receive queue (with a timeout), then any extra
    items that arrive shortly after (to detect duplicates)."""
    out = []
    st = time.time()
    while len(out) < n and time.time() - st < timeout:
        d = srv.get_recv_queue(timeout=0.1)
        if d is not None:
            out.append(d)
    time.sleep(0.2)
    while (d := srv.get_recv_queue()) is not None:
        out.append(d)
    return out


def test_mixed_messages_in_one_send(server):
    srv, sock = server
    obj1 = {"a": "x;y;z", "b": b";;;"}
    obj2 = list(range(5000)) + ["semi;colon"]
    sock.sendall(b"CMD1;" + pack(obj1) + pack(obj2, z=True) + b"CMD2;")
    assert drain(srv, 4) == ["CMD1", obj1, obj2, "CMD2"]


def test_split_packets_no_duplication(server):
    srv, sock = server
    big = {"k": "v;" * 20000}
    pk = pack(big) + b"CMD3;"
    n = len(pk)
    for p in (pk[:2], pk[2:7], pk[7:n // 3], pk[n // 3:-3], pk[-3:]):
        sock.sendall(p)
        time.sleep(0.05)
    assert drain(srv, 2) == [big, "CMD3"]

    obj = {"a": 1}
    zp = pack(obj, z=True)
    sock.sendall(zp[:5])
    time.sleep(0.05)
    sock.sendall(zp[5:])
    assert drain(srv, 1) == [obj]


def test_malformed_messages_are_skipped(server):
    srv, sock = server
    sock.sendall(b"NF abc,1,2;NF ;NF 1.0,2,3.5;" + b"PKL_3_xyz" + b"CMD4;")
    assert drain(srv, 1) == ["CMD4"]
    st = time.time()
    while len(srv.NF_signal) < 1 and time.time() - st < TIMEOUT:
        time.sleep(0.05)
    nf = srv.NF_signal
    assert len(nf) == 1 and nf.iloc[0]["TR"] == 2


def test_isalive_and_disconnect(server):
    srv, sock = server
    sock.sendall(b"IsAlive?;")
    assert sock.recv(100) == b"Yes."
    assert srv.connected
    sock.close()
    st = time.time()
    while srv.connected and time.time() - st < TIMEOUT:
        time.sleep(0.1)
    assert not srv.connected
