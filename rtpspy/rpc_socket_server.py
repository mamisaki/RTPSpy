#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remote Procedure Call (RPC) via a network socket.
RPCSocketServer: RPC socket server class.
RPCSocketCom: RPC client class.
pack_data: Pack (pickle) data to send.
recv_message, parse_message: Receive a message and unpack it.
get_rpc_token: The token that authenticates a message.

@author: mmisaki@libr.net
"""

# %% import ===================================================================
import logging
import sys
import traceback
from threading import Thread
import socketserver
import socket
import pickle
from pathlib import Path
import zlib
import time
import re
import json
import subprocess
import os
import hmac
import secrets
import shlex
import tempfile
import tkinter as tk
from tkinter import simpledialog


# %% pack_data ================================================================
def pack_data(data, compress=False):
    """
    Pack data to be sent. Adds a header 'PKL_{bytesize}_' followed by the
    pickled byte string data.

    Parameters
    ----------
    data : any type
        Any picklable data.
    compress : bool, optional
        Flag to compress the data with zlib. The default is False.

    Returns
    -------
    pkl_data : byte string
        Pickled byte string data.
    """
    try:
        pkl_data = pickle.dumps(data)
        if compress:
            pkl_data = zlib.compress(pkl_data)
            pkl_data = f"ZPKL_{len(pkl_data)}_".encode("utf-8") + pkl_data
        else:
            pkl_data = f"PKL_{len(pkl_data)}_".encode("utf-8") + pkl_data
    except Exception as e:
        errstr = str(e) + "\n" + traceback.format_exc()
        sys.stderr.write(errstr)
        pkl_data = None

    return pkl_data


# %% Authentication ==========================================================
# Every message to an RPCSocketServer starts with AUTH_<token>_. The token is
# a random secret in <config_path>.token, readable only by the user (0600).
# The servers listen on localhost only, but any local user could connect;
# without the token, a message is dropped before it is unpickled (unpickling
# can run arbitrary code).
_AUTH_PREFIX = b"AUTH_"
_TOKEN_LEN = 32  # hex characters
_AUTH_LEN = len(_AUTH_PREFIX) + _TOKEN_LEN + 1


def get_rpc_token(config_path):
    """Return the RPC token, creating <config_path>.token (mode 0600) if it
    does not exist. Safe when several processes start at once: the file is
    written to a temporary file and linked into place."""
    path = Path(config_path).with_suffix(".token")
    for _ in range(100):
        try:
            if path.stat().st_mode & 0o077:
                os.chmod(path, 0o600)  # Keep it private
            tok = path.read_text().strip()
            if len(tok) == _TOKEN_LEN:
                return tok
        except FileNotFoundError:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".rpc_token_")
            try:
                with os.fdopen(fd, "w") as fid:  # mkstemp makes it 0600
                    fid.write(secrets.token_hex(_TOKEN_LEN // 2))
                os.link(tmp, path)
            except FileExistsError:
                pass  # Made by another process
            finally:
                os.unlink(tmp)
            continue
        time.sleep(0.02)
    raise RuntimeError(f"Cannot read the RPC token in {path}")


# %% Message receiving =======================================================
_PKL_HEADER = re.compile(rb"Z?PKL_(\d+)_")


def recv_message(sock, logger, idle_timeout=1.5, max_sec=60.0):
    """Receive one message from sock.

    Reading stops when the peer closes the connection, when nothing arrives
    for idle_timeout seconds, when a pickled message (PKL_<size>_ header) is
    complete, or after max_sec seconds in total. The loop always ends, so an
    incomplete message (e.g., the sender timed out) cannot hang the caller.

    Returns the received bytes (may be incomplete; see parse_message).
    """
    sock.settimeout(idle_timeout)
    buf = bytearray()
    need = None
    t_end = time.time() + max_sec
    while time.time() < t_end:
        try:
            chunk = sock.recv(65536)
        except socket.timeout:
            break
        except OSError as e:
            logger.error(f"Error receiving data: {e}")
            break
        if not chunk:  # Closed by the peer
            break
        buf.extend(chunk)
        # A message to the server starts with the AUTH_<token>_ prefix
        off = _AUTH_LEN if buf[:len(_AUTH_PREFIX)] == _AUTH_PREFIX else 0
        if need is None and len(buf) > off and \
                buf[off:off + 1] in (b"P", b"Z"):
            ma = _PKL_HEADER.match(bytes(buf[off:off + 32]))
            if ma:
                need = off + ma.end() + int(ma.group(1))
        if need is not None and len(buf) >= need:
            break
    return buf


def parse_message(buf, logger):
    """Return the object (pickled message) or str in buf. None if buf is
    empty, or a pickled message is incomplete or broken."""
    if not len(buf):
        return None
    if buf.startswith(b"PKL_") or buf.startswith(b"ZPKL_"):
        ma = _PKL_HEADER.match(bytes(buf[:32]))
        if ma is None:
            logger.error("Invalid pickle header format")
            return None
        dsize = int(ma.group(1))
        payload = bytes(buf[ma.end():ma.end() + dsize])
        if len(payload) < dsize:
            logger.error(
                f"Incomplete message: {len(payload)} of {dsize} bytes "
                "received. The message is dropped."
            )
            return None
        try:
            if buf.startswith(b"Z"):
                payload = zlib.decompress(payload)
            return pickle.loads(payload)
        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            logger.error(f"Unpickling error: {errstr}")
            return None
    return buf.decode(errors="replace")


# %% RPCSocketCom =============================================================
class RPCSocketCom:

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, address_name, config_path):
        self.address_name = address_name
        self.config_path = Path(config_path)
        self._socket_name = address_name[2]
        self._logger = logging.getLogger(self._socket_name)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_port(self):
        """
        Get the port number of the socket (address_name[2]) from the
        config file. For a remote host (address_name[0]), the config file
        is read via ssh.

        Returns
        -------
        int or None
            The port number, or None if it is not found.
        """
        host = self.address_name[0]
        socket_name = self.address_name[2]

        if host is None or host in ("localhost", "127.0.0.1"):
            # Read local config
            if self.config_path.is_file():
                with open(self.config_path, "r") as fid:
                    rtpspy_config = json.load(fid)
                port = rtpspy_config.get(f"{socket_name}_port", None)
                if port is not None:
                    port = int(port)
            else:
                port = None
        else:
            # SSH access to read JSON file on a remote host
            try:
                # Check if a password is required
                ret = subprocess.run(
                    [
                        "ssh",
                        "-o",
                        "PasswordAuthentication=no",
                        "-o",
                        "BatchMode=yes",
                        host,
                        "exit",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ).returncode

                if ret != 0:
                    root = tk.Tk()
                    root.withdraw()
                    password = simpledialog.askstring(
                        f"Password to access {host}",
                        f"Enter password for {host}:",
                        show="*",
                    )
                    root.destroy()
                    if not password:
                        self._logger.error(
                            "Password required for SSH access")
                        return None
                    ssh_cmd = ["sshpass", "-p", password, "ssh", host]
                else:
                    ssh_cmd = ["ssh", host]

                # The config file on the remote host. A path under the local
                # home is taken as relative to the remote home.
                cfg_f = self.config_path
                if cfg_f.is_relative_to(Path.home()):
                    cfg_f = cfg_f.relative_to(Path.home())
                cmd = ssh_cmd + [f"cat {shlex.quote(str(cfg_f))}"]

                output = subprocess.check_output(cmd).decode()
                rtpspy_config = json.loads(output)
                port = rtpspy_config.get(f"{socket_name}_port", None)
                if port is not None:
                    port = int(port)

            except Exception as e:
                errstr = str(e) + "\n" + traceback.format_exc()
                error_message = f"Error accessing remote config file: {errstr}"
                self._logger.error(error_message)
                return None

        return port

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def call_rt_proc(
        self,
        data,
        pkl=False,
        get_return=False,
        timeout=1.5
    ):
        """
        Call a remote procedure.

        Parameters
        ----------
        data : any type
            Data to send.
        pkl : bool, optional
            Flag to pack the data in pickle format. Defaults to False.
        get_return : bool, optional
            Flag to receive a return value. Defaults to False.
        timeout : float, optional
            Timeout for the connection. Defaults to 1.5 seconds.

        Returns
        -------
        For data 'ping': (True if connected, port). Otherwise, the response
        of the remote procedure if get_return, or None.
        """
        remote_proc_address = tuple(self.address_name[:2])
        port = remote_proc_address[1]
        if port is None:
            port = self.get_port()
            if port is None:
                if data == 'ping':
                    return False, None
                return None
            else:
                remote_proc_address = (remote_proc_address[0], port)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.settimeout(timeout)
            sock.connect(remote_proc_address)
        except OSError:
            # Refused (not running), timed out (busy or hung), etc. No
            # wait here: callers that wait for a process sleep themselves,
            # and a wait here would delay every call to a dead process.
            sock.close()
            if data == 'ping':
                return False, port
            return

        if data == 'ping':
            sock.close()
            return True, port

        if not self.rpc_send_data(sock, data, pkl=pkl):
            errmsg = f'Failed to send {data}'
            self._logger.error(errmsg)
            sock.close()
            return None

        if get_return:
            # Tell the server that the call is complete. Without this, the
            # server reads until its 1.5 s receive timeout.
            sock.shutdown(socket.SHUT_WR)
            rcv_data = self.rpc_recv_data(sock)
            if rcv_data is None:
                errmsg = f'No response for call {data}'
                self._logger.warning(errmsg)

            self._logger.debug(f"Received data: {rcv_data}")
            sock.close()
            return rcv_data

        sock.close()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def rpc_ping(self, timeout=1.5):
        """
        Ping the remote procedure to check connectivity.

        Parameters
        ----------
        timeout : float, optional
            Timeout for the ping. Defaults to 1.5 seconds.

        Returns
        -------
        bool
            True if the ping is successful, False otherwise.
        """
        pong, port = self.call_rt_proc("ping", timeout=timeout)
        if not pong and port is not None:
            # The port may be outdated (e.g., the server restarted). Retry
            # with the port read from the config file.
            self.address_name[1] = port
            pong, port = self.call_rt_proc("ping", timeout=timeout)

        if not pong:
            self.address_name[1] = None

        return pong

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def rpc_send_data(self, sock, data, pkl=False):
        """
        Send data via a socket connected to the RPCSocketServer.

        Parameters
        ----------
        sock : socket.socket
            Connected socket.
        data : bytes, str or any picklable object
            Message to send.
        pkl : bool, optional
            Flag to send data as a pickled byte array. Defaults to False.

        Returns
        -------
        bool
            True if the data is sent successfully, False otherwise.
        """
        if pkl:
            data = pack_data(data)
        elif type(data) is str:
            data = data.encode("utf-8")
        try:
            token = get_rpc_token(self.config_path)
        except Exception as e:
            self._logger.error(f"No RPC token: {e}")
            return False
        data = _AUTH_PREFIX + token.encode() + b"_" + data

        try:
            sock.sendall(data)
            return True

        except OSError as e:  # Broken pipe, timeout, etc.
            self._logger.error(f"Failed to send data: {e}")
            return False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def rpc_recv_data(self, sock, timeout=10.0):
        """
        Receive the reply of the RPCSocketServer.

        Parameters
        ----------
        sock : socket.socket
            Connected socket.
        timeout : float
            Seconds to wait for the reply (and between its parts). The
            server closes the connection after replying, so a complete reply
            returns at once.

        Returns
        -------
        Received data, or None if there is no (complete) reply.
        """
        buf = recv_message(sock, self._logger, idle_timeout=timeout,
                           max_sec=max(30.0, timeout))
        return parse_message(buf, self._logger)


# %% class RPCSocketServer ====================================================
class RPCSocketServer:
    """
    RPC socket server class.
    Starts a socket server in __init__().
    Receives a call and passes it to the RPC_handler function.
    When the handler function returns a value, it is sent back to the client.
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(
        self,
        config_path,
        RPC_handler=print,
        handler_kwargs={},
        socket_name="RPCSocketServer",
        allow_remote_access=False,
    ):
        self._socket_name = socket_name
        self._logger = logging.getLogger(self._socket_name)
        self._config_path = Path(config_path)

        if allow_remote_access:
            host = "0.0.0.0"
        else:
            host = "localhost"

        # Boot server
        socketserver.TCPServer.allow_reuse_address = True
        try:
            self._server = socketserver.TCPServer(
                (host, 0), RPCSocketServer._recvDataHandler
            )
        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            self._logger.error("Failed to start TCPServer\n%s", errstr)
            self._server = None
            self.port = None
            return

        self.port = self._server.server_address[1]
        self._logger.debug(f"Server started on port {self.port}")
        self._server._callback = RPC_handler
        self._server._callback_kwargs = handler_kwargs
        self._server._token = get_rpc_token(self._config_path).encode()
        self._server._cancel = False
        self._server.socket_name = self._socket_name

        # Save the port number in the config file
        config_path = self._config_path
        if config_path.is_file():
            with open(config_path, "r") as fid:
                rtpspy_config = json.load(fid)
        else:
            if not config_path.parent.is_dir():
                config_path.parent.mkdir()
            rtpspy_config = {}

        rtpspy_config[f"{socket_name}_port"] = self.port
        with open(config_path, "w") as fid:
            json.dump(rtpspy_config, fid)

        # Start the server on another thread
        self._server_thread = Thread(
            target=self._server.serve_forever, args=(0.5,))
        # Make the server thread exit when the main thread terminates
        self._server_thread.daemon = True
        self._server_thread.start()

    # /////////////////////////////////////////////////////////////////////////
    class _recvDataHandler(socketserver.BaseRequestHandler):
        def handle(self):
            """Receive one message, pass it to the callback, and send back
            the return value."""
            self._logger = logging.getLogger(self.server.socket_name)

            client_addr, port = self.client_address
            self._logger.debug(f"Connected from {client_addr}:{port}")
            # --- Receive one message ---
            buf = recv_message(self.request, self._logger, idle_timeout=1.5)
            if len(buf):
                # Check the token before anything is unpickled
                expected = _AUTH_PREFIX + self.server._token + b"_"
                if not hmac.compare_digest(bytes(buf[:_AUTH_LEN]),
                                           expected):
                    self._logger.warning(
                        f"Dropped a message without the RPC token from "
                        f"{client_addr}:{port}")
                    return
                buf = buf[_AUTH_LEN:]
            data = parse_message(buf, self._logger)
            if data is not None:
                self._logger.debug(
                    f"Received {len(buf)} bytes from {client_addr}")

            # Always try to call the callback, even if data is None
            self._logger.debug(f"Calling callback with data: {type(data)}")

            if (
                hasattr(self.server, "_callback")
                and self.server._callback is not None
            ):
                try:
                    ret = self.server._callback(
                        data, **self.server._callback_kwargs
                    )
                    self._logger.debug(f"Callback returned: {type(ret)}")

                    if ret is not None:
                        if type(ret) is str:
                            ret = ret.encode("utf-8")
                        elif not isinstance(ret, bytes):
                            # Neither bytes nor str: pickle it
                            ret = pack_data(ret)

                        self.request.sendall(ret)
                        self._logger.debug(
                            f"Sent response of {len(ret)} bytes"
                        )
                    else:
                        self._logger.debug(
                            "Callback returned None; no response sent"
                        )

                except Exception as e:
                    errstr = str(e) + "\n" + traceback.format_exc()
                    self._logger.error(f"Callback execution error: {errstr}")
            else:
                self._logger.error(
                    "No callback function is set."
                )

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def shutdown(self):
        """
        Shut down the server and close its socket.
        """
        if getattr(self, "_server", None) is not None:
            self._server.shutdown()
            self._server.server_close()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        """
        Clean up resources and remove the port number from the config file.
        """
        config_f = getattr(self, "_config_path", None)
        if config_f is not None and config_f.is_file():
            try:
                with open(config_f, "r") as fid:
                    rtpspy_config = json.load(fid)
                rtpspy_config.pop(f"{self._server.socket_name}_port", None)
                with open(config_f, "w") as fid:
                    json.dump(rtpspy_config, fid)
            except Exception:
                pass

        self.shutdown()
