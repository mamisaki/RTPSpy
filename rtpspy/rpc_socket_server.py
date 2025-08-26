#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides all the utilities for Remote Procedure Call (RPC) via network socket.
RPCSocketServer: RPC socket server class
pack_data: utility function to pack data.
send_data: utility function to send data.
recv_resp: utility function to receive data.

mmisaki@libr.net
"""

# %% import ===================================================================
import logging
import subprocess
import sys
import traceback
from threading import Thread
import socketserver
import socket
import pickle
import zlib
import time
import re
from pathlib import Path
import json
import tkinter as tk
from tkinter import simpledialog


# %% pack_data ================================================================
def pack_data(data, compress=False):
    """
    Pack data to be sent. PKL_{bytesize}_' header is added
    followed by the pickled byte string data.

    Parameters
    ----------
    data : any type
        Any picklable data.
    compress : bool (optional)
        Flag to compress the data with zlib. The default is False.

    Returns
    -------
    pkl_data : byte string
        pickled byte string data.
    """
    try:
        pkl_data = pickle.dumps(data)
        if compress:
            pkl_data = zlib.compress(pkl_data)
            pkl_data = f"ZPKL_{len(pkl_data)}_".encode("utf-8") + pkl_data
        else:
            pkl_data = f"PKL_{len(pkl_data)}_".encode("utf-8") + pkl_data
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        errstr = "".join(traceback.format_exception(exc_type, exc_obj, exc_tb))
        sys.stderr.write(str(e) + "\n" + errstr)
        pkl_data = None

    return pkl_data


# %% rpc_send_data ============================================================
def rpc_send_data(sock, data, pkl=False, logger=None):
    """
    Send data via a socket connected to the RPCSocketServer.
    Parameters
    ----------
    data : bytes
        message to send.
    pkl : bool, optional
        send data as a pickled bytearray.
    """
    if pkl:
        data = pack_data(data)
    elif type(data) is str:
        data = data.encode("utf-8")

    try:
        sock.sendall(data)
        return True

    except BrokenPipeError:
        if logger:
            logger.error("No connection")
        else:
            sys.stderr.write("No connection")
        return False


# %% rpc_recv_data ============================================================
def rpc_recv_data(sock, logger=None):
    """
    Receive data via a socket connected to the RPCSocketServer.
    """

    # Receive data loop: Run until data is complete.
    serialized_data = bytearray()
    while True:
        # Append data in serialized_data until socket.timeout
        while True:  # loop until socket.timeout to collect all data
            try:
                recv_data = b""
                recv_data = sock.recv(4096)
                if not recv_data:
                    break
                serialized_data.extend(recv_data)

            except socket.timeout:
                if len(recv_data):
                    serialized_data.extend(recv_data)
                break
            except BrokenPipeError:
                if logger:
                    logger.error("Connection closed")
                break
            except BlockingIOError:
                # No more data available at the moment
                break
            except Exception:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                errstr = "".join(
                    traceback.format_exception(exc_type, exc_obj, exc_tb)
                )
                if logger:
                    logger.error(f"Data receiving error: {errstr}")
                break
            time.sleep(0.001)

        # Process the received data
        data = None
        if len(serialized_data):
            if serialized_data.startswith(
                "PKL_".encode("utf-8")
            ) or serialized_data.startswith("ZPKL_".encode("utf-8")):
                # Data is pickle
                # Get data size
                dstr = serialized_data.decode("utf-8", "backslashreplace")
                ma = re.search(r"Z*PKL_(\d+)_", dstr)
                dsize = int(ma.groups()[0])
                prefix = ma.group()
                # Check if data is complete
                pkldata = serialized_data.replace(
                    prefix.encode("utf-8"), "".encode("utf-8")
                )
                if len(pkldata) < dsize:
                    # Data is not complete.
                    # Keep the current data in serialized_data and
                    # continue receiving the remaining data.
                    continue

                # Extract pkldata part of the byte string
                pkldata = pkldata[:dsize]
                try:
                    # Unpickling the data
                    if prefix.startswith("Z"):
                        pkldata = zlib.decompress(pkldata)
                    data = pickle.loads(pkldata)
                except Exception:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    errstr = "".join(
                        traceback.format_exception(exc_type, exc_obj, exc_tb)
                    )
                    if logger:
                        logger.error(errstr)
            else:
                # Assume simple string
                data = serialized_data.decode()

        break

    return data


# %% get_port_by_name =========================================================
def get_port_by_name(socket_name, host=None):
    """
    Get the port number of a named socket.

    Parameters
    ----------
    socket_name : str
        Name of the socket.
    host : str or None, optional
        Hostname to query for the port. If None, reads local config.

    Returns
    -------
    tuple
        (port, None) if successful, or (None, error_message) if an error occurs
    """
    if host is None:  # Read local config
        config_f = Path.home() / ".RTPSpy" / "rtpspy"
        if config_f.is_file():
            with open(config_f, "r") as fid:
                rtpspy_config = json.load(fid)
            port = rtpspy_config.get(f"{socket_name}_port", None)
        else:
            port = None
    else:
        # ssh access to read json file in remote host
        try:
            # Check if password is required
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
                    return None, "Password required for SSH access"
                cmd = (
                    f"sshpass -p '{password}' ssh {host} 'cat .RTPSpy/rtpspy'"
                )
            else:
                cmd = f"ssh {host} 'cat .RTPSpy/rtpspy'"

            output = subprocess.check_output(cmd, shell=True).decode()
            rtpspy_config = json.loads(output)
            port = rtpspy_config.get(f"{socket_name}_port", None)

        except Exception as e:
            port = None
            error_message = f"Error accessing remote config file: {e}"
            return port, error_message

    return port, None


# %% class RPCSocketServer ====================================================
class RPCSocketServer:
    """
    RPC socket server class
    Start a socket server in __init__().
    Receive a call and pass it to the RPC_handler function.
    When the handler function returns a value, pass it back to the client.
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(
        self,
        RPC_handler=print,
        socket_name="RPCSocketServer",
        allow_remote_access=False,
    ):
        self._socket_name = socket_name
        self._logger = logging.getLogger(self._socket_name)

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
        except Exception:
            self._logger.error(
                "Failed to start TCPServer\n%s", traceback.format_exc()
            )
            self.server = None
            return

        self.port = self._server.server_address[1]
        self._logger.debug(f"Server started on port {self.port}")
        self._server._callback = RPC_handler
        self._server._cancel = False
        self._server.socket_name = self._socket_name

        # Save the port number in rtpspy config file
        config_f = Path.home() / ".RTPSpy" / "rtpspy"
        if config_f.is_file():
            with open(config_f, "r") as fid:
                rtpspy_config = json.load(fid)
        else:
            if not config_f.parent.is_dir():
                config_f.parent.mkdir()
            rtpspy_config = {}

        rtpspy_config[f"{socket_name}_port"] = self.port
        with open(config_f, "w") as fid:
            json.dump(rtpspy_config, fid)

        # Start the server on another thread.
        self._server_thread = Thread(
            target=self._server.serve_forever, args=(0.5,))
        # Make the server thread exit when the main thread terminates
        self._server_thread.daemon = True
        self._server_thread.start()

    # /////////////////////////////////////////////////////////////////////////
    class _recvDataHandler(socketserver.BaseRequestHandler):
        def handle(self):
            self._logger = logging.getLogger(self.server.socket_name)

            # --- Initialize ---
            client_addr, port = self.client_address
            self._logger.debug(f"Connected from {client_addr}:{port}")
            self.request.settimeout(1)

            # --- Request handling loop ---
            # Receive data loop: Run until data is complete.
            serialized_data = bytearray()
            while True:
                # Append data in serialized_data until socket.timeout
                while True:  # loop until socket.timeout to collect all data
                    try:
                        recv_data = b""
                        recv_data = self.request.recv(4096)
                        if not recv_data:
                            break
                        serialized_data.extend(recv_data)

                    except socket.timeout:
                        if len(recv_data):
                            serialized_data.extend(recv_data)
                        break
                    except BrokenPipeError:
                        self._logger.error(
                            f"Connection from {client_addr} closed"
                        )
                        return  # Exit the handler
                    except BlockingIOError:
                        # No more data available at the moment
                        break
                    except Exception:
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        errstr = "".join(
                            traceback.format_exception(
                                exc_type, exc_obj, exc_tb))
                        self._logger.error(f'Data receiving error: {errstr}')
                        break
                    time.sleep(0.001)

                # Process the received data
                data = None
                if len(serialized_data):
                    if serialized_data.startswith(
                        "PKL_".encode("utf-8")
                    ) or serialized_data.startswith("ZPKL_".encode("utf-8")):
                        # Data is pickle
                        # Get data size
                        dstr = serialized_data.decode(
                            "utf-8", "backslashreplace"
                        )
                        ma = re.search(r"Z*PKL_(\d+)_", dstr)
                        if ma is None:
                            self._logger.error("Invalid pickle header format")
                            break
                        dsize = int(ma.groups()[0])
                        prefix = ma.group()
                        # Check if data is complete
                        pkldata = serialized_data.replace(
                            prefix.encode("utf-8"), "".encode("utf-8")
                        )
                        if len(pkldata) < dsize:
                            # Data is not complete.
                            # Keep the current data in serialized_data and
                            # continue receiving the remaining data.
                            continue

                        # Extract pkldata part of the byte string
                        pkldata = pkldata[:dsize]
                        try:
                            # Unpickling the data
                            if prefix.startswith("Z"):
                                pkldata = zlib.decompress(pkldata)
                            data = pickle.loads(pkldata)
                            self._logger.debug(
                                f"Received {dsize} byte data"
                                + f" by {self.server.server_address}"
                            )
                        except Exception:
                            exc_type, exc_obj, exc_tb = sys.exc_info()
                            errstr = "".join(
                                traceback.format_exception(
                                    exc_type, exc_obj, exc_tb
                                )
                            )
                            self._logger.error(f"Unpickling error: {errstr}")
                            break
                    else:
                        # Assume simple string
                        data = serialized_data.decode()
                        self._logger.debug(
                            f"Received string data: {data[:100]}..."
                        )

                break

            # Always try to call the callback, even if data is None
            self._logger.debug(f"Calling callback with data: {type(data)}")

            if (
                hasattr(self.server, "_callback")
                and self.server._callback is not None
            ):
                try:
                    ret = self.server._callback(data)
                    self._logger.debug(f"Callback returned: {type(ret)}")

                    if ret is not None:
                        if type(ret) is str:
                            ret = ret.encode("utf-8")
                        elif not isinstance(ret, bytes):
                            # If it's not bytes or string, try to pack it
                            ret = pack_data(ret)

                        self.request.sendall(ret)
                        self._logger.debug(
                            f"Sent response of {len(ret)} bytes"
                        )
                    else:
                        self._logger.debug(
                            "Callback returned None, no response sent"
                        )

                except Exception as e:
                    errstr = str(e) + "\n" + traceback.format_exc()
                    self._logger.error(f"Callback execution error: {errstr}")
            else:
                self._logger.error(
                    "No callback function set or callback is None"
                )

            self._logger.debug(f"Close connection from {client_addr}:{port}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def shutdown(self):
        if hasattr(self, "_server"):
            self._server.shutdown()
            self._server.server_close()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        self.shutdown()
