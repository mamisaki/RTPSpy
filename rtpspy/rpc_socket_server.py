#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides utilities for Remote Procedure Call (RPC) via network socket.
RPCSocketServer: RPC socket server class
pack_data: Utility function to pack data.
send_data: Utility function to send data.
recv_resp: Utility function to receive data.

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


# %% RPCSocketCom =============================================================
class RPCSocketCom:

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, address_name, config_path):
        self.address_name = address_name
        self.config_path = config_path
        self._socket_name = address_name[2]
        self._logger = logging.getLogger(self._socket_name)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_port(self):
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
            (port, None) if successful, or (None, error_message) if an error
            occurs.
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
                        f"sshpass -p '{password}' ssh {host}"
                        "'cat .RTPSpy/rtpspy'"
                    )
                else:
                    cmd = f"ssh {host} 'cat .RTPSpy/rtpspy'"

                output = subprocess.check_output(cmd, shell=True).decode()
                rtpspy_config = json.loads(output)
                port = int(rtpspy_config.get(f"{socket_name}_port", None))

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
        Response from the remote procedure, if applicable.
        """
        remote_proc_address = tuple(self.address_name[:2])
        port = remote_proc_address[1]
        if port is None:
            port = self.get_port()
            if port is None:
                return False, None
            else:
                remote_proc_address = (remote_proc_address[0], port)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.settimeout(timeout)
            sock.connect(remote_proc_address)
        except ConnectionRefusedError:
            time.sleep(1)
            if data == 'ping':
                sock.close()
                return False, port
            sock.close()
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
            # Port might not be valid. Retry
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
        data : bytes
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
            sock.sendall(data)
            return True

        except BrokenPipeError:
            self._logger.error("No connection")
            return False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def rpc_recv_data(self, sock):
        """
        Receive data via a socket connected to the RPCSocketServer.

        Returns
        -------
        Received data, if applicable.
        """
        # Receive data loop: Run until data is complete.
        sock.settimeout(None)
        serialized_data = bytearray()
        while True:
            # Append data in serialized_data until socket.timeout
            while True:  # Loop until socket.timeout to collect all data
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
                    self._logger.error("Connection closed")
                    break
                except BlockingIOError:
                    # No more data available at the moment
                    time.sleep(0.001)
                    continue
                except Exception as e:
                    errstr = str(e) + "\n" + traceback.format_exc()
                    self._logger.error(f"Data receiving error: {errstr}")
                    break
                time.sleep(0.001)

            # Process the received data
            data = None
            if len(serialized_data):
                if serialized_data.startswith(
                    "PKL_".encode("utf-8")
                ) or serialized_data.startswith("ZPKL_".encode("utf-8")):
                    # Data is pickled
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
                    except Exception as e:
                        errstr = str(e) + "\n" + traceback.format_exc()
                        self._logger.error(errstr)
                else:
                    # Assume simple string
                    data = serialized_data.decode()

            break

        return data


# %% class RPCSocketServer ====================================================
class RPCSocketServer:
    """
    RPC socket server class.
    Starts a socket server in __init__().
    Receives a call and passes it to the RPC_handler function.
    When the handler function returns a value, passes it back to the client.
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
            self.server = None
            return

        self.port = self._server.server_address[1]
        self._logger.debug(f"Server started on port {self.port}")
        self._server._callback = RPC_handler
        self._server._callback_kwargs = handler_kwargs
        self._server._cancel = False
        self._server.socket_name = self._socket_name

        # Save the port number in the RT-MRI config file
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

        # Start the server on another thread.
        self._server_thread = Thread(
            target=self._server.serve_forever, args=(0.005,))
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
            self.request.settimeout(1.5)

            # --- Request handling loop ---
            # Receive data loop: Run until data is complete.
            serialized_data = bytearray()
            while True:
                # Append data in serialized_data until socket.timeout
                while True:  # Loop until socket.timeout to collect all data
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
                            f"Connection from {client_addr} closed")
                        break
                    except BlockingIOError:
                        # No more data available at the moment
                        break
                    except Exception as e:
                        errstr = str(e) + "\n" + traceback.format_exc()
                        self._logger.error(f'Data receiving error: {errstr}')
                        break
                    time.sleep(0.001)

                # Process the received data
                data = None
                if len(serialized_data):
                    if serialized_data.startswith(
                        "PKL_".encode("utf-8")
                    ) or serialized_data.startswith("ZPKL_".encode("utf-8")):
                        # Data is pickled
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
                        except Exception as e:
                            errstr = str(e) + "\n" + traceback.format_exc()
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
                    ret = self.server._callback(
                        data, **self.server._callback_kwargs
                    )
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

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def shutdown(self):
        """
        Shut down the server and close the connection.
        """
        if hasattr(self, "_server"):
            self._server.shutdown()
            self._server.server_close()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        """
        Clean up resources and remove the port number from the config file.
        """
        config_f = Path.home() / ".RTPSpy" / "rtmri_config.json"
        if config_f.is_file():
            try:
                with open(config_f, "r") as fid:
                    rtpspy_config = json.load(fid)
                rtpspy_config.pop(f"{self._server.socket_name}_port", None)
                with open(config_f, "w") as fid:
                    json.dump(rtpspy_config, fid)
            except Exception:
                pass

        self.shutdown()
