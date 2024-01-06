#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides all the utilities for Remote Procedure Call (RPC) via network socket.
RPCSocketServer: RPC sockect server class
pack_data: utility function to pack data.
send_data: utility function to send data.
recv_resp: utility function to receive data.

mmisaki@libr.net
"""

# %% import ===================================================================
import logging
import sys
import traceback
from threading import Thread
import socketserver
import socket
import pickle
import zlib
import time
import re


#  %% =========================================================================
def pack_data(data, compress=False):
    """
    Pack data to be sent. PKL_{bytesize}_' header is added
    followed by the pickled byte string data.

    Parameters
    ----------
    data : any type
        Any picklable data.
    compress : bool (optional)
        Falg to compress the data with zlib. The default is False.

    Returns
    -------
    pkl_data : byte string
        pickled byte string data.
    """
    try:
        pkl_data = pickle.dumps(data)
        if compress:
            pkl_data = zlib.compress(pkl_data)
            pkl_data = f"ZPKL_{len(pkl_data)}_".encode('utf-8') + pkl_data
        else:
            pkl_data = f"PKL_{len(pkl_data)}_".encode('utf-8') + pkl_data
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        errstr = ''.join(
            traceback.format_exception(exc_type, exc_obj, exc_tb))
        sys.stderr.write(str(e) + '\n' + errstr)
        pkl_data = None

    return pkl_data


#  %% =========================================================================
def rpc_send_data(sock, data, pkl=False, logger=None):
    """
    Send data via a sock connected to the RPCSocketServer.
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
        data = data.encode('utf-8')

    try:
        sock.sendall(data)
        return True

    except BrokenPipeError:
        if logger:
            logger.error('No connection')
        else:
            sys.stderr.write('No connection')
        return False


# %% ==========================================================================
def rpc_recv_data(sock, logger=None):
    """
    Receive data via a sock connected to the RPCSocketServer.
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
                    logger.error('Connction closed')
                break
            except BlockingIOError:
                # No more data available at the moment
                break
            except Exception:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                errstr = ''.join(
                    traceback.format_exception(
                        exc_type, exc_obj, exc_tb))
                if logger:
                    logger.error(f'Data receiving error: {errstr}')
                break
            time.sleep(0.001)

        # Process the received data
        data = None
        if len(serialized_data):
            if serialized_data.startswith('PKL_'.encode('utf-8')) or \
                    serialized_data.startswith('ZPKL_'.encode('utf-8')):
                # Data is pickle
                # Get data size
                dstr = serialized_data.decode(
                    'utf-8', 'backslashreplace')
                ma = re.search(r'Z*PKL_(\d+)_', dstr)
                dsize = int(ma.groups()[0])
                prefix = ma.group()
                # Check if data is complete
                pkldata = serialized_data.replace(
                    prefix.encode('utf-8'), ''.encode('utf-8'))
                if len(pkldata) < dsize:
                    # Data is not complete.
                    # Keep the current data in serialized_data and
                    # continue receiving the remaining data.
                    continue

                # Extract pkldata part of the byte string
                pkldata = pkldata[:dsize]
                try:
                    # Unpickling the data
                    if prefix.startswith('Z'):
                        pkldata = zlib.decompress(pkldata)
                    data = pickle.loads(pkldata)
                except Exception:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    errstr = ''.join(
                        traceback.format_exception(
                            exc_type, exc_obj, exc_tb))
                    if logger:
                        logger.error(errstr)
            else:
                # Assume simple string
                data = serialized_data.decode()

        break

    return data


# %% class RPCSocketServer ====================================================
class RPCSocketServer:
    """
    RPC sockect server class
    Start a socket server in __init__().
    Receive a call and pass it to the RPC_handler function.
    When the handler function returns a value, pass it back to the client.
    """
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, port, RPC_handler=print, socket_name='RPCSocketServer'):
        self._socket_name = socket_name
        self._logger = logging.getLogger(self._socket_name)

        # Boot server
        socketserver.TCPServer.allow_reuse_address = True
        try:
            self._server = socketserver.TCPServer(
                ('0.0.0.0', port), RPCSocketServer._recvDataHandler)
        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errstr = ''.join(
                traceback.format_exception(exc_type, exc_obj, exc_tb))
            self._logger.error(errstr)
            self.server = None
            return

        self._server._callback = RPC_handler
        self._server._cancel = False
        self._server.socket_name = self._socket_name

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
            self._logger.info(f"Connected from {client_addr}:{port}")
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
                            f'Connction from {client_addr} closed')
                        break
                    except BlockingIOError:
                        # No more data available at the moment
                        break
                    except Exception:
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        errstr = ''.join(
                            traceback.format_exception(
                                exc_type, exc_obj, exc_tb))
                        self._logger.error(f'Data receiving error: {errstr}')
                        break
                    time.sleep(0.001)

                # Process the received data
                data = None
                if len(serialized_data):
                    if serialized_data.startswith('PKL_'.encode('utf-8')) or \
                            serialized_data.startswith(
                                'ZPKL_'.encode('utf-8')):
                        # Data is pickle
                        # Get data size
                        dstr = serialized_data.decode(
                            'utf-8', 'backslashreplace')
                        ma = re.search(r'Z*PKL_(\d+)_', dstr)
                        dsize = int(ma.groups()[0])
                        prefix = ma.group()
                        # Check if data is complete
                        pkldata = serialized_data.replace(
                            prefix.encode('utf-8'), ''.encode('utf-8'))
                        if len(pkldata) < dsize:
                            # Data is not complete.
                            # Keep the current data in serialized_data and
                            # continue receiving the remaining data.
                            continue

                        # Extract pkldata part of the byte string
                        pkldata = pkldata[:dsize]
                        try:
                            # Unpickling the data
                            if prefix.startswith('Z'):
                                pkldata = zlib.decompress(pkldata)
                            data = pickle.loads(pkldata)
                            self._logger.info(
                                f'Received {dsize} byte data' +
                                f' by {self.server.server_address}')
                        except Exception:
                            exc_type, exc_obj, exc_tb = sys.exc_info()
                            errstr = ''.join(
                                traceback.format_exception(
                                    exc_type, exc_obj, exc_tb))
                            self._logger.error(errstr)
                    else:
                        # Assume simple string
                        data = serialized_data.decode()

                break

            if data:
                ret = self.server._callback(data)
                if ret is not None:
                    if type(ret) is str:
                        ret = ret.encode('utf-8')

                    self.request.sendall(ret)

            self._logger.debug(f"Close connection from {client_addr}:{port}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def shutdown(self):
        self._server.shutdown()
        self._server.server_close()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        self.shutdown()
