#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RPC socket server
mmisaki@laureateinstitute.org
"""

# %% import ===================================================================
import logging
import sys
import traceback
from threading import Thread
import socketserver
import socket


# %% class RPCSocketServer ====================================================
class RPCSocketServer:
    """
    RPC sockect server class
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
            call = None
            while not self.server._cancel:
                try:
                    call = self.request.recv(1024)
                    if len(call) == 0:
                        break
                except socket.timeout:
                    continue
                except BlockingIOError:
                    # No more data available at the moment
                    continue
                except BrokenPipeError:
                    break
                except Exception:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    errstr = ''.join(
                        traceback.format_exception(exc_type, exc_obj, exc_tb))
                    self._logger.error(f'Data receiving error: {errstr}')
                    break

                if call is not None:
                    call = call.decode()
                    if call == 'END_SERVER':
                        break

                    self._logger.info(f"Received {call}")
                    ret = self.server._callback(call)
                    if ret is not None:
                        self.request.sendall(ret)

                    call = None

            self._logger.debug(f"Close connection from {client_addr}:{port}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        self._server.shutdown()
        self._server.server_close()
