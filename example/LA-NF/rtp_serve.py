# -*- coding: utf-8 -*-
"""
RTP_SERVE: Server for an external application to communicate with RtpApp
(a copy of rtpspy/rtp_serve.py so that this example runs without rtpspy)

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
import socket
import threading
import socketserver
import queue
import subprocess
import re
import pickle
import sys
import time
import traceback
import shlex
import zlib
import logging

import pandas as pd


# Header of a pickled message made by pack_data: (Z)PKL_{size}_
_PKL_HEADER = re.compile(rb'(Z?)PKL_(\d+)_')
# A possibly incomplete header at the end of the received bytes
_PKL_HEADER_PART = re.compile(rb'Z?(P(K(L(_\d*)?)?)?)?')


# %% RTPMsgHandler ============================================================
class RTPMsgHandler(socketserver.StreamRequestHandler):
    """
    RTP message handling class
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def handle(self):
        """
        Keep running until a client closes the connection.
        """

        self._logger = logging.getLogger('RTP_SERVE')

        # --- Initialize ------------------------------------------------------
        addr_str = self.client_address[0]
        self._logger.debug(f"Open:{addr_str.encode('utf-8')}")

        self.server.connected = True
        self.server.client_address_str = addr_str
        self.server.send_queue.queue.clear()
        self.server.recv_queue.queue.clear()

        # --- Request handling loop -------------------------------------------
        recv_buf = b''  # Received bytes not yet processed
        self.request.settimeout(0.001)

        connected = True
        while not self.server.kill and connected:
            # --- Receiving data ----------------------------------------------
            while True:  # Receive until no more data arrives (timeout)
                # Processing each message as it arrives at very short
                # intervals (e.g., NF signals sent during retrospective
                # regression) can crash the server while the list shared
                # between threads (self.server.NF_signals) is reallocated.
                # To avoid this, messages arriving at short intervals are
                # processed together as one chunk.
                try:
                    recv_data = self.request.recv(4096)
                    if not recv_data:
                        # Empty data means the connection was closed.
                        connected = False
                        break
                    recv_buf += recv_data

                    time.sleep(0.001)  # Wait for the following messages

                except socket.timeout:
                    break

            recvs, recv_buf = self._split_messages(recv_buf)

            if len(recvs):
                self._logger.debug(f'Recv:{recvs}')

            # Process the received messages
            NF_data = []
            for data in recvs:
                if len(data) == 0:
                    continue

                ma = _PKL_HEADER.match(data)
                if ma:
                    # Pickled data: '(Z)PKL_{size}_' header followed by the
                    # pickled (and zlib-compressed for ZPKL) bytes
                    try:
                        pkldata = data[ma.end():]
                        if ma.group(1):
                            pkldata = zlib.decompress(pkldata)
                        data = pickle.loads(pkldata)
                    except Exception as e:
                        self._logger.error(
                            f"Failed to unpack the received data: {e}")
                        continue
                else:
                    data = data.decode('utf-8', 'backslashreplace')
                    if data.startswith('NF '):
                        try:
                            vals = [float(vv) for vv in
                                    data.replace('NF ', '').split(',')]
                            add_row = {'Time': vals[0], 'TR': int(vals[1]),
                                       'Signal': vals[2:]}
                        except (ValueError, IndexError):
                            self._logger.error(
                                f"Skip a malformed NF message: '{data}'")
                            continue

                        NF_data.append(add_row)
                        continue

                self.proc_recv_data(data)

            if len(NF_data):
                self.server.NF_signal_lock.acquire()
                self.server.NF_signals.extend(NF_data)
                self.server.NF_signal_lock.release()

            # --- Sending data ------------------------------------------------
            if not self.server.send_queue.empty():
                send_data = self.server.send_queue.get()

                try:
                    self._logger.debug(f'Send:{send_data}')
                    self.request.send(send_data)

                except socket.timeout:
                    self._logger.error(f'Failed_Send:{send_data}')

        # --- End request handling loop ---------------------------------------
        self.server.client_address_str = ''
        self.server.connected = False
        time.sleep(1)  # Wait before closing the connection

        self._logger.debug(f'Close:{addr_str}')
        self._logger.debug(
            f"client_address_str={self.server.client_address_str}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _split_messages(self, buf):
        """
        Split the received bytes into messages.

        A pickled message is framed by its '(Z)PKL_{size}_' header (see
        pack_data), so it may contain the separator. Other messages are
        terminated by self.server.data_sep (if data_sep is '', all the text
        received at once is one message).

        Returns
        -------
        msgs : list of bytes
            Complete messages.
        rest : bytes
            An incomplete message, kept to be completed by the next
            reception.
        """
        sep = self.server.data_sep.encode('utf-8')
        msgs = []
        while len(buf):
            ma = _PKL_HEADER.match(buf)
            if ma:
                msg_end = ma.end() + int(ma.group(2))
                if len(buf) < msg_end:
                    break  # Wait for the rest of the pickled data
                msgs.append(buf[:msg_end])
                buf = buf[msg_end:]
                if len(sep) and buf.startswith(sep):
                    buf = buf[len(sep):]
                continue

            if _PKL_HEADER_PART.fullmatch(buf):
                break  # The header itself is split; wait for the rest

            if len(sep) == 0:
                msg, buf = buf, b''
            elif sep in buf:
                msg, buf = buf.split(sep, 1)
            else:
                break  # Wait for the separator
            msgs.append(msg)

        return msgs, buf

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def proc_recv_data(self, data):
        sys.stdout.flush()
        if type(data) is str:
            if 'IsAlive?' in data:
                # Return 'Yes.' to an 'IsAlive?' inquiry.
                self.request.send('Yes.'.encode('utf-8'))
                self._logger.debug('Send:Yes.')
                return

        self.server.recv_queue.put(data)


# %% RTP_SERVE ================================================================
class RTP_SERVE():
    """
    Server for an external application (e.g., a neurofeedback presentation)
    to communicate with RtpApp via a TCP socket.
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, allow_remote_access=False, request_host=None,
                 handler_class=RTPMsgHandler, data_sep=';'):
        """
        Parameters
        ----------
        allow_remote_access : bool, optional
            Allow remote access (connection from other than localhost).
            The default is False.
        request_host : str, optional
            'host:port' to which the server address:port is sent.
            The default is None.
        handler_class : RTPMsgHandler class, optional
            Request handler class. The default is RTPMsgHandler.
        data_sep : str, optional
            Delimiter between messages. When multiple messages are
            received at once, they are split by this delimiter.
            The default is ';'.
        """
        self._logger = logging.getLogger('RTP_SERVE')

        if allow_remote_access:
            host = '0.0.0.0'
            host_addr = self._get_ip_address()
        else:
            host = 'localhost'
            host_addr = 'localhost'

        # Server
        socketserver.TCPServer.allow_reuse_address = True
        self.server = socketserver.TCPServer((host, 0),  RTPMsgHandler)

        # Set properties on self.server for access from a handler.
        self.server.data_sep = data_sep
        self.server.parent = self
        self.server.connected = False
        self.server.client_address_str = ''
        self.server.send_queue = queue.Queue()
        self.server.recv_queue = queue.Queue()
        self.server.NF_signals = []
        self.server.NF_signal_lock = threading.Lock()
        self.server.kill = False

        self._NF_signal = pd.DataFrame(columns=('Time', 'TR', 'Signal'))

        # Start the server on another thread.
        self.server_thread = threading.Thread(
            target=self.server.serve_forever,  args=(0.1,))
        # Make the server thread exit when the main thread terminates
        self.server_thread.daemon = True
        self.server_thread.start()

        if request_host is not None:
            # Send the opened address and port to the request host
            req_host, port = request_host.split(':')
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(5)
            host_port = self.server.server_address[1]
            sock.sendto(f"{host_addr}:{host_port}".encode('utf-8'),
                        (req_host, int(port)))
            sock.close()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Expose the self.server attributes as properties of this class
    @property
    def connected(self):
        return self.server.connected

    @property
    def client_address_str(self):
        return self.server.client_address_str

    @property
    def NF_signal(self):
        self.server.NF_signal_lock.acquire()
        if len(self.server.NF_signals) != len(self._NF_signal):
            self._NF_signal = pd.DataFrame(self.server.NF_signals)
        self.server.NF_signal_lock.release()

        return self._NF_signal

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def send(self, data):
        """
        Send data to RtpApp
        The data in send_queue will be sent by a handler.

        Parameters
        ----------
        data : byte
            Sending data.

        """

        self.server.send_queue.put(data)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_recv_queue(self, timeout=0):
        """
        Get data from recv_queue.
        The data sent by RtpApp is put in recv_queue by a handler with
        converting a str or any type unpickled.

        Parameters
        ----------
        timeout : float, optional
            Wait time until receiving the data. The default is 0.

        Returns
        -------
        data : str or any type unpickled from byte data
            DESCRIPTION.

        """
        if timeout > 0:
            st = time.time()
            while time.time() - st < timeout:
                if not self.server.recv_queue.empty():
                    break

        if self.server.recv_queue.empty():
            return None
        else:
            return self.server.recv_queue.get()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def flush_recv(self):
        """ Clear recv_queue """
        self.server.recv_queue.queue.clear()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def flush_send(self):
        """
        Wait until all data in send_queue has been sent.
        """
        while not self.server.send_queue.empty():
            time.sleep(0.001)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def reset_NF_signal(self):
        self.flush_recv()
        if self.server.NF_signal_lock.acquire():
            self.server.NF_signals = []
            self.server.NF_signal_lock.release()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _get_ip_address(self):
        """Get the IP address of the PC"""
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        self.server.kill = True
        self.server.shutdown()


# %% boot_RTP_SERVE_app =======================================================
def boot_RTP_SERVE_app(cmd, remote=False, timeout=5):
    """
    Boot an external application with RTP_SERVE

    Parameters
    ----------
    cmd : str
        Application boot command line.
    remote : bool, optional
        Whether the connection is remote or not (local). The default is False
        (local).
    timeout : float, optional
        Time out waiting for the application boot and receving the
        address:port. from an external application with RTP_SERVE.
        The default is 5.

    Returns
    -------
    (host, port), pr : (str, int), subprocess.Popen object
        host address and port of RTP_SERVE server and the process running
        the external application by 'cmd'.

    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 0))
    host, port = sock.getsockname()

    if remote:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            host_addr = s.getsockname()[0]
        cmd += " --allow_remote_access"
    else:
        host_addr = 'localhost'

    cmd += f" --request_host {host_addr}:{port}"
    pr = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    time.sleep(3)  # Wait for the process to start

    if pr.poll() is not None:
        errmsg = f"Failed: {cmd}"
        sys.stderr.write(f"{errmsg}\n")
        return None, errmsg
    else:
        sock.settimeout(timeout)
        try:
            data = sock.recv(1024)
            host, port = data.decode('utf-8').split(':')
            port = int(port)
            sock.close()
        except socket.timeout:
            errmsg = "No response to the request."
            sys.stderr.write(f"{errmsg}\n")
            sock.close()
            return None, errmsg

        return (host, port), pr


# %% pack_data ================================================================
def pack_data(data):
    """
    Pack data to send to the RTP_SERVE server: a header 'PKL_{bytesize}_'
    followed by the pickled bytes.

    Parameters
    ----------
    data : any type
        Any picklable data.

    Returns
    -------
    pkl_data : byte string
        Pickled byte string data.
    """

    try:
        pkl_data = pickle.dumps(data)
        pkl_data = f"PKL_{len(pkl_data)}_".encode('utf-8') + pkl_data
    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        errmsg = ''.join(
            traceback.format_exception(exc_type, exc_obj, exc_tb))
        sys.stderr.write(errmsg)
        pkl_data = None

    return pkl_data


# %% __main__ (test) ==========================================================
if __name__ == '__main__':

    # --- Initialize ----------------------------------------------------------
    rtp_srv = RTP_SERVE()
    srv_address = rtp_srv.server.server_address
    print("Open server at ", srv_address)

    # Open client socket (e.g., from RtpApp) and connect to rtp_srv
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(srv_address)
    time.sleep(0.001)

    # --- Connection test -----------------------------------------------------
    # 1. rtp_srv should respond to 'IsAlive?;'.
    #    ';' is the command separator.
    print("\n1. Send 'IsAlive?;'")
    sock.sendall('IsAlive?;'.encode('utf-8'))
    resp = sock.recv(1024).decode('utf-8', 'backslashreplace')
    print(f"Received '{resp}'")

    # 2. Multiple interactions in one session
    print("\n2. Multiple interactions")
    for ii in range(10):
        data = f"data 1-{ii};"
        print(f"Send '{data}'")
        sock.sendall(data.encode('utf-8'))
        time.sleep(0.001)

    data = rtp_srv.get_recv_queue(timeout=2)  # Wait for the data
    while data:
        print(f"Server recv_queue '{data}'")
        data = rtp_srv.get_recv_queue(timeout=0.5)

    # 3. Send NF signals
    print("\n3. Send NF signals")
    for ii in range(40):
        data = f"NF {ii*2.0},{ii}," + \
            ','.join(str(float(v)) for v in range(ii+1, ii+4)) + ';'
        print(f"Send '{data}'")
        sock.sendall(data.encode('utf-8'))

        # Check rtp_srv NF_signal
        print(rtp_srv.NF_signal)

    print('\nReset NF_signal')
    rtp_srv.reset_NF_signal()
    time.sleep(0.001)

    for ii in range(5):
        data = f"NF {ii*2.0},{ii}," + \
            ','.join(str(float(v)) for v in range(ii+1, ii+4)) + ';'
        print(f"Send '{data}'")
        sock.sendall(data.encode('utf-8'))
        time.sleep(0.001)

        # Check rtp_srv NF_signal
        print(rtp_srv.NF_signal)

    # Wait for NF_signal is filled.
    if len(rtp_srv.NF_signal) < 5:
        while len(rtp_srv.NF_signal) < 5:
            time.sleep(0.001)
        print(rtp_srv.NF_signal)

    # 4. '' does not close the connection.
    print("\n4. Check '' does not close the connection.")
    sock.sendall(''.encode('utf-8'))
    time.sleep(0.001)
    assert rtp_srv.connected, 'xxx Connection closed.'
    print("OK.")

    # 5. Pickled data can be sent
    data = {'test': 1}
    pdata = pack_data(data)
    print(f"\n5. Send pickled data '{data}'")
    sock.sendall(pdata)
    time.sleep(0.001)

    data = rtp_srv.get_recv_queue(timeout=2)  # Wait for the data
    while data:
        print(f"Server recv_queue '{data}'")
        data = rtp_srv.get_recv_queue(timeout=0.5)

    # 6. Receive message from rtp_srv
    data = 'data from server'
    print(f"\n6. Send message '{data}' from rtp_srv")
    rtp_srv.send(data.encode('utf-8'))
    resp = sock.recv(1024).decode('utf-8', 'backslashreplace')
    print(f"Client received '{resp}'")

    # 7. Close connection
    print('\n7. Close connection.')
    sock.close()
    time.sleep(0.001)
    assert not rtp_srv.connected, 'xxx Connection not closed.'
    print('OK.')

    del rtp_srv
