# -*- coding: utf-8 -*-
"""
RTP_SERVE: Server for an external application to communicate with RtpApp

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
import socket
import threading
import socketserver
from datetime import datetime
import queue
import subprocess
import re
import pickle
import sys
import time
import traceback
import shlex
import zlib
import tempfile
from pathlib import Path

import pandas as pd


# Header of a pickled message made by pack_data: (Z)PKL_{size}_
_PKL_HEADER = re.compile(rb'(Z?)PKL_(\d+)_')
# A possibly incomplete header at the end of the received bytes
_PKL_HEADER_PART = re.compile(rb'Z?(P(K(L(_\d*)?)?)?)?')


# %% _get_ip_address =========================================================
def _get_ip_address():
    """Return the IP address of this PC on the network.

    Connecting a UDP socket sends no packet; it only selects the network
    interface used to reach an external address.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


# %% RTPMsgHandler ============================================================
class RTPMsgHandler(socketserver.StreamRequestHandler):
    """Request handler that receives messages from and sends messages to
    RtpApp.
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def handle(self):
        """
        Keep running until a client closes the connection.
        """

        # --- Initialize ------------------------------------------------------
        addr_str = self.client_address[0]
        if self.server.verb:
            self._log(addr_str.encode('utf-8'), prefix='Open:')

        self.server.connected = True
        self.server.client_address_str = addr_str
        self.server.send_queue.queue.clear()
        self.server.recv_queue.queue.clear()

        # --- Request handling loop -------------------------------------------
        recv_buf = b''  # Received bytes not yet processed
        idle_timeout = 0.05  # Wait for a message (s); also the send interval
        burst_timeout = 0.005  # Wait for the following messages in a burst

        connected = True
        while not self.server.kill and connected:
            # --- Receiving data ----------------------------------------------
            self.request.settimeout(idle_timeout)
            while True:  # Receive until no more data arrives (timeout)
                # Processing each message as it arrives at very short
                # intervals (e.g., NF signals sent during retrospective
                # regression) can crash the server while the list shared
                # between threads (self.server.NF_signals) is reallocated.
                # To avoid this, messages arriving at short intervals are
                # processed together as one chunk.
                try:
                    recv_data = self.request.recv(102400)
                    if not recv_data:
                        # Empty data means the connection was closed.
                        connected = False
                        break
                    recv_buf += recv_data

                    # Wait briefly for the following messages to concatenate
                    self.request.settimeout(burst_timeout)

                except socket.timeout:
                    break

            recvs, recv_buf = self._split_messages(recv_buf)

            if self.server.verb and len(recvs):
                self._log(recvs, prefix='Recv:')

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
                        self._err(
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
                            self._err(
                                f"Skip a malformed NF message: '{data}'")
                            continue

                        NF_data.append(add_row)
                        continue

                self.proc_recv_data(data)

            if len(NF_data):
                with self.server.NF_signal_lock:
                    self.server.NF_signals.extend(NF_data)

            # --- Sending data ------------------------------------------------
            if not self.server.send_queue.empty():
                send_data = self.server.send_queue.get()

                try:
                    if self.server.verb:
                        self._log(send_data, prefix='Send:')
                    self.request.settimeout(1)
                    self.request.sendall(send_data)

                except socket.timeout:
                    if self.server.verb:
                        self._log(send_data, prefix='Failed_Send:')

        # --- End request handling loop ---------------------------------------
        self.server.client_address_str = ''
        self.server.connected = False
        time.sleep(1)  # Wait before closing the connection

        if self.server.verb:
            self._log(addr_str.encode('utf-8'), prefix='Close:')
            self._log(
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
        """Answer an 'IsAlive?' inquiry, or put data in recv_queue."""
        sys.stdout.flush()
        if type(data) is str and 'IsAlive?' in data:
            self.request.send('Yes.'.encode('utf-8'))
            if self.server.verb:
                self._log('Yes.', prefix='Send:')
            return

        self.server.recv_queue.put(data)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _err(self, errmsg):
        """Report an error regardless of the verb flag."""
        sys.stderr.write(f"RTP_SERVE: {errmsg}\n")
        if self.server.verb:
            self._log(errmsg, prefix='Error:')

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _log(self, datas, prefix=''):
        """Print the data and append it to RTP_SERVE_debug.log."""
        if type(datas) is not list:
            datas = [datas]

        logstrs = []
        for data in datas:
            if type(data) is str:
                logstrs.append(data)
            else:
                try:
                    logstrs.append(data.decode('utf-8', 'strict'))
                except Exception:
                    logstrs.append('encoded binary data')

        logstr = prefix + ';'.join(logstrs)
        dt = datetime.isoformat(datetime.now())
        print(f"RTP_SERVE,{dt},{logstr}")
        with open('RTP_SERVE_debug.log', 'a') as fd:
            print(f"RTP_SERVE,{dt},{logstr}", file=fd)


# %% RTP_SERVE ================================================================
class RTP_SERVE():
    """
    Server for an external application (e.g., a neurofeedback presentation)
    to communicate with RtpApp via a TCP socket.
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, allow_remote_access=False, request_host=None,
                 handler_class=RTPMsgHandler, data_sep=';', verb=False):
        """
        Parameters
        ----------
        allow_remote_access : bool, optional
            Allow remote access (connection from other than localhost).
            The default is False.
        request_host : str, optional
            'host:port' to which the server address:port is sent (UDP).
            The default is None.
        handler_class : RTPMsgHandler class, optional
            Request handler class. The default is RTPMsgHandler.
        data_sep : str, optional
            Delimiter between messages. When multiple messages are
            received at once, they are split by this delimiter.
            The default is ';'.
        verb : bool, optional
            Flag to print logs. The default is False.

        """
        if allow_remote_access:
            host = '0.0.0.0'
            host_addr = _get_ip_address()
        else:
            host = 'localhost'
            host_addr = 'localhost'

        # Start the server
        socketserver.TCPServer.allow_reuse_address = True
        try:
            self.server = socketserver.TCPServer((host, 0), handler_class)
        except Exception as e:
            errstr = str(e) + "\n" + traceback.format_exc()
            sys.stderr.write(f"Error starting RTP server: {errstr}\n")
            self.server = None
            return

        # Set attributes on self.server so that the handler can access them.
        self.server.data_sep = data_sep
        self.server.parent = self
        self.server.connected = False
        self.server.client_address_str = ''
        self.server.send_queue = queue.Queue()
        self.server.recv_queue = queue.Queue()
        self.server.NF_signals = []
        self.server.NF_signal_lock = threading.Lock()
        self.server.verb = verb
        self.server.kill = False

        self._NF_signal = pd.DataFrame(columns=('Time', 'TR', 'Signal'))

        # Run the server on a daemon thread so that it exits with the main
        # thread.
        self.server_thread = threading.Thread(
            target=self.server.serve_forever, args=(0.1,))
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
        """Received NF signals as a DataFrame (columns: Time, TR, Signal).
        The DataFrame is rebuilt only when new signals have arrived.
        """
        with self.server.NF_signal_lock:
            if len(self.server.NF_signals) != len(self._NF_signal):
                self._NF_signal = pd.DataFrame(self.server.NF_signals)

        return self._NF_signal

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def send(self, data):
        """
        Send data to RtpApp.
        The data is put in send_queue and sent by the handler.

        Parameters
        ----------
        data : bytes
            Data to send.

        """
        self.server.send_queue.put(data)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_recv_queue(self, timeout=0):
        """
        Get data from recv_queue.
        The handler puts the data sent by RtpApp in recv_queue, decoded to
        str or unpickled.

        Parameters
        ----------
        timeout : float, optional
            Maximum time (s) to wait for data. The default is 0.

        Returns
        -------
        data : str, unpickled object, or None
            Received data. None if no data has arrived.

        """
        if timeout > 0:
            st = time.time()
            while time.time() - st < timeout:
                if not self.server.recv_queue.empty():
                    break
                time.sleep(0.001)  # Do not busy-wait

        if self.server.recv_queue.empty():
            return None
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
        """Clear recv_queue and the received NF signals."""
        self.flush_recv()
        with self.server.NF_signal_lock:
            self.server.NF_signals = []

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        self.server.kill = True
        self.server.shutdown()


# %% boot_RTP_SERVE_app =======================================================
def _tail(fname, nbytes=2000):
    """Last nbytes of a text file ('' if it cannot be read)."""
    try:
        with open(fname, 'rb') as fd:
            fd.seek(0, 2)
            fd.seek(max(fd.tell() - nbytes, 0))
            return fd.read().decode('utf-8', 'replace')
    except Exception:
        return ''


def boot_RTP_SERVE_app(cmd, remote=False, timeout=5, verb=False,
                       out_file=None):
    """
    Boot an external application with RTP_SERVE.

    The standard output and error of the application are appended to
    out_file. They are not connected to pipes: a pipe that nobody reads
    fills up (64 KiB on Linux) and then blocks the application at its next
    print or warning.

    Parameters
    ----------
    cmd : str
        Application boot command line.
    remote : bool, optional
        Whether the connection is remote or local. The default is False
        (local).
    timeout : float or None, optional
        Time (s) to wait, in addition to 3 s for the start-up, for the
        application to send back its RTP_SERVE address:port. None waits
        until it is sent or the application ends. An application that does
        not answer in time is terminated. The default is 5.
    verb : bool, optional
        Print logs. The default is False.
    out_file : str or Path, optional
        File to which the standard output and error of the application are
        appended. Its directory is made if needed. The default is
        rtp_serve_app_<program>_<date>_<time>.log in the system temporary
        directory.

    Returns
    -------
    (host, port), pr : (str, int), subprocess.Popen object
        Host address and port of the RTP_SERVE server and the process running
        the external application via 'cmd'. pr.out_file is the output file.
        On failure, None and an error message (with the end of the
        application output) are returned.

    """
    if out_file is None:
        prog = Path(shlex.split(cmd)[0]).stem if len(cmd.split()) else 'app'
        out_file = Path(tempfile.gettempdir()) / (
            f"rtp_serve_app_{prog}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 0))
    host, port = sock.getsockname()

    if remote:
        host_addr = _get_ip_address()
        cmd += " --allow_remote_access"
    else:
        host_addr = 'localhost'

    cmd += f" --request_host {host_addr}:{port}"

    def failed(errmsg):
        out = _tail(out_file).strip()
        errmsg += f"\nOutput of the application ({out_file}):"
        errmsg += f"\n{out}" if out else " (none)"
        if verb:
            sys.stderr.write(f"{errmsg}\n")
        return None, errmsg

    # The child process gets its own handle of the file; the one here is
    # closed after starting it.
    with open(out_file, 'ab') as out_fd:
        out_fd.write(f"=== {datetime.now().isoformat()} {cmd}\n".encode())
        out_fd.flush()
        try:
            pr = subprocess.Popen(shlex.split(cmd), stdout=out_fd,
                                  stderr=subprocess.STDOUT)
        except Exception as e:
            sock.close()
            return failed(f"Failed: {cmd}\n{e}")
    pr.out_file = out_file

    # Wait for the address:port from the application, for 3 s (start-up)
    # plus timeout (None: no limit). Stop waiting if the application ends.
    deadline = None if timeout is None else time.time() + 3 + timeout
    sock.settimeout(0.2)
    try:
        while True:
            try:
                data = sock.recv(1024)
                break
            except socket.timeout:
                pass

            if pr.poll() is not None:
                return failed(f"Failed: {cmd} (exit code {pr.returncode})")

            if deadline is not None and time.time() > deadline:
                # The caller gets no handle of the process; do not leave it
                # running.
                pr.terminate()
                try:
                    pr.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pr.kill()
                return failed("No response to the request.")
    finally:
        sock.close()

    host, port = data.decode('utf-8').split(':')
    port = int(port)

    return (host, port), pr


# %% pack_data ================================================================
def pack_data(data, compress=False):
    """
    Pack data to be sent: a header 'PKL_{bytesize}_' followed by the pickled
    bytes ('ZPKL_{bytesize}_' and zlib-compressed bytes if compress=True).

    Parameters
    ----------
    data : any type
        Any picklable data.
    compress : bool, optional
        Flag to compress the data with zlib. The default is False.

    Returns
    -------
    pkl_data : bytes or None
        Packed data. None if the data cannot be pickled.
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


# %% __main__ (test) ==========================================================
if __name__ == '__main__':

    # --- Initialize ----------------------------------------------------------
    rtp_srv = RTP_SERVE()
    rtp_srv.verb = True
    srv_address = rtp_srv.server.server_address
    print("Open server at ", srv_address)

    # Open client socket (e.g., from RtpApp) and connect to rtp_srv
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(srv_address)
    time.sleep(0.001)

    # --- Connection test -----------------------------------------------------
    # 1. rtp_srv should respond to 'IsAlive?;'.
    #    ';' is a separator for the command.
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

    # Wait for NF_signal to be filled.
    if len(rtp_srv.NF_signal) < 5:
        while len(rtp_srv.NF_signal) < 5:
            time.sleep(0.001)
        print(rtp_srv.NF_signal)

    # 4. Sending '' does not close the connection.
    print("\n4. Check '' does not close the connection.")
    sock.sendall(''.encode('utf-8'))
    time.sleep(0.001)
    assert rtp_srv.connected, 'Connection closed unexpectedly.'
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

    # 6. Receive a message from rtp_srv
    data = 'data from server'
    print(f"\n6. Send message '{data}' from rtp_srv")
    rtp_srv.send(data.encode('utf-8'))
    resp = sock.recv(1024).decode('utf-8', 'backslashreplace')
    print(f"Client received '{resp}'")

    # 7. Close the connection
    print('\n7. Close the connection.')
    sock.close()
    time.sleep(0.001)
    assert not rtp_srv.connected, 'Connection not closed properly.'
    print('OK.')

    del rtp_srv
