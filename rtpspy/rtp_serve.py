# -*- coding: utf-8 -*-
"""
RTP_SERVE: Server class for the communication with RTPSpy application

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

import pandas as pd


# %%　RTP Message Handler =====================================================
class RTPMsgHandler(socketserver.StreamRequestHandler):
    """
    RTP message handling class
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
        self.request.settimeout(0.001)
        partial_recv_data = b''

        # Keep running until a client closes the connection.
        connected = True
        while not self.server.kill and connected:
            # --- Receiving data ---
            recv_data = partial_recv_data
            recvs = []
            while True:  # loop until socket.timeout to collect all data
                try:
                    recv_data += self.request.recv(1024)
                except socket.timeout:
                    break

                # To concatenate messages with short interval
                time.sleep(0.001)

            if len(recv_data):
                if len(self.server.data_sep) > 0:
                    # split the messages by self.server.data_sep
                    sep = self.server.data_sep.encode('utf-8')
                    recvs.extend(recv_data.split(sep))
                else:
                    recvs.append(recv_data)

            if self.server.verb:
                if len(recvs):
                    self._log(recvs, prefix='Recv:')

            # Process the received data list
            NF_data = []
            while len(recvs):
                data = recvs.pop(0)
                if len(data) == 0:
                    continue

                if data.startswith('PKL_'.encode('utf-8')):
                    # Data is pickle
                    # Get data size
                    dstr = data.decode('utf-8', 'backslashreplace')
                    ma = re.search(r'PKL_(\d+)_', dstr)
                    dsize = int(ma.groups()[0])
                    # Check if data is complete
                    pkldata = data.replace(ma.group().encode('utf-8'),
                                           ''.encode('utf-8'))

                    if len(pkldata) < dsize:
                        # Not complete data.
                        # Save the current data in partial_recv_data
                        # and break the received data process loop.
                        partial_recv_data = data
                        break
                    else:
                        partial_recv_data = b''
                        try:
                            data = pickle.loads(pkldata)
                        except Exception:
                            exc_type, exc_obj, exc_tb = sys.exc_info()
                            traceback.print_exception(exc_type,
                                                      exc_obj,
                                                      exc_tb)
                else:
                    data = data.decode('utf-8', 'backslashreplace')
                    if data.startswith('NF '):
                        vals = [float(vv)
                                for vv in data.replace('NF ', '').split(',')]
                        add_row = {'Time': vals[0], 'TR': int(vals[1]),
                                   'Signal': vals[2:]}

                        NF_data.append(add_row)
                        continue

                self.proc_recv_data(data)

            # Process NF data
            if len(NF_data):
                self.server.NF_signal_lock.acquire()
                self.server.NF_signals.extend(NF_data)
                self.server.NF_signal_lock.release()

            # --- Sending data ------------------------------------------------
            if not self.server.send_queue.empty():
                send_data = self.server.send_queue.get()

                try:
                    if self.server.verb:
                        self._log(send_data, prefix='Send:')
                    self.request.send(send_data)

                except socket.timeout:
                    if self.server.verb:
                        self._log(send_data, prefix='Failed_Send:')

        # --- End request handling loop ---------------------------------------
        self.server.client_address_str = ''
        self.server.connected = False
        time.sleep(1)  # Wait before closing a connection

        if self.server.verb:
            self._log(addr_str.encode('utf-8'), prefix='Close:')
            self._log(
                f"client_address_str={self.server.client_address_str}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def proc_recv_data(self, data):
        sys.stdout.flush()
        if type(data) is str:
            if 'IsAlive?' in data:
                # Return 'Yes. to 'IsAlive?' inquery.
                self.request.send('Yes.'.encode('utf-8'))
                if self.server.verb:
                    self._log('Yes.', prefix='Send:')
                return

        self.server.recv_queue.put(data)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _log(self, datas, prefix=''):
        # Log data
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


# %% RTP_SERVE ==============================================================
class RTP_SERVE():
    """
    Application server class to communicate with RTP_APP
    The class offers communication methods for an external application with
    RTP_APP via TCP socket.
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
            Host to receive the sever address:port. The default is None.
        handler_class : RTPMsgHandler classs, optional
            Request handler class. The default is RTPMsgHandler.
        data_sep : str, optional
            Delimiter to separate receved data. When multiple data are
            received at once, they were divided with this delimiter.
            The default is ';'.
        verb : bool, optional
            Flag to print log. The default is False.

        """
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
        self.server.verb = verb
        self.server.kill = False

        self._NF_signal = pd.DataFrame(columns=('Time', 'TR', 'Signal'))

        # Start the server on other thread.
        self.server_thread = threading.Thread(
            target=self.server.serve_forever,  args=(0.1,))
        # Make the server thread exit when the main thread terminates
        self.server_thread.daemon = True
        self.server_thread.start()

        if request_host is not None:
            # Return opened address and port to the request host
            req_host, port = request_host.split(':')
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(5)
            host_port = self.server.server_address[1]
            sock.sendto(f"{host_addr}:{host_port}".encode('utf-8'),
                        (req_host, int(port)))
            sock.close()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Make self.server properties accesible as the property of this class
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
        Send data to RTP_APP
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
        The data sent by RTP_APP is put in recv_queue by a handler with
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
        Wait for all data in send_queue being sent and the queue is empty.
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
        """ Get IP address of the PC """
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __del__(self):
        self.server.kill = True
        self.server.shutdown()


# %% boot_RTP_SERVE_app =====================================================
def boot_RTP_SERVE_app(cmd, remote=False, timeout=5, verb=False):
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
    verb : bool, optional
        Print logs. The default is False.

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
    time.sleep(3)  # Wait for opening the process

    if pr.poll() is not None:
        errmsg = f"Failed: {cmd}"
        errmsg += pr.stdout.read().decode()
        errmsg += pr.stderr.read().decode()
        if verb:
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
            errmsg = "No response to a request."
            if verb:
                sys.stderr.write(f"{errmg}\n")
            sock.close()
            return None, errmsg

        return (host, port), pr


# %% pack_data
def pack_data(data):
    """
    Pack data to send RTP_SERVE server. The data is pickled to byte string.

    Parameters
    ----------
    data : any type
        Any picklable data.

    Returns
    -------
    pkl_data : byte string
        pickled byte string data.
    """

    try:
        pkl_data = pickle.dumps(data)
        pkl_data = f"PKL_{len(pkl_data)}_".encode('utf-8') + pkl_data
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        errmsg = '{}, {}:{}\n'.format(
                exc_type, exc_tb.tb_frame.f_code.co_filename,
                exc_tb.tb_lineno)
        sys.stderr.write(str(e) + '\n' + errmsg,)
        pkl_data = None

    return pkl_data


# %% __main__ (test) ==========================================================
if __name__ == '__main__':

    # --- Initialize ----------------------------------------------------------
    rtp_srv = RTP_SERVE()
    rtp_srv.verb = True
    srv_address = rtp_srv.server.server_address
    print("Open server at ", srv_address)

    # Open client socket (e.g., from RTP_APP) and connect to rtp_srv
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(srv_address)
    time.sleep(0.001)

    # --- Connection test -----------------------------------------------------
    # 1. rtp_srv should respond to 'IsAlive?;'.
    #    ';' is a separater of the command.
    print("\n1. Send 'IsAlive?;'")
    sock.sendall('IsAlive?;'.encode('utf-8'))
    resp = sock.recv(1024).decode('utf-8', 'backslashreplace')
    print(f"Recevied '{resp}'")

    # 2. Multiple interactions in one session
    print("\n2. Multiple interactions")
    for ii in range(10):
        data = f"data 1-{ii};"
        print(f"Send '{data}'")
        sock.sendall(data.encode('utf-8'))
        time.sleep(0.001)

    data = rtp_srv.get_recv_queue()
    while data:
        print(f"Server recv_queue '{data}'")
        data = rtp_srv.get_recv_queue()

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
            ','.join(str(float(v)) for v in range(ii+1, ii+4))
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

    data = rtp_srv.get_recv_queue()
    while data:
        print(f"Server recv_queue '{data}'")
        data = rtp_srv.get_recv_queue()

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
