#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive test cells for the LA-NF feedback application, NF_psypy.py (run
cell by cell in an IDE in this directory). The cells send the messages that
LANF (la_nf.py) sends via RtpApp.send_extApp.

@author: mmisaki@laureateinstitute.org
"""


# %% import
import sys
import socket
from datetime import datetime
import numpy as np
from scipy.stats import norm
import time
from rtp_serve import boot_RTP_SERVE_app, pack_data


# %% Boot
def boot_connect_app():
    cmd = './NF_psypy.py --pos 0 0 --size 800 600 --log_dir ./'
    addr, proc = boot_RTP_SERVE_app(cmd)
    if addr is None:
        sys.stderr.write(proc+'\n')
        return None, None, None

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(addr)
    sock.settimeout(3)

    return sock, addr, proc


# %% isAlive_extApp
def isAlive_extApp(sock):
    sock.settimeout(3)
    sock.sendall('IsAlive?;'.encode('utf-8'))
    try:
        sock.recv(1024)
        return True
    except socket.timeout:
        return False


# %% Wait for the end of the session
def wait_end_session(sock):
    sock.settimeout(None)
    print(time.ctime(), sock.recv(1024).decode())
    sock.settimeout(3)
    time.sleep(3)


# %% Prepare a session and start the scan
def prep_and_start(sock, prep_cmd, task_param):
    # Prepare the session: the command followed by the pickled parameters
    sock.sendall(prep_cmd.encode('utf-8'))
    sock.sendall(pack_data(task_param))
    print(time.ctime(), sock.recv(1024).decode())  # 'PREPED;'

    # Send 'READY' and wait for 'READY'
    sock.sendall('READY;'.encode('utf-8'))
    print(time.ctime(), sock.recv(1024).decode())
    time.sleep(3)

    # Send the scan start
    scan_onset = time.time()
    onset_str = datetime.fromtimestamp(scan_onset).isoformat()
    sock.sendall(f'SCAN_START {onset_str};'.encode('utf-8'))

    return scan_onset


# %% boot app and open socket
sock, serv_addr, proc = boot_connect_app()


# %% Rest session
run = True

if run:
    if not isAlive_extApp(sock):
        sock.close()
        sock, serv_addr, proc = boot_connect_app()

    task_param = {'session': 'Rest', 'total_duration': 480}
    prep_and_start(sock, 'PREP_Rest;', task_param)
    wait_end_session(sock)


# %% Baseline
run = True

if run:
    if not isAlive_extApp(sock):
        sock.close()
        sock, serv_addr, proc = boot_connect_app()

    task_param = {'session': 'Baseline', 'TR': 2,
                  'timings': [96, 40, 40, 40, 4], 'nofb': True}
    prep_and_start(sock, 'PREP_Task;', task_param)
    wait_end_session(sock)


# %% NF
N_Run = 3

for nr in range(N_Run):
    if not isAlive_extApp(sock):
        sock.close()
        sock, serv_addr, proc = boot_connect_app()

    task_param = {'session': f'NF{nr+1}', 'TR': 2,
                  'timings': [96, 40, 40, 40, 4], 'nofb': False,
                  'target_level': 1}
    scan_onset = prep_and_start(sock, 'PREP_Task;', task_param)

    # simulate RTP
    TR = task_param['TR']
    timings = task_param['timings']
    tlen = (timings[0] + (timings[1] + timings[2] + timings[3]) * timings[4])
    NrVol = tlen / TR

    send_t = np.arange(NrVol) * TR + TR + 0.5  # Delay 1 TR + 0.5s (RTP time)
    for vi in range(int(NrVol)):
        if vi < 3:
            continue

        # Wait for send time
        while time.time() - scan_onset < send_t[vi]:
            time.sleep(0.01)

        # Send a volume signal: "NF {time},{volume_index},{signal_value};"
        val = norm.rvs()
        sock.sendall(
            f'NF {time.time() - scan_onset:.4f},{vi},{val:.6f};'.encode(
                'utf-8'))

    wait_end_session(sock)


# %% Transfer
run = True

if run:
    if not isAlive_extApp(sock):
        sock.close()
        sock, serv_addr, proc = boot_connect_app()

    task_param = {'session': 'Transfer', 'TR': 2,
                  'timings': [96, 40, 40, 40, 4], 'nofb': True}
    prep_and_start(sock, 'PREP_Task;', task_param)
    wait_end_session(sock)

sock.sendall('QUIT;'.encode('utf-8'))
proc.wait(timeout=10)
