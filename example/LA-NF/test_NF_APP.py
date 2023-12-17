#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mmisaki@laureateinstitute.org
"""


# %% import
import sys
import socket
from datetime import datetime
import numpy as np
from scipy.stats import norm
import time
from rtpspy.rtp_appserv import boot_RtpAppSERV_app, pack_data


# %% Boot
def boot_connect_app():
    cmd = './NF_APP.py --position 1920 0 --size 1280 1024 --log_dir ./'
    addr, proc = boot_RtpAppSERV_app(cmd)
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
    sock.sendall('Is alive?;'.encode('utf-8'))
    try:
        sock.recv(1024)
        return True
    except socket.timeout:
        return False


# %% boot app and open socket
sock, serv_addr, proc = boot_connect_app()


# %% Debug
addr = ('localhost', 34851)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(addr)
sock.settimeout(3)

isAlive_extApp(sock)

# Prepare the session
task_param = {'session': 'Rest1', 'total_duration': 480}

isAlive_extApp(sock)
sock.sendall('PREP REST;'.encode('utf-8'))

isAlive_extApp(sock)
sock.sendall(pack_data(task_param))

isAlive_extApp(sock)
sock.sendall('QUIT;'.encode('utf-8'))
sock.recv(1024).decode()

isAlive_extApp(sock)
sock.sendall('PREP REST;'.encode('utf-8'))

isAlive_extApp(sock)
sock.sendall(pack_data(task_param))


# %% Rest session
run = True

if run:
    if not isAlive_extApp(sock):
        sock.close()
        sock, serv_addr, proc = boot_connect_app()

    # Prepare the session
    task_param = {'session': 'Rest1', 'total_duration': 480}
    sock.sendall('PREP REST;'.encode('utf-8'))
    sock.sendall(pack_data(task_param))
    time.sleep(3)

    # Send 'READY'
    sock.sendall('READY;'.encode('utf-8'))
    # Wati for 'READY'
    print(time.ctime(), sock.recv(1024).decode())
    time.sleep(3)

    # Send 'SCAN_ONSET'
    scan_onset_time = datetime.now().isoformat()
    sock.sendall(f'SCAN_START:{scan_onset_time}'.encode('utf-8'))

    # Send 'QUIT'
    # sock.sendall('QUIT;'.encode('utf-8'))

    # Wati for 'END_SESSION'
    sock.settimeout(None)
    print(time.ctime(), sock.recv(1024).decode())
    sock.settimeout(3)
    time.sleep(3)


# %% Baseline
run = True

if run:
    if not isAlive_extApp(sock):
        sock.close()
        sock, serv_addr, proc = boot_connect_app()

    # Prepare
    task_param = {'session': 'Baseline', 'TR': 2,
                  'timings': [96, 40, 40, 40, 4], 'target_level': 1}
    sock.sendall('PREP NoNF;'.encode('utf-8')+pack_data(task_param))
    time.sleep(3)

    # Send 'READY'
    sock.sendall('READY;'.encode('utf-8'))
    print(time.ctime(), sock.recv(1024).decode())
    time.sleep(3)

    # Send 'SCAN_ONSET'
    scan_onset_time = datetime.now().isoformat()
    sock.sendall(f'SCAN_START {scan_onset_time}'.encode('utf-8'))

    # Wati for 'END_SESSION'
    sock.settimeout(None)
    print(time.ctime(), sock.recv(1024).decode())
    sock.settimeout(3)
    time.sleep(3)


# %% NF
N_Run = 3

for nr in range(N_Run):
    if not isAlive_extApp(sock):
        sock.close()
        sock, serv_addr, proc = boot_connect_app()

    # Prepare
    task_param = {'session': f'NF{nr+1}', 'TR': 2,
                  'timings': [96, 40, 40, 40, 4], 'target_level': 1}
    sock.sendall('PREP NoNF;'.encode('utf-8')+pack_data(task_param))
    time.sleep(3)

    # Send 'READY'
    sock.sendall('READY;'.encode('utf-8'))
    print(time.ctime(), sock.recv(1024).decode())
    time.sleep(3)

    # Send 'SCAN_ONSET'
    scan_onset_time = datetime.now().isoformat()
    sock.sendall(f'SCAN_START {scan_onset_time}'.encode('utf-8'))

    # simulate RTP
    TR = task_param['TR']
    timings = task_param['timings']
    tlen = (timings[0] + (timings[1] + timings[2] + timings[3]) * timings[4])
    NrVol = tlen / TR

    send_t = np.arange(NrVol) * TR + TR + 0.5  # Delay 1 TR + 0.5s (RTP time)
    st = time.time()
    for vi in range(int(NrVol)):
        if vi < 3:
            continue

        # Wait for send time
        while time.time()-st < send_t[vi]:
            time.sleep(0.01)

        # Send a volume
        val = norm.rvs()
        sock.sendall(f'NF {vi}:{val}'.encode('utf-8'))

    # Wati for 'END_SESSION'
    sock.settimeout(None)
    print(time.ctime(), sock.recv(1024).decode())
    sock.settimeout(3)
    time.sleep(3)


# %% Transfer
run = True

if run:
    if not isAlive_extApp(sock):
        sock.close()
        sock, serv_addr, proc = boot_connect_app()

    # Prepare
    task_param = {'session': 'Transfer', 'TR': 2,
                  'timings': [96, 40, 40, 40, 4], 'target_level': 1}
    sock.sendall('PREP NoNF;'.encode('utf-8')+pack_data(task_param))
    time.sleep(3)

    # Send 'READY'
    sock.sendall('READY;'.encode('utf-8'))
    print(time.ctime(), sock.recv(1024).decode())
    time.sleep(3)

    # Send 'SCAN_ONSET'
    scan_onset_time = datetime.now().isoformat()
    sock.sendall(f'SCAN_START {scan_onset_time}'.encode('utf-8'))

    # Wati for 'END_SESSION'
    sock.settimeout(None)
    print(time.ctime(), sock.recv(1024).decode())
    sock.settimeout(3)
    time.sleep(3)

sock.sendall('END_APP;'.encode('utf-8'))
proc.kill()
