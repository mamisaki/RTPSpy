#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Left amygdala neurofeedback application with PsychoPy

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
import sys
import os
from pathlib import Path
import argparse
import time
from datetime import datetime
import traceback

import numpy as np
import pandas as pd
from psychopy import visual, event, core, gui
from rtp_serve import RTP_SERVE, boot_RTP_SERVE_app


# %% Stimulus parameters ======================================================

# --- Instruction screen ---
# text size is ralative to the screen height (=1.0)
inst_txt_height = 0.04
# text wrap width is realative to scareen height (=1.0)
inst_txt_wrap = 0.95

# --- Box size, color, and text size in block screen ---
box_size = 0.16
blkMsg_txt_height = box_size/4.5  # text height in the box
# word size in describe block
desc_word_height = 0.04

box_color = {}
box_color['Rest'] = tuple((np.array([135, 206, 250])-128)/128)  # lightskyblue
box_color['Happy'] = tuple((np.array([255, 99, 71])-128)/128)  # tomato
box_color['Count'] = tuple((np.array([255, 215, 0])-128)/128)  # gold

# --- Set Feedback bar color and size ---
fbBar_pos_color = tuple((np.array([255, 0, 0])-128)/128)  # red
fbBar_neg_color = tuple((np.array([30, 144, 255])-128)/128)  # dodgeblue
targetBar_color = tuple((np.array([0, 0, 255])-128)/128)  # blue
fb_valTxt_color = (1, 1, 1)

# Feedback bar size
# screen height = 1.0
nf_range = [-1.6, 1.6]
bar_amax = 0.4  # maximum height of the bar
bar_scale = bar_amax/np.max(np.abs(nf_range))
bar_base = -box_size
bar_width = 0.11
fbBar_xpos = 0.15
targetBar_xpos = fbBar_xpos+bar_width+0.0025
val_word_height = bar_width/3.5  # word height in the feedback block

# --- rating ---
rating_txt_height = 0.05

# --- Response key assignment ---
rate_keys = {'L': 'g', 'E': 'y', 'R': 'b'}  # keys for rating

# --- background, line, and text colors ---
#  rgb tuple, [-1, 1]
bg_color = (0.25, 0.25, 0.25)  # background color
ln_color = (1.0, 1.0, 1.0)  # fixation line color
txt_color = (1.0, 1.0, 1.0)  # message text color

# --- Fixation size ---
fix_len = 0.05  # fixation cross line length; ralative to screen height
fix_lw = 2      # fixation cross line width in pixels

nf_range = [-1.5, 1.5]


# %% NFApp class ==============================================================
class NFApp(object):

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, allow_remote_access=False, request_host=None,
                 size=(800, 600), pos=None, screen=0, fullscr=False,
                 log_dir='./log', debug=False, open_param_dialog=False,
                 **kwargs):
        """
        Left amygdala neurofeedback with happy memory recall PyschoPy
        application.

        Parameters
        ----------
        allow_remote_access : bool, optional
            Allow remote access (connection from other than localhost).
            The default is False.
        request_host : str, optional
            Host to receive the sever address:port. The default is None.
        size : array or int, optional
            Size of the window in pixels [x, y]. The default is (800, 600).
        pos : array or int, optional
            Location of the top-left corner of the window on the screen [x, y].
            The default is None.
        screen : int, optional
            Screen number for psychopy.visual.Window. The default is 0.
        fullscr : bool, optional
            Create a window in ‘full-screen’ mode. The default is False.
        log_dir : Path or str, optional
            Log directory. The default is './log'.
        open_param_dialog : bool, optional
            Open parameter setup dialog. The default is False.
        debug : bool, optional
            Debug flag. The default is False.

        """
        self.DEBUG = debug
        self.class_name = self.__class__.__name__
        self.rtpapp_address_str = ''
        self.state = 'init'
        self.client_address_str = ''

        self.scan_onset = -1  # scan onset time (second)

        # --- Set parameters with a dialog ------------------------------------
        if open_param_dialog:
            params = {'allow_remote_access': allow_remote_access,
                      'size': ', '.join([str(v) for v in size])}
            if pos is not None:
                params['pos'] = ', '.join([str(v) for v in pos])
            else:
                params['pos'] = '1920, 0'
            params['screen'] = screen
            params['full screen'] = fullscr

            dlg = gui.DlgFromDict(dictionary=params,
                                  title="App server Parameters",
                                  sortKeys=False)
            if not dlg.OK:
                print("CANCELED")
                core.quit()

            allow_remote_access = params['allow_remote_access']
            size = [int(v.strip()) for v in params['size'].split(',')]
            pos = [int(v.strip()) for v in params['pos'].split(',')]
            fullscr = params['full screen']

        # --- Boot RTP_SERVE --------------------------------------------------
        # Boot the socketserver.TCPServer in RTP_SERVE to receive a signal from
        # the RTPSpy App/
        self.rtp_srv = RTP_SERVE(
            allow_remote_access=allow_remote_access,
            request_host=request_host, verb=True)  # self.DEBUG)

        # --- Open a log file -------------------------------------------------
        self.log_dir = log_dir
        self._set_log(self.log_dir)

        self.sess_log = None
        self.sess_log_dir = self.log_dir

        # --- Open the window and make visual objects -------------------------
        self.win = self._open_window(size=size, pos=pos, screen=screen,
                                     fullscr=fullscr)
        self._make_visual_objects(self.win)

        # --- Show welcome message ---
        msg = "Welcome!\n"
        self.msg_txt.setText(msg)
        self.msg_txt.draw()

        srv_addr, srv_port = self.rtp_srv.server.server_address
        sub_msg = f'App server is running on {srv_addr}:{srv_port}'
        self.sub_txt.setText(sub_msg)
        self.sub_txt.draw()

        self.win.flip()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_log(self, log_dir):
        try:
            if hasattr(self, '_log_fd') and self._log_fd != sys.stdout:
                self._log_fd.close()
        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_obj, exc_tb)
            pass

        if log_dir is not None:
            if self.DEBUG:
                log_f = Path(log_dir) / f"log_{self.class_name}_debug.log"
            else:
                log_f = Path(log_dir) / \
                    (f"log_{self.class_name}_" +
                     f"{time.strftime('%Y%m%d_%H%M%S')}.log")

            if not log_f.parent.is_dir():
                os.makedirs(log_f.parent)

            self._log_fd = open(log_f, 'w')
            log = "datetime, time_from_scan_onset, event, misc"
            self._log_fd.write(log + '\n')
            self._log_fd.flush()
        else:
            self._log_fd = sys.stdout

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _open_window(self, **kwargs):
        # Open the stimulu window
        if self.DEBUG:
            kwargs['allowGUI'] = True
            kwargs['fullscr'] = False
        else:
            kwargs['allowGUI'] = False

        win = visual.Window(units='height', color=bg_color, **kwargs)
        return win

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _make_visual_objects(self, win):
        # background
        self.bg_rect = visual.Rect(win, width=2.0*4/3, height=2.0,
                                   fillColor=bg_color,
                                   lineColor=bg_color)

        # Message text
        self.msg_txt = visual.TextStim(win, wrapWidth=inst_txt_wrap,
                                       height=inst_txt_height,
                                       anchorHoriz='center',
                                       color=txt_color)
        self.sub_txt = visual.TextStim(win, height=inst_txt_height/2,
                                       anchorHoriz='center',
                                       pos=[0, -0.5+inst_txt_height/2],
                                       color=txt_color)

        # fixation
        self.fix_hl = visual.Line(win, (-fix_len/2.0, 0.0),
                                  (fix_len/2.0, 0.0), lineWidth=fix_lw,
                                  units='height', lineColor=ln_color)
        self.fix_vl = visual.Line(win, (0.0, -fix_len/2.0),
                                  (0.0, fix_len/2.0), lineWidth=fix_lw,
                                  units='height', lineColor=ln_color)

        # Center square
        self.csquare = visual.Rect(win, pos=(0, 0), width=box_size,
                                   height=box_size, lineColor=(-1, -1, -1),
                                   fillColor=(1, 1, 1))
        self.blkMsg = visual.TextStim(win, pos=(0, 0),
                                      height=blkMsg_txt_height,
                                      color=(-1, -1, -1))

        # Block instruction text
        self.blkInstTxt = visual.TextStim(
            win, pos=(0, box_size/2+desc_word_height*1.1),
            height=desc_word_height, color=(-1, -1, -1))

        # feedback bars
        self.fb_rect = visual.Rect(win, pos=(fbBar_xpos, bar_base),
                                   width=bar_width, height=0,
                                   fillColor=fbBar_neg_color,
                                   lineColor=fbBar_neg_color)
        self.fb_valTxt = visual.TextStim(
            win, pos=(fbBar_xpos, bar_base-val_word_height/2),
            height=val_word_height, color=fb_valTxt_color)
        self.target_rect = visual.Rect(win,
                                       pos=(targetBar_xpos, bar_base),
                                       width=bar_width, height=0,
                                       fillColor=targetBar_color,
                                       lineColor=targetBar_color)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _log(self, event, misc=''):
        if self._log_fd is None:
            return

        ct = time.time()
        dt_ct = datetime.fromtimestamp(ct)
        tstr = dt_ct.strftime('%Y-%m-%dT%H:%M:%S.%f')
        if self.scan_onset > 0:
            et = core.getTime() - self.scan_onset
        else:
            et = -1

        wstr = f"{tstr},{et:.4f},{event},{misc}"

        self._log_fd.write(wstr + '\n')
        self._log_fd.flush()

        if self.sess_log is not None:
            self.sess_log = self.sess_log.append(
                {'datetime': tstr, 'time_from_scan_onset': et,
                 'event': event, 'misc': misc}, ignore_index=True)

        # echo to stdout
        if self.DEBUG and self._log_fd != sys.stdout:
            sys.stdout.write(wstr + '\n')
            sys.stdout.flush()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _make_task_seq(self, timings, target_level=None, **kwargs):
        InitDur, HappyDur, CountDur, RestDur, repeat_blocks = timings

        # Make event sequence
        self.seq = pd.DataFrame(columns=('onset', 'event', 'option'))
        count_intv = np.random.permutation([3, 4, 6, 7, 9])

        t = 0
        self.seq = self.seq.append(
            pd.Series({'onset': t, 'event': 'Rest'}), ignore_index=True)
        t += InitDur
        for rep in range(repeat_blocks):
            if HappyDur > 0:
                self.seq = self.seq.append(
                    pd.Series({'onset': t, 'event': 'Happy',
                               'option': target_level}),
                    ignore_index=True)
                t += HappyDur

            if CountDur > 0:
                self.seq = self.seq.append(
                    pd.Series({'onset': t, 'event': 'Count',
                               'option': count_intv[rep]}),
                    ignore_index=True)
                t += CountDur

            if RestDur > 0:
                self.seq = self.seq.append(
                    pd.Series({'onset': t, 'event': 'Rest'}),
                    ignore_index=True)
                t += RestDur

        self.seq = self.seq.append(
            pd.Series({'onset': t, 'event': 'END'}), ignore_index=True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def draw_connection_status(self):
        if self.rtp_srv.connected:
            self.sub_txt.setText(f"Connect from {self.client_address_str}")
        else:
            srv_addr, srv_port = self.rtp_srv.server.server_address
            sub_msg = 'App server is running on '
            sub_msg += f"{srv_addr}:{srv_port}"
            self.sub_txt.setText(sub_msg)

        self.sub_txt.draw()
        self.win.flip()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def command_loop(self):
        """ Command handling loop """

        self._log("Start App")
        self.state = 'cmd_loop'

        # Loop
        while True:
            # --- Key press event handling ------------------------------------
            k = event.getKeys(['escape', 'r', 't', 'n', 'a'])
            if k:
                if 'escape' in k:
                    # Abort when escape key is pressed
                    self._log('Press:[ESC]')
                    break

                elif 'a' in k:
                    # Show server address
                    self.draw_connection_status()

                elif 'r' in k:
                    # Rest
                    self._log("Press 'r' for Rest")

                    # Initialize session log
                    self.sess_log = pd.DataFrame(
                        columns=['datetime', 'time_from_scan_onset',
                                 'event', 'misc'])
                    event.clearEvents()

                    # Run Rest session
                    total_duration = 60
                    self.rest_run(total_duration=total_duration)

                    # End session
                    self.state = 'cmd_loop'

                    # Save session log
                    tstamp = time.strftime('%Y%m%d_%H%M%S')
                    sess_log_f = f"log_Rest_{tstamp}.csv"

                    if self.sess_log_dir is not None:
                        sess_log_f = Path(self.sess_log_dir) / sess_log_f

                    self.sess_log.to_csv(sess_log_f)
                    self._log(f'Save session log {sess_log_f}')

                    self.sess_log = None

                else:  # 't' or 'n'
                    # 't': start task run with feedback
                    # 'n': start task run without feedback

                    # Initialize session log
                    self.sess_log = pd.DataFrame(
                        columns=['datetime', 'time_from_scan_onset',
                                 'event', 'misc'])

                    # Run NF session
                    if k == 't':
                        nofb = False
                    elif k == 'n':
                        nofb = True

                    timings = [10, 40, 40, 40, 4]
                    self.task_run('NF', timings, TR=2.0, target_level=1.0,
                                  nofb=nofb)

                    # End session
                    self.state = 'cmd_loop'

                    # Save session log
                    tstamp = time.strftime('%Y%m%d_%H%M%S')
                    sess_log_f = f"log_Task_{tstamp}.csv"

                    if self.sess_log_dir is not None:
                        sess_log_f = Path(self.sess_log_dir) / sess_log_f

                    self.sess_log.to_csv(sess_log_f)
                    self._log(f'Save session log {sess_log_f}')

                    self.sess_log = None

            # --- Command on rtp_srv ------------------------------------------
            try:
                if self.rtp_srv.server.client_address_str != \
                        self.client_address_str:
                    # Update connection message
                    self.client_address_str = \
                        self.rtp_srv.server.client_address_str
                    self.draw_connection_status()
                    self._log(f'Change connection:{self.client_address_str}')

                # --- Check received data on self.rtp_srv ---------------------
                rcv_data = self.rtp_srv.get_recv_queue(timeout=0.01)
                if rcv_data is not None and type(rcv_data) is str:
                    cmd_str = rcv_data

                    if 'GET_STATE' in cmd_str:
                        # --- State request. Return self.state ----------------
                        self.rtp_srv.send(self.state.encode('utf-8'))
                        self._log(f'Send state {self.state}')

                    elif 'SET_LOGDIR' in cmd_str:
                        # --- Set log directory -------------------------------
                        log_dir = cmd_str.split()[1]
                        self.sess_log_dir = log_dir
                        self._log(f'Set log directory {self.sess_log_dir}')

                        if self.log_dir != log_dir:
                            self.log_dir = log_dir
                            self._set_log(self.log_dir)

                    elif 'PREP_Rest' in cmd_str:
                        # --- Prepare Rest session ----------------------------
                        self._log(f'Recv:{cmd_str}')

                        # Initialize session log
                        self.sess_log = pd.DataFrame(
                            columns=['datetime', 'time_from_scan_onset',
                                     'event', 'misc'])

                        # parameter dict should be sent following the PREP_*
                        params = self.rtp_srv.get_recv_queue(timeout=5)
                        if type(params) is not dict:
                            self._log(
                                'ERROR: received data is not a parameter dict')
                            continue

                        # Run Rest session
                        ret = self.rest_run(**params)

                        # End session
                        self.state = 'cmd_loop'

                        # Save session log
                        if 'session' in params:
                            sess = params['session'].replace(' ', '')
                        else:
                            sess = 'Rest'

                        tstamp = time.strftime('%Y%m%dT%H%M%S')
                        sess_log_f = f"log_{sess}_{tstamp}.csv"

                        if self.sess_log_dir is not None:
                            sess_log_f = Path(self.sess_log_dir) / sess_log_f

                        self.sess_log.to_csv(sess_log_f, index=False)
                        self._log(f'Save session log {sess_log_f}')

                        if ret == -1:
                            # QUIT
                            break

                        self.sess_log = None
                        self.rtp_srv.flush_send()
                        self.rtp_srv.flush_recv()

                    elif 'PREP_Task' in cmd_str:
                        # --- Prepare View/NF session
                        self._log(f'Recv:{cmd_str}')

                        # Initialize session log
                        self.sess_log = pd.DataFrame(
                            columns=['datetime', 'time_from_scan_onset',
                                     'event', 'misc'])

                        # parameter data should be sent following the PREP_*
                        params = self.rtp_srv.get_recv_queue(timeout=5)
                        if type(params) is not dict:
                            self._log(
                                'ERROR: received data is not a parameter dict')
                            continue

                        # Run NF session
                        ret = self.task_run(**params)

                        # End session
                        self.state = 'cmd_loop'

                        # Save session log
                        if 'session' in params:
                            sess = params['session'].replace(' ', '')
                        else:
                            sess = 'Task'

                        tstamp = time.strftime('%Y%m%dT%H%M%S')
                        sess_log_f = f"log_{sess}_{tstamp}.csv"

                        if self.sess_log_dir is not None:
                            sess_log_f = Path(self.sess_log_dir) / sess_log_f

                        self.sess_log.to_csv(sess_log_f)
                        self._log(f'Save session log {sess_log_f}')

                        if ret == -1:
                            # QUIT
                            break

                        self.sess_log = None
                        self.rtp_srv.flush_send()
                        self.rtp_srv.flush_recv()

                    elif cmd_str == 'QUIT':
                        self._log('Recv:QUIT')
                        break

            except Exception:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                traceback.print_exception(exc_type, exc_obj, exc_tb)
                continue

        # End
        time.sleep(1)
        self._log('End')

        if self._log_fd is not None and self._log_fd != sys.stdout:
            self._log_fd.close()

        self.win.close()
        del self.rtp_srv

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def rest_run(self, total_duration, **kwargs):
        """
        Rest run
        """

        self.state = 'rest_run'

        # Reset NF_signal
        self.rtp_srv.reset_NF_signal()

        # --- Instruction -----------------------------------------------------
        instruction_f = Path(__file__).parent / 'instruction_Rest.txt'
        inst_df = pd.read_csv(instruction_f, quotechar="'", header=None)
        instruction_txt = {}
        for idx1, row1 in inst_df.iterrows():
            sess = row1[0]
            text = row1[1]
            instruction_txt[sess] = text.replace('\\n', '\n')

        instruction_txt = instruction_txt['HEADER']
        self.msg_txt.setText(instruction_txt)
        self.msg_txt.draw()
        self.win.flip()
        self._log('Show Rest instruction')

        self.rtp_srv.send("PREPED;".encode('utf-8'))

        # -- Wait for ready --
        while True:
            # Check received data
            rcv_data = self.rtp_srv.get_recv_queue(timeout=0.1)
            if rcv_data is not None and type(rcv_data) is str:
                if rcv_data == 'READY':
                    self.rtp_srv.send('READY;'.encode('utf-8'))
                    break
                elif 'GET_STATE' in rcv_data:
                    self.rtp_srv.send(self.state.encode('utf-8'))
                    self._log(f'Send state {self.state}')
                elif rcv_data == 'END':
                    self.end_proc(abort=True)
                    return 0
                elif rcv_data == 'QUIT':
                    self.end_proc(abort=True)
                    return -1

            # Abort when escape key is pressed
            if event.getKeys(["escape"]):
                self._log('Press:[ESC]')
                self.end_proc(abort=True)
                return 0

            # Go ahead when space key is pressed
            if event.getKeys(["space"]):
                self.rtp_srv.send('READY;'.encode('utf-8'))
                break

        # -- Show 'Ready' --
        self.msg_txt.setText("Ready")
        self.msg_txt.draw()
        self.win.flip()
        self._log('Ready')

        # --- Wait for the scan start -----------------------------------------
        while True:
            rcv_data = self.rtp_srv.get_recv_queue(timeout=0.1)
            if rcv_data is not None and type(rcv_data) is str:
                if 'SCAN_START' in rcv_data:
                    self.scan_onset = core.getTime()
                    break
                elif 'GET_STATE' in rcv_data:
                    self.rtp_srv.send(self.state.encode('utf-8'))
                    self._log(f'Send state {self.state}')
                elif rcv_data == 'END':
                    self.end_proc(abort=True)
                    return 0
                elif rcv_data == 'QUIT':
                    self._log('QUIT')
                    self.end_proc(abort=True)
                    return -1

            # Abort when escape key is pressed
            if event.getKeys(["escape"]):
                self._log('Press:[ESC]')
                self.end_proc(abort=True)
                return 0

            # Go ahead when space key is pressed
            if event.getKeys(["space"]):
                self.scan_onset = core.getTime()
                dt_ons_str = datetime.isoformat(
                    datetime.fromtimestamp(time.time()))
                log = f"Press 'space' to start at {dt_ons_str}"
                self._log(log)
                break

        # --- Rest loop -------------------------------------------------------
        self.show_screen('FIXATION')

        abort = False
        while core.getTime()-self.scan_onset < total_duration and not abort:
            try:
                # Check received data
                rcv_data = self.rtp_srv.get_recv_queue(timeout=0.1)
                if rcv_data is not None and type(rcv_data) is str:
                    if 'GET_STATE' in rcv_data:
                        self.rtp_srv.send(self.state.encode('utf-8'))
                        self._log(f'Send state {self.state}')
                    elif rcv_data == 'END':
                        self._log('END')
                        self.end_proc(abort=True)
                        break
                    elif rcv_data == 'QUIT':
                        self._log('QUIT')
                        self.end_proc(abort=True)
                        return -1

                # Abort when escape key is pressed
                if event.getKeys(["escape"]):
                    self._log('Press:[ESC]')
                    abort = True
                    break

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                errmsg = '{}, {}:{}'.format(
                        exc_type, exc_tb.tb_frame.f_code.co_filename,
                        exc_tb.tb_lineno)
                errmsg += ' ' + str(e)
                self._log("!!!Error:{}".format(errmsg))

        # -- End --------------------------------------------------------------
        self.end_proc(abort)

        return 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def task_run(self, session, timings, TR, target_level=None, nofb=False,
                 **kwargs):
        """
        Running NF task
        """

        self.state = 'task_run'
        frameIntv = self.win.monitorFramePeriod / 2

        # Make a task sequence
        self._make_task_seq(timings, target_level=target_level)
        tEND = self.seq.iloc[-1, :].onset
        self.rcv_signal = np.ones(int(np.ceil(tEND/TR))) * np.nan
        self.last_vidx = -1

        self.rtp_srv.reset_NF_signal()  # Reset NF_signal in rtp_srv

        # --- Show instructions -----------------------------------------------
        if session in ('Baseline', 'Transfer'):
            instruction_f = Path(__file__).parent / 'instruction_NoNF.txt'
        else:
            instruction_f = Path(__file__).parent / 'instruction_NF.txt'
        inst_df = pd.read_csv(instruction_f, quotechar="'", header=None)
        instruction_txt = {}
        for idx1, row1 in inst_df.iterrows():
            sess = row1[0]
            text = row1[1]
            instruction_txt[sess] = text.replace('\\n', '\n')

        self.show_instruction_screen(instruction_txt)
        self.rtp_srv.send("PREPED;".encode('utf-8'))

        # --- Wait for receiving READY ----------------------------------------
        while True:
            # Check received data
            rcv_data = self.rtp_srv.get_recv_queue(timeout=0.1)
            if rcv_data is not None and type(rcv_data) is str:
                if rcv_data == 'READY':
                    self.rtp_srv.send('READY;'.encode('utf-8'))
                    break
                elif 'GET_STATE' in rcv_data:
                    self.rtp_srv.send(self.state.encode('utf-8'))
                    self._log(f'Send state {self.state}')
                elif rcv_data == 'END':
                    self.end_proc(abort=True)
                    return 0
                elif rcv_data == 'QUIT':
                    self._log('QUIT')
                    self.win.close()
                    return -1

            # Abort when escape key is pressed
            if event.getKeys(["escape"]):
                self._log('Press:[ESC]')
                self.win.close()
                return 0

            # Go ahead when space key is pressed
            if event.getKeys(["space"]):
                if self.rtp_srv.connected:
                    self.rtp_srv.send('READY;'.encode('utf-8'))
                break

        # -- Show 'Ready' --
        self.msg_txt.setText("Ready")
        self.msg_txt.draw()
        self.win.flip()
        self._log('Ready')

        # --- Wait for the scan start -----------------------------------------
        while True:
            rcv_data = self.rtp_srv.get_recv_queue(timeout=0.1)
            if rcv_data is not None and type(rcv_data) is str:
                if 'SCAN_START' in rcv_data:
                    self.scan_onset = core.getTime()
                    break
                elif 'GET_STATE' in rcv_data:
                    self.rtp_srv.send(self.state.encode('utf-8'))
                    self._log(f'Send state {self.state}')
                elif rcv_data == 'END':
                    self.end_proc(abort=True)
                    return 0
                elif rcv_data == 'QUIT':
                    self._log('QUIT')
                    self.end_proc(abort=True)
                    return -1

            # Abort when escape key is pressed
            if event.getKeys(["escape"]):
                self._log('Press:[ESC]')
                self.end_proc(abort=True)
                return 0

            # Go ahead when space key is pressed
            if event.getKeys(["space"]):
                self.scan_onset = core.getTime()
                dt_ons_str = datetime.isoformat(
                    datetime.fromtimestamp(time.time()))
                log = f"Press 'space' to start at {dt_ons_str}"
                self._log(log)
                break

        # --- Task loop -------------------------------------------------------
        # Initialize control parameter
        i_ev = 0
        tNext = self.seq.onset.iloc[i_ev]
        abort = False

        last_vidx = -1  # Latest volume index of the received NF signal.
        cur_vidx = -1  # Current volume index shown as NF.
        cur_ev = ''  # Current event
        blk_ons_vidx = -1  # volume index of the block onset

        baseMean = None  # Baseline signal mean
        baseSD = None  # Baseline signal SD
        restBlk_ons_vidx = None  # Previous rest block onset
        restBlk_offs_vidx = None  # Previous rest block offset
        fbval_hist = []  # feedback value history for moving average

        # loop
        while core.getTime()-self.scan_onset < tEND and not abort:
            try:
                # Check a message received
                rcv_data = self.rtp_srv.get_recv_queue(timeout=0.01)
                if rcv_data is not None and type(rcv_data) is str:
                    if 'GET_STATE' in rcv_data:
                        self.rtp_srv.send(self.state.encode('utf-8'))
                        self._log(f'Send state {self.state}')
                    elif rcv_data == 'END':
                        self._log('END')
                        self.end_proc(abort=True)
                        return 0
                    elif rcv_data == 'QUIT':
                        self._log('QUIT')
                        self.end_proc(abort=True)
                        return -1

                # Abort when escape key is pressed
                if event.getKeys(["escape"]):
                    self._log('Press:[ESC]')
                    abort = True
                    continue

                # --- Update event ---
                if core.getTime()-self.scan_onset >= tNext-frameIntv:
                    ev = self.seq.event.iloc[i_ev]
                    msgTxt = self.seq.option.iloc[i_ev]
                    if pd.isnull(msgTxt):
                        msgTxt = None
                    tCurr = tNext

                    # Update the end time of the event
                    i_ev += 1
                    if i_ev == len(self.seq):
                        tNext = np.inf
                    else:
                        tNext = self.seq.onset.iloc[i_ev]

                    # Screen change with the updated event
                    if cur_ev != ev:
                        # Set block onset
                        blk_ons_vidx = int(tCurr//TR)
                        self._log(f"Start {ev} block")
                        cur_ev = ev

                        if cur_ev == 'Rest':
                            # Show Rest screen
                            self.show_screen(cur_ev)
                            # Reset Rest paramter values
                            restBlk_ons_vidx = int(tCurr // TR)
                            restBlk_offs_vidx = int(tNext // TR)
                            baseMean = None
                            baseSD = None

                        elif cur_ev == 'Count':
                            # Show Count screen
                            self.show_screen(cur_ev, msgTxt=msgTxt)

                        elif cur_ev == 'Happy':
                            # Happy block
                            if nofb:
                                self.show_screen(cur_ev)
                                continue
                            else:
                                self.show_screen(cur_ev, fb_value=0,
                                                 msgTxt=msgTxt)

                # --- Update neurofeedback presentation ---
                if not nofb and cur_ev == 'Happy' and \
                        len(self.rtp_srv.NF_signal):
                    # Set the NF signal baseline
                    if baseSD is None and \
                            self.rtp_srv.NF_signal.TR.values[-1] >= \
                            restBlk_offs_vidx and baseSD is None:

                        # Get the previous Rest block signals
                        sigTR = self.rtp_srv.NF_signal.TR.values
                        signal = np.concatenate(
                            self.rtp_srv.NF_signal.Signal.values).ravel()
                        baseMask = (sigTR >= restBlk_ons_vidx+3) & \
                            (sigTR < restBlk_offs_vidx)
                        base_sig = signal[baseMask]

                        # Maximum last 12TR
                        base_sig = base_sig[-12:]
                        # Exclude outlier (> 2 *SD)
                        base_sig_mask = \
                            ((base_sig - np.nanmean(base_sig)) <
                             np.nanstd(base_sig) * 2)
                        base_sig = base_sig[base_sig_mask]

                        # Set baseline mean and SD
                        baseMean = np.nanmean(base_sig)
                        baseSD = np.nanstd(base_sig)

                        # Initialize signal history for moving average
                        fbval_hist = []
                        self._log("Set baseline " +
                                  f"mean={baseMean:.3f} (SD={baseSD:.3f})")

                    cur_vidx = self.rtp_srv.NF_signal.TR.values[-1]
                    if self.rtp_srv.NF_signal.TR.values[-1] < \
                            blk_ons_vidx+2:
                        # No feedback until receiving a valid NF signal
                        # (skip the initial 2 volumes )
                        continue

                    elif last_vidx != cur_vidx:
                        # Show feedback
                        sig = self.rtp_srv.NF_signal.Signal[
                            self.rtp_srv.NF_signal.TR == cur_vidx]
                        sig = sig.values[0][0]
                        fbsig = (sig - baseMean) / baseSD

                        fbval_hist.append(fbsig)
                        if len(fbval_hist) > 3:
                            # average 3 latest values
                            fbval_hist.pop(0)

                        fbsig = np.mean(fbval_hist)
                        self.show_screen(cur_ev, fb_value=fbsig,
                                         msgTxt=msgTxt)
                        last_vidx = cur_vidx

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                errmsg = '{}, {}:{}'.format(
                        exc_type, exc_tb.tb_frame.f_code.co_filename,
                        exc_tb.tb_lineno)
                errmsg += ' ' + str(e)
                self._log("!!!Error:{}".format(errmsg))

        # --- Loop end ---
        if not abort or core.getTime()-self.scan_onset > tEND:
            self._log('Session has completed')
            abort = False

        if not nofb:
            # Wait for RTP to process the last volume
            wait_start = core.getTime()
            while self.rtp_srv.NF_signal.TR.values[-1] < tEND/TR-1 and \
                    core.getTime()-wait_start < TR*3:
                time.sleep(0.1)

        # -- End --------------------------------------------------------------
        self.end_proc(abort)

        return 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def show_instruction_screen(self, instruction_txt):

        # Set background color
        self.bg_rect.draw()
        top_pos = 0.22

        if 'HEADER' in instruction_txt:
            titleTxt = visual.TextStim(self.win,
                                       pos=(0, top_pos+0.1+inst_txt_height),
                                       text=instruction_txt['HEADER'],
                                       height=inst_txt_height,
                                       color=(-1, -1, -1))
            titleTxt.draw()
            del instruction_txt['HEADER']

        # --- block instructions ---
        for ev, inst_txt in instruction_txt.items():
            ev_idx = ev.replace('\\n', '\n').split('\n')[0]
            csquare = visual.Rect(self.win, pos=(-0.35, top_pos),
                                  width=box_size, height=box_size,
                                  lineColor=(-1, -1, -1),
                                  fillColor=box_color[ev_idx])
            blkMsg = visual.TextStim(self.win, pos=(-0.35, top_pos),
                                     text=ev.replace('\\n', '\n'),
                                     anchorHoriz='center',
                                     height=blkMsg_txt_height,
                                     color=(-1, -1, -1))
            instTxt = visual.TextStim(
                    self.win, alignText='left',
                    pos=(-0.35+box_size/2+inst_txt_height*0.8,
                         top_pos+box_size/2),
                    anchorHoriz='left', anchorVert='top', wrapWidth=0.7,
                    text=inst_txt, height=inst_txt_height*0.8,
                    color=(-1, -1, -1))
            csquare.draw()
            blkMsg.draw()
            instTxt.draw()
            top_pos -= 0.22

        self.win.flip()
        self._log("Show instruction screen")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def show_screen(self, ev, fb_value=None, msgTxt=None):

        if ev == 'Rest':
            # Set background color
            self.bg_rect.draw()

            self.csquare.fillColor = box_color['Rest']
            self.csquare.draw()

            self.blkMsg.text = 'Rest'
            self.blkMsg.draw()

            ftime = self.win.flip()
            self._log(f"Show {ev} screen")

        elif 'Happy' in ev:
            # Set background color
            self.bg_rect.draw()

            self.csquare.fillColor = box_color['Happy']
            self.csquare.draw()

            self.blkMsg.text = ev.replace('\\n', '\n')
            self.blkMsg.draw()

            if fb_value is not None:
                self.fb_rect.height = min(np.abs(fb_value)*bar_scale, bar_amax)
                # pos is the position of the center of the rectangle
                self.fb_rect.pos = (fbBar_xpos, bar_base +
                                    np.sign(fb_value)*self.fb_rect.height/2)
                if fb_value < 0:
                    self.fb_rect.fillColor = fbBar_neg_color
                    self.fb_rect.lineColor = fbBar_neg_color
                else:
                    self.fb_rect.fillColor = fbBar_pos_color
                    self.fb_rect.lineColor = fbBar_pos_color

                self.fb_rect.draw()

                if msgTxt is not None:
                    target_level = float(msgTxt)
                    self.target_rect.height = target_level*bar_scale
                    self.target_rect.pos = (targetBar_xpos, bar_base +
                                            self.target_rect.height/2)
                    self.target_rect.draw()
                else:
                    target_level = 'None'

                if fb_value != 0.0:
                    self.fb_valTxt.setText("{:+.2f}".format(fb_value))
                    if fb_value < 0:
                        pos = (fbBar_xpos, bar_base + val_word_height/2 + 0.01)
                        """
                        pos = (fbBar_xpos, bar_base +
                                np.sign(fb_value)*self.fb_rect.height -
                                val_word_height/2 - 0.01)
                        """
                    else:
                        pos = (fbBar_xpos, bar_base - val_word_height/2 - 0.01)

                    self.fb_valTxt.pos = pos
                    self.fb_valTxt.draw()

            ftime = self.win.flip()

            if fb_value is not None:
                self._log("Show Happy screen:" +
                          f"FB={fb_value:+.2f}:Target={target_level}")
            else:
                self._log("Show Happy screen")

        elif ev == 'Count':
            # Set background color
            self.bg_rect.draw()

            self.csquare.fillColor = box_color['Count']
            self.csquare.draw()

            self.blkMsg.text = ev
            self.blkMsg.draw()

            if msgTxt is not None:
                d = int(msgTxt)
                txt = f"Count 300, {300-d}, {300-2*d}, ... (-{d})"
                self.blkInstTxt.setText(txt)
                self.blkInstTxt.draw()

            ftime = self.win.flip()
            self._log(f"Show {ev};{txt} screen")

        elif ev == 'FIXATION':
            # Set background color
            bg_col = bg_color
            self.bg_rect.fillColor = bg_col
            self.bg_rect.lineColor = bg_col
            self.bg_rect.draw()

            # Show fixation cross
            self.fix_hl.draw()
            self.fix_vl.draw()

            ftime = self.win.flip()
            self._log(f"Show {ev} screen")

        return ftime

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end_proc(self, abort=False):
        if abort:
            self.msg_txt.setText("Please wait for instructions.")
            self._log("Session was aborted.")

        else:
            self.msg_txt.setText("The session has completed.\n" +
                                 "Please wait for instructions.")

        self.msg_txt.draw()
        self.win.flip()
        self._log('End task')

        event.clearEvents()

        self.rtp_srv.send('END_SESSION;'.encode('utf-8'))
        self._log('Send:END_SESSION')

        self.scan_onset = -1


# %% main =====================================================================
if __name__ == '__main__':
    # --- parse auguments -----------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, nargs='+', help='window size',
                        default=[800, 600])
    parser.add_argument('--pos', type=int, nargs='+', help='window position')
    parser.add_argument('--screen', type=int, help='screen', default=0)
    parser.add_argument('--fullscr', action='store_true', default=False,
                        help='fullscreen mode')
    parser.add_argument('--log_dir', default='./log', help='log directory')
    parser.add_argument('--tell_port', action='store_true', default=False,
                        help='Print opened port, folk process to run the app,'
                        + ' and return immediately.')
    parser.add_argument("--debug", action='store_true', default=False,
                        help="debug mode")

    # WHen booted with boot_RTP_SERVE_app
    parser.add_argument('--allow_remote_access', action='store_true',
                        default=False, help='allow remote access')
    parser.add_argument('--request_host',
                        help="host:port to answer the sever's host:port")

    args = parser.parse_args()
    kwargs = args.__dict__

    if args.tell_port:
        argvs = sys.argv
        argvs.pop(argvs.index('--tell_port'))
        if '--allow_remote_access' in argvs:
            argvs.pop(argvs.index('--allow_remote_access'))
        if '--request_host' in argvs:
            idx = argvs.index('--request_host')
            argvs.pop(idx+1)
            argvs.pop(idx)

        cmd = ' ' .join(argvs)
        addr, pr = boot_RTP_SERVE_app(cmd, remote=args.allow_remote_access,
                                      timeout=None)
        print(addr)

    else:
        kwargs = args.__dict__

        # Make TaskApp instance
        app = NFApp(**kwargs)
        app.command_loop()

        # End
        del app

    sys.exit(0)
