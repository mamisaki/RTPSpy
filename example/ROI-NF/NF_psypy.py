#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple neurofeedback application with PsychoPy communicating with RTPSpy.

This example just shows a received value on screen.

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
import sys
from pathlib import Path
import argparse
import time
from datetime import datetime
import traceback

import numpy as np
from psychopy import visual, event, core
from rtp_serve import RTP_SERVE, boot_RTP_SERVE_app


# %% NFApp class ==============================================================
class NFApp():

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, allow_remote_access=False, request_host=None,
                 size=(800, 600), pos=None, screen=0, fullscr=False,
                 log_dir='./log', debug=False, **kwargs):
        """
        Example neurofeedback PyschoPy application.
        Receive signal via RTP_SERVE object and show the value with psychopy.

        Parameters
        ----------
        allow_remote_access : bool, optional
            Allow remote access (connection from other than localhost).
            The default is False.
        request_host : str, optional
            Host to return the server's address:port.
            The default is None.
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
        debug : bool, optional
            Debug flag. The default is False.

        """
        self.DEBUG = debug

        self.class_name = self.__class__.__name__
        self.rtpapp_address_str = ''

        self.scan_onset = -1  # scan onset time (second)

        # --- Boot RTP_SERVE --------------------------------------------------
        # Boot the socketserver.TCPServer in RTP_SERVE to receive a signal from
        # the RTPSpy App/
        self.rtp_srv = RTP_SERVE(
            allow_remote_access=allow_remote_access,
            request_host=request_host, verb=self.DEBUG)

        # --- Open a log file -------------------------------------------------
        self.log_dir = log_dir
        self._set_log(self.log_dir)

        self.sess_log = None
        self.sess_log_dir = self.log_dir

        # --- Open the window and make visual objects -------------------------
        self.win = self._open_window(size=size, pos=pos, screen=screen,
                                     fullscr=fullscr)
        self._make_visual_objects(self.win)

        if self.DEBUG:
            srv_addr, srv_port = self.rtp_srv.server.server_address
            print(srv_addr, srv_port)

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
                     f"{time.strftime('%Y%m%dT%H%M%S')}.log")
            self._log_fd = open(log_f, 'w')
            log = "datetime,time_from_scan_onset,event,misc"
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

        win = visual.Window(**kwargs)
        return win

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _make_visual_objects(self, win):
        # Message text
        self.msg_txt = visual.TextStim(
            win, wrapWidth=0.95, height=0.1, anchorHoriz='center',
            color=(1, 1, 1))

        self.sub_txt = visual.TextStim(
            win, wrapWidth=0.95, height=0.05,  pos=(0, -1),
            anchorHoriz='center',  anchorVert='bottom', color=(1, 1, 1))

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
    def run(self):
        """
        Running sessions.
        """
        while True:  # loop for running multiple sessions
            srv_addr, srv_port = self.rtp_srv.server.server_address
            self.sub_txt.setText(f"Server is running at {srv_addr}:{srv_port}")
            self.sub_txt.draw()
            self.win.flip()

            self.rtp_srv.reset_NF_signal()  # Reset NF_signal in rtp_srv
            END = False

            # --- Wait for receiving READY ------------------------------------
            while True:
                # Check received data.
                rcv_data = self.rtp_srv.get_recv_queue(timeout=0.1)
                if rcv_data is not None and type(rcv_data) == str:
                    if rcv_data == 'READY':
                        self.rtp_srv.send('READY;'.encode('utf-8'))
                        break
                    elif rcv_data == 'END':
                        END = True
                        break
                    elif rcv_data == 'QUIT':
                        self._log('QUIT')
                        self.win.close()
                        return 0

                # Abort when the escape key is pressed.
                if event.getKeys(["escape"]):
                    self._log('Press:[ESC]')
                    self.win.close()
                    return 0

                # Go ahead when the space key is pressed.
                if event.getKeys(["space"]):
                    if self.rtp_srv.connected:
                        self.rtp_srv.send('READY;'.encode('utf-8'))
                    break

            if END:
                continue

            # Show 'Ready'
            self.msg_txt.setText("Ready")
            self.msg_txt.draw()
            self.win.flip()
            self._log('Ready')

            # --- Wait for the scan start -------------------------------------
            while True:
                rcv_data = self.rtp_srv.get_recv_queue(timeout=0.1)
                if rcv_data is not None and type(rcv_data) == str:
                    if 'SCAN_START' in rcv_data:
                        self.scan_onset = core.getTime()
                        break
                    elif rcv_data == 'END':
                        END = True
                        break
                    elif rcv_data == 'QUIT':
                        self._log('QUIT')
                        self.win.close()
                        return 0

                    # Abort when the escape key is pressed.
                    if event.getKeys(["escape"]):
                        self._log('Press:[ESC]')
                        self.win.close()
                        return 0

                    # Go ahead when the space key is pressed.
                    if event.getKeys(["space"]):
                        self.scan_onset = core.getTime()
                        break

            if END:
                continue  # Retrun to wait for receiving the READY.

            # --- Feedback loop -----------------------------------------------
            self.win.flip()
            timeEND = np.inf  # Continue running unless receiving 'END'
            last_NF_idx = 0  # Index of the last read self.rtp_srv.NF_signal.
            last_NF_time = time.time()  # The last NF_signal reading time.

            while core.getTime()-self.scan_onset < timeEND:
                try:
                    # Check a message received
                    rcv_data = self.rtp_srv.get_recv_queue(timeout=0.01)
                    if rcv_data is not None and type(rcv_data) == str:
                        if 'GET_STATE' in rcv_data:
                            self.rtp_srv.send(self.state.encode('utf-8'))
                            self._log(f'Send state {self.state}')
                        elif rcv_data == 'END':
                            self._log('END')
                            break
                        elif rcv_data == 'QUIT':
                            self._log('QUIT')
                            self.win.close()
                            return 0

                    # Abort when the escape key is pressed
                    if event.getKeys(["escape"]):
                        self._log('Press:[ESC]')
                        break

                    # -- Update the feedback message --
                    if len(self.rtp_srv.NF_signal) > last_NF_idx:
                        if time.time()-last_NF_time < 0.5:
                            '''
                            Suppress too frequent updates of the signal.
                            This could happen when volumes in the RtpRegress's
                            wait periods are retrospectively processed and sent.
                            '''
                            continue

                        '''
                        self.rtp_srv.NF_signal is a pandas DataFrame with
                        columns=('Time', 'TR', 'Signal'), keeping the NF
                        signals received from RTPSpy RtpApp.
                        self.rtp_srv.NF_signal.Signal is a list of NF signals.
                        '''
                        last_NF_idx = len(self.rtp_srv.NF_signal)
                        last_NF_time = time.time()
                        TR = self.rtp_srv.NF_signal.TR.values[-1]
                        signal = self.rtp_srv.NF_signal.Signal.values[-1]

                        self.msg_txt.setText(f"{signal}")
                        self.msg_txt.draw()
                        self.win.flip()
                        self._log(f"Show feedback TR={TR} Signal={signal}")

                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    errmsg = '{}, {}:{}'.format(
                            exc_type, exc_tb.tb_frame.f_code.co_filename,
                            exc_tb.tb_lineno)
                    errmsg += ' ' + str(e)
                    self._log("!!!Error:{}".format(errmsg))

        if hasattr(self, 'win'):
            self.win.close()

        return 0


# %% main =====================================================================
if __name__ == '__main__':
    # --- parse auguments -----------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, nargs='+', help='window size',
                        default=[800, 600])
    parser.add_argument('--pos', type=int, nargs='+', help='window position')
    parser.add_argument('--screen', type=int, help='screen', default=0)
    parser.add_argument('--fullscr', action='store_true',
                        help='fullscreen mode')
    parser.add_argument('--log_dir', help='log directory')
    parser.add_argument('--tell_port', action='store_true',
                        help='Print opened port, folk process to run the app,'
                        + ' and return immediately.')
    parser.add_argument("--debug", action='store_true', help="debug mode")

    # WHen booted with boot_RTP_SERVE_app
    parser.add_argument('--allow_remote_access', action='store_true',
                        help='allow remote access')
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
    else:
        kwargs = args.__dict__

        # Make TaskApp instance
        app = NFApp(**kwargs)
        app.run()

        # End
        del app

    sys.exit(0)
