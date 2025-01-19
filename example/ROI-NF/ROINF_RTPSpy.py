#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run psychopy RTPSpy application

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import sys
from platform import uname
from datetime import datetime
import argparse
import logging
import shutil

import numpy as np
from PyQt5 import QtWidgets

# RTP application and utility functions and classes
from rtpspy.rtp_common import excepthook, save_parameters, load_parameters
from rtpspy import RtpGUI
from roi_nf import ROINF


# %% Default parameters =======================================================
# External NF application
this_dir = Path(__file__).absolute().parent
if 'Microsoft' in uname().release:
    cmd_path = r'/mnt/c/Program\ Files/PsychoPy3/pythonw.exe '
    cmd_path += str(this_dir / 'NF_psypy.py')
else:
    cmd_path = f"{this_dir / 'boot_psychopy.sh'} {this_dir / 'NF_psypy.py'}"
extApp_cmd = f"{cmd_path} --screen 0 --size 640 480 --pos 0 0"

# RTP pipeline
rtp_params = {'VOLREG': {'regmode': 'heptic'},
              'TSHIFT': {'method': 'cubic'},
              'REGRESS': {'wait_num': 30, 'max_poly_order': np.inf,
                          'mot_reg': 'mot12', 'GS_reg': True, 'WM_reg': True,
                          'Vent_reg': True, 'phys_reg': 'None'},
              'APP': {'enable_RTP': 2,
                      'run_extApp': True,  # Flag to use an external app
                      'extApp_cmd': extApp_cmd}
              }


# %% main =====================================================================
if __name__ == '__main__':
    dateStr = datetime.now().strftime("%Y%m%dT%H%M%S")
    log_file = Path(f'log/RNT-CNF_{dateStr}.log')

    # Parse arguments
    parser = argparse.ArgumentParser(description='ROI NF RTPSpy')
    parser.add_argument('--log_file', default=log_file,
                        help='Log file path')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    log_file = Path(args.log_file)
    debug = args.debug

    # --- Set logging ---------------------------------------------------------
    # Set log file
    log_dir = Path(log_file).parent
    if not log_dir.is_dir():
        log_dir.mkdir()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level, filename=log_file, filemode='a',
        format='%(asctime)s.%(msecs)04d,[%(levelname)s],%(name)s,%(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S')

    # --- Start application ---------------------------------------------------
    app = QtWidgets.QApplication(sys.argv)
    if sys.platform == 'darwin':
        app.setStyle("Fusion")

    # Make RtpApp instance
    rtp_app = ROINF(default_rtp_params=rtp_params)

    # Make GUI (RtpGUI) instance
    app_obj = {'ROI-NF': rtp_app}
    rtp_ui = RtpGUI(rtp_app.rtp_objs, app_obj, log_file=log_file)
    rtp_app.rtp_gui = rtp_ui

    # Get RTP object instances for loading and saving the parameters
    all_rtp_objs = rtp_app.rtp_objs
    all_rtp_objs.update(app_obj)

    # Ask if loading the previous settings.
    load = QtWidgets.QMessageBox.question(
        None, 'Load parameters', 'Load previous settings?',
        QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
    if load == QtWidgets.QMessageBox.Yes:
        # Load saved parameters: This will override the rtp_params settings
        props = load_parameters(all_rtp_objs, fname='RTPSpy_ROINF_params.pkl')

    # --- Run the application ---
    sys.excepthook = excepthook
    try:
        rtp_ui.show()
        rtp_ui.show_physio_chk(2)
        exit_code = app.exec_()

    except Exception as e:
        with open('rtpspy.error', 'w') as fd:
            fd.write(str(e))

        print(str(e))
        exit_code = -1

    # --- End ---
    # Save parameters
    # exit with 1 means exit wihtout saving (defined in RtpGUI)
    if exit_code != 1:
        save_parameters(all_rtp_objs, fname='RTPSpy_ROINF_params.pkl')

    # Copy log
    work_dir = Path(rtp_app.work_dir)
    log_dir = work_dir / 'log'
    if not log_dir.is_dir():
        log_dir.mkdir()
    if log_file.parent != log_dir:
        shutil.copy(log_file, log_dir / log_file.name)

    sys.exit(exit_code)
