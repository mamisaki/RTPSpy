#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run RTPSpy GUI application with simple ROI signal extraction.
The signal is saved in a file real-time.

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import sys

import numpy as np
from PyQt5 import QtWidgets

# RTP application and utility functions and classes
from rtpspy.rtp_common import excepthook, save_parameters, load_parameters
from rtpspy.rtp_app import RtpApp
from rtpspy.rtp_gui import RtpGUI


# %% Default parameters
same_dir = Path(__file__).absolute().parent

rtp_params = {'TSHIFT': {'method': 'cubic', 'ignore_init': 3, 'ref_time': 0},
              'VOLREG': {'regmode': 'cubic'},
              'SMOOTH': {'blur_fwhm': 6.0},
              'REGRESS': {'wait_num': 40, 'max_poly_order': np.inf,
                          'mot_reg': 'mot12', 'GS_reg': True, 'WM_reg': True,
                          'Vent_reg': True, 'phys_reg': 'RICOR8'},
              'APP': {'enable_RTP': True,
                      'template': same_dir / 'MNI152_2009_template.nii.gz',
                      'ROI_template': same_dir /
                      'MNI152_2009_template_LAmy.nii.gz'}
              }


# %% __main__ =================================================================
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    # Make RtpApp instance
    rtp_app = RtpApp()

    # Set default parameters
    for proc, params in rtp_params.items():
        if proc in rtp_app.rtp_objs:
            for attr, val in params.items():
                rtp_app.rtp_objs[proc].set_param(attr, val)

    for attr, val in rtp_params['APP'].items():
        rtp_app.set_param(attr, val)

    # Make RtpGUI instance
    rtp_ui = RtpGUI(rtp_app.rtp_objs, {'ROI-NF': rtp_app}, log_dir='./log')

    # Load saved parameters
    all_rtp_objs = rtp_app.rtp_objs
    all_rtp_objs.update({'ROI-NF': rtp_app})
    load_parameters(all_rtp_objs)

    # Ask watch_dir for RTPWatch
    ui_ret = rtp_ui.set_watchDir()

    # Ask work dir
    ui_ret = rtp_ui.set_workDir()

    # Application exec
    sys.excepthook = excepthook
    try:
        rtp_ui.show()
        exit_code = app.exec_()

    except Exception as e:
        with open('rtpspy.error', 'w') as fd:
            fd.write(str(e))

        print(str(e))
        exit_code = -1

    # End
    # Save parameters
    # exit with 1 means exit wihtout saving (defined in RtpGUI)
    if exit_code != 1:
        save_parameters(all_rtp_objs)

    sys.exit(exit_code)
