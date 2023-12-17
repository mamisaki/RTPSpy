#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run left amygdala neurofeedback (LA-NF) application

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import sys
from platform import uname

import numpy as np
from PyQt5 import QtWidgets

# RTP application and utility functions and classes
from rtpspy.rtp_common import excepthook, save_parameters, load_parameters
from rtpspy import RTP_UI
from la_nf import LANF


# %% Default parameters
# External NF application
this_dir = Path(__file__).absolute().parent
if 'Microsoft' in uname().release:
    cmd_path = r'/mnt/c/Program\ Files/PsychoPy3/pythonw.exe '
    cmd_path += str(this_dir / 'NF_psypy.py')
elif sys.platform == 'darwin':
    cmd_path = f"{this_dir / 'NF_psypy.py'}"
else:
    cmd_path = f"{this_dir / 'boot_psychopy.sh'} {this_dir / 'NF_psypy.py'}"
extApp_cmd = f"{cmd_path} --screen 0 --size 640 480 --pos 0 0"

rtp_params = {'VOLREG': {'regmode': 'heptic'},
              'TSHIFT': {'method': 'cubic', 'ref_time': 0},
              'SMOOTH': {'blur_fwhm': 6.0},
              'REGRESS': {'wait_num': 40, 'max_poly_order': np.inf,
                          'mot_reg': 'mot12', 'GS_reg': True, 'WM_reg': True,
                          'Vent_reg': True, 'phys_reg': 'RICOR8'},
              'APP': {'run_extApp': True,
                      'extApp_cmd': extApp_cmd,
                      'template': this_dir / 'MNI152_2009_template.nii.gz',
                      'ROI_template':
                          this_dir / 'MNI152_2009_template_LAmy.nii.gz',
                      'ROI_resample': 'NearestNeighbor',
                      'WM_template':
                          this_dir / 'MNI152_2009_template_WM.nii.gz',
                      'Vent_template':
                          this_dir / 'MNI152_2009_template_Vent.nii.gz',
                      'initDur': 96,  # seconds
                      'NFBlockDur': 40,
                      'CountBlockDur': 40,
                      'RestBlockDur': 40,
                      'NrBlockRep': 4,
                      'restDur': 480,
                      'NF_target_levels': {'Practice': 0.2, 'NF1': 0.4,
                                           'NF2': 0.8, 'NF3': 1.0},
                      'sham': False
                      }
              }


# %% main =====================================================================
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    # Make RtpApp instance
    rtp_app = LANF(default_rtp_params=rtp_params)
    rtp_app.run_extApp = True

    # Make RTP_UI instance
    app_obj = {'LA-NF': rtp_app}
    rtp_ui = RTP_UI(rtp_app.rtp_objs, app_obj, log_dir='./log')

    # Keep RTP objects for loading and saving the parameters
    all_rtp_objs = rtp_app.rtp_objs
    all_rtp_objs.update(app_obj)

    # Load saved parameters (without asking)
    all_rtp_objs = rtp_app.rtp_objs
    all_rtp_objs.update(app_obj)
    load_parameters(all_rtp_objs)

    # Ask watching dir
    ui_ret = rtp_ui.set_watchDir()

    # Ask working dir
    ui_ret = rtp_ui.set_workDir()

    # Run the application
    sys.excepthook = excepthook
    try:
        rtp_ui.show()
        exit_code = app.exec_()

    except Exception as e:
        with open('rtpspy.error', 'w') as fd:
            fd.write(str(e))

        print(str(e))
        exit_code = -1

    # --- End ---
    # Save parameters
    # exit with 1 means exit wihtout saving (defined in RTP_UI)
    if exit_code != 1:
        save_parameters(all_rtp_objs)

    sys.exit(exit_code)
