# -*- coding: utf-8 -*-
"""
GUI front end of RTPSpy (PyQt5)

The processing modules (rtpspy.rtp_*) do not import Qt. Their GUI panels
are here, one module per processing module (watch_ui, volreg_ui, tshift_ui,
smooth_ui, regress_ui, app_ui), with the main window RtpGUI, the Qt
notifier for the messages and dialogs of the modules, and the shared dialogs
and widgets.
"""
from .qt_notifier import QtNotifier
from .widgets import DlgProgressBar, MatplotlibWindow
from .module_ui import RtpModuleUI, bind_ui
from .rtp_gui import RtpGUI

__all__ = ["QtNotifier", "DlgProgressBar", "MatplotlibWindow",
           "RtpModuleUI", "bind_ui", "RtpGUI"]
