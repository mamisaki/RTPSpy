# -*- coding: utf-8 -*-
"""
RtpGUI has moved to rtpspy.ui.rtp_gui. This module keeps the old import
path (from rtpspy.rtp_gui import RtpGUI) working.
"""
from .ui.rtp_gui import RtpGUI, GPU_available  # noqa: F401
