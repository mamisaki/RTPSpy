#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Placement of the RTPSpy Qt windows (the main window and the plots) by the
layout of window_layout.rtpspy_layout(): side by side without overlap on
the primary monitor, within HDTV size (1920x1080).

    +--------+----------+------------+
    |        | Physio   | Latency    |
    |  Main  +----------+------------+
    |        |          | ROI signal |
    |        | Motion   +------------+
    +--------+----------+

The positions depend only on the primary monitor, so rt_physio (a separate
Tk process) is started at its place (RtpTTLPhysio passes --geometry) and
does not have to be moved after the main window opens.

The external application (e.g., the PsychoPy feedback window) is not
placed here; it is shown on another screen (its --screen option).

mmisaki@laureateinstitute.org
"""

# %% import ===================================================================
from ..window_layout import rtpspy_layout, rtpspy_tk_geometry


# %% helpers ==================================================================
def _main_width(main_win):
    """Minimum width of the main window (the slot is widened if the window
    cannot be as narrow as the slot)."""
    if main_win is None:
        return None
    return main_win.minimumSizeHint().width()


# %% place_main ===============================================================
def place_main(main_win):
    """Set the main window to its slot at the top left of the primary
    monitor (call before and after show(); the window manager may move a
    window when it is shown)."""
    x, y, w, h = rtpspy_layout(main_width=_main_width(main_win))["main"]
    h = max(h, main_win.minimumSizeHint().height())
    main_win.setGeometry(x, y, w, h)


# %% geometry =================================================================
def geometry(main_win, name, num_ROIs=1):
    """Client rectangle (x, y, w, h) of window name ('physio', 'motion',
    'roi', 'latency')."""
    return rtpspy_layout(main_width=_main_width(main_win),
                         num_ROIs=num_ROIs)[name]


def set_geometry(widget, main_win, name, **kwargs):
    """setGeometry of a Qt window to its slot."""
    widget.setGeometry(*geometry(main_win, name, **kwargs))


def tk_geometry(main_win, name):
    """Tk geometry string 'WxH+X+Y' of the slot (for rt_physio)."""
    return rtpspy_tk_geometry(name, main_width=_main_width(main_win))
