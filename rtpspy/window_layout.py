#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Default window layout of the RT-MRI GUI, fitted to the primary screen.

The windows are placed in the usable area of the primary monitor (the
monitor area minus panels such as the GNOME top bar). The browser and
motion windows are stacked at the left edge, the physio window is next to
the browser, and the small log window (main) is at the bottom right.

    +-------------+-----------+----------------+
    | MRI browser |           |                |
    |             |  Physio   |                |
    +-------------+           |                |
    | Motion      +-----------+                |
    +-------------+              +-------------+
    |                            | Log (main)  |
    +----------------------------+-------------+

Each process calls default_geometry(name) for its own window, so all
windows follow the same layout without passing positions between them.
A --geometry option of each process still overrides this.

mmisaki@laureateinstitute.org
"""

# %% import ===================================================================
import subprocess

# Fractions of the usable area
LEFT_WIDTH = 0.364      # Width of the browser and motion column
BROWSER_HEIGHT = 0.366  # Heights in the left column
MOTION_HEIGHT = 0.398
PHYSIO_WIDTH = 0.212    # Physio window, next to the browser
PHYSIO_HEIGHT = 0.366
LOG_WIDTH = 0.35        # Log window at the bottom right
LOG_HEIGHT = 0.25

# Height of a window title bar. Tk sets the size of the client area and the
# position of the frame, so the title bar is subtracted from each height.
TITLE_BAR = 37

# Minimum client sizes so that the widgets remain usable on small screens
MIN_SIZE = {
    "browser": (560, 340),
    "motion": (560, 280),
    "physio": (360, 300),
    "main": (500, 180),
}


# %% _primary_monitor =========================================================
def _primary_monitor():
    """Return (x, y, width, height) of the primary monitor."""
    try:
        from screeninfo import get_monitors

        monitors = get_monitors()
        if monitors:
            prim = [m for m in monitors if getattr(m, "is_primary", False)]
            if not prim:
                prim = [m for m in monitors if m.x == 0 and m.y == 0]
            mon = prim[0] if prim else monitors[0]
            return mon.x, mon.y, mon.width, mon.height
    except Exception:
        pass

    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return 0, 0, w, h
    except Exception:
        return 0, 0, 1920, 1080


# %% _work_area ===============================================================
def _work_area():
    """Return (x, y, width, height) of the area not covered by panels
    (_NET_WORKAREA of the window manager), or None if not available."""
    try:
        out = subprocess.run(
            ["xprop", "-root", "_NET_WORKAREA"],
            capture_output=True, text=True, timeout=2,
        ).stdout
        vals = [int(v) for v in out.split("=")[1].split(",")[:4]]
        return tuple(vals)
    except Exception:
        return None


# %% screen_area ==============================================================
def screen_area():
    """Return (x, y, width, height) of the usable area of the primary
    monitor."""
    mx, my, mw, mh = _primary_monitor()
    wa = _work_area()
    if wa is None:
        return mx, my, mw, mh

    # _NET_WORKAREA can cover all monitors. Use its intersection with the
    # primary monitor.
    wx, wy, ww, wh = wa
    x0, y0 = max(mx, wx), max(my, wy)
    x1, y1 = min(mx + mw, wx + ww), min(my + mh, wy + wh)
    if x1 - x0 < mw * 0.5 or y1 - y0 < mh * 0.5:
        return mx, my, mw, mh
    return x0, y0, x1 - x0, y1 - y0


# %% layout ===================================================================
def layout():
    """Return {window name: (width, height, x, y)} of the client areas."""
    x0, y0, W, H = screen_area()

    def size(name, w, h):
        min_w, min_h = MIN_SIZE[name]
        return max(int(w), min_w), max(int(h) - TITLE_BAR, min_h)

    geoms = {}
    # Left column, top to bottom
    bw, bh = size("browser", W * LEFT_WIDTH, H * BROWSER_HEIGHT)
    geoms["browser"] = (bw, bh, x0, y0)
    mw, mh = size("motion", W * LEFT_WIDTH, H * MOTION_HEIGHT)
    geoms["motion"] = (mw, mh, x0, y0 + bh + TITLE_BAR)

    # Physio: right of the browser
    pw, ph = size("physio", W * PHYSIO_WIDTH, H * PHYSIO_HEIGHT)
    geoms["physio"] = (pw, ph, x0 + bw, y0)

    # Log: bottom right
    lw, lh = size("main", W * LOG_WIDTH, H * LOG_HEIGHT)
    geoms["main"] = (lw, lh, x0 + W - lw, y0 + H - lh - TITLE_BAR)

    return geoms


# %% default_geometry =========================================================
def default_geometry(name):
    """Return a Tk geometry string, 'WxH+X+Y', for the window name:
    'browser', 'motion', 'physio' or 'main' (the log window)."""
    w, h, x, y = layout()[name]
    return f"{w}x{h}+{x}+{y}"


# %% __main__ =================================================================
if __name__ == "__main__":
    print("Screen area:", screen_area())
    for name in MIN_SIZE:
        print(f"{name}: {default_geometry(name)}")


# %% RTPSpy layout ============================================================
# Windows of RTPSpy (RtpGUI and the plots, and rt_physio started by
# RtpTTLPhysio), side by side on the primary monitor within HDTV size:
#
#    +--------+----------+------------+
#    |        | Physio   | Latency    |
#    |  Main  +----------+------------+
#    |        |          | ROI signal |
#    |        | Motion   +------------+
#    +--------+----------+
#
# The positions depend only on the primary monitor (not on where a window
# happens to be), so rt_physio can be started at its place before the main
# window opens.
RTPSPY_HDTV = (1920, 1080)
RTPSPY_GAP = 6
RTPSPY_MAIN_WIDTH = 620        # Width of the main window slot (client)
RTPSPY_COLUMN_WIDTHS = (440, 440)
# (slot, client height); None: the rest of the column ('roi' is computed)
RTPSPY_COLUMNS = ((("physio", 377), ("motion", 539)),
                  (("latency", 377), ("roi", None)))
RTPSPY_ROI_HEIGHT = (80, 70)   # ROI plot height = a + b * number of ROIs


def rtpspy_area():
    """Usable area (x, y, w, h) of the primary monitor within HDTV size."""
    x0, y0, W, H = screen_area()
    mx, my, _, _ = _primary_monitor()
    x1 = min(x0 + W, mx + RTPSPY_HDTV[0])
    y1 = min(y0 + H, my + RTPSPY_HDTV[1])
    return x0, y0, x1 - x0, y1 - y0


def rtpspy_layout(main_width=None, num_ROIs=1, area=None):
    """
    Client rectangles {name: (x, y, w, h)} of the RTPSpy windows: 'main',
    'physio', 'motion', 'roi', 'latency'.

    Parameters
    ----------
    main_width : int, optional
        Width of the main window if it is wider than RTPSPY_MAIN_WIDTH.
    num_ROIs : int
        Number of ROIs (the height of the ROI signal plot).
    area : tuple, optional
        (x, y, w, h) to use instead of rtpspy_area().
    """
    x0, y0, W, H = area if area is not None else rtpspy_area()
    title = TITLE_BAR
    main_w = max(RTPSPY_MAIN_WIDTH, main_width or 0)
    rects = {"main": (x0, y0 + title, main_w, H - title)}

    left = x0 + main_w + RTPSPY_GAP
    right = x0 + W
    tot = sum(RTPSPY_COLUMN_WIDTHS)
    scale = min(1.0, (right - left - RTPSPY_GAP) / tot)
    pref = {"roi": RTPSPY_ROI_HEIGHT[0] + RTPSPY_ROI_HEIGHT[1] * num_ROIs}

    x = left
    for column, cw in zip(RTPSPY_COLUMNS, RTPSPY_COLUMN_WIDTHS):
        col_w = int(cw * scale)
        x = min(x, right - col_w)  # On the monitor (overlaps if too small)
        n = len(column)
        avail_h = H - n * title - (n - 1) * RTPSPY_GAP
        hs = [h if h is not None else pref.get(slot) for slot, h in column]
        fixed = sum(h for h in hs if h is not None)
        n_rest = sum(h is None for h in hs)
        min_rest = 150 * n_rest
        if fixed + min_rest > avail_h:  # Scale down to fit the column
            s = max(avail_h - min_rest, 0) / fixed
            hs = [None if h is None else int(h * s) for h in hs]
            fixed = sum(h for h in hs if h is not None)
        rest = (avail_h - fixed) // n_rest if n_rest else 0
        y = y0 + title
        for (slot, _), h in zip(column, hs):
            h = rest if h is None else h
            rects[slot] = (x, y, col_w, h)
            y += h + RTPSPY_GAP + title
        x += col_w + RTPSPY_GAP
    return rects


def rtpspy_tk_geometry(name, **kwargs):
    """Tk geometry 'WxH+X+Y' of an RTPSpy window (Tk places the frame with
    the title bar at +X+Y; WxH is the client size)."""
    x, y, w, h = rtpspy_layout(**kwargs)[name]
    return f"{w}x{h}+{x}+{y - TITLE_BAR}"
