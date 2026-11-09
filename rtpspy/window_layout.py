#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Default window layout of the RT-MRI GUI, fitted to the primary screen.

The windows are placed in the usable area of the primary monitor (the
monitor area minus panels such as the GNOME top bar). The browser and
motion windows are stacked at the left edge, the physio window is next to
the browser, and the log window (main) is small at the bottom right.

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
