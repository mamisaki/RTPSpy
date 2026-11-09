#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Color theme of the RT-MRI GUI (dark, neon accents).

All windows use the colors defined here:
- apply_tk(root): Tk option database and ttk styles. Call it right after
  making the Tk root, before making widgets (the option database applies to
  widgets made after it).
- apply_mpl(): matplotlib rcParams. Call it before making figures.
- COLORS: colors used directly in the code (plot lines, alarm marks, etc.).

mmisaki@laureateinstitute.org
"""

# %% import ===================================================================
from tkinter import ttk

# %% Colors ===================================================================
COLORS = {
    "bg": "#0a0e1a",        # Window background
    "panel": "#111a2e",     # Controls (buttons, entries)
    "panel_hi": "#1c2a4a",  # Active (hover) controls
    "plot_bg": "#0d1324",   # Plot area
    "edge": "#2a3a66",      # Frames, plot spines
    "grid": "#1a2544",
    "fg": "#c9d4ff",        # Text
    "dim": "#6b7aa6",       # Secondary text, ticks
    "cyan": "#00e5ff",
    "magenta": "#ff2bd6",
    "yellow": "#ffe14d",
    "green": "#3dff9a",
    "red": "#ff3864",       # Errors and alarms
    "purple": "#9d4dff",
}

# Line colors of the plots, in the order of use
LINE_CYCLE = [
    COLORS["cyan"], COLORS["magenta"], COLORS["yellow"], COLORS["green"],
    COLORS["purple"], COLORS["red"],
]


# %% apply_tk =================================================================
def apply_tk(root):
    """Set the colors of Tk and ttk widgets made after this call."""
    c = COLORS
    opts = {
        "*Background": c["bg"],
        "*Foreground": c["fg"],
        "*activeBackground": c["panel_hi"],
        "*activeForeground": c["cyan"],
        "*highlightBackground": c["bg"],
        "*highlightColor": c["cyan"],
        "*selectBackground": c["magenta"],
        "*selectForeground": c["bg"],
        "*insertBackground": c["cyan"],
        "*troughColor": c["panel"],
        "*disabledForeground": c["dim"],
        "*Button.background": c["panel"],
        "*Button.foreground": c["cyan"],
        "*Button.relief": "flat",
        "*Button.highlightThickness": 1,
        "*Button.highlightBackground": c["edge"],
        "*Entry.background": c["plot_bg"],
        "*Entry.foreground": c["cyan"],
        "*Entry.relief": "flat",
        "*Entry.highlightThickness": 1,
        "*Entry.highlightBackground": c["edge"],
        "*Checkbutton.selectColor": c["plot_bg"],
        "*Menu.background": c["panel"],
        "*Menu.foreground": c["fg"],
        "*Menu.activeBackground": c["panel_hi"],
        "*Menu.activeForeground": c["cyan"],
        "*Menu.relief": "flat",
        "*Text.background": c["plot_bg"],
        "*Text.foreground": c["fg"],
        "*Canvas.background": c["bg"],
        "*Canvas.highlightThickness": 0,
        # Drop-down list of ttk.Combobox
        "*TCombobox*Listbox.background": c["panel"],
        "*TCombobox*Listbox.foreground": c["fg"],
        "*TCombobox*Listbox.selectBackground": c["magenta"],
        "*TCombobox*Listbox.selectForeground": c["bg"],
    }
    for pattern, value in opts.items():
        root.option_add(pattern, value)
    root.configure(background=c["bg"])

    style = ttk.Style(root)
    try:
        style.theme_use("clam")  # Allows setting the colors below
    except Exception:
        pass
    style.configure(
        ".", background=c["bg"], foreground=c["fg"],
        fieldbackground=c["plot_bg"], bordercolor=c["edge"],
        lightcolor=c["edge"], darkcolor=c["edge"], troughcolor=c["panel"],
        arrowcolor=c["cyan"], selectbackground=c["magenta"],
        selectforeground=c["bg"],
    )
    style.map(
        ".", background=[("active", c["panel_hi"])],
        foreground=[("disabled", c["dim"])],
    )
    style.configure(
        "TCombobox", fieldbackground=c["plot_bg"], foreground=c["cyan"],
        background=c["panel"],
    )
    style.map(
        "TCombobox", fieldbackground=[("readonly", c["plot_bg"])],
        foreground=[("readonly", c["cyan"])],
        selectbackground=[("readonly", c["plot_bg"])],
        selectforeground=[("readonly", c["cyan"])],
    )
    style.configure(
        "TProgressbar", background=c["cyan"], troughcolor=c["panel"],
        bordercolor=c["edge"],
    )
    for orient in ("Horizontal", "Vertical"):
        style.configure(
            f"{orient}.TScale", background=c["cyan"],
            troughcolor=c["panel"], bordercolor=c["edge"],
        )


# %% apply_mpl ================================================================
def apply_mpl():
    """Set matplotlib colors for figures made after this call."""
    import matplotlib as mpl
    from cycler import cycler

    c = COLORS
    mpl.rcParams.update({
        "figure.facecolor": c["bg"],
        "figure.edgecolor": c["bg"],
        "savefig.facecolor": c["bg"],
        "axes.facecolor": c["plot_bg"],
        "axes.edgecolor": c["edge"],
        "axes.labelcolor": c["fg"],
        "axes.prop_cycle": cycler(color=LINE_CYCLE),
        "grid.color": c["grid"],
        "text.color": c["fg"],
        "xtick.color": c["dim"],
        "ytick.color": c["dim"],
        "xtick.labelcolor": c["dim"],
        "ytick.labelcolor": c["dim"],
        "lines.linewidth": 1.2,
    })


# %% fix_mpl_canvas ===========================================================
def fix_mpl_canvas(mpl_canvas):
    """Set the theme background on the Tk widget of a FigureCanvasTkAgg.
    matplotlib makes it white, and it shows as white strips where the figure
    image does not cover the widget (e.g., when the Tk scaling is not 96 dpi
    or while resizing)."""
    mpl_canvas.get_tk_widget().configure(background=COLORS["bg"])
