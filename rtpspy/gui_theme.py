#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Color theme of the RT-MRI GUI.

Two themes: 'light' (default) and 'dark' (neon accents on a dark
background). Set the environment variable RTPSPY_THEME=dark to use the dark
theme. rt_physio (Tk) and the RTPSpy GUI (Qt) use the same theme.

All windows use the colors defined here:
- apply_tk(root): Tk option database and ttk styles (rt_physio). Call it
  right after making the Tk root, before making widgets (the option database
  applies to widgets made after it).
- apply_qt(app): Qt palette and style sheet (RTPSpy GUI). RtpGUI calls it.
- qt_button_style(kind): style sheet of an emphasized Qt button.
- apply_mpl(): matplotlib rcParams. Call it before making figures.
- COLORS: colors used directly in the code (plot lines, alarm marks, etc.).

mmisaki@laureateinstitute.org
"""

# Tk and Qt are imported in the functions that use them, so that each GUI
# (rt_physio with Tk, RTPSpy with Qt) does not load the other.

# %% Colors ===================================================================
import os

THEMES = {
    # Dark accents on a soft light background
    "light": {
        # Soft gray (not white) backgrounds, to reduce the glare
        "bg": "#dfe3ea",        # Window background
        "panel": "#d0d6e0",     # Controls (buttons, entries)
        "panel_hi": "#c0c9d8",  # Active (hover) controls
        "plot_bg": "#eef1f5",   # Plot area, text fields
        "edge": "#8392ae",      # Frames, plot spines
        "grid": "#cdd4e0",
        "fg": "#141a26",        # Text
        "dim": "#46516b",       # Secondary text, ticks
        "cyan": "#005f99",
        "magenta": "#a8134f",
        "yellow": "#a86f00",
        "green": "#0f6e3b",
        "red": "#c01e2e",       # Errors and alarms
        "purple": "#56309a",
        # Filled buttons (qt_button_style): yellow fill and the text colors
        "yellow_fill": "#e6ad35",
        "on_yellow": "#141a26",
        "on_red": "#ffffff",
        "on_purple": "#ffffff",
    },
    # Neon accents on a dark background
    "dark": {
        "bg": "#0a0e1a",
        "panel": "#111a2e",
        "panel_hi": "#1c2a4a",
        "plot_bg": "#0d1324",
        "edge": "#2a3a66",
        "grid": "#1a2544",
        "fg": "#c9d4ff",
        "dim": "#6b7aa6",
        "cyan": "#00e5ff",
        "magenta": "#ff2bd6",
        "yellow": "#ffe14d",
        "green": "#3dff9a",
        "red": "#ff3864",
        "purple": "#9d4dff",
        "yellow_fill": "#ffe14d",
        "on_yellow": "#0a0e1a",
        "on_red": "#0a0e1a",
        "on_purple": "#c9d4ff",
    },
}

THEME = os.environ.get("RTPSPY_THEME", "light").lower()
if THEME not in THEMES:
    THEME = "light"
COLORS = dict(THEMES[THEME])

# Line colors of the plots, in the order of use
LINE_CYCLE = [
    COLORS["cyan"], COLORS["magenta"], COLORS["yellow"], COLORS["green"],
    COLORS["purple"], COLORS["red"],
]


# %% apply_tk =================================================================
def apply_tk(root):
    """Set the colors of Tk and ttk widgets made after this call."""
    from tkinter import ttk

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


# %% apply_qt =================================================================
def apply_qt(app):
    """Set the Qt palette and style sheet of the application (QApplication)
    and the matplotlib colors, to match the rt_physio window."""
    from PyQt6 import QtGui, QtWidgets

    c = COLORS
    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

    R = QtGui.QPalette.ColorRole
    G = QtGui.QPalette.ColorGroup
    pal = QtGui.QPalette()
    roles = {
        R.Window: c["bg"], R.WindowText: c["fg"], R.Base: c["plot_bg"],
        R.AlternateBase: c["panel"], R.Text: c["fg"], R.Button: c["panel"],
        R.ButtonText: c["cyan"], R.BrightText: c["yellow"],
        R.Highlight: c["magenta"], R.HighlightedText: c["bg"],
        R.ToolTipBase: c["panel"], R.ToolTipText: c["fg"],
        R.PlaceholderText: c["dim"], R.Link: c["cyan"],
        R.Light: c["panel_hi"], R.Midlight: c["panel_hi"], R.Mid: c["edge"],
        R.Dark: c["bg"], R.Shadow: "#000000",
    }
    for role, col in roles.items():
        pal.setColor(role, QtGui.QColor(col))
    for role in (R.WindowText, R.Text, R.ButtonText):
        pal.setColor(G.Disabled, role, QtGui.QColor(c["dim"]))
    app.setPalette(pal)

    app.setStyleSheet(f"""
QToolTip {{ color: {c['fg']}; background-color: {c['panel']};
           border: 1px solid {c['edge']}; }}
QGroupBox {{ border: 1px solid {c['edge']}; border-radius: 4px;
            margin-top: 1.1em; padding-top: 4px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 3px;
                   color: {c['dim']}; }}
QPushButton {{ background-color: {c['panel']}; color: {c['cyan']};
              border: 1px solid {c['edge']}; border-radius: 3px;
              padding: 3px 10px; }}
QPushButton:hover {{ background-color: {c['panel_hi']};
                    border-color: {c['cyan']}; }}
QPushButton:pressed {{ background-color: {c['edge']}; }}
QPushButton:disabled {{ color: {c['dim']}; border-color: {c['grid']}; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit,
QListWidget, QTableWidget, QTreeWidget {{
    background-color: {c['plot_bg']}; color: {c['fg']};
    border: 1px solid {c['edge']}; border-radius: 2px;
    selection-background-color: {c['magenta']};
    selection-color: {c['bg']}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QComboBox:disabled {{ color: {c['dim']}; }}
QComboBox QAbstractItemView {{ background-color: {c['panel']};
    color: {c['fg']}; selection-background-color: {c['magenta']}; }}
QTabWidget::pane {{ border: 1px solid {c['edge']}; top: -1px; }}
QTabBar::tab {{ background-color: {c['panel']}; color: {c['dim']};
               border: 1px solid {c['edge']}; border-bottom: none;
               padding: 4px 12px; margin-right: 1px; }}
QTabBar::tab:selected {{ background-color: {c['panel_hi']};
                        color: {c['cyan']}; }}
QTabBar::tab:hover {{ color: {c['fg']}; }}
QMenuBar {{ background-color: {c['bg']}; color: {c['fg']}; }}
QMenuBar::item:selected, QMenu::item:selected {{
    background-color: {c['panel_hi']}; color: {c['cyan']}; }}
QMenu {{ background-color: {c['panel']}; color: {c['fg']};
        border: 1px solid {c['edge']}; }}
QProgressBar {{ background-color: {c['panel']}; color: {c['fg']};
               border: 1px solid {c['edge']}; border-radius: 2px;
               text-align: center; }}
QProgressBar::chunk {{ background-color: {c['cyan']}; }}
QSplitter::handle {{ background-color: {c['edge']}; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 12px; height: 12px;
    border: 1px solid {c['dim']}; background-color: {c['plot_bg']}; }}
QRadioButton::indicator {{ border-radius: 7px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border-color: {c['cyan']}; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {c['cyan']}; border-color: {c['cyan']}; }}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    border-color: {c['grid']}; }}
QCheckBox::indicator:checked:disabled,
QRadioButton::indicator:checked:disabled {{ background-color: {c['dim']}; }}
QScrollArea {{ border: none; }}
""")
    apply_mpl()


# %% qt_button_style ==========================================================
# Emphasized buttons: (background, text) colors
_QT_BUTTONS = {
    "accent": None,                        # Outlined (e.g., 'Set' buttons)
    "selected": ("yellow_fill", "on_yellow"),  # Selected (the app panel)
    "ready": ("yellow_fill", "on_yellow"),     # Ready
    "start": ("purple", "on_purple"),      # Manual start
    "stop": ("red", "on_red"),             # Cancel, quit, delete
}


def qt_button_style(kind="accent", extra=""):
    """
    Style sheet of an emphasized QPushButton (set with setStyleSheet).

    Parameters
    ----------
    kind : str
        'accent' (outlined in cyan), 'selected', 'ready', 'start', or
        'stop' (filled with the theme color).
    extra : str
        Additional declarations, e.g., 'height: 20px;'.
    """
    c = COLORS
    if _QT_BUTTONS[kind] is None:
        return (
            f"QPushButton {{ background-color: {c['panel']};"
            f" color: {c['cyan']}; border: 1px solid {c['cyan']}; {extra} }}"
            f" QPushButton:hover {{ background-color: {c['panel_hi']}; }}"
            f" QPushButton:disabled {{ color: {c['dim']};"
            f" border-color: {c['grid']}; }}"
        )
    bg, fg = (c[k] for k in _QT_BUTTONS[kind])
    return (
        f"QPushButton {{ background-color: {bg}; color: {fg};"
        f" border: 1px solid {bg}; font-weight: bold; {extra} }}"
        f" QPushButton:hover {{ border-color: {c['fg']}; }}"
        f" QPushButton:disabled {{ background-color: {c['panel']};"
        f" color: {c['dim']}; border-color: {c['grid']};"
        " font-weight: normal; }"
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
