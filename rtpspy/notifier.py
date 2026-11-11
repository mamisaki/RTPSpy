# -*- coding: utf-8 -*-
"""
Front-end hooks of the processing modules

The processing modules report errors to the user and ask the user for input
(a file, a number, a confirmation) through a notifier, so that they also run
without a GUI. They never import a GUI toolkit.

Notifier, the default, writes the messages to the log and answers every
request with its cancel/default value, so that a headless run never blocks.
A GUI installs its own notifier with set_notifier(): RtpGUI installs
rtpspy.ui.QtNotifier, which shows Qt dialogs. When no notifier is installed
and a Qt application is running (e.g., a script that made a QApplication
without RtpGUI), the Qt notifier is used, as in earlier versions.

@author: mmisaki@laureateinstitute.org
"""

# %% import ===================================================================
import sys
import logging


# %% Notifier =================================================================
class Notifier:
    """Headless front end: messages go to the log, requests are canceled."""

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def error(self, msg, title=None, parent=None):
        """Report an error. The callers log the error themselves, so it is
        logged here only at the DEBUG level."""
        logging.getLogger(title or "Notifier").debug(f"Error notice: {msg}")

    def info(self, msg, title=None, parent=None):
        """Report information."""
        logging.getLogger(title or "Notifier").debug(f"Notice: {msg}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def question(self, text, title="", informative="", buttons=("Yes", "No"),
                 default="No", icon="question", parent=None):
        """Ask the user to select one of buttons; returns the selected
        button label. The default answer is returned without a GUI."""
        return default

    def get_open_file(self, caption, directory, filt, parent=None):
        """Select an existing file. Returns (filename, filter); filename is
        '' if canceled."""
        return "", ""

    def get_save_file(self, caption, directory, filt, parent=None):
        """Select a file to save. Returns (filename, filter); filename is ''
        if canceled."""
        return "", ""

    def get_directory(self, caption, directory, parent=None):
        """Select a directory. Returns '' if canceled."""
        return ""

    def get_int(self, title, label, *args, parent=None):
        """Enter an integer. args are (value, min, max) as in
        QInputDialog.getInt. Returns (value, ok)."""
        return (args[0] if len(args) else 0), False

    def get_text(self, label, text="", parent=None):
        """Edit a text. Returns (text, ok)."""
        return text, False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def progress_dialog(self, **kwargs):
        """Open a progress dialog (see rtpspy.ui.DlgProgressBar for the
        arguments). Returns None without a GUI."""
        return None

    def process_events(self):
        """Let the GUI process the pending events (e.g., to update the
        display during a long process)."""
        pass


# %% Installed notifier =======================================================
_DEFAULT_NOTIFIER = Notifier()
_notifier = None


def set_notifier(notifier):
    """Install the notifier used by all modules (None: automatic)."""
    global _notifier
    _notifier = notifier


def get_notifier():
    """Return the installed notifier. Without one, the Qt notifier is used
    if a Qt application is running, else the default (headless) one."""
    if _notifier is not None:
        return _notifier

    # Check for a running Qt application without importing Qt
    qtw = sys.modules.get("PyQt6.QtWidgets")
    if qtw is not None and qtw.QApplication.instance() is not None:
        from .ui.qt_notifier import QtNotifier
        return QtNotifier()

    return _DEFAULT_NOTIFIER


def default_notifier():
    """Return the headless notifier."""
    return _DEFAULT_NOTIFIER
