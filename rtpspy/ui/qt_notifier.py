# -*- coding: utf-8 -*-
"""
Qt implementation of the notifier (see rtpspy.notifier)

@author: mmisaki@laureateinstitute.org
"""

# %% import ===================================================================
import sys

from PyQt6 import QtWidgets

from ..notifier import Notifier

_ICONS = {
    "question": QtWidgets.QMessageBox.Icon.Question,
    "warning": QtWidgets.QMessageBox.Icon.Warning,
    "critical": QtWidgets.QMessageBox.Icon.Critical,
    "information": QtWidgets.QMessageBox.Icon.Information,
}


# %% QtNotifier ===============================================================
class QtNotifier(Notifier):
    """Show the messages and requests in Qt dialogs."""

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def error(self, msg, title=None, parent=None):
        # No parent by default, since this can be called from another
        # thread.
        msgBox = QtWidgets.QMessageBox(parent)
        msgBox.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        msgBox.setText(str(msg))
        msgBox.setWindowTitle(title or "")
        msgBox.exec()

    def info(self, msg, title=None, parent=None):
        QtWidgets.QMessageBox.information(
            parent, title or "", str(msg),
            QtWidgets.QMessageBox.StandardButton.Ok)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def question(self, text, title="", informative="", buttons=("Yes", "No"),
                 default="No", icon="question", parent=None):
        msgBox = QtWidgets.QMessageBox(parent)
        msgBox.setIcon(_ICONS[icon])
        msgBox.setText(text)
        if informative:
            msgBox.setInformativeText(informative)
        msgBox.setWindowTitle(title)
        std_btns = {b: getattr(QtWidgets.QMessageBox, b) for b in buttons}
        btns = QtWidgets.QMessageBox.StandardButton.NoButton
        for b in std_btns.values():
            btns |= b
        msgBox.setStandardButtons(btns)
        msgBox.setDefaultButton(std_btns[default])
        # exec() returns an int in PyQt6; compare it as a StandardButton
        ret = QtWidgets.QMessageBox.StandardButton(msgBox.exec())
        for b, val in std_btns.items():
            if ret == val:
                return b
        return default

    def get_open_file(self, caption, directory, filt, parent=None):
        if sys.platform == "darwin":
            return QtWidgets.QFileDialog.getOpenFileName(
                parent, caption, str(directory), filt)

        options = QtWidgets.QFileDialog.Option(0)
        options |= QtWidgets.QFileDialog.Option.DontUseNativeDialog
        return QtWidgets.QFileDialog.getOpenFileName(
            parent, caption, str(directory), filt, options=options)

    def get_save_file(self, caption, directory, filt, parent=None):
        return QtWidgets.QFileDialog.getSaveFileName(
            parent, caption, str(directory), filt)

    def get_directory(self, caption, directory, parent=None):
        return QtWidgets.QFileDialog.getExistingDirectory(
            parent, caption, str(directory))

    def get_int(self, title, label, *args, parent=None):
        return QtWidgets.QInputDialog.getInt(parent, title, label, *args)

    def get_text(self, label, text="", parent=None):
        dlg = QtWidgets.QInputDialog(parent)
        dlg.setInputMode(QtWidgets.QInputDialog.InputMode.TextInput)
        dlg.setLabelText(label)
        dlg.setTextValue(text)
        dlg.resize(640, 100)
        okflag = dlg.exec()
        return dlg.textValue(), okflag

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def progress_dialog(self, **kwargs):
        from .widgets import DlgProgressBar
        return DlgProgressBar(**kwargs)

    def process_events(self):
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents()
