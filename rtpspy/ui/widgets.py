# -*- coding: utf-8 -*-
"""
Dialogs and widgets shared by the GUI panels

@author: mmisaki@laureateinstitute.org
"""

# %% import ===================================================================
from datetime import timedelta
import subprocess
import time

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import matplotlib as mpl
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

# Font size of the plot windows
mpl.rcParams['font.size'] = 8


# %% DlgProgressBar ===========================================================
class DlgProgressBar(QtWidgets.QDialog):
    """
    Progress bar dialog
    """

    class EmittingStream(QtCore.QObject):
        textWritten = QtCore.pyqtSignal(str)

        def write(self, text):
            self.textWritten.emit(str(text))

        def flush(self):
            pass

    def __init__(self, title='Progress', modal=True, parent=None,
                 win_size=(640, 320), show_button=True, st_time=None):
        super().__init__(parent)

        self.resize(*win_size)
        self.setWindowTitle(title)
        self.setModal(modal)
        vBoxLayout = QtWidgets.QVBoxLayout(self)

        # progress bar
        self.progBar = QtWidgets.QProgressBar(self)
        vBoxLayout.addWidget(self.progBar)

        # message text
        self.msgTxt = QtWidgets.QLabel()
        vBoxLayout.addWidget(self.msgTxt)

        # console output
        self.desc_pTxtEd = QtWidgets.QPlainTextEdit(self)
        self.desc_pTxtEd.setReadOnly(True)
        self.desc_pTxtEd.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        vBoxLayout.addWidget(self.desc_pTxtEd)

        # Cancel button
        bottomhLayout = QtWidgets.QHBoxLayout()
        bottomhLayout.addStretch()
        self.btnCancel = QtWidgets.QPushButton('Cancel')
        self.btnCancel.clicked.connect(self.close)
        self.btnCancel.setVisible(show_button)
        bottomhLayout.addWidget(self.btnCancel)
        vBoxLayout.addLayout(bottomhLayout)

        self.st_time = st_time
        self.title = title
        # Stream object that writes to the console text box (e.g., to
        # redirect sys.stdout)
        self.ostream = DlgProgressBar.EmittingStream(
            textWritten=self.update_console)

        self.show()

    def update_console(self, text):
        self.add_desc(text)

    def set_value(self, val):
        """
        Set the progress bar value (%). If st_time is set, the estimated
        remaining time is shown in the window title.
        """
        self.progBar.setValue(int(val))
        if self.st_time is not None:
            progress = self.progBar.value()
            if progress > 0:
                elapsed = time.time() - self.st_time
                remain = (elapsed/progress) * (100-progress)
                remain_str = str(timedelta(seconds=remain)).split('.')[0]
                self.setWindowTitle(f"{self.title} (ETA {remain_str})")

        self.repaint()
        QtWidgets.QApplication.instance().processEvents()

    def set_msgTxt(self, msg):
        self.msgTxt.setText(msg)
        self.repaint()
        QtWidgets.QApplication.instance().processEvents()

    def add_msgTxt(self, msg):
        self.msgTxt.setText(self.msgTxt.text()+msg)
        self.repaint()
        QtWidgets.QApplication.instance().processEvents()

    def add_desc(self, txt):
        self.desc_pTxtEd.moveCursor(QtGui.QTextCursor.End)
        self.desc_pTxtEd.insertPlainText(txt)

        sb = self.desc_pTxtEd.verticalScrollBar()
        sb.setValue(sb.maximum())

        self.repaint()
        QtWidgets.QApplication.instance().processEvents()

    def proc_print_progress(self, proc, bar_inc=None, ETA=None):
        """
        Show the output of a subprocess (proc, with stdout=PIPE) in the
        console text box until it ends. If bar_inc is set, the progress bar
        is advanced by up to bar_inc over the expected time ETA (seconds).
        Closing the dialog terminates proc (see closeEvent).

        Returns
        -------
        int or None
            Return code of proc.
        """
        if bar_inc is not None:
            bar_val0 = self.progBar.value()

        self.running_proc = proc
        st = time.time()
        while proc.poll() is None:
            try:
                out0 = proc.stdout.read(4)
                try:
                    out0 = out0.decode()
                except UnicodeDecodeError:
                    out0 = '\n'
                out = '\n'.join(out0.splitlines())
                if len(out0) and out0[-1] == '\n':
                    out += '\n'
                self.add_desc(out)

                if bar_inc is not None:
                    bar_inc0 = min((time.time() - st) / ETA * bar_inc,
                                   bar_inc)
                    if np.floor(bar_inc0+bar_val0) != self.progBar.value():
                        self.set_value(int(bar_inc0+bar_val0))

                QtWidgets.QApplication.instance().processEvents()
                time.sleep(0.001)

            except subprocess.TimeoutExpired:
                pass

            if not self.isVisible():
                break

        try:
            out = proc.stdout.read()
            try:
                out = out.decode()
                out = '\n'.join(out.splitlines()) + '\n\n'
                self.add_desc(out)
            except UnicodeDecodeError:
                pass

        except subprocess.TimeoutExpired:
            pass

        if bar_inc is not None:
            self.set_value(bar_inc+bar_val0)
            QtWidgets.QApplication.instance().processEvents()
            time.sleep(0.001)

        self.running_proc = None
        return proc.returncode

    def closeEvent(self, event):
        if hasattr(self, 'running_proc') and self.running_proc is not None:
            if self.running_proc.poll() is None:
                self.running_proc.terminate()


# %% MatplotlibWindow =========================================================
class MatplotlibWindow(QtWidgets.QMainWindow):
    """Window with a matplotlib figure canvas (self.canvas)."""

    def __init__(self, figsize=[5, 3]):
        super().__init__()

        self.canvas = FigureCanvas(Figure(figsize=figsize))
        self.setCentralWidget(self.canvas)
