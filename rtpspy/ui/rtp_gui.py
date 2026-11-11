# -*- coding: utf-8 -*-
"""
RTPSpy graphical user interface: the main window of an RTPSpy application

The window shows the GUI panel of each RTP module and application (made by
their ui_set_param(); see rtpspy.ui.module_ui) and installs the Qt notifier
(rtpspy.ui.QtNotifier) for the messages and dialogs of the modules.

@author: mmisaki@laureateinstitute.org
"""

# %% import ===================================================================
from pathlib import Path
import os
import shutil
from functools import partial
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import logging

import torch
from PyQt6 import QtCore, QtWidgets, QtGui

from ..rtp_common import save_parameters, load_parameters
from ..notifier import set_notifier
from .. import gui_theme
from .qt_notifier import QtNotifier
from . import window_place

GPU_available = (
    torch.cuda.is_available() or torch.backends.mps.is_available()
)

# RTP modules shown in the RTP tab and the parameter list, in processing
# order
_RTP_PROCS = ("WATCH", "VOLREG", "TSHIFT", "SMOOTH", "REGRESS", "EXTSIG")

# Log console: font size (pt) and the minimum number of characters per line
LOG_FONT_SIZE = 9
LOG_MIN_COLUMNS = 80

# Style of the selected module button
_SELECTED_BTN_STYLE = gui_theme.qt_button_style("selected")


# %% _same_geometry ===========================================================
def _same_geometry(geo, target, pos_tol=60, size_tol=10):
    """True if Tk geometry strings 'WxH+X+Y' geo and target agree (the
    position within pos_tol, as the window frame offsets depend on the
    window manager)."""
    import re

    def parse(g):
        m = re.match(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)", g or "")
        return tuple(int(v) for v in m.groups()) if m else None

    a, b = parse(geo), parse(target)
    if a is None or b is None:
        return False
    return (abs(a[0] - b[0]) <= size_tol and abs(a[1] - b[1]) <= size_tol
            and abs(a[2] - b[2]) <= pos_tol and abs(a[3] - b[3]) <= pos_tol)


# %% RtpGUI class =============================================================
class RtpGUI(QtWidgets.QMainWindow):
    """Main window of an RTPSpy application."""

    def __init__(
        self,
        rtp_objs,
        rtp_apps,
        log_file=None,
        winTitle="RTPSpy",
        onGPU=GPU_available,
    ):
        """
        Parameters
        ----------
        rtp_objs : dict
            RTP module objects given by RtpApp.rtp_objs.
        rtp_apps : dict
            {application name: RtpApp (or derived class) object}.
        log_file : str or Path, optional
            Log file to display in the window. The default is None.
        winTitle : str, optional
            Window title. The default is 'RTPSpy'.
        onGPU : bool, optional
            Use the GPU. The default is GPU_available.

        """
        QtWidgets.QMainWindow.__init__(self)
        self._logger = logging.getLogger("RtpGUI")

        # Show the messages and dialogs of the modules in Qt dialogs
        self._notifier = QtNotifier()
        set_notifier(self._notifier)

        # Colors of the rt_physio window (gui_theme)
        gui_theme.apply_qt(QtWidgets.QApplication.instance())

        self.rtp_objs = rtp_objs
        self.rtp_apps = rtp_apps

        self.GPU_available = GPU_available
        if onGPU and not GPU_available:
            onGPU = False

        self.watchDir_start = Path.cwd()

        # Set up the main UI
        self.setWindowTitle(winTitle)
        self.mainWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.mainWidget)

        self.make_menu()
        self.make_ui(rtp_objs, onGPU)
        self.layout_ui()

        # Initialize view
        self.on_clicked_setOption("WATCH")
        self.options_tab.setCurrentIndex(0)

        # Set log
        self._log_fd = None
        self._log_update_timer = QtCore.QTimer()
        if log_file:
            self.set_log(log_file)

        self._logger.info("=== Start application ===\n")

        # Main window at the top left of the primary monitor; the other
        # windows are placed next to it (window_place.py). It is placed
        # before and after show(), as the window manager may move it.
        window_place.place_main(self)
        self.show()
        window_place.place_main(self)

        # rt_physio is started at its place (RtpTTLPhysio passes
        # --geometry). Check it and move it if the window manager placed it
        # elsewhere.
        self.ensure_physio_geometry()
        # Keep the 'Physio' box in sync with the physio window, which
        # rt_physio opens by itself and its close button hides
        self._start_physio_watch()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _start_physio_watch(self, interval_ms=1000):
        """
        Check the state of the physio window (GET_WINDOW_STATE RPC of
        rt_physio) every interval_ms and set the 'Physio' check box to it,
        without showing or hiding the window. The state is read in a worker
        thread, so that a slow RPC never blocks the GUI.
        """
        if (
            self.rtp_objs.get("TTLPHYSIO") is None
            or not hasattr(self, "chbShowPhysio")
            or getattr(self, "_physio_watch_timer", None) is not None
        ):
            return
        self._physio_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="physio_state")
        self._physio_query = None  # (submit time, future)
        self._physio_chk_time = 0.0  # Last change of the box by the user
        self._physio_watch_timer = QtCore.QTimer(self)
        self._physio_watch_timer.timeout.connect(self._watch_physio_state)
        self._physio_watch_timer.start(interval_ms)
        self._watch_physio_state()

    def _watch_physio_state(self):
        """Timer callback of _start_physio_watch: apply the last state read
        and start the next read."""
        ttl = self.rtp_objs.get("TTLPHYSIO")
        if ttl is None:
            return
        if self._physio_query is not None:
            t_submit, fut = self._physio_query
            if not fut.done():
                return
            try:
                state = fut.result()
            except Exception:
                state = None
            # A state read before the user's last click may be stale
            if state is not None and t_submit > self._physio_chk_time:
                checked = state in ("normal", "iconic")
                if self.chbShowPhysio.isChecked() != checked:
                    self.chbShowPhysio.blockSignals(True)
                    self.chbShowPhysio.setChecked(checked)
                    self.chbShowPhysio.blockSignals(False)
        self._physio_query = (
            time.monotonic(),
            self._physio_pool.submit(ttl.get_window_state))

    def _stop_physio_watch(self):
        timer = getattr(self, "_physio_watch_timer", None)
        if timer is not None:
            timer.stop()
            self._physio_watch_timer = None
            self._physio_pool.shutdown(wait=False, cancel_futures=True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def _scroll_area(widget):
        """Put widget in a vertical scroll area (shown only when the window
        is shorter than the widget)."""
        area = QtWidgets.QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        area.setWidget(widget)
        # Width of the content and the scroll bar, so that no horizontal
        # scroll is needed
        area.setMinimumWidth(
            widget.minimumSizeHint().width()
            + area.verticalScrollBar().sizeHint().width())
        return area

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ensure_physio_geometry(self, tries=10, interval_ms=500):
        """
        Check the physio window geometry (GET_GEOMETRY RPC of rt_physio)
        and send SET_GEOMETRY until it is at its place in the layout. A
        geometry sent before the window opens is ignored by rt_physio, so
        this is repeated every interval_ms, up to tries times.
        """
        ttl = self.rtp_objs.get("TTLPHYSIO")
        if ttl is None or tries <= 0:
            return
        target = window_place.tk_geometry(self, "physio")
        if not _same_geometry(ttl.get_geometry(), target):
            ttl.move(target)
            QtCore.QTimer.singleShot(
                interval_ms,
                lambda: self.ensure_physio_geometry(tries - 1, interval_ms))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_physio_geometry(self):
        """Place the physio window to the right of the main window
        (window_place.py). rt_physio ignores it while its plot window is not
        open, so show_physio_chk calls this again after opening it."""
        if (
            "TTLPHYSIO" in self.rtp_objs and
            self.rtp_objs["TTLPHYSIO"] is not None
        ):
            self.rtp_objs["TTLPHYSIO"].move(
                window_place.tk_geometry(self, "physio"))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def make_menu(self):
        """Make the File menu."""
        # Save parameters
        saveOptAct = QtGui.QAction(QtGui.QIcon("save.png"), "&Save", self)
        saveOptAct.setShortcut("Ctrl+S")
        saveOptAct.setStatusTip("Save current parameters")
        saveOptAct.triggered.connect(self.ui_save_parameters)

        # Load parameters
        loadOptAct = QtGui.QAction(QtGui.QIcon("load.png"), "&Load", self)
        loadOptAct.setShortcut("Ctrl+L")
        loadOptAct.setStatusTip("Load parameters")
        loadOptAct.triggered.connect(self.ui_load_parameters)

        # Exit without saving
        exitNosaveAct = QtGui.QAction(
            QtGui.QIcon("exit.png"), "Exit&WithoutSaving", self
        )
        exitNosaveAct.setShortcut("Ctrl+W")
        exitNosaveAct.setStatusTip(
            "Exit application without saving parameters"
        )
        exitNosaveAct.triggered.connect(
            lambda: QtWidgets.QApplication.instance().exit(1))

        # Exit
        exitAct = QtGui.QAction(QtGui.QIcon("exit.png"), "&Exit", self)
        exitAct.setShortcut("Ctrl+Q")
        exitAct.setStatusTip("Exit application")
        exitAct.triggered.connect(self.close)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu("&File")
        fileMenu.addAction(saveOptAct)
        fileMenu.addAction(loadOptAct)
        fileMenu.addAction(exitNosaveAct)
        fileMenu.addAction(exitAct)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def make_ui(self, rtp_objs, onGPU=GPU_available):
        """Make GUI controls.

        Parameters
        ----------
        rtp_objs : dict
            RTP module objects.
        onGPU : bool, optional
            Initial state of the 'GPU' checkbox.
        """

        # region: Watch/Work dir controls -------------------------------------
        # Watching directory
        self.labelWatchDir = QtWidgets.QLabel(self.mainWidget)
        self.labelWatchDir.setText("Watch directory")

        self.lineEditWatchDir = QtWidgets.QLineEdit(self.mainWidget)
        self.lineEditWatchDir.setReadOnly(True)
        self.lineEditWatchDir.setStyleSheet("border: 0px none;")

        self.btnSetWatchDir = QtWidgets.QPushButton("Set", self.mainWidget)
        self.btnSetWatchDir.clicked.connect(self.set_watchDir)

        # Working directory
        self.labelWorkDir = QtWidgets.QLabel(self.mainWidget)
        self.labelWorkDir.setText("Working directory")

        self.lineEditWorkDir = QtWidgets.QLineEdit(self.mainWidget)
        self.lineEditWorkDir.setReadOnly(True)
        self.lineEditWorkDir.setStyleSheet("border: 0px none;")

        self.btnSetWorkDir = QtWidgets.QPushButton("Set", self.mainWidget)
        self.btnSetWorkDir.clicked.connect(self.set_workDir)
        # endregion: Watch/Work dir controls ----------------------------------

        # region: Devices and plot checkbox -----------------------------------
        # Show motion
        if "VOLREG" in rtp_objs:
            self.chbShowMotion = QtWidgets.QCheckBox(
                "Motion", self.mainWidget
            )
            self.chbShowMotion.setChecked(False)
            self.chbShowMotion.stateChanged.connect(
                lambda x: self.show_mot_chk(x)
            )

        # Show Physio
        if (
            "TTLPHYSIO" in rtp_objs and
            self.rtp_objs["TTLPHYSIO"] is not None
        ):
            self.chbShowPhysio = QtWidgets.QCheckBox(
                "Physio", self.mainWidget
            )
            self.chbShowPhysio.setChecked(False)
            self.chbShowPhysio.stateChanged.connect(
                lambda x: self.show_physio_chk(x)
            )

        # GPU
        if self.GPU_available:
            if torch.cuda.is_available():
                dev_name = torch.cuda.get_device_name(
                    torch.cuda.current_device()
                )
            elif torch.backends.mps.is_available():
                dev_name = torch.device("mps")
            self.chbUseGPU = QtWidgets.QCheckBox(
                "GPU", self.mainWidget
            )
            self.chbUseGPU.setToolTip(str(dev_name))
            self.chbUseGPU.setChecked(bool(onGPU))
            self.chbUseGPU.stateChanged.connect(self.enable_GPU)

        # endregion: Devices and plot checkbox --------------------------------

        # region: Experiment control space ------------------------------------
        # RtpApp.ui_set_param() places the experiment controls here, so this
        # must be made before calling it.
        self.hBoxExpCtrls = QtWidgets.QHBoxLayout()

        # --- Parameter setting -----------------------------------------------
        # --- tabs ---
        self.options_tab = QtWidgets.QTabWidget()
        self.options_tab.currentChanged.connect(self.show_options_list)
        self.options_tab.tabBarClicked.connect(self.show_options_list)

        # --- Application setting tab ---
        # Selection buttons
        self.Apps_btn = OrderedDict()
        for app_name in self.rtp_apps.keys():
            self.Apps_btn[app_name] = QtWidgets.QPushButton(
                app_name, self.mainWidget
            )
            self.Apps_btn[app_name].clicked.connect(
                partial(self.on_clicked_setOption, app_name)
            )
            self.Apps_btn[app_name].setStyleSheet(_SELECTED_BTN_STYLE)

        # Application setting pane stack
        self.stackedAppSetPanes = QtWidgets.QStackedWidget(self.mainWidget)
        self.AppSetPanes = OrderedDict()
        for app_name, app_obj in self.rtp_apps.items():
            self.AppSetPanes[app_name] = QtWidgets.QGroupBox(
                f"{app_name} setting", self.mainWidget
            )
            self.stackedAppSetPanes.addWidget(self.AppSetPanes[app_name])
            fLayout = QtWidgets.QFormLayout(self.AppSetPanes[app_name])
            app_obj.main_win = self

            ui_objs = app_obj.ui_set_param()
            if ui_objs is not None:
                for ui_row in ui_objs:
                    fLayout.addRow(*ui_row)

        # --- RTP setting tab ---
        # Selection buttons. The EXTSIG button (if any) comes first.
        self.RTP_btn = OrderedDict()
        btn_procs = list(_RTP_PROCS[:-1])
        if "EXTSIG" in rtp_objs:
            btn_procs.insert(0, "EXTSIG")
        for proc in btn_procs:
            label = "EXT SIGNAL" if proc == "EXTSIG" else proc
            self.RTP_btn[proc] = QtWidgets.QPushButton(label, self.mainWidget)
            self.RTP_btn[proc].clicked.connect(
                partial(self.on_clicked_setOption, proc)
            )

        # RTP setting pane stack
        self.stackedRTPSetPanes = QtWidgets.QStackedWidget(self.mainWidget)
        self.RTPSetPanes = {}
        for proc in _RTP_PROCS:
            self.RTPSetPanes[proc] = QtWidgets.QGroupBox(
                f"{proc} options", self.mainWidget
            )
            self.stackedRTPSetPanes.addWidget(self.RTPSetPanes[proc])
            fLayout = QtWidgets.QFormLayout(self.RTPSetPanes[proc])
            if proc not in rtp_objs:
                continue

            ui_objs = rtp_objs[proc].ui_set_param()
            if ui_objs is not None:
                for ui_row in ui_objs:
                    fLayout.addRow(*ui_row)

            rtp_objs[proc].main_win = self

        # --- Parameter list tab ---
        self.listAllParams = QtWidgets.QWidget(self.mainWidget)
        self.listParam_txtBrws = QtWidgets.QTextBrowser(self.listAllParams)

        #  endregion: Experiment control space --------------------------------

        # region: log console -------------------------------------------------
        self.logOutput_txtEd = QtWidgets.QTextEdit(self.mainWidget)
        self.logOutput_txtEd.setReadOnly(True)
        self.logOutput_txtEd.setAcceptRichText(True)
        self.logOutput_txtEd.setLineWrapMode(
            QtWidgets.QTextEdit.LineWrapMode.NoWrap)

        # Fixed-width font; the console shows at least LOG_MIN_COLUMNS
        # characters per line (the main window is widened to fit it)
        log_font = QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.SystemFont.FixedFont)
        log_font.setPointSize(LOG_FONT_SIZE)
        self.logOutput_txtEd.setFont(log_font)
        self.logOutput_txtEd.document().setDefaultFont(log_font)
        fm = QtGui.QFontMetrics(log_font)
        self.logOutput_txtEd.setMinimumWidth(
            fm.horizontalAdvance("x" * LOG_MIN_COLUMNS)
            + self.logOutput_txtEd.verticalScrollBar().sizeHint().width()
            + 2 * self.logOutput_txtEd.frameWidth()
            + 2 * int(self.logOutput_txtEd.document().documentMargin())
        )
        self.logOutput_txtEd.setMinimumHeight(128)
        # endregion: log console ----------------------------------------------

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def layout_ui(self):
        """Layout GUI controls"""

        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            QtWidgets.QSizePolicy.Policy.Fixed
        )

        # Root vertical box layout
        vBoxRoot = QtWidgets.QVBoxLayout(self.mainWidget)
        vSplitterRoot = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        vBoxRoot.addWidget(vSplitterRoot)

        topWidget = QtWidgets.QWidget(self.mainWidget)
        vSplitterRoot.addWidget(topWidget)
        vBoxTop = QtWidgets.QVBoxLayout(topWidget)

        # --- Watch/Work directory --------------------------------------------
        watchDirHBox = QtWidgets.QHBoxLayout()
        vBoxTop.addLayout(watchDirHBox)

        watchDirHBox.addWidget(self.labelWatchDir)
        watchDirHBox.addWidget(self.lineEditWatchDir)
        watchDirHBox.addWidget(self.btnSetWatchDir)

        workDirHBox = QtWidgets.QHBoxLayout()
        vBoxTop.addLayout(workDirHBox)

        workDirHBox.addWidget(self.labelWorkDir)
        workDirHBox.addWidget(self.lineEditWorkDir)
        workDirHBox.addWidget(self.btnSetWorkDir)

        # --- Checkboxes ------------------------------------------------------
        self.grpBoxChkBoxes = QtWidgets.QGroupBox(
            "Use / Show", self.mainWidget
        )
        vBoxTop.addWidget(self.grpBoxChkBoxes)
        self.ui_hChkBoxes = QtWidgets.QHBoxLayout(self.grpBoxChkBoxes)
        if hasattr(self, "chbUseGPU"):
            self.ui_hChkBoxes.addWidget(self.chbUseGPU)

        if hasattr(self, "chbUseGPU"):
            # A separator between 'Use' (GPU) and 'Show' (plots)
            sep = QtWidgets.QFrame()
            sep.setFrameShape(QtWidgets.QFrame.Shape.VLine)
            sep.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
            self.ui_hChkBoxes.addSpacing(6)
            self.ui_hChkBoxes.addWidget(sep)
            self.ui_hChkBoxes.addSpacing(6)

        if hasattr(self, "chbShowPhysio"):
            self.ui_hChkBoxes.addWidget(self.chbShowPhysio)

        if hasattr(self, "chbShowMotion"):
            self.ui_hChkBoxes.addWidget(self.chbShowMotion)

        for app_obj in self.rtp_apps.values():
            if hasattr(app_obj, "ui_showLatency_cbx"):
                self.ui_hChkBoxes.addWidget(app_obj.ui_showLatency_cbx)
        # The ROI signal at the right end
        for app_name, app_obj in self.rtp_apps.items():
            if hasattr(app_obj, "ui_showROISig_cbx"):
                ui_cbx = app_obj.ui_showROISig_cbx
                if len(self.rtp_apps) > 1:  # Which app's signal
                    ui_cbx.setText(f"{app_name}: {ui_cbx.text()}")
                self.ui_hChkBoxes.addWidget(ui_cbx)
        self.ui_hChkBoxes.addStretch(1)  # Pack the check boxes to the left

        # --- Parameter tabs --------------------------------------------------
        vBoxTop.addWidget(self.options_tab)

        # --- App setting ---
        AppSetWidget = QtWidgets.QWidget(self.mainWidget)
        self.options_tab.addTab(AppSetWidget, "App")
        vBoxAppSet = QtWidgets.QVBoxLayout(AppSetWidget)

        # Application selection buttons
        hBoxAppBtns = QtWidgets.QHBoxLayout()
        vBoxAppSet.addLayout(hBoxAppBtns)
        for btn in self.Apps_btn.values():
            hBoxAppBtns.addWidget(btn)

        # In a scroll area, so that the main window fits a small screen
        vBoxAppSet.addWidget(self._scroll_area(self.stackedAppSetPanes))

        # --- RTP setting ---
        RTPSetWidget = QtWidgets.QWidget(self.mainWidget)
        self.options_tab.addTab(RTPSetWidget, "RTP")
        vBoxRTPSet = QtWidgets.QVBoxLayout(RTPSetWidget)

        # RTP selection buttons
        hBoxRTPBtns = QtWidgets.QHBoxLayout()
        vBoxRTPSet.addLayout(hBoxRTPBtns)
        for btn in self.RTP_btn.values():
            hBoxRTPBtns.addWidget(btn)

        vBoxRTPSet.addWidget(self._scroll_area(self.stackedRTPSetPanes))

        # --- Parameter list ---
        vBoxListParam = QtWidgets.QVBoxLayout(self.listAllParams)
        vBoxListParam.addWidget(self.listParam_txtBrws)
        self.options_tab.addTab(self.listAllParams, "Parameter list")

        # --- Experiment control space ----------------------------------------
        vBoxTop.addLayout(self.hBoxExpCtrls)

        # --- log -------------------------------------------------------------
        vSplitterRoot.addWidget(self.logOutput_txtEd)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def err_popup(self, errmsg):
        """Show an error message dialog."""
        self._notifier.error(errmsg, title=self.__class__.__name__)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _all_objs(self):
        """Return a new dict of the RTP modules and applications.

        A new dict is made not to add the applications to self.rtp_objs,
        which is RtpApp.rtp_objs.
        """
        return {**self.rtp_objs, **self.rtp_apps}

    # --- slot functions ------------------------------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ui_save_parameters(self):
        """Save the parameters of all modules and applications to a file."""
        allObjs = self._all_objs()

        path = os.getcwd()
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save parameters", path, "pickle file (*.pkl)"
        )
        if fname == "":
            return

        if os.path.splitext(fname)[-1] != ".pkl":
            fname += ".pkl"

        save_parameters(allObjs, fname)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ui_load_parameters(self):
        """Load the parameters of all modules and applications from a file."""
        allObjs = self._all_objs()

        path = os.getcwd()
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load parameters", path, "pickle file (*.pkl)"
        )
        if fname == "":
            return
        load_parameters(allObjs, fname)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_workDir(self, wdir=None):
        """Set the working directory of all modules.

        A directory dialog opens if wdir is not given. A missing directory
        is created.
        """
        if wdir is None or wdir is False:  # Button press gives False
            wdir = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select working directory", str(self.watchDir_start)
            )
            if len(wdir) == 0:
                return -1
        else:
            if not Path(wdir).is_dir():
                Path(wdir).mkdir()

        for obj in self._all_objs().values():
            if hasattr(obj, "work_dir") and Path(obj.work_dir) != Path(wdir):
                obj.set_param("work_dir", Path(wdir))

        self.lineEditWorkDir.setText(str(wdir))

        return 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_watchDir(self, wdir=None):
        """Set the watching directory of the modules that have one.

        A directory dialog opens if wdir is not given.
        """
        if wdir is None or wdir is False:  # Button press gives False
            wdir = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select watching directory", str(self.watchDir_start)
            )
            if len(wdir) == 0:
                return -1
        else:
            if not Path(wdir).is_dir():
                errmsg = f"{wdir} is not a directory."
                self._logger.error(errmsg)
                self.err_popup(errmsg)
                return -1

        for obj in self._all_objs().values():
            if hasattr(obj, "watch_dir") and (
                obj.watch_dir is None or Path(obj.watch_dir) != Path(wdir)
            ):
                obj.set_param("watch_dir", Path(wdir))

        self.lineEditWatchDir.setText(str(wdir))

        return 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_log(self, log_file):
        """Display the log file in the window."""
        self._log_f = Path(log_file).absolute()

        # Wait up to 5 s for the log file to be created
        st = time.time()
        while not self._log_f.is_file() and time.time() - st < 5:
            time.sleep(0.5)

        if self._log_f.is_file():
            self._log_fd = open(self._log_f, "r")

        # Poll the log file for new lines
        self._log_update_timer.setInterval(333)
        self._log_update_timer.timeout.connect(self.update_log_display)
        self._log_update_timer.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def update_log_display(self):
        """Append new log lines to the log console.

        Lines with '!!!' or 'ERROR' are shown in red ('!!!' is removed).
        Lines with '<B>' are shown in bold ('<B>' is removed).
        """
        if self._log_fd is None:
            return

        new_entries = self._log_fd.read()
        if new_entries:
            self.logOutput_txtEd.moveCursor(
                QtGui.QTextCursor.MoveOperation.End)
            log_lines = new_entries.split("\n")
            for ii, log_line in enumerate(log_lines):
                # The last item is the rest after the last newline.
                if ii != len(log_lines) - 1:
                    add_line = log_line + "\n"
                else:
                    add_line = log_line

                if "!!!" in add_line or "ERROR" in add_line:
                    self.logOutput_txtEd.setTextColor(
                        QtGui.QColor(gui_theme.COLORS["red"]))
                    add_line = add_line.replace("!!!", "")
                else:
                    self.logOutput_txtEd.setTextColor(
                        QtGui.QColor(gui_theme.COLORS["fg"]))

                if "<B>" in add_line:
                    self.logOutput_txtEd.setFontWeight(QtGui.QFont.Weight.Bold)
                    add_line = add_line.replace("<B>", "")
                else:
                    self.logOutput_txtEd.setFontWeight(
                        QtGui.QFont.Weight.Normal)

                self.logOutput_txtEd.insertPlainText(add_line)

                # Scroll to the end
                sb = self.logOutput_txtEd.verticalScrollBar()
                sb.setValue(sb.maximum())

                # Reset the text format
                cursor = self.logOutput_txtEd.textCursor()
                default_format = QtGui.QTextCharFormat()
                default_format.clearForeground()
                cursor.setCharFormat(default_format)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def on_clicked_setOption(self, proc):
        """Show the setting pane of an application or RTP module."""
        if proc in self.AppSetPanes:
            self.options_tab.setCurrentIndex(0)
            self.stackedAppSetPanes.setCurrentWidget(self.AppSetPanes[proc])
            buttons = self.Apps_btn
        elif proc in self.RTPSetPanes:
            self.options_tab.setCurrentIndex(1)
            self.stackedRTPSetPanes.setCurrentWidget(self.RTPSetPanes[proc])
            buttons = self.RTP_btn
        else:
            return

        # Highlight the selected button
        for name, btnObj in buttons.items():
            if name == proc:
                btnObj.setStyleSheet(_SELECTED_BTN_STYLE)
            else:
                btnObj.setStyleSheet("")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def show_mot_chk(self, state):
        """Open or close the motion plot."""
        if "VOLREG" not in self.rtp_objs:
            return

        if state == 2:
            self.rtp_objs["VOLREG"].open_motion_plot()
        else:
            self.rtp_objs["VOLREG"].close_motion_plot()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def show_physio_chk(self, state):
        """Show or hide the physio window."""
        if (
            "TTLPHYSIO" not in self.rtp_objs or
            self.rtp_objs["TTLPHYSIO"] is None
        ):
            return

        self._physio_chk_time = time.monotonic()
        if state > 0:
            self.rtp_objs["TTLPHYSIO"].show()
            # Keep it at its place in the layout
            self.ensure_physio_geometry()
        else:
            self.rtp_objs["TTLPHYSIO"].hide()

        # Sync the checkbox when this is called from the program
        if hasattr(self, "chbShowPhysio"):
            self.chbShowPhysio.blockSignals(True)
            self.chbShowPhysio.setChecked(state > 0)
            self.chbShowPhysio.blockSignals(False)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def enable_GPU(self, *args):
        """Set the GPU use of all modules from the checkbox."""
        for obj in self._all_objs().values():
            if hasattr(obj, "onGPU"):
                obj.onGPU = self.chbUseGPU.isChecked()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def show_options_list(self, html=True, hide_sham=True):
        """Show the parameter list in the text browser.

        Parameters
        ----------
        html : bool, optional
            If False, return the parameter dictionary without showing it.
        hide_sham : bool, optional
            Hide the parameters with 'sham' in the name.
        """
        all_params = OrderedDict()
        for rtp in _RTP_PROCS:
            if rtp in self.rtp_objs and self.rtp_objs[rtp].enabled:
                all_params[rtp] = self.rtp_objs[rtp].get_params()

        # enable_RTP of the last enabled application decides whether the RTP
        # module parameters are shown.
        enable_RTP = True
        for app_name, app_obj in self.rtp_apps.items():
            if not app_obj.enabled:
                continue

            all_params[app_name] = app_obj.get_params()
            try:
                enable_RTP = app_obj.enable_RTP
            except Exception:
                enable_RTP = True

        if not html:
            return all_params

        param_list = ""
        for rtp, opt_dict in all_params.items():
            if not enable_RTP and rtp in _RTP_PROCS:
                continue

            param_list += "<p>"
            param_list += f"<font size='+1'><b>{rtp}</b></font><br/>\n"

            for k, val in opt_dict.items():
                if k == "proc_times":
                    continue

                if hide_sham and "sham" in k.lower():
                    continue

                param_list += f"<b>{k}</b>: {val}<br/>\n"

            param_list += "</p>\n"

        self.listParam_txtBrws.setText(param_list)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def keyPressEvent(self, event):
        # Pass the key press to the applications (e.g., manual scan start)
        for rtp_app in self.rtp_apps.values():
            rtp_app.keyPressEvent(event)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def closeEvent(self, event):
        # Do not close while dcm2niix is running
        for rtp_app in self.rtp_apps.values():
            if rtp_app._is_running_dcm2nii():
                event.ignore()
                return

        # Stop physio
        self._stop_physio_watch()
        if (
            "TTLPHYSIO" in self.rtp_objs
            and self.rtp_objs["TTLPHYSIO"] is not None
        ):
            self.rtp_objs["TTLPHYSIO"].quit()

        # Copy the log file shown in this window (set_log) to the work_dir
        # of the modules and applications
        dst_fnames = set()
        if hasattr(self, "_log_f") and self._log_f.is_file():
            for rtp in self._all_objs().values():
                if (
                    hasattr(rtp, "work_dir")
                    and str(rtp.work_dir) != ""
                    and os.path.realpath(rtp.work_dir) != os.getcwd()
                ):
                    dst_f = Path(rtp.work_dir).absolute() / self._log_f.name
                    if dst_f != self._log_f:
                        dst_fnames.add(dst_f)

        for dst_fname in dst_fnames:
            if not dst_fname.parent.is_dir():
                os.makedirs(dst_fname.parent)

            shutil.copy(self._log_f, dst_fname)

        self._log_update_timer.stop()
