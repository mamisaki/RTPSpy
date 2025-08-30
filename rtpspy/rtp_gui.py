# -*- coding: utf-8 -*-
"""
RTPSpy graphical user interface

@author: mmisaki@laureateinstitute.org
"""

# %% import ===================================================================
from pathlib import Path
import os
import shutil
from functools import partial
import time
from collections import OrderedDict
import logging
import sys

import torch
from PyQt5 import QtCore, QtWidgets, QtGui

try:
    from .rtp_common import save_parameters, load_parameters

except Exception:
    from rtpspy.rtp_common import save_parameters, load_parameters

GPU_available = (
    torch.backends.mps.is_available() or torch.backends.mps.is_available()
)


# %% RtpGUI class =============================================================
class RtpGUI(QtWidgets.QMainWindow):
    """RtpGUI class"""

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
        rtp_objs : dictionary, optional
            RTP objects given by RtpApp.rtp_objs.
        rtp_apps : TYPE, optional
            RtpApp or derived class object.
        log_file : str or Path, optional
            Log file. The default is None
        winTitle : str, optional
            WIndow title. The default is 'RTPSpy'.
        onGPU : bool, optional
            Flag to use GPU. The default is given by torch.cuda.is_available()
            or torch.backends.mps.is_available().

        """
        QtWidgets.QMainWindow.__init__(self)
        self._logger = logging.getLogger("RtpGUI")

        self.rtp_objs = rtp_objs
        self.rtp_apps = rtp_apps

        self.GPU_available = GPU_available
        if onGPU and not GPU_available:
            onGPU = False

        self.watchDir_start = Path.cwd()

        # Setup main UI
        self.setWindowTitle(winTitle)  # Set window title
        self.mainWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.mainWidget)

        self.make_menu()
        self.make_ui(rtp_objs, onGPU)
        self.layout_ui()

        # Initialize view
        self.on_clicked_setOption("WATCH")
        self.options_tab.setCurrentIndex(0)

        # Move window to screen top,left
        self.move(0, 0)

        # Set log
        self._log_fd = None
        self._log_update_timer = QtCore.QTimer()
        if log_file:
            self.set_log(log_file)

        msg = "=== Start application ===\n"
        self._logger.info(msg)

        # Wait until the window is ready
        QtWidgets.QApplication.processEvents()
        time.sleep(1)

        # Move physio window next to the main window
        if (
            "TTLPHYSIO" in self.rtp_objs and
            self.rtp_objs["TTLPHYSIO"] is not None
        ):
            # Get top right corner of the main window
            geo = self.geometry()
            x = geo.x() + geo.width() + 55
            y = geo.y()
            physio_geometry = f"450x450+{x}+{y}"
            self.rtp_objs["TTLPHYSIO"].move(physio_geometry)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def make_menu(self):
        # --- menu bar --------------------------------------------------------
        # Save parameters
        saveOptAct = QtWidgets.QAction(QtGui.QIcon("save.png"), "&Save", self)
        saveOptAct.setShortcut("Ctrl+S")
        saveOptAct.setStatusTip("Save current parameters")
        saveOptAct.triggered.connect(self.ui_save_parameters)

        # Load parameters
        loadOptAct = QtWidgets.QAction(QtGui.QIcon("load.png"), "&Load", self)
        loadOptAct.setShortcut("Ctrl+L")
        loadOptAct.setStatusTip("Load parameters")
        loadOptAct.triggered.connect(self.ui_load_parameters)

        # Exit without saving
        exitNosaveAct = QtWidgets.QAction(
            QtGui.QIcon("exit.png"), "Exit&WithoutSaving", self
        )
        exitNosaveAct.setShortcut("Ctrl+W")
        exitNosaveAct.setStatusTip(
            "Exit application without saving parameters"
        )
        exitNosaveAct.triggered.connect(lambda: QtWidgets.qApp.exit(1))

        # Exit
        exitAct = QtWidgets.QAction(QtGui.QIcon("exit.png"), "&Exit", self)
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
        """Make GUI controls

        Options
        -------
        rpt_mods: dictionary
            dictionary of RTP module instances
        """

        # region: Watch/Work dir controls -------------------------------------
        # Watching Directory Label
        self.labelWatchDir = QtWidgets.QLabel(self.mainWidget)
        self.labelWatchDir.setText("rtfMRI watching directory")

        # Watching Directory LineEdit
        self.lineEditWatchDir = QtWidgets.QLineEdit(self.mainWidget)
        self.lineEditWatchDir.setReadOnly(True)
        self.lineEditWatchDir.setStyleSheet("border: 0px none;")

        # Set watching directory button
        self.btnSetWatchDir = QtWidgets.QPushButton("Set", self.mainWidget)
        self.btnSetWatchDir.clicked.connect(self.set_watchDir)

        # Working Directory Label
        self.labelWorkDir = QtWidgets.QLabel(self.mainWidget)
        self.labelWorkDir.setText("Working directory")

        # Working Directory LineEdit
        self.lineEditWorkDir = QtWidgets.QLineEdit(self.mainWidget)
        self.lineEditWorkDir.setReadOnly(True)
        self.lineEditWorkDir.setStyleSheet("border: 0px none;")

        # Set working directory button
        self.btnSetWorkDir = QtWidgets.QPushButton("Set", self.mainWidget)
        self.btnSetWorkDir.clicked.connect(self.set_workDir)
        # endregion: Watch/Work dir controls ----------------------------------

        # region: Devices and plot checkbox -----------------------------------
        # Show motion
        if "VOLREG" in rtp_objs:
            self.chbShowMotion = QtWidgets.QCheckBox(
                "Show motion", self.mainWidget
            )
            self.chbShowMotion.setCheckState(0)
            self.chbShowMotion.stateChanged.connect(
                lambda x: self.show_mot_chk(x)
            )

        # Show Physio
        if (
            "TTLPHYSIO" in rtp_objs and
            self.rtp_objs["TTLPHYSIO"] is not None
        ):
            self.chbShowPhysio = QtWidgets.QCheckBox(
                "Show physio", self.mainWidget
            )
            self.chbShowPhysio.setCheckState(0)
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
                f"Use GPU\n{dev_name}", self.mainWidget
            )
            self.chbUseGPU.setCheckState(onGPU * 2)
            self.chbUseGPU.stateChanged.connect(self.enable_GPU)

        # endregion: Devices and plot checkbox --------------------------------

        # region: Experiment control space ------------------------------------
        # Will be used by a RtpApp obejct to place a experiment controls.
        # This needs to be defined before the app_obj.ui_set_param() call.
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
            self.Apps_btn[app_name].setStyleSheet(
                "background-color: rgb(255,201,32)"
            )

        # Application setting pane stack
        self.stackedAppSetPanes = QtWidgets.QStackedWidget(self.mainWidget)
        self.AppSetPanes = OrderedDict()
        for app_name, app_obj in self.rtp_apps.items():
            self.AppSetPanes[app_name] = QtWidgets.QGroupBox(
                "{} setting".format(app_name), self.mainWidget
            )
            self.stackedAppSetPanes.addWidget(self.AppSetPanes[app_name])
            fLayout = QtWidgets.QFormLayout(self.AppSetPanes[app_name])
            app_obj.main_win = self

            ui_objs = app_obj.ui_set_param()
            if ui_objs is not None:
                for ui_row in ui_objs:
                    fLayout.addRow(*ui_row)

        # --- RTP setting tab ---
        # Selection buttons
        self.RTP_btn = OrderedDict()

        # EXTSIG
        if "EXTSIG" in rtp_objs:
            self.RTP_btn["EXTSIG"] = QtWidgets.QPushButton(
                "EXT SIGNAL", self.mainWidget
            )
            self.RTP_btn["EXTSIG"].clicked.connect(
                partial(self.on_clicked_setOption, "EXTSIG")
            )
        # WATCH
        self.RTP_btn["WATCH"] = QtWidgets.QPushButton("WATCH", self.mainWidget)
        self.RTP_btn["WATCH"].clicked.connect(
            partial(self.on_clicked_setOption, "WATCH")
        )
        # VOLREG
        self.RTP_btn["VOLREG"] = QtWidgets.QPushButton(
            "VOLREG", self.mainWidget
        )
        self.RTP_btn["VOLREG"].clicked.connect(
            partial(self.on_clicked_setOption, "VOLREG")
        )
        # TSHIFT
        self.RTP_btn["TSHIFT"] = QtWidgets.QPushButton(
            "TSHIFT", self.mainWidget
        )
        self.RTP_btn["TSHIFT"].clicked.connect(
            partial(self.on_clicked_setOption, "TSHIFT")
        )
        # SMOOTH
        self.RTP_btn["SMOOTH"] = QtWidgets.QPushButton(
            "SMOOTH", self.mainWidget
        )
        self.RTP_btn["SMOOTH"].clicked.connect(
            partial(self.on_clicked_setOption, "SMOOTH")
        )
        # REGRESS
        self.RTP_btn["REGRESS"] = QtWidgets.QPushButton(
            "REGRESS", self.mainWidget
        )
        self.RTP_btn["REGRESS"].clicked.connect(
            partial(self.on_clicked_setOption, "REGRESS")
        )

        # RTP setting pane stack
        self.stackedRTPSetPanes = QtWidgets.QStackedWidget(self.mainWidget)
        self.RTPSetPanes = {}
        for proc in [
            "WATCH",
            "VOLREG",
            "TSHIFT",
            "SMOOTH",
            "REGRESS",
            "EXTSIG",
        ]:
            self.RTPSetPanes[proc] = QtWidgets.QGroupBox(
                "{} options".format(proc), self.mainWidget
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
        self.logOutput_txtEd.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)

        self.logOutput_txtEd.font().setFamily("Courier")
        self.logOutput_txtEd.font().setPointSize(9)
        self.logOutput_txtEd.setMinimumHeight(128)
        # endregion: log console ----------------------------------------------

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def layout_ui(self):
        """Layout GUI controls"""

        self.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed
        )

        # Root vertical box layout
        vBoxRoot = QtWidgets.QVBoxLayout(self.mainWidget)
        vSplitterRoot = QtWidgets.QSplitter(QtCore.Qt.Vertical)
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
        # watchDirHBox.addWidget(self.afniBootBtn)

        workDirHBox = QtWidgets.QHBoxLayout()
        vBoxTop.addLayout(workDirHBox)

        workDirHBox.addWidget(self.labelWorkDir)
        workDirHBox.addWidget(self.lineEditWorkDir)
        workDirHBox.addWidget(self.btnSetWorkDir)

        # --- Checkboxes ------------------------------------------------------
        self.grpBoxChkBoxes = QtWidgets.QGroupBox(
            "GPU/Signal/Plot", self.mainWidget
        )
        vBoxTop.addWidget(self.grpBoxChkBoxes)
        self.ui_hChkBoxes = QtWidgets.QHBoxLayout(self.grpBoxChkBoxes)
        if hasattr(self, "chbUseGPU"):
            self.ui_hChkBoxes.addWidget(self.chbUseGPU)

        if hasattr(self, "chbShowPhysio"):
            self.ui_hChkBoxes.addWidget(self.chbShowPhysio)

        if hasattr(self, "chbShowMotion"):
            self.ui_hChkBoxes.addWidget(self.chbShowMotion)

        for app_name in self.rtp_apps.keys():
            if hasattr(self.rtp_apps[app_name], "ui_showROISig_cbx"):
                ui_cbx = self.rtp_apps[app_name].ui_showROISig_cbx
                ui_cbx.setText(f"{app_name}:{ui_cbx.text()}")
                self.ui_hChkBoxes.addWidget(ui_cbx)

        # --- Parameter tabs --------------------------------------------------
        vBoxTop.addWidget(self.options_tab)

        # --- App setting ---
        AppSetWidget = QtWidgets.QWidget(self.mainWidget)
        self.options_tab.addTab(AppSetWidget, "App")
        vBoxAppSet = QtWidgets.QVBoxLayout(AppSetWidget)

        # Application selection buttons
        hBoxAppBtns = QtWidgets.QHBoxLayout()
        vBoxAppSet.addLayout(hBoxAppBtns)
        for app_name in self.Apps_btn.keys():
            hBoxAppBtns.addWidget(self.Apps_btn[app_name])

        vBoxAppSet.addWidget(self.stackedAppSetPanes)

        # --- RTP setting ---
        RTPSetWidget = QtWidgets.QWidget(self.mainWidget)
        self.options_tab.addTab(RTPSetWidget, "RTP")
        vBoxRTPSet = QtWidgets.QVBoxLayout(RTPSetWidget)

        # RTP selection buttons
        hBoxRTPBtns = QtWidgets.QHBoxLayout()
        vBoxRTPSet.addLayout(hBoxRTPBtns)
        for rtp in self.RTP_btn.keys():
            hBoxRTPBtns.addWidget(self.RTP_btn[rtp])

        vBoxRTPSet.addWidget(self.stackedRTPSetPanes)

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
        msgBox = QtWidgets.QMessageBox()
        msgBox.setIcon(QtWidgets.QMessageBox.Critical)
        msgBox.setText(errmsg)
        msgBox.setWindowTitle(self.__class__.__name__)
        msgBox.exec()

    # --- slot functions ------------------------------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ui_save_parameters(self):
        allObjs = self.rtp_objs
        allObjs.update(self.rtp_apps)

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
        allObjs = self.rtp_objs
        allObjs.update(self.rtp_apps)

        path = os.getcwd()
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load parameters", path, "pickle file (*.pkl)"
        )
        if fname == "":
            return
        load_parameters(allObjs, fname)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_workDir(self, wdir=None):
        if wdir is None or wdir is False:  # Button press gives False
            wdir = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select working directory", str(self.watchDir_start)
            )
            if len(wdir) == 0:
                return -1
        else:
            if not Path(wdir).is_dir():
                Path(wdir).mkdir()

        for obj in self.rtp_objs.values():
            if hasattr(obj, "work_dir") and Path(obj.work_dir) != Path(wdir):
                obj.set_param("work_dir", Path(wdir))

        self.lineEditWorkDir.setText(str(wdir))

        return 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_watchDir(self, wdir=None):
        if wdir is None or wdir is False:  # Button press gives False
            wdir = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select Watch directory", str(self.watchDir_start)
            )
            if len(wdir) == 0:
                return -1
        else:
            if not Path(wdir).is_dir():
                errmsg = f"{wdir} is not a directory."
                self._logger.error(errmsg)
                self.err_popup(errmsg)
                return -1

        for obj in self.rtp_objs.values():
            if hasattr(obj, "watch_dir") and (
                obj.watch_dir is None or Path(obj.watch_dir) != Path(wdir)
            ):
                obj.set_param("watch_dir", Path(wdir))

        self.lineEditWatchDir.setText(str(wdir))

        return 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_log(self, log_file):
        # Open a log file to display in the window
        self._log_f = Path(log_file).absolute()

        # Wait for the log file is ready
        st = time.time()
        while not Path(self._log_f).is_file() and time.time() - st < 5:
            time.sleep(0.5)

        # Open log file
        if self._log_f.is_file():
            log_fd = open(self._log_f, "r")

            if log_fd is None:
                sys.stderr.write(f"Failed to open {self._log_f}")
            else:
                self._log_fd = log_fd

        # Set log update timer
        self._log_update_timer.setInterval(333)
        self._log_update_timer.timeout.connect(self.update_log_display)
        self._log_update_timer.start()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def update_log_display(self):
        if self._log_fd is None:
            return

        new_entries = self._log_fd.read()
        if new_entries:
            self.logOutput_txtEd.moveCursor(QtGui.QTextCursor.End)
            log_lines = new_entries.split("\n")
            for ii, log_line in enumerate(log_lines):
                if ii != len(log_lines) - 1:
                    add_line = log_line + "\n"
                else:
                    add_line = log_line

                # Determine if this line should be red
                is_error_line = "!!!" in add_line or "ERROR" in add_line

                # Font Color
                if is_error_line:
                    # Red color
                    self.logOutput_txtEd.setTextColor(QtGui.QColor(255, 0, 0))
                    add_line = add_line.replace("!!!", "")
                else:
                    # Black color for normal lines
                    self.logOutput_txtEd.setTextColor(QtGui.QColor(0, 0, 0))

                # Font weight
                if "<B>" in add_line:
                    # Bold face
                    self.logOutput_txtEd.setFontWeight(QtGui.QFont.Bold)
                    add_line = add_line.replace("<B>", "")
                else:
                    self.logOutput_txtEd.setFontWeight(QtGui.QFont.Normal)

                # insert
                self.logOutput_txtEd.insertPlainText(add_line)

                # Move scroll bar
                sb = self.logOutput_txtEd.verticalScrollBar()
                sb.setValue(sb.maximum())

                # Reset text color
                cursor = self.logOutput_txtEd.textCursor()
                default_format = QtGui.QTextCharFormat()
                default_format.clearForeground()
                cursor.setCharFormat(default_format)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def on_clicked_setOption(self, proc):
        if proc in self.AppSetPanes:
            self.options_tab.setCurrentIndex(0)
            self.stackedAppSetPanes.setCurrentWidget(self.AppSetPanes[proc])
            # Set button color
            for app_name, btnObj in self.Apps_btn.items():
                if app_name == proc:
                    btnObj.setStyleSheet("background-color: rgb(255,201,32)")
                else:
                    btnObj.setStyleSheet("")

        elif proc in self.RTPSetPanes:
            self.options_tab.setCurrentIndex(1)
            self.stackedRTPSetPanes.setCurrentWidget(self.RTPSetPanes[proc])
            # Set button color
            for rtp_name, btnObj in self.RTP_btn.items():
                if rtp_name == proc:
                    btnObj.setStyleSheet("background-color: rgb(255,201,32)")
                else:
                    btnObj.setStyleSheet("")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def show_mot_chk(self, state):
        if "VOLREG" not in self.rtp_objs:
            return

        if state == 2:
            self.rtp_objs["VOLREG"].open_motion_plot()
        else:
            self.rtp_objs["VOLREG"].close_motion_plot()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def show_physio_chk(self, state):
        if (
            "TTLPHYSIO" not in self.rtp_objs or
            self.rtp_objs["TTLPHYSIO"] is None
        ):
            return

        if state > 0:
            self.rtp_objs["TTLPHYSIO"].show()
            if hasattr(self, "chbShowPhysio"):
                self.chbShowPhysio.blockSignals(True)
                self.chbShowPhysio.setCheckState(2)
                self.chbShowPhysio.blockSignals(False)
        else:
            self.rtp_objs["TTLPHYSIO"].hide()
            if hasattr(self, "chbShowPhysio"):
                self.chbShowPhysio.blockSignals(True)
                self.chbShowPhysio.setCheckState(0)
                self.chbShowPhysio.blockSignals(False)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def enable_GPU(self, *args):
        for obj in self.rtp_objs.values():
            if hasattr(obj, "onGPU"):
                obj.onGPU = self.chbUseGPU.checkState() > 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def show_options_list(self, html=True, hide_sham=True):
        """Print parameter list on text browser"""

        all_params = OrderedDict()
        for rtp in (
            "WATCH", "VOLREG", "TSHIFT", "SMOOTH", "REGRESS", "EXTSIG"
        ):
            if rtp not in self.rtp_objs:
                continue

            if not self.rtp_objs[rtp].enabled:
                continue

            all_params[rtp] = self.rtp_objs[rtp].get_params()

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
            if not enable_RTP and rtp in (
                "WATCH",
                "VOLREG",
                "TSHIFT",
                "SMOOTH",
                "REGRESS",
                "EXTSIG",
            ):
                continue

            param_list += "<p>"
            param_list += f"<font size='+1'><b>{rtp}</b></font><br/>\n"

            for k, val in opt_dict.items():
                if k == "proc_times":
                    # hide options
                    continue

                if hide_sham and "sham" in k.lower():
                    continue

                param_list += "<b>{}</b>: {}<br/>\n".format(k, val)

            param_list += "</p>\n"

        self.listParam_txtBrws.setText(param_list)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def keyPressEvent(self, event):
        for rtp_app in self.rtp_apps.values():
            rtp_app.keyPressEvent(event)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def closeEvent(self, event):
        # Check is the dcm2nnix is running
        for rtp_app in self.rtp_apps.values():
            if rtp_app._is_running_dcm2nii():
                event.ignore()
                return

        # Stop physio
        if (
            "TTLPHYSIO" in self.rtp_objs
            and self.rtp_objs["TTLPHYSIO"] is not None
        ):
            self.rtp_objs["TTLPHYSIO"].quit()

        # Move logfile to work_dir
        cpfnames = {}
        for rtp in list(self.rtp_objs.values()) + list(self.rtp_apps.values()):
            if (
                hasattr(rtp, "_log_f")
                and hasattr(rtp, "work_dir")
                and os.path.realpath(rtp.work_dir) != os.getcwd()
            ):
                logf = self._log_f
                if logf not in cpfnames:
                    cpfnames[logf] = Path(rtp.work_dir) / logf.name

        for src_fname, dst_fname in cpfnames.items():
            if not Path(src_fname).is_file():
                continue

            if not Path(dst_fname).parent.is_dir():
                os.makedirs(Path(dst_fname).parent)

            shutil.copy(src_fname, dst_fname)

        self._log_update_timer.stop()
