# -*- coding: utf-8 -*-
"""
RTP common utility classes

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import os
import sys
import re
from datetime import timedelta
import subprocess
import shlex
from multiprocessing import Value, Array, Lock
import pickle
import time
import traceback
import socket
import logging

import nibabel as nib
import numpy as np
import pandas as pd

import serial
from PyQt5 import QtWidgets, QtCore, QtGui
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.qt_compat as qt_compat
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure


# %% RTP base class ===========================================================
class RTP(object):
    """ Base class for real-time processing modules"""

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, ignore_init=0, next_proc=None, save_proc=False,
                 online_saving=False, save_delay=False, work_dir='',
                 max_scan_length=300, main_win=None, **kwargs):

        # Set arguments
        self.ignore_init = ignore_init
        self.next_proc = next_proc
        self.save_proc = save_proc
        self.online_saving = online_saving
        self.save_delay = save_delay
        self.work_dir = work_dir
        self.max_scan_length = max_scan_length
        self.main_win = main_win

        # Initialize parameters
        self._vol_num = 0  # 1-base, Number of processed (received) volumes
        self._proc_start_idx = -1  # 0-base index
        self._proc_time = []
        self.proc_delay = []

        self.saved_files = []
        self.saved_filename = None  # filename svaing canatenated data
        self.saved_data = None
        self.enabled = True

        self._logger = logging.getLogger(self.__class__.__name__)
        self._proc_ready = False

        self._std_out = sys.stdout
        self._err_out = sys.stderr

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @property
    def std_out(self):
        return self._std_out

    @std_out.setter
    def std_out(self, std_out):
        self._std_out = std_out

    @property
    def err_out(self):
        return self._err_out

    @err_out.setter
    def err_out(self, err_out):
        self._err_out = err_out

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ready_proc(self):
        """
        Ready the process.
        If self.next_proc is not None, self.next_proc.ready_proc() is called.
        Return all the process is ready (True) or not (False).

        This should be overridden.
        """

        self._vol_num = 0
        self._proc_start_idx = -1
        self._proc_time = []
        self.proc_delay = []
        self.saved_files = []
        self.saved_data = None

        if self.next_proc:
            self._proc_ready &= self.next_proc.ready_proc()

        return self._proc_ready

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def do_proc(self, fmri_img, vol_idx=None, pre_proc_time=0):
        """
        Do the process.
        If self.next_proc is not None, self.next_proc.do_proc() is called.

        This should be overridden.

        Parameters
        ----------
        fmri_img : nibabel image object
            nibabel image object.
        vol_idx : int, optional
            VOlume index. The default is None.
        pre_proc_time : float, optional
            Processing time until the previous modelue. The default is 0.

        """

        if self.next_proc is not None:
            self.next_proc.do_proc(fmri_img, vol_idx=vol_idx,
                                   pre_proc_time=pre_proc_time)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end_reset(self):
        """
        End process and reset process parameters.

        If self.next_proc is not None, self.next_proc.end_reset() is called.
        Return all the process is ready (True) or not (False).

        """

        out_files = {}

        # Save proc_delay
        if self.save_delay and len(self.proc_delay) > 0:
            out_f = self.save_proc_delay()
            out_files[f"{self.__class__.__name__}:process time"] = out_f

        # concatenate saved volumes
        if self.saved_data is not None:
            self.save_processed_data(self.saved_data, self.saved_data_affine,
                                     saved_files=self.saved_files)
            out_files[f"{self.__class__.__name__}:processed image"] = \
                self.saved_filename

        # Reset running variables
        self._vol_num = 0
        self._proc_start_idx = -1
        self._proc_time = []
        self.proc_delay = []
        self.saved_files = []
        self.saved_data = None

        # Reset child process
        if self.next_proc:
            out_fs = self.next_proc.end_reset()
            if out_fs is not None:
                out_files.update(out_fs)

        return out_files

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_param(self, attr, val, echo=True):
        setattr(self, attr, val)
        if echo:
            print("{}.".format(self.__class__.__name__) + attr, '=',
                  getattr(self, attr))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ui_set_param(self):
        """
        Retrun the list of usrer interfaces for setting parameters
        Each element is placed in QFormLayout.
        """

        pass

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def select_file_dlg(self, caption, directory, filt):
        """
        If the user presses Cancel, it returns a tuple of empty string.
        """
        fname = QtWidgets.QFileDialog.getOpenFileName(
            None, caption, str(directory), filt)
        return fname

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def keep_processed_image(self, fmri_img, vol_num=None, save_temp=False):
        """
        The data is kept in self.saved_data. At the end of the scan,
        self.saved_data will be saved as a 4D file by
        self.save_processed_data() .

        Parameters
        ----------
        fmri_img : nibabel image
            nibabel image object.
        vol_num : int, optional
            Saving volume index. If vol_num is None, self._vol_num is used.
            The default is None.
        save_temp : bool, optional
            Save the current data online as a single volume filet for debugging
            or something. The online save files will be deleted after the
            concatenated 4D file is saved.

        """

        savefname = fmri_img.get_filename()
        if Path(savefname).is_file():
            savefname = 'RTP_image'
            if self._vol_num > 0:
                savefname += f"_{self._vol_num:04d}"
            savefname += '.nii.gz'

        savefname = Path(self.work_dir) / 'RTP' / savefname
        self.saved_files.append(str(savefname))

        if self.saved_data is None:
            self.saved_data = np.zeros(
                [*fmri_img.shape, self.max_scan_length], dtype=np.float32)

        if vol_num is None:
            vol_num = self._vol_num

        if save_temp:
            self.save_data(fmri_img, savefname)

        if vol_num == self.saved_data.shape[-1]:
            # Expand self.saved_data
            save_data = self.saved_data
            self.saved_data = np.zeros(
                [*save_data.shape[:3], vol_num+100], dtype=save_data.dtype)
            self.saved_data[:, :, :, :save_data.shape[3]] = save_data

        self.saved_data[:, :, :, vol_num] = fmri_img.get_fdata()
        self.saved_data_affine = fmri_img.affine

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save_data(self, fmri_img, savefname):
        save_dir = Path(savefname).parent
        if not save_dir.is_dir():
            save_dir.mkdir()

        nib.save(fmri_img, savefname)
        msg = f"Save data as {savefname}"
        self._logger.info(msg)

        return savefname

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save_proc_delay(self):
        save_dir = Path(self.work_dir) / 'RTP'
        if not save_dir.is_dir():
            save_dir.mkdir()

        clname = self.__class__.__name__
        save_fname = os.path.join(
                save_dir, clname + '_proc_delay_' +
                time.strftime('%Y%m%d_%H%M%S') + '.txt')
        np.savetxt(save_fname, self.proc_delay, delimiter='\n')
        return save_fname

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_params(self):
        opts = dict()
        excld_opts = ('main_win',
                      'saved_files', 'next_proc',
                      'enabled', 'save_proc',
                      'save_delay', 'proc_delay', 'done_proc:', 'proc_data',
                      'saved_data', 'max_scan_length', 'done_proc',
                      'saved_data_affine', 'saved_filename',
                      'online_saving')

        for var_name, val in self.__dict__.items():
            if var_name in excld_opts or var_name[0] == '_' or \
                    'ui_' in var_name or \
                    isinstance(val, RTP) or \
                    isinstance(val, serial.Serial) or \
                    isinstance(val, QtCore.QThread):
                continue

            if isinstance(val, Path):
                get_val = str(val)
            else:
                get_val = val

            try:
                pickle.dumps(get_val)
                opts[var_name] = get_val
            except Exception:
                continue

        return opts

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save_processed_data(self, saved_data, affine, saved_files=[],
                            save_prefix=''):
        """
        Seve processd data
        """

        # --- Start message ---
        if self.main_win is not None:
            self._logger.info(
                "<B>Saving the processed data. Please wait ...")
            QtWidgets.QApplication.instance().processEvents()
        else:
            self._logger.info("Saving the processed data. Please wait ...")

        # --- Set filename ---
        if len(saved_files):
            commstr = re.sub(r'\.nii.*', '', saved_files[0])
            for ff in saved_files[1:]:
                commstr = ''.join(
                    [c for ii, c in enumerate(str(ff)[:len(commstr)])
                     if c == commstr[ii]])

            savefname0 = re.sub(r'_nr_\d*', '', commstr)
            savefname0 = re.sub(r'_ch\d*', '', savefname0)
            savefname = savefname0
        else:
            savefname = save_prefix + '.nii.gz'

        # Add file number postfix
        fn = 1
        while os.path.isfile(savefname + '.nii.gz'):
            savefname = f"{savefname0}_{fn}"
            fn += 1
        savefname += '.nii.gz'
        savefname = Path(savefname)

        if not savefname.parent.is_dir():
            os.makedirs(savefname.parent)

        # Preapre data array
        stidx = max(self._proc_start_idx, 0)
        img_data = saved_data[:, :, :, stidx:self._vol_num]
        simg = nib.Nifti1Image(img_data, affine)
        simg.to_filename(savefname)

        # --- Remove individual saved_files ---
        if len(saved_files):
            for ff in saved_files:
                if Path(ff).is_file():
                    self._logger.info(f'Delete {ff}')
                    Path(ff).unlink()

        self.saved_filename = savefname
        self._logger.info(f"Done. Processed data is saved as {savefname}")
        if self.main_win is not None:
            QtWidgets.QApplication.instance().processEvents()

    # # # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # def logmsg(self, msg, ret_str=False, show_ui=True):
    #     self._logger

    #     tstr = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S.%f')
    #     msg = "{}:[{}]: {}".format(tstr, self.__class__.__name__, msg)
    #     if ret_str:
    #         return msg

    #     if isinstance(self._std_out, LogDev):
    #         self._std_out.write(msg + '\n', show_ui=show_ui)
    #     else:
    #         self._std_out.write(msg + '\n')

    #     if hasattr(self._std_out, 'flush'):
    #         self._std_out.flush()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def err_popup(self, errmsg):
        if self.main_win is not None:
            # 'parent' cannot be self.main_win since this could be called by
            # other thread.
            msgBox = QtWidgets.QMessageBox()
            msgBox.setIcon(QtWidgets.QMessageBox.Critical)
            msgBox.setText(str(errmsg))
            msgBox.setWindowTitle(self.__class__.__name__)
            msgBox.exec()


# %% save_parameters ==========================================================
def save_parameters(objs, fname='RTPSpy_params.pkl'):

    props = dict()
    for k in objs.keys():
        if k == 'RETROTS':
            continue

        props[k] = dict()
        for var_name, var_val in objs[k].get_params().items():
            if var_name in ('main_win', '_std_out', '_err_out') or \
                    'ui_' in var_name or \
                    isinstance(var_val, RTP) or \
                    isinstance(var_val, serial.Serial) or \
                    isinstance(var_val, QtCore.QThread):
                continue

            try:
                pickle.dumps(var_val)
                props[k][var_name] = var_val
            except Exception:
                continue

    with open(fname, 'wb') as fd:
        pickle.dump(props, fd)


# %% load_parameters ==========================================================
def load_parameters(objs, fname='RTPSpy_params.pkl'):

    if not Path(fname).is_file():
        # sys.stderr.write("Not found parameter file: {}".format(fname))
        return False

    try:
        sys.stdout.write(f"Load parameters from {Path(fname).absolute()}\n")
        with open(fname, 'rb') as fd:
            props = pickle.load(fd)

        for mod in props.keys():
            if mod in objs:
                obj = objs[mod]
                for var_name, val in props[mod].items():
                    if not hasattr(obj, var_name) or var_name is None:
                        continue
                    try:
                        obj.set_param(var_name, val)
                    except TypeError:
                        errmsg = "Error setting "
                        errmsg += f"{obj.__class__.__name__}.{var_name}"
                        errmsg += f" = {str(val)}"
                        sys.stderr.write(errmsg)

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        errmsg = '{}, {}:{}'.format(
                    exc_type, exc_tb.tb_frame.f_code.co_filename,
                    exc_tb.tb_lineno)
        print(errmsg)


# %% LogDev ===================================================================
# class LogDev(QtCore.QObject):
#     write_log = QtCore.pyqtSignal(str)

#     def __init__(self, fname=None, ui_obj=None):
#         super().__init__()

#         self.fname = fname
#         if fname is not None:
#             if fname == sys.stdout:
#                 self.fd = fname
#             else:
#                 self.fd = open(fname, 'a')
#         else:
#             self.fd = None

#         self.ui_obj = ui_obj
#         if self.ui_obj is not None:
#             self.write_log.connect(self.ui_obj.print_log)

#     def write(self, txt, show_ui=True):
#         if self.ui_obj is not None and show_ui:
#             # GUI handling across threads is not allowed. So logging from
#             # other threads (e.g., rtp thread by watchdog) must be called via
#             # signal.
#             self.write_log.emit(txt)

#         if self.fd is not None:
#             self.fd.write(txt)

#     def flush(self):
#         if self.fd is not None:
#             self.fd.flush()

#     def __del__(self):
#         if self.fd is not None and self.fd != sys.stdout:
#             self.fd.close()


# %% DlgProgressBar ===========================================================
class DlgProgressBar(QtWidgets.QDialog):
    """
    Progress bar dialog
    """

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
        """
        fmt = QtGui.QTextBlockFormat()
        fmt.setLineHeight(0, QtGui.QTextBlockFormat.SingleHeight)
        self.desc_pTxtEd.textCursor().setBlockFormat(fmt)
        """

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

        self.show()

    def set_value(self, val):
        """
        Set progress bar value
        """
        self.progBar.setValue(int(val))
        if self.st_time is not None:
            ep = self.progBar.value()
            if ep > 0:
                et = time.time() - self.st_time
                last_t = (et/ep) * (100-ep)
                last_t_str = str(timedelta(seconds=last_t))
                last_t_str = last_t_str.split('.')[0]
                tstr = f"{self.title} (ETA {last_t_str})"
                self.setWindowTitle(tstr)

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


# %% plot_pause ===============================================================
def plot_pause(interval):
    """
    Pause function for matplotlib to update a plot in background
    https://stackoverflow.com/questions/45729092/make-interactive-matplotlib-window-not-pop-to-front-on-each-update-windows-7/45734500#45734500
    """

    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()

            canvas.start_event_loop(interval)
            return


# %% RingBuffer ===============================================================
class RingBuffer:
    """ Ring buffer with shared memory Array """

    def __init__(self, max_size):
        self.cur = Value('i', 0)
        self.max = max_size
        self.data = Array('d', np.ones(max_size)*np.nan)
        self._lock = Lock()

    def append(self, x):
        """ Append an element overwriting the oldest one. """
        with self._lock:
            self.data[self.cur.value] = x
            self.cur.value = (self.cur.value+1) % self.max

    def get(self):
        """ return list of elements in correct order """
        return np.concatenate([self.data[self.cur.value:],
                               self.data[:self.cur.value]])


# %% MatplotlibWindow =========================================================
class MatplotlibWindow(qt_compat.QtWidgets.QMainWindow):
    def __init__(self, figsize=[5, 3]):
        super().__init__()

        self.canvas = FigureCanvas(Figure(figsize=figsize))
        self.setCentralWidget(self.canvas)


# %% make_design_matrix =======================================================
def make_design_matrix(stim_times, scan_len, TR, out_f):

    NT = int(scan_len//TR)
    cmd = "3dDeconvolve -x1D_stop -nodata {} {}".format(NT, TR)
    cmd += " -local_times -x1D stdout:"
    cmd += " -num_stimts {}".format(len(stim_times))
    ii = 0
    names = []
    for optstr in stim_times:
        ii += 1
        times, opt, name = optstr.split(';')
        names.append(name)
        cmd += " -stim_times {} '{}' '{}'".format(ii, times, opt)
        cmd += " -stim_label {} {}".format(ii, name)

    pr = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE)
    out = pr.stdout.decode()

    ma = re.search(r'\# \>\n(.+)\n# \<', out, re.MULTILINE | re.DOTALL)
    mtx = ma.groups()[0]
    mtx = np.array([[float(v) for v in ll.split()] for ll in mtx.split('\n')])
    mtx = mtx[:, -len(stim_times):]

    desMtx = pd.DataFrame(mtx, columns=names)
    desMtx.to_csv(out_f, index=False)


# %% boot_afni ================================================================
def boot_afni(main_win=None, boot_dir='./', rt=True, TRUSTHOST=None):

    if rt:
        # Set TRUSTHOST
        ip = socket.gethostbyname(socket.gethostname())
        if TRUSTHOST is None or main_win is None:
            TRUSTHOST = '.'.join(ip.split('.')[:-1])
        elif TRUSTHOST == 'ask':
            dlg = QtWidgets.QInputDialog(main_win)
            dlg.setInputMode(QtWidgets.QInputDialog.TextInput)
            dlg.setLabelText('AFNI_TRUSTHOST')
            dlg.setTextValue('.'.join(ip.split('.')[:-1]))
            dlg.resize(640, 100)
            okflag = dlg.exec_()
            if not okflag:
                return

            TRUSTHOST = dlg.textValue()

    # Check afni process
    cmd0 = f"afni .* {boot_dir}"
    pret = subprocess.run(shlex.split(f"pgrep -af '{cmd0}'"),
                          stdout=subprocess.PIPE)
    procs = pret.stdout
    procs = [ll for ll in procs.decode().rstrip().split('\n')
             if "pgrep -af 'afni" not in ll and 'RTafni' not in ll and
             len(ll) > 0]
    if len(procs) > 0:
        for ll in procs:
            llsp = ll.split()
            pid = int(llsp[0])
            cmdl = ' '.join(llsp[4:])
            if 'RTafni' in cmdl:
                continue

            # Warning dialog
            msgBox = QtWidgets.QMessageBox()
            msgBox.setIcon(QtWidgets.QMessageBox.Warning)
            msgBox.setText("Kill existing afni process ?")
            msgBox.setInformativeText("pid={}: {}".format(pid, cmdl))
            msgBox.setWindowTitle("Existing afni process")
            msgBox.setStandardButtons(QtWidgets.QMessageBox.Yes |
                                      QtWidgets.QMessageBox.Ignore |
                                      QtWidgets.QMessageBox.Cancel)
            msgBox.setDefaultButton(QtWidgets.QMessageBox.Cancel)
            ret = msgBox.exec()
            if ret == QtWidgets.QMessageBox.Yes:
                os.kill(pid, 9)
            elif ret == QtWidgets.QMessageBox.Cancel:
                return

    # Boot afni at boot_dir
    if main_win is not None:
        xpos = 0
        ypos = main_win.frameGeometry().height()+25  # 25 => titlebar height
    else:
        xpos = 0
        ypos = 0

    cmd = 'afni'
    if rt:
        cmd += f" -rt -DAFNI_TRUSTHOST={TRUSTHOST} -DAFNI_REALTIME_WRITEMODE=1"
    cmd += " -yesplugouts -DAFNI_IMSAVE_WARNINGS=NO"
    cmd += f" -com 'OPEN_WINDOW A geom=+{xpos}+{ypos}'"
    cmd += f" {boot_dir}"
    subprocess.call(shlex.split(cmd), cwd=boot_dir)


# %% excepthook ===============================================================
def excepthook(exc_type, exc_value, exc_tb):
    with open('rtpspy.error', 'a') as fd:
        traceback.print_exception(exc_type, exc_value, exc_tb, file=fd)

    traceback.print_exception(exc_type, exc_value, exc_tb)
