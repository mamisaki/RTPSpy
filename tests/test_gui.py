"""RtpGUI (offscreen): the module panels, and the synchronization of the
widgets and the parameters in both directions."""
import os
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest

QtWidgets = pytest.importorskip("PyQt5.QtWidgets")

from rtpspy import RtpApp, RtpGUI  # noqa: E402
from rtpspy import notifier as nt  # noqa: E402
from rtpspy.ui import QtNotifier, RtpModuleUI  # noqa: E402
from rtpspy.ui.module_ui import ZERO_OUT_MASK_TEXT  # noqa: E402

TEST_DIR = Path(__file__).resolve().parent


def _proc_argv_alive(argv):
    pids = []
    for d in Path("/proc").glob("[0-9]*"):
        try:
            a = (d / "cmdline").read_bytes().split(b"\0")[:-1]
        except OSError:
            continue
        if [x.decode(errors="replace") for x in a] == list(argv):
            pids.append(int(d.name))
    return pids


@pytest.fixture(scope="module")
def gui_app(tmp_path_factory):
    work_dir = tmp_path_factory.mktemp("gui")
    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    sock_name = f"RtTTLPhysioSocketServer_testgui_{os.getpid()}"
    app = RtpApp(work_dir=work_dir,
                 physio_log_file=work_dir / "rt_physio.log",
                 rt_physio_address_name=["localhost", None, sock_name],
                 config_path=work_dir / "rtmri_config.json")
    proc = app.rtp_objs["TTLPHYSIO"]._rt_physio_proc
    physio_argv = list(map(str, proc.args)) if proc is not None else None
    try:
        gui = RtpGUI(app.rtp_objs, {"RTP App": app}, log_file=None)
        qt_app.processEvents()
        yield qt_app, app, gui, work_dir
        gui.close()
        qt_app.processEvents()
    finally:
        nt.set_notifier(None)
        app.rtp_objs["TTLPHYSIO"].quit()
        if physio_argv is not None:
            for _ in range(50):
                if not _proc_argv_alive(physio_argv):
                    break
                time.sleep(0.1)
            assert _proc_argv_alive(physio_argv) == []


def test_panels(gui_app):
    qt_app, app, gui, _ = gui_app
    assert isinstance(nt.get_notifier(), QtNotifier)
    n_rows = {"WATCH": 8, "VOLREG": 7, "TSHIFT": 8, "SMOOTH": 5,
              "REGRESS": 13}
    for proc, n in n_rows.items():
        obj = app.rtp_objs[proc]
        assert isinstance(obj._ui, RtpModuleUI)
        assert obj.main_win is gui
        assert gui.RTPSetPanes[proc].layout().rowCount() == n, proc
        assert obj.ui_enabled_rdb.isChecked() == obj.enabled
        assert len(obj.ui_objs) > 0
    assert isinstance(app._ui, RtpModuleUI)
    assert gui.AppSetPanes["RTP App"].layout().rowCount() == 3
    tabs = app.ui_top_tabs
    assert [tabs.tabText(i) for i in range(tabs.count())] == [
        "Preprocessing", "Template", "Processed images", "Output"]
    assert not app.ui_ready_btn.isEnabled()
    # TTLPHYSIO has no panel
    assert app.rtp_objs["TTLPHYSIO"].ui_set_param() is None


def test_widget_to_param(gui_app):
    qt_app, app, gui, _ = gui_app
    W, V, T, S, R = (app.rtp_objs[m] for m in
                     ("WATCH", "VOLREG", "TSHIFT", "SMOOTH", "REGRESS"))
    W.ui_dcmreadTimeout_dSpBx.setValue(4.5)
    W.ui_pollingObserver_chb.setChecked(True)
    W.ui_watchPat_lnEd.setText(r"nr_\d+.*\.nii")
    W.ui_watchPat_lnEd.editingFinished.emit()
    V.ui_regmode_cmbBx.setCurrentIndex(1)
    V.ui_maxLen_spBx.setValue(500)
    V.ui_maxLen_spBx.editingFinished.emit()
    T.ui_method_cmbBx.setCurrentText("linear")
    T.ui_TR_dSpBx.setValue(1.5)
    T.ui_SlTiming_lnEd.setText("[0.0, 0.5, 1.0]")
    T.ui_SlTiming_lnEd.editingFinished.emit()
    S.ui_FWHM_dSpBx.setValue(4.0)
    R.ui_motReg_cmbBx.setCurrentIndex(2)
    R.ui_GS_reg_chb.setChecked(True)
    R.ui_censorFD_spBx.setValue(0.3)
    R.ui_censorFD_spBx.editingFinished.emit()
    app.ui_ROI_resample_cmbBx.setCurrentIndex(1)
    app.ui_no_FastSeg_chb.setChecked(True)
    qt_app.processEvents()

    assert W.read_timeout == 4.5
    assert W.polling_observer is True
    assert W.watch_file_pattern == r"nr_\d+.*\.nii"
    assert V.regmode == "cubic"
    assert V.max_scan_length == 500
    assert T.method == "linear"
    assert T.TR == 1.5
    assert T.slice_timing == [0.0, 0.5, 1.0]
    assert S.blur_fwhm == 4.0
    assert R.mot_reg == "mot12"
    assert R.GS_reg is True
    assert R.censor_FD == pytest.approx(0.3)
    # wait_num follows the number of regressors in the GUI default mode
    assert R.wait_num == R.get_reg_num() + 1
    assert app.ROI_resample == "linear"
    assert app.no_FastSeg

    # Disabling a module disables its widgets
    V.ui_enabled_rdb.setChecked(False)
    assert V.enabled is False
    assert not V.ui_regmode_cmbBx.isEnabled()
    V.ui_enabled_rdb.setChecked(True)
    assert V.enabled and V.ui_regmode_cmbBx.isEnabled()


def test_param_to_widget(gui_app):
    qt_app, app, gui, work_dir = gui_app
    W, V, T, S, R = (app.rtp_objs[m] for m in
                     ("WATCH", "VOLREG", "TSHIFT", "SMOOTH", "REGRESS"))
    img = nib.load(TEST_DIR / "func_epi.nii.gz")
    mask_f = work_dir / "mask.nii.gz"
    nib.save(nib.Nifti1Image(
        (np.asanyarray(img.dataobj[..., 0]) > 500).astype(np.int16),
        img.affine), mask_f)
    desmtx_f = work_dir / "desmtx.csv"
    np.savetxt(desmtx_f, np.ones((150, 2)), delimiter=",")

    W.set_param("read_timeout", 2.25)
    W.set_param("file_type", "AFNI_BRIK")
    W.set_param("watch_dir", work_dir)
    V.set_param("regmode", "quintic")
    V.set_param("ref_vol", str(TEST_DIR / "func_epi.nii.gz") + "[0]")
    T.set_param("set_from_sample", TEST_DIR / "func_epi.nii.gz")
    S.set_param("mask_file", mask_f)
    R.set_param("mot_reg", "mot6")
    R.set_param("max_poly_order", 3)
    R.set_param("wait_num", 40)
    R.set_param("desMtx_f", desmtx_f)
    R.set_param("GS_mask", mask_f)
    app.set_param("enable_RTP", 0)
    app.set_param("ROI_orig", mask_f)
    qt_app.processEvents()

    assert W.ui_dcmreadTimeout_dSpBx.value() == 2.25
    assert W.ui_watchFType_cmbBx.currentText() == "AFNI_BRIK"
    assert W.ui_wdir_lnEd.text() == str(work_dir)
    assert gui.lineEditWatchDir.text() == str(work_dir)
    assert V.ui_regmode_cmbBx.currentText() == "Quintic"
    assert V.ui_baseVol_lnEd.text().endswith("func_epi.nii.gz[0]")
    assert T.ui_TR_dSpBx.value() == pytest.approx(T.TR)
    assert T.ui_SlTiming_lnEd.text() == str(T.slice_timing)
    assert S.ui_mask_lnEd.text() == str(mask_f)
    assert S.ui_mask_cmbBx.currentIndex() == 0
    assert R.ui_motReg_cmbBx.currentIndex() == 1
    assert R.ui_maxPoly_cmbBx.currentIndex() == 1
    assert R.ui_maxPoly_lb.text().endswith("up to 3")
    assert R.ui_waitNum_cmbBx.currentIndex() == 1
    assert R.ui_waitNum_lb.text() == \
        "Wait REGRESS until receiving 40 volumes"
    assert R.ui_loadDesMtx_btn.text() == "Unset"
    assert R.ui_showDesMtx_btn.isEnabled()
    assert R.ui_GS_mask_lnEd.text() == str(mask_f)
    assert app.ui_enableRTP_chb.checkState() == 0
    assert app.ui_ROI_orig_lnEd.text() == str(mask_f)

    S.set_param("mask_file", 0)
    R.set_param("desMtx_f", "unset")
    app.set_param("ROI_orig", "")
    assert S.ui_mask_lnEd.text() == ZERO_OUT_MASK_TEXT
    assert R.ui_loadDesMtx_btn.text() == "Set"
    assert not R.ui_showDesMtx_btn.isEnabled()
    assert app.ui_ROI_orig_lnEd.text() == ""

    # ui_setEnabled disables the widgets of all modules
    app.ui_setEnabled(False)
    assert not W.ui_watchPat_lnEd.isEnabled()
    assert not gui.btnSetWorkDir.isEnabled()
    app.ui_setEnabled(True)
    assert W.ui_watchPat_lnEd.isEnabled()


def test_plot_windows(gui_app):
    qt_app, app, gui, _ = gui_app
    vr_ui = app.rtp_objs["VOLREG"]._ui
    gui.chbShowMotion.setCheckState(2)
    qt_app.processEvents()
    assert vr_ui.thPltMotion.isRunning()
    gui.chbShowMotion.setCheckState(0)
    qt_app.processEvents()
    assert not hasattr(vr_ui, "thPltMotion")

    app.ui_showROISig_cbx.setCheckState(2)
    qt_app.processEvents()
    assert app._ui.pltROISig.isRunning()
    app.ui_showROISig_cbx.setCheckState(0)
    qt_app.processEvents()
    assert not hasattr(app._ui, "pltROISig")
