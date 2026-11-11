"""Run RTPSpy without PyQt6 (helper of test_headless.py).

PyQt6 imports are blocked with a sys.meta_path finder before rtpspy is
imported, so any Qt import in the processing code raises ImportError.

usage: python _headless_run.py <chain|make_masks> <work_dir>

chain       Run the full processing chain WATCH -> VOLREG -> TSHIFT ->
            SMOOTH -> REGRESS -> RtpApp with RtpApp.RTP_setup, ready_to_run,
            manual_start, and end_run on tests/func_epi.nii.gz (cropped in
            plane) with synthetic masks, feeding the files directly to WATCH
            (as rtpspy_system_check.py --debug does).
make_masks  Run RtpApp.make_masks with the image processing steps
            (RtpImgProc) replaced by stubs that write dummy files, with and
            without progress_dlg.

A JSON summary is printed on the last line of stdout.
"""
import sys
import os
import json
import time
import shutil
from pathlib import Path


# %% Block PyQt6 ==============================================================
class _BlockQt:
    def find_spec(self, name, path=None, target=None):
        if name == "PyQt6" or name.startswith("PyQt6."):
            raise ImportError(f"PyQt6 is blocked ({name})")
        return None


sys.meta_path.insert(0, _BlockQt())

TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_DIR.parent))

import numpy as np  # noqa: E402
import nibabel as nib  # noqa: E402

import rtpspy  # noqa: E402
from rtpspy import RtpApp  # noqa: E402
import rtpspy.rtp_app as rtp_app_mod  # noqa: E402


# %% helpers ==================================================================
def proc_argv_alive(argv):
    """PIDs of the processes whose argv equals argv exactly."""
    pids = []
    for d in Path("/proc").glob("[0-9]*"):
        try:
            a = (d / "cmdline").read_bytes().split(b"\0")[:-1]
        except OSError:
            continue
        if [x.decode(errors="replace") for x in a] == list(argv):
            pids.append(int(d.name))
    return pids


def make_app(work_dir):
    """RtpApp with a unique rt_physio socket name and a scratch config."""
    sock_name = f"RtTTLPhysioSocketServer_headless_{os.getpid()}"
    app = RtpApp(work_dir=work_dir,
                 physio_log_file=work_dir / "rt_physio.log",
                 rt_physio_address_name=["localhost", None, sock_name],
                 config_path=work_dir / "rtmri_config.json")
    proc = app.rtp_objs["TTLPHYSIO"]._rt_physio_proc
    argv = list(map(str, proc.args)) if proc is not None else None
    return app, argv


def quit_physio(app, argv):
    app.rtp_objs["TTLPHYSIO"].quit()
    if argv is None:
        return []
    for _ in range(50):
        left = proc_argv_alive(argv)
        if not left:
            break
        time.sleep(0.1)
    return left


def save_mask(data, affine, fname):
    nib.save(nib.Nifti1Image(data.astype(np.int16), affine), fname)
    return fname


# %% run_chain ================================================================
def run_chain(work_dir, n_vols=40):
    # Crop the test data in plane (the center 64 x 64 voxels) to run fast
    img = nib.load(TEST_DIR / "func_epi.nii.gz")
    img = img.slicer[32:96, 32:96, :, :n_vols]
    data = np.asanyarray(img.dataobj)
    mean = data.mean(axis=-1)

    # fMRI parameter reference with the slice timing sidecar
    mask_dir = work_dir / "masks"
    mask_dir.mkdir()
    func_f = mask_dir / "func_ref.nii.gz"
    nib.save(img, func_f)
    shutil.copy(TEST_DIR / "func_epi.json", mask_dir / "func_ref.json")

    # Synthetic masks in the function space
    brain = mean > np.percentile(mean, 60)
    shape = brain.shape
    c = [s // 2 for s in shape]
    box = np.zeros(shape, bool)
    box[c[0]-3:c[0]+3, c[1]-3:c[1]+3, c[2]-2:c[2]+2] = True
    wm = np.zeros(shape, bool)
    wm[c[0]-6:c[0]-2, c[1]-2:c[1]+2, c[2]-2:c[2]+2] = True
    vent = np.zeros(shape, bool)
    vent[c[0]+2:c[0]+6, c[1]-2:c[1]+2, c[2]-2:c[2]+2] = True
    masks = {
        "RTP_mask": save_mask(brain, img.affine, mask_dir / "RTP_mask.nii.gz"),
        "GSR_mask": save_mask(brain, img.affine, mask_dir / "GSR_mask.nii.gz"),
        "WM_orig": save_mask(wm, img.affine,
                             mask_dir / "WM.nii.gz"),
        "Vent_orig": save_mask(vent, img.affine, mask_dir / "Vent.nii.gz"),
        "ROI_orig": save_mask(box, img.affine, mask_dir / "ROI.nii.gz"),
    }
    func_orig = mask_dir / "vr_base.nii.gz"
    nib.save(nib.Nifti1Image(data[..., 0], img.affine), func_orig)

    watch_dir = work_dir / "watch"
    watch_dir.mkdir()
    app, physio_argv = make_app(work_dir)
    res = {}
    try:
        app.func_param_ref = func_f
        app.func_orig = func_orig
        for attr, f in masks.items():
            setattr(app, attr, f)

        rtp_params = {
            "WATCH": {"watch_dir": watch_dir,
                      "watch_file_pattern": r"nr_\d+.*\.nii",
                      "file_type": "Nifti"},
            "VOLREG": {"regmode": "cubic"},
            "TSHIFT": {"method": "cubic", "ignore_init": 2},
            "SMOOTH": {"blur_fwhm": 6.0},
            "REGRESS": {"mot_reg": "mot12", "GS_reg": True, "WM_reg": True,
                        "Vent_reg": True, "phys_reg": "None",
                        "wait_num": 20, "onGPU": False},
        }
        res["RTP_setup"] = app.RTP_setup(rtp_params=rtp_params)
        for m in ("WATCH", "VOLREG", "TSHIFT", "SMOOTH", "REGRESS"):
            app.rtp_objs[m].save_proc = True

        proc_chain = app.ready_to_run()
        res["chain_head"] = type(proc_chain).__name__
        app.rtp_objs["WATCH"].stop_watching()
        app.manual_start()

        for ii in range(n_vols):
            f = watch_dir / f"headless_nr_{ii + 1:04d}.nii.gz"
            nib.save(nib.Nifti1Image(data[..., ii], img.affine), f)
            proc_chain.do_proc(f)

        res["n_roi_signal"] = len(app._roi_sig[0])
        saved = app.end_run(scan_name="headless")
        res["saved"] = sorted(saved.keys())
        res["saved_exist"] = all(Path(f).is_file() for f in saved.values())
        regress_img = saved.get("RtpRegress:processed image")
        if regress_img is not None:
            res["regress_shape"] = list(nib.load(regress_img).shape)
    finally:
        res["physio_left"] = quit_physio(app, physio_argv)

    return res


# %% run_make_masks ===========================================================
class StubImgProc(rtp_app_mod.RtpImgProc):
    """RtpImgProc whose processing steps write dummy files."""

    def _touch(self, out_dir, name):
        f = Path(out_dir) / name
        nib.save(nib.Nifti1Image(np.ones((4, 4, 4), np.int16), np.eye(4)), f)
        return f

    def copy_deoblique(self, src, dst, progress_bar=None, overwrite=False):
        nib.save(nib.load(src), dst)

    def run_fast_seg(self, out_dir, anat, total_ETA, progress_bar=None,
                     overwrite=False):
        return [self._touch(out_dir, f"anat_{n}.nii.gz")
                for n in ("Brain", "WM", "Vent", "aseg")]

    def align_anat2epi(self, out_dir, brain, func, total_ETA,
                       progress_bar=None, ask_cmd=False, overwrite=False):
        f = self._touch(out_dir, "anat_Brain_alFunc.nii.gz")
        (Path(out_dir) / "anat_Brain_alFunc_mat.aff12.1D").write_text("0\n")
        return f

    def make_RTP_GSR_masks(self, out_dir, func, total_ETA, ref_vi=0,
                           alAnat=None, progress_bar=None, ask_cmd=False,
                           overwrite=False):
        return [self._touch(out_dir, "RTP_mask.nii.gz"),
                self._touch(out_dir, "GSR_mask.nii.gz")]

    def resample_segmasks(self, out_dir, seg_f, segname, erode, func,
                          total_ETA, aff1D_f=None, progress_bar=None,
                          ask_cmd=False, overwrite=False):
        return self._touch(out_dir, f"{segname}_alFunc.nii.gz")


def run_make_masks(work_dir):
    rtp_app_mod.RtpImgProc = StubImgProc
    app, physio_argv = make_app(work_dir)
    res = {}
    try:
        func = TEST_DIR / "func_epi.nii.gz"
        anat = work_dir / "anat.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((8, 8, 8), np.int16), np.eye(4)),
                 anat)
        for key, kwargs in (("plain", {}),
                            ("progress_dlg", {"progress_dlg": True,
                                              "overwrite": True})):
            ret = app.make_masks(func_orig=str(func) + "'[0]'",
                                 anat_orig=anat, no_FastSeg=False, **kwargs)
            res[key] = {
                "ret": ret,
                "masks": {a: Path(getattr(app, a)).name for a in
                          ("RTP_mask", "GSR_mask", "WM_orig", "Vent_orig",
                           "aseg_orig", "alAnat")},
                "stdout_restored": sys.stdout is sys.__stdout__,
            }
    finally:
        res["physio_left"] = quit_physio(app, physio_argv)

    return res


# %% main =====================================================================
if __name__ == "__main__":
    mode, work_dir = sys.argv[1], Path(sys.argv[2])
    work_dir.mkdir(parents=True, exist_ok=True)
    if mode == "chain":
        res = run_chain(work_dir)
    else:
        res = run_make_masks(work_dir)
    res["rtpspy"] = rtpspy.__file__
    res["qt_modules"] = sorted(m for m in sys.modules
                               if m.startswith("PyQt6"))
    print(json.dumps(res))
