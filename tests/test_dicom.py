"""DICOM handling on synthetic DICOM files made with pydicom (no real data):
RtpWatch reading (partial files, timeouts, file pattern, volume index),
RtDcmMonitor (exit, truncated files, TR of a series), DicomReader and
DicomConverter (file listing, real-time conversion)."""
import inspect
import logging
import shutil
import threading
import time
import types
from threading import Lock

import nibabel as nib
import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

import rtpspy.dicom_converter as dcmod
import rtpspy.rt_dicom_monitor as dmon
from rtpspy.dicom_converter import DicomConverter
from rtpspy.dicom_reader import DicomReader
from rtpspy.rtp_tshift import RtpTshift
from rtpspy.rtp_watch import RtpWatch


def synth_dicom(path, rows=8, cols=8, pixels=True, series=1, inst=1,
                acq=True):
    """Write a synthetic single-frame MR DICOM file (phantom, no PHI)."""
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    ds.SOPInstanceUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    ds.PatientName = "Test^Phantom"
    ds.ImageType = ["ORIGINAL", "PRIMARY", "M", "ND"]
    ds.SeriesDescription = "synth"
    ds.SeriesNumber = series
    ds.InstanceNumber = inst
    ds.AcquisitionNumber = 1
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyDate = "20240101"
    ds.StudyTime = "120000.000"
    if acq:
        ds.AcquisitionDateTime = "20240101120000.000000"
    else:
        ds.ContentDate = "20240101"
        ds.ContentTime = "120001.5"
    if pixels:
        ds.Rows = rows
        ds.Columns = cols
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelData = np.arange(rows * cols, dtype=np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)
    return path


@pytest.fixture
def full_and_part(tmp_path):
    """A complete 64x64 DICOM file and a copy truncated by 1000 bytes."""
    full = synth_dicom(tmp_path / "full.dcm", rows=64, cols=64)
    data = full.read_bytes()
    part = tmp_path / "part.dcm"
    part.write_bytes(data[:-1000])
    return full, part, data


# %% RtpWatch =================================================================
def test_watch_read_retries_partial_file(tmp_path, full_and_part):
    full, part, data = full_and_part
    w = RtpWatch(watch_dir=str(tmp_path), read_timeout=1.0)

    def finish():
        time.sleep(0.3)
        part.write_bytes(data)

    th = threading.Thread(target=finish)
    th.start()
    st = time.time()
    d = w._read_dicom_file(part)
    el = time.time() - st
    th.join()
    assert d is not None and d.pixel_array.shape == (64, 64)
    assert 0.25 < el < 1.0

    # Permanent truncation: gives up after read_timeout
    part.write_bytes(data[:-1000])
    st = time.time()
    assert w._read_dicom_file(part) is None
    assert 0.9 < time.time() - st < 1.5


def test_watch_non_image_and_garbage(tmp_path):
    w = RtpWatch(watch_dir=str(tmp_path), read_timeout=1.0)
    nopix = synth_dicom(tmp_path / "nopix.dcm", pixels=False)
    st = time.time()
    assert w._read_dicom_file(nopix) is not None
    assert time.time() - st < 0.3
    assert w.process_Dicom(nopix, 1) is None  # skipped: no image

    garbage = tmp_path / "g.dcm"
    garbage.write_bytes(b"x" * 10)
    st = time.time()
    assert w.process_Dicom(garbage, 1) is None
    assert time.time() - st < 1.5


def test_watch_pixel_array_rereads_file(tmp_path, full_and_part):
    full, part, data = full_and_part
    w = RtpWatch(watch_dir=str(tmp_path), read_timeout=1.0)
    dtr = pydicom.dcmread(full)
    dtr.PixelData = dtr.PixelData[:100]
    assert w._get_pixel_array(dtr, full).shape == (64, 64)
    with pytest.raises(Exception):
        w._get_pixel_array(dtr, None)


def test_watch_process_nibabel(tmp_path):
    w = RtpWatch(watch_dir=str(tmp_path), read_timeout=1.0)
    bad = tmp_path / "bad.nii"
    bad.write_bytes(b"0" * 100)
    st = time.time()
    assert w.process_nibabel(bad, 1) is None
    assert time.time() - st < 1.5
    good = tmp_path / "img_0001.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((4, 4, 3), np.float32), np.eye(4)),
             good)
    r = w.process_nibabel(good, 1)
    assert r is not None and r.shape == (4, 4, 3)


def test_watch_file_pattern_and_on_moved():
    calls = []
    h = RtpWatch.RTPFileHandler(r".+\.dcm$", callback=calls.append)

    def ev(**k):
        return types.SimpleNamespace(is_directory=False, **k)

    h.on_created(ev(src_path="/x/.a.dcm.AbC123"))  # temporary file
    h.on_created(ev(src_path="/x/a.dcm"))
    h.on_moved(ev(src_path="/x/.b.dcm.XyZ", dest_path="/x/b.dcm"))
    h.on_moved(ev(src_path="/x/c.dcm", dest_path="/x/c.dcm.bak"))
    assert calls == ["/x/a.dcm", "/x/b.dcm"]
    default = inspect.signature(RtpWatch.__init__).parameters[
        "watch_file_pattern"].default
    assert default == r".+\.dcm$"


def test_watch_vol_index_skips_unread_files(tmp_path):
    class Next:
        def __init__(self):
            self.idx = []

        def do_proc(self, img, vol_idx=None, pre_proc_time=None, **kw):
            self.idx.append(vol_idx)
            np.asanyarray(img.dataobj)[:] = -1  # modified in place

    nxt = Next()
    w = RtpWatch(watch_dir=str(tmp_path), next_proc=nxt, save_proc=True,
                 work_dir=str(tmp_path))
    w._vol_num = 0
    w._last_proc_f = ""
    seq = iter([None, "img", None, None, "img", "img", None, "img"])
    names = []

    def fake_process(file_path, vol_num):
        if next(seq) is None:
            return None
        names.append(vol_num)
        img = nib.Nifti1Image(np.full((2, 2, 2), vol_num, np.int16),
                              np.eye(4))
        img.set_filename(f"v_{vol_num:04d}.nii.gz")
        return img

    w.process_Dicom = fake_process
    for i in range(8):
        f = tmp_path / f"f{i}.dcm"
        f.write_bytes(b"1")
        w.do_proc(f)
    assert nxt.idx == [0, 1, 2, 3]
    assert w._vol_num == 4 and names == [1, 2, 3, 4]
    saved = w.saved_data
    assert saved is not None
    # Saved data keeps its own values despite the change downstream
    assert [saved[..., i].max() for i in range(4)] == [1, 2, 3, 4]


def test_watch_start_watching_twice(tmp_path):
    w = RtpWatch(watch_dir=str(tmp_path))
    try:
        w.start_watching()
        o1 = w._observer
        w.start_watching()
        assert not o1.is_alive() and w._observer.is_alive()
        assert not isinstance(w.watch_dir, str)
    finally:
        w.stop_watching()


def test_watch_read_timeout_param(tmp_path, full_and_part):
    full, part, data = full_and_part
    w = RtpWatch(watch_dir=str(tmp_path), read_timeout=3)
    w.set_param("read_timeout", 0.5, reset_fn=lambda x: None)
    assert w.read_timeout == 0.5
    w.set_param("read_timeout", 0.4)
    assert w.read_timeout == 0.4
    assert w.get_params().get("read_timeout") == 0.4
    st = time.time()
    assert w._read_dicom_file(part) is None
    assert 0.3 < time.time() - st < 0.9


# %% RtDcmMonitor =============================================================
class _Com:
    def rpc_ping(self):
        return False

    def call_rt_proc(self, *a, **k):
        return None


class _Srv:
    def __init__(self):
        self.n = 0

    def shutdown(self):
        self.n += 1


def make_monitor(watch_dir):
    """RtDcmMonitor without __init__ (no RPC server, no observer)."""
    m = object.__new__(dmon.RtDcmMonitor)
    m._logger = logging.getLogger("test")
    m.watch_dir = watch_dir
    m.work_root = watch_dir / "work"
    m.study_prefix = "P"
    m.study_ID_field = None
    m.series_timeout = 7
    m.make_brik = False
    m.no_online = True
    m.rt_mri_main_com = _Com()
    m.rt_physio_com = _Com()
    m.rt_mrib_com = _Com()
    m._dcmread_timeout = 1
    m.dcm_reader = DicomReader()
    m._current_study = -1
    m._current_series = -1
    m._isRun_series = False
    m._last_dicom_time = None
    m._last_dicom_header = None
    m._last_proc_f = None
    m._study_dir = None
    m._series_nr = 0
    m._TR = None
    m._acqNr = 0
    m._NVol = 0
    m._process_lock = Lock()
    m._cancel = False
    m._end_complete = False
    m._save_physio = False
    m.socket_srv = _Srv()
    return m


def test_monitor_exit(tmp_path):
    (tmp_path / "empty_sub").mkdir()
    m = make_monitor(tmp_path)
    m.exit()
    m.exit()  # the second call does nothing
    assert m.socket_srv.n == 1
    assert (tmp_path / "empty_sub").is_dir()  # no folder is deleted

    m2 = make_monitor(tmp_path)

    def fail():
        raise RuntimeError("stop_watching failed")

    m2.stop_watching = fail
    m2.exit()
    assert m2.socket_srv.n == 1  # shutdown runs anyway

    m3 = make_monitor(tmp_path)
    del m3.socket_srv  # __init__ failed before the server was made
    m3.exit()


def test_monitor_truncated_and_complete_file(tmp_path, full_and_part):
    full, part, data = full_and_part
    m = make_monitor(tmp_path)
    st = time.time()
    m.do_proc(str(part))
    el = time.time() - st
    assert 0.9 < el < 2.0
    assert m._NVol == 0 and m._study_dir is None
    assert not m._process_lock.locked()

    m = make_monitor(tmp_path)
    m.do_proc(str(full))
    assert m._NVol == 1 and m._isRun_series


def test_monitor_series_TR(tmp_path, full_and_part):
    full = full_and_part[0]
    m = make_monitor(tmp_path)
    m.init_series(pydicom.dcmread(full))
    assert m._TR == 7.0  # no TR: series_timeout

    d2 = pydicom.dcmread(full)
    d2.SharedFunctionalGroupsSequence = [Dataset()]
    m = make_monitor(tmp_path)
    m.init_series(d2)
    assert m._TR == 7.0  # shared group without MR timing: fallback

    d3 = pydicom.dcmread(full)
    d3.RepetitionTime = 1500
    m = make_monitor(tmp_path)
    m.init_series(d3)
    assert m._TR == 1.5


# %% DicomReader, RtpTshift ===================================================
def test_dicom_reader_defaults():
    dr = DicomReader()
    ps = inspect.signature(dr.dcm2nii).parameters
    assert ps["dcms"].default is None
    assert "timeout" in ps and "time_out" not in ps
    assert dr.dcm2nii() is None
    mv = Dataset()
    mv.SoftwareVersions = ["syngo MR XA30", "x"]
    assert dr._software_versions(mv) == "syngo MR XA30\\x"


def test_tshift_rejects_empty_slice_timing():
    ts = RtpTshift(TR=2.0)
    ts.err_popup = lambda *a, **k: None
    assert ts.ready_proc() is False  # default []
    ts.slice_timing = np.array([])
    assert ts.ready_proc() is False
    ts.slice_timing = [0.0, 1.0]
    assert ts.ready_proc() is True


# %% DicomConverter ===========================================================
def test_converter_list_files(tmp_path):
    src = tmp_path / "conv_src"
    src.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    for i in range(3):
        synth_dicom(src / f"a{i}.dcm", series=3, inst=i + 1)
    synth_dicom(src / "noacq.dcm", series=3, inst=4, acq=False)
    (src / "junk.dcm").write_bytes(b"junk")

    conv = DicomConverter()
    info = conv._list_dicom_files(src, out)
    assert len(info) == 4
    assert info.SeriesNumber.dtype.kind == "i" and info.IsImage.all()
    # ContentDate + ContentTime fallback
    assert abs(info.loc[str(src / "noacq.dcm"), "AcquisitionDateTime"]
               - 20240101120001.5) < 1e-3
    assert len(conv._list_dicom_files(src, out)) == 4  # from the saved CSV


@pytest.mark.skipif(shutil.which("rsync") is None, reason="needs rsync")
def test_converter_keeps_late_file(tmp_path, monkeypatch):
    src = tmp_path / "rt_src"
    src.mkdir()
    outd = tmp_path / "rt_out"
    outd.mkdir()
    for i in range(3):
        synth_dicom(src / f"s{i}.dcm", series=5, inst=i + 1)

    real_cc = dcmod.subprocess.check_call
    n = [0]

    def check_call(cmd, *a, **k):
        r = real_cc(cmd, *a, **k)
        n[0] += 1
        if n[0] == 1:  # a file arrives after the first copy
            synth_dicom(src / "s9.dcm", series=5, inst=10)
        return r

    monkeypatch.setattr(dcmod.subprocess, "check_call", check_call)
    conv = DicomConverter()
    conv.convert_dicom = lambda *a, **k: None
    conv.rt_convert_dicom(src, outd)

    assert sorted(p.name for p in src.iterdir()) == ["s9.dcm"]
    assert sorted(p.name for p in (outd / "dicom").iterdir()) == [
        "s0.dcm", "s1.dcm", "s2.dcm"]
