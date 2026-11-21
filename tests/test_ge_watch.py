"""RtpWatch with GE DICOM (one slice per file): slice assembly into volumes,
temporal order, series selection, and the receiver of ge_dicom_sender.py.
Synthetic phantom data only (no PHI)."""
import socket
import struct
import time

import nibabel as nib
import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from rtpspy.rtp_watch import RtpWatch

NX, NY, NSL, NT = 6, 5, 4, 3


def ge_slice(path, series_uid, tpos, pos, nt=NT, nsl=NSL, echo=1,
             series_nr=5, desc="fMRI", no_tpos=False):
    """Write a synthetic GE slice file. The pixel value encodes the volume,
    the slice, the row and the column: 1000*tpos + 100*pos + 10*row + col.
    no_tpos: no TemporalPositionIdentifier (older GE software)."""
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    ds.SOPInstanceUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    ds.Manufacturer = "GE MEDICAL SYSTEMS"
    ds.PatientName = "Test^Phantom"
    ds.ImageType = ["ORIGINAL", "PRIMARY", "EPI"]
    ds.SeriesDescription = desc
    ds.SeriesNumber = series_nr
    ds.StudyInstanceUID = "1.2.3.4"
    ds.SeriesInstanceUID = series_uid
    ds.StudyDate = "20240101"
    ds.StudyTime = "120000"
    ds.RepetitionTime = 2000
    ds.NumberOfTemporalPositions = nt
    ds.ImagesInAcquisition = nt * nsl
    ds.add_new((0x0021, 0x104F), "SS", nsl)  # Locations in acquisition
    if not no_tpos:
        ds.TemporalPositionIdentifier = tpos
    ds.InStackPositionNumber = pos
    ds.EchoNumbers = echo
    ds.InstanceNumber = (tpos - 1) * nsl + pos
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [-10.0, -12.0, 3.0 * (pos - 1)]
    ds.PixelSpacing = [2.0, 2.5]  # Row spacing, column spacing
    ds.SliceThickness = 3.0
    ds.Rows, ds.Columns = NY, NX
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    r, c = np.mgrid[0:NY, 0:NX]
    arr = (1000 * tpos + 100 * pos + 10 * r + c).astype(np.uint16)
    ds.PixelData = arr.tobytes()
    ds.save_as(path, enforce_file_format=True)
    return path


def _feed(w, files):
    out = []
    for f in files:
        imgs = w.process_Dicom(f, w._vol_num + 1)
        if imgs:
            out.extend(imgs)
            w._vol_num += len(imgs)
    return out


def test_ge_volumes_in_temporal_order(tmp_path):
    w = RtpWatch(watch_dir=str(tmp_path), work_dir=str(tmp_path))
    w._ge_reset()
    uid = generate_uid()
    files = []
    # Volume 2 is completed before volume 1 (multiband reconstruction);
    # slices come out of order; a second echo is ignored.
    order = [(2, p) for p in (3, 1, 4, 2)] + [(1, p) for p in (2, 4, 1)] \
        + [(1, 3)] + [(3, p) for p in (1, 2, 3, 4)]
    for i, (t, p) in enumerate(order):
        files.append(ge_slice(tmp_path / f"i{i}.MRDC.{i + 1}.dcm", uid, t, p))
    echo2 = ge_slice(tmp_path / "e2.dcm", uid, 3, 1, echo=2)

    out = []
    for f in files[:4]:
        out += _feed(w, [f])
    assert out == []  # Volume 2 waits for volume 1
    out += _feed(w, files[4:8])
    assert len(out) == 2  # Volume 1 completes: volumes 1 and 2 go out
    out += _feed(w, [echo2])
    out += _feed(w, files[8:])
    assert len(out) == 3

    for t, img in enumerate(out, start=1):
        data = np.asanyarray(img.dataobj)
        assert data.shape == (NX, NY, NSL)
        assert img.header.get_zooms()[:3] == pytest.approx((2.5, 2.0, 3.0))
        # LPI: the row axis is flipped (the first stored row is the last
        # in the image)
        for pos in range(1, NSL + 1):
            sl = data[:, :, pos - 1]
            r, c = np.mgrid[0:NY, 0:NX]
            expect = (1000 * t + 100 * pos + 10 * r + c).T[:, ::-1]
            assert np.array_equal(sl, expect)
        # The world coordinate of voxel (0, NY-1, 0) is the first stored
        # pixel: LPS (-10, -12, 0) -> RAS (10, 12, 0)
        xyz = img.affine @ [0, NY - 1, 0, 1]
        assert xyz[:3] == pytest.approx([10.0, 12.0, 0.0])
    assert img.get_filename().endswith("_0003.nii.gz")


def test_ge_volumes_without_temporal_position(tmp_path):
    """Older GE software writes no TemporalPositionIdentifier: the volume is
    taken from InstanceNumber."""
    w = RtpWatch(watch_dir=str(tmp_path), work_dir=str(tmp_path))
    w._ge_reset()
    uid = generate_uid()
    files = [ge_slice(tmp_path / f"i{t}_{p}.dcm", uid, t, p, no_tpos=True)
             for t in range(1, NT + 1) for p in range(1, NSL + 1)]
    assert "TemporalPositionIdentifier" not in pydicom.dcmread(files[-1])
    out = _feed(w, files)
    assert len(out) == NT
    for t, img in enumerate(out, start=1):
        data = np.asanyarray(img.dataobj)
        assert data.min() // 1000 == t and data.max() // 1000 == t


def test_ge_series_selection(tmp_path):
    w = RtpWatch(watch_dir=str(tmp_path), work_dir=str(tmp_path))
    w._ge_reset()
    # A localizer (one volume) is not processed
    loc = [ge_slice(tmp_path / f"l{p}.dcm", "1.9.1", 1, p, nt=1, desc="loc")
           for p in range(1, NSL + 1)]
    assert _feed(w, loc) == []
    assert w._ge_series is None
    # The first fMRI series is kept; files of another series are ignored
    uid = generate_uid()
    a = [ge_slice(tmp_path / f"a{p}.dcm", uid, 1, p) for p in range(1, 3)]
    other = [ge_slice(tmp_path / f"o{p}.dcm", "1.9.2", 1, p)
             for p in range(1, NSL + 1)]
    b = [ge_slice(tmp_path / f"b{p}.dcm", uid, 1, p) for p in range(3, 5)]
    assert _feed(w, a + other) == []
    assert len(_feed(w, b)) == 1
    assert w._ge_series == uid


def test_ge_missing_volume_is_skipped(tmp_path, monkeypatch):
    w = RtpWatch(watch_dir=str(tmp_path), work_dir=str(tmp_path))
    w._ge_reset()
    uid = generate_uid()
    v1 = [ge_slice(tmp_path / f"a{p}.dcm", uid, 1, p) for p in range(1, 4)]
    v2 = [ge_slice(tmp_path / f"b{p}.dcm", uid, 2, p)
          for p in range(1, NSL + 1)]
    v3 = [ge_slice(tmp_path / f"c{p}.dcm", uid, 3, p)
          for p in range(1, NSL + 1)]
    assert _feed(w, v1 + v2) == []  # Volume 1 lacks a slice
    t0 = time.time()
    monkeypatch.setattr(time, "time", lambda: t0 + 31.0)
    out = _feed(w, v3)
    # After max(30 s, 10 TR), volume 1 is skipped: volumes 2 and 3 go out
    assert len(out) == 2
    assert np.asanyarray(out[0].dataobj).min() >= 2000


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_ge_receiver_writes_into_watch_dir(tmp_path):
    port = _free_port()
    w = RtpWatch(watch_dir=str(tmp_path), work_dir=str(tmp_path),
                 ge_port=port, ge_allow_hosts="127.0.0.1")
    try:
        assert w._ge_server is not None
        src = ge_slice(tmp_path / "src.dcm", generate_uid(), 1, 1)
        data = src.read_bytes()
        name = b"p1/e2/s3/i100.MRDC.1"
        with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
            s.sendall(b"GEDC" + struct.pack(">HI", len(name), len(data))
                      + name + data)
            assert s.recv(1) == b"K"
            bad = b"../x"
            s.sendall(b"GEDC" + struct.pack(">HI", len(bad), 1) + bad + b"0")
            assert s.recv(1) == b"E"
        out_f = tmp_path / "p1/e2/s3/i100.MRDC.1.dcm"
        assert out_f.read_bytes() == data
        assert pydicom.dcmread(out_f).TemporalPositionIdentifier == 1
        assert not (tmp_path.parent / "x.dcm").exists()

        # Changing the port restarts the receiver
        port2 = _free_port()
        w.set_param("ge_port", port2)
        with socket.create_connection(("127.0.0.1", port2), timeout=5):
            pass
        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=1)
    finally:
        w.stop_ge_receiver()

