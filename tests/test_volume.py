"""RtpVolume, the volume object passed along the processing chain, and the
handling of the volume in RTP.do_proc (a module outputs a new volume and
does not change the one it receives)."""
import copy

import nibabel as nib
import numpy as np

from rtpspy import RTP, RtpVolume
from rtpspy.rtp_smooth import RtpSmooth


def _volume(**kw):
    data = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    affine = np.diag([2.0, 2.0, 3.0, 1.0])
    return RtpVolume(data, affine, filename="sub_nr_0005.nii.gz", **kw)


class Recorder:
    """Downstream process that records the received volumes."""

    def __init__(self):
        self.imgs = []
        self.kw = []

    def do_proc(self, img, vol_idx=None, pre_proc_time=None, **kwargs):
        self.imgs.append(img)
        self.kw.append((vol_idx, pre_proc_time))


class Double(RTP):
    """Module doubling the data."""
    _proc_label = "Double"
    _out_prefix = "x."

    def _process(self, fmri_img, vol_idx, pre_proc_time):
        return fmri_img.with_data(
            np.asarray(fmri_img.dataobj, dtype=np.float32) * 2)


class PassThrough(RTP):
    """Module returning the received volume."""
    _proc_label = "Pass"
    _out_prefix = "p."

    def _process(self, fmri_img, vol_idx, pre_proc_time):
        return fmri_img


def test_volume_is_nifti1image():
    v = _volume(vol_idx=4, acq_time=10.0)
    assert isinstance(v, nib.Nifti1Image)
    assert v.get_filename() == "sub_nr_0005.nii.gz"
    assert v.vol_idx == 4 and v.acq_time == 10.0 and v.proc_time is None
    assert v.get_fdata().shape == (2, 3, 4)
    assert np.array_equal(v.affine, np.diag([2.0, 2.0, 3.0, 1.0]))
    assert v.get_data_dtype() == np.int16
    # nibabel operations keep working
    assert isinstance(copy.deepcopy(v), RtpVolume)
    assert v.slicer[:1].shape == (1, 3, 4)


def test_with_data_new_volume_input_unchanged():
    v = _volume(vol_idx=4, acq_time=10.0, proc_time=11.0)
    data0 = np.asanyarray(v.dataobj).copy()
    hdr0 = v.header.binaryblock
    new = np.ones((2, 3, 4), np.float32)
    w = v.with_data(new)
    assert w is not v and isinstance(w, RtpVolume)
    assert w.header is not v.header
    # the input is not changed
    assert np.array_equal(np.asanyarray(v.dataobj), data0)
    assert v.header.binaryblock == hdr0
    assert v.get_filename() == "sub_nr_0005.nii.gz"
    # the new volume
    assert np.asanyarray(w.dataobj) is new
    assert w.get_data_dtype() == np.float32
    assert np.array_equal(w.affine, v.affine)
    assert w.get_filename() == "sub_nr_0005.nii.gz"
    assert (w.vol_idx, w.acq_time, w.proc_time) == (4, 10.0, 11.0)
    # filename and vol_idx can be replaced
    u = v.with_data(new, filename="other.nii.gz", vol_idx=2)
    assert u.get_filename() == "other.nii.gz" and u.vol_idx == 2
    assert v.get_filename() == "sub_nr_0005.nii.gz" and v.vol_idx == 4
    # changing the new header does not change the input
    w.header["descrip"] = b"changed"
    assert v.header.binaryblock == hdr0


def test_as_volume():
    v = _volume()
    assert RtpVolume.as_volume(v) is v
    img = nib.Nifti1Image(np.zeros((2, 2, 2), np.float32), np.eye(4))
    img.set_filename("plain.nii.gz")
    rv = RtpVolume.as_volume(img, vol_idx=3)
    assert isinstance(rv, RtpVolume) and rv is not img
    assert rv.get_filename() == "plain.nii.gz" and rv.vol_idx == 3
    assert np.asanyarray(rv.dataobj) is np.asanyarray(img.dataobj)
    assert not hasattr(img, "vol_idx")


def test_do_proc_plain_nib_input_and_no_mutation():
    rec = Recorder()
    mod = Double(next_proc=rec, save_proc=True)
    data = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    img = nib.Nifti1Image(data, np.eye(4))
    img.set_filename("run_nr_0001.nii.gz")
    mod.do_proc(img, 0, 123.0)

    # The plain nibabel input is accepted and not changed
    assert type(img) is nib.Nifti1Image
    assert img.get_filename() == "run_nr_0001.nii.gz"
    assert np.array_equal(np.asanyarray(img.dataobj), data)

    out = rec.imgs[0]
    assert isinstance(out, RtpVolume) and isinstance(out, nib.Nifti1Image)
    assert out.get_filename() == "x.run_nr_0001.nii.gz"
    assert np.array_equal(out.get_fdata(), data * 2)
    assert out.vol_idx == 0 and out.proc_time == rec.kw[0][1]
    # The module output is kept for saving
    assert mod.saved_files[-1].endswith("x.run_nr_0001.nii.gz")
    assert np.array_equal(mod.saved_data[..., 0], data * 2)
    assert np.asanyarray(mod.proc_data) is np.asanyarray(out.dataobj)


def test_do_proc_returned_input_is_not_renamed():
    rec = Recorder()
    mod = PassThrough(next_proc=rec)
    v = _volume(vol_idx=0)
    mod.do_proc(v, 0, None)
    assert v.get_filename() == "sub_nr_0005.nii.gz"
    out = rec.imgs[0]
    assert out is not v and out.get_filename() == "p.sub_nr_0005.nii.gz"


def test_kept_output_unchanged_by_downstream(tmp_path):
    """A module's kept output is not changed by the next module (no restore
    is needed): Double -> RtpSmooth."""
    smooth = RtpSmooth(blur_fwhm=4.0, save_proc=True, work_dir=tmp_path)
    mod = Double(next_proc=smooth, save_proc=True, work_dir=tmp_path)
    rng = np.random.default_rng(0)
    data = rng.random((8, 8, 6)).astype(np.float32) + 1
    mask = np.ones(data.shape, np.uint8)
    smooth.set_mask(mask)
    affine = np.diag([3.0, 3.0, 3.0, 1.0])
    v = RtpVolume(data, affine, filename="t_nr_0001.nii.gz")
    mod.do_proc(v, 0, None)
    assert np.array_equal(mod.saved_data[..., 0], data * 2)
    assert mod.saved_files[-1].endswith("x.t_nr_0001.nii.gz")
    assert smooth.saved_files[-1].endswith("sm.x.t_nr_0001.nii.gz")
    assert not np.array_equal(smooth.saved_data[..., 0], data * 2)
