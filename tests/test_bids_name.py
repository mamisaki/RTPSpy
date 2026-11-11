"""BIDS names of the RTPSpy output files (rtpspy.bids_name)."""
from types import SimpleNamespace

import pandas as pd

from rtpspy import bids_name as bn


def test_parse_names():
    # Name of the DICOM converter (dcm2niix: sub-%n_ses-%t_ser-%s_desc-%d)
    ent = bn.entities_from_filename(
        "sub-AB123_ses-20250101120000_ser-5_desc-epi_rest.nii.gz")
    assert ent == {"sub": "AB123", "ses": "20250101120000",
                   "task": "epirest", "run": 5}
    # Volume of the processing chain (prefixes and volume number)
    ent = bn.entities_from_filename(
        "regRes.sm.vr.sub-AB123_ses-1_task-rest_run-05_bold_0001.nii.gz")
    assert ent == {"sub": "AB123", "ses": "1", "task": "rest", "run": 5}
    # Name without entities: sub-unknown, the stem identifies the image
    assert bn.entities_from_filename("func_epi_nr_0001.nii") == \
        {"sub": "unknown", "task": "funcepi"}
    assert bn.entities_from_filename("vr_base_func_epi.nii.gz'[0]'") == \
        {"sub": "unknown", "task": "funcepi"}
    assert bn.entities_from_filename("anat_mprage.nii.gz", kind="anat") \
        == {"sub": "unknown", "acq": "anatmprage"}


def test_dicom_and_name():
    hdr = {"StudyDate": "20250101", "StudyTime": "120000.123"}
    dcm = SimpleNamespace(PatientName="AB123^x", SeriesDescription="NF run-1",
                          SeriesNumber=7, get=lambda k, d="": hdr.get(k, d),
                          **hdr)
    ent = bn.entities_from_dicom(dcm)
    assert ent == {"sub": "AB123", "ses": "20250101120000",
                   "task": "NFrun1", "run": 7}
    assert bn.bids_name(ent, "bold", ".nii.gz", desc="regress") == \
        "sub-AB123_ses-20250101120000_task-NFrun1_run-07_desc-regress" \
        "_bold.nii.gz"


def test_resolve_run_and_repeat(tmp_path):
    ent = {"sub": "x", "task": "rest"}
    assert bn.resolve_scan_entities(ent, [tmp_path])["run"] == 1
    (tmp_path / "sub-x_task-rest_run-01_desc-roi_timeseries.tsv").touch()
    assert bn.resolve_scan_entities(ent, [tmp_path])["run"] == 2
    # Same series again: rep2 in desc, not overwritten
    ent = bn.resolve_scan_entities({**ent, "run": 1}, [tmp_path])
    assert ent["rep"] == 2
    assert bn.bids_name(ent, "physio", ".tsv.gz") == \
        "sub-x_task-rest_run-01_desc-rep2_physio.tsv.gz"
    assert bn.bids_name(ent, "bold", desc="volreg") == \
        "sub-x_task-rest_run-01_desc-volregrep2_bold"


def test_write_tsv(tmp_path):
    f = bn.write_tsv(pd.DataFrame({"a": [1.0, float("nan")]}),
                     tmp_path / "x_timeseries.tsv",
                     columns_info={"a": {"Units": "s"}})
    assert f.read_text().splitlines() == ["a", "1", "n/a"]
    assert (tmp_path / "x_timeseries.json").is_file()


def test_mask_names(tmp_path):
    m = bn.MaskNames(tmp_path, "anat_mprage.nii.gz",
                     "sub-A_ses-1_ser-5_desc-rest.nii.gz",
                     template_f="MNI152_2009_template.nii.gz")
    f = "sub-A_ses-1_task-rest_run-05"
    assert m.boldref.name == f + "_boldref.nii.gz"
    assert m.func_mask("RTP").name == f + "_desc-RTP_mask.nii.gz"
    assert m.func_mask("WM", space="sdc").name == \
        f + "_space-sdc_label-WM_mask.nii.gz"
    assert m.roi("MNI152_2009_template_LAmy.nii.gz").name == \
        f + "_label-LAmy_mask.nii.gz"
    a = "sub-unknown_acq-anatmprage"
    assert m.brain.name == a + "_desc-brain_T1w.nii.gz"
    assert m.alanat.name == a + "_space-func_desc-brain_T1w.nii.gz"
    assert m.alanat_xfm.name == \
        a + "_from-T1w_to-func_mode-image_xfm.aff12.1D"
    assert m.template_xfm("affine").name == \
        a + "_from-MNI1522009template_to-func_mode-image_desc-affine_xfm.mat"
