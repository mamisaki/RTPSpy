#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BIDS-style names and formats of the RTPSpy output files.

The output files follow the BIDS derivatives conventions as far as the
real-time setting allows:

* Names are '<entities>_<suffix><extension>'. The entities are in the BIDS
  order (sub, ses, task, acq, ..., run, ..., space, label, desc), and their
  values have only letters and digits.
* Tables are TSV files with a header row, 'n/a' for missing values, and a
  JSON sidecar describing the columns.
* Images have a JSON sidecar when they need metadata (e.g., RepetitionTime).

The entities of a scan are taken from the DICOM header (entities_from_dicom)
or from the image file name (entities_from_filename). The subject is the
patient name (the ID of two letters and three digits if it has one; the
same rule as the DICOM converter), the session is the study date and time
(YYYYMMDDhhmmss, as dcm2niix's %t), the task is the series description,
and the run is the series number.

Output entities of a scan (sub, ses, task, run) are fixed once per scan by
resolve_scan_entities: a missing run gets the next free run number, and if
files of the same entities already exist (e.g., the same DICOM series
replayed), 'rep<k>' is added to the desc entity of the new files instead of
overwriting the old ones.
"""

import json
import re
from pathlib import Path

import numpy as np

# BIDS entity order (raw and derivatives)
ENTITY_ORDER = (
    "sub", "ses", "sample", "task", "tracksys", "acq", "nuc", "voi", "ce",
    "trc", "stain", "rec", "dir", "run", "mod", "echo", "flip", "inv", "mt",
    "part", "proc", "hemi", "space", "split", "recording", "chunk", "seg",
    "res", "den", "label", "from", "to", "mode", "desc",
)

# Keys read from a file name. 'ser' (series number) and a 'desc' of the
# DICOM-converted names (series description) are mapped to run and task.
_NAME_KEYS = set(ENTITY_ORDER) | {"ser"}

# Image file extensions removed to get the name stem
_IMG_EXT = re.compile(
    r"(\.nii(\.gz)?|\+\w+\.(HEAD|BRIK)(\.gz)?|\.(HEAD|BRIK)(\.gz)?|"
    r"\.json|\.tsv(\.gz)?|\.csv|\.txt|\.1D)$")


# %% Labels and names =========================================================
def bids_label(value):
    """Return a BIDS label of value: letters and digits only (other
    characters are removed). Returns None for an empty label."""
    if value is None:
        return None
    label = re.sub(r"[^a-zA-Z0-9]", "", str(value))
    return label if label else None


def bids_index(value):
    """Return a BIDS index (non-negative integer) of value, or None."""
    try:
        idx = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return idx if idx >= 0 else None


def format_entity(key, value):
    """'key-value' with the value made BIDS-safe. The run index is
    zero-padded to two digits."""
    if key in ("run", "echo", "chunk", "split", "inv", "flip", "mt"):
        idx = bids_index(value)
        if idx is None:
            return None
        return f"{key}-{idx:02d}" if key == "run" else f"{key}-{idx}"
    label = bids_label(value)
    return None if label is None else f"{key}-{label}"


def bids_name(entities, suffix, ext="", **extra):
    """
    Make a BIDS file name.

    Parameters
    ----------
    entities : dict
        Entities, e.g., {'sub': 'AB123', 'ses': '20250101120000',
        'task': 'rest', 'run': 5}. Keys not in ENTITY_ORDER are ignored,
        except 'rep' (see below).
    suffix : str
        BIDS suffix (e.g., 'bold', 'mask', 'timeseries', 'physio').
    ext : str, optional
        Extension with the leading dot (e.g., '.nii.gz', '.tsv').
    extra : dict
        Additional entities (e.g., desc='regress', label='WM', space='func').
        An entity of value None is not added.

    entities['rep'] (an int > 1) marks a repeat of a scan whose files exist:
    'rep<k>' is added to the desc entity (desc-<desc>rep<k>, or desc-rep<k>
    without desc).
    """
    ent = {k: v for k, v in dict(entities or {}).items()
           if v is not None}
    ent.update({k: v for k, v in extra.items() if v is not None})

    rep = ent.pop("rep", None)
    if rep is not None and int(rep) > 1:
        ent["desc"] = f"{ent.get('desc', '')}rep{int(rep)}"

    parts = []
    for key in ENTITY_ORDER:
        if key in ent:
            part = format_entity(key, ent[key])
            if part is not None:
                parts.append(part)
    parts.append(suffix)
    return "_".join(parts) + ext


def name_stem(fname):
    """File name without the directory and the image/table extension."""
    return _IMG_EXT.sub("", Path(str(fname)).name)


# %% Entities from a file name, a JSON sidecar, or a DICOM header =============
def parse_entities(fname):
    """
    Read the BIDS entities in a file name.

    Processing prefixes (e.g., 'regRes.sm.ts.vr.'), volume numbers
    ('_nr_0001', '_0001') and the BIDS suffix ('_bold') are ignored.
    The legacy names of RTPSpy and the DICOM converter
    ('sub-<name>_ses-<date>_ser-<N>_desc-<series description>') are read
    as sub, ses, run (= ser), and task (= desc, when there is no task).

    Returns
    -------
    dict
        Found entities (values as in the name, not sanitized).
    """
    stem = name_stem(fname)
    stem = re.sub(r"(_nr)?_\d{4,}$", "", stem)  # volume number
    # Processing prefixes before the first entity (e.g., 'vr.')
    ma = re.search(r"(^|[._])(sub|ses|task|run|ser)-", stem)
    if ma is None:
        return {}
    stem = stem[ma.start(2):]

    # 'key-value' tokens. A value runs to the next '_key-' token; a legacy
    # 'desc' (the series description) can include '_'.
    tokens = re.split(r"_(?=[a-zA-Z]+-)", stem)
    ent = {}
    for tok in tokens:
        key, _, val = tok.partition("-")
        if key not in _NAME_KEYS or not val:
            continue
        if key != "desc":
            val = val.split("_")[0]  # drop a trailing suffix ('_bold')
        else:
            # Drop a trailing BIDS suffix of an RTPSpy name (lowercase
            # word without '-', e.g., '_bold'); a series description is
            # kept whole.
            val = re.sub(r"_(bold|boldref|T1w|mask|dseg|physio|"
                         r"timeseries|xfm)$", "", val)
        ent[key] = val

    if "ser" in ent:
        ser = ent.pop("ser")
        if "run" not in ent and bids_index(ser) is not None:
            ent["run"] = bids_index(ser)
    if "desc" in ent and "task" not in ent and "sub" in ent:
        # Series description of a DICOM-converted name
        ent["task"] = ent.pop("desc")
    return ent


def entities_from_dicom(dcm):
    """
    Entities of a scan from its DICOM header (pydicom Dataset).

    sub: patient name (the ID of two letters and three digits if the first
    name component has that form), ses: study date and time
    (YYYYMMDDhhmmss), task: series description, run: series number.
    """
    ent = {}
    try:
        patient = str(dcm.PatientName).split("^")
        if re.match(r"\w\w\d\d\d", patient[0]):
            # Subject ID format of two letters and three digits
            ent["sub"] = patient[0]
        else:
            ent["sub"] = "_".join(patient)
    except Exception:
        pass
    try:
        date = str(dcm.get("StudyDate", ""))
        tm = str(dcm.get("StudyTime", "")).split(".")[0]
        if date:
            ent["ses"] = date + tm
    except Exception:
        pass
    try:
        ent["task"] = str(dcm.SeriesDescription)
    except Exception:
        pass
    try:
        ent["run"] = int(dcm.SeriesNumber)
    except Exception:
        pass
    return clean_entities(ent)


def entities_from_json(json_f):
    """Entities from a dcm2niix JSON sidecar: task (SeriesDescription) and
    run (SeriesNumber). Empty if the file cannot be read."""
    try:
        with open(json_f) as fd:
            info = json.load(fd)
    except Exception:
        return {}
    ent = {}
    if info.get("SeriesDescription"):
        ent["task"] = info["SeriesDescription"]
    if bids_index(info.get("SeriesNumber")) is not None:
        ent["run"] = bids_index(info["SeriesNumber"])
    return ent


def clean_entities(ent):
    """Entities with BIDS-safe values (entities that become empty are
    dropped)."""
    out = {}
    for key, val in ent.items():
        if key in ("run", "rep", "echo"):
            idx = bids_index(val)
            if idx is not None:
                out[key] = idx
        else:
            label = bids_label(val)
            if label is not None:
                out[key] = label
    return out


def entities_from_filename(fname, kind="func"):
    """
    Entities of an image from its file name, completed from the JSON
    sidecar next to it (dcm2niix: SeriesDescription, SeriesNumber).

    kind : 'func' or 'anat'
        For 'anat', only sub, ses, and acq are kept. If the name has no
        subject entity, sub is 'unknown' and the name stem identifies the
        image (task for 'func', acq for 'anat').
    """
    fname = Path(re.sub(r"[\'\"]*\[\d+\][\'\"]*$", "", str(fname)))
    ent = parse_entities(fname)
    if "task" not in ent or "run" not in ent:
        json_f = fname.parent / (name_stem(fname) + ".json")
        for key, val in entities_from_json(json_f).items():
            ent.setdefault(key, val)

    # Label of the name stem, without the volume number and the prefix of
    # the base function image of the former RTPSpy ('vr_base_')
    stem_label = bids_label(re.sub(
        r"(_nr)?_\d{4,}$", "", re.sub(r"^vr_base_", "", name_stem(fname))))
    if "sub" not in ent:
        ent["sub"] = "unknown"
        if kind == "anat":
            ent.setdefault("acq", stem_label)
        else:
            ent.setdefault("task", stem_label)

    if kind == "anat":
        ent = {k: v for k, v in ent.items() if k in ("sub", "ses", "acq")}
    elif "task" not in ent:
        ent["task"] = stem_label or "rtp"
    return clean_entities(ent)


# %% Scan entities and repeats ================================================
def _existing_names(dirs):
    names = []
    for dd in dirs:
        dd = Path(dd)
        if dd.is_dir():
            names.extend(f.name for f in dd.iterdir())
    return names


def resolve_scan_entities(ent, dirs):
    """
    Fix the output entities of a scan.

    If ent has no run, the next free run number of the same sub/ses/task in
    dirs is set. If files of these entities exist in dirs, 'rep' is set to
    the next repeat number (see bids_name).

    Parameters
    ----------
    ent : dict
        Entities (sub, ses, task, run).
    dirs : list of Path
        Output directories to check (e.g., work_dir/RTP and work_dir/log).

    Returns
    -------
    dict
        Entities with run (and rep if needed).
    """
    ent = {k: v for k, v in clean_entities(ent).items() if k != "rep"}
    names = _existing_names(dirs)
    if "run" not in ent:
        base = bids_name({k: v for k, v in ent.items()
                          if k in ("sub", "ses", "task", "acq")}, "")
        base = base.rstrip("_")
        runs = [int(m.group(1)) for n in names
                for m in [re.match(re.escape(base) + r"_run-(\d+)_", n)]
                if m]
        ent["run"] = max(runs, default=0) + 1
        return ent

    prefix = bids_name(ent, "").rstrip("_") + "_"
    same = [n for n in names if n.startswith(prefix)]
    if same:
        reps = [1]
        for n in same:
            ma = re.search(r"_desc-[a-zA-Z]*?rep(\d+)_", n)
            if ma:
                reps.append(int(ma.group(1)))
        ent["rep"] = max(reps) + 1
    return ent


# %% Writers ==================================================================
def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def write_json(path, data):
    """Write a JSON file (numpy values and Paths are converted)."""
    with open(path, "w") as fd:
        json.dump(data, fd, indent=4, default=_json_default)
    return Path(path)


def write_tsv(df, path, columns_info=None, float_format="%.6g",
              sidecar=None):
    """
    Write a pandas DataFrame as a BIDS TSV file (header row, tab
    separated, 'n/a' for missing values, no index) with a JSON sidecar.

    Parameters
    ----------
    df : pandas.DataFrame
    path : Path
        Output .tsv file.
    columns_info : dict, optional
        {column: {'Description': ..., 'Units': ...}} for the sidecar.
    sidecar : dict, optional
        Other sidecar fields.
    """
    path = Path(path)
    df.to_csv(path, sep="\t", index=False, na_rep="n/a",
              float_format=float_format)
    meta = dict(sidecar or {})
    for col, info in (columns_info or {}).items():
        if col in df.columns:
            meta[col] = info
    if meta:
        write_json(path.with_suffix(".json"), meta)
    return path


def tsv_json_path(tsv_path):
    """JSON sidecar of a .tsv or .tsv.gz file."""
    name = re.sub(r"\.tsv(\.gz)?$", ".json", Path(tsv_path).name)
    return Path(tsv_path).parent / name


# %% Names of the mask files made by RtpApp.make_masks =======================
class MaskNames:
    """
    BIDS names of the files made by RtpApp.make_masks in out_dir
    (work_dir/RTP).

    Anatomy images are named with the anatomy entities (sub, ses[, acq]);
    images on the grid of the base function image are named with its
    entities (sub, ses, task, run), and the anatomy aligned to the function
    image is in 'space-func' (a custom space whose reference is the base
    function image; see the SpatialReference of its sidecar). Images made
    in the distortion-corrected space are in 'space-sdc'.
    """

    def __init__(self, out_dir, anat_f, func_f, template_f=None):
        self.out_dir = Path(out_dir)
        self.anat_ent = entities_from_filename(anat_f, kind="anat")
        self.func_ent = entities_from_filename(func_f, kind="func")
        # The base function image copied by make_masks is already in
        # out_dir with a BIDS name; read its entities as they are.
        self.template_label = None
        if template_f:
            self.template_label = bids_label(name_stem(template_f))

    def _a(self, suffix, ext=".nii.gz", **extra):
        return self.out_dir / bids_name(self.anat_ent, suffix, ext, **extra)

    def _f(self, suffix, ext=".nii.gz", **extra):
        return self.out_dir / bids_name(self.func_ent, suffix, ext, **extra)

    # --- Anatomy space ---
    @property
    def anat(self):
        return self._a("T1w")

    @property
    def anat_biascorr(self):
        return self._a("T1w", desc="biascorr")

    @property
    def brain(self):
        return self._a("T1w", desc="brain")

    def anat_mask(self, seg):
        """WM/Vent mask (label-<seg>_mask) or aseg (desc-aseg_dseg) in the
        anatomy space."""
        if seg == "aseg":
            return self._a("dseg", desc="aseg")
        return self._a("mask", label=seg)

    # --- Function space ---
    @property
    def boldref(self):
        return self._f("boldref")

    @property
    def boldref_json(self):
        return self._f("boldref", ".json")

    @property
    def boldref_sdc(self):
        return self._f("boldref", space="sdc")

    @property
    def sdc_invwarp(self):
        """Composite transform from the corrected to the original space."""
        return self._f("xfm", **{"from": "sdc", "to": "orig",
                                 "mode": "image"})

    @property
    def alanat(self):
        """Brain image aligned to the base function image."""
        return self._a("T1w", space="func", desc="brain")

    @property
    def alanat_xfm(self):
        """Affine (AFNI aff12.1D) from the anatomy to the function."""
        return self._a("xfm", ".aff12.1D", **{"from": "T1w", "to": "func",
                                              "mode": "image"})

    def template_xfm(self, kind):
        """ANTs transform from the template to alAnat: kind 'affine'
        (.mat), 'warp', or 'invwarp' (.nii.gz)."""
        ext = ".mat" if kind == "affine" else ".nii.gz"
        return self._a("xfm", ext, **{"from": self.template_label or
                                      "template", "to": "func",
                                      "mode": "image", "desc": kind})

    def template_on_anat(self, seg):
        """Template mask (WM/Vent) warped to the alAnat grid."""
        return self._a("mask", space="func", label=seg, desc="template")

    def func_mask(self, seg, space=None):
        """Mask on the base function grid: 'RTP', 'GSR', 'automask'
        (desc-<name>_mask), 'WM'/'Vent' (label-<name>_mask), 'aseg'
        (desc-aseg_dseg)."""
        if seg == "aseg":
            return self._f("dseg", space=space, desc="aseg")
        if seg in ("RTP", "GSR", "automask", "anatmask"):
            return self._f("mask", space=space, desc=seg)
        return self._f("mask", space=space, label=seg)

    def roi(self, roi_template_f, space=None):
        """Template ROI warped to the base function grid."""
        label = bids_label(name_stem(roi_template_f)) or "ROI"
        if self.template_label and label.startswith(self.template_label) \
                and len(label) > len(self.template_label):
            # e.g., MNI152_2009_template_LAmy -> LAmy
            label = label[len(self.template_label):]
        return self._f("mask", space=space, label=label)

    def spatial_reference(self, img_f):
        """Write the sidecar of a space-func image with its
        SpatialReference (the base function image)."""
        img_f = Path(img_f)
        json_f = img_f.parent / (name_stem(img_f) + ".json")
        write_json(json_f, {"SpatialReference": self.boldref.name})
        return json_f
