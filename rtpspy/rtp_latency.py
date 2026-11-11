#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latency of the real-time processing for each fMRI volume.

For each volume, three times are recorded (time.time() of this PC; the TTL
times from rt_physio use the same clock):
    onset : acquisition onset of the volume, from the scanner TTL pulse
    read  : the image file was read by RtpWatch
    sent  : the feedback signal was sent to the external application (or
            saved to the signal file)
and two latencies are derived:
    acq_to_read : onset -> read (image reconstruction and transfer)
    read_to_fb  : read -> sent (RTPSpy processing)

The TTL of a volume is the TTL pulse nearest to its expected onset: the
onset of the previous matched volume plus TR (or scan_onset + vol_idx * TR
for the first one), within ONSET_TOL * TR. This works with one pulse per
volume and with one pulse per slice (the first slice pulse of the volume is
the nearest). If no pulse is found (e.g., a manual start without the TTL),
the expected onset is used and the volume is marked onset_estimated.

Volumes processed after later volumes had been read (retrospectively, e.g.,
those in the wait period of RtpRegress) are marked retrospective; their
read_to_fb is not a real-time latency.

@author: mmisaki@laureateinstitute.org
"""

# %% import ===================================================================
from pathlib import Path

import numpy as np

try:
    from .bids_name import write_json
except ImportError:
    from rtpspy.bids_name import write_json

COLUMNS = ("vol", "onset", "onset_estimated", "read", "sent",
           "acq_to_read", "read_to_fb", "acq_to_fb", "retrospective")

# Column descriptions (JSON sidecar of the saved TSV)
_CLOCK = "time.time() of the RTPSpy PC (Unix time)"
COLUMNS_INFO = {
    "vol": {"Description": "0-based index of the volume from the scan "
            "start"},
    "onset": {"Description": "Acquisition onset of the volume (the "
              "matched scanner TTL pulse, or the expected onset if "
              f"onset_estimated is 1); {_CLOCK}", "Units": "s"},
    "onset_estimated": {"Description": "The onset is estimated (1) "
                        "because no TTL pulse was matched, or measured (0)"},
    "read": {"Description": f"The image file was read; {_CLOCK}",
             "Units": "s"},
    "sent": {"Description": "The feedback signal was sent (or saved); "
             f"{_CLOCK}", "Units": "s"},
    "acq_to_read": {"Description": "read - onset", "Units": "s"},
    "read_to_fb": {"Description": "sent - read (RTPSpy processing)",
                   "Units": "s"},
    "acq_to_fb": {"Description": "sent - onset", "Units": "s"},
    "retrospective": {"Description": "Processed after later volumes had "
                      "been read (1), e.g., in the wait period of REGRESS; "
                      "read_to_fb is then not a real-time latency"},
}


# %% VolumeLatency ============================================================
class VolumeLatency:
    """Per-volume latency record (see the module docstring)."""

    ONSET_TOL = 0.3  # Tolerance of the TTL match (fraction of TR)

    def __init__(self):
        self.reset()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def reset(self):
        """Clear the records (at the start of a run)."""
        self.data = {c: [] for c in COLUMNS}
        self._last_match = None  # (vol_idx, onset) of the last TTL match

    def __len__(self):
        return len(self.data["vol"])

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def match_onset(self, vol_idx, ttl_onsets, scan_onset, TR):
        """
        Acquisition onset of volume vol_idx (0-based).

        Parameters
        ----------
        vol_idx : int
            Volume index from the scan onset.
        ttl_onsets : array
            TTL onset times (any order; NaN is ignored).
        scan_onset : float
            Scan onset time (the first TTL, or the manual start time).
        TR : float
            Repetition time (s).

        Returns
        -------
        onset : float
        estimated : bool
            True if no TTL pulse matched and the expected onset is returned.
        """
        if self._last_match is not None and self._last_match[0] <= vol_idx:
            last_vi, last_on = self._last_match
            expected = last_on + (vol_idx - last_vi) * TR
        else:
            expected = scan_onset + vol_idx * TR

        ttl = np.asarray(ttl_onsets, dtype=float)
        ttl = ttl[~np.isnan(ttl)]
        if TR and TR > 0 and len(ttl):
            ttl = ttl[ttl >= scan_onset - self.ONSET_TOL * TR]
            if len(ttl):
                near = ttl[np.argmin(np.abs(ttl - expected))]
                if abs(near - expected) <= self.ONSET_TOL * TR:
                    self._last_match = (vol_idx, float(near))
                    return float(near), False

        return float(expected), True

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def record(self, vol_idx, onset, onset_estimated, t_read, t_sent,
               retrospective):
        """Add the times of a volume. A volume recorded twice (e.g., sent
        twice) keeps the first record."""
        if vol_idx in self.data["vol"]:
            return
        d = self.data
        d["vol"].append(int(vol_idx))
        d["onset"].append(onset)
        d["onset_estimated"].append(bool(onset_estimated))
        d["read"].append(t_read)
        d["sent"].append(t_sent)
        nan = float("nan")
        d["acq_to_read"].append(
            t_read - onset if None not in (t_read, onset) else nan)
        d["read_to_fb"].append(
            t_sent - t_read if None not in (t_sent, t_read) else nan)
        d["acq_to_fb"].append(
            t_sent - onset if None not in (t_sent, onset) else nan)
        d["retrospective"].append(bool(retrospective))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save_tsv(self, fname):
        """Save the records as a BIDS TSV file ('n/a' for missing values)
        with a JSON sidecar describing the columns. Returns fname, or None
        if there is no record."""
        if len(self) == 0:
            return None
        fname = Path(fname)
        with open(fname, "w") as fd:
            fd.write("\t".join(COLUMNS) + "\n")
            for row in zip(*(self.data[c] for c in COLUMNS)):
                vals = []
                for c, v in zip(COLUMNS, row):
                    if isinstance(v, bool):
                        vals.append(str(int(v)))
                    elif v is None or (isinstance(v, float) and
                                       np.isnan(v)):
                        vals.append("n/a")
                    elif c in ("onset", "read", "sent"):
                        vals.append(f"{v:.6f}")
                    elif isinstance(v, float):
                        vals.append(f"{v:.4f}")
                    else:
                        vals.append(str(v))
                fd.write("\t".join(vals) + "\n")
        write_json(fname.parent / (fname.name.replace(".tsv", ".json")),
                   COLUMNS_INFO)
        return fname

    # Former name
    save_csv = save_tsv
