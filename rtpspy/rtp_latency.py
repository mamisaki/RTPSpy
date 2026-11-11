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
import numpy as np

COLUMNS = ("vol", "onset", "onset_estimated", "read", "sent",
           "acq_to_read", "read_to_fb", "acq_to_fb", "retrospective")


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
    def save_csv(self, fname):
        """Save the records as a CSV file. Returns fname, or None if there
        is no record."""
        if len(self) == 0:
            return None
        with open(fname, "w") as fd:
            fd.write(",".join(COLUMNS) + "\n")
            for row in zip(*(self.data[c] for c in COLUMNS)):
                vals = []
                for c, v in zip(COLUMNS, row):
                    if isinstance(v, bool):
                        vals.append(str(int(v)))
                    elif v is None:
                        vals.append("")
                    elif c in ("onset", "read", "sent"):
                        vals.append(f"{v:.6f}")
                    elif isinstance(v, float):
                        vals.append(f"{v:.4f}")
                    else:
                        vals.append(str(v))
                fd.write(",".join(vals) + "\n")
        return fname
