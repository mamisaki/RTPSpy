#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alarm sound of the RT-MRI GUI (motion alarm, missing TTL alarm).

print("\\a") makes no sound when stdout is not a terminal (e.g., when run by
the launcher or the systemd service). Instead, a WAV file made here is played
with a sound player found on the system, so no system sound file is needed.

mmisaki@laureateinstitute.org
"""

# %% import ===================================================================
import os
import atexit
import shutil
import subprocess
import tempfile
import wave
import logging
from pathlib import Path

import numpy as np

# Players tried in this order: PulseAudio/PipeWire, ALSA (Linux), and macOS
ALARM_PLAYERS = (["paplay"], ["pw-play"], ["aplay", "-q"], ["afplay"])


# %% make_alarm_wav ===========================================================
def make_alarm_wav(path, freqs=(880, 1320), tone_sec=0.15, n_rep=2,
                   rate=44100, volume=0.5):
    """Write the alarm sound (alternating tones) to a WAV file."""
    t = np.arange(int(tone_sec * rate)) / rate
    fade = np.minimum(1.0, np.minimum(t, tone_sec - t) / 0.01)  # no clicks
    tones = [np.sin(2 * np.pi * f * t) * fade for f in freqs]
    sig = np.concatenate(tones * n_rep) * volume
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes((sig * 32767).astype("<i2").tobytes())


# %% find_alarm_player ========================================================
def find_alarm_player():
    """Return the command to play the alarm sound, or None."""
    player = next(
        (cmd for cmd in ALARM_PLAYERS if shutil.which(cmd[0])), None
    )
    if player is None:
        return None
    try:
        fd, wav_f = tempfile.mkstemp(prefix="rtmri_alarm_", suffix=".wav")
        os.close(fd)
        make_alarm_wav(wav_f)
        atexit.register(lambda: Path(wav_f).unlink(missing_ok=True))
    except Exception:
        return None
    return [shutil.which(player[0])] + player[1:] + [wav_f]


# %% AlarmSound ===============================================================
class AlarmSound:
    """Play the alarm sound without blocking. A new sound is not started
    while the previous one is playing. Without a sound player, the bell
    (Qt or Tk) is used."""

    def __init__(self):
        self._logger = logging.getLogger("AlarmSound")
        self._cmd = find_alarm_player()
        self._proc = None
        if self._cmd is None:
            self._logger.warning(
                "No sound player for the alarm "
                f"({', '.join(c[0] for c in ALARM_PLAYERS)}). "
                "The X bell is used."
            )

    @property
    def cmd(self):
        """Command to play the alarm sound, or None."""
        return self._cmd

    def play(self, tk_widget=None, bell=None):
        """Play the alarm sound, or ring the bell (the bell function, e.g.,
        QApplication.beep, or the bell of tk_widget) if there is no sound
        player or the player fails."""
        if self._cmd is not None:
            if self._proc is not None and self._proc.poll() is None:
                return
            try:
                self._proc = subprocess.Popen(
                    self._cmd, stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                return
            except Exception as e:
                self._logger.error(f"Failed to play the alarm sound: {e}")
        if bell is not None:
            try:
                bell()
            except Exception:
                pass
        elif tk_widget is not None:
            try:
                tk_widget.bell()
            except Exception:
                pass
