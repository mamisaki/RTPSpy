"""Framewise displacement (rtp_volreg.framewise_displacement), shared by the
motion plot, REGRESS (QC and censoring), and the offline comparison."""
import numpy as np

from rtpspy.rtp_volreg import framewise_displacement, FD_HEAD_RADIUS


def test_framewise_displacement():
    # roll, pitch, yaw (deg), dS, dL, dP (mm)
    mot = np.array([[0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],      # 1 deg roll
                    [1, 0, 0, 0.5, -0.2, 0],  # translations
                    [0, -1, 0, 0.5, -0.2, 0]])
    fd = framewise_displacement(mot)
    deg = FD_HEAD_RADIUS * np.pi / 180  # 0.873 mm per degree
    assert np.allclose(fd, [0, deg, 0.7, 2 * deg])


def test_single_volume():
    assert np.array_equal(framewise_displacement(np.zeros(6)), [0.0])
