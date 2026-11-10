# -*- coding: utf-8 -*-
"""
Volume object passed along the RTP processing chain

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
import nibabel as nib


# %% RtpVolume ================================================================
class RtpVolume(nib.Nifti1Image):
    """
    A volume of the real-time processing chain.

    RtpVolume is a nibabel Nifti1Image (get_fdata, dataobj, affine, header,
    get_filename, nib.save, ... work as usual) with the volume metadata as
    attributes:

    vol_idx : int or None
        0-based index of the volume from the scan start.
    acq_time : float or None
        Time stamp when the volume was received (e.g., the creation time of
        the image file read by RtpWatch).
    proc_time : float or None
        Time stamp when the last module finished processing the volume.

    A module does not change the volume it receives. It outputs a new volume
    made by with_data(), so the volume kept by a module (e.g., for saving)
    is not changed by the following modules. A volume received from the
    chain (e.g., in RtpApp) should also be treated as read-only.
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, dataobj, affine, header=None, extra=None,
                 file_map=None, dtype=None, *, filename=None, vol_idx=None,
                 acq_time=None, proc_time=None):
        """
        Parameters
        ----------
        dataobj, affine, header, extra, file_map, dtype :
            As nibabel.Nifti1Image. The header is copied.
        filename : str or Path, optional
            Filename of the volume (see set_filename). It is used as the
            name of the volume (e.g., in the log and the saved filename).
        vol_idx : int, optional
            0-based index of the volume.
        acq_time : float, optional
            Time stamp when the volume was received.
        proc_time : float, optional
            Time stamp when the volume was processed.
        """
        super().__init__(dataobj, affine, header=header, extra=extra,
                         file_map=file_map, dtype=dtype)
        self.vol_idx = vol_idx
        self.acq_time = acq_time
        self.proc_time = proc_time
        if filename is not None:
            self.set_filename(filename)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @classmethod
    def as_volume(cls, img, vol_idx=None):
        """
        Return img as an RtpVolume.

        An RtpVolume is returned as it is. Another nibabel image is converted
        to a new RtpVolume sharing its data object, with a copy of its
        header and its filename; img is not changed.

        Parameters
        ----------
        img : nibabel image object
            Volume image.
        vol_idx : int, optional
            0-based index of the volume, set to a converted image.
        """
        if isinstance(img, cls):
            return img

        return cls(img.dataobj, img.affine, header=img.header,
                   extra=img.extra.copy(), filename=img.get_filename(),
                   vol_idx=vol_idx)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def with_data(self, data, filename=None, vol_idx=None):
        """
        Return a new RtpVolume with the data array replaced. This volume is
        not changed.

        The header is copied, with the data type set to that of data. The
        filename is that of this volume unless filename is given. The
        metadata (vol_idx, acq_time, proc_time) are carried over; vol_idx
        is replaced if it is given.

        Parameters
        ----------
        data : array
            Data of the new volume.
        filename : str or Path, optional
            Filename of the new volume.
        vol_idx : int, optional
            0-based index of the new volume.
        """
        if filename is None:
            filename = self.get_filename()
        if vol_idx is None:
            vol_idx = self.vol_idx

        return self.__class__(
            data, self.affine, header=self.header, extra=self.extra.copy(),
            dtype=getattr(data, 'dtype', None), filename=filename,
            vol_idx=vol_idx,
            acq_time=self.acq_time, proc_time=self.proc_time)
