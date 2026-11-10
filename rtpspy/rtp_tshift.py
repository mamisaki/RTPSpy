#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTP online slice-timing correction

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import time
import json

import numpy as np
import nibabel as nib

try:
    from .rtp_common import RTP
except Exception:
    from rtpspy.rtp_common import RTP


# %% class RtpTshift =========================================================
class RtpTshift(RTP):
    """ Online slice-timing correction """

    # Log label and output filename prefix (see RTP.do_proc)
    _proc_label = 'Slice-timing correction'
    _out_prefix = 'ts.'

    # GUI panel (rtpspy.ui.tshift_ui)
    _ui_class = "tshift_ui:RtpTshiftUI"

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, method='cubic', ref_time=0, TR=2.0,
                 slice_timing=[], slice_dim=2, **kwargs):
        """
        TR and slice_timing can be set from a sample fMRI data file with the
        'set_from_sample' method.

        Parameters
        ----------
        method : str ['linear'|'cubic'], optional
            Temporal interpolation method for the correction.
            The default is 'cubic'.
        ref_time : float, optional
            Reference time (seconds) to which all slices are aligned.
            The default is 0.
        TR : float, optional
            Scan interval (seconds). The default is 2.0.
        slice_timing : array like, optional
            Acquisition time (seconds) of each slice. The default is [].
        slice_dim : int, optional
            Slice dimension
                0: x axis (sagittal slice)
                1: y axis (coronal slice)
                2: z axis (axial slice)
            The default is 2.

        """
        super(RtpTshift, self).__init__(**kwargs)

        # Set instance parameters
        self.method = method
        self.ref_time = ref_time

        self.TR = TR
        self.slice_timing = slice_timing
        self.slice_dim = slice_dim

        # Initialize the process state
        self.prep_weight = False  # True when the weights are calculated
        self._weight_shape = None  # Volume shape of the weights
        self.pre_data = []  # Previous data volumes

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ready_proc(self):
        if self.slice_timing is None or len(self.slice_timing) == 0 or \
                self.TR is None:
            errmsg = "Slice timing is not set. "
            self._logger.error(errmsg)
            self.err_popup(errmsg)
            self._proc_ready = False
            return False

        self._proc_ready = True  # ready in any case
        if self.next_proc:
            self._proc_ready &= self.next_proc.ready_proc()

        return self._proc_ready

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _process(self, fmri_img, vol_idx, pre_proc_time):
        """
        Slice-timing correction of the volume (see RTP.do_proc).

        The correction of a volume uses the previous volumes. The first
        volume (and, with the cubic method, the second one) is held and
        corrected retrospectively when the next volume arrives; then the
        held volume is passed to the next process before the current one
        (self._correct_previous_volume), and None is returned for a volume
        that is held.
        """
        # --- Initialize ------------------------------------------------------
        if fmri_img.shape[self.slice_dim] != len(self.slice_timing):
            errmsg = 'Slice timing array does not match the data.'
            self._logger.error(errmsg)
            return None

        # Set the interpolation weights (again if the volume shape changed).
        if not self.prep_weight or \
                self._weight_shape != tuple(fmri_img.shape[:3]):
            self._calc_weights(fmri_img.shape[:3])
            if not self.prep_weight:
                return None

        dataV = np.asarray(fmri_img.dataobj, dtype=np.float32)
        if dataV.ndim > 3:
            dataV = np.squeeze(dataV)

        # The first volume cannot be corrected alone. Keep it and correct it
        # retrospectively when the next volume arrives.
        if len(self.pre_data) == 0:
            self.pre_fmri_img = fmri_img  # not changed by other modules
            self.pre_data.append(dataV)
            return None

        # --- Run the process -------------------------------------------------
        if hasattr(self, 'pre_fmri_img'):
            # Correct the kept previous volume and pass it on.
            self._correct_previous_volume(dataV, vol_idx)

            if self.method == 'cubic' and len(self.pre_data) < 2:
                # Cubic interpolation needs two previous volumes. Keep the
                # current volume too and correct it retrospectively when the
                # next volume arrives.
                self.pre_data.append(dataV)
                self.pre_fmri_img = fmri_img
                return None

            del self.pre_fmri_img

        # Slice-timing correction of the current volume
        if self.method == 'linear':
            shift_dataV = self.wm1 * self.pre_data[-1] + self.w0 * dataV
        elif self.method == 'cubic':
            # The next volume is not available; the current volume is used
            # in its place (w0p1 = w0 + wp1).
            shift_dataV = \
                self.wm2 * self.pre_data[-2] + \
                self.wm1 * self.pre_data[-1] + \
                self.w0p1 * dataV

        # Update the previous volume list
        self.pre_data.append(dataV)
        self.pre_data.pop(0)

        # Output the corrected data in a new volume
        return fmri_img.with_data(shift_dataV)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _correct_previous_volume(self, dataV, vol_idx):
        """
        Correct the kept previous volume (self.pre_fmri_img) retrospectively,
        using the current volume (dataV), and pass it to the next process.

        The previous volume is the one before vol_idx (index vol_idx - 1).
        """
        if self.method == 'linear':
            retro_shft_dataV = \
                self.r1wm1 * self.pre_data[-1] + self.r1w0 * dataV

        elif self.method == 'cubic':
            # The volume after the current one is not available yet; the
            # current volume is used in its place (r1w0p1). With only one
            # earlier volume, it is used for both earlier time points
            # (r1wm2m1).
            if len(self.pre_data) == 1:
                retro_shft_dataV = \
                    self.r1wm2m1 * self.pre_data[-1] + \
                    self.r1w0p1 * dataV
            elif len(self.pre_data) == 2:
                retro_shft_dataV = \
                    self.r1wm2 * self.pre_data[-2] + \
                    self.r1wm1 * self.pre_data[-1] + \
                    self.r1w0p1 * dataV

        pre_img = self.pre_fmri_img

        # Log message
        fname = pre_img.get_filename() or "unknown.nii.gz"
        fname = Path(fname).name.replace('.nii.gz', '')
        msg = f"#{vol_idx};"
        msg += f"Retrospective slice-timing correction;{fname}"
        self._logger.info(msg)

        # The corrected volume with the save filename
        out_img = pre_img.with_data(
            retro_shft_dataV, filename=self._out_prefix + fname + '.nii.gz',
            vol_idx=vol_idx-1)

        # Keep this module's output (see RTP._pass_to_next), and output the
        # corrected volume.
        if self.next_proc:
            self.proc_data = np.asanyarray(out_img.dataobj)
        self._output_earlier_volume(out_img, vol_idx-1,
                                    save_temp=self.online_saving)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def end_reset(self):
        """ End process and reset process parameters. """

        self._logger.info(f"Reset {self.__class__.__name__} module.")

        self.pre_data = []
        if hasattr(self, 'pre_fmri_img'):
            del self.pre_fmri_img
        self.prep_weight = False

        return super(RtpTshift, self).end_reset()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_from_sample(self, fmri_img):
        """
        Set TR, slice_timing, and slice_dim from a sample fMRI data.

        The values are read from a JSON sidecar file made by dcm2niix
        (same name with '.json'), if it exists, or from the image header.

        Parameters
        ----------
        fmri_img: str, Path, or nibabel image object
            If fmri_img is str or Path, it is the filename of the sample
            data (NIfTI or AFNI BRIK).
        """
        # Read header
        if isinstance(fmri_img, (str, Path)):
            img = nib.load(fmri_img)
            vol_shape = img.shape[:3]
            header = img.header
            fname = fmri_img
        else:
            vol_shape = fmri_img.shape[:3]
            fname = fmri_img.get_filename()
            header = fmri_img.header

        # Read json
        if '.gz' in Path(fname).suffixes:
            fname_stem = Path(Path(fname).stem).stem
        else:
            fname_stem = Path(fname).stem
        json_f = Path(fname).parent / (fname_stem + '.json')
        if json_f.is_file():
            with open(json_f) as fd:
                img_info = json.load(fd)
        else:
            img_info = None

        # Get TR
        TR = None
        if img_info and 'RepetitionTime' in img_info:
            # from json by dcm2niix
            TR = float(img_info['RepetitionTime'])
        elif hasattr(header, 'info') and 'TAXIS_FLOATS' in header.info:
            # from AFNI header
            TR = header.info['TAXIS_FLOATS'][1]
        else:
            zooms = header.get_zooms()
            if len(zooms) > 3:
                TR = zooms[3]

        # Get slice timing
        slice_timing = []
        if img_info and 'SliceTiming' in img_info:
            # from json by dcm2niix
            slice_timing = img_info['SliceTiming']
        elif hasattr(header, 'info') and 'TAXIS_OFFSETS' in header.info:
            slice_timing = header.info['TAXIS_OFFSETS']
        elif hasattr(header, 'get_slice_times'):
            try:
                slice_timing = header.get_slice_times()
            except nib.spatialimages.HeaderDataError:
                pass

        # Get the slice dimension: the array axis whose size matches the
        # number of slice times. dcm2niix and RtpWatch store the slices along
        # the third array axis, whatever the slice orientation is, so the
        # third axis is preferred when several axes match.
        slice_dim = None
        if len(slice_timing):
            dims = [d for d in range(3) if vol_shape[d] == len(slice_timing)]
            if 2 in dims:
                slice_dim = 2
            elif len(dims):
                slice_dim = dims[0]

        # Set parameters
        notset_params = []
        if TR is not None:
            self.set_param('TR', TR)
            self._logger.info(f'TR = {self.TR}')
        else:
            notset_params.append('TR')

        if len(slice_timing):
            self.set_param('slice_timing', slice_timing)
            self._logger.info(f'slice_timing = {self.slice_timing}')
        else:
            notset_params.append('slice_timing')

        if slice_dim is not None:
            self.set_param('slice_dim', slice_dim)
            self._logger.info(f'slice_dim = {self.slice_dim}')
        else:
            notset_params.append('slice_dim')

        if len(notset_params):
            errmsg = ', '.join(notset_params)
            errmsg += ' could not be read from the sample.'
            self._logger.warning(errmsg)
            self.err_popup(errmsg)

        # Set the interpolation weights
        self._calc_weights(vol_shape)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _calc_weights(self, vol_shape):
        """
        Calculate the weights for temporal interpolation in advance.
        Each weight is a per-slice array shaped to broadcast over a volume.

        Set variables
        -------------
        Interpolation weights
        self.wm1: weight for the volume at t-1
        self.w0: weight for the current volume (t)

        Only for cubic interpolation
        self.wm2: weight for the volume at t-2
        self.wp1: weight for the volume at t+1
        self.w0p1: w0 + wp1, used because the volume at t+1 is not
                   available and the current volume is used in its place

        Retrospective (1 TR back) extrapolation weights
        self.r1wm1: weight for the volume at t-1
        self.r1w0: weight for the current volume (t)

        Only for cubic interpolation
        self.r1wm2: weight for the volume at t-2
        self.r1wp1: weight for the volume at t+1
        self.r1w0p1: r1w0 + r1wp1 (the current volume is used for t+1)
        self.r1wm2m1: r1wm2 + r1wm1 (the volume at t-1 is used for t-2)
        """

        if self.slice_timing is None or self.TR is None:
            errmsg = "Slice timing is not set."
            self._logger.error(errmsg)
            self.err_popup(errmsg)
            return

        if self.method not in ['linear', 'cubic']:
            errmsg = f"{self.method} is not supported."
            self._logger.error(errmsg)
            self.err_popup(errmsg)
            return

        if self.slice_dim not in (0, 1, 2) or \
                vol_shape[self.slice_dim] != len(self.slice_timing):
            errmsg = 'Slice timing array does not match the data.'
            self._logger.error(errmsg)
            return

        # Slice timing shift from ref_time (relative to TR)
        shf = (np.asarray(self.slice_timing, dtype=np.float64) -
               self.ref_time) / self.TR

        # Shape the per-slice weights to broadcast over a volume.
        wshape = [1, 1, 1]
        wshape[self.slice_dim] = len(shf)

        def _w(w):
            return np.reshape(w, wshape).astype(np.float32)

        if self.method == 'linear':
            self.wm1 = _w(shf)  # weight for t-1
            self.w0 = _w(1.0 - shf)  # weight for t
            # Retrospective weights
            self.r1wm1 = _w(shf + 1.0)
            self.r1w0 = _w(-shf)

        elif self.method == 'cubic':
            aa = 1.0 - shf
            wm2 = aa * (1.0-aa) * (aa-2.0) * 0.1666667
            wm1 = (aa+1.0) * (aa-1.0) * (aa-2.0) * 0.5
            w0 = aa * (aa+1.0) * (2.0-aa) * 0.5
            wp1 = aa * (aa+1.0) * (aa-1.0) * 0.1666667
            self.wm2, self.wm1 = _w(wm2), _w(wm1)
            self.w0, self.wp1 = _w(w0), _w(wp1)
            self.w0p1 = _w(w0 + wp1)

            # Retrospective weights (1 TR back)
            aa = 1.0 - shf - 1.0
            r1wm2 = aa * (1.0-aa) * (aa-2.0) * 0.1666667
            r1wm1 = (aa+1.0) * (aa-1.0) * (aa-2.0) * 0.5
            r1w0 = aa * (aa+1.0) * (2.0-aa) * 0.5
            r1wp1 = aa * (aa+1.0) * (aa-1.0) * 0.1666667
            self.r1wm2, self.r1wm1 = _w(r1wm2), _w(r1wm1)
            self.r1w0, self.r1wp1 = _w(r1w0), _w(r1wp1)
            self.r1w0p1 = _w(r1w0 + r1wp1)
            self.r1wm2m1 = _w(r1wm2 + r1wm1)

        self._weight_shape = tuple(vol_shape)
        self.prep_weight = True

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_slice_timing_uis(self):
        """Show TR and slice_timing in the GUI panel, if any."""
        self._ui_update('TR', self.TR)
        self._ui_update('slice_timing', self.slice_timing)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_param(self, attr, val=None, reset_fn=None, echo=False):
        """
        When reset_fn is None, set_param is considered to be called from
        the load_parameters function.
        """
        self._logger.debug(f"set_param: {attr} = {val}")

        # -- Check value --
        if attr == 'enabled':
            self._ui_update(attr, val)

        elif attr == 'work_dir':
            if val is None or not Path(val).is_dir():
                return

            val = Path(val)
            setattr(self, attr, val)

            if self.main_win is not None:
                self.main_win.set_workDir(val)

        elif attr in ('method', 'ref_time') and reset_fn is None:
            self._ui_update(attr, val)

        elif attr == 'set_from_sample':
            if val is None:
                fname = self.select_file_dlg(
                        'TSHIFT: Select a slice timing sample',
                        self.work_dir, "*.BRIK* *.nii*")
                if fname[0] == '':
                    return -1

                val = fname[0]

            self.set_from_sample(val)
            self.set_slice_timing_uis()
            return 0

        elif attr == 'TR' and reset_fn is None:
            self._ui_update(attr, val)

        elif attr == 'slice_timing':
            if type(val) is str:
                try:
                    val = eval(val)
                except Exception:
                    if reset_fn:
                        reset_fn(str(getattr(self, attr)))
                    return
            else:
                self._ui_update(attr, val)

        elif attr == 'slice_dim' and reset_fn is None:
            self._ui_update(attr, val)

        elif attr == 'save_proc':
            self._ui_update(attr, val)

        elif reset_fn is None:
            # Ignore an unrecognized parameter
            if not hasattr(self, attr):
                errmsg = f"{attr} is an unrecognized parameter."
                self._logger.error(errmsg)
                return

        # -- Set value --
        setattr(self, attr, val)
        if attr in ('method', 'ref_time', 'TR', 'slice_timing', 'slice_dim'):
            # The interpolation weights must be computed again
            self.prep_weight = False
        if echo:
            print("{}.".format(self.__class__.__name__) + attr, '=',
                  getattr(self, attr))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_params(self):
        all_opts = super().get_params()
        # Exclude the process state and the interpolation weights
        excld_opts = ('work_dir', 'pre_data', 'pre_fmri_img', 'prep_weight',
                      'wm2', 'wm1', 'w0', 'wp1', 'w0p1',
                      'r1wm2', 'r1wm1', 'r1w0', 'r1wp1', 'r1w0p1', 'r1wm2m1')
        sel_opts = {}
        for k, v in all_opts.items():
            if k in excld_opts:
                continue
            if isinstance(v, Path):
                v = str(v)
            sel_opts[k] = v

        return sel_opts


# %% __main__ (test) ==========================================================
if __name__ == '__main__':
    # --- Test ---
    # test data directory
    test_dir = Path(__file__).absolute().parent.parent / 'tests'

    # Load test data
    testdata_f = test_dir / 'func_epi.nii.gz'
    assert testdata_f.is_file()
    img = nib.load(testdata_f)
    img_data = np.asanyarray(img.dataobj)
    N_vols = img.shape[-1]

    work_dir = test_dir / 'work'
    if not work_dir.is_dir():
        work_dir.mkdir()

    # Create RtpTshift instance
    rtp_tshift = RtpTshift()
    rtp_tshift.method = 'cubic'
    rtp_tshift.ref_time = 0

    # Set slice timing from a sample data
    rtp_tshift.set_from_sample(testdata_f)

    # Set parameters for debug
    rtp_tshift.work_dir = work_dir
    rtp_tshift.save_proc = True
    rtp_tshift.save_delay = True

    # Run tshift
    rtp_tshift.end_reset()

    N_vols = img.shape[-1]
    for ii in range(N_vols):
        save_filename = f"test_nr_{ii:04d}.nii.gz"
        fmri_img = nib.Nifti1Image(img_data[:, :, :, ii], affine=img.affine)
        fmri_img.set_filename(save_filename)
        st = time.time()
        rtp_tshift.do_proc(fmri_img, ii, st)

    proc_delay = rtp_tshift.proc_delay
    rtp_tshift.end_reset()

    # --- Load processed data  ---
    img_ts = nib.load(rtp_tshift.saved_filename)
    img_data_ts = img_ts.get_fdata()

    # --- Plot ---
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(proc_delay[1:], bins='auto')
    print(f"Median processing delay: {np.median(proc_delay[1:]):.4f}s")

    plot_range = 30  # TRs
    sli = 1
    xi = np.arange(plot_range)
    plt.figure()
    xraw = (np.array(xi)+rtp_tshift.ignore_init) * rtp_tshift.TR + \
        rtp_tshift.slice_timing[sli]
    plt.plot(xraw, img_data[64, 64, sli, rtp_tshift.ignore_init+xi],
             linestyle='-', marker='o', label='raw')
    xshift = (np.array(xi)+rtp_tshift.ignore_init) * rtp_tshift.TR + \
        rtp_tshift.ref_time
    plt.plot(xshift, img_data_ts[64, 64, sli, xi],
             linestyle='--', marker='x', label='shift')
    plt.legend()
    plt.title(f"Shift -{rtp_tshift.slice_timing[sli]:.3f}s")
    plt.xlabel('Time (s)')
