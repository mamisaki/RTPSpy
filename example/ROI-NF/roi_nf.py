#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LA-NF application with RtpApp

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import time
import sys
import numpy as np
import nibabel as nib
from rtpspy import RtpApp


# %% ROINF class =============================================================
class ROINF(RtpApp):
    """
    Example class for overriding a neurofeedback signal calculation in the
    do_proc method.
    """
    def do_proc(self, fmri_img, vol_idx=None, pre_proc_time=0):
        """
        Extract the neurofeedback signal from the processed fMRI image, and
        send it to an external application.

        Parameters
        ----------
        fmri_img : nibabel NIfTI1 imageobject
            Processed fMRI image.
        vol_idx : int, optional
            Volume index from the scan start (0 base). The default is None.
        pre_proc_time : float, optional
            Time (seconds) of previous process end. The default is 0.

        """
        try:
            # Load ROI if self.ROI_mask is not set.
            if Path(self.ROI_orig).is_file() and self.ROI_mask is None:
                # Load ROI mask
                self.ROI_mask = np.asanarry(nib.load(self.ROI_orig).dataobj)

            self._vol_num += 1  # 1- base number of volumes recieved by this
            if vol_idx is None:
                vol_idx = self._vol_num - 1  # 0-base index

            if vol_idx < self.ignore_init:
                # Skip ignore_init volumes
                return

            if self._proc_start_idx < 0:
                self._proc_start_idx = vol_idx

            dataV = fmri_img.get_fdata()  # Get data

            # --- Extract and send the ROI mean signal ------------------------
            roimask = (self.ROI_mask > 0) & (np.abs(dataV) > 0.0)
            mean_sig = np.nanmean(dataV[roimask])
            self._roi_sig[0].append(mean_sig)

            if self.extApp_sock is None:
                # Error: Socket is not opened.
                self._logger.error('No socket to an external app.')

            else:
                # Send data to an external application via socket.
                # Message format should be;
                # "NF {time},{volume_index},{signal_value}(,{signal_value}...)"
                try:
                    scan_onset = self.rtp_objs['TTLPHYSIO'].scan_onset
                    val_str = f"{time.time()-scan_onset:.4f},"
                    val_str += f"{vol_idx},{mean_sig:.6f}"
                    msg = f"NF {val_str};"

                    self.send_extApp(msg.encode(), no_err_pop=True)
                    self._logger.info(f"Sent '{msg}' to an external app")

                except Exception as e:
                    self._logger.error(str(e))

            # --- Post procress -----------------------------------------------
            # Record the processing time
            tstamp = time.time()
            self._proc_time.append(tstamp)
            if pre_proc_time is not None:
                proc_delay = self._proc_time[-1] - pre_proc_time
                if self.save_delay:
                    self.proc_delay.append(proc_delay)

            # Log message
            f = Path(fmri_img.get_filename()).name
            msg = f"#{vol_idx+1};ROI signal extraction;{f}"
            msg += f";tstamp={tstamp}"
            if pre_proc_time is not None:
                msg += f';took {proc_delay:.4f}s'
            self._logger.info(msg)

            # Update the signal plot
            self._plt_xi.append(vol_idx+1)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errmsg = '{}, {}:{}'.format(
                    exc_type, exc_tb.tb_frame.f_code.co_filename,
                    exc_tb.tb_lineno)
            errmsg = str(e) + '\n' + errmsg
            self._logger.error(errmsg)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ready_to_run(self):
        super().ready_to_run()

        # Send READY message to extApp
        if self.send_extApp('READY;'.encode()):
            recv = self.recv_extApp(timeout=3)
            if recv is not None:
                self._logger.debug(f"Recv {recv.decode()}")
