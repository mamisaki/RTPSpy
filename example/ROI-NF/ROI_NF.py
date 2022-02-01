#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LA-NF application with RTP_APP

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import time
import sys
import numpy as np
import nibabel as nib
from rtpspy import RTP_APP


# %% ROI_NF class =============================================================
class ROI_NF(RTP_APP):
    """
    Example class to override a neurofeedback signal calculation in do_proc
    method.
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

            self.vol_num += 1  # Number of volumes recieved by this module.
            if vol_idx is None:
                vol_idx = self.vol_num

            if vol_idx < self.ignore_init:
                # Skip ignore_init volumes
                return

            if self.proc_start_idx < 0:
                self.proc_start_idx = vol_idx

            dataV = fmri_img.get_fdata()  # Get data

            # --- Extract and send the ROI mean signal ------------------------
            roimask = (self.ROI_mask > 0) & (np.abs(dataV) > 0.0)
            mean_sig = np.nanmean(dataV[roimask])
            self.roi_sig[0].append(mean_sig)

            if self.extApp_sock is None:
                # Error: Socket is not opened.
                self.errmsg('No socket to an external app.', no_pop=True)

            else:
                # Send data to an external application via socket.
                # Message format should be;
                # "NF {time},{volume_index},{signal_value}(,{signal_value}...)"
                try:
                    scan_onset = self.rtp_objs['SCANONSET'].scan_onset
                    val_str = f"{time.time()-scan_onset:.4f},"
                    val_str += f"{vol_idx},{mean_sig:.6f}"
                    msg = f"NF {val_str};"

                    self.send_extApp(msg.encode(), no_pop_err=True)
                    if self._verb:
                        self.logmsg(f"Sent '{msg}' to an external app")

                except Exception as e:
                    self.errmsg(str(e), no_pop=True)

            # --- Post procress -----------------------------------------------
            # Record the processing time
            self.proc_time.append(time.time())
            if pre_proc_time is not None:
                proc_delay = self.proc_time[-1] - pre_proc_time
                if self.save_delay:
                    self.proc_delay.append(proc_delay)

            # Log message
            if self._verb:
                f = Path(fmri_img.get_filename()).name
                msg = f'#{vol_idx}, ROI signal extraction is done for {f}'
                if pre_proc_time is not None:
                    msg += f' (took {proc_delay:.4f}s)'
                msg += '.'
                self.logmsg(msg)

            # Update the signal plot
            self.plt_xi.append(vol_idx)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            errmsg = '{}, {}:{}'.format(
                    exc_type, exc_tb.tb_frame.f_code.co_filename,
                    exc_tb.tb_lineno)
            self.errmsg(str(e) + '\n' + errmsg, no_pop=True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ready_to_run(self):
        super().ready_to_run()

        # Send READY message to extApp
        if self.send_extApp('READY;'.encode()):
            recv = self.recv_extApp(timeout=3)
            if recv is not None:
                if self._verb:
                    self.logmsg(f"Recv {recv.decode()}")
