# -*- coding: utf-8 -*-
"""
RTP retrots for making online RETROICOR regressors

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
import warnings
import numpy as np
from scipy.signal import lfilter, firwin, hilbert, convolve
from scipy.interpolate import interp1d


# %% RtpRetroTS class ========================================================
class RtpRetroTS():
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, zero_phase_offset=0,
                 respiration_cutoff_frequency=3,
                 cardiac_cutoff_frequency=[0.1, 3],
                 cardiac_detend_window=5,
                 fir_order=40):
        """
        Parameters
        zero_phase_offset: float (optional)
        """
        self.zero_phase_offset = zero_phase_offset
        self.respiration_cutoff_frequency = respiration_cutoff_frequency
        self.cardiac_cutoff_frequency = cardiac_cutoff_frequency
        self.cardiac_detend_window = cardiac_detend_window
        self.fir_order = fir_order

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def RetroTs(self, resp, card, TR, physFS, tshift=0, Nvol=None):
        """
        RetroTS process function
        Return regressors for tshift timiming slice.

        Options
        -------
        Resp : array
            Respiration signal data array
        Card : array
            Cardiac signal data array
        NReg : int
            Length of regressor
        Retrun
        ------
        RetroTS regressor
        """

        # Set parameters
        self._phys_fs = physFS
        self._TR = TR
        self._slice_offset = tshift
        resp = self._clean_resamp(resp)
        card = self._clean_resamp(card)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            # Find peaks
            respiration_peak = self.peak_finder(
                resp, self.respiration_cutoff_frequency)

            cardiac_peak = self.peak_finder(
                card, self.respiration_cutoff_frequency,
                detend_window=self.cardiac_detend_window)

        # Phase estimate and create regressors
        resp_reg = self.phase_estimator(1, respiration_peak)
        card_reg = self.phase_estimator(0, cardiac_peak)

        if Nvol is None:
            Nvol = min(resp_reg.shape[0], card_reg.shape[0])

        resp_reg = resp_reg[:Nvol, :]
        card_reg = card_reg[:Nvol, :]
        reg = np.concatenate([resp_reg, card_reg], axis=1)

        return reg

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def peak_finder(self, v, frequency_cutoff, detend_window=None):

        ret = {}  # return dict

        # De-mean
        v = v - np.mean(v)
        ret['v'] = v

        # Filtering with frequency_cutoff
        # FIR filter
        if type(frequency_cutoff) in (list, np.array):
            pass_zero = 'bandpass'
        else:
            pass_zero = 'lowpass'
        b = firwin(numtaps=(self.fir_order + 1), cutoff=frequency_cutoff,
                   window="hamming", pass_zero=pass_zero, fs=self._phys_fs)

        # Filter both ways to cancel phase shift
        v = lfilter(b, 1, v, axis=0)
        v = np.flipud(v)
        v = lfilter(b, 1, v)
        v = np.flipud(v)

        if detend_window is not None:
            # Detrend with moving average of this second
            ww = int(detend_window * self._phys_fs)
            trend = convolve(v, np.ones(ww)/ww, mode='same')
            n_conv = int(np.ceil(ww/2))
            trend[0] *= ww/n_conv
            for i in range(1, n_conv):
                trend[i] *= ww/(i+n_conv)
                trend[-i] *= ww/(i + n_conv - (ww % 2))
            v = v - trend

        # analytic_signal
        x = hilbert(v)

        # Find peaks
        fsi = 1.0 / self._phys_fs
        t = np.arange(0, (len(x)+1)*fsi, fsi)
        t = t[:len(x)]
        img_chg = (x[:-1].imag * x[1:].imag)
        iz = np.nonzero(img_chg <= 0)[0]
        polall = -np.sign(x[:-1].imag - x[1:].imag)
        pk = (x[iz]).real
        pol = polall[iz]

        # Polishing: Find a peak around the identified points
        window_width = 0.2  # Window for adjusting peak location in seconds
        nww = int(np.ceil((window_width / 2) * self._phys_fs))
        pkp_ = pk
        for ii in range(len(iz)):
            n0 = int(max(2, iz[ii]-nww))
            n1 = int(min(len(x), iz[ii]+nww))
            wval = (x[n0:n1+1]).real
            if pol[ii] > 0:
                xx, ixx = np.max(wval), np.argmax(wval)
            else:
                xx, ixx = np.min(wval), np.argmin(wval)
            iz[ii] = n0 + ixx - 1
            pkp_[ii] = xx
        tizp_ = t[iz]

        ppp = np.nonzero(pol > 0)[0]
        p_trace = pkp_[ppp]
        tp_trace = tizp_[ppp]

        npp = np.nonzero(pol < 0)[0]
        n_trace = pkp_[npp]
        tn_trace = tizp_[npp]

        ntrace = min(len(p_trace), len(n_trace))
        p_trace = p_trace[:ntrace]
        tp_trace = tp_trace[:ntrace]
        n_trace = n_trace[:ntrace]
        tn_trace = tn_trace[:ntrace]

        # remove duplicates
        okflag = (np.diff(tp_trace) > 0.3) & (np.diff(tn_trace) != 0)
        okflag = [True, *okflag]
        p_trace = p_trace[okflag]
        tp_trace = tp_trace[okflag]
        n_trace = n_trace[okflag]
        tn_trace = tn_trace[okflag]

        # Remove peak points in the opposite side
        irr_ppidx = []
        for ppidx, pv in enumerate(p_trace):
            mean_pp = np.mean(p_trace[max(ppidx-5, 0):ppidx+5])
            mean_np = np.mean(n_trace[max(ppidx-5, 0):ppidx+5])
            if np.abs(pv - mean_pp) > np.abs(pv - mean_np):
                irr_ppidx.append(ppidx)

        irr_npidx = []
        for npidx, pv in enumerate(n_trace):
            mean_pp = np.mean(p_trace[max(npidx-5, 0):npidx+5])
            mean_np = np.mean(n_trace[max(npidx-5, 0):npidx+5])
            if np.abs(pv - mean_pp) < np.abs(pv - mean_np):
                irr_npidx.append(npidx)

        p_trace = np.delete(p_trace, irr_ppidx)
        tp_trace = np.delete(tp_trace, irr_ppidx)

        n_trace = np.delete(n_trace, irr_npidx)
        tn_trace = np.delete(tn_trace, irr_npidx)

        # Positive and negative peaks must appear alternately.
        tpn = np.concatenate([tp_trace, tn_trace])
        pol = np.concatenate([np.ones_like(tp_trace),
                              np.ones_like(tn_trace)*-1])
        tpn_sidx = np.argsort(tpn).ravel()
        tpn = tpn[tpn_sidx]
        pol = pol[tpn_sidx]
        not_alt_idx = np.argwhere(np.diff(pol) == 0).ravel()
        rm_tp = []
        rm_tn = []
        for nli in not_alt_idx:
            t0 = tpn[nli]
            t1 = tpn[nli+1]
            if t1 - t0 < window_width:
                if pol[nli] > 0:
                    v0 = p_trace[tp_trace == t0][0]
                    v1 = p_trace[tp_trace == t1][0]
                    if v0 > v1:
                        rm_tp.append(t1)
                    else:
                        rm_tp.append(t0)
                else:
                    v0 = n_trace[tn_trace == t0][0]
                    v1 = n_trace[tn_trace == t1][0]
                    if v0 < v1:
                        rm_tn.append(t1)
                    else:
                        rm_tn.append(t0)
                continue

            tmask = (t > t0) & (t < t1)
            trange = t[tmask]
            vals = x[tmask].real
            if pol[nli] > 0:
                peak_v = np.min(vals)
                # If the negative peak is on the positive side, two positive
                # peaks are merged into the higher one.
                tr_idx = np.argwhere(tp_trace == t0).ravel()[0]
                p_mean = np.mean(p_trace[max(0, tr_idx-5): tr_idx+5])
                n_mean = np.mean(n_trace[max(0, tr_idx-5): tr_idx+5])
                if np.abs(peak_v - p_mean) < np.abs(peak_v - n_mean):
                    # Remove the lower peak points
                    p0 = p_trace[tp_trace == t0][0]
                    p1 = p_trace[tp_trace == t1][0]
                    if p0 < p1:
                        rm_tp.append(t0)
                    else:
                        rm_tp.append(t1)
                else:
                    # Add negative peak points
                    n_trace = np.append(n_trace, peak_v)
                    tn_trace = np.append(tn_trace, trange[np.argmin(vals)])
            else:
                peak_v = np.max(vals)
                # If the positive peak is on the negative side, two negative
                # peaks are merged into the lower one.
                tr_idx = np.argwhere(tn_trace == t0).ravel()[0]
                p_mean = np.mean(p_trace[max(0, tr_idx-5): tr_idx+5])
                n_mean = np.mean(n_trace[max(0, tr_idx-5): tr_idx+5])
                if np.abs(peak_v - p_mean) > np.abs(peak_v - n_mean):
                    # Remove the lower peak points
                    p0 = n_trace[tn_trace == t0][0]
                    p1 = n_trace[tn_trace == t1][0]
                    if p0 > p1:
                        rm_tn.append(t0)
                    else:
                        rm_tn.append(t1)
                else:
                    # Add positive peak points
                    p_trace = np.append(p_trace, peak_v)
                    tp_trace = np.append(tp_trace, trange[np.argmin(vals)])

        if len(rm_tp):
            rmidx = np.argwhere([tt in rm_tp for tt in tp_trace]).ravel()
            p_trace = np.delete(p_trace, rmidx)
            tp_trace = np.delete(tp_trace, rmidx)

        if len(rm_tn):
            rmidx = np.argwhere([tt in rm_tn for tt in tn_trace]).ravel()
            n_trace = np.delete(n_trace, rmidx)
            tn_trace = np.delete(tn_trace, rmidx)

        sidx = np.argsort(tp_trace).ravel()
        p_trace = p_trace[sidx]
        tp_trace = tp_trace[sidx]

        sidx = np.argsort(tn_trace).ravel()
        n_trace = n_trace[sidx]
        tn_trace = tn_trace[sidx]

        # Calculate the period
        prd = np.diff(tp_trace)

        '''
        from pylab import plot, subplot, show, text, figure
        figure(1)
        plot(t, np.real(x), "g")
        plot(tp_trace, p_trace, "r-")
        plot(tp_trace, p_trace, "r.")
        plot(tn_trace, n_trace, "b-")
        plot(tn_trace, n_trace, "b.")
        show()
        '''
        ret.update(
            {'t': t, 'p_trace': p_trace, 'tp_trace': tp_trace,
             'tn_trace': tn_trace, 'prd': prd})

        return ret

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def phase_estimator(self, amp_type, peak_vars):

        '''
        import numpy as np
        v = phasee['v'].copy()
        t = phasee['t'].copy()
        tp_trace = phasee['tp_trace'].copy()
        tn_trace = phasee['tn_trace'].copy()
        TR = phasee["volume_tr"]
        number_of_slices = phasee["number_of_slices"]
        slice_offset = phasee["slice_offset"]
        zero_phase_offset = phasee["zero_phase_offset"]
        prd = phasee["prd"]
        '''

        v = peak_vars['v']
        t = peak_vars['t']
        p_trace = peak_vars['p_trace']
        tp_trace = peak_vars['tp_trace']
        tn_trace = peak_vars['tn_trace']
        prd = peak_vars['prd']

        if amp_type == 0:
            # Calculate the phase of the trace, with the peak to be the start
            # of the phase
            phase = -2 * np.ones_like(t)
            for ii, tp0 in enumerate(tp_trace[:-1]):
                tp1 = tp_trace[ii+1]
                for jj in np.argwhere((t >= tp0) & (t < tp1)).ravel():
                    phase[jj] = (t[jj] - tp0) / prd[ii] + \
                        self.zero_phase_offset
                    if phase[jj] < 0:
                        phase[jj] = -phase[jj]
                    if phase[jj] > 1:
                        phase[jj] -= 1

            # Remove the points flagged as unset
            phase[np.nonzero(phase < -1)[0]] = 0.0
            # Change phase to radians
            phase = phase * 2 * np.pi
        else:
            '''
            import numpy as np
            p_trace = phasee["p_trace"]
            tp_trace = phasee["tp_trace"][:-1]
            tn_trace = phasee["tn_trace"][:-1]
            v = phasee["v"]
            t = phasee["t"]
            prd = phasee["prd"]
            '''
            # phase based on amplitude
            # scale to the max
            mxamp = np.max(p_trace)
            vmin = np.min(v)
            vmax = np.max(v)
            # Scale, per Glover 2000's paper
            gR = ((v - vmin) / (vmax - vmin)) * vmax
            bins = np.arange(0.01, 1.01, 0.01) * mxamp
            hb_value = self._my_hist(gR, bins)

            # Set the phase polarity of each time point in v:
            # rising = 1, falling = -1
            phase_pol = np.zeros_like(v)
            tp_tr = tp_trace.copy()
            tn_tr = tn_trace.copy()

            if tp_tr[0] < tn_tr[0]:
                pp = tp_tr[0]
                cpol = -1
                tp_tr = tp_tr[1:]
            else:
                pp = tn_tr[0]
                cpol = 1
                tn_tr = tn_tr[1:]
            phase_pol[t == pp] = cpol

            while len(tp_tr) and len(tn_tr):
                pp0 = pp
                if tp_tr[0] < tn_tr[0]:
                    pp = tp_tr[0]
                    cpol = 1
                    tp_tr = tp_tr[1:]
                else:
                    pp = tn_tr[0]
                    cpol = -1
                    tn_tr = tn_tr[1:]
                phase_pol[(t > pp0) & (t <= pp)] = cpol

            pp0 = pp
            if len(tp_tr):
                pp = tp_tr[0]
                cpol = 1
                tp_tr = tp_tr[1:]
            elif len(tn_tr):
                pp = tn_tr[0]
                cpol = -1
                tn_tr = tn_tr[1:]
            phase_pol[(t > pp0) & (t <= pp)] = cpol
            phase_pol[t > pp] = cpol * -1

            gR = np.clip(np.floor(gR / mxamp * 100).astype(int), 0, 99)
            hbsum = np.cumsum(hb_value/np.sum(hb_value))
            phase = np.pi * hbsum[gR] * phase_pol

        # Time series time vector
        time_series_time = np.arange(
            0, t.max()-self._TR/2 + np.finfo(float).eps, self._TR)
        phase_slice_reg = np.zeros([len(time_series_time), 4])
        tslc = time_series_time + self._slice_offset
        ph_idx = [min(max(0, int(np.round(ts*self._phys_fs))), len(phase))
                  for ts in tslc]
        phase_slice = phase[ph_idx]

        # Make regressors for each slice
        phase_slice_reg[:, 0] = np.sin(phase_slice)
        phase_slice_reg[:, 1] = np.cos(phase_slice)
        phase_slice_reg[:, 2] = np.sin(2*phase_slice)
        phase_slice_reg[:, 3] = np.cos(2*phase_slice)

        return phase_slice_reg

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _my_hist(self, x, bin_centers):
        """
        http://stackoverflow.com/questions/18065951/why-does-numpy-histogram-python-leave-off-one-element-as-compared-to-hist-in-m
        :param x:dataset
        :param bin_centers:bin values in a list to be moved from edges to
        centers
        :return: counts = the data in bin centers ready for pyplot.bar
        """
        bin_edges = np.r_[-np.inf, 0.5*(bin_centers[:-1]+bin_centers[1:]),
                          np.inf]
        counts, edges = np.histogram(x, bin_edges)
        return counts

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _clean_resamp(self, v):
        idx_good = np.argwhere(~np.isnan(v)).ravel()
        if idx_good[0] > 0:
            v[:idx_good[0]] = v[idx_good[0]]
        if idx_good[-1] < len(v)-1:
            v[idx_good[-1]:] = v[idx_good[-1]]

        idx_bad = np.argwhere(np.isnan(v)).ravel()
        if len(idx_bad):
            x = np.arange(0, len(v))
            xg = np.setdiff1d(x, idx_bad)
            v[idx_bad] = interp1d(xg, v[xg], bounds_error=None)(idx_bad)

        return v


# %% __main__ (test) ==========================================================
if __name__ == '__main__':
    rtp_retrots = RtpRetroTS()
    rtp_retrots.TR = 1.5
    # if not rtp_retrots.ready_proc():
    #     sys.stderr.write("Failed to be ready for proc.\n")

    # if not call_rt_physio(rtp_retrots.rtp_physio_address, 'ping'):
    #     sys.exit()

    # # call_rt_physio(rtp_retrots.rtp_physio_address, 'WAIT_TTL_ON')
    # # time.sleep(30)

    # NVol = 20
    # for _ in range(10):
    #     reg = rtp_retrots.do_proc(NVol)
    #     print(reg)
    #     time.sleep(1.5)
    #     NVol += 1

    from pathlib import Path
    card_f = Path('/data/rt/S20240104150322/Card_500Hz_ser-12.1D')
    resp_f = Path('/data/rt/S20240104150322/Resp_500Hz_ser-12.1D')
    resp = np.loadtxt(resp_f)
    card = np.loadtxt(card_f)
    rtp_retrots._phys_fs = 500
    reg = rtp_retrots.RetroTs(resp, card)

    # for _ in range(10):
    #     st = time.time()
    #     reg = rtp_retrots.RetroTs(resp, card)
    #     print(f'total: {time.time() - st}')

    # for n in range(20, 201):
    #     num_points = int(n * TR * PhysFS)
    #     reg = restrots.do_proc(Resp[:num_points], Card[:num_points], TR,
    #                            PhysFS)
