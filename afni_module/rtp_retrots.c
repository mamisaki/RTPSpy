/*
 * rt_RetroTS.c
 *
 *  Created by Nafise Barzigar <nbarzigar@laureateinstitute.org>
 *  Edited by Masaya Misaki <mmisaki@laureateinstitute.org>
 *
 *  RetroTS.m MATLAB code in AFNI package is converted to C code by Nafise
 *  Barzigar.
 *  Masaya Misaki modified the code to apply online processing and added garbage
 *  collection.
 */

#include "rtp_retrots.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <fftw3.h>

/*--- Define variable types ---*/
#define PI	M_PI	/* pi to machine precision, defined in math.h */
#define HISTSIZE 100
//#define RIC_HISTFUDGE 0.0000001

static int extp = 0;  // bool flag to extrapolate at interp1

typedef struct {
	int sig_len; // length of input signal
	double *v; // input signal
	double* t; // time (sec) of input signal
	fftw_complex* X; // (double) complex representation of input signal
	double* ptrace; // real values of X at positive change points
	double * tptrace; // time (sec) of positive change points
	int ptracesize; // length of positive change points (ptrace)
	double * ntrace; // real values of X negative change points
	double * tntrace; // time (sec) of negative change points
	int ntracesize; // length of negative change points (ntrace)
	int * iz; // indices where imag(X[0:len-1]).*imag(X[1:len]) <= 0
	double * prd; //period between positive peaks
	double * tmidprd; //middle time of each period
	double * phz; //phase at each time point (sig_len length)
	double * RV; // respiration volume (ptracesize length)
	double * RVT; // respiration volume per time RV/prd (ptracesize length)
	double * ptraceR; // resampled ptrace values at each t point (linearly interpolated)
	double * ntraceR; // resampled ntrace values at each t point (linearly interpolated)
	double * prdR; // resampled period values at each t point (linearly interpolated)
	int * phz_pol; // phase polarity of each time point in v
	double * tst; // time of MRI volume
	int tstsize; // length of tst (MRI volume)
	double * phz_slc; // Phase time series for each slice (only 0 slice here)
	double ** phz_slc_reg; // phase based regresor (4 basis sin,cos is used)
	double * RVR; // resampled respiration volume at each t point (linearly interpolated)
	double * RVTR; // resampled respiration volume per time at each t point (linearly interpolated)
	double * RVTRS; // smoothed resampled respiration volume per time at each t point (linearly interpolated)
	double ** RVTRS_slc; //RVT regressor
	int RVTshiftslength; // number of RVT shift
} R_E;

typedef struct {
	int *array;
	size_t used;
	size_t size;
} Array;

#define max(a,b) \
		({ __typeof__ (a) _a = (a); \
		__typeof__ (b) _b = (b); \
		_a > _b ? _a : _b; })
#define min(a,b) \
		({ __typeof__ (a) _a = (a); \
		__typeof__ (b) _b = (b); \
		_a < _b ? _a : _b; })

/*--- Function prototypes ---*/
static void PeakFinder(double* vvec, int len, RetroTSOpt* Opt_RT, R_E* RE);
static void RVT_from_PeakFinder(R_E * RE, RetroTSOpt* Opt_RT);
static void PhaseEstimator(R_E * RE, RetroTSOpt * Opt_RT);

static double* my_fir1(int N, double Wn);
static void filter(int ord, double* b, int len, double* x, double* y);
static void flipV(double* x, int len, double* y);
static fftw_complex* analytic_signal(double * vi, int nvi, double windwidth,
		int percover, int hamwin);
static void fftsegs(double ww, int po, int nv, int* num, Array* ble, Array* bli);
static void initArray(Array* a, size_t initialSize);
static void insertArray(Array* a, int element);
static void freeArray(Array* a);
static void hamming(double* out, int windowLength);
static int* findImag(int nvi, fftw_complex* x, int limit, int* izSize);
static int* signImag(int len, fftw_complex* x);
static void max_arrayReal(fftw_complex* a, int n0, int n1, double* xx, int* ixx);
static void min_arrayReal(fftw_complex* a, int n0, int n1, double* xx, int* ixx);
static int remove_duplicates(double *t, double *v, int len);
static double* interp1(double* x, double* y, int in_size, double* xi,
		int out_size, int extp);
static int min_index_array(double *a, int sizea);
static double max_array(double *a, int sizea);
static double* zscale(double *x, int xsize, double ub, double lb);
static int * histC(double *x, int len, int nbin);
static double* daxpy(int len, double* x, int a);
static void sin_array(double* x, int len, double* y);
static void cos_array(double* x, int len, double* y);
static void freeR_E(R_E* RE);

//static void print_mtx(int n, int m, double* X);
//static void write_int_mtx(unsigned long n, unsigned long m, int* X);
//static void write_flt_mtx(unsigned long n, unsigned long m, float* X);
//static void write_dbl_mtx(unsigned long n, unsigned long m, double* X);

/*==============================================================================
 * RetroTS
 */
int rtp_retrots(RetroTSOpt *Opt, double *Resp, double *ECG, int len,
		float *regOut) {
	/*
	 *
	 *
	 */
	RetroTSOpt *OptR, *OptE;
	R_E *R, *E;

	//-- Initialize parameters --
	Opt->zerophaseoffset = 0;
	Opt->RVTshifts[0] = 0;
	Opt->RVTshifts[1] = 5;
	Opt->RVTshifts[2] = 10;
	Opt->RVTshifts[3] = 15;
	Opt->RVTshifts[4] = 20;
	Opt->RVTshiftslength = 5;
	Opt->RespCutoffFreq = 3;
	Opt->CardCutoffFreq = 3;
	strcpy(Opt->ResamKernel, "linear");
	Opt->FIROrder = 40;

	//-- Create option copy for Respiration signal --
	OptR = (RetroTSOpt*) malloc(sizeof(RetroTSOpt));
	memcpy(OptR, Opt, sizeof(RetroTSOpt));
	OptR->fcutoff = Opt->RespCutoffFreq;
	OptR->AmpPhase = 1; //bool flag for amplitude based phase for respiration
	// OptR.as_percover = 50; %percent overlap of windows for fft
	// OptR.as_windwidth = 0; %window width in seconds for fft, 0 for full window
	// OptR.as_fftwin = 0 ; %1 == hamming window. 0 == no windowing

	//-- Create option copy for ECG signal --
	OptE = (RetroTSOpt*) malloc(sizeof(RetroTSOpt));
	memcpy(OptE, Opt, sizeof(RetroTSOpt));
	OptE->fcutoff = Opt->CardCutoffFreq;
	OptE->AmpPhase = 0; //time based phase for cardiac signal

	//-- Get the peaks for Respiration and E --
	R = (R_E*) malloc(sizeof(R_E));
	memset(R, 0, sizeof(R_E)); //Initialize with 0,NULL
	PeakFinder(Resp, len, OptR, R);

	E = (R_E*) malloc(sizeof(R_E));
	memset(E, 0, sizeof(R_E)); //Initialize with 0,NULL
	PeakFinder(ECG, len, OptE, E);

	//-- Phase estimation for Resp and ECG --
	PhaseEstimator(R, OptR);
	PhaseEstimator(E, OptE);

	//-- Computing RVT from peaks --
	RVT_from_PeakFinder(R, OptR);

	//-- Put results in regOut --
	int i, t; // iteration indices
	int reg_len = R->tstsize;
	int num_RVT = OptR->RVTshiftslength;
	int reg_num = num_RVT + 4 + 4; // number of regressors
	int reg_pos = 0; // writing position of regOut

	// RVT
	for (i = 0; i < OptR->RVTshiftslength; i++) {
		for (t = 0; t < reg_len; t++) {
			regOut[reg_pos + t * reg_num] = (float) R->RVTRS_slc[i][t];
		}
		reg_pos++;
	}

	// Resp ricor
	for (i = 0; i < 4; i++) {
		for (t = 0; t < reg_len; t++) {
			regOut[reg_pos + t * reg_num] = (float) R->phz_slc_reg[i][t];
		}
		reg_pos++;
	}

	// ECG ricor
	for (i = 0; i < 4; i++) {
		for (t = 0; t < reg_len; t++) {
			regOut[reg_pos + t * reg_num] = (float) E->phz_slc_reg[i][t];
		}
		reg_pos++;
	}

	//-- Free memory --
	free(OptR);
	free(OptE);
	freeR_E(R);
	free(R);
	freeR_E(E);
	free(E);

	return 0;
}

/*==============================================================================
 * Peak finder
 */
void PeakFinder(double* vvec, int sig_len, RetroTSOpt* Opt_RT, R_E* RE) {
	/* Find peaks in vvec
	 * Input:
	 * 	vvec; pointer to input vector values
	 * 	len; lenght of vvec
	 * 	Opt_RT; options
	 * 	RE; output
	 */

	int i; // loop counter

	//-- Set options if not set --
	if (Opt_RT->fcutoff == 0) //( Opt_RT->fcutoff == NULL) |
		Opt_RT->fcutoff = 10;

	if (Opt_RT->FIROrder == 0)
		Opt_RT->FIROrder = 80;

	if (strlen(Opt_RT->ResamKernel) == 0)
		strcpy(Opt_RT->ResamKernel, "linear");

	//-- Some filtering --

	// Prepare fir filter
	double fnyq = Opt_RT->PhysFS / 2.0; // Nyquist frequency
	double w = Opt_RT->fcutoff / fnyq;  // upper cut off frequency normalized
	double* b = my_fir1(Opt_RT->FIROrder, w); //FIR filter

	// Copy vvec in v
	double *v;
	v = (double*) malloc(sizeof(double) * sig_len);
	memcpy(v, vvec, sizeof(double) * sig_len);

	// getting mean of v
	double meanv, sum;
	sum = 0.0;
	for (i = 0; i < sig_len; ++i)
		sum += v[i];

	meanv = sum / sig_len;

	for (i = 0; i < sig_len; ++i)
		v[i] = v[i] - meanv; //remove the mean

	RE->v = v;
	RE->sig_len = sig_len;

	// filter both ways to cancel phase shift
	double *tmpv, *filtv; // Temporary space for filtering and flipping vector
	tmpv = (double *) malloc(sizeof(double) * sig_len);
	filtv = (double *) malloc(sizeof(double) * sig_len);
	double *a;
	a = calloc(Opt_RT->FIROrder + 1, sizeof(double));
	a[0] = 1;
	filter(Opt_RT->FIROrder, b, sig_len, RE->v, tmpv);
	flipV(tmpv, sig_len, filtv);
	filter(Opt_RT->FIROrder, b, sig_len, filtv, tmpv);
	flipV(tmpv, sig_len, filtv);
	free(a);
	free(b);
	free(tmpv);

	//-- Get complex of filtered signal --
	double windw = Opt_RT->as_windwidth * Opt_RT->PhysFS;
	RE->X = analytic_signal(filtv, sig_len, windw, Opt_RT->as_percover,
			Opt_RT->as_fftwin);
	free(filtv);

	//-- Set time of sample in t --
	double samp_intv = 1 / Opt_RT->PhysFS; // sampling interval
	RE->t = (double *) malloc(sizeof(double) * sig_len);
	for (i = 0; i < sig_len; ++i)
		RE->t[i] = i * samp_intv;

	//-- Get signal change points --
	int* iz; // indices where imag(RE->X[0:len-1]).*imag(Re->X[1:len]) <= 0
	int izSize; //length of iz
	iz = findImag(sig_len, RE->X, 0, &izSize);
	int* polall; //polarity of imag(X) change (sig_len-1 length)
	polall = signImag(sig_len, RE->X);

	double *pk; //real values of RE->X at iz
	int *pol; //polarity at at iz
	double *tiz; // time at iz
	pk = (double *) malloc(sizeof(double) * izSize);
	pol = (int *) malloc(sizeof(int) * izSize);
	tiz = (double *) malloc(sizeof(double) * izSize);
	int izi;
	int pos_pol_n = 0; // number of positive pol
	int neg_pol_n = 0; // number of negative pol
	for (i = 0; i < izSize; ++i) {
		izi = iz[i];
		pk[i] = RE->X[izi][0];
		pol[i] = polall[izi];
		if (pol[i] > 0)
			pos_pol_n++;
		else if (pol[i] < 0)
			neg_pol_n++;

		tiz[i] = RE->t[izi];
	}
	free(polall);

	double *ptrace, *ntrace; //value at positive/negative change point
	double *tptrace, *tntrace; //time at positive/negative change point
	ptrace = (double *) malloc(sizeof(double) * pos_pol_n);
	tptrace = (double *) malloc(sizeof(double) * pos_pol_n);
	ntrace = (double *) malloc(sizeof(double) * neg_pol_n);
	tntrace = (double *) malloc(sizeof(double) * neg_pol_n);
	pos_pol_n = 0;
	neg_pol_n = 0;
	for (i = 0; i < izSize; ++i) {
		if (pol[i] > 0) {
			ptrace[pos_pol_n] = pk[i];
			tptrace[pos_pol_n] = tiz[i];
			pos_pol_n++;
		}
		if (pol[i] < 0) {
			ntrace[neg_pol_n] = pk[i];
			tntrace[neg_pol_n] = tiz[i];
			neg_pol_n++;
		}
	}

	//-- Some polishing --
	double windwidth = 0.2; //window for adjusting peak location in seconds
	int nww = ceil(windwidth / 2 * Opt_RT->PhysFS);
	RE->iz = iz;
	int n0, n1, ixx;
	double xx;
	for (i = 0; i < izSize; ++i) {
		n0 = max(1, iz[i] - nww);
		n1 = min(sig_len, iz[i] + nww + 1);
		if (pol[i] > 0)
			max_arrayReal(RE->X, n0, n1, &xx, &ixx);
		else
			min_arrayReal(RE->X, n0, n1, &xx, &ixx);

		RE->iz[i] = ixx;
		pk[i] = xx;
		tiz[i] = RE->t[ixx];
	}

	pos_pol_n = 0;
	neg_pol_n = 0;
	for (i = 0; i < izSize; ++i) {
		if (pol[i] > 0) {
			ptrace[pos_pol_n] = pk[i];
			tptrace[pos_pol_n] = tiz[i];
			pos_pol_n++;
		}
		if (pol[i] < 0) {
			ntrace[neg_pol_n] = pk[i];
			tntrace[neg_pol_n] = tiz[i];
			neg_pol_n++;
		}
	}

	// Save values in RE struct variable
	RE->ptracesize = pos_pol_n;
	RE->ptrace = ptrace;
	RE->tptrace = tptrace;

	RE->ntracesize = neg_pol_n;
	RE->ntrace = ntrace;
	RE->tntrace = tntrace;

	free(pol);
	free(tiz);
	free(pk);

	//remove duplicates that might come up when improving peak location
	RE->ptracesize = remove_duplicates(RE->tptrace, RE->ptrace, RE->ptracesize);
	RE->ntracesize = remove_duplicates(RE->tntrace, RE->ntrace, RE->ntracesize);

	//-- Calculate the period--
	int nptrc = RE->ptracesize;
	RE->prd = (double *) malloc((nptrc - 1) * sizeof(double));
	RE->tmidprd = (double *) malloc((nptrc - 1) * sizeof(double));
	for (i = 0; i < nptrc - 1; i++) {
		RE->prd[i] = RE->tptrace[i + 1] - RE->tptrace[i];
		RE->tmidprd[i] = (RE->tptrace[i + 1] + RE->tptrace[i]) / 2.0;
	}

	if (strlen(Opt_RT->ResamKernel) != 0) {
		//interpolate to slice sampling time grid:
		//linear interpolation
		RE->ptraceR = interp1(RE->tptrace, RE->ptrace, RE->ptracesize, RE->t,
				RE->sig_len, extp);
		RE->ntraceR = interp1(RE->tntrace, RE->ntrace, RE->ntracesize, RE->t,
				RE->sig_len, extp);
		RE->prdR = interp1(RE->tmidprd, RE->prd, RE->ptracesize - 1, RE->t,
				RE->sig_len, extp);
		/* In matlab code, you may get NaN when t exceeds original signal time,
		 * so set those to the last interpolated value, but in this code we
		 * never get the NaN, so no need for clean_resamp function.
		 */
	}

}

/*==============================================================================
 * Get the phase
 */
void PhaseEstimator(R_E* RE, RetroTSOpt* Opt_RT) {
	int i, j, tt;
	if ((Opt_RT->AmpPhase == 0)) {
		//-- Cardiac--
		RE->phz = (double *) malloc(RE->sig_len * sizeof(double));
		for (tt = 0; tt < RE->sig_len; tt++)
			RE->phz[tt] = -2;

		j = 0;
		for (i = 0; i < RE->ptracesize - 1; i++) {
			while (RE->t[j] < RE->tptrace[i + 1]) {
				if (RE->t[j] >= RE->tptrace[i]) {
					/* Note: Using a constant244 period for each interval causes
					 * slope discontinuity within a period. One should resample
					 * prd(i) so that it is estimated at each time in RE->t[j]
					 * dunno if that makes much of a difference in the end
					 * however.
					 */
					RE->phz[j] = (RE->t[j] - (RE->tptrace[i])) / RE->prd[i]
							+ Opt_RT->zerophaseoffset;

					if (RE->phz[j] < 0)
						RE->phz[j] = -RE->phz[j];

					if (RE->phz[j] > 1)
						RE->phz[j] = RE->phz[j] - 1;
				}
				j = j + 1;
			}
		}

		// remove the points flagged as unset
		for (tt = 0; tt < RE->sig_len; tt++) {
			if (RE->phz[tt] < -1)
				RE->phz[tt] = 0.0;
		}

		// change phase to radians
		for (tt = 0; tt < RE->sig_len; tt++)
			RE->phz[tt] = RE->phz[tt] * 2 * PI;

	} else {
		//-- Respiration- phase based on amplitude --
		// At first, scale input signal to 0-max(ptrace)
		double mxamp = max_array(RE->ptrace, RE->ptracesize);
		double *gR; // scaled signal
		gR = zscale(RE->v, RE->sig_len, mxamp, 0.0); //scale to 0-mxamp

		// Get histogram
		int* hb = histC(gR, RE->sig_len, HISTSIZE);

		//	find the polarity of each time point in v
		RE->phz_pol = (int *) calloc(RE->sig_len, sizeof(int));
		i = 0;
		// keep 0 until min(RE->tptrace[0],RE->tntrace[0])
		while (i < RE->sig_len && RE->t[i] < RE->tptrace[0]
				&& RE->t[i] < RE->tntrace[0])
			i++;

		int cpol;
		int itp = 0;
		int inp = 0;
		if (RE->tptrace[0] < RE->tntrace[0]) {
			cpol = -1; // expiring phase, peak behind us
			itp = 1;
		} else {
			cpol = 1; // inspiring phase (bottom behind us)
			inp = 1;
		}

		while (i < RE->sig_len) {
			RE->phz_pol[i] = cpol;
			if (RE->t[i] == RE->tptrace[itp]) {
				cpol = -1;
				itp = min(itp + 1, RE->ptracesize - 1);
			} else if (RE->t[i] == RE->tntrace[inp]) {
				cpol = +1;
				inp = min(inp + 1, RE->ntracesize - 1);
			}
			i++;
		}

		/* Now that we have the polarity, without computing sign(dR/dt) as in
		 * Glover et al 2000, calculate the phase per eq. 3 of that paper
		 */
		// First the sum in the numerator
		for (i = 0; i < RE->sig_len; i++) {
			gR[i] = round(gR[i] / mxamp * HISTSIZE);
			if (gR[i] > HISTSIZE - 1)
				gR[i] = HISTSIZE - 1;
		}

		int shb = 0; // sum of hb
		for (i = 0; i < HISTSIZE; ++i)
			shb += hb[i];

		// Cumulative density
		double* hbsum; // Cumulative density
		hbsum = (double*) calloc(HISTSIZE, sizeof(double));
		hbsum[0] = (double) (hb[0]) / shb;
		for (i = 1; i < HISTSIZE; i++)
			hbsum[i] = hbsum[i - 1] + (double) (hb[i]) / shb;

		RE->phz = (double *) malloc(RE->sig_len * sizeof(double));
		for (i = 0; i < RE->sig_len; i++)
			RE->phz[i] = PI * hbsum[(int) round(gR[i])] * RE->phz_pol[i];

		free(hbsum);
		free(gR);
		free(hb);
	}

	//-- Calculating phz_slc_reg--
	RE->tstsize = (int) ((RE->t[RE->sig_len - 1] - 0.5 * Opt_RT->VolTR)
			/ Opt_RT->VolTR) + 1;
	RE->tst = (double *) malloc(sizeof(double) * RE->tstsize);
	for (i = 0; i < RE->tstsize; i++)
		RE->tst[i] = i * Opt_RT->VolTR;

	RE->phz_slc = (double *) malloc(sizeof(double) * RE->tstsize);
	RE->phz_slc_reg = (double **) calloc(4, sizeof(double*));
	for (i = 0; i < 4; i++)
		RE->phz_slc_reg[i] = (double *) malloc(RE->tstsize * sizeof(double));

	// Estimate only for the one slice at Opt_RT->tshift offest
	int idmin;
	double* temp = (double *) malloc(sizeof(double) * (RE->sig_len));

	for (i = 0; i < RE->tstsize; i++) {
		for (tt = 0; tt < RE->sig_len; tt++)
			temp[tt] = fabs(RE->tst[i] + Opt_RT->tshift - RE->t[tt]);

		idmin = min_index_array(temp, RE->sig_len);
		RE->phz_slc[i] = RE->phz[idmin];
	}
	free(temp);

	// Make regressors
	sin_array(RE->phz_slc, RE->tstsize, RE->phz_slc_reg[0]);
	cos_array(RE->phz_slc, RE->tstsize, RE->phz_slc_reg[1]);

	double* phz_slcCol = daxpy(RE->tstsize, RE->phz_slc, 2.0); // 2*RE->phz_slc
	sin_array(phz_slcCol, RE->tstsize, RE->phz_slc_reg[2]);
	cos_array(phz_slcCol, RE->tstsize, RE->phz_slc_reg[3]);
	free(phz_slcCol);

}

/*==============================================================================
 * RVT from PeakFinder
 */
void RVT_from_PeakFinder(R_E * RE, RetroTSOpt* Opt_RT) {
	/* Calculate RVT
	 */

	// DEBUG
	// FILE *fp;

	int i; // loop counter

	if (RE->ptracesize != RE->ntracesize) {
		int dd = abs(RE->ptracesize - RE->ntracesize);
		if (dd > 1) { // have not seen this yet, trap for it.
			fprintf(stderr, "RT: rt_retroTS, Error RVT_from_PeakFinder:\n");
			fprintf(stderr, "    Peak trace lengths differ by %d\n", dd);
		} else { // When the difference is 1, discard one sample
			if (RE->ptracesize < RE->ntracesize) {
				RE->ntracesize = RE->ptracesize;
				RE->ntrace = (double*) realloc(RE->ntrace,
						RE->ntracesize * sizeof(double));
			} else {
				RE->ptracesize = RE->ntracesize;
				RE->ptrace = (double*) realloc(RE->ptrace,
						RE->ptracesize * sizeof(double));
			}
		}
	}

	RE->RV = (double *) malloc(sizeof(double) * RE->ptracesize);
	for (i = 0; i < RE->ptracesize; ++i)
		RE->RV[i] = RE->ptrace[i] - RE->ntrace[i];
	/* Need to consider which starts first and whether to initialize first two
	 * values by means and also, what to do when we are left with one
	 * incomplete pair at the end.
	 */

	RE->RVT = (double *) malloc(sizeof(double) * (RE->ptracesize - 1));
	for (i = 0; i < RE->ptracesize - 1; ++i)
		RE->RVT[i] = RE->RV[i] / RE->prd[i];

	if (RE->ptraceR) { // When Opt_RT->ResamKernel is set ptraceR is calculated
		RE->RVR = (double *) malloc(sizeof(double) * RE->sig_len);
		RE->RVTR = (double *) malloc(sizeof(double) * RE->sig_len);
		for (i = 0; i < RE->sig_len; ++i) {
			RE->RVR[i] = RE->ptraceR[i] - RE->ntraceR[i];
			RE->RVTR[i] = RE->RVR[i] / RE->prdR[i];
		}

#ifdef DEBUG
		fp = fopen("RVTR.bin", "wb");
		fwrite(RE->RVTR, sizeof(double), RE->sig_len, fp);
		fclose(fp);
#endif

		// smooth RVT so that we can resample it at VolTR later
		double fnyq = Opt_RT->PhysFS / 2.0; // nyquist frequency of physio sampling
		double w = Opt_RT->fcutoff / fnyq; // normalized upper cut off frequency
		double* b = my_fir1(Opt_RT->FIROrder, w); // FIR filter

		double sum, mean;

		sum = 0.0;
		for (i = 0; i < RE->sig_len; ++i)
			sum += RE->RVTR[i];

		mean = sum / RE->sig_len;
		double* v = (double *) malloc(sizeof(double) * RE->sig_len);
		for (i = 0; i < RE->sig_len; ++i)
			v[i] = RE->RVTR[i] - mean; //remove the mean

		// filter both ways to cancel phase shift
		double* a = calloc(Opt_RT->FIROrder + 1, sizeof(double));
		a[0] = 1;
		double* y = (double *) malloc(sizeof(double) * RE->sig_len);
		double* z = (double *) malloc(sizeof(double) * RE->sig_len);
		filter(Opt_RT->FIROrder, b, RE->sig_len, v, y);

		//flipV(y, RE->sig_len, z);
		filter(Opt_RT->FIROrder, b, RE->sig_len, z, y);
		//flipV(y, RE->sig_len, v);

		RE->RVTRS = (double *) malloc(sizeof(double) * RE->sig_len);
		for (i = 0; i < RE->sig_len; ++i)
			RE->RVTRS[i] = v[i] + mean; // Add back mean

		free(a);
		free(b);
		free(v);
		free(y);
		free(z);
	}

#ifdef DEBUG
	fp = fopen("RVTRS.bin", "wb");
	fwrite(RE->RVTRS, sizeof(double), RE->sig_len, fp);
	fclose(fp);
#endif

	//-- Create RVT regressors --

	// Initialize RVTRS_slc
	RE->RVTshiftslength = Opt_RT->RVTshiftslength;
	RE->RVTRS_slc = (double **) calloc(Opt_RT->RVTshiftslength,
			sizeof(double*));

	int shf;
	double nsamp;
	int j;
	int* sind = (int *) malloc(sizeof(int) * RE->sig_len);
	double* temp = (double *) malloc(sizeof(double) * (RE->sig_len));

	for (i = 0; i < (Opt_RT->RVTshiftslength); ++i) {
		shf = Opt_RT->RVTshifts[i];
		nsamp = round(shf * Opt_RT->PhysFS);
		for (j = 0; j < RE->sig_len; ++j) {
			sind[j] = j + nsamp;
			if (sind[j] < 0)
				sind[j] = 0;

			if (sind[j] >= RE->sig_len)
				sind[j] = RE->sig_len - 1;

			temp[j] = RE->RVTRS[sind[j]];
		}

		//linear interpolation
		RE->RVTRS_slc[i] = interp1(RE->t, temp, RE->sig_len, RE->tst,
				RE->tstsize, extp);

#ifdef DEBUG
		if (i == 4) {
		 	fp = fopen("test_temp.bin", "wb");
			fwrite(temp, sizeof(double), RE->sig_len, fp);
			fclose(fp);
		}
#endif
	}

	free(sind);
	free(temp);
}

/*==============================================================================
 Utility functions
 =============================================================================*/

/*==============================================================================
 * FIR filter
 */
double* my_fir1(int N, double Wn) {
	/* FIR filter
	 * Input:
	 * 	N; length of filter (FIRorder)
	 * 	Wn; normalized upper cut off frequency
	 * Output:
	 * 	return pointer to filter vector (N+1 length must is allocated)
	 */

	int Pr_L = N + 1;
	double* bb = (double *) malloc(sizeof(double) * Pr_L);
	double gain = 0.0000000;

	N = N + 1;
	int odd = N - (N / 2) * 2; /* odd = rem(N,2) */

	int i;
	double wind[Pr_L]; // wind = hamming(N)
	for (i = 0; i < Pr_L; i++)
		wind[i] = 0.54 - 0.46 * cos((2 * PI * i) / (N - 1));

	double f1 = Wn / 2.0;
	double c1 = f1;
	int nhlf = (N + 1) / 2;

	//-- Low-pass --
	double b[Pr_L / 2];
	if (odd)
		b[0] = 2 * c1;

	// b[odd:nhlf]=(sin(c3)/c)
	double c, c3;
	for (i = odd; i < nhlf; i++) {
		c = PI * (i + 0.5 * (1 - odd));
		c3 = 2 * c1 * c;
		b[i] = sin(c3) / c;
	}

	/* bb = real([b(nhlf:-1:i1) b(1:nhlf)].*wind(:)') */
	for (i = 0; i < nhlf; i++)
		bb[i] = b[nhlf - 1 - i];

	for (i = nhlf; i < Pr_L; i++)
		bb[i] = b[i - Pr_L / 2];

	for (i = 0; i < Pr_L; i++)
		bb[i] = bb[i] * wind[i];

	/* gain = abs(polyval(b,1)); */
	for (i = 0; i < Pr_L; i++)
		gain += bb[i];

	/* b = b / gain */
	for (i = 0; i < Pr_L; i++)
		bb[i] = bb[i] / gain;

	return bb;
}

/*==============================================================================
 * filter
 */
void filter(int ord, double *b, int len, double *x, double *y) {
	/* 1-D digital filter
	 * Input:
	 * 	ord; order of filter (length of b is ord+1)
	 * 	b; filter coefficients
	 * 	len; length of signal
	 * 	x; signal vector
	 * 	y; output, n length should be allocated
	 *
	 * 	2018/08/01:	DEBUG by Masaya Misaki
	 */

	double acc;
	double *bp, *xp, *xbuf;
	int i, j; // loop counters

	xbuf = (double *) calloc(ord + len, sizeof(double));
	memcpy(xbuf + ord, x, len * sizeof(double));

	// apply filter to each input sample
	for (i = 0; i < len; i++) {
		// calculate output n
		bp = b;
		xp = xbuf + ord + i;
		acc = 0;
		for (j = 0; j < ord + 1; j++) {
			acc += (*bp++) * (*xp--);
		}
		y[i] = acc;
	}

	free(xbuf);

}

/*==============================================================================
 * flipV
 */
void flipV(double* x, int len, double* y) {
	/* Flip vector
	 * Input:
	 * 	x; input vector
	 * 	len; length of vector
	 * Output:
	 * 	y; flipped vector
	 */

	int i;
	for (i = 0; i < len; i++)
		y[i] = x[len - 1 - i];
}

/*==============================================================================
 * analytic_signal
 */
fftw_complex* analytic_signal(double* vi, int nvi, double windwidth,
		int percover, int hamwin) {
	/* Return real and image (complex) vector for input vector vi
	 *
	 */

	// Get the segements that are to be used for fft calculations
	// (fftsegs(100,70,1000,bli, ble, num);)
	int* num = (int*) malloc(sizeof(int) * nvi);
	Array bli;
	Array ble;
	initArray(&ble, 1);
	initArray(&bli, 1);
	fftsegs(windwidth, percover, nvi, num, &ble, &bli);

	// variables for fft with fftw3
	// http://www.fftw.org/
	fftw_complex *in_cmplx, *spect_cmplx;

	// other variables
	int bi; // loop counter across blocks
	int ii; // loop counter within loop
	int nv; // length of block
	double *v; // working vector of block
	double *ham; // buffer for hamming
	double *wind; // window for FFT filter

	// output vector
	fftw_complex *h = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nvi);
	memset(h, 0, sizeof(double) * nvi * 2);

	//-- Process for each segment --
	for (bi = 0; bi < bli.size; bi++) {
		// Extract segment
		nv = abs(bli.array[bi] - ble.array[bi]);
		v = (double*) malloc(sizeof(double) * nv);
		for (ii = 0; ii < nv; ii++)
			v[ii] = vi[ii + bli.array[bi]];

		// Prepare complex variables
		fftw_plan fftplan;
		in_cmplx = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nv);
		spect_cmplx = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nv);

		if (hamwin == 1) {
			// apply hamming window
			ham = (double*) malloc(sizeof(double) * nvi);
			hamming(ham, nv);
			for (ii = 0; ii < nv; ii++)
				v[ii] *= ham[ii];

			free(ham);
		}

		// set value in complex for FFT
		for (ii = 0; ii < nv; ii++) {
			in_cmplx[ii][0] = v[ii];
			in_cmplx[ii][1] = 0.0;
		}

		// calculate FFT
		fftplan = fftw_plan_dft_1d(nv, in_cmplx, spect_cmplx, FFTW_FORWARD,
		FFTW_ESTIMATE);
		fftw_execute(fftplan);

		// Prepare filter window
		// (zero negative frequencies, double positive frequencies)
		wind = (double *) calloc(nv, sizeof(double));
		if (nv % 2 == 0) { // x is even
			wind[0] = 1; // keep DC
			wind[nv / 2] = 1;
			for (ii = 1; ii < nv / 2; ii++)
				wind[ii] = 2; // double positive freq.
		} else { // x is odd
			wind[0] = 1; // keep DC
			for (ii = 1; ii < (nv + 1) / 2; ii++)
				wind[ii] = 2; // double positive freq.
		}

		// Apply filter to spectrum
		for (ii = 0; ii < nv; ii++) {
			spect_cmplx[ii][0] *= wind[ii];
			spect_cmplx[ii][1] *= wind[ii];
		}

		// calculate IFFT
		fftw_plan ifftplan;
		ifftplan = fftw_plan_dft_1d(nv, spect_cmplx, in_cmplx, FFTW_BACKWARD,
		FFTW_ESTIMATE);
		fftw_execute(ifftplan);

		// Save segment result into output complex vector
		// (ifft should be normalized by the signal length)
		for (ii = 0; ii < nv; ii++) {
			h[ii + bli.array[bi]][0] += in_cmplx[ii][0] / nv;
			h[ii + bli.array[bi]][1] += in_cmplx[ii][1] / nv;
		}

		// free working vector
		fftw_free(in_cmplx);
		fftw_free(spect_cmplx);
		fftw_destroy_plan(fftplan);
		fftw_destroy_plan(ifftplan);
		free(v);
		free(wind);
	}

	// divide h by the number of evaluated times in segment (num)
	for (bi = 0; bi < nvi; bi++) {
		h[bi][0] /= num[bi];
		h[bi][0] /= num[bi];
	}

	free(num);
	freeArray(&bli);
	freeArray(&ble);

	return h;
}

/*==============================================================================
 * fftsegs
 */
void fftsegs(double ww, int po, int nv, int* num, Array* ble, Array* bli) {
	/* Returns the segements that are to be used for fft calculations.
	 * Input:
	 * 	ww: Segment width (in number of samples)
	 * 	po: Percent segment overlap
	 * 	nv: Total number of samples in original symbol
	 * Output:
	 * 	num: An nv x 1 vector containing the number of segments each sample
	 * 	     belongs to.
	 * 	bli, ble: Two Nblck x 1 vectors defining the segments' starting and
	 * 	          ending indices.
	 */

	int out, cnt, i;
	int jmp, nblck, ib, mina;

	if (ww == 0) {
		po = 0;
		ww = nv;
	} else if (ww < 32 || ww > nv) {
		printf("Error fftsegs: Bad value for window width of %f\n", ww);
		return;
	}

	out = 0;
	while (out == 0) {
		freeArray(ble);
		freeArray(bli);

		//How many blocks?
		jmp = floor((100 - po) * ww / 100); //jump from block to block
		nblck = nv / jmp;  //number of jumps
		ib = 0;
		cnt = -1;

		while (cnt < 0 || ble->array[cnt] < nv) {
			cnt++;
			insertArray(bli, ib);  // automatically resizes as necessary
			mina = min(ib + ww, nv);
			insertArray(ble, mina);  // automatically resizes as necessary
			ib = ib + jmp;
		}

		//if the last block is too small, spread the love
		if ((ble->array[cnt] - bli->array[cnt]) < 0.1 * ww) {
			// too small a last block, merge
			ble->array[cnt - 1] = ble->array[cnt]; // into previous
			cnt--;
			out = 1;
		} else if (ble->array[cnt] - bli->array[cnt] < 0.75 * ww) {
			// too large to merge, spread it
			ww = ww + floor((ble->array[cnt] - bli->array[cnt]) / nblck);
			out = 0;
		} else
			//last block big enough, proceed
			out = 1;
	}

	//now figure out the number of estimates each point of the time series gets
	memset(num, 0, nv * sizeof(int));
	cnt = 0;
	while (cnt < ble->size) {
		for (i = bli->array[cnt]; i < ble->array[cnt]; ++i)
			num[i] += 1;
		cnt++;
	}
}

/*==============================================================================
 * initArray
 */
void initArray(Array *a, size_t initialSize) {
	a->array = (int *) malloc(initialSize * sizeof(int));
	a->used = 0;
	a->size = initialSize;
}

/*==============================================================================
 * insertArray
 */
void insertArray(Array *a, int element) {
	if (a->used == a->size) {
		//    a->size *= 2;
		a->size++;
		a->array = (int *) realloc(a->array, a->size * sizeof(int));
	}
	a->array[a->used++] = element;
}

/*==============================================================================
 * freeArray
 */
void freeArray(Array *a) {
	if (a->array != NULL) {
		free(a->array);
		a->array = NULL;
	}
	a->used = a->size = 0;
}

/*==============================================================================
 * hamming
 */
void hamming(double *out, int windowLength) {
	int n;
	int m = windowLength - 1;
	int halfLength = windowLength / 2;

	// Calculate taps
	// Due to symmetry, only need to calculate half the window
	for (n = 0; n <= halfLength; n++) {
		double val = 0.54 - 0.46 * cos(2.0 * M_PI * n / m);
		out[n] = val;
		out[windowLength - n - 1] = val;
	}
}

/*==============================================================================
 * findImag
 */
int* findImag(int nvi, fftw_complex* x, int limit, int * izSize) {
	/* Return indices where the sign of imag(x) is changed
	 * <=> x[t][1]*x[t+1][1] <= limit (=0)
	 * Input:
	 * 	nvi; length of x
	 * 	x; complex vector, each componet's [0] = real, [1] = image
	 * 	limit; threshhold of difference
	 * Output:
	 * 	izSize; length of indices found
	 * 	return pointer of int vector of indices
	 */

	int* index = (int *) malloc(sizeof(double) * nvi); //return vector
	int cnt = 0; // counter of indices
	int ii; // loop counter
	for (ii = 0; ii < nvi - 1; ii++) {
		if (x[ii][1] * x[ii + 1][1] <= limit) {
			index[cnt] = ii;
			cnt++;
		}
	}

	*izSize = cnt;
	index = realloc(index, sizeof(int) * cnt);
	return index;
}

/*==============================================================================
 * signImag
 */
int* signImag(int len, fftw_complex* x) {
	/* Return sign of imag change in complex vector
	 * Input:
	 * 	len; length of vector
	 * 	x; complex vector, [0]->real, [1]->image in each element
	 * Output:
	 * 	pointer to len length int vector of change polarity
	 */

	int* pol = (int*) malloc(sizeof(int) * (len - 1));
	int ii;
	double ch_imag; //change of image value
	for (ii = 0; ii < len - 1; ii++) {
		ch_imag = x[ii + 1][1] - x[ii][1];
		if (ch_imag > 0.0)
			pol[ii] = 1;
		else if (ch_imag < 0.0)
			pol[ii] = -1;
		else
			pol[ii] = 0;
	}
	return pol;
}

/*==============================================================================
 * max_arrayReal
 */
void max_arrayReal(fftw_complex* a, int n0, int n1, double* xx, int* ixx) {
	/* Get maximum value and index in real values of complex vector within a
	 * certain range.
	 * Input:
	 * 	a; complex vector
	 * 	n0,n1; begin and end positions of vector to find maximum
	 * Output:
	 * 	xx; maximum value
	 * 	ixx; index of maximum value, absolute in a
	 */
	int i, idx;
	double max = -DBL_MAX;
	for (i = n0; i < n1; i++) {
		if (a[i][0] > max) {
			max = a[i][0];
			idx = i;
		}
	}
	*xx = max;
	*ixx = idx;
}

/*==============================================================================
 * min_arrayReal
 */
void min_arrayReal(fftw_complex* a, int n0, int n1, double* xx, int* ixx) {
	/* Get minimum value and index in real values of complex vector within a
	 * certain range.
	 * Input:
	 * 	a; complex vector
	 * 	n0,n1; begin and end positions of vector to find minimum
	 * Output:
	 * 	xx; minimum value
	 * 	ixx; index of maximum value, absolute position of a
	 */
	int i, idx;
	double min = DBL_MAX;
	for (i = n0; i < n1; i++) {
		if (a[i][0] < min) {
			min = a[i][0];
			idx = i;
		}
	}
	*xx = min;
	*ixx = idx;
}

/*==============================================================================
 * remove_duplicates
 */
int remove_duplicates(double *t, double *v, int len) {
	/* remove duplicate
	 * if diff(t) <= 0.3 sec the later point is removed.
	 * Input:
	 * 	t; sequence of time
	 * 	v; value at t
	 * 	len; length of vector
	 * Output:
	 * 	return new length
	 */

	int i;
	int ok_i = 0;
	for (i = 1; i < len; i++) {
		if ((t[i] != t[i - 1]) && (t[i] - t[i - 1] > 0.3)) {
			//minimum time before next beat
			ok_i = ok_i + 1;
			t[ok_i] = t[i];
			v[ok_i] = v[i];
		}
	}

	return ok_i + 1;
}

/*==============================================================================
 * interp1
 */
double* interp1(double* x, double* y, int in_size, double* xi, int out_size,
		int extp) {
	/* Liner interpolation. Vector y is resampled at xi points.
	 * x and xi must be monotonically increasing.
	 * Note: In MATLAB's interp1, xi outside of x is filled by edge value of y.
	 * Here, if extp == 1 outside of x is linearly extrapolated.
	 *
	 * Input:
	 * 	x, y; input vector, x;sampling points, y; values
	 * 	in_size; size of x and y
	 * 	xi; resampling points
	 * 	out_size; size of resampling vector
	 * 	extp; bool flag for extrapolation
	 */

	int ii;
	double * yi = (double *) malloc(out_size * sizeof(double));

	// indices
	int idyi = 0; // index for the interpolating points

	double slope;

	// extrapolate left points
	slope = (y[1] - y[0]) / (x[1] - x[0]);
	while (idyi < out_size && xi[idyi] < x[0]) {
		if (extp) // Extrapolate left points
			yi[idyi] = y[0] - (slope * (x[0] - xi[idyi]));
		else
			yi[idyi] = y[0];

		++idyi;
	}

	// fill point in the same interval as the original values
	int idx = 1; // index for the original points
	while (idyi < out_size && idx < in_size) {
		//Debug
		if (idyi >= out_size || idx >= in_size) {
			printf("Invalid index idyi:%d>=%d, idx:%d>=%d\n", idyi, out_size,
					idx, in_size);
			exit(0);
		}

		//Move reference point of original value
		while (idx < in_size && xi[idyi] > x[idx]) {
			++idx;
			if (idx == in_size) //Debug
				break;
		}

		if (idx < in_size) {
			slope = (y[idx] - y[idx - 1]) / (x[idx] - x[idx - 1]);
			yi[idyi] = y[idx] - (slope * (x[idx] - xi[idyi]));
			++idyi;
		}
	}

	// extrapolate right points
	if (idyi < out_size) {
		slope = (y[in_size - 1] - y[in_size - 2])
				/ (x[in_size - 1] - x[in_size - 2]);
		while (idyi < out_size) {
			if (extp) // Extrapolate right points
				yi[idyi] = y[in_size - 1]
						+ (slope * (xi[idyi] - x[in_size - 1]));
			else
				// Fill with edge value
				yi[idyi] = yi[idyi - 1];

			++idyi;
		}
	}

	return yi;
}

/*==============================================================================
 * min_array
 */
double min_array(double *a, int sizea) {
	int i;
	double max = DBL_MAX;

	for (i = 0; i < sizea; i++) {
		if (a[i] < max) {
			max = a[i];
		}
	}
	return max;
}

/*==============================================================================
 * max_array
 */
double max_array(double *a, int sizea) {
	int i;
	double max = -DBL_MAX;

	for (i = 0; i < sizea; i++) {
		if (a[i] > max) {
			max = a[i];
		}
	}
	return max;
}

/*==============================================================================
 * min_index_array
 */
int min_index_array(double *a, int sizea) {
	int i, id;
	double max = DBL_MAX;

	for (i = 0; i < sizea; i++) {
		if (a[i] < max) {
			max = a[i];
			id = i;
		}
	}
	return id;
}

/*==============================================================================
 * zscale
 */
double* zscale(double *x, int xsize, double ub, double lb) {
	/* Comment for the original MATLAB code;
	 * This function scales  X into Y such that its maximum value is ub and
	 * minimum value is lb.
	 * If X is all constants, it gets scaled to ub;
	 * 	Ziad, Oct 30 96 / modified March 18 97
	 */

	double* y; //output vector
	y = (double *) malloc(xsize * sizeof(double));
	double xmin = min_array(x, xsize);
	double xmax = max_array(x, xsize);

	int i;
	if (xmin == xmax) {
		// If X is all constants, it gets scaled to ub;
		for (i = 0; i < xsize; i++)
			y[i] = ub;
	} else {
		for (i = 0; i < xsize; i++)
			y[i] = (((x[i] - xmin) / (xmax - xmin)) * (ub - lb)) + lb;
	}

	return y;
}

/*==============================================================================
 * Get the histogram
 */
int* histC(double* x, int len, int nbins) {
	/* Return histogram with nbins size
	 * bin centers = [1:nbins]./nbins.*(max(x)-min(x)) + min(x);
	 *
	 * TODO: For compatibility with MATLAB output, current implementation is
	 * not smart. Much faster algorithm should be implemented
	 */

	// Initialize output vector
	int* h = (int*) calloc(nbins, sizeof(int));

	// set bin width
	double maxx = max_array(x, len);
	double minx = min_array(x, len);
	double bin_width = (maxx - minx) / nbins;
	double bin_center, bin_left, bin_right;

	int i, j; // loop counter
	for (i = 0; i < nbins; i++) {
		bin_center = (i + 1) * bin_width;
		if (i == 0) {
			bin_left = -DBL_MAX;
			bin_right = bin_center + bin_width / 2.0;
		} else {
			bin_left = bin_right;
			bin_right = bin_left + bin_width;
		}

		if (i == nbins - 1)
			bin_right = DBL_MAX;

		for (j = 0; j < len; j++) {
			if (x[j] > bin_left && x[j] <= bin_right)
				h[i]++;
		}
	}

	return h;
}

/*==============================================================================
 * daxpy
 */
double* daxpy(int len, double* x, int a) {
	/* element-wise multiple vector x by a
	 */

	int i;
	double* y = (double *) malloc(len * sizeof(double));
	for (i = 0; i < len; i++)
		y[i] = x[i] * a;

	return y;
}

/*==============================================================================
 * sin_array
 */
void sin_array(double* x, int len, double* y) {
	/* Get the sin of array
	 */
	int i;
	for (i = 0; i < len; i++)
		y[i] = sin(x[i]);
}

/*==============================================================================
 * cos_array
 */
void cos_array(double* x, int len, double* y) {
	/* Get the cos of array
	 */
	int i;
	for (i = 0; i < len; i++)
		y[i] = cos(x[i]);
}

/*==============================================================================
 * freeR_E
 */
void freeR_E(R_E* RE) {
	/* free memory in RE
	 */

	if (RE->v != NULL)
		free(RE->v);

	if (RE->X != NULL)
		fftw_free(RE->X);

	if (RE->t != NULL)
		free(RE->t);

	if (RE->iz != NULL)
		free(RE->iz);

	if (RE->ptrace != NULL)
		free(RE->ptrace);

	if (RE->tptrace != NULL)
		free(RE->tptrace);

	if (RE->ntrace != NULL)
		free(RE->ntrace);

	if (RE->tntrace != NULL)
		free(RE->tntrace);

	if (RE->prd != NULL)
		free(RE->prd);

	if (RE->tmidprd != NULL)
		free(RE->tmidprd);

	if (RE->ptraceR != NULL)
		free(RE->ptraceR);

	if (RE->ntraceR != NULL)
		free(RE->ntraceR);

	if (RE->prdR != NULL)
		free(RE->prdR);

	if (RE->phz != NULL)
		free(RE->phz);

	if (RE->phz_pol != NULL)
		free(RE->phz_pol);

	if (RE->tst != NULL)
		free(RE->tst);

	if (RE->phz_slc != NULL)
		free(RE->phz_slc);

	int i;
	for (i = 0; i < 4; i++)
		if (RE->phz_slc_reg[i] != NULL)
			free(RE->phz_slc_reg[i]);

	free(RE->phz_slc_reg);

	if (RE->RV != NULL)
		free(RE->RV);

	if (RE->RVT != NULL)
		free(RE->RVT);

	if (RE->RVR != NULL)
		free(RE->RVR);

	if (RE->RVTR != NULL)
		free(RE->RVTR);

	if (RE->RVTRS != NULL)
		free(RE->RVTRS);

	for (i = 0; i < RE->RVTshiftslength; i++)
		if (RE->RVTRS_slc[i] != NULL)
			free(RE->RVTRS_slc[i]);

	free(RE->RVTRS_slc);

}

/*==============================================================================
 * print. write functions for debug
 */

//void print_mtx(int n, int m, double* X) {
//	int i, j;
//	for (i = 0; i < n; i++) {
//		for (j = 0; j < m; j++)
//			printf("%.4f ", X[i * m + j]);
//
//		printf("\n");
//	}
//}
//
//void write_int_mtx(unsigned long n, unsigned long m, int* X) {
//	FILE *fid;
//	char fname[] = "tmp.out";
//
//	fid = fopen(fname, "w");
//	int i, j;
//	for (i = 0; i < n; i++) {
//		for (j = 0; j < m; j++) {
//			fprintf(fid, "%d", X[i * m + j]);
//			if (j != m - 1)
//				fprintf(fid, " ");
//		}
//
//		fprintf(fid, "\n");
//	}
//	fclose(fid);
//
//}
//
//void write_flt_mtx(unsigned long n, unsigned long m, float* X) {
//	FILE *fid;
//	char fname[] = "tmp.out";
//
//	fid = fopen(fname, "w");
//	int i, j;
//	for (i = 0; i < n; i++) {
//		for (j = 0; j < m; j++) {
//			fprintf(fid, "%f", X[i * m + j]);
//			if (j != m - 1)
//				fprintf(fid, " ");
//		}
//
//		fprintf(fid, "\n");
//	}
//	fclose(fid);
//}
//
//void write_dbl_mtx(unsigned long n, unsigned long m, double* X) {
//	FILE *fid;
//	char fname[] = "tmp.out";
//
//	fid = fopen(fname, "w");
//	int i, j;
//	for (i = 0; i < n; i++) {
//		for (j = 0; j < m; j++) {
//			fprintf(fid, "%f", X[i * m + j]);
//			if (j != m - 1)
//				fprintf(fid, " ");
//		}
//
//		fprintf(fid, "\n");
//	}
//	fclose(fid);
//}
