/*
 * rt_RetroTS.h
 *
 *  Created on: Nov 27, 2013
 *      Author: mmisaki
 */

#ifndef RT_RETROTS_H_
#define RT_RETROTS_H_

typedef struct {
	float VolTR; // TR of MRI acquisition in second
	float PhysFS; // Sampling frequency of physiological signal data
	float tshift;  // slice timing offset
	float zerophaseoffset;
	int RVTshifts[256];
	int RVTshiftslength;
	int RespCutoffFreq;
	int fcutoff;
	int AmpPhase;
	int CardCutoffFreq;
	char ResamKernel[256];
	int FIROrder;
	int as_windwidth;
	int as_percover;
	int as_fftwin;
} RetroTSOpt;
RetroTSOpt rtsOpt;

int rtp_retrots(RetroTSOpt *rtsOpt, double *Resp, double *ECG, int len,
		float *regOut);

#endif /* RT_RETROTS_H_ */
