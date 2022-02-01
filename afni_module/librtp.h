/*
 * librtp.h
 *
 *  Created on: Jul 31, 2018
 *      Author: mmisaki
 */

#ifndef LIBRTP_H_

#include "mrilib.h"
// #include <mkl.h>
// #include <omp.h>

/* --- volreg parameters --- */
#define MAX_ITER     5
#define DXY_THRESH   0.07         /* pixels */
#define PHI_THRESH   0.21         /* degrees */
#define DFAC         (PI/180.0)

static float dxy_thresh = DXY_THRESH, phi_thresh = PHI_THRESH, delfac = 0.7;
static float init_dth1 = 0.0, init_dth2 = 0.0, init_dth3 = 0.0;
static float init_dx = 0.0, init_dy = 0.0, init_dz = 0.0;
static int max_iter = MAX_ITER;
static int dcode = -1;

static int xfade, yfade, zfade;

static float sinit = 1.0;

#define NONZERO_INITVALS                                        \
 ( init_dth1 != 0.0 || init_dth2 != 0.0 || init_dth3 != 0.0 ||  \
   init_dx   != 0.0 || init_dy   != 0.0 || init_dz   != 0.0   )

#ifndef CC
#define CC(i,j) cc[(i)+(j)*nref]
#endif

/* --- smooth parameters --- */
#undef  INMASK
#define INMASK(i) (mask == NULL || mask[i] != 0)

/* -- prototype -- */

int rtp_align_setup(float *base_im, int nx, int ny, int nz, float dx, float dy,
		float dz, int ax1, int ax2, int ax3, int regmode, int nref,
		float *ref_ims, double *chol_fitim);

int rtp_align_one(float *fim, int nx, int ny, int nz, float dx, float dy,
		float dz, float *fitim, double *chol_fitim, int nref, int ax1, int ax2,
		int ax3, float *init_motpar, int regmode, float *tim, float *motpar);

void rtp_delayed_lsqfit(int veclen, float * data, int nref, float *ref[],
		double * cc, double *rr);

int rtp_smooth(float *im, int nx, int ny, int nz, float dx, float dy, float dz,
		unsigned short *mask, float fwhm);

#define LIBRTP_H_
#endif /* LIBRTP_H_ */
