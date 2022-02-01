/* Functions for RTPfMRI python module
 * Modified form mri_3dalign.c in afni source
 */

#include "librtp.h"

float Max(float *arr, int n) {
	int ii;
	float max;

	for (ii = 0; ii < n; ii++)
		max = MAX(max, arr[ii]);

	return max;
}

float Min(float *arr, int n) {
	int ii;
	float min;

	for (ii = 0; ii < n; ii++)
		min = MIN(min, arr[ii]);

	return min;
}

int rtp_align_setup(float *base_im, int nx, int ny, int nz, float dx, float dy,
		float dz, int ax1, int ax2, int ax3, int regmode, int nref,
		float *ref_ims, double *chol_fitim) {
	/*
	 * Modified from mri_3dalign_setup in mri_3dalign.c of afni source
	 */

	int ii, nxyz;
	float clip, delta;
	int nxy;
	float *wt, *pim, *mim, *dim, **refar;

	nxyz = nx * ny * nz;

	/* -- make weight map from the base  -- */
	wt = (float *) malloc(sizeof(float) * nxyz);
	memcpy(wt, base_im, sizeof(float) * nxyz);
	for (ii = 0; ii < nxyz; ii++)
		wt[ii] = fabs(wt[ii]);

	EDIT_blur_volume_3d(nx, ny, nz, dx, dy, dz, MRI_float, wt, 3.0 * dx,
			3.0 * dy, 3.0 * dz);

	clip = 0.025 * Max(wt, nxyz);
	for (ii = 0; ii < nxyz; ii++)
		if (wt[ii] < clip)
			wt[ii] = 0.0;

	nxy = nx * ny;
	xfade = (int) (0.05 * nx + 0.5);
	yfade = (int) (0.05 * ny + 0.5);
	zfade = (int) (0.05 * nz + 0.5);

	float *f = wt;
	int jj, kk, ff;
#define FF(i,j,k) f[(i)+(j)*nx+(k)*nxy]

	for (jj = 0; jj < ny; jj++)
		for (ii = 0; ii < nx; ii++)
			for (ff = 0; ff < zfade; ff++)
				FF(ii,jj,ff)= FF(ii,jj,nz-1-ff) = 0.0;

	for (kk = 0; kk < nz; kk++)
		for (jj = 0; jj < ny; jj++)
			for (ff = 0; ff < xfade; ff++)
				FF(ff,jj,kk)= FF(nx-1-ff,jj,kk) = 0.0;

	for (kk = 0; kk < nz; kk++)
		for (ii = 0; ii < nx; ii++)
			for (ff = 0; ff < yfade; ff++)
				FF(ii,ff,kk)= FF(ii,ny-1-ff,kk) = 0.0;

				/* -- make reference image array -- */
	refar = (float **) malloc(sizeof(float *) * nref);
	for (ii = 0; ii < nref; ii++)
		refar[ii] = ref_ims + ii * nxyz;

	/* refar[0]: base image --*/
	memcpy(refar[0], base_im, sizeof(float) * nxyz);

	/* -- gradient images -- */
	pim = (float *) malloc(sizeof(float) * nxyz);
	mim = (float *) malloc(sizeof(float) * nxyz);

	THD_rota_method(regmode);  // setup interpolation method

	/*-- refar[1]: d/d(th1) image --*/
	delta = 2.0 * delfac / (nx + ny + nz);
	memcpy(pim, base_im, sizeof(float) * nxyz);
	THD_rota_vol(nx, ny, nz, dx, dy, dz, pim, ax1, delta, ax2, 0.0, ax3, 0.0,
			dcode, 0.0, 0.0, 0.0);
	memcpy(mim, base_im, sizeof(float) * nxyz);
	THD_rota_vol(nx, ny, nz, dx, dy, dz, mim, ax1, -delta, ax2, 0.0, ax3, 0.0,
			dcode, 0.0, 0.0, 0.0);
	dim = refar[1];
	delta = sinit * 0.5 * DFAC / delta;
	for (ii = 0; ii < nxyz; ii++)
		dim[ii] = delta * (mim[ii] - pim[ii]);

	/*-- refar[2]: d/d(th2) image --*/
	delta = 2.0 * delfac / (nx + ny + nz);
	memcpy(pim, base_im, sizeof(float) * nxyz);
	THD_rota_vol(nx, ny, nz, dx, dy, dz, pim, ax1, 0.0, ax2, delta, ax3, 0.0,
			dcode, 0.0, 0.0, 0.0);
	memcpy(mim, base_im, sizeof(float) * nxyz);
	THD_rota_vol(nx, ny, nz, dx, dy, dz, mim, ax1, 0.0, ax2, -delta, ax3, 0.0,
			dcode, 0.0, 0.0, 0.0);
	delta = sinit * 0.5 * DFAC / delta;
	dim = refar[2];
	for (ii = 0; ii < nxyz; ii++)
		dim[ii] = delta * (mim[ii] - pim[ii]);

	/*-- refar[3]: d/d(th3) image --*/
	delta = 2.0 * delfac / (nx + ny + nz);
	memcpy(pim, base_im, sizeof(float) * nxyz);
	THD_rota_vol(nx, ny, nz, dx, dy, dz, pim, ax1, 0.0, ax2, 0.0, ax3, delta,
			dcode, 0.0, 0.0, 0.0);
	memcpy(mim, base_im, sizeof(float) * nxyz);
	THD_rota_vol(nx, ny, nz, dx, dy, dz, mim, ax1, 0.0, ax2, 0.0, ax3, -delta,
			dcode, 0.0, 0.0, 0.0);
	delta = sinit * 0.5 * DFAC / delta;
	dim = refar[3];
	for (ii = 0; ii < nxyz; ii++)
		dim[ii] = delta * (mim[ii] - pim[ii]);

	/*-- refar[4]: d/dx image --*/
	delta = delfac * dx;
	memcpy(pim, base_im, sizeof(float) * nxyz);
	THD_rota_vol(nx, ny, nz, dx, dy, dz, pim, ax1, 0.0, ax2, 0.0, ax3, 0.0,
			dcode, delta, 0.0, 0.0);
	memcpy(mim, base_im, sizeof(float) * nxyz);
	THD_rota_vol(nx, ny, nz, dx, dy, dz, mim, ax1, 0.0, ax2, 0.0, ax3, 0.0,
			dcode, -delta, 0.0, 0.0);
	delta = sinit * 0.5 / delta;
	dim = refar[4];
	for (ii = 0; ii < nxyz; ii++)
		dim[ii] = delta * (mim[ii] - pim[ii]);

	/*-- refar[5]: d/dy image --*/
	delta = delfac * dy;
	memcpy(pim, base_im, sizeof(float) * nxyz);
	THD_rota_vol(nx, ny, nz, dx, dy, dz, pim, ax1, 0.0, ax2, 0.0, ax3, 0.0,
			dcode, 0.0, delta, 0.0);
	memcpy(mim, base_im, sizeof(float) * nxyz);
	THD_rota_vol(nx, ny, nz, dx, dy, dz, mim, ax1, 0.0, ax2, 0.0, ax3, 0.0,
			dcode, 0.0, -delta, 0.0);
	delta = sinit * 0.5 / delta;
	dim = refar[5];
	for (ii = 0; ii < nxyz; ii++)
		dim[ii] = delta * (mim[ii] - pim[ii]);

	/*-- refar[6]: d/dz image --*/
	delta = delfac * dz;
	memcpy(pim, base_im, sizeof(float) * nxyz);
	THD_rota_vol(nx, ny, nz, dx, dy, dz, pim, ax1, 0.0, ax2, 0.0, ax3, 0.0,
			dcode, 0.0, 0.0, delta);
	memcpy(mim, base_im, sizeof(float) * nxyz);
	THD_rota_vol(nx, ny, nz, dx, dy, dz, mim, ax1, 0.0, ax2, 0.0, ax3, 0.0,
			dcode, 0.0, 0.0, -delta);
	delta = sinit * 0.5 / delta;
	dim = refar[6];
	for (ii = 0; ii < nxyz; ii++)
		dim[ii] = delta * (mim[ii] - pim[ii]);

	/*-- initialize linear least squares --*/
	double *cc;
	cc = startup_lsqfit(nxyz, wt, nref, refar);
	memcpy(chol_fitim, cc, sizeof(double) * nref * nref);

	free(wt);
	free(refar);
	free(pim);
	free(mim);
	free(cc);

	return 0;
}

int rtp_align_one(float *fim, int nx, int ny, int nz, float dx, float dy,
		float dz, float *fitim, double *chol_fitim, int nref, int ax1, int ax2,
		int ax3, float *init_motpar, int regmode, float *tim, float *motpar) {

	int ii, iter, good, nxyz;
	float dxt, dyt, dzt, ftop, fbot;
	float *fit, *dfit, **refar;
	double *rr;

	iter = 0;

	rr = (double*) malloc(nref * sizeof(double));

	THD_rota_method(regmode);

	/* convert displacement threshold from voxels to mm in each direction */
	dxt = fabs(dx) * dxy_thresh;
	dyt = fabs(dy) * dxy_thresh;
	dzt = fabs(dz) * dxy_thresh;

	/* set refar */
	nxyz = nx * ny * nz;
	refar = (float **) malloc(sizeof(float *) * nref);
	for (ii = 0; ii < nref; ii++)
		refar[ii] = fitim + ii * nxyz;

	/* set initial motion parameter */
	init_dth1 = init_motpar[0];
	init_dth2 = init_motpar[1];
	init_dth3 = init_motpar[2];
	init_dx = init_motpar[3];
	init_dy = init_motpar[4];
	init_dz = init_motpar[5];

	/*-- initial fit --*/
	fit = motpar;
	if ( NONZERO_INITVALS) {
		fit[0] = 1.0;
		fit[1] = init_dth1;
		fit[2] = init_dth2;
		fit[3] = init_dth3; /* degrees */
		fit[4] = init_dx;
		fit[5] = init_dy;
		fit[6] = init_dz; /* mm      */

		good = 1;
	} else {
		rtp_delayed_lsqfit(nxyz, fim, nref, refar, chol_fitim, rr);
		for( ii=0 ; ii < nref ; ii++ )
			fit[ii] = rr[ii];

		good = (10.0 * fabs(fit[4]) > dxt || 10.0 * fabs(fit[5]) > dyt
				|| 10.0 * fabs(fit[6]) > dzt || 10.0 * fabs(fit[1]) > phi_thresh
				|| 10.0 * fabs(fit[2]) > phi_thresh
				|| 10.0 * fabs(fit[3]) > phi_thresh);
	}

	/*-- iterate fit --*/
	dfit = (float *) malloc(nref*sizeof(float));
	while (good) {
		memcpy(tim, fim, nxyz*sizeof(float));
		THD_rota_vol(nx, ny, nz, dx, dy, dz, tim, ax1, fit[1] * DFAC, ax2,
				fit[2] * DFAC, ax3, fit[3] * DFAC, dcode, fit[4], fit[5],
				fit[6]);

		rtp_delayed_lsqfit(nxyz, tim, nref, refar, chol_fitim, rr); /* delta angle/shift */
		for( ii=0 ; ii < nref ; ii++ )
			dfit[ii] = rr[ii];

		/* accumulate angle/shift */
		for( ii=1; ii<nref; ii++)
			fit[ii] += dfit[ii];

		good = (++iter < max_iter)
				&& (fabs(dfit[4]) > dxt || fabs(dfit[5]) > dyt
						|| fabs(dfit[6]) > dzt || fabs(dfit[1]) > phi_thresh
						|| fabs(dfit[2]) > phi_thresh
						|| fabs(dfit[3]) > phi_thresh);

	} /* end while */
	free(dfit);
	free(rr);

	/*-- save final alignment parameters --*/
	// fit[1] *= DFAC; /* convert to radians */
	// fit[2] *= DFAC;
	// fit[3] *= DFAC;

	/*-- do the actual realignment --*/
	memcpy(tim, fim, nxyz * sizeof(float));
	THD_rota_vol(nx, ny, nz, dx, dy, dz, tim, ax1, fit[1] * DFAC, ax2,
			fit[2] * DFAC, ax3, fit[3] * DFAC, dcode, fit[4], fit[5], fit[6]);

	ftop = Max(fim, nxyz);
	fbot = Min(fim, nxyz);
	for (ii = 0; ii < nxyz; ii++) {
		if (tim[ii] < fbot)
			tim[ii] = fbot;
		else if (tim[ii] > ftop)
			tim[ii] = ftop;
	}

	free(refar);

	return 0;
}

void rtp_delayed_lsqfit(int veclen, float * data, int nref, float *ref[],
		double * cc, double *rr) {
	int ii, jj;
	float *alpha = NULL;
	register double sum;

	/*** form RHS vector into rr ***/
	for (ii = 0; ii < nref; ii++) {
		sum = 0.0;
		for (jj = 0; jj < veclen; jj++)
			sum += ref[ii][jj] * data[jj];
		rr[ii] = sum;
	}

	/*** forward solve ***/

	for (ii = 0; ii < nref; ii++) {
		sum = rr[ii];
		for (jj = 0; jj < ii; jj++)
			sum -= CC(ii, jj) * rr[jj];
		rr[ii] = sum / CC(ii, ii);
	}

	/*** backward solve ***/

	for (ii = nref - 1; ii >= 0; ii--) {
		sum = rr[ii];
		for (jj = ii + 1; jj < nref; jj++)
			sum -= CC(jj, ii) * rr[jj];
		rr[ii] = sum / CC(ii, ii);
	}
}

int rtp_smooth(float *im, int nx, int ny, int nz, float dx, float dy, float dz,
		unsigned short *mask, float fwhm) {

	int nrep = -1;
	float fx = -1.0f, fy = -1.0f, fz = -1.0f;

	mri_blur3D_getfac(fwhm, dx, dy, dz, &nrep, &fx, &fy, &fz);
	if (nrep < 0 || fx < 0.0f || fy < 0.0f || fz < 0.0f)
		return -1;

	// Below is a copy from mri_blur3D_inmask in mri_blur3d_variable.c in afni source

	int nxy, nxyz;
	float *qar;
	int ijk, ii, jj, kk, nn, nfloat_err = 0;
	register float vcc, vsub, vout, vx, vy, vz;

	nxy = nx * ny;
	nxyz = nxy * nz;

	vx = 2.0f * fx;
	if (nx < 2)
		vx = 0.0f;
	vy = 2.0f * fy;
	if (ny < 2)
		vy = 0.0f;
	vz = 2.0f * fz;
	if (nz < 2)
		vz = 0.0f;
	if (vx <= 0.0f && vy <= 0.0f && vz <= 0.0f)
		return -1;

#pragma omp critical (MALLOC)
	qar = (float *) calloc(sizeof(float), nxyz);

	for (nn = 0; nn < nrep; nn++) {
		for (ijk = kk = 0; kk < nz; kk++) {
			for (jj = 0; jj < ny; jj++) {
				for (ii = 0; ii < nx; ii++, ijk++) {
					if (!INMASK(ijk))
						continue;
					vout = vcc = im[ijk];
					if (vx > 0.0f) { /* distribute (diffuse) in the x-direction */
						if (ii - 1 >= 0 && INMASK(ijk - 1)) {
							vsub = vx * vcc;
							qar[ijk - 1] += vsub;
							vout -= vsub;
						}
						if (ii + 1 < nx && INMASK(ijk + 1)) {
							vsub = vx * vcc;
							qar[ijk + 1] += vsub;
							vout -= vsub;
						}
					}
					if (vy > 0.0f) { /* distribute (diffuse) in the y-direction */
						if (jj - 1 >= 0 && INMASK(ijk - nx)) {
							vsub = vy * vcc;
							qar[ijk - nx] += vsub;
							vout -= vsub;
						}
						if (jj + 1 < ny && INMASK(ijk + nx)) {
							vsub = vy * vcc;
							qar[ijk + nx] += vsub;
							vout -= vsub;
						}
					}
					if (vz >= 0.0f) { /* distribute (diffuse) in the z-direction */
						if (kk - 1 >= 0 && INMASK(ijk - nxy)) {
							vsub = vz * vcc;
							qar[ijk - nxy] += vsub;
							vout -= vsub;
						}
						if (kk + 1 < nz && INMASK(ijk + nxy)) {
							vsub = vz * vcc;
							qar[ijk + nxy] += vsub;
							vout -= vsub;
						}
					}

					qar[ijk] += vout; /* whatever wasn't diffused away from this voxel */
				}
			}
		}
		AAmemcpy(im, qar, sizeof(float) * nxyz);
		if (nn != nrep - 1) {
			AAmemset(qar, 0, sizeof(float) * nxyz);
		}
	}

#pragma omp critical (MALLOC)
	free((void * )qar);

	return 0;
}
