#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 17:22:25 2021

@author: mmisaki@laureateinstitute.org
"""

# %% import ===================================================================
from pathlib import Path
import nibabel as nib
import ants
import argparse


# %% read_to_ANTs =============================================================
def read_to_ANTs(file):
    if '.nii' in Path(file).suffixes:
        ants_img = ants.image_read(str(file))
    else:
        img = nib.load(file)
        nii_img = nib.Nifti1Image(img.get_fdata(), img.affine)
        ants_img = ants.utils.convert_nibabel.from_nibabel(nii_img)

    return ants_img


# %% ants_registration ========================================================
def ants_registration(fix_f, move_f, outprefix, verbose=True):
    fixed = read_to_ANTs(fix_f)
    moving = read_to_ANTs(move_f)

    warp_params = ants.registration(
        fixed, moving, outprefix=outprefix, type_of_transform='SyN',
        reg_iterations=[100, 70, 50, 0], verbose=verbose)

    return warp_params


# %% ants_warp_resample =======================================================
def ants_warp_resample(fix_f, move_f, out_f, transformlist,
                       interpolator='linear', imagetype=0, verbose=True):
    fixed = read_to_ANTs(fix_f)
    moving = read_to_ANTs(move_f)

    warped = ants.apply_transforms(
        fixed, moving, transformlist, interpolator=interpolator,
        imagetype=imagetype, verbose=verbose)

    warped.to_filename(str(out_f))


# %% __main__ =================================================================
if __name__ == '__main__':
    # --- Get options ---
    parser = argparse.ArgumentParser()
    parser.add_argument('run', help='[registration|warp_resample]')
    parser.add_argument('-f', '--fixed', help='fixed image file')
    parser.add_argument('-m', '--moving', help='moving image file')
    parser.add_argument('-t', '--transforms', nargs='*',
                        help='transformation files')
    parser.add_argument('-o', '--out', help='output file (prefix)')
    parser.add_argument('-i', '--interpolation', help='resample interpolation')
    parser.add_argument('-v', '--verbose', action='store_true')

    opts = parser.parse_args()
    run = opts.run
    fix_f = opts.fixed
    move_f = opts.moving
    transforms = opts.transforms
    out_f = opts.out
    interpolation = opts.interpolation
    verbose = opts.verbose

    if run == 'registration':
        ants_registration(fix_f, move_f, out_f, verbose=verbose)

    elif run == 'warp_resample':
        ants_warp_resample(fix_f, move_f, out_f, transforms,
                           interpolator=interpolation, verbose=verbose)
