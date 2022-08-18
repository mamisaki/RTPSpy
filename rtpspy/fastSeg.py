#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anatomical image segmentation for real-time fMRI processing using FastSurfer
(https://github.com/Deep-MI/FastSurfer).

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
import numpy as np
import argparse
from pathlib import Path
import subprocess
import shlex
from platform import uname
import shutil

import torch
import nibabel as nib

# Debug
if '__file__' not in locals():
    __file__ = 'this.py'

no_cuda = (not torch.cuda.is_available())

aseg_mask_IDs = {'Brain': ['>0'],
                 'WM': [2, 41, 192, 251, 252, 253, 254, 255],
                 'Vent': [4, 43]}


# %% prep_files ===============================================================
def prep_files(opts):
    """
    Parameters
    ----------
    opts : argparse,Namespace
        opts.in_f : input file name
        opts.prefix : output prefix

    Returns
    -------
    in_f : Path
        input file path for FastSurferCNN
    prefix : Path
        output file prefix
    """

    # --- Get input file type ---
    in_f = Path(opts.in_f).resolve()
    exts = in_f.suffixes
    if exts[-1] == '.gz':
        ftype = exts[-2]
    else:
        ftype = exts[-1]

    # --- Set prefix ---
    prefix = opts.prefix
    if prefix is None:
        out_d = in_f.absolute().parent
        prefix = out_d / (in_f.name.replace(''.join(exts[-2:]), '')
                          + '_fastSeg')
    else:
        prefix = Path(opts.prefix).absolute()

    # --- Convert BRIK to NIfTI ---
    if ftype in ('.HEAD', '.BRIK'):
        prefix = Path(str(prefix).replace('+orig', '').replace('+tlrc', ''))
        nii_f = prefix.absolute().parent / \
            (in_f.name.replace(''.join(exts), '') + '.nii.gz')
        cmd = f"3dAFNItoNIFTI -overwrite -prefix {nii_f} {in_f}"
        subprocess.check_call(cmd, shell=True, stderr=subprocess.PIPE)
        in_f = nii_f

    return (in_f, prefix)


# %% run_FastSurferCNN ========================================================
def run_FastSurferCNN(eval_cmd, fastsurfer_d, in_f, prefix, batch_size):
    """
    Parameters
    ----------
    eval_cmd : Path
        path to FastSurferCNN eval.py command.
    fastsurfer_d : Path
        path to FastSurfer install directory.
    in_f : Path
        input file.
    prefix : Path
        output prefix.
    batch_size : int
        batch size for inference.

    Returns
    -------
    out_mgz : Path
        FastSurferCNN segmentation .mgz image.
    """

    # Set output filename
    fsSeg_mgz = str(prefix) + '.mgz'

    # Run
    if 'Microsoft' in uname().release:
        FastSurferDir = Path('/mnt/c/FastSurfer')
        in_f_tmp = FastSurferDir / in_f.name
        shutil.copy(in_f, in_f_tmp)
        out_f_tmp = FastSurferDir / Path(fsSeg_mgz).name

        cmd = f"cmd.exe /c run_fastsurfer.bat ./{in_f_tmp.name}"
        cmd += f" ./{out_f_tmp.name}"
        subprocess.check_call(cmd, shell=True, cwd=FastSurferDir)
        if in_f_tmp.is_file():
            in_f_tmp.unlink()

        if out_f_tmp.is_file():
            shutil.copy(out_f_tmp, fsSeg_mgz)
            out_f_tmp.unlink()

    else:
        work_dir = Path(fsSeg_mgz).parent
        cmd = f"python {eval_cmd.relative_to(fastsurfer_d)}"
        cmd += f" --in_name {in_f} --conf_name {work_dir / 'orig.mgz'}"
        cmd += f" --out_name {fsSeg_mgz} --simple_run"
        cmd += f" --batch_size {batch_size}"
        if no_cuda:
            cmd += ' --no_cuda'

        subprocess.check_call(shlex.split(cmd), cwd=fastsurfer_d)

    return fsSeg_mgz


# %% make_seg_images ==========================================================
def make_seg_images(fsSeg_mgz, prefix, aseg_mask_IDs=aseg_mask_IDs):

    aseg_img = nib.load(fsSeg_mgz)
    header = aseg_img.header
    affine = aseg_img.affine
    aseg_V = np.asarray(aseg_img.dataobj)

    # Save aseg as nii.gz
    aseg_nii_f = prefix.parent / (prefix.name + '_aseg.nii.gz')
    simg = nib.Nifti1Image(aseg_V, affine)
    nib.save(simg, aseg_nii_f)
    subprocess.check_call(
        shlex.split(f"3drefit -space ORIG -view orig {aseg_nii_f}"))

    out_fs = []
    for seg_name, seg_idx in aseg_mask_IDs.items():
        out_f = str(prefix) + f"_{seg_name}.nii.gz"

        seg = np.zeros_like(aseg_V)
        for idx in seg_idx:
            if type(idx) == str:
                seg += eval(f"aseg_V {idx}")
            else:
                seg += aseg_V == idx

        tmp_f = prefix.parent / f'rm_{seg_name}.nii.gz'
        seg_nii_img = nib.Nifti1Image(seg, affine, header)
        nib.save(seg_nii_img, str(tmp_f))

        cmd = ''
        if seg_name == 'Brain':
            cmd += f"3dmask_tool -overwrite -input {tmp_f}"
            cmd += f" -prefix {tmp_f}"
            cmd += " -frac 1.0 -dilate_inputs 5 -5 -fill_holes &&"

        cmd += f"3dfractionize -overwrite -template {in_f} -input {tmp_f}"
        cmd += f" -prefix {tmp_f} -clip 0.5 &&"

        if seg_name == 'Brain':
            cmd += f"3dcalc -overwrite -prefix {out_f} -a {tmp_f} -b {in_f}"
            cmd += " -expr 'step(a)*b'"
        else:
            cmd += f"3dcalc -overwrite -prefix {out_f} -a {tmp_f}"
            cmd += " -expr 'step(a)'"

        subprocess.check_call(cmd, shell=True)
        if tmp_f.is_file():
            tmp_f.unlink()

        out_fs.append(out_f)

    return out_fs


# %% __main__ =================================================================
if __name__ == "__main__":
    # --- Get options ---
    parser = argparse.ArgumentParser()
    parser.add_argument('in_f', help='input file')
    parser.add_argument('-o', '--out', dest='prefix', help='output prefix')
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size for inference. Default: 8")
    parser.add_argument('--fastsurfer_dir', help="FastSurfer directory")
    opts = parser.parse_args()

    # --- Check FastSurfer installation ---
    fastsurfer_dir = opts.fastsurfer_dir
    if fastsurfer_dir is None:
        fastsurfer_dir = Path(__file__).absolute().parent / 'FastSurfer'

    fastsurfer_dir = Path(fastsurfer_dir)
    assert fastsurfer_dir.is_dir(), f"Not found {fastsurfer_dir} directory.\n"

    eval_cmd = fastsurfer_dir / 'FastSurferCNN' / 'eval.py'
    assert eval_cmd.is_file(), f"Not found {eval_cmd}\n"

    # --- Prepare files ---
    in_f, prefix = prep_files(opts)

    # --- Run FastSurferCNN ---
    if no_cuda and opts.batch_size == 8:
        batch_size = 32
    else:
        batch_size = opts.batch_size
    fsSeg_mgz = run_FastSurferCNN(eval_cmd, fastsurfer_dir, in_f, prefix,
                                  batch_size)

    # --- Get Brain, WM, Vent masks ---
    out_fs = make_seg_images(fsSeg_mgz, prefix, aseg_mask_IDs)

    # --- Clean intermediate files ---
    if Path(fsSeg_mgz).is_file():
        cmd = f""
        Path(fsSeg_mgz).unlink()

    work_dir = Path(fsSeg_mgz).parent
    for mgz in work_dir.glob('*.mgz'):
        mgz.unlink()

    if (work_dir / 'orig').is_dir():
        shutil.rmtree((work_dir / 'orig'))

    if Path(opts.in_f).stat().st_ino != Path(in_f).stat().st_ino and \
            in_f.is_file():
        in_f.unlink()

    out_fs_str = '\n  '.join([str(p) for p in out_fs])
    print(f"Output images: \n  {out_fs_str}")
