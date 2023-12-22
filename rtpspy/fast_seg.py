#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anatomical image segmentation for real-time fMRI processing using FastSurfer
(https://github.com/Deep-MI/FastSurfer).

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
import numpy as np
from pathlib import Path
import subprocess
import shlex
import shutil

import torch
import nibabel as nib

# Debug
if '__file__' not in locals():
    __file__ = 'this.py'

no_cuda = (not torch.cuda.is_available())


# %%
class FastSeg:
    aseg_mask_IDs = {'aseg': None,
                     'Brain': ['>0'],
                     'GM': ['>0', -2, -7, -41, -46, -192, -251, -252, -253,
                            -254, -255, -4, -5, -14, -15, -31, -43, -44, -63,
                            -72, -77],
                     'WM': [2, 41, 192, 251, 252, 253, 254, 255],
                     'wmparc': [2, 4, 43, 41, 192, 251, 252, 253, 254, 255],
                     'Vent': [4, 43],
                     'Vent_all': [4, 5, 14, 15, 43, 44, 72, 213, 221],
                     'Left-Thalamus-Proper': [10],
                     'Left-Caudate': [11],
                     'Left-Putamen': [12],
                     'Left-Pallidum': [13],
                     'Brain-Stem': [16],
                     'Left-Hippocampus': [17],
                     'Left-Amygdala': [18],
                     'Left-Accumbens-area': [26],
                     'Left-VentralDC': [28],
                     'Right-Thalamus-Proper': [49],
                     'Right-Caudate': [50],
                     'Right-Putamen': [51],
                     'Right-Pallidum': [52],
                     'Right-Hippocampus': [53],
                     'Right-Amygdala': [54],
                     'Right-Accumbens-area': [58],
                     'Right-VentralDC': [60],
                     'ctx-lh-caudalanteriorcingulate': [1002],
                     'ctx-lh-caudalmiddlefrontal': [1003],
                     'ctx-lh-corpuscallosum': [1004],
                     'ctx-lh-cuneus': [1005],
                     'ctx-lh-entorhinal': [1006],
                     'ctx-lh-fusiform': [1007],
                     'ctx-lh-inferiorparietal': [1008],
                     'ctx-lh-inferiortemporal': [1009],
                     'ctx-lh-isthmuscingulate': [1010],
                     'ctx-lh-lateraloccipital': [1011],
                     'ctx-lh-lateralorbitofrontal': [1012],
                     'ctx-lh-lingual': [1013],
                     'ctx-lh-medialorbitofrontal': [1014],
                     'ctx-lh-middletemporal': [1015],
                     'ctx-lh-parahippocampal': [1016],
                     'ctx-lh-paracentral': [1017],
                     'ctx-lh-parsopercularis': [1018],
                     'ctx-lh-parsorbitalis': [1019],
                     'ctx-lh-parstriangularis': [1020],
                     'ctx-lh-pericalcarine': [1021],
                     'ctx-lh-postcentral': [1022],
                     'ctx-lh-posteriorcingulate': [1023],
                     'ctx-lh-precentral': [1024],
                     'ctx-lh-precuneus': [1025],
                     'ctx-lh-rostralanteriorcingulate': [1026],
                     'ctx-lh-rostralmiddlefrontal': [1027],
                     'ctx-lh-superiorfrontal': [1028],
                     'ctx-lh-superiorparietal': [1029],
                     'ctx-lh-superiortemporal': [1030],
                     'ctx-lh-supramarginal': [1031],
                     'ctx-lh-frontalpole': [1032],
                     'ctx-lh-temporalpole': [1033],
                     'ctx-lh-transversetemporal': [1034],
                     'ctx-lh-insula': [1035],
                     'ctx-rh-unknown': [2000],
                     'ctx-rh-bankssts': [2001],
                     'ctx-rh-caudalanteriorcingulate': [2002],
                     'ctx-rh-caudalmiddlefrontal': [2003],
                     'ctx-rh-corpuscallosum': [2004],
                     'ctx-rh-cuneus': [2005],
                     'ctx-rh-entorhinal': [2006],
                     'ctx-rh-fusiform': [2007],
                     'ctx-rh-inferiorparietal': [2008],
                     'ctx-rh-inferiortemporal': [2009],
                     'ctx-rh-isthmuscingulate': [2010],
                     'ctx-rh-lateraloccipital': [2011],
                     'ctx-rh-lateralorbitofrontal': [2012],
                     'ctx-rh-lingual': [2013],
                     'ctx-rh-medialorbitofrontal': [2014],
                     'ctx-rh-middletemporal': [2015],
                     'ctx-rh-parahippocampal': [2016],
                     'ctx-rh-paracentral': [2017],
                     'ctx-rh-parsopercularis': [2018],
                     'ctx-rh-parsorbitalis': [2019],
                     'ctx-rh-parstriangularis': [2020],
                     'ctx-rh-pericalcarine': [2021],
                     'ctx-rh-postcentral': [2022],
                     'ctx-rh-posteriorcingulate': [2023],
                     'ctx-rh-precentral': [2024],
                     'ctx-rh-precuneus': [2025],
                     'ctx-rh-rostralanteriorcingulate': [2026],
                     'ctx-rh-rostralmiddlefrontal': [2027],
                     'ctx-rh-superiorfrontal': [2028],
                     'ctx-rh-superiorparietal': [2029],
                     'ctx-rh-superiortemporal': [2030],
                     'ctx-rh-supramarginal': [2031],
                     'ctx-rh-frontalpole': [2032],
                     'ctx-rh-temporalpole': [2033],
                     'ctx-rh-transversetemporal': [2034],
                     'ctx-rh-insula': [2035]}

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, fastsurfer_dir=None):
        if fastsurfer_dir is None:
            fastsurfer_dir = Path(__file__).absolute().parent / 'FastSurfer'

        self.fastsurfer_dir = Path(fastsurfer_dir)
        self.run_cmd = self.fastsurfer_dir / 'run_fastsurfer.sh'

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run(self, in_f, prefix=None, batch_size=1,
            segs=['aseg', 'Brain', 'WM', 'Vent'], overwrite=False):
        # --- Prepare files ---
        in_f, prefix = self.prep_files(in_f, prefix)

        # --- Run FastSurferCNN ---
        fsSeg_mgz = self.run_seg_only(in_f, prefix, batch_size)

        # --- Get segmentation ---
        out_fs = self.make_seg_images(fsSeg_mgz, prefix, segs)

        # --- Clean intermediate files ---
        if Path(prefix).is_dir():
            shutil.rmtree(prefix)

        if Path(in_f).stat().st_ino != Path(in_f).stat().st_ino and \
                in_f.is_file():
            in_f.unlink()

        out_fs_str = '\n  '.join([str(p) for p in out_fs])
        print(f"Output images: \n  {out_fs_str}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def prep_files(self, in_f, prefix=None):
        """ Convert BRIK to NIfTI
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
        exts = in_f.suffixes
        if exts[-1] == '.gz':
            ftype = exts[-2]
        else:
            ftype = exts[-1]

        # --- Set prefix ---
        if prefix is None:
            out_d = in_f.absolute().parent
            prefix = out_d / (in_f.name.replace(''.join(exts[-2:]), '')
                              + '_fastSeg')
        else:
            prefix = Path(prefix).absolute()

        # --- Convert BRIK to NIfTI ---
        if ftype in ('.HEAD', '.BRIK'):
            prefix = Path(
                str(prefix).replace('+orig', '').replace('+tlrc', ''))
            nii_f = prefix.absolute().parent / \
                (in_f.name.replace(''.join(exts), '') + '.nii.gz')
            cmd = f"3dAFNItoNIFTI -overwrite -prefix {nii_f} {in_f}"
            subprocess.check_call(cmd, shell=True, stderr=subprocess.PIPE)
            in_f = nii_f

        return (in_f, prefix)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run_seg_only(self, in_f, prefix, batch_size=1, seg_cereb=False):
        """
        run_FastSurferCNN
        Parameters
        ----------
        in_f : Path
            input file.
        prefix : Path
            output prefix.
        batch_size : int
            batch size for inference.
        seg_cereb : bool
            Run cerebellum segmentation

        Returns
        -------
        out_mgz : Path
            FastSurferCNN segmentation .mgz image.
        """

        # Run
        work_dir = Path(prefix).parent
        cmd = f"./{self.run_cmd.relative_to(self.fastsurfer_dir)}"
        cmd += f" --t1 {in_f} --sd {work_dir}"
        cmd += f" --sid {Path(prefix).name} --seg_only --no_biasfield"
        if not seg_cereb:
            cmd += " --no_cereb"
        cmd += f" --batch {batch_size}"
        if no_cuda:
            cmd += ' --no_cuda'

        # Spawn the process
        proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                cwd=self.fastsurfer_dir)

        fsSeg_mgz = Path(prefix) / 'mri' / 'aparc.DKTatlas+aseg.deep.mgz'

        return proc, fsSeg_mgz

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def make_seg_images(self, in_f, fsSeg_mgz, prefix,
                        segs=['aseg', 'Brain', 'WM', 'Vent'],
                        show_proc_progress=None):

        aseg_img = nib.load(fsSeg_mgz)
        header = aseg_img.header
        affine = aseg_img.affine
        aseg_V = np.asarray(aseg_img.dataobj)

        # Get space of input file for AFNI
        cmd = f"3dinfo -space {in_f}"
        img_space = subprocess.check_output(shlex.split(cmd)).decode()

        out_fs = []
        for seg_name in segs:
            out_f = str(prefix) + f"_{seg_name}.nii.gz"
            if seg_name == 'aseg':
                seg = aseg_V
            else:
                seg_idx = FastSeg.aseg_mask_IDs[seg_name]

                seg = np.zeros_like(aseg_V)
                for idx in seg_idx:
                    if type(idx) is str:
                        seg += eval(f"aseg_V {idx}")
                    elif idx < 0:
                        seg -= aseg_V == -idx
                    else:
                        seg += aseg_V == idx

            tmp_f = prefix.parent / f'rm_{seg_name}.nii.gz'
            seg_nii_img = nib.Nifti1Image(seg, affine, header)
            nib.save(seg_nii_img, str(tmp_f))

            if seg_name == 'Brain':
                cmd = f"3dmask_tool -overwrite -input {tmp_f}"
                cmd += f" -prefix {tmp_f}"
                cmd += " -frac 1.0 -dilate_inputs 5 -5 -fill_holes"
                proc = subprocess.Popen(
                    shlex.split(cmd), stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=prefix.parent)
                ret = show_proc_progress(proc)
                if ret != 0:
                    return None

            if seg_name == 'aseg':
                cmd = f"3dresample -overwrite -master {in_f} -input {tmp_f}"
                cmd += f" -prefix {out_f} -rmode NN"
                proc = subprocess.Popen(
                    shlex.split(cmd), stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=prefix.parent)
                ret = show_proc_progress(proc)
                if ret != 0:
                    return None
            else:
                cmd = f"3dfractionize -overwrite -template {in_f}"
                cmd += f" -input {tmp_f} -prefix {tmp_f} -clip 0.5"
                proc = subprocess.Popen(
                    shlex.split(cmd), stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=prefix.parent)
                ret = show_proc_progress(proc)
                if ret != 0:
                    return None

                if seg_name == 'Brain':
                    cmd = f"3dcalc -overwrite -prefix {out_f}"
                    cmd += f"  -a {tmp_f} -b {in_f} -expr 'step(a)*b'"
                else:
                    cmd = f"3dcalc -overwrite -prefix {out_f} -a {tmp_f}"
                    cmd += " -expr 'step(a)'"
                proc = subprocess.Popen(
                    shlex.split(cmd), stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=prefix.parent)
                ret = show_proc_progress(proc)
                if ret != 0:
                    return None

            cmd = f"3drefit -space {img_space} {out_f}"
            ret = show_proc_progress(proc)
            if ret != 0:
                return None

            if tmp_f.is_file():
                tmp_f.unlink()

            out_fs.append(out_f)

        return out_fs
