#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anatomical image segmentation for real-time fMRI processing using FastSurfer
(https://github.com/Deep-MI/FastSurfer).

@author: mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
import os
import sys
from pathlib import Path
import subprocess
import shlex
import shutil

import numpy as np
import torch
import ants

# Device for FastSurfer inference
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'


# %% FastSeg =================================================================
class FastSeg:
    """
    Segment an anatomical image with FastSurfer (segmentation only) and make
    brain, white matter, ventricle, and other masks from the segmentation.
    """

    # Segmentation labels of each mask (FreeSurfer aseg/DKT atlas IDs).
    # A positive ID adds the label, a negative ID removes it, and a string is
    # a condition on the label value (e.g., '>0': all labels). None (aseg)
    # means the label image itself.
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
        """
        Parameters
        ----------
        fastsurfer_dir : str or Path, optional
            FastSurfer directory. The default is the FastSurfer directory
            next to this file.
        """
        if fastsurfer_dir is None:
            fastsurfer_dir = Path(__file__).absolute().parent / 'FastSurfer'

        self.fastsurfer_dir = Path(fastsurfer_dir).absolute()
        self.run_cmd = self.fastsurfer_dir / 'run_fastsurfer.sh'

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run(self, in_f, bias_correction=False, prefix=None, batch_size=1,
            segs=['aseg', 'Brain', 'WM', 'Vent']):
        """
        Run all the steps: prep_files, run_seg_only, and make_seg_images.
        The output images are prefix + f'_{seg}.nii.gz' for each seg in
        segs.
        """
        src_f = Path(in_f).absolute()

        in_f, prefix = self.prep_files(src_f, bias_correction, prefix)
        fsSeg_mgz = self.run_seg_only(in_f, prefix, batch_size)
        out_fs = self.make_seg_images(in_f, fsSeg_mgz, prefix, segs)

        # --- Clean intermediate files ---
        # FastSurfer subject directory
        if Path(prefix).is_dir():
            shutil.rmtree(prefix)

        # Delete the intermediate input (NIfTI conversion or bias-corrected
        # image), but never the source file.
        if in_f.is_file() and not in_f.samefile(src_f):
            in_f.unlink()

        out_fs_str = '\n  '.join([str(p) for p in out_fs])
        print(f"Output images:\n  {out_fs_str}")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def _run_cmd(cmd, cwd, show_proc_progress=None, ETA=None, env=None):
        """
        Run a command. With show_proc_progress, the command is spawned and
        show_proc_progress(proc, ETA) waits for it and shows the progress.

        Returns
        -------
        bool
            True if the command succeeded.
        """
        if show_proc_progress is None:
            subprocess.check_call(cmd, cwd=cwd, env=env)
            return True

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, cwd=cwd, env=env)
        return show_proc_progress(proc, ETA) == 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def prep_files(self, in_f, bias_correction=False, prefix=None,
                   show_proc_progress=None, ETA=None):
        """
        Prepare the input file for FastSurfer: convert an AFNI BRIK file to
        NIfTI and optionally apply N4 bias field correction.

        Parameters
        ----------
        in_f : Path
            Input anatomical image file.
        bias_correction : bool, optional
            Apply N4 bias field correction. The default is False.
        prefix : str or Path, optional
            Output prefix. The default is in_f without extensions +
            '_fastSeg', in the directory of in_f.
        show_proc_progress : function, optional
            Function to wait for a spawned process while showing its
            progress; called as show_proc_progress(proc, ETA) and returns 0
            on success. If None, the command is run with check_call.
        ETA : float, optional
            Expected time (seconds) passed to show_proc_progress.

        Returns
        -------
        in_f : Path
            Input file for FastSurfer. (None, None) is returned on failure.
        prefix : Path
            Output prefix.
        """

        # Image space (e.g., ORIG, TLRC) to set on the output files
        cmd = f"3dinfo -space {in_f}"
        img_space = subprocess.check_output(
            shlex.split(cmd)).decode().rstrip()

        exts = in_f.suffixes
        if exts[-1] == '.gz':
            ftype = exts[-2]
        else:
            ftype = exts[-1]

        # --- Set the prefix ---
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

        if bias_correction:
            in_img = ants.image_read(str(in_f))
            in_img_bc = in_img.n4_bias_field_correction()
            in_f = in_f.parent / in_f.name.replace('.nii', '_N4BCOR.nii')
            ants.image_write(in_img_bc, str(in_f))

            # ANTs does not keep the AFNI space; set it again.
            cmd = f"3drefit -space {img_space} {in_f}"
            if not self._run_cmd(shlex.split(cmd), prefix.parent,
                                 show_proc_progress, ETA):
                return (None, None)

        return (in_f, prefix)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run_seg_only(self, in_f, prefix, batch_size=1, seg_cereb=False,
                     show_proc_progress=None, ETA=None):
        """
        Run the FastSurfer segmentation (--seg_only).

        Parameters
        ----------
        in_f : Path
            Input file.
        prefix : Path
            Output prefix. The FastSurfer subject directory is made at this
            path.
        batch_size : int
            Batch size for inference.
        seg_cereb : bool
            Run the cerebellum segmentation.
        show_proc_progress : function, optional
            See prep_files.
        ETA : float, optional
            Expected time (seconds) passed to show_proc_progress.

        Returns
        -------
        fsSeg_mgz : Path
            FastSurfer segmentation .mgz image. None on failure.
        """

        # FastSurfer requires absolute paths for --t1 and --sd.
        in_f = Path(in_f).absolute()
        prefix = Path(prefix).absolute()
        work_dir = prefix.parent
        cmd = [str(self.run_cmd),
               '--t1', str(in_f), '--sd', str(work_dir),
               '--sid', prefix.name, '--seg_only', '--no_biasfield',
               '--no_hypothal', '--no_cc',
               '--batch', str(batch_size), '--device', DEVICE,
               # Run FastSurfer with the same Python as RTPSpy
               '--py', sys.executable]
        if not seg_cereb:
            cmd.append('--no_cereb')
        env = os.environ.copy()
        env['FASTSURFER_HOME'] = str(self.fastsurfer_dir)
        env['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

        if not self._run_cmd(cmd, self.fastsurfer_dir, show_proc_progress,
                             ETA, env=env):
            return None

        fsSeg_mgz = prefix / 'mri' / 'aparc.DKTatlas+aseg.deep.mgz'

        return fsSeg_mgz

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def make_seg_images(self, in_f, fsSeg_mgz, prefix,
                        segs=['aseg', 'Brain', 'WM', 'Vent'],
                        show_proc_progress=None, ETA=None):
        """
        Make mask images in the in_f space from the FastSurfer segmentation.

        Parameters
        ----------
        in_f : Path
            Input file of FastSurfer.
        fsSeg_mgz : Path
            FastSurfer segmentation image (output of run_seg_only).
        prefix : Path
            Output prefix. The output files are prefix + f'_{seg}.nii.gz'.
        segs : list of str, optional
            Mask names (keys of FastSeg.aseg_mask_IDs). 'Brain' is the
            skull-stripped in_f image.
        show_proc_progress : function, optional
            See prep_files.
        ETA : float, optional
            Expected time (seconds) passed to show_proc_progress.

        Returns
        -------
        out_fs : list of Path
            Output files. None on failure.
        """

        # FastSurfer conforms the input to orig.mgz. Register orig.mgz to
        # in_f to bring the segmentation back to the in_f space.
        orig_mgz = fsSeg_mgz.parent / 'orig.mgz'

        orig_img = ants.image_read(str(orig_mgz))
        in_img = ants.image_read(str(in_f))
        reg = ants.registration(in_img, orig_img, 'Translation')

        aseg_img = ants.image_read(str(fsSeg_mgz))
        aseg_V = aseg_img.numpy()

        # Image space (e.g., ORIG, TLRC) to set on the output files
        cmd = f"3dinfo -space {in_f}"
        img_space = subprocess.check_output(
            shlex.split(cmd)).decode().rstrip()

        out_fs = []
        for seg_name in segs:
            out_f = str(prefix) + f"_{seg_name}.nii.gz"
            if seg_name == 'aseg':
                out_img = ants.apply_transforms(
                    in_img, aseg_img, reg['fwdtransforms'],
                    'nearestNeighbor'
                    )

            elif seg_name == 'Brain':
                # Skull-stripped in_f
                mask_img = ants.image_read(str(fsSeg_mgz.parent / 'mask.mgz'))
                mask_in = ants.apply_transforms(
                    in_img, mask_img, reg['fwdtransforms'], 'nearestNeighbor'
                    )
                out_img = in_img * mask_in

            else:
                seg_idx = FastSeg.aseg_mask_IDs[seg_name]
                seg_V = np.zeros_like(aseg_V)
                for idx in seg_idx:
                    if type(idx) is str:
                        # Condition on the label value, e.g., '>0'
                        seg_V += eval(f"aseg_V {idx}")
                    elif idx < 0:
                        seg_V -= aseg_V == -idx
                    else:
                        seg_V += aseg_V == idx
                seg_img = aseg_img.new_image_like(seg_V)
                out_img = ants.apply_transforms(
                    in_img, seg_img,
                    reg['fwdtransforms'], 'nearestNeighbor'
                    )

            ants.image_write(out_img, str(out_f))
            out_fs.append(Path(out_f))

            cmd = f"3drefit -space {img_space} {out_f}"
            if not self._run_cmd(shlex.split(cmd), prefix.parent,
                                 show_proc_progress, ETA):
                return None

        return out_fs
