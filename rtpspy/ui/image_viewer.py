#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image viewer to check the masks: three orthogonal slices (sagittal,
coronal, axial) of a base image with an overlay.

The overlay is resampled to the grid of the base image by the world (scanner)
coordinates, so images in different spaces (e.g., a mask in the function
space on the aligned anatomy) are shown in register. A mask (few integer
values) is shown in one color; a continuous image (e.g., a function image)
with a color map.

One window is reused: set_images() changes the images shown. The Base and
Overlay menus choose the images from a list of named images (set_choices;
e.g., the function, anatomy and mask images of RtpApp) or from a file.

The views are in the neurological convention (the subject's left on the
left of the screen; letters at the edges show the orientation). Click on a
view to move the cursor; the mouse wheel moves the slice of the view. The
slider sets the opacity of the overlay.

mmisaki@laureateinstitute.org
"""

# %% import ===================================================================
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from PyQt6 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap

from .. import gui_theme

# Axes of the canonical (RAS) volume shown in each view: (slice axis,
# displayed (row, column) axes). Neurological convention: the subject's
# left is on the left of the screen.
_VIEWS = (("Sagittal", 0, (2, 1)), ("Coronal", 1, (2, 0)),
          ("Axial", 2, (1, 0)))
# Orientation letters of the (left, right, bottom, top) edges of each view
_ORIENT = (("P", "A", "I", "S"), ("L", "R", "I", "S"), ("L", "R", "P", "A"))

_FILE_FILTER = "Images (*.nii *.nii.gz *.HEAD);;All files (*)"


# %% load_image ===============================================================
def _load_3d(fname):
    """Load an image (NIfTI or AFNI BRIK) as a 3D image in the closest
    canonical (RAS) orientation. The first volume of a 4D image is used."""
    img = nib.load(str(fname))
    if len(img.shape) > 3:
        data = np.asanyarray(img.dataobj[..., 0])
        img = nib.Nifti1Image(data, img.affine)
    elif not isinstance(img, nib.Nifti1Image):
        img = nib.Nifti1Image(np.asanyarray(img.dataobj), img.affine)
    return nib.as_closest_canonical(img)


def _is_mask(data):
    """True if data looks like a mask or a label image."""
    vals = np.unique(data[np.isfinite(data)])
    return len(vals) <= 64 and np.allclose(vals, np.round(vals))


# %% ImageViewer ==============================================================
class ImageViewer(QtWidgets.QMainWindow):
    """Window showing a base image with an overlay in three views."""

    def __init__(self, base_f, ovl_f=None, title=None, choices=None,
                 start_dir=None, parent=None):
        """
        Parameters
        ----------
        base_f, ovl_f : str or Path
            Base and overlay (optional) images.
        title : str, optional
            Window title.
        choices : dict, optional
            {name: file} of the images in the Base and Overlay menus.
        start_dir : str or Path, optional
            Folder of the file dialogs ('Other file...').
        """
        super().__init__(parent)
        self._choices = {}
        self._start_dir = str(start_dir) if start_dir else None
        self._base_f = None
        self._ovl_f = None
        self._base = None
        self._ovl = None
        self._alpha = 0.5

        # --- Widgets ---
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)

        self._name_lbl = QtWidgets.QLabel()
        self._name_lbl.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        vbox.addWidget(self._name_lbl)

        self.canvas = FigureCanvas(Figure(figsize=(9, 3.4)))
        vbox.addWidget(self.canvas)
        fig = self.canvas.figure
        fig.set_layout_engine("constrained", w_pad=0.02, h_pad=0.02)
        self._axes = fig.subplots(1, 3)

        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        self._pos_lbl = QtWidgets.QLabel()
        hbox.addWidget(self._pos_lbl, 1)
        self._alpha_lbl = QtWidgets.QLabel("Overlay opacity")
        hbox.addWidget(self._alpha_lbl)
        self._alpha_sld = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._alpha_sld.setRange(0, 100)
        self._alpha_sld.setValue(int(self._alpha * 100))
        self._alpha_sld.setFixedWidth(160)
        self._alpha_sld.valueChanged.connect(self._set_alpha)
        hbox.addWidget(self._alpha_sld)

        # --- Menus ---
        self._base_menu = self.menuBar().addMenu("&Base")
        self._ovl_menu = self.menuBar().addMenu("&Overlay")
        self._base_group = QtGui.QActionGroup(self)
        self._ovl_group = QtGui.QActionGroup(self)

        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

        self.set_choices(choices)
        self.set_images(base_f, ovl_f, title)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_choices(self, choices, start_dir=None):
        """Set the images of the Base and Overlay menus ({name: file})."""
        if choices is not None:
            self._choices = {k: str(v) for k, v in choices.items() if v}
        if start_dir:
            self._start_dir = str(start_dir)
        self._make_menus()

    def _make_menus(self):
        """Make the Base and Overlay menus (only from set_choices: an
        action must not be deleted in its own triggered handler)."""
        self._menu_files = {"base": [], "ovl": []}
        for menu, group, which in (
            (self._base_menu, self._base_group, "base"),
            (self._ovl_menu, self._ovl_group, "ovl"),
        ):
            menu.clear()
            for act in group.actions():
                group.removeAction(act)
            if which == "ovl":
                act = menu.addAction("None")
                act.setCheckable(True)
                group.addAction(act)
                self._menu_files[which].append((act, None))
                act.triggered.connect(lambda _=False: self._choose_ovl(None))
                menu.addSeparator()
            for name, fname in self._choices.items():
                act = menu.addAction(name)
                act.setCheckable(True)
                act.setToolTip(fname)
                act.setEnabled(Path(fname).is_file())
                group.addAction(act)
                self._menu_files[which].append((act, fname))
                if which == "base":
                    act.triggered.connect(
                        lambda _=False, f=fname: self._choose_base(f))
                else:
                    act.triggered.connect(
                        lambda _=False, f=fname: self._choose_ovl(f))
            menu.setToolTipsVisible(True)
            if self._choices:
                menu.addSeparator()
            other = menu.addAction("Other file...")
            other.triggered.connect(
                lambda _=False, w=which: self._choose_file(w))
        self._update_menus()

    def _update_menus(self):
        """Check the menu items of the images shown."""
        if not hasattr(self, "_menu_files"):
            return
        for which, current in (("base", self._base_f), ("ovl", self._ovl_f)):
            for act, fname in self._menu_files[which]:
                if fname is None:
                    act.setChecked(current is None)
                else:
                    act.setChecked(current is not None and
                                   Path(fname) == Path(current))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _choose_base(self, fname):
        self._set_images_safe(fname, self._ovl_f)

    def _choose_ovl(self, fname):
        self._set_images_safe(self._base_f, fname)

    def _choose_file(self, which):
        start = self._start_dir or str(Path(self._base_f).parent)
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Base image" if which == "base" else "Overlay image",
            start, _FILE_FILTER)
        if not fname:
            self._update_menus()  # Restore the checked item
            return
        self._start_dir = str(Path(fname).parent)
        if which == "base":
            self._choose_base(fname)
        else:
            self._choose_ovl(fname)

    def _set_images_safe(self, base_f, ovl_f):
        """set_images with an error message (menus)."""
        try:
            self.set_images(base_f, ovl_f)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Image viewer", f"Failed to show the image:\n{e}")
            self._update_menus()

    def _name_of(self, fname):
        """Menu name of an image file, or the file name."""
        for name, f in self._choices.items():
            if Path(f) == Path(fname):
                return name
        return Path(fname).name

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_images(self, base_f, ovl_f=None, title=None):
        """Show base_f with the overlay ovl_f (None: no overlay). Raises an
        exception if an image cannot be read (the images shown are kept)."""
        c = gui_theme.COLORS
        base = _load_3d(base_f)
        base_data = base.get_fdata(dtype=np.float32)
        ovl_data = None
        ovl_mask = False
        if ovl_f is not None:
            ovl = _load_3d(ovl_f)
            ovl_mask = _is_mask(np.asanyarray(ovl.dataobj))
            ovl_r = resample_from_to(ovl, base, order=0 if ovl_mask else 1)
            ovl_data = ovl_r.get_fdata(dtype=np.float32)

        # The cursor: the center of the overlay (mask) when the overlay is
        # new; the same voxel when only the base image of the same grid
        # changed; otherwise the volume center.
        same_grid = self._base is not None and \
            self._base.shape == base_data.shape
        new_ovl = ovl_f is not None and (
            self._ovl_f is None or Path(ovl_f) != Path(self._ovl_f))
        if ovl_data is not None and np.any(ovl_data > 0) and \
                (new_ovl or not same_grid):
            cur = [int(round(v)) for v in
                   np.mean(np.argwhere(ovl_data > 0), axis=0)]
        elif same_grid:
            cur = list(self._cur)
        else:
            cur = [s // 2 for s in base_data.shape]

        self._base_f = str(base_f)
        self._ovl_f = None if ovl_f is None else str(ovl_f)
        self._base = base_data
        self._zooms = base.header.get_zooms()[:3]
        pos = self._base[self._base > 0]
        self._clim = tuple(np.percentile(pos, (1, 99.5))) if len(pos) \
            else (0, 1)
        self._ovl = ovl_data
        self._ovl_mask = ovl_mask
        self._cur = cur

        # Overlay color map: one color for a mask, a color map otherwise
        if self._ovl_mask:
            self._ovl_cmap = ListedColormap([c["red"]])
            self._ovl_clim = (0.5, 1.5)
        else:
            self._ovl_cmap = "hot"
            v = self._ovl[self._ovl > 0] if self._ovl is not None else []
            self._ovl_clim = tuple(np.percentile(v, (1, 99.5))) \
                if len(v) else (0, 1)

        # Labels and menus
        if title is None:
            title = self._name_of(base_f) if ovl_f is None else \
                f"{self._name_of(ovl_f)} on {self._name_of(base_f)}"
        self.setWindowTitle(f"Image viewer: {title}")
        names = Path(base_f).name
        if ovl_f is not None:
            names = f"{Path(ovl_f).name}  on  {Path(base_f).name}"
        self._name_lbl.setText(names)
        self._name_lbl.setToolTip(f"Base: {base_f}\nOverlay: {ovl_f}")
        has_ovl = self._ovl is not None
        self._alpha_lbl.setEnabled(has_ovl)
        self._alpha_sld.setEnabled(has_ovl)
        self._update_menus()

        self._init_views()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _slice(self, vol, view):
        """2D slice of vol at the cursor for view (rows: superior or
        anterior up)."""
        _, ax_s, (ax_r, ax_c) = _VIEWS[view]
        idx = [slice(None)] * 3
        idx[ax_s] = self._cur[ax_s]
        sl = vol[tuple(idx)]
        # The remaining axes are in increasing order; make (row, col)
        rem = [a for a in range(3) if a != ax_s]
        return sl if rem.index(ax_r) == 0 else sl.T

    def _aspect(self, view):
        _, _, (ax_r, ax_c) = _VIEWS[view]
        return self._zooms[ax_r] / self._zooms[ax_c]

    def _init_views(self):
        c = gui_theme.COLORS
        self._im_base, self._im_ovl, self._cross = [], [], []
        for vi, ax in enumerate(self._axes):
            ax.clear()
            ax.set_title(_VIEWS[vi][0], fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("black")
            ib = ax.imshow(self._slice(self._base, vi), cmap="gray",
                           origin="lower", vmin=self._clim[0],
                           vmax=self._clim[1], aspect=self._aspect(vi),
                           interpolation="nearest")
            io = None
            if self._ovl is not None:
                io = ax.imshow(self._ovl_slice(vi), cmap=self._ovl_cmap,
                               origin="lower", vmin=self._ovl_clim[0],
                               vmax=self._ovl_clim[1], alpha=self._alpha,
                               aspect=self._aspect(vi),
                               interpolation="nearest")
            for (x, y), txt in zip(((0.02, 0.5), (0.98, 0.5), (0.5, 0.02),
                                    (0.5, 0.98)), _ORIENT[vi]):
                ax.text(x, y, txt, transform=ax.transAxes, color=c["yellow"],
                        fontsize=9, fontweight="bold", ha="center",
                        va="center")
            hl = ax.axhline(0, color=c["cyan"], lw=0.6)
            vl = ax.axvline(0, color=c["cyan"], lw=0.6)
            self._im_base.append(ib)
            self._im_ovl.append(io)
            self._cross.append((hl, vl))
        self._update()

    def _ovl_slice(self, view):
        sl = self._slice(self._ovl, view)
        return np.ma.masked_where(~(sl > 0), sl)

    def _update(self):
        for vi in range(3):
            self._im_base[vi].set_data(self._slice(self._base, vi))
            if self._im_ovl[vi] is not None:
                self._im_ovl[vi].set_data(self._ovl_slice(vi))
                self._im_ovl[vi].set_alpha(self._alpha)
            _, _, (ax_r, ax_c) = _VIEWS[vi]
            hl, vl = self._cross[vi]
            hl.set_ydata([self._cur[ax_r]] * 2)
            vl.set_xdata([self._cur[ax_c]] * 2)
        txt = f"voxel (i, j, k) = {tuple(self._cur)}"
        if self._ovl is not None:
            txt += f"   overlay = {self._ovl[tuple(self._cur)]:.3g}"
        self._pos_lbl.setText(txt)
        self.canvas.draw_idle()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_alpha(self, val):
        self._alpha = val / 100
        self._update()

    def _view_of(self, event):
        for vi, ax in enumerate(self._axes):
            if event.inaxes is ax:
                return vi
        return None

    def _on_click(self, event):
        vi = self._view_of(event)
        if vi is None or event.xdata is None:
            return
        _, _, (ax_r, ax_c) = _VIEWS[vi]
        self._cur[ax_c] = int(np.clip(round(event.xdata), 0,
                                      self._base.shape[ax_c] - 1))
        self._cur[ax_r] = int(np.clip(round(event.ydata), 0,
                                      self._base.shape[ax_r] - 1))
        self._update()

    def _on_scroll(self, event):
        vi = self._view_of(event)
        if vi is None:
            return
        ax_s = _VIEWS[vi][1]
        step = 1 if event.button == "up" else -1
        self._cur[ax_s] = int(np.clip(self._cur[ax_s] + step, 0,
                                      self._base.shape[ax_s] - 1))
        self._update()
