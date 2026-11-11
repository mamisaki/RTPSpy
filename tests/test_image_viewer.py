"""Image viewer to check the masks (rtpspy.ui.image_viewer)."""
import numpy as np
import nibabel as nib
import pytest

QtWidgets = pytest.importorskip("PyQt6.QtWidgets")


@pytest.fixture(scope="module")
def qt_app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_mask_on_base_in_another_grid(qt_app, tmp_path):
    from rtpspy.ui.image_viewer import ImageViewer

    # Base: 1 mm, 64^3; mask: 2 mm grid with a block at world (10, -5, 4)
    base = np.random.default_rng(0).random((64, 64, 64)).astype(np.float32)
    aff_b = np.diag([1.0, 1.0, 1.0, 1.0])
    aff_b[:3, 3] = -32
    nib.save(nib.Nifti1Image(base, aff_b), tmp_path / "base.nii.gz")

    mask = np.zeros((32, 32, 32), np.int16)
    aff_m = np.diag([2.0, 2.0, 2.0, 1.0])
    aff_m[:3, 3] = -32
    ijk = (np.array([10, -5, 4]) + 32) // 2
    mask[ijk[0] - 1:ijk[0] + 2, ijk[1] - 1:ijk[1] + 2,
         ijk[2] - 1:ijk[2] + 2] = 1
    nib.save(nib.Nifti1Image(mask, aff_m), tmp_path / "mask.nii.gz")

    v = ImageViewer(tmp_path / "base.nii.gz", tmp_path / "mask.nii.gz")
    assert v._ovl_mask and v._ovl.shape == base.shape
    # The cursor starts at the mask center, in base voxels (world + 32)
    assert np.allclose(v._cur, np.array([10, -5, 4]) + 32, atol=1)
    assert v._ovl[tuple(v._cur)] == 1
    v._set_alpha(20)
    assert v._alpha == 0.2
    v.close()


def test_4d_base_only(qt_app, tmp_path):
    from rtpspy.ui.image_viewer import ImageViewer

    data = np.random.default_rng(1).random((20, 22, 18, 3))
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)),
             tmp_path / "func.nii.gz")
    v = ImageViewer(tmp_path / "func.nii.gz")
    assert v._base.shape == (20, 22, 18) and v._ovl is None
    v.close()


def test_reuse_and_menus(qt_app, tmp_path):
    from rtpspy.ui.image_viewer import ImageViewer

    rng = np.random.default_rng(2)
    nib.save(nib.Nifti1Image(rng.random((20, 22, 18)).astype(np.float32),
                             np.eye(4)), tmp_path / "func.nii.gz")
    nib.save(nib.Nifti1Image(rng.random((40, 44, 36)).astype(np.float32),
                             np.diag([0.5, 0.5, 0.5, 1])),
             tmp_path / "anat.nii.gz")
    mask = np.zeros((20, 22, 18), np.int16)
    mask[5:8, 6:9, 4:7] = 1
    nib.save(nib.Nifti1Image(mask, np.eye(4)), tmp_path / "roi.nii.gz")
    choices = {"Base function image": tmp_path / "func.nii.gz",
               "Anatomy image (aligned)": tmp_path / "anat.nii.gz",
               "ROI mask": tmp_path / "roi.nii.gz"}

    v = ImageViewer(tmp_path / "func.nii.gz", choices=choices)
    names = [a.text() for a in v._base_menu.actions()]
    assert names[:3] == list(choices) and names[-1] == "Other file..."
    ovl_names = [a.text() for a in v._ovl_menu.actions()]
    assert ovl_names[0] == "None" and "ROI mask" in ovl_names
    assert not v._alpha_sld.isEnabled()

    # Overlay chosen from the menu: the cursor goes to the mask
    act = [a for a in v._ovl_menu.actions() if a.text() == "ROI mask"][0]
    act.trigger()
    assert v._ovl is not None and v._alpha_sld.isEnabled()
    assert np.allclose(v._cur, (6, 7, 5))
    assert act.isChecked()
    assert not v._ovl_menu.actions()[0].isChecked()  # "None"

    # Base changed in the same window: overlay resampled to the new grid
    [a for a in v._base_menu.actions()
     if a.text() == "Anatomy image (aligned)"][0].trigger()
    assert v._base.shape == (40, 44, 36) and v._ovl.shape == (40, 44, 36)
    assert "ROI mask on Anatomy" in v.windowTitle()

    # Overlay None
    v._ovl_menu.actions()[0].trigger()
    assert v._ovl is None and not v._alpha_sld.isEnabled()
    v.close()
