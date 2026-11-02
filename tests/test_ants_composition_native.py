"""Optional native ANTs numerical check of the session composite conventions."""
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

ants = pytest.importorskip("ants")
from ants.internal import get_lib_fn
from fmri_rt_preproc.offline_assets import compose_transforms, apply_image


def native_run(args):
    result = get_lib_fn(args[0])(list(map(str, args[1:])))
    if result:
        raise RuntimeError(f"Native ANTs returned {result}: {args}")


def test_noncommuting_forward_and_inverse_composites_match_native_chains(tmp_path):
    shape = (20, 21, 22)
    xyz = np.indices(shape).astype(np.float32)
    data = 100 + xyz[0] + 2 * xyz[1] + 3 * xyz[2]
    image = tmp_path / "input.nii"
    nib.save(nib.Nifti1Image(data, np.eye(4)), image)
    theta = .08
    matrix = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1.02]]
    transform_a = ants.create_ants_transform(transform_type="AffineTransform", dimension=3,
                                            matrix=matrix, translation=(.4, .7, -.2))
    transform_b = ants.create_ants_transform(transform_type="AffineTransform", dimension=3,
                                            translation=(1.2, -1.3, .6))
    a, b = tmp_path / "a.mat", tmp_path / "b.mat"
    ants.write_transform(transform_a, str(a))
    ants.write_transform(transform_b, str(b))
    for label, chain in (("forward", [a, b]), ("inverse", [f"[{b},1]", f"[{a},1]"])):
        composite = tmp_path / (label + ".h5")
        compose_transforms(chain, image, composite, native_run)
        direct, composed = tmp_path / (label + "_direct.nii"), tmp_path / (label + "_composed.nii")
        apply_image(image, image, direct, chain, native_run)
        apply_image(image, image, composed, [composite], native_run)
        np.testing.assert_allclose(nib.load(direct).get_fdata(), nib.load(composed).get_fdata(), atol=5e-4)
    # Order is observable: this guards against a test using commuting translations.
    wrong = tmp_path / "wrong_order.nii"
    apply_image(image, image, wrong, [b, a], native_run)
    correct = nib.load(tmp_path / "forward_direct.nii").get_fdata()
    assert np.max(np.abs(correct[4:-4, 4:-4, 4:-4] - nib.load(wrong).get_fdata()[4:-4, 4:-4, 4:-4])) > .05
