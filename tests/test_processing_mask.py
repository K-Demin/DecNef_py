"""Check that processing margins retain actual interpolation contributors."""
from itertools import product

import numpy as np
import pytest

from fmri_rt_preproc.processing_mask import (
    PROCESSING_MASK_POLICY,
    PROCESSING_MASK_POLICY_VERSION,
    make_processing_mask,
    processing_margin_vox,
)


@pytest.mark.parametrize("interpolation,margin", [("linear", 1), ("lanczos", 3)])
def test_kernel_margin_preserves_anatomical_input(interpolation, margin):
    brain = np.zeros((11, 12, 13), np.uint8)
    brain[5, 6, 7] = 1
    before = brain.copy()
    processing = make_processing_mask(brain, interpolation)
    expected = np.zeros(brain.shape, bool)
    expected[5-margin:6+margin, 6-margin:7+margin, 7-margin:8+margin] = True
    np.testing.assert_array_equal(processing, expected)
    np.testing.assert_array_equal(brain, before)
    assert processing.dtype == bool
    assert not np.shares_memory(processing, brain)
    assert processing_margin_vox(interpolation) == margin


def _contributors(point, interpolation):
    """Independent kernel-support definition; no production interpolation call."""
    axes = []
    for coordinate in point:
        base = int(np.floor(coordinate))
        if coordinate == base:
            axes.append([base])
        elif interpolation == "linear":
            axes.append([base, base + 1])
        else:
            axes.append(list(range(base - 2, base + 4)))
    return product(*axes)


@pytest.mark.parametrize("interpolation", ["linear", "lanczos"])
def test_all_nonzero_kernel_contributors_fit_for_interior_and_edge_centres(interpolation):
    shape = np.array([13, 14, 15])
    centres = [(6, 7, 8), *product(*[(0, int(size - 1)) for size in shape])]
    fractions = list(product((-.499999, -.2, 0., .2, .499999), repeat=3))
    for centre in centres:
        brain = np.zeros(tuple(shape), bool)
        brain[centre] = True
        processing = make_processing_mask(brain, interpolation)
        for offset in fractions:
            point = np.asarray(centre) + offset
            np.testing.assert_array_equal(np.floor(point + .5).astype(int), centre)
            for contributor in _contributors(point, interpolation):
                index = np.asarray(contributor)
                if np.all(index >= 0) and np.all(index < shape):
                    assert processing[contributor], (interpolation, point, contributor)
    # Same source grid, clipped at its boundaries; no extra acquired voxels.
    assert processing.shape == tuple(shape)


def test_lanczos_needs_a_cube_not_a_radius_three_sphere():
    brain = np.zeros((11, 11, 11), bool)
    brain[5, 5, 5] = True
    point = np.array([5.2, 5.2, 5.2])
    contributor = (8, 8, 8)
    assert contributor in list(_contributors(point, "lanczos"))
    assert np.linalg.norm(np.asarray(contributor) - [5, 5, 5]) > 3
    assert make_processing_mask(brain, "lanczos")[contributor]


@pytest.mark.parametrize("interpolation", ["linear", "lanczos"])
def test_no_spatial_resampling_returns_independent_unchanged_binary_mask(interpolation):
    brain = np.zeros((3, 4, 5), np.float32)
    brain[1, 2, 3] = 1
    processing = make_processing_mask(brain, interpolation, spatial_resampling=False)
    assert processing_margin_vox(interpolation, spatial_resampling=False) == 0
    np.testing.assert_array_equal(processing, brain.astype(bool))
    assert not np.shares_memory(processing, brain)


@pytest.mark.parametrize("mask", [
    np.ones((3, 3)),
    np.ones((3, 3, 3, 1)),
    np.ones((3, 0, 3)),
    np.zeros((3, 3, 3)),
    np.full((3, 3, 3), .5),
    np.full((3, 3, 3), np.nan),
    np.full((3, 3, 3), np.inf),
    np.full((3, 3, 3), -1),
    np.full((3, 3, 3), 2),
    np.ones((3, 3, 3), complex),
    np.full((3, 3, 3), "1"),
])
def test_rejects_invalid_brain_masks(mask):
    with pytest.raises(ValueError, match="Brain mask"):
        make_processing_mask(mask)


def test_policy_metadata_and_interpolation_validation():
    assert PROCESSING_MASK_POLICY == "source_grid_cubic_halo"
    assert PROCESSING_MASK_POLICY_VERSION == 1
    assert processing_margin_vox(" Lanczos ") == 3
    with pytest.raises(ValueError, match="interpolation"):
        processing_margin_vox("cubic", spatial_resampling=False)
    with pytest.raises(ValueError, match="boolean"):
        processing_margin_vox("linear", spatial_resampling="false")
