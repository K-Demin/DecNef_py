import numpy as np
import pytest

from fmri_rt_preproc.voxel_normalization import VoxelNormalizer


@pytest.mark.parametrize("mode", ["mean", "zscore"])
@pytest.mark.parametrize("reference", ["fixed", "running"])
def test_matches_independent_observed_only_numpy_history(mode, reference):
    rng = np.random.default_rng(701)
    shape = (2, 2, 3)
    brain = np.ones(shape, bool)
    brain[1, 1, 2] = False
    normalizer = VoxelNormalizer(mode, reference, 3, brain)
    history = [[] for _ in range(brain.size)]
    reference_count = 0
    for t in range(10):
        observed = rng.random(shape) > .2
        if t == 1:
            observed[:] = False
        observed.flat[0] = t != 1
        sample = rng.normal(size=shape) * 7. + 30.
        sample[~observed] = np.nan
        # A caller may scale residuals while observing raw intensities for mean.
        current = sample.copy() if mode == "zscore" else sample - 30.
        active = observed & brain
        expected = np.zeros(shape, np.float32)
        expected_support = np.zeros(shape, bool)
        if reference_count >= 3:
            for index, samples in enumerate(history):
                if not active.flat[index]:
                    continue
                if mode == "zscore" and len(samples) >= 2:
                    sd = np.std(samples, ddof=1)
                    if sd > 1e-6:
                        expected.flat[index] = (current.flat[index] - np.mean(samples)) / sd
                        expected_support.flat[index] = True
                elif mode == "mean" and samples and abs(np.mean(samples)) > 1e-6:
                    expected.flat[index] = 100. * current.flat[index] / np.mean(samples)
                    expected_support.flat[index] = True
        count_before = normalizer.count.copy()
        result = normalizer.transform(current, observed)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(normalizer.output_support, expected_support)
        np.testing.assert_array_equal(normalizer.count, count_before)
        assert result.dtype == np.float32
        normalizer.observe(sample, observed)
        if active.any() and (reference == "running" or reference_count < 3):
            for index in np.flatnonzero(active):
                history[index].append(sample.flat[index])
            reference_count += 1
        assert normalizer.reference_count == reference_count
        assert normalizer.ready == (reference_count >= 3)
        for index, samples in enumerate(history):
            assert normalizer.count.flat[index] == len(samples)
            np.testing.assert_allclose(normalizer.mean.flat[index], np.mean(samples) if samples else 0., atol=1e-12)
            np.testing.assert_allclose(normalizer.std.flat[index], np.std(samples, ddof=1) if len(samples) >= 2 else 0., atol=1e-12)


def test_running_current_sample_is_excluded_until_explicit_observe():
    normalizer = VoxelNormalizer(reference="running", ref_volumes=2)
    normalizer.observe(np.full((1, 1, 1), 1.))
    normalizer.observe(np.full((1, 1, 1), 3.))
    current = np.full((1, 1, 1), 100.)
    np.testing.assert_allclose(normalizer.transform(current), (100. - 2.) / np.sqrt(2.))
    assert normalizer.reference_count == 2
    normalizer.observe(current)
    np.testing.assert_allclose(normalizer.mean, np.mean([1., 3., 100.]))
    np.testing.assert_allclose(normalizer.std, np.std([1., 3., 100.], ddof=1))


def test_missing_samples_do_not_count_and_fixed_reference_cannot_gain_later_coverage():
    normalizer = VoxelNormalizer(ref_volumes=2)
    support = np.array([[[True, False, True]]])
    normalizer.observe(np.array([[[0., np.nan, -2.]]]), support)
    normalizer.observe(np.array([[[2., np.nan, 2.]]]), support)
    np.testing.assert_array_equal(normalizer.count, [[[2, 0, 2]]])
    np.testing.assert_allclose(normalizer.mean, [[[1., 0., 0.]]])
    normalizer.observe(np.array([[[100., 100., 100.]]]))
    output = normalizer.transform(np.array([[[0., 100., -2.]]]))
    np.testing.assert_allclose(output, [[[-1 / np.sqrt(2), 0., -1 / np.sqrt(2)]]])
    np.testing.assert_array_equal(normalizer.output_support, [[[True, False, True]]])
    assert normalizer.reference_count == 2


def test_global_readiness_does_not_imply_each_voxel_has_sufficient_variance():
    normalizer = VoxelNormalizer(reference="running", ref_volumes=2)
    normalizer.observe(np.array([[[1., 5., 7., 3.]]]), np.array([[[True, True, False, True]]]))
    normalizer.observe(np.array([[[3., 5., 8., 3. + 1e-8]]]))
    assert normalizer.ready
    output = normalizer.transform(np.array([[[5., 10., 9., 4.]]]))
    np.testing.assert_array_equal(normalizer.output_support, [[[True, False, False, False]]])
    np.testing.assert_array_equal(output[..., 1:], 0.)
    normalizer.observe(np.array([[[4., 5., 10., 3.]]]))
    normalizer.transform(np.array([[[5., 10., 9., 4.]]]))
    assert normalizer.output_support[0, 0, 2]


@pytest.mark.parametrize("mode", ["mean", "zscore"])
def test_no_clipping_and_support_is_recomputed_for_each_output(mode):
    normalizer = VoxelNormalizer(mode=mode, ref_volumes=2)
    normalizer.observe(np.array([[[1., -1.]]]))
    normalizer.observe(np.array([[[3., -3.]]]))
    result = normalizer.transform(np.array([[[10000., 10000.]]]))
    assert np.abs(result).min() > 200.
    support = np.array([[[False, True]]])
    result = normalizer.transform(np.array([[[np.nan, 0.]]]), support)
    np.testing.assert_allclose(result, [[[0., np.sqrt(2.) if mode == "zscore" else 0.]]])
    np.testing.assert_array_equal(normalizer.output_support, support)
    normalizer.transform(np.array([[[-1., 0.]]]))
    assert normalizer.output_support.all()


def test_mean_uses_raw_reference_not_mean_of_residuals():
    normalizer = VoxelNormalizer(mode="mean", ref_volumes=2)
    normalizer.observe(np.array([[[900., -900., 0.]]]))
    normalizer.observe(np.array([[[1100., -1100., 0.]]]))
    result = normalizer.transform(np.array([[[-20., 20., 15.]]]))
    np.testing.assert_allclose(result, [[[-2., -2., 0.]]])
    np.testing.assert_array_equal(normalizer.output_support, [[[True, True, False]]])


def test_empty_coverage_does_not_advance_reference_or_leak_output():
    normalizer = VoxelNormalizer(ref_volumes=2)
    data = np.ones((2, 2, 2))
    missing = np.zeros_like(data, bool)
    normalizer.observe(np.full_like(data, np.nan), missing)
    assert normalizer.reference_count == 0
    np.testing.assert_array_equal(normalizer.transform(data), 0.)
    assert not normalizer.output_support.any()
    normalizer.observe(data)
    normalizer.observe(data * 2)
    normalizer.transform(data)
    assert normalizer.output_support.all()
    np.testing.assert_array_equal(normalizer.transform(np.full_like(data, np.nan), missing), 0.)
    assert not normalizer.output_support.any()


@pytest.mark.parametrize("operation", ["observe", "transform"])
def test_rejects_nonfinite_acquired_values_without_mutating_statistics(operation):
    normalizer = VoxelNormalizer(ref_volumes=2, brain_mask=np.array([[[True, False]]]))
    normalizer.observe(np.array([[[1., np.nan]]]))
    normalizer.observe(np.array([[[3., np.nan]]]))
    before = (normalizer.mean.copy(), normalizer.count.copy(), normalizer.std.copy())
    with pytest.raises(ValueError, match="Nonfinite acquired"):
        getattr(normalizer, operation)(np.array([[[np.nan, 5.]]]))
    for actual, expected in zip((normalizer.mean, normalizer.count, normalizer.std), before):
        np.testing.assert_array_equal(actual, expected)
    assert normalizer.reference_count == 2


@pytest.mark.parametrize("kwargs", [
    {"mode": "none"}, {"reference": "rolling"}, {"ref_volumes": 1},
    {"ref_volumes": 2.5}, {"ref_volumes": True}, {"ref_volumes": np.inf},
    {"mode": "mean", "ref_volumes": 0}, {"brain_mask": np.ones((2, 2))},
    {"brain_mask": np.array([[[np.nan]]])},
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        VoxelNormalizer(**kwargs)


def test_shape_and_nonbinary_support_are_rejected():
    normalizer = VoxelNormalizer(ref_volumes=2, brain_mask=np.ones((2, 2, 2), np.uint8))
    with pytest.raises(ValueError, match="shape"):
        normalizer.observe(np.ones((2, 2, 3)))
    with pytest.raises(ValueError, match="shapes"):
        normalizer.observe(np.ones((2, 2, 2)), np.ones((2, 2, 3), bool))
    with pytest.raises(ValueError, match="binary"):
        normalizer.transform(np.ones((2, 2, 2)), np.full((2, 2, 2), .5))
    with pytest.raises(ValueError, match="3D"):
        normalizer.observe(np.ones((2, 2)))
    assert normalizer.reference_count == 0


def test_mean_mode_allows_one_reference_volume():
    normalizer = VoxelNormalizer(mode="mean", ref_volumes=1)
    normalizer.observe(np.array([[[10.]]]))
    assert normalizer.ready
    np.testing.assert_allclose(normalizer.transform(np.array([[[2.]]])), 20.)
