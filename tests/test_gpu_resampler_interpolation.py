"""Interpolation selection, exact-grid compatibility, and acquired support."""
import builtins
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
from scipy.ndimage import map_coordinates
import torch

import gpu_ants_like_resampler as gpu


def grid_file(path, shape=(6, 7, 8), points=None, exact=False):
    if points is None:
        coordinates = np.moveaxis(np.indices(shape, dtype=np.float32), 0, -1)
    else:
        coordinates = np.asarray(points, np.float64).reshape(-1, 1, 1, 3)
    fixed = coordinates.shape[:3]
    dims = np.asarray(shape)
    valid = np.isfinite(coordinates).all(axis=-1)
    valid &= ((coordinates >= -.5) & (coordinates < dims - .5)).all(axis=-1)
    normalized = 2 * np.clip(coordinates, 0, dims - 1) / (dims - 1) - 1
    normalized[~valid] = 3
    extra = dict(coordinates_xyz=coordinates, center_valid=valid) if exact else {}
    np.savez(path, grid=np.transpose(normalized.astype(np.float32), (2,1,0,3))[None],
             moving_shape=np.asarray(shape), fixed_shape=np.asarray(fixed), fixed_affine=np.eye(4), **extra)
    return path, coordinates, valid


def test_linear_default_does_not_import_lanczos_and_keeps_signed_values(tmp_path, monkeypatch):
    path, _, _ = grid_file(tmp_path / 'grid.npz')
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name in {'triton', 'gpu_lanczos_resampler'}:
            raise AssertionError('Linear should not import optional Lanczos dependencies')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded)
    sampler = gpu.GpuAntsLikeResampler(path, device='cpu')
    data = np.random.default_rng(31).normal(size=sampler.moving_shape).astype(np.float32)
    result = sampler.resample_array(data)
    np.testing.assert_allclose(result, data, rtol=1e-5, atol=1e-5)
    assert sampler.interpolation == 'linear'
    assert sampler.warmup_seconds == 0


def test_support_never_uses_the_intensity_kernel(tmp_path):
    path, _, _ = grid_file(tmp_path / 'grid.npz')
    sampler = gpu.GpuAntsLikeResampler(path, device='cpu')
    support = np.ones(sampler.moving_shape, bool)
    support[2, 3, 4] = False
    expected_coverage = np.zeros(sampler.fixed_shape, bool)
    sampler._lanczos = SimpleNamespace(
        resample_array=lambda data: np.full(sampler.fixed_shape, -999, np.float32),
        resample_coverage=lambda data: expected_coverage.copy(),
    )
    np.testing.assert_array_equal(sampler.resample_array(support), -999)
    np.testing.assert_allclose(sampler.resample_support(support), support, atol=1e-6)
    np.testing.assert_array_equal(sampler.resample_coverage(support), expected_coverage)


def test_linear_shared_buffer_is_serialized(tmp_path):
    path, _, _ = grid_file(tmp_path / 'grid.npz')
    sampler = gpu.GpuAntsLikeResampler(path, device='cpu')
    sources = [np.full(sampler.moving_shape, value, np.float32) for value in range(-10, 10)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(sampler.resample_array, sources))
    for expected, actual in zip(sources, outputs):
        np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_lanczos_rejects_cpu_and_legacy_grid(tmp_path, monkeypatch):
    path, _, _ = grid_file(tmp_path / 'grid.npz')
    with pytest.raises(RuntimeError, match='CUDA'):
        gpu.GpuAntsLikeResampler(path, device='cpu', interpolation='lanczos')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    with pytest.raises(ValueError, match='exact-coordinate grid'):
        gpu.GpuAntsLikeResampler(path, device='cuda', interpolation='lanczos')
    with pytest.raises(ValueError, match='interpolation'):
        gpu.GpuAntsLikeResampler(path, device='cpu', interpolation='unknown')


def fake_ants(monkeypatch):
    """Linear pull sampler with ITK's half-voxel center domain and edge clamp."""
    calls = []
    def run(command):
        calls.append(command)
        source = nib.load(command[command.index('-i') + 1])
        target = nib.load(command[command.index('-r') + 1])
        xyz = np.indices(target.shape[:3], dtype=np.float64).reshape(3, -1)
        physical = target.affine @ np.vstack([xyz, np.ones(xyz.shape[1])])
        points = (np.linalg.inv(source.affine) @ physical)[:3]
        for index, token in enumerate(command):
            if token == '-t':
                points[0] += float(Path(command[index + 1]).read_text())
        dims = np.asarray(source.shape[:3])[:, None]
        valid = ((points >= -.5) & (points < dims - .5)).all(axis=0)
        clipped = np.clip(points, 0, dims - 1)
        values = map_coordinates(source.get_fdata(), clipped, order=1, mode='nearest')
        values[~valid] = 0
        nib.save(nib.Nifti1Image(values.reshape(target.shape).astype(np.float32), target.affine),
                 command[command.index('-o') + 1])
    monkeypatch.setattr(gpu, '_run_cmd', run)
    return calls


def test_lanczos_grid_preserves_half_voxel_coordinates_and_linear_cache(tmp_path, monkeypatch):
    moving = tmp_path / 'moving.nii'
    fixed = tmp_path / 'fixed.nii'
    transform = tmp_path / 'shift.txt'
    transform.write_text('0')
    source = np.arange(6 * 7 * 8, dtype=np.float32).reshape(6, 7, 8)
    nib.save(nib.Nifti1Image(source, np.eye(4)), moving)
    affine = np.diag([.5, 1., 1., 1.])
    affine[:3, 3] = [-.25, 3., 3.]
    nib.save(nib.Nifti1Image(np.ones((4, 2, 2), np.float32), affine), fixed)
    calls = fake_ants(monkeypatch)
    args = (moving, fixed, [transform], tmp_path / 'grid')
    path = gpu.precompute_sampling_grid(*args)
    original = gpu.GpuAntsLikeResampler(path, device='cpu').resample_array(source)
    gpu.precompute_sampling_grid(*args)
    assert len(calls) == 4
    gpu.precompute_sampling_grid(*args, interpolation='lanczos')
    assert len(calls) == 8
    with np.load(path) as payload:
        np.testing.assert_allclose(payload['coordinates_xyz'][:,0,0,0], [-.25,.25,.75,1.25])
        assert payload['center_valid'].all()
        assert json.loads(payload['provenance'].item())['version'] == 3
    compatible = gpu.GpuAntsLikeResampler(path, device='cpu').resample_array(source)
    np.testing.assert_allclose(compatible, original, atol=1e-5)
    gpu.precompute_sampling_grid(*args, interpolation='lanczos')
    gpu.precompute_sampling_grid(*args)  # Enhanced grid also serves Linear.
    assert len(calls) == 8
    transform.write_text('1')
    gpu.precompute_sampling_grid(*args, interpolation='lanczos')
    assert len(calls) == 12


def test_constructor_does_not_mutate_float64_coordinates(monkeypatch):
    import gpu_lanczos_resampler as backend
    coordinates = np.asarray([[1,2,3],[-.5001,2,3],[5.5,2,3],[np.nan,1,1]], np.float64).reshape(4,1,1,3)
    original = coordinates.copy()
    payload = dict(grid=np.zeros((1,1,1,4,3), np.float32), moving_shape=np.asarray([6,7,8]),
                   fixed_shape=np.asarray([4,1,1]), fixed_affine=np.eye(4))
    class Allocation:
        def __init__(self, array): self.array = np.array(array, copy=True)
        def to(self, *args, **kwargs): return self
        def numel(self): return self.array.size
        def element_size(self): return self.array.dtype.itemsize
    monkeypatch.setattr(backend.np, 'load', lambda *a, **k: nullcontext(payload))
    monkeypatch.setattr(backend, 'triton', object())
    monkeypatch.setattr(backend.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(backend.torch.cuda, 'synchronize', lambda *a: None)
    monkeypatch.setattr(backend.torch, 'from_numpy', Allocation)
    monkeypatch.setattr(backend.torch, 'empty', lambda shape, **kw: Allocation(np.zeros(shape, np.float32)))
    sampler = backend.GpuLanczosResampler(Path('unused.npz'), coordinates_xyz=coordinates)
    np.testing.assert_array_equal(coordinates, original)
    np.testing.assert_array_equal(sampler._valid.array, [1,0,0,0])


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec('triton') is None,
                    reason='Actual CUDA/Triton coverage test')
def test_cuda_coverage_respects_zero_weights_and_acquisition_holes(tmp_path):
    shape = (9, 10, 11)
    points = [[0,0,0], [8,9,10], [4,4,4], [4.25,4,4], [4.25,4.25,4],
              [4.25,4.25,4.25], [-.25,4,4], [8.5,4,4]]
    path, coordinates, valid = grid_file(tmp_path / 'exact.npz', shape, points, exact=True)
    sampler = gpu.GpuAntsLikeResampler(path, interpolation='lanczos')
    support = np.ones(shape, bool)
    support[3,3,3] = False
    expected = [True,True,True,True,True,False,False,False]
    np.testing.assert_array_equal(sampler.resample_coverage(support).reshape(-1), expected)
    support[2,4,4] = False
    expected[3] = False
    expected[4] = False
    np.testing.assert_array_equal(sampler.resample_coverage(support).reshape(-1), expected)
    # Coverage remains a boolean operation; interpolated ones need not equal1.
    assert sampler.resample_support(np.ones(shape, np.float32)).reshape(-1)[0] == 1
    np.testing.assert_array_equal(sampler.resample_array(np.zeros(shape, np.float32)), 0)
