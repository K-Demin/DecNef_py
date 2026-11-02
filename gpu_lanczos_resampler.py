"""Cached CUDA Lanczos-3 intensity and strict acquired-support interpolation.

ANTs 2.6.2 semantics: 6 samples per axis, unnormalized sinc(d)*sinc(d/3),
zero-valued out-of-array neighbors, and a half-open [-.5, n-.5) center domain.
Source/output arrays are XYZ, matching GpuAntsLikeResampler. Kernel storage is
ZYX with X contiguous. Precomputation/JIT are outside warm resample_array calls.

Imported lazily by gpu_ants_like_resampler only when Lanczos is selected.
Exact coordinates must come from padded coordinate ramps; legacy Linear
coordinate-image grids clamp the outer half-voxel and are rejected upstream.
"""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _lanczos3_kernel(
        source, bases, weights, validity, output,
        N: tl.constexpr, SX: tl.constexpr, SY: tl.constexpr, SZ: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        exists = p < N
        valid = exists & (tl.load(validity + p, mask=exists, other=0) != 0)
        bx = tl.load(bases + p, mask=exists, other=0)
        by = tl.load(bases + N + p, mask=exists, other=0)
        bz = tl.load(bases + 2 * N + p, mask=exists, other=0)
        accum = tl.full((BLOCK,), 0.0, tl.float32)
        for iz in range(0, 6):
            z = bz + iz - 2
            wz = tl.load(weights + (12 + iz) * N + p, mask=exists, other=0.0)
            plane = tl.full((BLOCK,), 0.0, tl.float32)
            for iy in range(0, 6):
                y = by + iy - 2
                wy = tl.load(weights + (6 + iy) * N + p, mask=exists, other=0.0)
                row = tl.full((BLOCK,), 0.0, tl.float32)
                for ix in tl.static_range(0, 6):
                    x = bx + ix - 2
                    wx = tl.load(weights + ix * N + p, mask=exists, other=0.0)
                    inside = valid & (x >= 0) & (x < SX) & (y >= 0) & (y < SY) & (z >= 0) & (z < SZ)
                    value = tl.load(source + (z * SY + y) * SX + x, mask=inside, other=0.0)
                    row += value * wx
                plane += row * wy
            accum += plane * wz
        tl.store(output + p, tl.where(valid, accum, 0.0), mask=exists)


    @triton.jit
    def _lanczos3_coverage_kernel(
        source, bases, weights, validity, output,
        N: tl.constexpr, SX: tl.constexpr, SY: tl.constexpr, SZ: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        exists = p < N
        covered = exists & (tl.load(validity + p, mask=exists, other=0) != 0)
        bx = tl.load(bases + p, mask=exists, other=0)
        by = tl.load(bases + N + p, mask=exists, other=0)
        bz = tl.load(bases + 2 * N + p, mask=exists, other=0)
        for iz in range(0, 6):
            z = bz + iz - 2
            wz = tl.load(weights + (12 + iz) * N + p, mask=exists, other=0.0)
            for iy in range(0, 6):
                y = by + iy - 2
                wy = tl.load(weights + (6 + iy) * N + p, mask=exists, other=0.0)
                for ix in tl.static_range(0, 6):
                    x = bx + ix - 2
                    wx = tl.load(weights + ix * N + p, mask=exists, other=0.0)
                    needed = (wx != 0.0) & (wy != 0.0) & (wz != 0.0)
                    inside = (x >= 0) & (x < SX) & (y >= 0) & (y < SY) & (z >= 0) & (z < SZ)
                    present = tl.load(source + (z * SY + y) * SX + x,
                                      mask=covered & needed & inside, other=0) != 0
                    covered = covered & ((~needed) | (inside & present))
        tl.store(output + p, covered.to(tl.uint8), mask=exists)


def axis_lanczos3_weights(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return floor(c) and six unnormalized ITK-equivalent weights in float64."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if not np.isfinite(coordinates).all():
        raise ValueError('Non-finite coordinates must be excluded before weight calculation.')
    base = np.floor(coordinates).astype(np.int32)
    fraction = coordinates - base
    distances = fraction[None, :] + 2.0 - np.arange(6, dtype=np.float64)[:, None]
    weights = np.sinc(distances) * np.sinc(distances / 3.0)
    integer = fraction == 0.0
    weights[:, integer] = 0.0
    weights[2, integer] = 1.0
    return base, weights


class GpuLanczosResampler:
    """Cached Lanczos-3 backend; CUDA is required, with no fallback.

    coordinates_xyz, when supplied, has fixed-grid XYZ shape plus a final XYZ
    coordinate axis: (*fixed_shape, 3). validity has shape fixed_shape. These
    override the normalized production grid with exact pull coordinates.
    """
    def __init__(self, grid_path: Path, device: str = 'cuda', *,
                 coordinates_xyz: np.ndarray | None = None,
                 validity: np.ndarray | None = None,
                 block_size: int = 128):
        start = time.perf_counter()
        if triton is None:
            raise ImportError('GPU Lanczos requires Triton in the CUDA environment.')
        self.device = torch.device(device)
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('GPU Lanczos requires an available CUDA device; CPU fallback is unsupported.')
        self.block_size = int(block_size)
        if self.block_size not in (64, 128, 256):
            raise ValueError('block_size must be 64, 128, or 256.')
        with np.load(str(grid_path)) as payload:
            self.moving_shape = tuple(int(x) for x in payload['moving_shape'])
            self.fixed_shape = tuple(int(x) for x in payload['fixed_shape'])
            self.fixed_affine = np.asarray(payload['fixed_affine'], dtype=np.float64)
            grid = np.asarray(payload['grid'], dtype=np.float64)[0]
        expected_grid_shape = self.fixed_shape[::-1] + (3,)
        if grid.shape != expected_grid_shape:
            raise ValueError(f'Unexpected normalized grid shape: {grid.shape}.')
        dims = np.asarray(self.moving_shape, dtype=np.float64)
        if coordinates_xyz is None:
            coordinates = (grid + 1.0) * ((dims - 1.0) / 2.0)
            self.coordinate_source = 'production normalized grid; outer-half-voxel clamping remains'
        else:
            exact = np.asarray(coordinates_xyz, dtype=np.float64)
            if exact.shape != self.fixed_shape + (3,):
                raise ValueError('Exact coordinates must have shape (*fixed_shape, 3).')
            coordinates = np.transpose(exact, (2, 1, 0, 3))
            self.coordinate_source = 'explicit source voxel coordinates'
        # Own this array before replacing invalid coordinates. For singleton
        # dimensions ascontiguousarray can otherwise alias a caller's float64
        # coordinates, silently mutating the oracle/input used by another call.
        points = np.array(coordinates.reshape(-1, 3), dtype=np.float64, order='C', copy=True)
        valid = np.isfinite(points).all(axis=1)
        valid &= ((points >= -.5) & (points < dims - .5)).all(axis=1)
        if coordinates_xyz is None:
            valid &= np.isfinite(grid.reshape(-1, 3)).all(axis=1)
            valid &= (np.abs(grid.reshape(-1, 3)) < 2.0).all(axis=1)
        if validity is not None:
            valid_xyz = np.asarray(validity, dtype=bool)
            if valid_xyz.shape != self.fixed_shape:
                raise ValueError('Explicit validity must have shape fixed_shape.')
            valid &= np.transpose(valid_xyz, (2, 1, 0)).reshape(-1)
        points[~valid] = 0.0
        basis, weight = zip(*(axis_lanczos3_weights(points[:, axis]) for axis in range(3)))
        base_array = np.ascontiguousarray(np.stack(basis), dtype=np.int32)
        weight_array = np.ascontiguousarray(np.concatenate(weight), dtype=np.float32)
        self._bases = torch.from_numpy(base_array).to(self.device)
        self._weights = torch.from_numpy(weight_array).to(self.device)
        self._valid = torch.from_numpy(valid.astype(np.uint8)).to(self.device)
        self._input = torch.empty(self.moving_shape[::-1], dtype=torch.float32, device=self.device)
        self._output = torch.empty(self.fixed_shape[::-1], dtype=torch.float32, device=self.device)
        self._support_input = torch.empty(self.moving_shape[::-1], dtype=torch.uint8, device=self.device)
        self._coverage_output = torch.empty(self.fixed_shape[::-1], dtype=torch.uint8, device=self.device)
        self._n_output = int(np.prod(self.fixed_shape))
        self.cached_bytes = sum(t.numel() * t.element_size() for t in
                                (self._bases, self._weights, self._valid, self._input, self._output,
                                 self._support_input, self._coverage_output))
        torch.cuda.synchronize(self.device)
        self.setup_seconds = time.perf_counter() - start

    @torch.inference_mode()
    def resample_array(self, data_xyz: np.ndarray) -> np.ndarray:
        data = np.asarray(data_xyz)
        if data.shape != self.moving_shape:
            raise ValueError(f'Input shape {data.shape} does not match {self.moving_shape}.')
        src_zyx = np.transpose(data.astype(np.float32, copy=False), (2, 1, 0))
        source = torch.from_numpy(src_zyx).to(self.device, non_blocking=True)
        self._input.copy_(source)
        with torch.cuda.device(self.device):
            _lanczos3_kernel[(triton.cdiv(self._n_output, self.block_size),)](
                self._input, self._bases, self._weights, self._valid, self._output,
                self._n_output, *self.moving_shape, self.block_size,
                num_warps=4, enable_fp_fusion=False,
            )
        out_zyx = self._output.detach().cpu().numpy()
        return np.transpose(out_zyx, (2, 1, 0)).astype(np.float32, copy=False)

    @torch.inference_mode()
    def resample_coverage(self, support_xyz: np.ndarray) -> np.ndarray:
        """Require every nonzero kernel tap to sample acquired source data.

        This is a boolean geometry operation, never Lanczos interpolation of
        an all-ones mask. At exact integer centers only the center tap is
        needed. Calls are serialized by the public resampler's lock.
        """
        support = np.asarray(support_xyz)
        if support.shape != self.moving_shape:
            raise ValueError(f'Support shape {support.shape} does not match {self.moving_shape}.')
        if support.dtype != np.bool_:
            support = np.isfinite(support) & (support >= 1.0 - 1e-5)
        src_zyx = np.transpose(support.astype(np.uint8), (2, 1, 0))
        source = torch.from_numpy(src_zyx).to(self.device, non_blocking=True)
        self._support_input.copy_(source)
        with torch.cuda.device(self.device):
            _lanczos3_coverage_kernel[(triton.cdiv(self._n_output, self.block_size),)](
                self._support_input, self._bases, self._weights, self._valid, self._coverage_output,
                self._n_output, *self.moving_shape, self.block_size,
                num_warps=4,
            )
        out_zyx = self._coverage_output.detach().cpu().numpy()
        return np.transpose(out_zyx, (2, 1, 0)).astype(bool, copy=False)

    def resample_nifti_to_array(self, nii_path: Path) -> np.ndarray:
        import nibabel as nib
        return self.resample_array(np.asanyarray(nib.load(str(nii_path)).dataobj))

    def resample_nifti_to_nifti(self, in_path: Path, out_path: Path) -> Path:
        import nibabel as nib
        result = self.resample_nifti_to_array(in_path)
        nib.save(nib.Nifti1Image(result, self.fixed_affine), str(out_path))
        return out_path
