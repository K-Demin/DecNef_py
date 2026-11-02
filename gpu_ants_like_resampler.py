from __future__ import annotations

import subprocess
import hashlib
import json
import zipfile
import threading
import time
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


def _run_cmd(cmd: Sequence[str | Path]) -> None:
    cmd_str = [str(c) for c in cmd]
    subprocess.run(cmd_str, check=True)


def _torch_load_compat(path: Path):
    """
    PyTorch >=2.6 defaults torch.load(..., weights_only=True), which rejects
    non-tensor Python objects. Our grid payload is produced locally and trusted.
    """
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch versions do not expose the weights_only argument.
        return torch.load(str(path), map_location="cpu")


def _interpolation_name(value: str) -> str:
    name = str(value).strip().lower()
    if name not in {"linear", "lanczos"}:
        raise ValueError("GPU interpolation must be 'linear' or 'lanczos'.")
    return name


def sampling_grid_identity(moving_ref: Path, fixed_ref: Path, transforms: Sequence[Path],
                           *, interpolation: str = "linear") -> dict:
    """Identify the grids and ordered transform contents used for pull-resampling."""
    def geometry(path):
        img = nib.load(str(path))
        return {
            "shape": list(img.shape[:3]),
            "affine": img.affine.tolist(),
            "zooms": [float(v) for v in img.header.get_zooms()[:3]],
            "qform": img.get_qform().tolist(),
            "qform_code": int(img.header["qform_code"]),
            "sform": img.get_sform().tolist(),
            "sform_code": int(img.header["sform_code"]),
        }

    identities = []
    for transform in transforms:
        path = Path(transform)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        identities.append({"path": str(path.resolve()), "sha256": digest.hexdigest()})
    result = {"version": 2, "moving": geometry(moving_ref), "fixed": geometry(fixed_ref),
              "transforms": identities}
    if _interpolation_name(interpolation) == "lanczos":
        result.update(version=3, coordinate_representation="padded_voxel_ramps_v1", coordinate_padding=3)
    return result


def make_voxel_coordinate_images(moving_ref: Path, out_dir: Path, *, padding: int = 0) -> dict[str, Path]:
    """
    Create scalar NIfTI images on the moving grid whose values are voxel x/y/z indices.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img = nib.load(str(moving_ref))
    source_shape = img.shape[:3]
    shape = tuple(n + 2 * padding for n in source_shape)
    affine = img.affine.copy()
    header = img.header.copy()
    if padding:
        affine[:3, 3] = nib.affines.apply_affine(img.affine, [-padding] * 3)
        header.set_data_dtype(np.float32)

    x = np.arange(-padding, source_shape[0] + padding, dtype=np.float32)[:, None, None]
    y = np.arange(-padding, source_shape[1] + padding, dtype=np.float32)[None, :, None]
    z = np.arange(-padding, source_shape[2] + padding, dtype=np.float32)[None, None, :]
    xx = np.broadcast_to(x, shape).astype(np.float32)
    yy = np.broadcast_to(y, shape).astype(np.float32)
    zz = np.broadcast_to(z, shape).astype(np.float32)

    suffix = f"_padded{padding}" if padding else ""
    paths = {axis: out_dir / f"coord_{axis}_epi_vox{suffix}.nii" for axis in "xyz"}
    nib.save(nib.Nifti1Image(xx, affine, header), str(paths["x"]))
    nib.save(nib.Nifti1Image(yy, affine, header), str(paths["y"]))
    nib.save(nib.Nifti1Image(zz, affine, header), str(paths["z"]))
    return paths


def precompute_sampling_grid(
    moving_ref: Path,
    fixed_ref: Path,
    transforms: Sequence[Path],
    out_dir: Path,
    grid_name: str = "sampling_grid.npz",
    *,
    interpolation: str = "linear",
) -> Path:
    """
    Build the fixed pull lookup. Linear retains its original cache format.

    Lanczos adds padded coordinate ramps so the outer half-voxel is not
    clamped. Its payload also contains the original clamped Linear lookup
    used for support interpolation. A version-3 cache can serve Linear too.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    interpolation = _interpolation_name(interpolation)
    identity = sampling_grid_identity(moving_ref, fixed_ref, transforms, interpolation=interpolation)
    grid_path = out_dir / grid_name
    if grid_path.exists():
        try:
            with np.load(str(grid_path)) as payload:
                stored = json.loads(str(payload["provenance"].item()))
                matches = stored == identity
                if interpolation == "linear" and stored.get("version") == 3:
                    compatible = {key: value for key, value in stored.items()
                                  if key not in {"coordinate_representation", "coordinate_padding"}}
                    compatible["version"] = 2
                    matches = (stored.get("coordinate_representation") == "padded_voxel_ramps_v1"
                               and stored.get("coordinate_padding") == 3 and compatible == identity)
                if interpolation == "lanczos":
                    matches = matches and {"coordinates_xyz", "center_valid"}.issubset(payload.files)
                if matches:
                    return grid_path
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            pass  # Legacy, incomplete, or incompatible grids must be rebuilt.

    # Regenerate every coordinate image together. Reusing an axis by filename
    # would retain the old transform even after the grid itself is invalidated.
    coord_paths = make_voxel_coordinate_images(moving_ref, out_dir, padding=3 if interpolation == "lanczos" else 0)
    moving_img = nib.load(str(moving_ref))
    coverage_path = out_dir / "coverage_epi.nii"
    nib.save(nib.Nifti1Image(np.ones(moving_img.shape[:3], np.float32),
                            moving_img.affine, moving_img.header), str(coverage_path))
    coord_paths["coverage"] = coverage_path

    warped_paths: dict[str, Path] = {}
    for axis, coord_img in coord_paths.items():
        warped_img = out_dir / f"coord_{axis}_in_fixed.nii"
        cmd = [
            "antsApplyTransforms",
            "-d", "3",
            "-i", str(coord_img),
            "-r", str(fixed_ref),
            "-o", str(warped_img),
        ]
        for transform in transforms:
            cmd.extend(["-t", str(transform)])
        cmd.extend(["-n", "Linear", "--float", "1"])
        _run_cmd(cmd)
        warped_paths[axis] = warped_img

    fixed_img = nib.load(str(fixed_ref))
    moving_shape = np.array(moving_img.shape[:3], dtype=np.float32)
    cx = np.asanyarray(nib.load(str(warped_paths["x"])).dataobj).astype(np.float32)
    cy = np.asanyarray(nib.load(str(warped_paths["y"])).dataobj).astype(np.float32)
    cz = np.asanyarray(nib.load(str(warped_paths["z"])).dataobj).astype(np.float32)

    coordinates_xyz = np.stack([cx, cy, cz], axis=-1)
    coverage = np.asarray(nib.load(str(warped_paths["coverage"])).dataobj)
    # ANTs fills out-of-field coordinate images with 0: that is otherwise
    # indistinguishable from the legitimate source voxel (0, 0, 0).
    # A separate ones image identifies these locations before normalization.
    valid = np.isfinite(coverage) & (coverage >= 1.0 - 1e-5)
    extra = {}
    linear_coordinates = coordinates_xyz
    if interpolation == "lanczos":
        valid &= np.isfinite(coordinates_xyz).all(axis=-1)
        valid &= ((coordinates_xyz >= -.5) & (coordinates_xyz < moving_shape - .5)).all(axis=-1)
        extra = {"coordinates_xyz": coordinates_xyz, "center_valid": valid}
        linear_coordinates = np.clip(coordinates_xyz, 0, moving_shape - 1)
    denominator = np.maximum(moving_shape - 1, 1)
    grid_xyz = 2.0 * linear_coordinates / denominator - 1.0
    grid_xyz[~valid] = 3.0  # Fully outside grid_sample's source extent.
    # nibabel uses XYZ; grid_sample stores ZYX and final XYZ components.
    grid_zyx = np.transpose(grid_xyz, (2, 1, 0, 3))
    grid = torch.from_numpy(grid_zyx[None, ...]).float()

    np.savez_compressed(
        str(grid_path),
        grid=grid.numpy(),
        moving_shape=np.asarray(moving_img.shape[:3], dtype=np.int32),
        fixed_shape=np.asarray(fixed_img.shape[:3], dtype=np.int32),
        fixed_affine=fixed_img.affine.astype(np.float64),
        provenance=json.dumps(identity, sort_keys=True),
        **extra,
    )
    return grid_path


class GpuAntsLikeResampler:
    def __init__(
        self,
        grid_path: Path,
        device: str = "cuda",
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        *,
        interpolation: str = "linear",
    ):
        payload = self._load_payload(grid_path)
        self.interpolation = _interpolation_name(interpolation)
        self._lock = threading.RLock()
        selected_device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.device = torch.device(selected_device)
        if self.interpolation == "lanczos":
            if self.device.type != "cuda" or not torch.cuda.is_available():
                raise RuntimeError("GPU Lanczos requires an available CUDA device; CPU fallback is unsupported.")
            if mode != "bilinear" or padding_mode != "zeros":
                raise ValueError("Lanczos uses its own intensity kernel with zero padding; keep mode='bilinear' for support.")
            if payload.get("coordinates_xyz") is None or payload.get("center_valid") is None:
                raise ValueError("Lanczos requires an exact-coordinate grid; rebuild with interpolation='lanczos'.")
        self.mode = mode
        self.padding_mode = padding_mode
        self.grid = payload["grid"].to(self.device, non_blocking=True)
        self.moving_shape = tuple(payload["moving_shape"])
        self.fixed_shape = tuple(payload["fixed_shape"])
        self.fixed_affine = np.asarray(payload["fixed_affine"])
        self._input = torch.empty(
            (1, 1, self.moving_shape[2], self.moving_shape[1], self.moving_shape[0]),
            dtype=torch.float32,
            device=self.device,
        )
        self._lanczos = None
        self.warmup_seconds = 0.0
        if self.interpolation == "lanczos":
            # Optional dependency stays out of the unchanged Linear path.
            from gpu_lanczos_resampler import GpuLanczosResampler
            self._lanczos = GpuLanczosResampler(grid_path, device=str(self.device),
                                               coordinates_xyz=payload["coordinates_xyz"],
                                               validity=payload["center_valid"])
            start = time.perf_counter()
            self.resample_array(np.zeros(self.moving_shape, np.float32))
            self.resample_support(np.ones(self.moving_shape, np.float32))
            torch.cuda.synchronize(self.device)
            self.warmup_seconds = time.perf_counter() - start

    @staticmethod
    def _load_payload(grid_path: Path) -> dict:
        if grid_path.suffix == ".npz":
            with np.load(str(grid_path)) as npz:
                result = {
                    "grid": torch.from_numpy(npz["grid"]).float(),
                    "moving_shape": tuple(int(v) for v in npz["moving_shape"].tolist()),
                    "fixed_shape": tuple(int(v) for v in npz["fixed_shape"].tolist()),
                    "fixed_affine": np.asarray(npz["fixed_affine"], dtype=np.float64),
                }
                if "coordinates_xyz" in npz and "center_valid" in npz:
                    result["coordinates_xyz"] = np.asarray(npz["coordinates_xyz"], dtype=np.float32)
                    result["center_valid"] = np.asarray(npz["center_valid"], dtype=bool)
                return result

        # Backward compatibility with older .pt payloads.
        payload = _torch_load_compat(grid_path)
        payload["fixed_affine"] = np.asarray(payload["fixed_affine"], dtype=np.float64)
        return payload

    @torch.inference_mode()
    def resample_array(self, data_xyz: np.ndarray) -> np.ndarray:
        """Resample intensities; only this operation uses the selected kernel."""
        with self._lock:
            if self._lanczos is not None:
                return self._lanczos.resample_array(data_xyz)
            return self._resample_linear(data_xyz, mode=self.mode, padding_mode=self.padding_mode)

    def _resample_linear(self, data_xyz: np.ndarray, *, mode: str = "bilinear", padding_mode: str = "zeros") -> np.ndarray:
        data_xyz = np.asarray(data_xyz)
        if data_xyz.shape != self.moving_shape:
            raise ValueError(f"Input shape {data_xyz.shape} does not match {self.moving_shape}.")
        src_zyx = np.transpose(data_xyz.astype(np.float32, copy=False), (2, 1, 0))
        src_t = torch.from_numpy(src_zyx).to(self.device, non_blocking=True)
        self._input[0, 0].copy_(src_t)
        out = F.grid_sample(
            self._input,
            self.grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=True,
        )
        out_zyx = out[0, 0].detach().cpu().numpy()
        return np.transpose(out_zyx, (2, 1, 0)).astype(np.float32, copy=False)

    @torch.inference_mode()
    def resample_support(self, data_xyz: np.ndarray) -> np.ndarray:
        """Resample fractional support using Linear/zero padding in every mode."""
        with self._lock:
            return self._resample_linear(data_xyz)

    @torch.inference_mode()
    def resample_coverage(self, source_support_xyz: np.ndarray) -> np.ndarray:
        """Diagnostic: return full-kernel target support as a boolean XYZ array.

        This conservative diagnostic requires every nonzero Lanczos tap to be
        acquired and in-bounds. Integer-axis delta weights avoid unnecessary
        erosion at exact voxel centers. Linear uses the
        legacy Linear >= 1-1e-5 coverage rule. Runtime uses resample_support
        plus a separately mapped anatomical mask instead of this erosion.
        """
        with self._lock:
            if self._lanczos is not None:
                return self._lanczos.resample_coverage(source_support_xyz)
            source = np.asarray(source_support_xyz)
            source = np.isfinite(source) & (source >= 1.0 - 1e-5)
            return self._resample_linear(source.astype(np.float32)) >= 1.0 - 1e-5

    def resample_nifti_to_array(self, nii_path: Path) -> np.ndarray:
        img = nib.load(str(nii_path))
        return self.resample_array(np.asanyarray(img.dataobj))

    def resample_nifti_to_nifti(self, in_path: Path, out_path: Path) -> Path:
        out_data = self.resample_nifti_to_array(in_path)
        nib.save(nib.Nifti1Image(out_data, self.fixed_affine), str(out_path))
        return out_path
