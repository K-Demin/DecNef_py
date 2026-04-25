from __future__ import annotations

import subprocess
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


def make_voxel_coordinate_images(moving_ref: Path, out_dir: Path) -> dict[str, Path]:
    """
    Create scalar NIfTI images on the moving grid whose values are voxel x/y/z indices.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img = nib.load(str(moving_ref))
    shape = img.shape[:3]

    x = np.arange(shape[0], dtype=np.float32)[:, None, None]
    y = np.arange(shape[1], dtype=np.float32)[None, :, None]
    z = np.arange(shape[2], dtype=np.float32)[None, None, :]
    xx = np.broadcast_to(x, shape).astype(np.float32)
    yy = np.broadcast_to(y, shape).astype(np.float32)
    zz = np.broadcast_to(z, shape).astype(np.float32)

    paths = {
        "x": out_dir / "coord_x_epi_vox.nii",
        "y": out_dir / "coord_y_epi_vox.nii",
        "z": out_dir / "coord_z_epi_vox.nii",
    }
    nib.save(nib.Nifti1Image(xx, img.affine, img.header), str(paths["x"]))
    nib.save(nib.Nifti1Image(yy, img.affine, img.header), str(paths["y"]))
    nib.save(nib.Nifti1Image(zz, img.affine, img.header), str(paths["z"]))
    return paths


def precompute_sampling_grid(
    moving_ref: Path,
    fixed_ref: Path,
    transforms: Sequence[Path],
    out_dir: Path,
    grid_name: str = "sampling_grid.npz",
) -> Path:
    """
    Build a fixed-grid lookup for PyTorch grid_sample that approximates ANTs pull-resampling.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    coord_paths = make_voxel_coordinate_images(moving_ref, out_dir)

    warped_paths: dict[str, Path] = {}
    for axis, coord_img in coord_paths.items():
        warped_img = out_dir / f"coord_{axis}_in_fixed.nii"
        if not warped_img.exists():
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

    moving_img = nib.load(str(moving_ref))
    fixed_img = nib.load(str(fixed_ref))
    moving_shape = np.array(moving_img.shape[:3], dtype=np.float32)
    cx = np.asanyarray(nib.load(str(warped_paths["x"])).dataobj).astype(np.float32)
    cy = np.asanyarray(nib.load(str(warped_paths["y"])).dataobj).astype(np.float32)
    cz = np.asanyarray(nib.load(str(warped_paths["z"])).dataobj).astype(np.float32)

    gx = 2.0 * cx / max(float(moving_shape[0] - 1), 1.0) - 1.0
    gy = 2.0 * cy / max(float(moving_shape[1] - 1), 1.0) - 1.0
    gz = 2.0 * cz / max(float(moving_shape[2] - 1), 1.0) - 1.0

    # nibabel uses [X,Y,Z], grid_sample expects [Z,Y,X] layout and [x,y,z] components.
    grid_xyz = np.stack([gx, gy, gz], axis=-1)
    grid_zyx = np.transpose(grid_xyz, (2, 1, 0, 3))
    grid = torch.from_numpy(grid_zyx[None, ...]).float()

    grid_path = out_dir / grid_name
    np.savez_compressed(
        str(grid_path),
        grid=grid.numpy(),
        moving_shape=np.asarray(moving_img.shape[:3], dtype=np.int32),
        fixed_shape=np.asarray(fixed_img.shape[:3], dtype=np.int32),
        fixed_affine=fixed_img.affine.astype(np.float64),
    )
    return grid_path


class GpuAntsLikeResampler:
    def __init__(
        self,
        grid_path: Path,
        device: str = "cuda",
        mode: str = "bilinear",
        padding_mode: str = "zeros",
    ):
        payload = self._load_payload(grid_path)
        selected_device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.device = torch.device(selected_device)
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

    @staticmethod
    def _load_payload(grid_path: Path) -> dict:
        if grid_path.suffix == ".npz":
            with np.load(str(grid_path)) as npz:
                return {
                    "grid": torch.from_numpy(npz["grid"]).float(),
                    "moving_shape": tuple(int(v) for v in npz["moving_shape"].tolist()),
                    "fixed_shape": tuple(int(v) for v in npz["fixed_shape"].tolist()),
                    "fixed_affine": np.asarray(npz["fixed_affine"], dtype=np.float64),
                }

        # Backward compatibility with older .pt payloads.
        payload = _torch_load_compat(grid_path)
        payload["fixed_affine"] = np.asarray(payload["fixed_affine"], dtype=np.float64)
        return payload

    @torch.inference_mode()
    def resample_array(self, data_xyz: np.ndarray) -> np.ndarray:
        if data_xyz.shape[:3] != self.moving_shape:
            raise ValueError(f"Input shape {data_xyz.shape[:3]} does not match {self.moving_shape}.")
        src_zyx = np.transpose(data_xyz.astype(np.float32, copy=False), (2, 1, 0))
        src_t = torch.from_numpy(src_zyx).to(self.device, non_blocking=True)
        self._input[0, 0].copy_(src_t)
        out = F.grid_sample(
            self._input,
            self.grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=True,
        )
        out_zyx = out[0, 0].detach().cpu().numpy()
        return np.transpose(out_zyx, (2, 1, 0)).astype(np.float32, copy=False)

    def resample_nifti_to_array(self, nii_path: Path) -> np.ndarray:
        img = nib.load(str(nii_path))
        return self.resample_array(np.asanyarray(img.dataobj))

    def resample_nifti_to_nifti(self, in_path: Path, out_path: Path) -> Path:
        out_data = self.resample_nifti_to_array(in_path)
        nib.save(nib.Nifti1Image(out_data, self.fixed_affine), str(out_path))
        return out_path
