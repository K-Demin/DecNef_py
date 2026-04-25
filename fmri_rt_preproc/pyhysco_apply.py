from __future__ import annotations

from pathlib import Path
import logging
import sys
from uuid import uuid4

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

PYHYSCO_SRC = Path(__file__).resolve().parent / "PyHySCO-main" / "src"
if str(PYHYSCO_SRC) not in sys.path:
    sys.path.append(str(PYHYSCO_SRC))

from EPI_MRI.EPIMRIDistortionCorrection import DataObject, EPIMRIDistortionCorrection
from EPI_MRI.utils import m_plus

log = logging.getLogger(__name__)


def _inverse_permutation(perm: list[int]) -> list[int]:
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def _load_internal_fieldmap(fieldmap_path: Path, data_obj: DataObject) -> torch.Tensor:
    """
    Load a PyHySCO fieldmap saved with original orientation and convert it to
    PyHySCO's internal orientation.
    """
    field_img = nib.load(str(fieldmap_path))
    field = torch.tensor(np.asarray(field_img.dataobj), dtype=data_obj.dtype, device=data_obj.device)

    expected_external_shape = tuple(m_plus(data_obj.m).cpu().numpy()[data_obj.p])
    if tuple(field.shape) != expected_external_shape:
        raise ValueError(
            "PyHySCO fieldmap shape does not match target EPI geometry. "
            f"Expected {expected_external_shape}, got {tuple(field.shape)}."
        )

    internal_perm = _inverse_permutation(data_obj.p)
    return field.permute(internal_perm).contiguous()


def _to_internal_volume(
    volume_xyz: np.ndarray,
    data_obj: DataObject,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert an external-orientation volume to PyHySCO's internal orientation."""
    internal_perm = _inverse_permutation(data_obj.p)
    vol_t = torch.as_tensor(volume_xyz, dtype=dtype, device=data_obj.device)
    return vol_t.permute(internal_perm).contiguous()


def _select_device_like_gpu_resampler(device: str | None) -> str:
    """
    Mirror gpu_ants_like_resampler device-selection behavior.

    - default device string is "cuda"
    - if explicit/default device is "cuda" but CUDA is unavailable, fall back to "cpu"
    - otherwise pass through device as-is
    """
    requested = "cuda" if device is None else str(device)
    return requested if (requested != "cuda" or torch.cuda.is_available()) else "cpu"


class PreloadedPyHyscoApplier:
    """
    Reusable PyHySCO fieldmap applier for fixed geometry.

    This avoids reloading/initializing DataObject and EPIMRIDistortionCorrection for
    every single volume.
    """

    def __init__(
        self,
        prototype_vol_path: Path,
        fieldmap_path: Path,
        phase_encoding_direction: int = 1,
        polarity: int = 1,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.data_obj = DataObject(
            str(prototype_vol_path),
            str(prototype_vol_path),
            phase_encoding_direction=phase_encoding_direction,
            do_normalize=False,
            dtype=dtype,
            device=device,
        )
        self.corr_obj = EPIMRIDistortionCorrection(self.data_obj, alpha=1.0, beta=0.0)

        b = _load_internal_fieldmap(fieldmap_path, self.data_obj)
        if polarity < 0:
            b = -b
        self.fieldmap_internal = b.contiguous()
        self.device = str(self.data_obj.device)
        self._init_fast_apply_cache()

    @staticmethod
    def _idx_to_norm(idx: torch.Tensor, size: int) -> torch.Tensor:
        if size <= 1:
            return torch.zeros_like(idx)
        return 2.0 * idx / float(size - 1) - 1.0

    def _init_fast_apply_cache(self) -> None:
        """
        Precompute a fixed sampling grid and Jacobian for fast per-volume apply.

        For fixed fieldmap b:
            TI(xc) = I(xc + bc) * (1 + dbc)
        where bc, dbc, and therefore xc+bc/Jac are constant across incoming volumes.
        """
        m = tuple(int(v) for v in self.data_obj.m.tolist())
        if len(m) != 3:
            raise ValueError(f"Fast fieldmap apply currently expects 3D volumes, got shape rank={len(m)}.")

        # Precompute deformation and Jacobian terms from fixed fieldmap.
        bc = self.corr_obj.A.mat_mul(self.fieldmap_internal)
        dbc = self.corr_obj.D.mat_mul(self.fieldmap_internal)
        jac = (1.0 + dbc).reshape(m).contiguous()
        xt = (self.corr_obj.xc + bc).reshape(m)

        # Convert physical coordinates to interpolation indices along distortion dim.
        x_idx = (xt - self.data_obj.omega[-2]) / self.data_obj.h[-1] - 0.5
        x_idx = torch.clamp(x_idx, 0, m[-1] - 1)

        # Identity coordinates for non-distortion axes.
        z_idx = torch.arange(m[0], device=self.data_obj.device, dtype=self.data_obj.dtype).view(m[0], 1, 1).expand(m)
        y_idx = torch.arange(m[1], device=self.data_obj.device, dtype=self.data_obj.dtype).view(1, m[1], 1).expand(m)

        x_norm = self._idx_to_norm(x_idx, m[2]).float()
        y_norm = self._idx_to_norm(y_idx, m[1]).float()
        z_norm = self._idx_to_norm(z_idx, m[0]).float()

        # grid_sample expects [x, y, z] components in the last dimension.
        self.grid = torch.stack([x_norm, y_norm, z_norm], dim=-1).unsqueeze(0).contiguous()
        self.jac = jac.float()

    def apply_volume(self, volume_xyz: np.ndarray) -> np.ndarray:
        """Apply cached fieldmap/correction objects to one 3D volume."""
        vol_internal = _to_internal_volume(volume_xyz, self.data_obj, dtype=self.data_obj.dtype)
        src = vol_internal.float().unsqueeze(0).unsqueeze(0)
        sampled = F.grid_sample(
            src,
            self.grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        corr_vol = sampled[0, 0] * self.jac
        corr_vol = corr_vol.permute(self.data_obj.p)
        return corr_vol.detach().cpu().numpy().astype(np.float32, copy=False)


def apply_pyhysco_fieldmap(
    epi_path: Path,
    fieldmap_path: Path,
    out_path: Path,
    phase_encoding_direction: int = 1,
    polarity: int = 1,
    device: str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Path:
    """
    Apply a pre-estimated PyHySCO fieldmap to a 3D or 4D EPI.

    Parameters
    ----------
    epi_path:
        Input distorted 3D or 4D EPI.
    fieldmap_path:
        Fieldmap saved by PyHySCO (`*-EstFieldMap.nii.gz`).
    out_path:
        Output corrected 4D EPI.
    phase_encoding_direction:
        PyHySCO phase encoding axis index (1-based).
    polarity:
        +1 applies correction in the same direction as I1, -1 as I2.
    device:
        Torch device; defaults to cuda if available.
    dtype:
        Torch dtype.
    """
    device = _select_device_like_gpu_resampler(device)

    epi_img = nib.load(str(epi_path))
    epi = np.asarray(epi_img.dataobj)
    if epi.ndim == 3:
        epi_work = epi[..., np.newaxis]
        squeeze_output = True
    elif epi.ndim == 4:
        epi_work = epi
        squeeze_output = False
    else:
        raise ValueError(f"Expected 3D or 4D EPI, got shape {epi.shape}")

    corrected = np.zeros_like(epi_work, dtype=np.float32)
    run_tag = uuid4().hex
    proto_path = epi_path.parent / f".__pyhysco_tmp_{epi_path.stem}_{run_tag}_proto.nii.gz"
    try:
        nib.save(nib.Nifti1Image(epi_work[..., 0], epi_img.affine, epi_img.header), str(proto_path))
        applier = PreloadedPyHyscoApplier(
            prototype_vol_path=proto_path,
            fieldmap_path=fieldmap_path,
            phase_encoding_direction=phase_encoding_direction,
            polarity=polarity,
            device=device,
            dtype=dtype,
        )
        log.info(
            "Applying PyHySCO fieldmap on device=%s (cuda_available=%s, mps_available=%s)",
            applier.device,
            torch.cuda.is_available(),
            torch.backends.mps.is_available(),
        )

        with torch.inference_mode():
            for t in range(epi_work.shape[-1]):
                corrected[..., t] = applier.apply_volume(epi_work[..., t])
    finally:
        proto_path.unlink(missing_ok=True)

    out_data = corrected[..., 0] if squeeze_output else corrected
    nib.save(nib.Nifti1Image(out_data, epi_img.affine, epi_img.header), str(out_path))
    return out_path


def apply_pyhysco_fieldmap_to_4d(*args, **kwargs) -> Path:
    """Backward-compatible alias."""
    return apply_pyhysco_fieldmap(*args, **kwargs)
