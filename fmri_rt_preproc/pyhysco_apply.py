from __future__ import annotations

from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import torch

PYHYSCO_SRC = Path(__file__).resolve().parent / "PyHySCO-main" / "src"
if str(PYHYSCO_SRC) not in sys.path:
    sys.path.append(str(PYHYSCO_SRC))

from EPI_MRI.EPIMRIDistortionCorrection import DataObject, EPIMRIDistortionCorrection
from EPI_MRI.utils import m_plus


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
    return field.permute(internal_perm).contiguous().view(-1, 1)


def apply_pyhysco_fieldmap_to_4d(
    epi_4d: Path,
    fieldmap_path: Path,
    out_path: Path,
    phase_encoding_direction: int = 1,
    polarity: int = 1,
    device: str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Path:
    """
    Apply a pre-estimated PyHySCO fieldmap to every volume of a 4D EPI.

    Parameters
    ----------
    epi_4d:
        Input distorted 4D EPI.
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
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    epi_img = nib.load(str(epi_4d))
    epi = np.asarray(epi_img.dataobj)
    if epi.ndim != 4:
        raise ValueError(f"Expected 4D EPI, got shape {epi.shape}")

    corrected = np.zeros_like(epi, dtype=np.float32)

    for t in range(epi.shape[-1]):
        vol_path = epi_4d.parent / f".__pyhysco_tmp_vol_{t:04d}.nii.gz"
        nib.save(nib.Nifti1Image(epi[..., t], epi_img.affine, epi_img.header), str(vol_path))

        data_obj = DataObject(
            str(vol_path),
            str(vol_path),
            phase_encoding_direction=phase_encoding_direction,
            do_normalize=False,
            dtype=dtype,
            device=device,
        )
        corr_obj = EPIMRIDistortionCorrection(data_obj, alpha=1.0, beta=0.0)
        b = _load_internal_fieldmap(fieldmap_path, data_obj)
        if polarity < 0:
            b = -b

        corr_vol, _, _, _ = corr_obj.mp_transform(corr_obj.dataObj.I1, b, do_derivative=False)
        corr_vol = corr_vol.reshape(tuple(corr_obj.dataObj.m)).permute(corr_obj.dataObj.p)
        corrected[..., t] = corr_vol.detach().cpu().numpy().astype(np.float32)

        vol_path.unlink(missing_ok=True)

    nib.save(nib.Nifti1Image(corrected, epi_img.affine, epi_img.header), str(out_path))
    return out_path


def apply_pyhysco_fieldmap_to_3d(
    epi_3d: Path,
    fieldmap_path: Path,
    out_path: Path,
    phase_encoding_direction: int = 1,
    polarity: int = 1,
    device: str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Path:
    """
    Apply a pre-estimated PyHySCO fieldmap to a single 3D EPI volume.
    """
    img = nib.load(str(epi_3d))
    data = np.asarray(img.dataobj)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D EPI, got shape {data.shape}")

    tmp_4d = epi_3d.parent / f".__pyhysco_tmp_{epi_3d.stem}_4d.nii.gz"
    tmp_uw_4d = epi_3d.parent / f".__pyhysco_tmp_{epi_3d.stem}_uw4d.nii.gz"
    try:
        nib.save(
            nib.Nifti1Image(data[..., np.newaxis], img.affine, img.header),
            str(tmp_4d),
        )
        apply_pyhysco_fieldmap_to_4d(
            epi_4d=tmp_4d,
            fieldmap_path=fieldmap_path,
            out_path=tmp_uw_4d,
            phase_encoding_direction=phase_encoding_direction,
            polarity=polarity,
            device=device,
            dtype=dtype,
        )
        uw_img = nib.load(str(tmp_uw_4d))
        uw_data = np.asarray(uw_img.dataobj)[..., 0]
        nib.save(nib.Nifti1Image(uw_data, img.affine, img.header), str(out_path))
    finally:
        tmp_4d.unlink(missing_ok=True)
        tmp_uw_4d.unlink(missing_ok=True)
    return out_path
