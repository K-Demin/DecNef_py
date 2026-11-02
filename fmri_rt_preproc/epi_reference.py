"""One-time N4 enhancement of EPI registration copies, never live BOLD data."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from .native_fieldmap import require_same_grid
from .offline_assets import file_identity


EPI_REFERENCE_N4_RECIPE = "masked_n4_s2_b200_50x50x30x20_v1"
_FRESH = "Use a fresh preparation directory for EPI reference N4."


def epi_reference_metadata_path(output):
    output = Path(output)
    return output.with_name(output.name.removesuffix(".gz").removesuffix(".nii") + ".json")


def _inputs(epi, mask, output):
    epi, mask, output = map(Path, (epi, mask, output))
    if output.resolve() in {epi.resolve(), mask.resolve()} or (
        output.exists() and any(output.samefile(p) for p in (epi, mask))
    ):
        raise ValueError("N4 registration output must be separate from the EPI and mask inputs.")
    image, mask_image = nib.load(str(epi)), nib.load(str(mask))
    values, selected = np.asanyarray(image.dataobj), np.asanyarray(mask_image.dataobj)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("EPI registration reference must be a finite 3D image.")
    if (selected.ndim != 3 or not np.isfinite(selected).all()
            or not np.all((selected == 0) | (selected == 1)) or not selected.any()):
        raise ValueError("EPI N4 mask must be a finite, nonempty, binary 3D image.")
    require_same_grid(mask_image, image)
    if not np.any(values[selected.astype(bool)] > 0):
        raise ValueError("EPI N4 mask contains no positive reference signal.")
    return image, {"version": 1, "recipe": EPI_REFERENCE_N4_RECIPE,
                   "source": file_identity(epi), "mask": file_identity(mask)}


def _check_output(output, reference):
    image = nib.load(str(output))
    values = np.asanyarray(image.dataobj)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("N4 registration output must be a finite 3D image.")
    require_same_grid(image, reference)
    # Detach from an uncompressed NIfTI memmap before rewriting its header.
    return np.array(values, copy=True)


def validate_epi_registration_reference(epi, mask, output):
    """Validate an existing N4 copy and its provenance without running N4."""
    output = Path(output)
    reference, expected = _inputs(epi, mask, output)
    sidecar = epi_reference_metadata_path(output)
    if not output.is_file() or not sidecar.is_file():
        raise ValueError(f"Incomplete EPI N4 reference cache. {_FRESH}")
    try:
        saved = json.loads(sidecar.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ValueError(f"Unreadable EPI N4 reference provenance. {_FRESH}") from exc
    expected["output"] = file_identity(output)
    if saved != expected:
        raise ValueError(f"EPI N4 inputs, recipe or output changed. {_FRESH}")
    _check_output(output, reference)
    return saved


def prepare_epi_registration_reference(epi, mask, output, run_command):
    """Create or reuse a masked N4 fitting image on the unchanged EPI grid.

    The validated preparation recipe uses shrink=2, spline spacing=200 mm,
    iterations=50x50x30x20 and tolerance=1e-7, without intensity rescaling.
    The original mean remains the motion-correction/intensity reference.
    A partial or stale cache is never silently overwritten.
    """
    epi, mask, output = map(Path, (epi, mask, output))
    sidecar = epi_reference_metadata_path(output)
    if output.exists() or sidecar.exists():
        validate_epi_registration_reference(epi, mask, output)
        return output
    reference, record = _inputs(epi, mask, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    run_command(["N4BiasFieldCorrection", "-d", "3", "-i", str(epi),
                 "-x", str(mask), "-s", "2", "-b", "[200]",
                 "-c", "[50x50x30x20,1e-7]", "-r", "0", "-o", str(output)])
    if not output.is_file() or not output.stat().st_size:
        raise RuntimeError("N4 did not produce the EPI registration reference.")
    values = _check_output(output, reference)
    # N4 changes intensities only. Keep both original spatial forms/codes, and
    # store floating-point data rather than quantizing to the source datatype.
    corrected = reference.__class__(np.asarray(values, dtype=np.float32), reference.affine,
                                    reference.header.copy(), extra=reference.extra.copy())
    corrected.set_data_dtype(np.float32)
    corrected.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    corrected.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    nib.save(corrected, str(output))
    record["output"] = file_identity(output)
    temporary = sidecar.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(sidecar)
    return output
