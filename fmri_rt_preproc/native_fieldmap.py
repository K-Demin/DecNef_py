"""Calibration-local fields for raw BOLD unwarping (no AP/PA registration)."""
import json
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np

PIPELINE_ORDER = "unwarp_then_mc_v1"
NATIVE_DIR = "native_unwarp_v1"
# DecNef local modification (3255b60): prepared references and transforms
# depend on the estimator recipe used to generate their calibration field.
ESTIMATOR_RECIPE = "pyhysco_gn_positive_jacobians_v2"
_FRESH_CALIBRATION = (
    "Use a fresh preparation directory to re-estimate the AP/PA calibration "
    "and rebuild its dependent EPI references and transforms; existing files were not overwritten."
)


def source_identity(path):
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def find_source(folder, name):
    for suffix in (".nii", ".nii.gz"):
        path = Path(folder) / (name + suffix)
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing raw {name}.nii[.gz] in {folder}; derived AP/PA means are not inputs.")


def sidecar(path):
    path = Path(path)
    name = path.name.removesuffix(".gz").removesuffix(".nii") + ".json"
    meta = path.with_name(name)
    return json.loads(meta.read_text()) if meta.exists() else {}


def calibration_spec(ap, pa):
    ap_meta, pa_meta = sidecar(ap), sidecar(pa)
    pe = [m.get("PhaseEncodingDirection") for m in (ap_meta, pa_meta)]
    known = [v for v in pe if v]
    if any(v not in ("i", "i-", "j", "j-", "k", "k-") for v in known):
        raise ValueError(f"Invalid PhaseEncodingDirection: {pe}")
    if len(known) == 2 and (pe[0][0] != pe[1][0] or pe[0] == pe[1]):
        raise ValueError("AP and PA must have opposite signs on the same voxel axis.")
    readouts = [m.get("TotalReadoutTime") for m in (ap_meta, pa_meta)]
    if all(v is not None for v in readouts) and not np.isclose(*readouts, rtol=1e-4, atol=1e-6):
        raise ValueError("This PyHySCO path requires matching AP/PA readout times.")
    # Existing PA protocol used dimension 2. Metadata overrides this legacy default.
    axis = "ijk".index(known[0][0]) + 1 if known else 2
    return {"order": PIPELINE_ORDER, "motion_correction": "none",
            "estimator_recipe": ESTIMATOR_RECIPE,
            "ap": source_identity(ap), "pa": source_identity(pa),
            "phase_encoding_axis": axis, "ap_pe": pe[0], "pa_pe": pe[1],
            "readout_time": next((v for v in readouts if v is not None), None)}


def require_same_grid(image, reference):
    if image.shape[:3] != reference.shape[:3] or not np.allclose(image.affine, reference.affine, atol=1e-4, rtol=0):
        raise ValueError("Raw BOLD/calibration/reference grids differ; header relabeling is not registration.")


def save_native_mean(source, out):
    image = nib.load(str(source))
    data = np.asarray(image.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data.mean(axis=3)
    elif data.ndim != 3:
        raise ValueError(f"Expected 3D/4D calibration: {source}")
    nib.save(nib.Nifti1Image(data, image.affine), str(out))


def native_paths(pair_dir):
    folder = Path(pair_dir) / NATIVE_DIR
    return folder, folder / "pyhysco_native-EstFieldMap.nii", folder / "calibration.json"


def _file_key(path):
    identity = source_identity(path)
    return identity["path"], identity["size"], identity["mtime_ns"]


# DecNef local modification (3255b60): reuse successful saved-field checks
# for the same field, AP/PA prototypes and PE axis. A changed file identity
# causes a fresh check the next time the calibration is loaded.
@lru_cache(maxsize=32)
def _validate_field_cached(field_key, ap_key, pa_key, phase_encoding_axis):
    """Validate once per file identity, including both PE polarities.

    The field header is intentionally not physical image geometry. PyHySCO
    saves an array on a staggered grid; its spacing comes from the AP/PA images.
    Float64 checks the saved field mathematically. The runtime applier also
    checks its actual float32 derivative when its sampling grid is constructed.
    """
    ap, pa = nib.load(ap_key[0]), nib.load(pa_key[0])
    require_same_grid(pa, ap)
    axis = int(phase_encoding_axis) - 1
    if len(ap.shape) != 3 or axis not in (0, 1, 2):
        raise ValueError("Native calibration requires a 3D prototype and a valid PE axis.")
    field = np.asarray(nib.load(field_key[0]).dataobj, dtype=np.float64)
    shape = list(ap.shape)
    shape[axis] += 1
    if field.shape != tuple(shape):
        raise ValueError("Native PyHySCO field has the wrong staggered shape. " + _FRESH_CALIBRATION)
    spacing = float(np.linalg.norm(ap.affine[:3, axis]))
    if not np.isfinite(spacing) or spacing <= 0 or not np.isfinite(field).all():
        raise ValueError("Native PyHySCO field or its PE spacing is nonfinite/invalid. " + _FRESH_CALIBRATION)
    # Adjacent field samples bracket one EPI cell. This difference uses the
    # physical PE spacing from the AP prototype and is evaluated in float64.
    derivative = np.diff(field, axis=axis) / spacing
    if not np.isfinite(derivative).all() or np.any(np.abs(derivative) >= 1):
        raise ValueError(
            "Native PyHySCO field has nonfinite or nonpositive AP/PA Jacobians "
            "(both 1+Db and 1-Db must be strictly positive). " + _FRESH_CALIBRATION)
    return {"min_ap_jacobian": float(np.min(1 + derivative)),
            "min_pa_jacobian": float(np.min(1 - derivative))}


def validate_native_field(pair_dir, phase_encoding_axis):
    """Cached whole-field validity check; no brain mask or erosion is involved."""
    folder, field, _ = native_paths(pair_dir)
    return _validate_field_cached(_file_key(field), _file_key(folder / "AP_mean.nii"),
                                  _file_key(folder / "PA_mean.nii"), int(phase_encoding_axis))


def load_calibration(pair_dir):
    folder, field, manifest = native_paths(pair_dir)
    spec = json.loads(manifest.read_text())
    expected = calibration_spec(find_source(pair_dir, "AP"), find_source(pair_dir, "PA"))
    if spec != expected or not all(p.exists() for p in (field, folder / "AP_mean.nii", folder / "PA_mean.nii")):
        raise ValueError("Native calibration estimator recipe/inputs are stale or incomplete. " + _FRESH_CALIBRATION)
    validate_native_field(pair_dir, spec["phase_encoding_axis"])
    return spec


def validate_bold(image, pair_dir, polarity, source_path=None):
    spec = load_calibration(pair_dir)
    folder, _, _ = native_paths(pair_dir)
    require_same_grid(image, nib.load(str(folder / "AP_mean.nii")))
    if polarity not in ("AP", "PA"):
        raise ValueError("epi_phase_encoding must be AP or PA")
    meta = sidecar(source_path) if source_path else {}
    expected_pe = spec["ap_pe" if polarity == "AP" else "pa_pe"]
    if meta.get("PhaseEncodingDirection") and expected_pe and meta["PhaseEncodingDirection"] != expected_pe:
        raise ValueError("BOLD phase encoding does not match the selected calibration polarity.")
    if meta.get("TotalReadoutTime") is not None and spec["readout_time"] is not None:
        if not np.isclose(meta["TotalReadoutTime"], spec["readout_time"], rtol=1e-4, atol=1e-6):
            raise ValueError("BOLD/calibration readout times differ; displacement scaling is not implemented.")
    return spec
