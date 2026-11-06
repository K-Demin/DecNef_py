"""Processing-mask provenance and the historical source-grid margin helper.

Current response processing uses the final analysis grid after spatial mapping;
denoising precedes optional smoothing. Native nuisance sampling stays unsmoothed. The cubic
margin helper retains its original behavior for callers requiring native-grid
support; it does not define the current response GLM/normalization mask.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter


PROCESSING_MASK_POLICY = "source_grid_cubic_halo"
PROCESSING_MASK_POLICY_VERSION = 1
FINAL_PROCESSING_MASK_POLICY = "final_analysis_grid"
FINAL_PROCESSING_MASK_POLICY_VERSION = 1
FINAL_NUISANCE_SOURCE = "native_unsmoothed_mc"
FINAL_PROCESSING_ORDER = (
    "spatial_transform", "denoising", "smoothing", "normalization",
)
LEGACY_FINAL_PROCESSING_ORDER = (
    "spatial_transform", "smoothing", "denoising", "normalization",
)


def canonical_final_processing_mask(mask: dict, smoothing_fwhm_mm=None) -> dict:
    """Validate final-grid provenance and retain equivalent unsmoothed runs.

    The former smoothing-before-denoising order is equivalent only when the
    run explicitly records zero Gaussian FWHM. Missing widths, booleans and
    nonfinite values cannot establish that. Never modify the saved metadata.
    Callers comparing runs must still compare their actual FWHM settings.
    """
    if not isinstance(mask, dict):
        raise ValueError("Missing final-grid processing metadata.")
    order = mask.get("processing_order")
    if (mask.get("policy") != FINAL_PROCESSING_MASK_POLICY
            or not isinstance(mask.get("version"), int)
            or mask.get("version") != FINAL_PROCESSING_MASK_POLICY_VERSION
            or isinstance(mask.get("version"), bool)
            or mask.get("nuisance_source") != FINAL_NUISANCE_SOURCE
            or not isinstance(order, (tuple, list))):
        raise ValueError("Invalid final-grid order/nuisance-source policy.")
    if tuple(order) == LEGACY_FINAL_PROCESSING_ORDER:
        explicit_zero = (
            isinstance(smoothing_fwhm_mm, (int, float, np.integer, np.floating))
            and not isinstance(smoothing_fwhm_mm, (bool, np.bool_))
            and np.isfinite(smoothing_fwhm_mm)
            and smoothing_fwhm_mm == 0
        )
        if not explicit_zero:
            raise ValueError(
                "Smoothing-before-denoising references require explicit zero FWHM "
                "for reuse; reprocess smoothed or unverifiable references."
            )
    elif tuple(order) != FINAL_PROCESSING_ORDER:
        raise ValueError("Unsupported final-grid processing order.")
    result = dict(mask)
    result["processing_order"] = list(FINAL_PROCESSING_ORDER)
    return result


def canonical_upstream_processing(spec: dict) -> dict:
    """Keep comparable processing semantics, excluding acquisition identities."""
    if not isinstance(spec, dict) or spec.get("version") != 1:
        raise ValueError("Missing or unsupported upstream processing metadata.")
    stage = spec.get("input_stage")
    normalization = spec.get("normalization")
    if stage not in {"normalized", "raw_mc"}:
        raise ValueError("Upstream processing must identify normalized or raw_mc input.")
    if stage == "normalized" and (not isinstance(normalization, dict) or not normalization):
        raise ValueError("Upstream processing lacks voxel normalization metadata.")
    if stage == "raw_mc" and normalization is not None:
        raise ValueError("Raw MC inputs must not claim voxel normalization.")
    coverage = spec.get("coverage_policy")
    mask = spec.get("processing_mask")
    if not isinstance(coverage, str) or not coverage or not isinstance(mask, dict):
        raise ValueError("Upstream processing lacks coverage/mask policy metadata.")
    policy, version, interpolation = (mask.get(key) for key in ("policy", "version", "interpolation"))
    if (not isinstance(policy, str) or not policy or not isinstance(version, int)
            or isinstance(version, bool) or version < 1
            or interpolation not in {"linear", "lanczos"}):
        raise ValueError("Upstream processing lacks a valid mask policy/version/interpolation.")
    canonical_mask = {"policy": policy, "version": version, "interpolation": interpolation}
    if policy == FINAL_PROCESSING_MASK_POLICY:
        order = mask.get("processing_order")
        if (version != FINAL_PROCESSING_MASK_POLICY_VERSION
                or not isinstance(order, (tuple, list))
                or tuple(order) != FINAL_PROCESSING_ORDER
                or mask.get("nuisance_source") != FINAL_NUISANCE_SOURCE):
            raise ValueError("Final-grid upstream processing lacks its supported order/nuisance-source policy.")
        canonical_mask.update(processing_order=list(FINAL_PROCESSING_ORDER),
                              nuisance_source=FINAL_NUISANCE_SOURCE)
    return {"version": 1, "input_stage": stage,
            "normalization": dict(normalization) if normalization is not None else None,
            "coverage_policy": coverage,
            "processing_mask": canonical_mask}


def upstream_processing_spec(metadata: dict, input_stage: str = "normalized") -> dict | None:
    """Capture verified modern-run semantics; None means genuinely legacy input.

    A new normalization declaration without its spatial policy is incomplete,
    not a legacy run. Source mask paths/hashes, grid and halo width intentionally
    stay out of this contract because RS and NF acquisitions can differ.
    """
    if "normalization" not in metadata:
        return None
    if not isinstance(metadata["normalization"], dict) or not metadata["normalization"]:
        raise ValueError("Run declares normalization but does not record its specification.")
    mask = metadata.get("processing_mask")
    if isinstance(mask, dict) and mask.get("policy") == FINAL_PROCESSING_MASK_POLICY:
        mask = canonical_final_processing_mask(mask, metadata.get("smoothing_fwhm_mm"))
    return canonical_upstream_processing({
        "version": 1, "input_stage": input_stage,
        "normalization": metadata["normalization"] if input_stage == "normalized" else None,
        "coverage_policy": metadata.get("coverage_policy"),
        "processing_mask": mask,
    })


def processing_margin_vox(interpolation: str, spatial_resampling: bool = True) -> int:
    """Return the conservative source-grid margin for the selected kernel.

    Lanczos-3 uses ``floor(c)-2 .. floor(c)+3`` along each axis; trilinear
    interpolation uses ``floor(c) .. floor(c)+1``. Relative to a nearest-neighbor
    source centre, these fit in cubic radii three and one, respectively.
    """
    name = str(interpolation).strip().lower()
    if name not in {"linear", "lanczos"}:
        raise ValueError("Processing-mask interpolation must be 'linear' or 'lanczos'.")
    if not isinstance(spatial_resampling, (bool, np.bool_)):
        raise ValueError("spatial_resampling must be boolean.")
    if not spatial_resampling:
        return 0
    return 3 if name == "lanczos" else 1


def make_processing_mask(brain_mask, interpolation: str = "linear",
                         spatial_resampling: bool = True) -> np.ndarray:
    """Return a separate boolean mask containing a cubic interpolation margin.

    ``brain_mask`` must be a nonempty, finite 3D binary array on the source EPI
    grid. The input is never changed. Padding is clipped to the existing array;
    selecting a voxel here does not assert it was acquired or temporally valid.

    This margin contains every in-array kernel contributor for target points
    whose nearest-neighbor source centre belongs to ``brain_mask``. It does not
    repair an anatomical mask that omits desired target brain locations.
    """
    margin = processing_margin_vox(interpolation, spatial_resampling)
    values = np.asarray(brain_mask)
    if values.ndim != 3 or any(size == 0 for size in values.shape):
        raise ValueError("Brain mask must be a nonempty 3D array.")
    if (values.dtype.kind not in "biuf" or not np.isfinite(values).all()
            or not np.all((values == 0) | (values == 1))):
        raise ValueError("Brain mask must contain only finite binary values (0/1).")
    mask = values.astype(bool, copy=True)
    if not mask.any():
        raise ValueError("Brain mask must contain at least one brain voxel.")
    if margin == 0:
        return mask
    return maximum_filter(mask, size=2 * margin + 1, mode="constant", cval=False)
