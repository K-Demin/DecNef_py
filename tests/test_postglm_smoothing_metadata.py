"""Changing smoothing order preserves only demonstrably equivalent references."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fmri_rt_preproc import reference_pool
from fmri_rt_preproc.processing_mask import (
    FINAL_NUISANCE_SOURCE, FINAL_PROCESSING_MASK_POLICY,
    FINAL_PROCESSING_MASK_POLICY_VERSION, FINAL_PROCESSING_ORDER,
    LEGACY_FINAL_PROCESSING_ORDER, canonical_final_processing_mask,
    canonical_upstream_processing, upstream_processing_spec,
)
from test_reference_pool import native_runs


def processing(order=FINAL_PROCESSING_ORDER):
    return {"policy": FINAL_PROCESSING_MASK_POLICY,
            "version": FINAL_PROCESSING_MASK_POLICY_VERSION,
            "nuisance_source": FINAL_NUISANCE_SOURCE,
            "processing_order": list(order), "interpolation": "linear"}


def metadata(order=FINAL_PROCESSING_ORDER, fwhm=0):
    return {"normalization": {"mode": "zscore", "reference": "fixed", "ref_volumes": 30},
            "coverage_policy": "separate_brain_processing_observed_support_v3",
            "processing_mask": processing(order), "smoothing_fwhm_mm": fwhm}


@pytest.mark.parametrize("zero", [0, 0., np.int64(0), np.float32(0)])
def test_only_explicit_zero_makes_old_order_equivalent_without_mutation(zero):
    old = processing(LEGACY_FINAL_PROCESSING_ORDER)
    before = deepcopy(old)
    actual = canonical_final_processing_mask(old, zero)
    assert actual["processing_order"] == list(FINAL_PROCESSING_ORDER)
    assert actual is not old
    assert old == before


@pytest.mark.parametrize("width", [None, False, True, "0", 6, -1, np.nan, np.inf])
def test_old_order_with_missing_smoothed_or_invalid_width_is_rejected(width):
    with pytest.raises(ValueError, match="explicit zero FWHM"):
        canonical_final_processing_mask(processing(LEGACY_FINAL_PROCESSING_ORDER), width)


@pytest.mark.parametrize("width", [None, 0, 6])
def test_current_order_is_retained_without_inferring_a_width(width):
    current = processing()
    assert canonical_final_processing_mask(current, width) == current


@pytest.mark.parametrize("change", ["unknown_order", "version", "source", "policy"])
def test_compatibility_does_not_override_other_processing_contracts(change):
    old = processing(LEGACY_FINAL_PROCESSING_ORDER)
    if change == "unknown_order":
        old["processing_order"] = ["denoising", "normalization", "smoothing"]
    elif change == "version":
        old["version"] = True
    elif change == "source":
        old["nuisance_source"] = "smoothed"
    else:
        old["policy"] = "unverified"
    with pytest.raises(ValueError):
        canonical_final_processing_mask(old, 0)


def test_verified_zero_width_run_can_prepare_new_pca_contract_but_old_contract_cannot():
    old = metadata(LEGACY_FINAL_PROCESSING_ORDER)
    before = deepcopy(old)
    canonical = upstream_processing_spec(old)
    assert canonical["processing_mask"]["processing_order"] == list(FINAL_PROCESSING_ORDER)
    assert old == before
    # A previously serialized PCA contract omitted FWHM. Its order alone
    # cannot prove whether the source data had zero spatial smoothing.
    old_contract = deepcopy(canonical)
    old_contract["processing_mask"]["processing_order"] = list(LEGACY_FINAL_PROCESSING_ORDER)
    with pytest.raises(ValueError, match="order/nuisance-source"):
        canonical_upstream_processing(old_contract)


@pytest.mark.parametrize("width", [None, 6])
def test_old_run_without_zero_width_cannot_prepare_pca_contract(width):
    old = metadata(LEGACY_FINAL_PROCESSING_ORDER, width)
    if width is None:
        old.pop("smoothing_fwhm_mm")
    with pytest.raises(ValueError, match="explicit zero FWHM"):
        upstream_processing_spec(old)


def test_pool_accepts_old_zero_runs_and_records_current_processing(native_runs):
    runs, output, kwargs = native_runs
    for folder in runs:
        path = folder / "session_metadata.json"
        info = json.loads(path.read_text())
        info["processing_mask"]["processing_order"] = list(LEGACY_FINAL_PROCESSING_ORDER)
        info["temporal_processing_order"] = "spatial_smooth_glm_norm_v1"
        path.write_text(json.dumps(info))
    stats = reference_pool.pool_native_references(runs, output, **kwargs)
    assert stats["n"] == 4
    saved = json.loads((output / "session_metadata.json").read_text())
    assert saved["processing_mask"]["processing_order"] == list(FINAL_PROCESSING_ORDER)
    assert saved["temporal_processing_order"] == kwargs["runtime"].TEMPORAL_PROCESSING_ORDER
    for folder in runs:
        original = json.loads((folder / "session_metadata.json").read_text())
        assert original["processing_mask"]["processing_order"] == list(LEGACY_FINAL_PROCESSING_ORDER)


@pytest.mark.parametrize("width", [None, 6])
def test_pool_rejects_old_nonzero_or_unrecorded_smoothing(native_runs, width):
    runs, output, kwargs = native_runs
    path = runs[0] / "session_metadata.json"
    info = json.loads(path.read_text())
    info["processing_mask"]["processing_order"] = list(LEGACY_FINAL_PROCESSING_ORDER)
    info["temporal_processing_order"] = "spatial_smooth_glm_norm_v1"
    if width is None:
        info.pop("smoothing_fwhm_mm")
    else:
        info["smoothing_fwhm_mm"] = width
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="explicit zero FWHM"):
        reference_pool.pool_native_references(runs, output, **kwargs)
    assert not output.exists()


@pytest.mark.parametrize("current,recorded,accepted", [
    (0, 0, True), (6, 0, False), (0, 6, False), (6, 6, False),
    (0, None, False), (0, False, False),
])
def test_scalar_old_order_compatibility_requires_both_widths_zero(
        native_runs, current, recorded, accepted):
    runs, _, kwargs = native_runs
    kwargs["settings"].smoothing_fwhm_mm = current
    folder = runs[0]
    path = folder / "session_metadata.json"
    info = json.loads(path.read_text())
    info["processing_mask"]["processing_order"] = list(LEGACY_FINAL_PROCESSING_ORDER)
    if recorded is None:
        info.pop("smoothing_fwhm_mm")
    else:
        info["smoothing_fwhm_mm"] = recorded
    path.write_text(json.dumps(info))
    load = kwargs["runtime"].load_reference_score_stats
    cfg = SimpleNamespace(day_root=Path("unused"))
    if accepted:
        actual = load(cfg, str(folder))
        assert (actual["mean"], actual["n"]) == (4., 2)
    else:
        with pytest.raises(ValueError, match="spatial masking/interpolation policy"):
            load(cfg, str(folder))
