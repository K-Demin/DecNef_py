"""Pool completed native-pipeline score logs without changing their voxel units.

Each source retains its ordinary per-volume outputs. The pooled directory is a
normal ``--rs`` reference: it contains raw scores with readiness flags and native
method metadata, rather than a new online scoring or normalization pathway.
"""
from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .offline_assets import file_identity
from .processing_mask import canonical_final_processing_mask


REGRESSION_KEYS = (
    "enable_motion_regression", "mot_reg", "max_poly_order", "use_gs", "use_wm", "use_vent",
    "enable_fd_censor_reg", "fd_thr", "enable_dvars_censor_reg", "dvars_thr_robust_z",
    "censor_plus1", "dvars_warmup", "dvars_mask_source", "analysis_space",
    "voxel_norm_ref_volumes", "voxel_norm_mode", "voxel_norm_reference",
)
RECEIPT_KEYS = (*REGRESSION_KEYS, "TR", "skip_first_trs", "smoothing_fwhm_mm",
                "analysis_mask", "analysis_mask_space", "gpu_resampler_interpolation",
                "enable_biopac_physio", "biopac_phys_reg", "epi_phase_encoding")


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected metadata object: {path}")
    return value


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _canonical(value):
    # Native metadata historically serialized max_poly_order as Infinity; JSON
    # profiles serialize the same setting as 'inf'. Retain strict JSON in pools.
    if isinstance(value, (float, np.floating)) and np.isposinf(value):
        return "inf"
    return value


def _equal(actual, expected):
    return _canonical(actual) == _canonical(expected)


def _same_file(recorded, expected, label):
    if (recorded is None) != (expected is None):
        raise ValueError(f"Different {label} in reference source")
    if expected is None:
        return None
    path = Path(recorded)
    if not path.is_file() or file_identity(path)["sha256"] != expected["sha256"]:
        raise ValueError(f"Different or unavailable {label} in reference source: {path}")
    return file_identity(path)


def _table(path, expected_count, fields):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not set(fields).issubset(reader.fieldnames or []):
            raise ValueError(f"Required columns {fields} missing in {path}")
        rows = {}
        for row in reader:
            try:
                volume = int(row["volume_idx"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid volume index in {path}") from exc
            if volume in rows:
                raise ValueError(f"Duplicate volume {volume} in {path}")
            rows[volume] = row
    if sorted(rows) != list(range(1, expected_count + 1)):
        raise ValueError(f"Incomplete or unexpected TRs in {path}; expected 1..{expected_count}")
    return rows


def _ready(value, path):
    if str(value) not in {"0", "1"}:
        raise ValueError(f"Readiness must be explicitly 0 or 1 in {path}")
    return str(value) == "1"


def _check_receipt(path, expected, decoder, roi, subject, count, run):
    if not path.exists():
        return None  # Ordinary RS logs predate the recorded-replay receipt.
    receipt = _json(path)
    if (receipt.get("version") != 1 or receipt.get("status") != "complete"
            or str(receipt.get("subject")) != str(subject)
            or str(receipt.get("run")) != str(run)
            or receipt.get("expected_volumes") != count
            or receipt.get("inputs_unchanged") is not True
            or receipt.get("processing_error_count") != 0
            or receipt.get("worker_exit_code", 0) != 0
            or receipt.get("decoder_unchanged", True) is not True):
        raise ValueError(f"Recorded replay was not completed cleanly: {path}")
    recorded = receipt.get("settings")
    if not isinstance(recorded, dict) or any(
            key not in recorded or not _equal(recorded[key], expected[key]) for key in RECEIPT_KEYS):
        raise ValueError(f"Recorded replay preprocessing settings differ: {path}")
    for key, identity in (("decoder", decoder), ("roi_txt", roi)):
        actual = receipt.get(key)
        if (identity is None) != (actual is None) or (identity is not None and (
                not isinstance(actual, dict) or actual.get("sha256") != identity["sha256"])):
            raise ValueError(f"Recorded replay used a different {key}: {path}")
    snapshot = receipt.get("inputs_snapshot")
    if isinstance(snapshot, dict):
        groups = list(snapshot.values())
        bold_snapshot = snapshot.get(str(run))
        if not isinstance(bold_snapshot, list) or len(bold_snapshot) != count:
            raise ValueError(f"Recorded replay has an inconsistent BOLD snapshot: {path}")
    else:
        groups = [snapshot]
    if (not groups or any(not isinstance(group, list) or not group for group in groups)
            or any(not isinstance(record, dict) or not record.get("path")
                   or not isinstance(record.get("size"), int) or record["size"] <= 0
                   or not isinstance(record.get("mtime_ns"), int)
                   for group in groups for record in group)):
        raise ValueError(f"Recorded replay lacks its input snapshot: {path}")
    return file_identity(path)


def pool_native_references(run_dirs, output, *, expected_volumes, settings, subject,
                           decoder_template, decoder_roi_txt=None, runtime=None):
    """Validate complete native runs and export a pooled population reference.

    ``expected_volumes`` has one positive count per source directory. Per-source
    masks, calibration pairs and registrations may differ; method/decoder/units
    must match. Existing directories are never overwritten. No new output is
    created until every source validates. Full replay receipts, when present,
    are checked; ordinary RS logs retain their original metadata-only provenance.
    """
    directories = [Path(path).expanduser().resolve() for path in run_dirs]
    counts = list(expected_volumes)
    if not directories or len(directories) != len(counts) or len(set(directories)) != len(directories):
        raise ValueError("Provide distinct run directories and one expected TR count per run")
    if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1 for n in counts):
        raise ValueError("Expected TR counts must be positive integers")
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use a new pooled reference directory: {output}")
    decoder = file_identity(decoder_template)
    roi = file_identity(decoder_roi_txt) if decoder_roi_txt is not None else None
    if runtime is None:
        import rt_pipeline as runtime
    runtime.REGRESSOR_SETTINGS.update(vars(settings))
    expected = settings.to_json_dict()
    cfg = SimpleNamespace(day_root=output.parent, subject=str(subject),
                          decoder_template=Path(decoder_template), decoder_roi_txt=decoder_roi_txt)
    pooled, sources, first_metadata = [], [], None
    expected_processing = {
        "policy": runtime.FINAL_PROCESSING_MASK_POLICY,
        "version": runtime.FINAL_PROCESSING_MASK_POLICY_VERSION,
        "nuisance_source": runtime.FINAL_NUISANCE_SOURCE,
        "processing_order": list(runtime.FINAL_PROCESSING_ORDER),
        "interpolation": settings.gpu_resampler_interpolation,
    }
    for number, (folder, count) in enumerate(zip(directories, counts), 1):
        metadata_path = folder / "session_metadata.json"
        scores_path = folder / "scores.csv"
        status_path = folder / "regression_status_rt.csv"
        metadata = _json(metadata_path)
        if str(metadata.get("subject")) != str(subject):
            raise ValueError(f"Reference source belongs to a different subject: {folder}")
        processing = canonical_final_processing_mask(
            metadata.get("processing_mask"), metadata.get("smoothing_fwhm_mm"))
        if any(processing.get(k) != v for k, v in expected_processing.items()):
            raise ValueError(f"Reference source is not the current native processing method: {folder}")
        temporal_order = metadata.get("temporal_processing_order")
        # The old marker is equivalent only when both source and requested
        # profiles disable Gaussian smoothing. All other settings stay checked.
        if (temporal_order == "spatial_smooth_glm_norm_v1"
                and metadata.get("smoothing_fwhm_mm") == 0
                and not isinstance(metadata.get("smoothing_fwhm_mm"), bool)
                and settings.smoothing_fwhm_mm == 0):
            temporal_order = runtime.TEMPORAL_PROCESSING_ORDER
        if (metadata.get("preprocessing_order") != runtime.PIPELINE_ORDER
                or temporal_order != runtime.TEMPORAL_PROCESSING_ORDER):
            raise ValueError(f"Reference source uses a different processing order: {folder}")
        for key, metadata_key in (("TR", "tr"), ("smoothing_fwhm_mm", "smoothing_fwhm_mm"),
                                  ("analysis_mask", "analysis_mask")):
            if metadata_key not in metadata or not _equal(metadata[metadata_key], expected[key]):
                raise ValueError(f"Reference source uses different {key}: {folder}")
        regression = metadata.get("regression")
        if not isinstance(regression, dict) or any(
                key not in regression or not _equal(regression[key], expected[key]) for key in REGRESSION_KEYS):
            raise ValueError(f"Reference source uses different nuisance/censor/normalization settings: {folder}")
        biopac = metadata.get("biopac")
        if not isinstance(biopac, dict) or biopac.get("enabled") != expected["enable_biopac_physio"]:
            raise ValueError(f"Reference source uses different BIOPAC settings: {folder}")
        if settings.enable_biopac_physio and biopac.get("phys_reg") != settings.biopac_phys_reg:
            raise ValueError(f"Reference source uses different physiology regressors: {folder}")
        source_decoder = _same_file(metadata.get("decoder_template"), decoder, "decoder")
        source_roi = _same_file(metadata.get("decoder_roi_txt"), roi, "decoder ROI-TXT")
        if settings.analysis_mask == "custom":
            if metadata.get("analysis_mask_space") != settings.analysis_mask_space:
                raise ValueError(f"Reference source uses different custom-mask coordinates: {folder}")
            _same_file(metadata.get("analysis_mask_file"), file_identity(settings.analysis_mask_file), "analysis mask")
        # Reuse the existing loader's exact normalization/coverage checks and
        # finite+ready selection, with no adapter allowance or first-ready drop.
        scores = _table(scores_path, count, ("volume_idx", "score_raw", "reg_ready"))
        statuses = _table(status_path, count, ("volume_idx", "reg_ready"))
        stats = runtime.load_reference_score_stats(cfg, str(folder))
        if stats is None:
            raise ValueError(f"Reference source has no usable score distribution: {folder}")
        values = []
        for volume in range(1, count + 1):
            score_ready = _ready(scores[volume]["reg_ready"], scores_path)
            status_ready = _ready(statuses[volume]["reg_ready"], status_path)
            if score_ready != status_ready:
                raise ValueError(f"Score/status readiness disagrees at TR {volume}: {folder}")
            try:
                value = float(scores[volume]["score_raw"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Malformed raw score at TR {volume}: {scores_path}") from exc
            if score_ready and np.isfinite(value):
                values.append(value)
            pooled.append({"volume_idx": len(pooled) + 1, "score_raw": value,
                           "reg_ready": int(score_ready), "source_run": number,
                           "source_run_dir": str(folder), "source_volume_idx": volume})
        values = np.asarray(values, dtype=np.float64)
        if (len(values) != stats["n"] or not np.isclose(values.mean(), stats["mean"], rtol=1e-12, atol=1e-12)
                or not np.isclose(values.std(ddof=0), stats["std"], rtol=1e-12, atol=1e-12)):
            raise ValueError(f"Reference source selection differs from the ordinary loader: {folder}")
        receipt = _check_receipt(folder / "recorded_reference.json", expected, decoder, roi,
                                 subject, count, metadata.get("run", folder.name))
        sources.append({"run_dir": str(folder), "expected_volumes": int(count),
                        "ready_finite_scores": int(len(values)), "mean": stats["mean"], "std": stats["std"],
                        "scores": file_identity(scores_path), "metadata": file_identity(metadata_path),
                        "regression_status": file_identity(status_path), "decoder": source_decoder,
                        "roi_txt": source_roi, "recorded_replay_receipt": receipt,
                        "provenance_scope": "completed_recorded_replay" if receipt else "ordinary_RS_metadata_only"})
        if first_metadata is None:
            first_metadata = metadata
    values = np.asarray([row["score_raw"] for row in pooled
                         if row["reg_ready"] and np.isfinite(row["score_raw"])], dtype=np.float64)
    mean, std = float(values.mean()), float(values.std(ddof=0))
    if len(values) < 2 or not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Pooled reference needs at least two finite ready scores and positive population SD")
    stats = {"run": str(output), "mean": mean, "std": std, "n": int(len(values)),
             "n_runs": len(sources), "ddof": 0, "aggregation": "pooled_ready_samples",
             "used_reg_ready": True, "skipped_first_reg_ready": False,
             "censor_filter": "include flagged ready scores, matching the native reference loader"}
    metadata = {"created_at": datetime.now(timezone.utc).isoformat(), "subject": str(subject),
                "decoder_template": decoder["path"], "decoder_roi_txt": roi["path"] if roi else None,
                "normalization": copy.deepcopy(first_metadata["normalization"]),
                "coverage_policy": first_metadata["coverage_policy"], "processing_mask": expected_processing,
                "preprocessing_order": runtime.PIPELINE_ORDER,
                "temporal_processing_order": runtime.TEMPORAL_PROCESSING_ORDER,
                "tr": settings.TR, "smoothing_fwhm_mm": settings.smoothing_fwhm_mm,
                "analysis_mask": settings.analysis_mask, "analysis_mask_space": settings.analysis_mask_space,
                "analysis_mask_file": settings.analysis_mask_file,
                "regression": {key: expected[key] for key in REGRESSION_KEYS},
                "biopac": {"enabled": settings.enable_biopac_physio, "phys_reg": settings.biopac_phys_reg},
                "reference_pool": {"version": 1, "status": "complete", "source_type": "native_pipeline_scores",
                                   "decoder": decoder, "roi_txt": roi, "sources": sources,
                                   "selection": "finite_raw_and_normalization_ready",
                                   "aggregation": stats["aggregation"], "ddof": 0,
                                   "per_run_temporal_state": "independent_source_runs"}}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pooled[0]))
        writer.writeheader()
        writer.writerows(pooled)
    _write_json(output / "reference_stats.json", stats)
    _write_json(output / "session_metadata.json", metadata)  # Completion marker written last.
    return stats
