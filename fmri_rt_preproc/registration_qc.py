"""Image registration diagnostics; no fitting, resampling, or universal cutoff.

All images must already share the fixed image's physical grid. The caller owns
session-level approval. A report ID binds that approval to inputs and policy.
"""
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


VERSION = 1
_THRESHOLDS = {
    "pearson_r_min": ("pearson_r", -1., 1.),
    "nmi_min": ("nmi", 1., 2.),
    "edge_correlation_min": ("edge_correlation", -1., 1.),
    "signal_overlap_min": ("signal_overlap_fraction", 0., 1.),
    "geometric_coverage_min": ("geometric_coverage_fraction", 0., 1.),
    "mask_dice_min": ("mask_dice", 0., 1.),
}


def file_identity(path):
    """Content identity, independent of modification time."""
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "size": path.stat().st_size,
            "sha256": digest.hexdigest()}


def inputs_match(report):
    """Reject approval reuse after any image, mask, or transform changes."""
    try:
        return bool(report["inputs"]) and all(
            file_identity(record["path"]) == record
            for record in report["inputs"].values()
        )
    except (OSError, KeyError, TypeError):
        return False


def load_registration_qc_report(path):
    """Load a report only if its content and current input hashes still match."""
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("version") != VERSION:
        raise ValueError(f"Unsupported registration QC report: {path}")
    payload = {key: value for key, value in report.items() if key not in {"report_id", "review_png"}}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if digest != report.get("report_id") or not inputs_match(report):
        raise ValueError(f"Registration QC report or input files changed; regenerate QC: {path}")
    if report.get("gate_status") not in {"pass", "review_required", "fail"}:
        raise ValueError(f"Invalid registration QC gate status: {path}")
    return report


def approve_registration_qc(report_path, *, reviewer, note=""):
    """Record an EXPLICIT human review; never call automatically at NF startup.

    The caller must first show the report/images to the named reviewer. This
    helper records the review, not performs it. Failed reports cannot be waived.
    """
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("A named reviewer is required for registration QC approval")
    if not isinstance(note, str):
        raise ValueError("The review note must be text")
    report = load_registration_qc_report(report_path)
    if report["gate_status"] == "fail" or report["hard_failures"] or report["threshold_failures"]:
        raise ValueError("Failed registration QC cannot be approved; correct the inputs/registration and regenerate QC")
    approval = {"version": VERSION, "report_id": report["report_id"],
                "reviewer": reviewer.strip(), "note": note}
    destination = Path(report_path).with_suffix(".approval.json")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(approval, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def assert_qc_gate(report_paths):
    """Raise before NF unless every expected report passes or has current review.

    Pass the session's explicit required report list. A directory is shorthand
    for its ``registration_qc*.json`` reports, excluding approval sidecars; the
    caller must still ensure all expected registrations have been reported.
    """
    if isinstance(report_paths, (str, Path)):
        path = Path(report_paths)
        report_paths = sorted(p for p in path.glob("registration_qc*.json")
                              if not p.name.endswith(".approval.json")) if path.is_dir() else [path]
    paths = list(report_paths)
    if not paths:
        raise ValueError("No registration QC reports; prepare and review this session before neurofeedback")
    reports = []
    for path in paths:
        path = Path(path)
        report = load_registration_qc_report(path)
        if report["gate_status"] == "fail" or report["hard_failures"] or report["threshold_failures"]:
            raise ValueError(f"Registration QC failed: {path}")
        if report["gate_status"] == "review_required":
            approval_path = path.with_suffix(".approval.json")
            try:
                approval = json.loads(approval_path.read_text(encoding="utf-8"))
                approved = (isinstance(approval, dict) and approval.get("version") == VERSION
                            and approval.get("report_id") == report["report_id"]
                            and isinstance(approval.get("reviewer"), str)
                            and bool(approval["reviewer"].strip()))
            except (OSError, ValueError):
                approved = False
            if not approved:
                raise ValueError(f"Registration QC needs visual approval for the current inputs: {path}")
        elif not report.get("thresholds_validated") or not report.get("complete_threshold_policy"):
            raise ValueError(f"Registration QC cannot auto-pass without a validated complete threshold policy: {path}")
        reports.append(report)
    return reports


def _load(path, reference=None):
    image = nib.load(str(path))
    if len(image.shape) != 3 or not np.all(np.isfinite(image.affine)):
        raise ValueError(f"Registration QC requires a finite-affine 3D image: {path}")
    if abs(np.linalg.det(image.affine[:3, :3])) < 1e-12:
        raise ValueError(f"Registration QC image has singular geometry: {path}")
    if reference is not None and (
        image.shape != reference.shape
        or not np.allclose(image.affine, reference.affine, atol=1e-4, rtol=0)
    ):
        raise ValueError(f"Registration QC grids differ; resample with the transform first: {path}")
    return image, np.asarray(image.dataobj, dtype=np.float64)


def _pearson(x, y):
    if x.size < 3:
        return None
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if denominator == 0 or not np.isfinite(denominator):
        return None
    return float(np.clip(np.dot(x, y) / denominator, -1, 1))


def _nmi(x, y, bins):
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    joint = np.histogram2d(x, y, bins=bins)[0]
    joint /= joint.sum()

    def entropy(probabilities):
        p = probabilities[probabilities > 0]
        return float(-np.sum(p * np.log(p)))

    return float((entropy(joint.sum(axis=0)) + entropy(joint.sum(axis=1)))
                 / entropy(joint))


def _interior(mask):
    """One-voxel face erosion avoids measuring mask/zero-fill boundaries."""
    out = mask.copy()
    for axis in range(3):
        if mask.shape[axis] < 3:
            return np.zeros_like(mask)
        left, middle, right = [slice(None)] * 3, [slice(None)] * 3, [slice(None)] * 3
        left[axis], middle[axis], right[axis] = slice(None, -2), slice(1, -1), slice(2, None)
        out[tuple(middle)] &= mask[tuple(left)] & mask[tuple(right)]
        for index in (0, -1):
            boundary = [slice(None)] * 3
            boundary[axis] = index
            out[tuple(boundary)] = False
    return out


def _gradient_magnitude(data, affine):
    gradients = []
    for axis, spacing in enumerate(np.linalg.norm(affine[:3, :3], axis=0)):
        gradients.append(np.gradient(data, spacing, axis=axis)
                         if data.shape[axis] > 1 else np.zeros_like(data))
    return np.sqrt(sum(g * g for g in gradients))


def _policy(thresholds, validated, comparison):
    if not isinstance(validated, bool):
        raise ValueError("thresholds_validated must be a boolean")
    policy = dict(thresholds or {})
    for key, value in policy.items():
        if key not in _THRESHOLDS:
            raise ValueError(f"Unknown registration QC threshold: {key}")
        _, low, high = _THRESHOLDS[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} must be a finite number in [{low}, {high}]")
        policy[key] = float(value)
    if comparison == "epi_t1" and "pearson_r_min" in policy:
        raise ValueError("Positive global Pearson correlation is not an EPI-T1 acceptance criterion")
    return policy


def build_registration_qc(fixed_file, aligned_file, fixed_mask_file, *,
                          comparison="epi_t1", aligned_mask_file=None,
                          coverage_file=None, transform_files=(), thresholds=None,
                          thresholds_validated=False, output_json=None,
                          review_png=None, histogram_bins=32):
    """Return diagnostics and a conservative decision for an already aligned pair.

    ``coverage_file`` is a transformed acquisition-support mask, not a brain
    intensity mask. Masks must have finite values, with positive voxels included.
    Similarity excludes zero fill in either image and is never computed from
    common background zeros. Thresholds are explicit lower bounds (see docs).
    No validated policy means visual review is required, even for perfect r.
    Input/grid errors raise ValueError; measurable failures produce status fail.
    """
    if comparison not in {"epi_epi", "epi_t1", "t1_t1"}:
        raise ValueError("comparison must be 'epi_epi', 'epi_t1', or 't1_t1'")
    if isinstance(histogram_bins, bool) or not isinstance(histogram_bins, int) or not 2 <= histogram_bins <= 256:
        raise ValueError("histogram_bins must be an integer between 2 and 256")
    policy = _policy(thresholds, thresholds_validated, comparison)
    paths = {"fixed": fixed_file, "aligned": aligned_file, "fixed_mask": fixed_mask_file}
    if aligned_mask_file is not None:
        paths["aligned_mask"] = aligned_mask_file
    if coverage_file is not None:
        paths["coverage"] = coverage_file
    paths.update({f"transform_{index}": path for index, path in enumerate(transform_files)})
    identities = {key: file_identity(path) for key, path in paths.items()}
    reference, fixed = _load(fixed_file)
    _, aligned = _load(aligned_file, reference)
    masks = {}
    for key in ("fixed_mask", "aligned_mask", "coverage"):
        if key in paths:
            _, values = _load(paths[key], reference)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Registration QC {key} contains nonfinite values")
            masks[key] = values > 0
    brain = masks["fixed_mask"]
    count = int(brain.sum())
    supported = brain & masks.get("coverage", True)
    domain = supported & masks.get("aligned_mask", True)
    finite = np.isfinite(fixed) & np.isfinite(aligned)
    overlap = domain & finite & (fixed != 0) & (aligned != 0)
    n_overlap = int(overlap.sum())
    hard_failures = []
    if count == 0:
        hard_failures.append("empty_fixed_mask")
    if np.any(brain & ~np.isfinite(fixed)) or np.any(supported & ~np.isfinite(aligned)):
        hard_failures.append("nonfinite_image_values_in_evaluated_region")
    if n_overlap < 3:
        hard_failures.append("insufficient_signal_overlap")
    x, y = fixed[overlap], aligned[overlap]
    nmi = _nmi(x, y, histogram_bins)
    if n_overlap >= 3 and (np.ptp(x) == 0 or np.ptp(y) == 0):
        hard_failures.append("constant_image_in_overlap")
    interior = _interior(overlap)
    edge_r = None
    if interior.sum() >= 3:
        # Values outside the evaluated region never contribute through NaNs.
        a = _gradient_magnitude(np.where(finite, fixed, 0.), reference.affine)
        b = _gradient_magnitude(np.where(finite, aligned, 0.), reference.affine)
        edge_r = _pearson(a[interior], b[interior])
    metrics = {
        "fixed_mask_voxels": count, "overlap_voxels": n_overlap,
        "pearson_r": _pearson(x, y) if comparison in {"epi_epi", "t1_t1"} else None,
        "nmi": nmi, "edge_correlation": edge_r,
        "edge_voxels": int(interior.sum()),
        "signal_overlap_fraction": n_overlap / count if count else 0.,
        "geometric_coverage_fraction": float(supported.sum() / count)
        if count and "coverage" in masks else None,
        "mask_dice": None,
    }
    if "aligned_mask" in masks:
        other = masks["aligned_mask"]
        denominator = count + int(other.sum())
        metrics["mask_dice"] = float(2 * np.sum(brain & other) / denominator) if denominator else None
    threshold_failures = []
    for key, limit in policy.items():
        name = _THRESHOLDS[key][0]
        value = metrics[name]
        if value is None or value < limit:
            threshold_failures.append({"metric": name, "value": value, "minimum": limit})
    similarity_policy = "pearson_r_min" in policy if comparison in {"epi_epi", "t1_t1"} else bool(
        {"nmi_min", "edge_correlation_min"} & policy.keys())
    # A high similarity on a tiny surviving patch must not approve the run.
    complete_policy = similarity_policy and "signal_overlap_min" in policy
    if hard_failures or threshold_failures:
        status = "fail"
    elif thresholds_validated and complete_policy:
        status = "pass"
    else:
        status = "review_required"
    report = {
        "version": VERSION, "comparison": comparison, "inputs": identities,
        "parameters": {"histogram_bins": histogram_bins, "edge_erosion_voxels": 1,
                       "nmi_definition": "(H(fixed)+H(aligned))/H(joint)"},
        "mask_semantics": {
            "fixed_mask": "caller-supplied evaluation support on the fixed image grid",
            "aligned_mask": "separate supplied mask" if aligned_mask_file is not None else None,
            "coverage": "transformed acquisition-support mask" if coverage_file is not None else None,
            "signal_overlap_fraction": "finite nonzero paired signal / evaluation-support voxels",
        },
        "thresholds": policy, "thresholds_validated": thresholds_validated,
        "complete_threshold_policy": complete_policy, "metrics": metrics,
        "hard_failures": hard_failures, "threshold_failures": threshold_failures,
        "gate_status": status,
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report["report_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if review_png is not None:
        _review_png(reference, fixed, aligned, brain, report, review_png)
        report["review_png"] = str(Path(review_png).resolve())
    # Detect files changed while reading/hashing; never approve inconsistent data.
    if not inputs_match(report):
        raise ValueError("Registration QC inputs changed while generating the report; retry after preparation finishes")
    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    return report


def _review_png(reference, fixed, aligned, mask, report, destination):
    """Three native-grid orthogonal views; source axes/obliquity are explicit."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 9), dpi=140, facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 3)
    positions = np.argwhere(mask)
    center = np.median(positions, axis=0).astype(int) if positions.size else np.array(mask.shape) // 2

    def scaled(data):
        values = data[mask & np.isfinite(data) & (data != 0)]
        lo, hi = np.percentile(values, [2, 98]) if values.size else (0., 1.)
        return np.clip((np.nan_to_num(data, nan=lo, posinf=hi, neginf=lo) - lo) / max(hi - lo, 1e-12), 0, 1)

    first, second = scaled(fixed), scaled(aligned)
    codes = nib.aff2axcodes(reference.affine)
    for axis in range(3):
        a, b = np.take(first, center[axis], axis=axis).T, np.take(second, center[axis], axis=axis).T
        blend = np.stack([a, b, b], axis=-1)
        for row, data in enumerate((a, b, blend)):
            axes[row, axis].imshow(data, cmap="gray", origin="lower", interpolation="nearest")
            axes[row, axis].set_xticks([])
            axes[row, axis].set_yticks([])
        axes[0, axis].set_title(f"Voxel axis {axis} ({codes[axis]}) · slice {center[axis]}", fontsize=11)
    for row, label in enumerate(("Fixed reference", "Aligned image", "Overlay: fixed red / aligned cyan")):
        axes[row, 0].set_ylabel(label, fontsize=10)
    figure.suptitle(f"Registration review · {report['comparison']} · {report['gate_status']}", fontsize=16)
    figure.text(.5, .022, "Native voxel planes (may be oblique). Independently scaled intensities; inspect anatomy and coverage.\n"
                f"Report {report['report_id'][:16]} · Similarity alone does not establish correct alignment.",
                ha="center", fontsize=10)
    figure.subplots_adjust(left=.09, right=.98, top=.92, bottom=.09, wspace=.07, hspace=.06)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor="white")
