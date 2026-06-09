from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from multiprocessing import Queue
from pathlib import Path
from queue import Full
from typing import Optional
import subprocess

import numpy as np


@dataclass(frozen=True)
class PCACondition:
    condition_id: str
    symbol: str
    roi: str
    pc: str
    direction: str

    @property
    def score_label(self) -> str:
        return str(self.pc).upper()

    @property
    def score_column(self) -> str:
        return f"{self.roi}_{self.score_label}"

    @property
    def direction_sign(self) -> float:
        return -1.0 if self.direction.lower() in {"down", "decrease", "-"} else 1.0


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def build_run_dir(base_data: Path, subj: str, day: str, run: str) -> Path:
    sub_tag = subj if str(subj).startswith("sub-") else f"sub-{subj}"
    day_dir = Path(base_data) / sub_tag / str(day)
    func_dir = day_dir / "func"
    candidates = [func_dir / str(run)]
    if not str(run).startswith("run-"):
        candidates.append(func_dir / f"run-{run}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def build_pca_root(day_dir: Path, run: str, pca_input: str) -> Path:
    run_name = Path(str(run)).name
    return day_dir / "PCA" / run_name / f"pca_{pca_input}"


def default_condition_paths(subject_root: Path) -> tuple[Path, Path]:
    return (
        Path(subject_root) / "pca_condition_key_private.json",
        Path(subject_root) / "pca_condition_schedule_public.json",
    )


def build_reference_stats_path(
    base_data: Path,
    subj: str,
    day: str,
    run: str,
    pca_input: str,
) -> Path:
    run_dir = build_run_dir(base_data, subj, day, run)
    day_dir = run_dir.parent.parent
    return build_pca_root(day_dir, run_dir.name, pca_input) / "pca_reference_stats.json"


def build_reference_scores_path(
    base_data: Path,
    subj: str,
    day: str,
    run: str,
    pca_input: str,
) -> Path:
    run_dir = build_run_dir(base_data, subj, day, run)
    day_dir = run_dir.parent.parent
    return build_pca_root(day_dir, run_dir.name, pca_input) / "scores_pca_all_rois.csv"


def _find_first_existing(paths: list[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _same_zooms(a: tuple[float, ...], b: tuple[float, ...], tol: float = 0.05) -> bool:
    if len(a) < 3 or len(b) < 3:
        return False
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a[:3], b[:3]))


def _crop_img_to_mask(img, mask_data: np.ndarray, padding_vox: int):
    import nibabel as nib

    mask_idx = np.argwhere(mask_data > 0)
    if mask_idx.size == 0:
        return img
    start = np.maximum(mask_idx.min(axis=0) - int(padding_vox), 0)
    stop = np.minimum(mask_idx.max(axis=0) + int(padding_vox) + 1, img.shape[:3])
    slices = tuple(slice(int(start[axis]), int(stop[axis])) for axis in range(3))
    cropped = np.asanyarray(img.dataobj)[slices].astype(np.float32, copy=False)
    affine = img.affine.copy()
    affine[:3, 3] = affine[:3, 3] + affine[:3, :3] @ start.astype(float)
    return nib.Nifti1Image(cropped, affine, img.header)


def _make_truncated_t1_reference(
    *,
    full_ref_img,
    full_ref_path: Path,
    trans_dir: Path,
    out_path: Path,
    padding_vox: int,
):
    import nibabel as nib

    epi_mask = _find_first_existing([
        trans_dir / "epi_mask_mean.nii",
        trans_dir / "epi_mask_mean.nii.gz",
        trans_dir / "rt_ref_epi_mask.nii",
        trans_dir / "rt_ref_epi_mask.nii.gz",
    ])
    epi2t1 = trans_dir / "epi2t1_Composite.h5"
    if epi_mask is None or not epi2t1.exists():
        nib.save(full_ref_img, str(out_path))
        return out_path

    mask_on_full = out_path.with_name(out_path.stem + "_epi_mask_on_full.nii.gz")
    if not mask_on_full.exists():
        cmd = [
            "antsApplyTransforms",
            "-d",
            "3",
            "-i",
            str(epi_mask),
            "-r",
            str(full_ref_path),
            "-o",
            str(mask_on_full),
            "-t",
            str(epi2t1),
            "-n",
            "NearestNeighbor",
            "--float",
            "1",
        ]
        subprocess.run(cmd, check=True)

    mask_data = np.asanyarray(nib.load(str(mask_on_full)).dataobj)
    cropped_img = _crop_img_to_mask(full_ref_img, mask_data, padding_vox)
    cropped_img.set_data_dtype(np.float32)
    nib.save(cropped_img, str(out_path))
    return out_path


def ensure_pca_t1_reference(
    subject_root: Path,
    trans_dir: Path,
    explicit_path: Optional[Path] = None,
    resolution: str = "epi",
    truncate_to_epi_fov: bool = True,
    padding_vox: int = 2,
) -> Path:
    if explicit_path is not None:
        explicit_path = Path(explicit_path)
        if not explicit_path.exists():
            raise FileNotFoundError(f"PCA reference image not found: {explicit_path}")
        return explicit_path

    resolution = str(resolution).lower()
    if resolution not in {"epi", "t1"}:
        raise ValueError(f"Unknown PCA reference resolution: {resolution}")

    t1_path = _find_first_existing([
        subject_root / "anat" / "T1_N4.nii",
        subject_root / "anat" / "T1_N4.nii.gz",
        subject_root / "anat" / "T1.nii",
        subject_root / "anat" / "T1.nii.gz",
    ])
    if t1_path is None:
        raise FileNotFoundError(
            f"Could not find a T1 image for PCA reference under {subject_root / 'anat'}"
        )

    if resolution == "t1":
        suffix = "_trunc" if truncate_to_epi_fov else ""
        out_path = trans_dir / f"pca_t1_reference_t1res{suffix}.nii.gz"
        if out_path.exists():
            return out_path
        if truncate_to_epi_fov:
            full_ref_path = trans_dir / "pca_t1_reference_t1res_full.nii.gz"
            import shutil
            import nibabel as nib

            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not full_ref_path.exists():
                shutil.copyfile(t1_path, full_ref_path)
            full_ref_img = nib.load(str(full_ref_path))
            return _make_truncated_t1_reference(
                full_ref_img=full_ref_img,
                full_ref_path=full_ref_path,
                trans_dir=trans_dir,
                out_path=out_path,
                padding_vox=padding_vox,
            )
        else:
            import shutil

            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(t1_path, out_path)
        return out_path

    suffix = "_trunc" if truncate_to_epi_fov else ""
    out_path = trans_dir / f"pca_t1_reference_epires{suffix}.nii.gz"
    epi_ref = _find_first_existing([
        trans_dir / "epi_unwarped_mean.nii",
        trans_dir / "epi_unwarped_mean.nii.gz",
        trans_dir / "rt_ref_epi.nii",
        trans_dir / "rt_ref_epi.nii.gz",
    ])
    if epi_ref is None:
        raise FileNotFoundError(
            f"Could not find an EPI reference in {trans_dir} to define PCA reference resolution."
        )

    import nibabel as nib
    from nibabel.processing import resample_to_output

    epi_zooms = nib.load(str(epi_ref)).header.get_zooms()[:3]
    if out_path.exists():
        existing_zooms = nib.load(str(out_path)).header.get_zooms()[:3]
        if _same_zooms(existing_zooms, epi_zooms):
            return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t1_img = nib.load(str(t1_path))
    ref_img = resample_to_output(t1_img, voxel_sizes=epi_zooms, order=1)
    ref_img.set_data_dtype(np.float32)
    if truncate_to_epi_fov:
        full_ref_path = trans_dir / "pca_t1_reference_epires_full.nii.gz"
        if not full_ref_path.exists():
            nib.save(ref_img, str(full_ref_path))
        return _make_truncated_t1_reference(
            full_ref_img=ref_img,
            full_ref_path=full_ref_path,
            trans_dir=trans_dir,
            out_path=out_path,
            padding_vox=padding_vox,
        )

    nib.save(ref_img, str(out_path))
    return out_path


def load_decoder_artifacts(roi_dir: Path) -> dict[str, np.ndarray]:
    bundle_path = roi_dir / "decoder_bundle.npz"
    if bundle_path.exists():
        z = np.load(bundle_path)
        artifacts = {
            "voxel_indices": z["voxel_indices"].astype(np.int64, copy=False),
            "weights": z["weights"].astype(np.float32, copy=False),
            "norm_mean": z["norm_mean"].astype(np.float32, copy=False),
            "norm_std": z["norm_std"].astype(np.float32, copy=False),
        }
        if "explained" in z.files:
            artifacts["explained"] = z["explained"].astype(np.float32, copy=False)
        return artifacts

    artifacts = {
        "voxel_indices": np.load(roi_dir / "decoder_voxel_indices.npy").astype(np.int64, copy=False),
        "weights": np.load(roi_dir / "decoder_weights.npy").astype(np.float32, copy=False),
        "norm_mean": np.load(roi_dir / "decoder_norm_mean.npy").astype(np.float32, copy=False),
        "norm_std": np.load(roi_dir / "decoder_norm_std.npy").astype(np.float32, copy=False),
    }
    explained_path = roi_dir / "pca_explained.npy"
    if explained_path.exists():
        artifacts["explained"] = np.load(explained_path).astype(np.float32, copy=False)
    return artifacts


LEGACY_COMPONENT_METRICS = {"projection", "cosine"}
MULTI_PC_METRICS = {"subspace_cosine", "subspace_distance", "projection_norm"}
ALL_SCORE_METRICS = LEGACY_COMPONENT_METRICS | MULTI_PC_METRICS


def normalize_score_metric(score_metric: str) -> str:
    metric = str(score_metric).strip().lower()
    aliases = {
        "multi_cosine": "subspace_cosine",
        "top_pc_cosine": "subspace_cosine",
        "top_pcs_cosine": "subspace_cosine",
        "distance": "subspace_distance",
        "cosine_distance": "subspace_distance",
        "norm": "projection_norm",
        "pc_norm": "projection_norm",
    }
    metric = aliases.get(metric, metric)
    if metric not in ALL_SCORE_METRICS:
        raise ValueError(
            f"Unknown PCA score metric: {score_metric}. "
            f"Use one of {sorted(ALL_SCORE_METRICS)}."
        )
    return metric


def _normalized_roi_vector(
    volume_3d: np.ndarray,
    decoder: dict[str, np.ndarray],
    normalization: str,
) -> np.ndarray:
    flat = volume_3d.reshape(-1).astype(np.float32, copy=False)
    x = flat[decoder["voxel_indices"]]

    if normalization == "zscore":
        safe_std = np.where(decoder["norm_std"] == 0, 1.0, decoder["norm_std"])
        return ((x - decoder["norm_mean"]) / safe_std).astype(np.float32, copy=False)
    if normalization == "demean":
        return (x - decoder["norm_mean"]).astype(np.float32, copy=False)
    if normalization == "none":
        return x.astype(np.float32, copy=False)
    raise ValueError(f"Unknown PCA normalization: {normalization}")

def score_pca_volume(
    volume_3d: np.ndarray,
    decoder: dict[str, np.ndarray],
    *,
    normalization: str = "zscore",
    score_metric: str = "projection",
) -> np.ndarray:
    metric = normalize_score_metric(score_metric)
    if metric not in LEGACY_COMPONENT_METRICS:
        raise ValueError(
            f"score_pca_volume returns per-PC values and only supports "
            f"{sorted(LEGACY_COMPONENT_METRICS)}. Use score_pca_scalar for {metric}."
        )
    x = _normalized_roi_vector(volume_3d, decoder, normalization)
    weights = decoder["weights"]
    if metric == "projection":
        return (weights @ x).astype(np.float32, copy=False)
    if metric == "cosine":
        x_norm = float(np.linalg.norm(x))
        if x_norm == 0:
            return np.zeros((weights.shape[0],), dtype=np.float32)
        w_norm = np.linalg.norm(weights, axis=1)
        w_norm_safe = np.where(w_norm == 0, 1.0, w_norm)
        return ((weights @ x) / (w_norm_safe * x_norm)).astype(np.float32, copy=False)
    raise ValueError(f"Unknown PCA score metric: {score_metric}")


def _target_pc_index(target_pc: str | None) -> int:
    pc = "PC01" if target_pc is None else str(target_pc).strip().upper()
    try:
        return int(pc.replace("PC", "")) - 1
    except ValueError as exc:
        raise ValueError(f"Invalid PCA target PC: {target_pc!r}") from exc


def _format_variance_label(top_pc_variance: float) -> str:
    pct = float(top_pc_variance) * 100.0
    if abs(pct - round(pct)) < 1e-6:
        return f"{int(round(pct))}PCT"
    text = f"{pct:.3g}".replace(".", "P")
    return f"{text}PCT"


def _format_min_variance_label(min_variance: float) -> str:
    pct = float(min_variance) * 100.0
    if abs(pct - round(pct)) < 1e-6:
        return f"{int(round(pct))}PCT"
    text = f"{pct:.3g}".replace(".", "P")
    return f"{text}PCT"


def _parse_min_variance_top_pcs(top_pcs: str | int) -> Optional[float]:
    text = str(top_pcs).strip().lower()
    for prefix in ("var:", "evr:", "minvar:", ">="):
        if text.startswith(prefix):
            return float(text[len(prefix):])
    return None


def resolve_score_label(
    *,
    score_metric: str,
    target_pc: str | None = None,
    top_pcs: str | int = "auto",
    top_pc_variance: float = 0.10,
) -> str:
    metric = normalize_score_metric(score_metric)
    if metric in LEGACY_COMPONENT_METRICS:
        return f"PC{_target_pc_index(target_pc) + 1:02d}"
    top_pcs_text = str(top_pcs).strip().lower()
    min_variance = _parse_min_variance_top_pcs(top_pcs)
    if min_variance is not None:
        return f"TOPPC{_format_min_variance_label(min_variance)}"
    if top_pcs_text == "auto":
        return f"CUM{_format_variance_label(top_pc_variance)}"
    if top_pcs_text == "all":
        return "TOPPCALL"
    return f"TOPPC{int(top_pcs):02d}"


def select_top_pc_indices(
    decoder: dict[str, np.ndarray],
    *,
    top_pcs: str | int = "auto",
    top_pc_variance: float = 0.10,
) -> np.ndarray:
    n_components = int(decoder["weights"].shape[0])
    if n_components <= 0:
        raise ValueError("PCA decoder has no components")

    top_pcs_text = str(top_pcs).strip().lower()
    min_variance = _parse_min_variance_top_pcs(top_pcs)
    if top_pcs_text == "all":
        n_keep = n_components
    elif min_variance is not None:
        explained = decoder.get("explained")
        if explained is not None and len(explained):
            explained = np.asarray(explained, dtype=float)
            valid = np.where(np.nan_to_num(explained, nan=-np.inf) >= min_variance)[0]
            n_keep = int(valid[-1] + 1) if valid.size else 1
        else:
            n_keep = min(3, n_components)
    elif top_pcs_text == "auto":
        explained = decoder.get("explained")
        if explained is not None and len(explained):
            explained = np.asarray(explained, dtype=float)
            if np.any(np.isfinite(explained)):
                cumulative = np.cumsum(np.nan_to_num(explained, nan=0.0))
                threshold = float(top_pc_variance)
                if cumulative.size and threshold <= float(cumulative[-1]):
                    n_keep = int(np.searchsorted(cumulative, threshold, side="left") + 1)
                else:
                    n_keep = n_components
            else:
                n_keep = min(3, n_components)
        else:
            n_keep = min(3, n_components)
    else:
        n_keep = int(top_pcs)

    n_keep = max(1, min(int(n_keep), n_components))
    return np.arange(n_keep, dtype=np.int64)


def describe_top_pc_selection(
    decoder: dict[str, np.ndarray],
    *,
    top_pcs: str | int = "auto",
    top_pc_variance: float = 0.10,
) -> dict:
    indices = select_top_pc_indices(
        decoder,
        top_pcs=top_pcs,
        top_pc_variance=top_pc_variance,
    )
    explained = decoder.get("explained")
    selected_explained = None
    cumulative_explained = None
    auto_threshold_reached = None
    if explained is not None and len(explained):
        explained = np.asarray(explained, dtype=float)
        selected_explained = [float(explained[i]) for i in indices if i < len(explained)]
        cumulative_explained = float(np.sum(selected_explained))
        if str(top_pcs).strip().lower() == "auto":
            auto_threshold_reached = cumulative_explained >= float(top_pc_variance)
    return {
        "top_pcs": str(top_pcs),
        "min_pc_variance": _parse_min_variance_top_pcs(top_pcs),
        "top_pc_variance": float(top_pc_variance),
        "auto_threshold_reached": auto_threshold_reached,
        "selected_pc_indices_1based": [int(i) + 1 for i in indices],
        "selected_explained": selected_explained,
        "selected_cumulative_explained": cumulative_explained,
    }


def score_pca_scalar(
    volume_3d: np.ndarray,
    decoder: dict[str, np.ndarray],
    *,
    normalization: str = "zscore",
    score_metric: str = "subspace_cosine",
    target_pc: str | None = None,
    top_pcs: str | int = "auto",
    top_pc_variance: float = 0.10,
) -> float:
    metric = normalize_score_metric(score_metric)
    if metric in LEGACY_COMPONENT_METRICS:
        scores = score_pca_volume(
            volume_3d,
            decoder,
            normalization=normalization,
            score_metric=metric,
        )
        pc_idx = _target_pc_index(target_pc)
        if pc_idx < 0 or pc_idx >= scores.shape[0]:
            raise ValueError(
                f"PCA decoder has {scores.shape[0]} PCs, cannot use "
                f"{target_pc or 'PC01'}"
            )
        return float(scores[pc_idx])

    x = _normalized_roi_vector(volume_3d, decoder, normalization)
    weights = decoder["weights"]
    indices = select_top_pc_indices(
        decoder,
        top_pcs=top_pcs,
        top_pc_variance=top_pc_variance,
    )
    selected = weights[indices]
    projections = selected @ x
    if metric == "projection_norm":
        return float(np.linalg.norm(projections))

    x_norm = float(np.linalg.norm(x))
    if x_norm == 0.0:
        return 1.0 if metric == "subspace_distance" else 0.0
    w_norm = np.linalg.norm(selected, axis=1)
    w_norm_safe = np.where(w_norm == 0, 1.0, w_norm)
    cosines = projections / (w_norm_safe * x_norm)
    closeness = float(np.sqrt(np.sum(cosines * cosines)))
    closeness = float(np.clip(closeness, 0.0, 1.0))
    if metric == "subspace_cosine":
        return closeness
    if metric == "subspace_distance":
        return 1.0 - closeness
    raise ValueError(f"Unknown PCA score metric: {score_metric}")


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def load_reference_stats(path: Optional[Path]) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("columns", payload)


def load_reference_stats_from_score_csv(
    scores_csv: Path,
    columns: list[str],
) -> dict[str, dict[str, float]]:
    scores_csv = Path(scores_csv)
    if not scores_csv.exists():
        raise FileNotFoundError(f"PCA reference scores CSV not found: {scores_csv}")
    values: dict[str, list[float]] = {column: [] for column in columns}
    with scores_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"PCA reference scores CSV has no header: {scores_csv}")
        missing = [column for column in columns if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"PCA reference scores CSV missing columns: {missing}")
        for row in reader:
            for column in columns:
                try:
                    value = float(row[column])
                except (TypeError, ValueError, KeyError):
                    continue
                if np.isfinite(value):
                    values[column].append(value)

    stats: dict[str, dict[str, float]] = {}
    for column, buf in values.items():
        arr = np.asarray(buf, dtype=float)
        if arr.size < 2:
            raise ValueError(
                f"PCA reference column {column!r} has fewer than 2 valid samples "
                f"in {scores_csv}"
            )
        std = float(arr.std())
        if not np.isfinite(std) or std == 0.0:
            raise ValueError(f"PCA reference column {column!r} has invalid std: {std}")
        stats[column] = {
            "mean": float(arr.mean()),
            "std": std,
            "n": int(arr.size),
            "p05": float(np.percentile(arr, 5)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
        }
    return stats


def plot_pca_scores_motion(
    *,
    run_dir: Path,
    scores_csv: Path,
    out_png: Path,
    score_columns: Optional[list[str]] = None,
    title: str = "PCA scores and motion",
) -> Optional[Path]:
    scores_csv = Path(scores_csv)
    motion_path = Path(run_dir) / "motion_rt.1D"
    if not scores_csv.exists():
        return None

    with scores_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None
        volume_field = "volume_idx" if "volume_idx" in reader.fieldnames else "tr"
        if volume_field not in reader.fieldnames:
            return None
        metadata_columns = {
            "volume_idx",
            "tr",
            "timestamp",
            "condition_id",
            "symbol",
            "roi",
            "pc",
            "score_label",
            "direction",
        }
        candidate_columns = score_columns or [
            column for column in reader.fieldnames if column not in metadata_columns
        ]
        volumes: list[int] = []
        values: dict[str, list[float]] = {column: [] for column in candidate_columns}
        for row in reader:
            try:
                volume_idx = int(float(row[volume_field]))
            except (TypeError, ValueError):
                continue
            row_values: dict[str, float] = {}
            has_score = False
            for column in candidate_columns:
                try:
                    value = float(row.get(column, ""))
                except (TypeError, ValueError):
                    value = float("nan")
                if np.isfinite(value):
                    has_score = True
                row_values[column] = value
            if not has_score:
                continue
            volumes.append(volume_idx)
            for column in candidate_columns:
                values[column].append(row_values[column])

    score_columns = [
        column
        for column in candidate_columns
        if column in values and np.any(np.isfinite(np.asarray(values[column], dtype=float)))
    ]
    if not volumes or not score_columns:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_motion = motion_path.exists()
    motion = None
    motion_vols = None
    if has_motion:
        try:
            motion = np.loadtxt(motion_path)
            if motion.ndim == 1:
                motion = motion[None, :]
            all_motion_vols = np.arange(1, motion.shape[0] + 1)
            include_motion = np.isin(all_motion_vols, np.asarray(volumes, dtype=int))
            motion = motion[include_motion]
            motion_vols = all_motion_vols[include_motion]
            has_motion = motion.shape[0] > 0
        except Exception:
            has_motion = False

    n_panels = 2 if has_motion else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 8 if has_motion else 5), sharex=False)
    if n_panels == 1:
        axes = [axes]

    for column in score_columns:
        axes[0].plot(volumes, values[column], label=column)
    axes[0].set_title(title)
    axes[0].set_xlabel("Volume")
    axes[0].set_ylabel("PCA score")
    axes[0].legend(loc="upper right", fontsize=8)

    if has_motion and motion is not None and motion_vols is not None:
        for idx in range(min(motion.shape[1], 6)):
            axes[1].plot(motion_vols, motion[:, idx], label=f"Motion {idx + 1}")
        axes[1].set_xlabel("Volume")
        axes[1].set_ylabel("Motion")
        axes[1].legend(loc="upper right", ncol=3, fontsize=8)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def feedback_from_score(
    raw_score: float,
    condition: PCACondition,
    reference_stats: dict[str, dict[str, float]],
) -> dict[str, float]:
    directed_score = raw_score * condition.direction_sign
    stats = reference_stats.get(condition.score_column)
    if not stats:
        return {
            "directed_score": directed_score,
            "score_z": float("nan"),
            "feedback_score": directed_score,
        }
    std = float(stats["std"])
    if not np.isfinite(std) or std == 0.0:
        raise ValueError(f"Invalid reference std for {condition.score_column}: {std}")
    raw_z = (raw_score - float(stats["mean"])) / std
    directed_z = raw_z * condition.direction_sign
    return {
        "directed_score": directed_score,
        "score_z": directed_z,
        "feedback_score": float(np.clip(_norm_cdf(directed_z) * 100.0, 0.0, 100.0)),
    }


def _make_conditions(
    roi_labels: list[str],
    direction_labels: list[str],
    symbols: list[str],
    target_pc: str,
    symbol_seed: Optional[int],
) -> list[PCACondition]:
    n_conditions = len(roi_labels) * len(direction_labels)
    if len(symbols) < n_conditions:
        raise ValueError(f"Need at least {n_conditions} condition symbols, got {len(symbols)}")
    if len(set(symbols)) < n_conditions:
        raise ValueError("Condition symbols must be unique for blinded PCA feedback.")
    shuffled_symbols = symbols[:]
    random.Random(symbol_seed).shuffle(shuffled_symbols)
    conditions: list[PCACondition] = []
    symbol_iter = iter(shuffled_symbols)
    for roi in roi_labels:
        for direction in direction_labels:
            symbol = next(symbol_iter)
            conditions.append(
                PCACondition(
                    condition_id=f"condition_{symbol}",
                    symbol=symbol,
                    roi=roi,
                    pc=target_pc,
                    direction=direction,
                )
            )
    return conditions


def load_or_create_condition_schedule(
    *,
    private_path: Path,
    public_path: Path,
    roi_labels: list[str],
    direction_labels: list[str],
    symbols: list[str],
    target_pc: str = "PC01",
    condition_seed: Optional[int] = None,
    symbol_seed: Optional[int] = None,
) -> dict:
    schema_version = 2
    if private_path.exists():
        with private_path.open("r", encoding="utf-8") as f:
            schedule = json.load(f)
        expected = {
            "schema_version": schema_version,
            "roi_labels": roi_labels,
            "direction_labels": direction_labels,
        }
        mismatches = [k for k, v in expected.items() if schedule.get(k) != v]
        if not mismatches:
            if not public_path.exists():
                public_schedule = {
                    "schema_version": schema_version,
                    "created_at": schedule.get("created_at"),
                    "condition_seed": schedule.get("condition_seed"),
                    "symbol_seed": schedule.get("symbol_seed"),
                    "order": schedule.get("order", []),
                    "conditions": [
                        {
                            "condition_id": c["condition_id"],
                            "symbol": c["symbol"],
                        }
                        for c in schedule.get("conditions", [])
                    ],
                }
                public_path.parent.mkdir(parents=True, exist_ok=True)
                public_path.write_text(
                    json.dumps(public_schedule, indent=2) + "\n",
                    encoding="utf-8",
                )
            return schedule
        raise ValueError(
            f"Existing PCA condition key does not match requested setup: {private_path}. "
            f"Mismatched fields: {mismatches}. Move/delete it to create a new blinded mapping."
        )

    conditions = _make_conditions(
        roi_labels,
        direction_labels,
        symbols,
        target_pc,
        symbol_seed,
    )
    order = [c.condition_id for c in conditions]
    random.Random(condition_seed).shuffle(order)
    private_schedule = {
        "schema_version": schema_version,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "roi_labels": roi_labels,
        "direction_labels": direction_labels,
        "target_pc": target_pc,
        "condition_seed": condition_seed,
        "symbol_seed": symbol_seed,
        "order": order,
        "conditions": [asdict(c) for c in conditions],
    }
    public_schedule = {
        "schema_version": schema_version,
        "created_at": private_schedule["created_at"],
        "condition_seed": condition_seed,
        "symbol_seed": symbol_seed,
        "order": order,
        "conditions": [
            {"condition_id": c.condition_id, "symbol": c.symbol}
            for c in conditions
        ],
    }
    private_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_text(json.dumps(private_schedule, indent=2) + "\n", encoding="utf-8")
    public_path.write_text(json.dumps(public_schedule, indent=2) + "\n", encoding="utf-8")
    return private_schedule


def condition_for_index(
    schedule: dict,
    index: int,
    *,
    score_label: Optional[str] = None,
) -> PCACondition:
    order = schedule["order"]
    if not order:
        raise ValueError("Condition schedule has an empty order")
    idx = (max(1, int(index)) - 1) % len(order)
    condition_id = order[idx]
    lookup = {c["condition_id"]: c for c in schedule["conditions"]}
    cond = lookup[condition_id]
    return PCACondition(
        condition_id=cond["condition_id"],
        symbol=cond["symbol"],
        roi=cond["roi"],
        pc=score_label or cond.get("pc") or schedule.get("target_pc", "PC01"),
        direction=cond["direction"],
    )


def volume_path_for_kind(run_dir: Path, volume_idx: int, volume_kind: str) -> Path:
    if volume_kind == "reg":
        return run_dir / "reg" / f"vol_{volume_idx:05d}_reg.nii"
    if volume_kind == "mc":
        return run_dir / "mc" / f"vol_{volume_idx:05d}_mc.nii"
    if volume_kind == "unwarped":
        return run_dir / "unwarped" / f"vol_{volume_idx:05d}_mc_uw.nii"
    if volume_kind == "t1":
        return run_dir / "t1" / f"vol_{volume_idx:05d}_t1.nii"
    if volume_kind == "mni":
        return run_dir / "mni" / f"vol_{volume_idx:05d}_mni.nii"
    raise ValueError(f"Unknown PCA volume kind: {volume_kind}")


def append_pca_score(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    fieldnames = [
        "volume_idx",
        "timestamp",
        "condition_id",
        "symbol",
        "score_label",
        "raw_component_score",
        "directed_score",
        "score_z",
        "feedback_score",
    ]
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def run_realtime_pca_scorer(
    *,
    run_dir: Path,
    pca_root: Path,
    condition: PCACondition,
    score_queue: Queue,
    stop_event,
    reference_stats_path: Optional[Path] = None,
    reference_stats: Optional[dict[str, dict[str, float]]] = None,
    volume_kind: str = "reg",
    normalization: str = "zscore",
    score_metric: str = "subspace_cosine",
    target_pc: Optional[str] = None,
    top_pcs: str | int = "auto",
    top_pc_variance: float = 0.10,
    max_trs: Optional[int] = None,
    poll_interval: float = 0.05,
) -> None:
    import nibabel as nib

    decoder = load_decoder_artifacts(pca_root / condition.roi)
    loaded_reference_stats = reference_stats or load_reference_stats(reference_stats_path)
    metric = normalize_score_metric(score_metric)
    if metric in LEGACY_COMPONENT_METRICS and target_pc is None:
        target_pc = condition.score_label

    out_csv = run_dir / "pca_realtime_scores.csv"
    processed: set[int] = set()
    volume_idx = 1
    while True:
        if max_trs is not None and volume_idx > max_trs:
            break
        if stop_event.is_set() and volume_idx in processed:
            break
        if volume_idx in processed:
            volume_idx += 1
            continue

        vol_path = volume_path_for_kind(run_dir, volume_idx, volume_kind)
        if not vol_path.exists():
            if stop_event.is_set():
                break
            time.sleep(poll_interval)
            continue

        try:
            img = nib.load(str(vol_path))
            raw_score = score_pca_scalar(
                np.asanyarray(img.dataobj),
                decoder,
                normalization=normalization,
                score_metric=score_metric,
                target_pc=target_pc,
                top_pcs=top_pcs,
                top_pc_variance=top_pc_variance,
            )
            feedback = feedback_from_score(raw_score, condition, loaded_reference_stats)
            row = {
                "volume_idx": volume_idx,
                "timestamp": time.time(),
                "condition_id": condition.condition_id,
                "symbol": condition.symbol,
                "score_label": condition.score_label,
                "raw_component_score": raw_score,
                **feedback,
            }
            append_pca_score(out_csv, row)
            try:
                score_queue.put_nowait(
                    {
                        **row,
                        "score_raw": row["feedback_score"],
                        "reg_ready": True,
                        "blind_condition": True,
                    }
                )
            except Full:
                pass
            processed.add(volume_idx)
            volume_idx += 1
        except Exception as exc:
            try:
                score_queue.put_nowait(
                    {
                        "volume_idx": volume_idx,
                        "timestamp": time.time(),
                        "reg_ready": False,
                        "error": str(exc),
                    }
                )
            except Full:
                pass
            processed.add(volume_idx)
            volume_idx += 1
