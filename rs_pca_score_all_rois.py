#!/usr/bin/env python
"""
Score a run with PCA decoders for all ROIs produced by roi_rs_pca_decoder_prep.py.

Example
-------
python rs_pca_score_all_rois.py \
  --subj 00085 --day 3 --run 1 --base-data ./data --pca-input reg

Outputs
-------
Writes per-TR component scores to:
  .../day/PCA/<run>/pca_<mode>/scores_pca_all_rois.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np


def _build_run_dir(base_data: Path, subj: str, day: str, run: str) -> Path:
    sub_tag = subj if str(subj).startswith("sub-") else f"sub-{subj}"
    day_tag = day if str(day).startswith("day") else f"day{day}"
    run_tag = run if str(run).startswith("run") else f"run{run}"
    return base_data / sub_tag / day_tag / "func" / run_tag


def _load_4d(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D NIfTI, got shape={data.shape} at {path}")
    return data


def _iter_roi_dirs(pca_root: Path) -> Iterable[Path]:
    for d in sorted(pca_root.iterdir()):
        if d.is_dir() and (d / "decoder_metadata.json").exists():
            yield d


def _load_decoder_artifacts(roi_dir: Path) -> dict:
    bundle_path = roi_dir / "decoder_bundle.npz"
    if bundle_path.exists():
        z = np.load(bundle_path)
        return {
            "voxel_indices": z["voxel_indices"].astype(np.int64, copy=False),
            "weights": z["weights"].astype(np.float32, copy=False),
            "norm_mean": z["norm_mean"].astype(np.float32, copy=False),
            "norm_std": z["norm_std"].astype(np.float32, copy=False),
        }

    return {
        "voxel_indices": np.load(roi_dir / "decoder_voxel_indices.npy").astype(np.int64, copy=False),
        "weights": np.load(roi_dir / "decoder_weights.npy").astype(np.float32, copy=False),
        "norm_mean": np.load(roi_dir / "decoder_norm_mean.npy").astype(np.float32, copy=False),
        "norm_std": np.load(roi_dir / "decoder_norm_std.npy").astype(np.float32, copy=False),
    }


def _score_one_vol(
    volume_3d: np.ndarray,
    voxel_indices: np.ndarray,
    weights: np.ndarray,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    normalization: str,
) -> np.ndarray:
    flat = volume_3d.reshape(-1).astype(np.float32, copy=False)
    x = flat[voxel_indices]

    if normalization == "zscore":
        safe_std = np.where(norm_std == 0, 1.0, norm_std)
        x = (x - norm_mean) / safe_std
    elif normalization == "demean":
        x = x - norm_mean
    elif normalization == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    return (weights @ x).astype(np.float32, copy=False)


def _cosine_one_vol(
    volume_3d: np.ndarray,
    voxel_indices: np.ndarray,
    weights: np.ndarray,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    normalization: str,
) -> np.ndarray:
    flat = volume_3d.reshape(-1).astype(np.float32, copy=False)
    x = flat[voxel_indices]

    if normalization == "zscore":
        safe_std = np.where(norm_std == 0, 1.0, norm_std)
        x = (x - norm_mean) / safe_std
    elif normalization == "demean":
        x = x - norm_mean
    elif normalization == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    x_norm = float(np.linalg.norm(x))
    if x_norm == 0:
        return np.zeros((weights.shape[0],), dtype=np.float32)

    w_norm = np.linalg.norm(weights, axis=1)
    w_norm_safe = np.where(w_norm == 0, 1.0, w_norm)
    sims = (weights @ x) / (w_norm_safe * x_norm)
    return sims.astype(np.float32, copy=False)


def _load_reference_stats(reference_csv: Path, columns: list[str]) -> dict[str, dict[str, float]]:
    if not reference_csv.exists():
        raise FileNotFoundError(f"Reference scores CSV not found: {reference_csv}")
    with open(reference_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Reference scores CSV has no header: {reference_csv}")
        missing = [c for c in columns if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Reference scores CSV missing columns: {missing}")

        buf: dict[str, list[float]] = {c: [] for c in columns}
        for row in reader:
            for c in columns:
                try:
                    v = float(row[c])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(v):
                    buf[c].append(v)

    stats: dict[str, dict[str, float]] = {}
    for c in columns:
        arr = np.asarray(buf[c], dtype=float)
        if arr.size < 2:
            raise ValueError(f"Reference column '{c}' has insufficient valid rows (<2).")
        std = float(arr.std())
        if not np.isfinite(std) or std == 0.0:
            raise ValueError(f"Reference column '{c}' has invalid std: {std}")
        stats[c] = {"mean": float(arr.mean()), "std": std, "n": int(arr.size)}
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Score all PCA ROIs for an RS run")
    parser.add_argument("--subj", required=True)
    parser.add_argument("--day", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--base-data", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--pca-input", choices=["auto", "mc", "reg"], default="reg")
    parser.add_argument(
        "--normalization",
        choices=["zscore", "demean", "none"],
        default="zscore",
        help="Voxel normalization before PCA projection. zscore is recommended for consistency with PCA training.",
    )
    parser.add_argument(
        "--score-metric",
        choices=["projection", "cosine"],
        default="projection",
        help="projection: weights @ voxels. cosine: cosine similarity between voxel vector and each PC.",
    )
    parser.add_argument(
        "--post-normalization",
        choices=["none", "zscore"],
        default="none",
        help="Optional second-stage normalization of per-component scores using a reference CSV.",
    )
    parser.add_argument(
        "--reference-scores-csv",
        type=Path,
        default=None,
        help="Reference scores CSV (same columns as output) used when --post-normalization=zscore.",
    )
    parser.add_argument("--input-4d", type=Path, default=None, help="Optional explicit 4D volume to score.")
    args = parser.parse_args()
    if args.post_normalization == "zscore" and args.reference_scores_csv is None:
        raise ValueError("--reference-scores-csv is required when --post-normalization=zscore")

    run_dir = _build_run_dir(args.base_data, args.subj, args.day, args.run)
    day_dir = run_dir.parent.parent
    pca_root = day_dir / "PCA" / run_dir.name / f"pca_{args.pca_input}"
    if not pca_root.exists():
        raise FileNotFoundError(f"PCA directory not found: {pca_root}")

    if args.input_4d is not None:
        score_4d_path = args.input_4d
    else:
        score_4d_path = run_dir / "analysis" / f"rs_4d_{args.pca_input}.nii.gz"
        if not score_4d_path.exists() and args.pca_input == "auto":
            for candidate in [run_dir / "analysis" / "rs_4d_reg.nii.gz", run_dir / "analysis" / "rs_4d_mc.nii.gz"]:
                if candidate.exists():
                    score_4d_path = candidate
                    break

    if not score_4d_path.exists():
        raise FileNotFoundError(f"Could not locate 4D scoring input: {score_4d_path}")

    data_4d = _load_4d(score_4d_path)
    n_trs = data_4d.shape[3]

    roi_decoders: dict[str, dict] = {}
    for roi_dir in _iter_roi_dirs(pca_root):
        roi = roi_dir.name
        roi_decoders[roi] = _load_decoder_artifacts(roi_dir)

    if not roi_decoders:
        raise RuntimeError(f"No ROI decoder folders found under {pca_root}")

    pc_counts = {roi: dec["weights"].shape[0] for roi, dec in roi_decoders.items()}
    max_pcs = max(pc_counts.values())

    out_csv = pca_root / "scores_pca_all_rois.csv"
    fieldnames = ["tr"]
    for roi in sorted(roi_decoders):
        for pc in range(1, max_pcs + 1):
            fieldnames.append(f"{roi}_PC{pc:02d}")
    reference_stats: dict[str, dict[str, float]] | None = None
    if args.post_normalization == "zscore":
        reference_stats = _load_reference_stats(args.reference_scores_csv, fieldnames[1:])
        for key in fieldnames[1:]:
            fieldnames.append(f"{key}_z")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for tr in range(n_trs):
            vol = data_4d[..., tr]
            row = {"tr": tr + 1}
            for roi in sorted(roi_decoders):
                dec = roi_decoders[roi]
                if args.score_metric == "projection":
                    pcs = _score_one_vol(
                        vol,
                        dec["voxel_indices"],
                        dec["weights"],
                        dec["norm_mean"],
                        dec["norm_std"],
                        normalization=args.normalization,
                    )
                else:
                    pcs = _cosine_one_vol(
                        vol,
                        dec["voxel_indices"],
                        dec["weights"],
                        dec["norm_mean"],
                        dec["norm_std"],
                        normalization=args.normalization,
                    )
                for idx in range(max_pcs):
                    key = f"{roi}_PC{idx + 1:02d}"
                    row[key] = float(pcs[idx]) if idx < pcs.shape[0] else ""
                    if reference_stats is not None and row[key] != "":
                        row[f"{key}_z"] = (float(row[key]) - reference_stats[key]["mean"]) / reference_stats[key]["std"]
                    elif reference_stats is not None:
                        row[f"{key}_z"] = ""
            writer.writerow(row)

    summary = {
        "subj": args.subj,
        "day": args.day,
        "run": args.run,
        "pca_root": str(pca_root),
        "input_4d": str(score_4d_path),
        "normalization": args.normalization,
        "score_metric": args.score_metric,
        "post_normalization": args.post_normalization,
        "reference_scores_csv": str(args.reference_scores_csv) if args.reference_scores_csv else None,
        "n_trs": int(n_trs),
        "rois": sorted(roi_decoders.keys()),
        "pc_counts": pc_counts,
        "scores_csv": str(out_csv),
    }
    with open(pca_root / "scores_pca_all_rois_metadata.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved PCA ROI scores: {out_csv}")


if __name__ == "__main__":
    main()
