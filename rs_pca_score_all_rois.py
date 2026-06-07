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
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np

from rs_pca_runtime import (
    build_pca_root,
    build_run_dir,
    load_decoder_artifacts,
    score_pca_volume,
)


def _find_3d_series(run_dir: Path, pca_input: str) -> list[Path]:
    folders = {
        "mc": ("mc", "_mc"),
        "reg": ("reg", "_reg"),
        "t1": ("t1", "_t1"),
        "unwarped": ("unwarped", "_mc_uw"),
    }
    if pca_input not in folders:
        return []
    folder_name, suffix = folders[pca_input]
    src_dir = run_dir / folder_name
    if not src_dir.exists():
        return []
    return sorted(
        [
            p
            for p in src_dir.iterdir()
            if p.is_file()
            and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))
            and suffix in p.name
        ]
    )


def _ensure_score_4d(run_dir: Path, pca_input: str) -> Path:
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_path = analysis_dir / f"rs_4d_{pca_input}.nii.gz"
    if out_path.exists():
        return out_path

    vols = _find_3d_series(run_dir, pca_input)
    if not vols:
        return out_path

    cmd = ["fslmerge", "-t", str(out_path), *[str(v) for v in vols]]
    subprocess.run(cmd, check=True)
    return out_path


def _load_4d(path: Path) -> np.ndarray:
    import nibabel as nib

    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D NIfTI, got shape={data.shape} at {path}")
    return data


def _iter_roi_dirs(pca_root: Path) -> Iterable[Path]:
    for d in sorted(pca_root.iterdir()):
        if d.is_dir() and (d / "decoder_metadata.json").exists():
            yield d


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


def _write_reference_stats(
    *,
    scores_csv: Path,
    columns: list[str],
    out_path: Path,
    metadata: dict,
) -> None:
    values: dict[str, list[float]] = {c: [] for c in columns}
    with scores_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for column in columns:
                try:
                    value = float(row[column])
                except (TypeError, ValueError, KeyError):
                    continue
                if np.isfinite(value):
                    values[column].append(value)

    payload = {"metadata": metadata, "columns": {}}
    for column, buf in values.items():
        arr = np.asarray(buf, dtype=float)
        if arr.size < 2:
            continue
        payload["columns"][column] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "n": int(arr.size),
            "p05": float(np.percentile(arr, 5)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score all PCA ROIs for an RS run")
    parser.add_argument("--subj", required=True)
    parser.add_argument("--day", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--base-data", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--pca-input", choices=["auto", "mc", "reg", "t1"], default="reg")
    parser.add_argument(
        "--decoder-run",
        default=None,
        help="Run whose PCA decoder bundles should be used. Defaults to --run.",
    )
    parser.add_argument(
        "--decoder-day",
        default=None,
        help="Day/session whose PCA decoder bundles should be used. Defaults to --day.",
    )
    parser.add_argument(
        "--decoder-pca-root",
        type=Path,
        default=None,
        help="Explicit PCA decoder root containing ROI decoder folders.",
    )
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
    parser.add_argument(
        "--reference-stats-out",
        type=Path,
        default=None,
        help="Write daily RS reference stats JSON for PCA feedback normalization.",
    )
    parser.add_argument(
        "--write-reference-stats",
        action="store_true",
        help="Write reference stats to the default score output folder.",
    )
    args = parser.parse_args()
    if args.post_normalization == "zscore" and args.reference_scores_csv is None:
        raise ValueError("--reference-scores-csv is required when --post-normalization=zscore")

    run_dir = build_run_dir(args.base_data, args.subj, args.day, args.run)
    day_dir = run_dir.parent.parent
    score_root = build_pca_root(day_dir, run_dir.name, args.pca_input)
    decoder_run = args.decoder_run or args.run
    decoder_day = args.decoder_day or args.day
    if args.decoder_pca_root is not None:
        decoder_pca_root = args.decoder_pca_root
    else:
        decoder_run_dir = build_run_dir(args.base_data, args.subj, decoder_day, decoder_run)
        decoder_day_dir = decoder_run_dir.parent.parent
        decoder_pca_root = build_pca_root(
            decoder_day_dir,
            decoder_run_dir.name,
            args.pca_input,
        )
    if not decoder_pca_root.exists():
        raise FileNotFoundError(f"PCA decoder directory not found: {decoder_pca_root}")

    if args.input_4d is not None:
        score_4d_path = args.input_4d
    else:
        if args.pca_input == "auto":
            score_4d_path = run_dir / "analysis" / "rs_4d_auto.nii.gz"
        else:
            score_4d_path = _ensure_score_4d(run_dir, args.pca_input)
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
    for roi_dir in _iter_roi_dirs(decoder_pca_root):
        roi = roi_dir.name
        roi_decoders[roi] = load_decoder_artifacts(roi_dir)

    if not roi_decoders:
        raise RuntimeError(f"No ROI decoder folders found under {decoder_pca_root}")

    pc_counts = {roi: dec["weights"].shape[0] for roi, dec in roi_decoders.items()}
    max_pcs = max(pc_counts.values())

    score_root.mkdir(parents=True, exist_ok=True)
    out_csv = score_root / "scores_pca_all_rois.csv"
    fieldnames = ["tr"]
    for roi in sorted(roi_decoders):
        for pc in range(1, max_pcs + 1):
            fieldnames.append(f"{roi}_PC{pc:02d}")
    score_columns = fieldnames[1:]
    reference_stats: dict[str, dict[str, float]] | None = None
    if args.post_normalization == "zscore":
        reference_stats = _load_reference_stats(args.reference_scores_csv, score_columns)
        for key in score_columns:
            fieldnames.append(f"{key}_z")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for tr in range(n_trs):
            vol = data_4d[..., tr]
            row = {"tr": tr + 1}
            for roi in sorted(roi_decoders):
                dec = roi_decoders[roi]
                pcs = score_pca_volume(
                    vol,
                    dec,
                    normalization=args.normalization,
                    score_metric=args.score_metric,
                )
                for idx in range(max_pcs):
                    key = f"{roi}_PC{idx + 1:02d}"
                    row[key] = float(pcs[idx]) if idx < pcs.shape[0] else ""
                    if reference_stats is not None and row[key] != "":
                        row[f"{key}_z"] = (
                            float(row[key]) - reference_stats[key]["mean"]
                        ) / reference_stats[key]["std"]
                    elif reference_stats is not None:
                        row[f"{key}_z"] = ""
            writer.writerow(row)

    summary = {
        "subj": args.subj,
        "day": args.day,
        "run": args.run,
        "score_root": str(score_root),
        "decoder_day": decoder_day,
        "decoder_run": decoder_run,
        "decoder_pca_root": str(decoder_pca_root),
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
    with open(score_root / "scores_pca_all_rois_metadata.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    stats_out = args.reference_stats_out
    if stats_out is None and args.write_reference_stats:
        stats_out = score_root / "pca_reference_stats.json"
    if stats_out is not None:
        _write_reference_stats(
            scores_csv=out_csv,
            columns=score_columns,
            out_path=stats_out,
            metadata=summary,
        )
        summary["reference_stats_json"] = str(stats_out)
        with open(score_root / "scores_pca_all_rois_metadata.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print(f"Saved PCA ROI scores: {out_csv}")


if __name__ == "__main__":
    main()
