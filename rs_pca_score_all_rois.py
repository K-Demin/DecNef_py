#!/usr/bin/env python
"""
Daily resting-state realtime PCA scoring.

This script mirrors the realtime RS/NF path: it waits for scanner DICOMs,
runs the normal rt_pipeline processing with an empty display, and scores all
PCA ROI components as processed realtime volumes appear.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing as mp
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Full
from typing import Optional

import numpy as np

import rs_pca_runtime as pca_rt
from rs_realtime_parallel import (
    _merge_session_metadata,
    _plot_qc,
    _run_biopac_listener,
    _run_pipeline_with_settings,
    _run_prep_surface_rois,
    _run_preproc,
    _wait_for_epi_dicoms,
    run_fixation_presentation,
)
from rt_global_settings import load_regressor_settings


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rs_pca_score_all_rois")


def _iter_roi_dirs(pca_root: Path):
    for d in sorted(pca_root.iterdir()):
        if d.is_dir() and (d / "decoder_metadata.json").exists():
            yield d


def _load_all_roi_decoders(pca_root: Path) -> dict[str, dict[str, np.ndarray]]:
    decoders: dict[str, dict[str, np.ndarray]] = {}
    for roi_dir in _iter_roi_dirs(pca_root):
        decoders[roi_dir.name] = pca_rt.load_decoder_artifacts(roi_dir)
    if not decoders:
        raise RuntimeError(f"No ROI decoder folders found under {pca_root}")
    return decoders


def _score_columns(pc_counts: dict[str, int]) -> list[str]:
    columns: list[str] = []
    for roi in sorted(pc_counts):
        for pc in range(1, pc_counts[roi] + 1):
            columns.append(f"{roi}_PC{pc:02d}")
    return columns


def _load_reg_ready_map(run_dir: Path) -> Optional[dict[int, bool]]:
    status_path = run_dir / "regression_status_rt.csv"
    if not status_path.exists():
        return None
    ready: dict[int, bool] = {}
    with status_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "volume_idx" not in reader.fieldnames or "reg_ready" not in reader.fieldnames:
            return None
        for row in reader:
            try:
                ready[int(row["volume_idx"])] = bool(int(row["reg_ready"]))
            except (TypeError, ValueError):
                continue
    return ready


def _requires_regression_ready_filter(volume_kind: str) -> bool:
    return str(volume_kind).lower() in {"reg", "t1", "mni"}


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
            raise ValueError(f"Reference column {column!r} has fewer than 2 valid samples")
        std = float(arr.std())
        if not np.isfinite(std) or std == 0.0:
            raise ValueError(f"Reference column {column!r} has invalid std: {std}")
        payload["columns"][column] = {
            "mean": float(arr.mean()),
            "std": std,
            "n": int(arr.size),
            "p05": float(np.percentile(arr, 5)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_realtime_all_roi_pca_scorer(
    *,
    run_dir: Path,
    pca_root: Path,
    score_root: Path,
    score_queue,
    stop_event,
    reference_stats_out: Optional[Path],
    volume_kind: str,
    normalization: str,
    score_metric: str,
    max_trs: Optional[int],
    poll_interval: float,
    metadata: dict,
) -> None:
    import nibabel as nib

    decoders = _load_all_roi_decoders(pca_root)
    pc_counts = {roi: int(dec["weights"].shape[0]) for roi, dec in decoders.items()}
    score_columns = _score_columns(pc_counts)
    fieldnames = ["volume_idx", "timestamp", *score_columns]

    score_root.mkdir(parents=True, exist_ok=True)
    out_csv = score_root / "scores_pca_all_rois.csv"
    require_reg_ready = _requires_regression_ready_filter(volume_kind)
    filter_meta = {
        "source": "regression_status_rt.csv" if require_reg_ready else "none",
        "require_regression_ready": require_reg_ready,
        "drop_first_ready": require_reg_ready,
        "first_reg_ready_volume": None,
        "excluded_through_volume": None,
        "skipped_not_ready": 0,
        "skipped_first_ready": 0,
    }

    processed: set[int] = set()
    scored_count = 0
    volume_idx = 1

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        while True:
            if max_trs is not None and volume_idx > max_trs:
                break
            if volume_idx in processed:
                volume_idx += 1
                continue

            vol_path = pca_rt.volume_path_for_kind(run_dir, volume_idx, volume_kind)
            if not vol_path.exists():
                if stop_event.is_set():
                    break
                time.sleep(poll_interval)
                continue

            if require_reg_ready:
                reg_ready_map = _load_reg_ready_map(run_dir)
                if reg_ready_map is None or volume_idx not in reg_ready_map:
                    if stop_event.is_set():
                        break
                    time.sleep(poll_interval)
                    continue
                if not reg_ready_map[volume_idx]:
                    filter_meta["skipped_not_ready"] += 1
                    processed.add(volume_idx)
                    volume_idx += 1
                    continue
                if filter_meta["first_reg_ready_volume"] is None:
                    filter_meta["first_reg_ready_volume"] = volume_idx
                    filter_meta["excluded_through_volume"] = volume_idx
                    filter_meta["skipped_first_ready"] = 1
                    processed.add(volume_idx)
                    volume_idx += 1
                    continue

            try:
                img = nib.load(str(vol_path))
                vol = np.asanyarray(img.dataobj)
                row = {
                    "volume_idx": volume_idx,
                    "timestamp": time.time(),
                }
                for roi in sorted(decoders):
                    scores = pca_rt.score_pca_volume(
                        vol,
                        decoders[roi],
                        normalization=normalization,
                        score_metric=score_metric,
                    )
                    for pc_idx, value in enumerate(scores, start=1):
                        row[f"{roi}_PC{pc_idx:02d}"] = float(value)
                writer.writerow(row)
                f.flush()
                scored_count += 1
                try:
                    score_queue.put_nowait({"volume_idx": volume_idx})
                except Full:
                    pass
            except Exception as exc:
                log.exception("Failed PCA scoring for volume %05d: %s", volume_idx, exc)
            finally:
                processed.add(volume_idx)
                volume_idx += 1

    summary = {
        **metadata,
        "score_root": str(score_root),
        "pca_root": str(pca_root),
        "volume_kind": volume_kind,
        "normalization": normalization,
        "score_metric": score_metric,
        "n_scored": int(scored_count),
        "n_seen_or_skipped": int(len(processed)),
        "initial_volume_filter": filter_meta,
        "rois": sorted(decoders.keys()),
        "pc_counts": pc_counts,
        "scores_csv": str(out_csv),
    }
    with (score_root / "scores_pca_all_rois_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    stats_out = reference_stats_out or (score_root / "pca_reference_stats.json")
    if scored_count < 2:
        raise ValueError(f"Need at least 2 scored TRs for PCA reference stats, got {scored_count}")
    _write_reference_stats(
        scores_csv=out_csv,
        columns=score_columns,
        out_path=stats_out,
        metadata=summary,
    )
    summary["reference_stats_json"] = str(stats_out)
    with (score_root / "scores_pca_all_rois_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved realtime PCA ROI scores: %s", out_csv)
    log.info("Saved realtime PCA reference stats: %s", stats_out)


def _coalesce_sub(args) -> str:
    sub = args.sub or args.subj
    if not sub:
        raise ValueError("Provide --sub")
    return str(sub)


def main() -> None:
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(
        description="Run daily realtime RS with blank screen and score all PCA ROIs."
    )
    parser.add_argument("--sub", default=None, help="Subject ID, e.g. 00086")
    parser.add_argument("--subj", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--day", required=True, help="Day/session, e.g. 3")
    parser.add_argument("--run", required=True, help="Run number, e.g. 4")
    parser.add_argument(
        "--incoming-root",
        required=True,
        help="Folder where scanner writes DICOMs in real time.",
    )
    parser.add_argument(
        "--base-data",
        default="/SSD2/DecNef_py/data",
        help="Base preproc data folder.",
    )
    parser.add_argument("--struct-block", type=int)
    parser.add_argument("--ap-block", type=int)
    parser.add_argument("--pa-block", type=int)
    parser.add_argument(
        "--epi-block",
        type=int,
        help="Block/run id for EPI DICOMs inside incoming-root; defaults to --run.",
    )
    parser.add_argument(
        "--wait-epi-min",
        type=int,
        default=20,
        help="Minimum EPI DICOMs to wait for before offline preproc.",
    )
    parser.add_argument(
        "--prep-surface-rois",
        action="store_true",
        help="Run fmri_rt_preproc.prep_surface_rois after offline preproc.",
    )
    parser.add_argument(
        "--pca-day",
        default=None,
        help="Day/session whose PCA decoder bundles should be used. Defaults to --day.",
    )
    parser.add_argument(
        "--pca-run",
        default=None,
        help="Run whose PCA decoder bundles should be used. Defaults to --run.",
    )
    parser.add_argument(
        "--pca-root",
        type=Path,
        default=None,
        help="Explicit PCA decoder root containing ROI folders.",
    )
    parser.add_argument(
        "--pca-input",
        choices=["auto", "mc", "reg", "t1"],
        default="t1",
        help="PCA decoder folder name.",
    )
    parser.add_argument(
        "--pca-space",
        choices=["epi", "t1", "mni"],
        default="t1",
        help="rt_pipeline output space to create for PCA scoring.",
    )
    parser.add_argument(
        "--pca-reference-image",
        type=Path,
        default=None,
        help="Explicit reference image/grid for PCA-space transforms.",
    )
    parser.add_argument(
        "--pca-reference-resolution",
        choices=["epi", "t1"],
        default="epi",
        help="Default PCA T1 reference resolution when --pca-reference-image is not provided.",
    )
    parser.add_argument(
        "--pca-reference-stats-out",
        type=Path,
        default=None,
        help="Output JSON for daily PCA reference stats.",
    )
    parser.add_argument(
        "--pca-volume-kind",
        choices=["reg", "mc", "unwarped", "t1", "mni"],
        default=None,
        help="Realtime volume folder to score. Defaults from --pca-space.",
    )
    parser.add_argument(
        "--pca-normalization",
        choices=["zscore", "demean", "none"],
        default="zscore",
        help="Voxel normalization before PCA projection.",
    )
    parser.add_argument(
        "--pca-score-metric",
        choices=["projection", "cosine"],
        default="projection",
        help="PCA score metric.",
    )
    parser.add_argument(
        "--pca-poll-interval",
        type=float,
        default=0.05,
        help="Seconds between checks for newly processed realtime volumes.",
    )
    parser.add_argument("--max-trs", type=int, default=None)
    parser.add_argument(
        "--duration-min",
        type=float,
        default=None,
        help="Stop after this many minutes using TR from settings. Ignored if --max-trs is set.",
    )
    parser.add_argument(
        "--skip-first-trs",
        type=int,
        default=10,
        help="TRs to label as warmup in session metadata.",
    )
    parser.add_argument(
        "--baseline-trs",
        type=int,
        default=None,
        help="TRs for voxel-wise normalization baseline.",
    )
    parser.add_argument("--settings-file", default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--rt-max-scan-length", type=int, default=None)
    parser.add_argument("--biopac-enable", action="store_true")
    parser.add_argument("--biopac-host", default="0.0.0.0")
    parser.add_argument("--biopac-port", type=int, default=15000)
    parser.add_argument("--biopac-timeout", type=float, default=0.3)
    parser.add_argument(
        "--biopac-phys-reg",
        default="RICOR8",
        choices=["RICOR8", "RVT5", "RVT+RICOR13"],
    )
    parser.add_argument("--biopac-handshake", action="store_true", default=True)
    parser.add_argument("--biopac-start-online", action="store_true", default=False)
    parser.add_argument("--biopac-mode", default="tcp", choices=["tcp", "file"])
    parser.add_argument("--biopac-file", default=None)
    parser.add_argument("--biopac-poll", type=float, default=0.05)
    parser.add_argument("--biopac-listener", action="store_true")
    args = parser.parse_args()

    from biopac_rt.biopac_receiver import BiopacReceiverConfig
    from rt_pipeline import REGRESSOR_SETTINGS, RTSessionConfig

    if args.settings_file:
        loaded = load_regressor_settings(args.settings_file)
        REGRESSOR_SETTINGS.update(vars(loaded))

    sub = _coalesce_sub(args)
    max_trs = args.max_trs
    if max_trs is None and args.duration_min is not None:
        max_trs = int(round((float(args.duration_min) * 60.0) / float(REGRESSOR_SETTINGS.TR)))
    if args.skip_first_trs < 0:
        raise ValueError("--skip-first-trs must be >= 0")
    baseline_trs = (
        int(args.baseline_trs)
        if args.baseline_trs is not None
        else int(REGRESSOR_SETTINGS.voxel_norm_ref_volumes)
    )
    if baseline_trs < 0:
        raise ValueError("--baseline-trs must be >= 0")

    try:
        epi_block = int(args.epi_block if args.epi_block is not None else args.run)
    except ValueError as exc:
        raise ValueError("--run must be numeric when --epi-block is not provided.") from exc

    incoming_root = Path(args.incoming_root)
    base_data = Path(args.base_data)
    if not incoming_root.exists():
        raise FileNotFoundError(f"Incoming directory does not exist: {incoming_root}")

    subject_root = base_data / f"sub-{sub}"
    run_dir = subject_root / args.day / "func" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    score_queue = ctx.Queue(maxsize=100)
    fixation_stop = ctx.Event()
    fixation_process = ctx.Process(
        target=run_fixation_presentation,
        args=(score_queue, max_trs, fixation_stop),
    )
    fixation_process.start()

    _merge_session_metadata(
        run_dir,
        {
            "fixation_display": {
                "description": "Grey screen only; daily RS PCA all-ROI scoring.",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "skip_first_trs": args.skip_first_trs,
                "baseline_trs": baseline_trs,
            }
        },
    )

    biopac_process = None
    biopac_stop = None
    try:
        if args.biopac_listener:
            if not args.biopac_enable:
                raise ValueError("--biopac-listener requires --biopac-enable")
            if args.biopac_mode != "file":
                raise ValueError("--biopac-listener requires --biopac-mode=file")
            biopac_output = Path(args.biopac_file) if args.biopac_file else (run_dir / "biopac_regressors_rx.csv")
            args.biopac_file = str(biopac_output)
            biopac_stop = ctx.Event()
            expected_regressors = {
                "RICOR8": 8,
                "RVT5": 5,
                "RVT+RICOR13": 13,
            }.get(args.biopac_phys_reg, 8)
            biopac_cfg = BiopacReceiverConfig(
                host=args.biopac_host,
                port=args.biopac_port,
                timeout=args.biopac_timeout,
                expected_regressors=expected_regressors,
                handshake_tr=REGRESSOR_SETTINGS.TR if args.biopac_handshake else None,
                subject=sub,
                day=args.day,
                run=args.run,
                output_path=biopac_output,
            )
            biopac_process = ctx.Process(
                target=_run_biopac_listener,
                args=(biopac_cfg, biopac_stop),
            )
            biopac_process.start()

        _wait_for_epi_dicoms(incoming_root, epi_block, min_vols=args.wait_epi_min)
        _run_preproc(
            sub=sub,
            day=args.day,
            base_data=base_data,
            incoming_root=incoming_root,
            struct_block=args.struct_block,
            ap_block=args.ap_block,
            pa_block=args.pa_block,
            epi_block=epi_block,
        )
        if args.prep_surface_rois:
            _run_prep_surface_rois(base_data, sub, args.day)

        pca_reference_image = None
        if args.pca_space in {"t1", "mni"}:
            trans_dir = subject_root / args.day / "func" / "trans"
            pca_reference_image = pca_rt.ensure_pca_t1_reference(
                subject_root,
                trans_dir,
                args.pca_reference_image,
                resolution=args.pca_reference_resolution,
            )

        pca_day = args.pca_day or args.day
        pca_run = args.pca_run or args.run
        if args.pca_root is not None:
            pca_root = args.pca_root
        else:
            pca_run_dir = pca_rt.build_run_dir(base_data, sub, pca_day, pca_run)
            pca_root = pca_rt.build_pca_root(pca_run_dir.parent.parent, pca_run_dir.name, args.pca_input)
        if not pca_root.exists():
            raise FileNotFoundError(f"PCA decoder root not found: {pca_root}")

        score_root = pca_rt.build_pca_root(run_dir.parent.parent, run_dir.name, args.pca_input)
        pca_volume_kind = args.pca_volume_kind
        if pca_volume_kind is None:
            pca_volume_kind = "reg" if args.pca_space == "epi" else args.pca_space

        settings_payload = vars(REGRESSOR_SETTINGS).copy()
        settings_payload.update(
            {
                "skip_first_trs": args.skip_first_trs,
                "voxel_norm_ref_volumes": max(1, baseline_trs),
                "enable_biopac_physio": args.biopac_enable,
                "biopac_host": args.biopac_host,
                "biopac_port": args.biopac_port,
                "biopac_timeout": args.biopac_timeout,
                "biopac_phys_reg": args.biopac_phys_reg,
                "biopac_handshake": args.biopac_handshake,
                "biopac_start_online_only": args.biopac_start_online,
                "biopac_mode": args.biopac_mode,
                "biopac_file": Path(args.biopac_file) if args.biopac_file else None,
                "biopac_poll_interval": args.biopac_poll,
                "analysis_space": args.pca_space,
                "truncate_t1_to_epi_fov": False,
            }
        )
        if args.max_workers is not None:
            settings_payload["max_workers"] = max(1, int(args.max_workers))
        if args.rt_max_scan_length is not None:
            settings_payload["rt_max_scan_length"] = max(1, int(args.rt_max_scan_length))

        cfg = RTSessionConfig(
            subject=sub,
            day=args.day,
            run=args.run,
            incoming_root=incoming_root,
            base_data=base_data,
            decoder_template=pca_reference_image,
            enable_scoring=False,
        )

        metadata = {
            "subj": sub,
            "day": args.day,
            "run": args.run,
            "decoder_day": pca_day,
            "decoder_run": pca_run,
            "decoder_pca_root": str(pca_root),
            "pca_input": args.pca_input,
            "pca_space": args.pca_space,
            "pca_reference_image": str(pca_reference_image) if pca_reference_image else None,
            "pca_reference_resolution": args.pca_reference_resolution,
        }
        _merge_session_metadata(
            run_dir,
            {
                "pca_daily_rs": {
                    **metadata,
                    "score_root": str(score_root),
                    "volume_kind": pca_volume_kind,
                    "normalization": args.pca_normalization,
                    "score_metric": args.pca_score_metric,
                    "reference_stats_out": (
                        str(args.pca_reference_stats_out)
                        if args.pca_reference_stats_out
                        else str(score_root / "pca_reference_stats.json")
                    ),
                }
            },
        )

        pipeline_process = ctx.Process(
            target=_run_pipeline_with_settings,
            args=(cfg, score_queue, settings_payload),
        )
        pca_stop = ctx.Event()
        pca_process = ctx.Process(
            target=run_realtime_all_roi_pca_scorer,
            kwargs={
                "run_dir": run_dir,
                "pca_root": pca_root,
                "score_root": score_root,
                "score_queue": score_queue,
                "stop_event": pca_stop,
                "reference_stats_out": args.pca_reference_stats_out,
                "volume_kind": pca_volume_kind,
                "normalization": args.pca_normalization,
                "score_metric": args.pca_score_metric,
                "max_trs": max_trs,
                "poll_interval": args.pca_poll_interval,
                "metadata": metadata,
            },
        )
        pipeline_process.start()
        pca_process.start()

        pipeline_process.join()
        pca_stop.set()
        pca_process.join(timeout=60)
        if pca_process.is_alive():
            pca_process.terminate()
            pca_process.join(timeout=5)
            raise RuntimeError("Realtime PCA all-ROI scorer did not finish cleanly")
        if pipeline_process.exitcode not in (0, None):
            raise RuntimeError(f"rt_pipeline failed with exit code {pipeline_process.exitcode}")
        if pca_process.exitcode not in (0, None):
            raise RuntimeError(f"Realtime PCA all-ROI scorer failed with exit code {pca_process.exitcode}")
        _plot_qc(run_dir)
    finally:
        fixation_stop.set()
        fixation_process.join(timeout=5)
        if biopac_stop is not None:
            biopac_stop.set()
        if biopac_process is not None:
            biopac_process.join(timeout=5)


if __name__ == "__main__":
    main()
