#!/usr/bin/env python
import argparse
import csv
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import Queue
from queue import Empty
from pathlib import Path
import multiprocessing as mp
import time
from typing import Optional

import numpy as np

from rt_global_settings import load_regressor_settings
import rs_pca_runtime as pca_rt

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rs_realtime_parallel")

DICOM_LIKE_SUFFIXES = {".dcm", ".ima"}


@dataclass(frozen=True)
class Condition:
    condition_id: str
    roi: str
    direction: str
    symbol: str


def _merge_session_metadata(run_dir: Path, payload: dict) -> None:
    metadata_path = run_dir / "session_metadata.json"
    data = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            data = {}
    data.update(payload)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_reg_ready_map(run_dir: Path) -> Optional[dict[int, bool]]:
    reg_path = run_dir / "regression_status_rt.csv"
    if not reg_path.exists():
        return None
    with open(reg_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "volume_idx" not in reader.fieldnames or "reg_ready" not in reader.fieldnames:
            return None
        reg_ready_map: dict[int, bool] = {}
        for row in reader:
            try:
                vol = int(row["volume_idx"])
                reg_ready_map[vol] = bool(int(row["reg_ready"]))
            except (TypeError, ValueError):
                continue
    return reg_ready_map


def _is_dicom_like(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in DICOM_LIKE_SUFFIXES


def _parse_dicom_name(name: str) -> Optional[tuple[int, int, int]]:
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) != 3:
        return None
    try:
        series_id = int(parts[0])
        run_id = int(parts[1])
        scan = int(parts[2])
    except ValueError:
        return None
    return series_id, run_id, scan


def _count_epi_dicoms(
    incoming_dir: Path,
    epi_block: int,
    keep_start: int = 11,
    keep_end: int = 30,
) -> int:
    count = 0
    for f in incoming_dir.iterdir():
        if not _is_dicom_like(f):
            continue
        parsed = _parse_dicom_name(f.name)
        if parsed is None:
            continue
        _, run_id, scan = parsed
        if run_id == epi_block and keep_start <= scan <= keep_end:
            count += 1
    return count


def _wait_for_epi_dicoms(
    incoming_dir: Path,
    epi_block: int,
    min_vols: int = 20,
    keep_start: int = 11,
    keep_end: int = 30,
    poll_interval: float = 1.0,
) -> None:
    log.info(
        "Waiting for at least %d EPI DICOMs (block %s, scans %d-%d) in %s",
        min_vols,
        epi_block,
        keep_start,
        keep_end,
        incoming_dir,
    )
    while True:
        count = _count_epi_dicoms(incoming_dir, epi_block, keep_start, keep_end)
        if count >= min_vols:
            log.info("Detected %d EPI DICOMs; proceeding with preprocessing.", count)
            return
        time.sleep(poll_interval)


def _run_preproc(
    sub: str,
    day: str,
    base_data: Path,
    incoming_root: Path,
    struct_block: Optional[int],
    ap_block: Optional[int],
    pa_block: Optional[int],
    epi_block: int,
) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_preproc.py"),
        "--sub",
        sub,
        "--day",
        day,
        "--base-data",
        str(base_data),
        "--incoming-root",
        str(incoming_root),
        "--epi-block",
        str(epi_block),
    ]
    if struct_block is not None:
        cmd += ["--struct-block", str(struct_block)]
    if ap_block is not None:
        cmd += ["--ap-block", str(ap_block)]
    if pa_block is not None:
        cmd += ["--pa-block", str(pa_block)]
    log.info("Running offline preprocessing: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _run_prep_surface_rois(base_data: Path, sub: str, day: str) -> None:
    cmd = [
        sys.executable,
        "-m",
        "fmri_rt_preproc.prep_surface_rois",
        "--root",
        str(base_data),
        "--subj",
        sub,
        "--day",
        day,
    ]
    log.info("Running surface ROI preparation: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _run_pca_prep(base_data: Path, sub: str, day: str, run: str, pca_input: str) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "roi_rs_pca_decoder_prep.py"),
        "-subj",
        sub,
        "-day",
        day,
        "-run",
        run,
        "--base-data",
        str(base_data),
        "--pca-input",
        pca_input,
    ]
    log.info("Running PCA prep: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _run_pca_score(
    *,
    base_data: Path,
    sub: str,
    day: str,
    run: str,
    decoder_day: Optional[str],
    decoder_run: Optional[str],
    pca_input: str,
    normalization: str,
    score_metric: str,
    reference_stats_out: Optional[Path],
) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "rs_pca_score_all_rois.py"),
        "--subj",
        sub,
        "--day",
        day,
        "--run",
        run,
        "--base-data",
        str(base_data),
        "--pca-input",
        pca_input,
        "--normalization",
        normalization,
        "--score-metric",
        score_metric,
    ]
    if decoder_day is not None:
        cmd += ["--decoder-day", decoder_day]
    if decoder_run is not None:
        cmd += ["--decoder-run", decoder_run]
    if reference_stats_out is not None:
        cmd += ["--reference-stats-out", str(reference_stats_out)]
    else:
        cmd += ["--write-reference-stats"]
    log.info("Running PCA score/reference stats: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _plot_qc(run_dir: Path, prefer_reg_ready: bool = True) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores_path = run_dir / "scores.csv"
    motion_path = run_dir / "motion_rt.1D"
    if not scores_path.exists() or not motion_path.exists():
        return

    reg_ready_map = _load_reg_ready_map(run_dir) if prefer_reg_ready else None
    if prefer_reg_ready and reg_ready_map is None:
        return

    first_reg_ready_vol = None
    if reg_ready_map:
        ready_vols = [vol for vol, ready in reg_ready_map.items() if ready]
        if ready_vols:
            first_reg_ready_vol = min(ready_vols)

    qc_exclude_until_vol = 0
    metadata_path = run_dir / "session_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            qc_exclude_until_vol = max(
                int(metadata.get("voxel_norm_ref_volumes", 0) or 0),
                int(metadata.get("pre_trial_scans", 0) or 0),
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            qc_exclude_until_vol = 0

    vols = []
    scores = []
    with open(scores_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                vol = int(row["volume_idx"])
                score = float(row["score_raw"])
            except (TypeError, ValueError):
                continue
            if reg_ready_map is not None and not reg_ready_map.get(vol, False):
                continue
            if first_reg_ready_vol is not None and vol <= first_reg_ready_vol:
                continue
            if vol <= qc_exclude_until_vol:
                continue
            vols.append(vol)
            scores.append(score)

    if not scores:
        return

    motion = np.loadtxt(motion_path)
    if motion.ndim == 1:
        motion = motion[None, :]

    motion_vols = np.arange(1, motion.shape[0] + 1)
    include_motion = np.isin(motion_vols, np.asarray(vols, dtype=int))
    motion = motion[include_motion]
    motion_vols = motion_vols[include_motion]

    if motion.shape[0] == 0:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    axes[0].plot(vols, scores, label="Decoder score (regressed)")
    axes[0].set_xlabel("Volume")
    axes[0].set_ylabel("Score")
    axes[0].legend(loc="upper right")

    for idx in range(min(motion.shape[1], 6)):
        axes[1].plot(motion_vols, motion[:, idx], label=f"Motion {idx + 1}")
    axes[1].set_xlabel("Volume")
    axes[1].set_ylabel("Motion")
    axes[1].legend(loc="upper right", ncol=3, fontsize=8)

    fig.tight_layout()
    out_png = run_dir / "qc_scores_motion.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _run_biopac_listener(config: "BiopacReceiverConfig", stop_event: mp.Event) -> None:
    from biopac_rt.biopac_receiver import BiopacRetroTSReceiver

    receiver = BiopacRetroTSReceiver(config)
    receiver.start()
    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    finally:
        receiver.stop()


def _run_pipeline_with_settings(cfg: "RTSessionConfig", score_queue: Queue, settings: dict) -> None:
    from rt_pipeline import REGRESSOR_SETTINGS, run_rt_pipeline

    REGRESSOR_SETTINGS.update(settings)
    run_rt_pipeline(cfg, score_queue)

def _fieldmap_pair_dir(
    base_data: Path,
    sub: str,
    day: str,
    ap_block: Optional[int],
    pa_block: Optional[int],
) -> Path:
    fmap_root = base_data / f"sub-{sub}" / str(day) / "fmap"
    if ap_block is None or pa_block is None:
        return fmap_root
    return fmap_root / f"pair-ap{ap_block:03d}_pa{pa_block:03d}"

def _check_rt_fieldmap_exists(fmap_dir: Path) -> None:
    candidates = [
        fmap_dir / "pyhysco_epi-EstFieldMap.nii",
        fmap_dir / "pyhysco_epi-EstFieldMap.nii.gz",
        fmap_dir / "pyhysco-EstFieldMap.nii",
        fmap_dir / "pyhysco-EstFieldMap.nii.gz",
    ]
    if not any(p.exists() for p in candidates):
        raise FileNotFoundError(
            "No PyHySCO fieldmap found for realtime unwarping.\n"
            f"Expected one of:\n  " + "\n  ".join(str(p) for p in candidates)
        )


def _build_presentation_window(visual, color):
    default_size = (1000, 700)
    window_kwargs = {
        "size": default_size,
        "color": color,
        "units": "pix",
        "screen": 0,
        "fullscr": False,
    }
    try:
        import pyglet

        screens = pyglet.canvas.get_display().get_screens()
        if len(screens) > 1:
            second_screen = screens[1]
            window_kwargs.update(
                {
                    "size": (second_screen.width, second_screen.height),
                    "screen": 1,
                    "fullscr": True,
                }
            )

    except Exception as exc:
        log.warning("Could not detect external monitor; using default window size: %s", exc)
    return visual.Window(**window_kwargs)


def run_fixation_presentation(
    score_queue: Queue,
    max_trs: Optional[int],
    stop_event: mp.Event,
) -> None:
    from psychopy import core, event, visual

    win = _build_presentation_window(visual, color=[-0.004, -0.004, -0.004])
    seen_vols: set[int] = set()

    while True:

        win.fullscr = True
        win.flip()
        try:
            while True:
                message = score_queue.get_nowait()
                vol_idx = int(message.get("volume_idx", 0))
                if vol_idx:
                    seen_vols.add(vol_idx)
        except Empty:
            pass

        if max_trs is not None and len(seen_vols) >= max_trs:
            break
        if stop_event.is_set():
            break
        if "escape" in event.getKeys():
            break
        core.wait(0.02)

    win.close()


def main() -> None:
    mp.set_start_method("spawn", force=True)  # <-- IMPORTANT: before CUDA touches anything
    parser = argparse.ArgumentParser(
        description=(
            "Run rt_pipeline in parallel with a grey fixation display (no score output)."
        )
    )
    parser.add_argument("--sub", required=True, help="Subject ID, e.g. 00086")
    parser.add_argument("--day", required=True, help="Day/session, e.g. 3")
    parser.add_argument("--run", required=True, help="Run number, e.g. 4")
    parser.add_argument(
        "--incoming-root",
        required=False,
        default="/home/sin/DecNef_pain_Dec23/realtime/incoming/pain7T/20251105.20251105_00085.Kostya",
        help="Folder where scanner writes DICOMs in real-time.",
    )
    parser.add_argument(
        "--base-data",
        required=False,
        default="/SSD2/DecNef_py/data",
        help="Base preproc data folder (same as offline pipeline).",
    )
    parser.add_argument(
        "--decoder-template",
        required=False,
        help="Optional decoder template path to override the default.",
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Disable decoder scoring during rt_pipeline.",
    )
    parser.add_argument(
        "--struct-block",
        type=int,
        help="Block/run id for the structural (UNI/T1) DICOMs inside incoming-root.",
    )
    parser.add_argument(
        "--ap-block",
        type=int,
        help="Block/run id for the AP fieldmap DICOMs inside incoming-root.",
    )
    parser.add_argument(
        "--pa-block",
        type=int,
        help="Block/run id for the PA fieldmap DICOMs inside incoming-root.",
    )
    parser.add_argument(
        "--epi-block",
        type=int,
        help="Block/run id for the EPI DICOMs inside incoming-root (defaults to --run).",
    )
    parser.add_argument(
        "--wait-epi-min",
        type=int,
        default=20,
        help="Minimum EPI DICOMs to wait for before running offline preproc.",
    )
    parser.add_argument(
        "--prep-surface-rois",
        action="store_true",
        help="Run fmri_rt_preproc.prep_surface_rois after offline preproc.",
    )
    parser.add_argument(
        "--pca-prep",
        action="store_true",
        help="Run roi_rs_pca_decoder_prep.py after the run completes.",
    )
    parser.add_argument(
        "--pca-score",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--pca-day",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--pca-run",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--pca-input",
        choices=["auto", "mc", "reg", "t1"],
        default="t1",
        help="PCA input/output mode used for PCA prep and scoring.",
    )
    parser.add_argument(
        "--pca-space",
        choices=["epi", "t1", "mni"],
        default="t1",
        help="rt_pipeline output space to create for PCA workflows.",
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
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--pca-normalization",
        choices=["zscore", "demean", "none"],
        default="zscore",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--pca-score-metric",
        choices=["projection", "cosine"],
        default="projection",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--biopac-enable",
        action="store_true",
        help="Enable BIOPAC RetroTS regressors via TCP/file input.",
    )
    parser.add_argument(
        "--biopac-host",
        default="0.0.0.0",
        help="Host to bind BIOPAC receiver.",
    )
    parser.add_argument(
        "--biopac-port",
        type=int,
        default=15000,
        help="Port to bind BIOPAC receiver.",
    )
    parser.add_argument(
        "--biopac-timeout",
        type=float,
        default=0.3,
        help="Seconds to wait for physio regressors before zero-fill.",
    )
    parser.add_argument(
        "--biopac-phys-reg",
        default="RICOR8",
        choices=["RICOR8", "RVT5", "RVT+RICOR13"],
        help="Physio regressor family to expect from BIOPAC stream.",
    )
    parser.add_argument(
        "--biopac-handshake",
        action="store_true",
        default=True,
        help="Send a handshake with TR to the BIOPAC streamer.",
    )
    parser.add_argument(
        "--biopac-start-online",
        action="store_true",
        default=False,
        help="Defer BIOPAC receiver start until after offline DICOM processing.",
    )
    parser.add_argument(
        "--biopac-mode",
        default="tcp",
        choices=["tcp", "file"],
        help="BIOPAC input mode: tcp (listen on socket) or file (tail CSV).",
    )
    parser.add_argument(
        "--biopac-file",
        default=None,
        help="Path to BIOPAC regressors CSV when using --biopac-mode=file.",
    )
    parser.add_argument(
        "--biopac-poll",
        type=float,
        default=0.05,
        help="Polling interval (seconds) for file-backed BIOPAC buffer.",
    )
    parser.add_argument(
        "--biopac-listener",
        action="store_true",
        help="Spawn a dedicated BIOPAC listener process that writes regressors to CSV.",
    )
    parser.add_argument(
        "--max-trs",
        type=int,
        default=None,
        help="Stop the run after this many TRs (or press ESC).",
    )
    parser.add_argument(
        "--skip-first-trs",
        type=int,
        default=10,
        help="TRs to exclude from downstream analyses (warmup after trigger).",
    )
    parser.add_argument(
        "--baseline-trs",
        type=int,
        default=None,
        help="TRs for voxel-wise normalization baseline (defaults to settings voxel_norm_ref_volumes).",
    )
    parser.add_argument(
        "--settings-file",
        default=None,
        help="Optional JSON file with global runtime settings (TR, censor thresholds, BIOPAC defaults, etc.).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum parallel processing workers for DICOM handling.",
    )
    parser.add_argument(
        "--rt-max-scan-length",
        type=int,
        default=None,
        help="Maximum TR count preallocated by RTPSpy regression.",
    )

    args = parser.parse_args()
    if args.pca_score:
        raise ValueError(
            "--pca-score no longer belongs in rs_realtime_parallel.py. "
            "Run rs_pca_score_all_rois.py directly for daily realtime PCA all-ROI scoring."
        )
    from rt_pipeline import RTSessionConfig, REGRESSOR_SETTINGS

    if args.settings_file:
        loaded = load_regressor_settings(args.settings_file)
        REGRESSOR_SETTINGS.update(vars(loaded))
    if args.skip_first_trs < 0:
        raise ValueError("--skip-first-trs must be >= 0")
    baseline_trs = (
        int(args.baseline_trs)
        if args.baseline_trs is not None
        else int(REGRESSOR_SETTINGS.voxel_norm_ref_volumes)
    )
    if baseline_trs < 0:
        raise ValueError("--baseline-trs must be >= 0")
    from biopac_rt.biopac_receiver import BiopacReceiverConfig

    if args.epi_block is not None:
        epi_block = args.epi_block
    else:
        try:
            epi_block = int(args.run)
        except ValueError as exc:
            raise ValueError("--run must be numeric when --epi-block is not provided.") from exc
    incoming_root = Path(args.incoming_root)
    base_data = Path(args.base_data)

    if not incoming_root.exists():
        raise FileNotFoundError(f"Incoming directory does not exist: {incoming_root}")

    run_dir = base_data / f"sub-{args.sub}" / args.day / "func" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    pca_workflow = args.pca_prep
    pca_reference_image = None

    ctx = mp.get_context("spawn")
    score_queue = ctx.Queue(maxsize=100)
    fixation_stop = ctx.Event()
    fixation_process = ctx.Process(
        target=run_fixation_presentation,
        args=(score_queue, args.max_trs, fixation_stop),
    )
    fixation_process.start()

    _merge_session_metadata(
        run_dir,
        {
            "fixation_display": {
                "description": "Grey screen only.",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "skip_first_trs": args.skip_first_trs,
                "baseline_trs": baseline_trs,
            }
        },
    )

    biopac_process = None
    biopac_stop = None
    if args.biopac_listener:
        if not args.biopac_enable:
            raise ValueError("--biopac-listener requires --biopac-enable")
        if args.biopac_mode != "file":
            raise ValueError("--biopac-listener requires --biopac-mode=file")
        biopac_output = Path(args.biopac_file) if args.biopac_file else (run_dir / "biopac_regressors_rx.csv")
        args.biopac_file = str(biopac_output)
        biopac_stop = mp.get_context("spawn").Event()
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
            subject=args.sub,
            day=args.day,
            run=args.run,
            output_path=biopac_output,
        )
        biopac_process = mp.get_context("spawn").Process(
            target=_run_biopac_listener,
            args=(biopac_cfg, biopac_stop),
        )
        biopac_process.start()

    try:
        _wait_for_epi_dicoms(
            incoming_root,
            epi_block,
            min_vols=args.wait_epi_min,
        )
        _run_preproc(
            sub=args.sub,
            day=args.day,
            base_data=base_data,
            incoming_root=incoming_root,
            struct_block=args.struct_block,
            ap_block=args.ap_block,
            pa_block=args.pa_block,
            epi_block=epi_block,
        )
        if args.prep_surface_rois:
            _run_prep_surface_rois(base_data, args.sub, args.day)
        if pca_workflow and args.pca_space in {"t1", "mni"}:
            subject_root = base_data / f"sub-{args.sub}"
            trans_dir = subject_root / args.day / "func" / "trans"
            pca_reference_image = pca_rt.ensure_pca_t1_reference(
                subject_root,
                trans_dir,
                args.pca_reference_image,
                resolution=args.pca_reference_resolution,
                truncate_to_epi_fov=bool(REGRESSOR_SETTINGS.truncate_t1_to_epi_fov),
                padding_vox=int(REGRESSOR_SETTINGS.truncate_t1_padding_vox),
            )

        decoder_selected = bool(args.decoder_template)
        enable_scoring = decoder_selected and not args.no_score
        if not decoder_selected:
            log.info("[SCORE] No decoder selected (--decoder-template not provided); scoring will be skipped.")
        elif args.no_score:
            log.info("[SCORE] Decoder was provided but --no-score is set; scoring will be skipped.")

        fieldmap_dir = _fieldmap_pair_dir(
            base_data=base_data,
            sub=args.sub,
            day=args.day,
            ap_block=args.ap_block,
            pa_block=args.pa_block,
        )

        _check_rt_fieldmap_exists(fieldmap_dir)
        log.info("[FMAP] RT will use fieldmap dir: %s", fieldmap_dir)

        cfg = RTSessionConfig(
            subject=args.sub,
            day=args.day,
            run=args.run,
            incoming_root=incoming_root,
            base_data=base_data,
            decoder_template=(
                Path(args.decoder_template)
                if args.decoder_template
                else pca_reference_image
            ),
            enable_scoring=enable_scoring,
            fieldmap_dir=fieldmap_dir,
        )

        # pca_root = cfg.day_root / "PCA"
        # if args.decoder_template is None and not pca_root.exists():
        #     raise FileNotFoundError(
        #         f"PCA folder not found at {pca_root}. "
        #         "Provide --decoder-template or run PCA preparation first."
        #     )

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
            }
        )
        if args.rt_max_scan_length is not None:
            settings_payload["rt_max_scan_length"] = max(1, int(args.rt_max_scan_length))
        if pca_workflow:
            settings_payload["analysis_space"] = args.pca_space

        pipeline_process = ctx.Process(
            target=_run_pipeline_with_settings,
            args=(cfg, score_queue, settings_payload),
        )
        pipeline_process.start()

        try:
            pipeline_process.join()
        finally:
            if pipeline_process.is_alive():
                pipeline_process.terminate()
            pipeline_process.join(timeout=5)
            _plot_qc(cfg.rt_work_dir)

        if args.pca_prep:
            _run_pca_prep(base_data, args.sub, args.day, args.run, args.pca_input)
        if args.pca_score:
            _run_pca_score(
                base_data=base_data,
                sub=args.sub,
                day=args.day,
                run=args.run,
                decoder_day=args.pca_day,
                decoder_run=args.pca_run,
                pca_input=args.pca_input,
                normalization=args.pca_normalization,
                score_metric=args.pca_score_metric,
                reference_stats_out=args.pca_reference_stats_out,
            )
    finally:
        fixation_stop.set()
        fixation_process.join(timeout=5)
        if biopac_stop is not None:
            biopac_stop.set()
        if biopac_process is not None:
            biopac_process.join(timeout=5)


if __name__ == "__main__":
    main()
