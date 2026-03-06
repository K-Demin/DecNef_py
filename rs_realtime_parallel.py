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


def _run_pca_prep(base_data: Path, sub: str, day: str, run: str) -> None:
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
    ]
    log.info("Running PCA prep: %s", " ".join(cmd))
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
            vols.append(vol)
            scores.append(score)

    if not scores:
        return

    motion = np.loadtxt(motion_path)
    if motion.ndim == 1:
        motion = motion[None, :]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    axes[0].plot(vols, scores, label="Decoder score (regressed)")
    axes[0].set_xlabel("Volume")
    axes[0].set_ylabel("Score")
    axes[0].legend(loc="upper right")

    for idx in range(min(motion.shape[1], 6)):
        axes[1].plot(motion[:, idx], label=f"Motion {idx + 1}")
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

    for key, value in settings.items():
        if hasattr(REGRESSOR_SETTINGS, key):
            setattr(REGRESSOR_SETTINGS, key, value)
    run_rt_pipeline(cfg, score_queue)


def _build_presentation_window(visual, color):
    default_size = (1000, 700)
    window_kwargs = {
        "size": default_size,
        "color": color,
        "units": "height",
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
        else:
            window_kwargs.update(
                {
                "screen": 0,
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

    args = parser.parse_args()
    from rt_pipeline import RTSessionConfig, REGRESSOR_SETTINGS

    if args.settings_file:
        loaded = load_regressor_settings(args.settings_file)
        REGRESSOR_SETTINGS.update(vars(loaded))
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

        cfg = RTSessionConfig(
            subject=args.sub,
            day=args.day,
            run=args.run,
            incoming_root=incoming_root,
            base_data=base_data,
            decoder_template=Path(args.decoder_template) if args.decoder_template else None,
            enable_scoring=not args.no_score,
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
            _run_pca_prep(base_data, args.sub, args.day, args.run)
    finally:
        fixation_stop.set()
        fixation_process.join(timeout=5)
        if biopac_stop is not None:
            biopac_stop.set()
        if biopac_process is not None:
            biopac_process.join(timeout=5)


if __name__ == "__main__":
    main()
