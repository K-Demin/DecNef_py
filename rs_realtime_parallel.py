#!/usr/bin/env python
import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import Queue
from queue import Empty
from pathlib import Path
import multiprocessing as mp
import time
from typing import Optional

import numpy as np


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


def _run_pipeline_with_settings(cfg: "RTSessionConfig", score_queue: Queue, settings: dict) -> None:
    from rt_pipeline import REGRESSOR_SETTINGS, run_rt_pipeline

    for key, value in settings.items():
        if hasattr(REGRESSOR_SETTINGS, key):
            setattr(REGRESSOR_SETTINGS, key, value)
    run_rt_pipeline(cfg, score_queue)


def run_fixation_presentation(score_queue: Queue, max_trs: Optional[int]) -> None:
    from psychopy import core, event, visual

    win = visual.Window(size=(1000, 700), color=[0.5, 0.5, 0.5], units="pix")
    fixation = visual.TextStim(win, text="+", color="black", height=60)
    seen_vols: set[int] = set()

    while True:
        fixation.draw()
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
        "--max-trs",
        type=int,
        default=None,
        help="Stop the run after this many TRs (or press ESC).",
    )
    args = parser.parse_args()
    from rt_pipeline import RTSessionConfig

    cfg = RTSessionConfig(
        subject=args.sub,
        day=args.day,
        run=args.run,
        incoming_root=Path(args.incoming_root),
        base_data=Path(args.base_data),
        decoder_template=Path(args.decoder_template) if args.decoder_template else None,
    )

    pca_root = cfg.day_root / "PCA"
    if args.decoder_template is None and not pca_root.exists():
        raise FileNotFoundError(
            f"PCA folder not found at {pca_root}. "
            "Provide --decoder-template or run PCA preparation first."
        )

    settings_payload = {
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

    run_dir = cfg.rt_work_dir
    _merge_session_metadata(
        run_dir,
        {
            "fixation_display": {
                "description": "Grey screen with fixation cross only.",
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
        },
    )

    ctx = mp.get_context("spawn")
    score_queue = ctx.Queue(maxsize=100)
    pipeline_process = ctx.Process(
        target=_run_pipeline_with_settings,
        args=(cfg, score_queue, settings_payload),
    )
    pipeline_process.start()

    try:
        run_fixation_presentation(score_queue, args.max_trs)
    finally:
        if pipeline_process.is_alive():
            pipeline_process.terminate()
        pipeline_process.join(timeout=5)
        _plot_qc(run_dir)


if __name__ == "__main__":
    main()
