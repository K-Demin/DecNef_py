#!/usr/bin/env python
import argparse
import csv
import json
import random
from collections import deque
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


@dataclass(frozen=True)
class Condition:
    condition_id: str
    roi: str
    direction: str
    symbol: str


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_conditions(roi_labels: list[str], directions: list[str], symbols: list[str]) -> list[Condition]:
    conditions: list[Condition] = []
    for roi in roi_labels:
        for direction in directions:
            condition_id = f"{roi}_{direction}".lower().replace(" ", "_")
            conditions.append(
                Condition(
                    condition_id=condition_id,
                    roi=roi,
                    direction=direction,
                    symbol="",
                )
            )

    if len(symbols) < len(conditions):
        raise ValueError(
            f"Need at least {len(conditions)} symbols, got {len(symbols)}"
        )

    enriched: list[Condition] = []
    for cond, symbol in zip(conditions, symbols):
        enriched.append(
            Condition(
                condition_id=cond.condition_id,
                roi=cond.roi,
                direction=cond.direction,
                symbol=symbol,
            )
        )
    return enriched


def _shuffle_symbols(symbols: list[str], seed: int | None) -> list[str]:
    rng = random.Random(seed)
    shuffled = symbols[:]
    rng.shuffle(shuffled)
    return shuffled


def _load_or_create_schedule(
    schedule_path: Path,
    conditions: list[Condition],
    seed: int | None = None,
    symbol_seed: int | None = None,
) -> dict:
    if schedule_path.exists():
        with open(schedule_path, "r", encoding="utf-8") as f:
            schedule = json.load(f)
        existing = {c["condition_id"] for c in schedule.get("conditions", [])}
        expected = {cond.condition_id for cond in conditions}
        if existing == expected and schedule.get("order"):
            return schedule

    rng = random.Random(seed)
    condition_ids = [cond.condition_id for cond in conditions]
    rng.shuffle(condition_ids)
    schedule = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "symbol_seed": symbol_seed,
        "order": condition_ids,
        "conditions": [
            {
                "condition_id": cond.condition_id,
                "roi": cond.roi,
                "direction": cond.direction,
                "symbol": cond.symbol,
            }
            for cond in conditions
        ],
    }
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    with open(schedule_path, "w", encoding="utf-8") as f:
        json.dump(schedule, f, indent=2)
    return schedule


def _condition_for_run(schedule: dict, run: str) -> Condition:
    order = schedule["order"]
    condition_lookup = {c["condition_id"]: c for c in schedule["conditions"]}
    try:
        run_idx = int(run) - 1
    except ValueError:
        run_idx = 0
    if run_idx < 0:
        run_idx = 0
    if run_idx >= len(order):
        run_idx = run_idx % len(order)
    condition_id = order[run_idx]
    cond = condition_lookup[condition_id]
    return Condition(
        condition_id=cond["condition_id"],
        roi=cond["roi"],
        direction=cond["direction"],
        symbol=cond["symbol"],
    )


def _write_run_assignment(run_path: Path, condition: Condition, schedule_path: Path) -> None:
    payload = {
        "condition_id": condition.condition_id,
        "roi": condition.roi,
        "direction": condition.direction,
        "symbol": condition.symbol,
        "schedule_path": str(schedule_path),
    }
    run_path.mkdir(parents=True, exist_ok=True)
    with open(run_path / "condition_assignment.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _append_condition_score(csv_path: Path, message: dict, condition: Condition) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(
                [
                    "volume_idx",
                    "timestamp",
                    "score_raw",
                    "condition_id",
                    "roi",
                    "direction",
                    "symbol",
                ]
            )
        writer.writerow(
            [
                message.get("volume_idx"),
                message.get("timestamp"),
                message.get("score_raw"),
                condition.condition_id,
                condition.roi,
                condition.direction,
                condition.symbol,
            ]
        )


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


def run_psychopy_presentation(
    score_queue: Queue,
    max_points: int,
    condition: Condition,
    condition_scores_path: Path,
    max_trs: Optional[int],
) -> None:
    from psychopy import core, event, visual

    win = visual.Window(size=(1000, 700), color="black", units="pix")
    title = visual.TextStim(win, text="Real-time Scores", pos=(0, 300), color="white")
    condition_text = visual.TextStim(
        win,
        text=condition.symbol,
        pos=(0, 200),
        color="white",
        height=80,
    )
    waiting_text = visual.TextStim(
        win,
        text="Waiting for scanner trigger ('s')...",
        pos=(0, 0),
        color="white",
    )

    margins = {"left": 80, "right": 40, "bottom": 80, "top": 80}
    plot_width = win.size[0] - margins["left"] - margins["right"]
    plot_height = win.size[1] - margins["bottom"] - margins["top"]
    origin_x = -win.size[0] / 2 + margins["left"]
    origin_y = -win.size[1] / 2 + margins["bottom"]

    x_axis = visual.Line(
        win,
        start=(origin_x, origin_y),
        end=(origin_x + plot_width, origin_y),
        lineColor="white",
    )
    y_axis = visual.Line(
        win,
        start=(origin_x, origin_y),
        end=(origin_x, origin_y + plot_height),
        lineColor="white",
    )
    # start as a degenerate 2-point line (valid Nx2)
    score_line = visual.ShapeStim(
        win,
        vertices=[(origin_x, origin_y), (origin_x, origin_y)],
        closeShape=False,
        lineColor="cyan",
    )
    last_score_text = visual.TextStim(win, text="", pos=(0, -300), color="white")

    scores = deque(maxlen=max_points)
    needs_redraw = True
    reg_ready_seen = False
    seen_vols: set[int] = set()

    waiting = True
    while waiting:
        waiting_text.draw()
        win.flip()
        keys = event.getKeys()
        if "s" in keys:
            waiting = False
        if "escape" in keys:
            win.close()
            return
        core.wait(0.02)

    def update_plot() -> None:
        nonlocal needs_redraw
        if not scores:
            score_line.vertices = [(origin_x, origin_y), (origin_x, origin_y)]
            if reg_ready_seen:
                last_score_text.text = "Waiting for scores…"
            else:
                last_score_text.text = "Waiting for regression…"
            needs_redraw = True
            return

        # If only 1 point, duplicate it so vertices is still Nx2 with N>=2
        data = list(scores)
        if len(data) == 1:
            data = [data[0], data[0]]

        min_score = min(data)
        max_score = max(data)
        if min_score == max_score:
            min_score -= 0.5
            max_score += 0.5

        span = max_score - min_score
        x_step = plot_width / max(1, max_points - 1)

        vertices = []
        for idx, score in enumerate(data):
            x = origin_x + idx * x_step
            y_norm = (score - min_score) / span
            y = origin_y + y_norm * plot_height
            vertices.append((x, y))

        score_line.vertices = vertices
        last_score_text.text = f"Last score: {scores[-1]:.4f}"
        needs_redraw = True

    while True:
        updated = False
        try:
            while True:
                message = score_queue.get_nowait()
                vol_idx = int(message.get("volume_idx", 0))
                if vol_idx:
                    seen_vols.add(vol_idx)
                if message.get("reg_ready", True):
                    reg_ready_seen = True
                    scores.append(float(message["score_raw"]))
                    _append_condition_score(condition_scores_path, message, condition)
                    updated = True
                else:
                    updated = True
        except Empty:
            pass

        if max_trs is not None and len(seen_vols) >= max_trs:
            break

        if updated or needs_redraw:
            update_plot()
            win.clearBuffer()
            title.draw()
            condition_text.draw()
            x_axis.draw()
            y_axis.draw()
            score_line.draw()
            last_score_text.draw()
            win.flip()
            needs_redraw = False

        if "escape" in event.getKeys():
            break

        core.wait(0.02)

    win.close()


def main() -> None:
    mp.set_start_method("spawn", force=True)  # <-- IMPORTANT: before CUDA touches anything
    parser = argparse.ArgumentParser(
        description=(
            "Run rt_pipeline in parallel with a PsychoPy visualization of the last 20 scores."
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
        "--max-points",
        type=int,
        default=20,
        help="Number of most recent scores to plot.",
    )
    parser.add_argument(
        "--decoder-template",
        required=False,
        help="Optional decoder template path to override the default.",
    )
    parser.add_argument(
        "--roi-labels",
        default="LPFC,Sensorimotor,EVC",
        help="Comma-separated ROI labels for condition mapping.",
    )
    parser.add_argument(
        "--direction-labels",
        default="up,down",
        help="Comma-separated regulation directions.",
    )
    parser.add_argument(
        "--condition-symbols",
        default="A,B,C,D,E,F",
        help="Comma-separated symbols to display for each condition.",
    )
    parser.add_argument(
        "--condition-seed",
        type=int,
        default=None,
        help="Optional random seed for condition order (per day)."
    )
    parser.add_argument(
        "--symbol-seed",
        type=int,
        default=None,
        help="Optional random seed for symbol assignment (per subject).",
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
        "--rs",
        dest="reference_score_run",
        help="Reference run ID for z-scoring (uses scores.csv from that run).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum parallel processing workers for DICOM handling.",
    )
    parser.add_argument(
        "--settings-file",
        default=None,
        help="Optional JSON file with global runtime settings (TR, censor thresholds, BIOPAC defaults, etc.).",
    )
    args = parser.parse_args()
    from rt_pipeline import RTSessionConfig, REGRESSOR_SETTINGS

    if args.settings_file:
        loaded = load_regressor_settings(args.settings_file)
        REGRESSOR_SETTINGS.update(vars(loaded))
    from biopac_rt.biopac_receiver import BiopacReceiverConfig
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

    roi_labels = _parse_csv_list(args.roi_labels)
    direction_labels = _parse_csv_list(args.direction_labels)
    symbols = _parse_csv_list(args.condition_symbols)
    symbol_seed = args.symbol_seed
    if symbol_seed is None:
        try:
            symbol_seed = int(args.sub)
        except ValueError:
            symbol_seed = None
    symbols = _shuffle_symbols(symbols, symbol_seed)
    conditions = _build_conditions(roi_labels, direction_labels, symbols)
    schedule_path = cfg.subject_root / "condition_schedule.json"
    schedule = _load_or_create_schedule(
        schedule_path,
        conditions,
        seed=args.condition_seed,
        symbol_seed=symbol_seed,
    )
    condition = _condition_for_run(schedule, args.run)
    run_dir = cfg.rt_work_dir
    _write_run_assignment(run_dir, condition, schedule_path)
    condition_scores_path = run_dir / "scores_with_conditions.csv"

    _merge_session_metadata(
        run_dir,
        {
            "psychopy": {
                "max_points": args.max_points,
                "roi_labels": roi_labels,
                "direction_labels": direction_labels,
                "condition_symbols": symbols,
                "condition_seed": args.condition_seed,
                "symbol_seed": symbol_seed,
                "condition_schedule": str(schedule_path),
                "condition_assignment": {
                    "condition_id": condition.condition_id,
                    "roi": condition.roi,
                    "direction": condition.direction,
                    "symbol": condition.symbol,
                },
            }
        },
    )

    ctx = mp.get_context("spawn")
    score_queue = ctx.Queue(maxsize=100)
    biopac_process = None
    biopac_stop = None
    if args.biopac_listener:
        if not args.biopac_enable:
            raise ValueError("--biopac-listener requires --biopac-enable")
        if args.biopac_mode != "file":
            raise ValueError("--biopac-listener requires --biopac-mode=file")
        biopac_output = settings_payload["biopac_file"] or (run_dir / "biopac_regressors_rx.csv")
        settings_payload["biopac_file"] = biopac_output
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
            subject=args.sub,
            day=args.day,
            run=args.run,
            output_path=biopac_output,
        )
        biopac_process = ctx.Process(
            target=_run_biopac_listener,
            args=(biopac_cfg, biopac_stop),
        )
        biopac_process.start()

    pipeline_process = ctx.Process(
        target=_run_pipeline_with_settings,
        args=(cfg, score_queue, settings_payload),
    )
    pipeline_process.start()

    try:
        run_psychopy_presentation(
            score_queue,
            args.max_points,
            condition,
            condition_scores_path,
            args.max_trs,
        )
    finally:
        if pipeline_process.is_alive():
            pipeline_process.terminate()
        pipeline_process.join(timeout=5)
        if biopac_stop is not None:
            biopac_stop.set()
        if biopac_process is not None:
            biopac_process.join(timeout=5)
        _plot_qc(run_dir)


if __name__ == "__main__":
    main()
