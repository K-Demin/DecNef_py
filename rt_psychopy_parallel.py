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


def _load_or_create_schedule(
    schedule_path: Path,
    conditions: list[Condition],
    seed: int | None = None,
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


def run_psychopy_presentation(
    score_queue: Queue,
    max_points: int,
    condition: Condition,
    condition_scores_path: Path,
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
            last_score_text.text = "Waiting for scores…"
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
                scores.append(float(message["score_raw"]))
                _append_condition_score(condition_scores_path, message, condition)
                updated = True
        except Empty:
            pass

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
        help="Optional random seed for condition order (per day).",
    )
    args = parser.parse_args()
    from rt_pipeline import RTSessionConfig, run_rt_pipeline
    cfg = RTSessionConfig(
        subject=args.sub,
        day=args.day,
        run=args.run,
        incoming_root=Path(args.incoming_root),
        base_data=Path(args.base_data),
        decoder_template=Path(args.decoder_template) if args.decoder_template else None,
    )

    roi_labels = _parse_csv_list(args.roi_labels)
    direction_labels = _parse_csv_list(args.direction_labels)
    symbols = _parse_csv_list(args.condition_symbols)
    conditions = _build_conditions(roi_labels, direction_labels, symbols)
    schedule_path = cfg.day_root / "func" / "condition_schedule.json"
    schedule = _load_or_create_schedule(schedule_path, conditions, seed=args.condition_seed)
    condition = _condition_for_run(schedule, args.run)
    run_dir = cfg.rt_work_dir
    _write_run_assignment(run_dir, condition, schedule_path)
    condition_scores_path = run_dir / "scores_with_conditions.csv"

    ctx = mp.get_context("spawn")
    score_queue = ctx.Queue(maxsize=100)
    pipeline_process = ctx.Process(target=run_rt_pipeline, args=(cfg, score_queue))
    pipeline_process.start()

    try:
        run_psychopy_presentation(
            score_queue,
            args.max_points,
            condition,
            condition_scores_path,
        )
    finally:
        if pipeline_process.is_alive():
            pipeline_process.terminate()
        pipeline_process.join(timeout=5)


if __name__ == "__main__":
    main()
