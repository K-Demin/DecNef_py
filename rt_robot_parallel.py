#!/usr/bin/env python
import argparse
import csv
import json
import logging
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
import rs_pca_runtime as pca_rt


log = logging.getLogger(__name__)


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


def _write_pca_run_assignment(
    run_path: Path,
    condition: pca_rt.PCACondition,
    public_schedule_path: Path,
) -> None:
    public_payload = {
        "condition_id": condition.condition_id,
        "symbol": condition.symbol,
        "schedule_path": str(public_schedule_path),
    }
    run_path.mkdir(parents=True, exist_ok=True)
    with open(run_path / "condition_assignment.json", "w", encoding="utf-8") as f:
        json.dump(public_payload, f, indent=2)


def _append_condition_score(csv_path: Path, message: dict, condition: Condition) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    fieldnames = [
        "volume_idx",
        "timestamp",
        "score_raw",
        "feedback_score",
        "score_z",
        "raw_component_score",
        "directed_score",
        "condition_id",
        "roi",
        "score_label",
        "pc",
        "direction",
        "symbol",
    ]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        blind_condition = bool(message.get("blind_condition", False))
        writer.writerow(
            {
                "volume_idx": message.get("volume_idx"),
                "timestamp": message.get("timestamp"),
                "score_raw": message.get("score_raw"),
                "feedback_score": message.get("feedback_score"),
                "score_z": message.get("score_z"),
                "raw_component_score": message.get("raw_component_score"),
                "directed_score": message.get("directed_score"),
                "condition_id": condition.condition_id,
                "roi": "" if blind_condition else getattr(condition, "roi", ""),
                "score_label": message.get("score_label"),
                "pc": "" if blind_condition else getattr(condition, "pc", ""),
                "direction": "" if blind_condition else getattr(condition, "direction", ""),
                "symbol": condition.symbol,
            }
        )




def _append_acquisition_speed(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "volume_idx",
                "estimated_trigger_timestamp",
                "watchdog_timestamp",
                "analysis_timestamp",
                "trigger_to_watchdog_s",
                "watchdog_to_analysis_s",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)

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



def _release_mouse_cursor(win) -> None:
    try:
        win.mouseVisible = True
    except Exception as exc:
        log.warning("Could not show PsychoPy mouse cursor: %s", exc)

    handles = [getattr(win, "winHandle", None)]
    backend = getattr(win, "backend", None)
    if backend is not None:
        handles.append(getattr(backend, "winHandle", None))

    for handle in handles:
        if handle is None or not hasattr(handle, "set_exclusive_mouse"):
            continue
        try:
            handle.set_exclusive_mouse(False)
        except Exception as exc:
            log.warning("Could not release exclusive mouse control: %s", exc)


# ==========================================================================
# ROBOT NEUROFEEDBACK DISPLAY
# Continuous "two robots fighting" task, driven by the live score stream.
# Adapted from robot_task.py into the rt_psychopy_parallel orchestration:
#   * the top line is driven by the decoder/PCA score (not the mouse wheel),
#   * each score is plotted the instant the pipeline emits it (no artificial
#     hold), like rt_psychopy_parallel,
#   * per-run CSVs for level-ups and win/loss are written next to the usual
#     score CSVs, in addition to the normal scores_with_conditions.csv.
# Window setup matches rt_psychopy_parallel (units='pix', no Monitor spec,
# second-screen detection); robot geometry is authored in height-fractions and
# scaled to pixels so it stays centered on any monitor.
# ==========================================================================

# ---- robot tuning constants (safe to tune) --------------------------------
# OUR level (power meter): meter fills at the (delayed) line value per second.
OUR_LEVEL_COST = 3500.0     # meter units for the first level-up
COST_GROWTH = 0.20          # each level costs this much more (0 = constant)
# ENEMY level (win streak):
WIN_STREAK = 3              # consecutive wins that level the enemy up
# fight strength (both sides):
STR_BASE = 50.0             # strength at level 1
STR_STEP = 12.0             # strength gained per level
GAME_HP = 100.0
DMG_K = 0.22                # damage per hit = DMG_K * strength
ATTACK_INTERVAL = 0.5       # seconds between hits
DMG_JITTER = 0.30           # +/- randomness per hit (equal levels ~= chance)
MAX_LEVEL = 12              # safety cap for both sides
# trace geometry:
WINDOW_SEC = 14.0           # seconds of signal across the screen
N_TRACE = 240
SAMPLE_DT = WINDOW_SEC / N_TRACE


def our_cost(lvl):
    return OUR_LEVEL_COST * (1 + COST_GROWTH * (lvl - 1))


def strength(lvl):
    return STR_BASE + STR_STEP * (lvl - 1)


def scale_for(lvl, base):
    return base * min(0.72 + 0.12 * (lvl - 1), 2.2)


def _rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


ROBOT_PARTS = [
    ("leg_l", _rect(-0.22, 0.00, -0.06, 0.28), "dark"),
    ("leg_r", _rect(0.06, 0.00, 0.22, 0.28), "dark"),
    ("arm_b", _rect(-0.46, 0.34, -0.30, 0.68), "dark"),
    ("body", _rect(-0.30, 0.26, 0.30, 0.74), "main"),
    ("core", _rect(-0.10, 0.42, 0.10, 0.60), "glow"),
    ("arm_f", _rect(0.30, 0.40, 0.54, 0.58), "main"),
    ("fist", _rect(0.50, 0.36, 0.68, 0.62), "glow"),
    ("neck", _rect(-0.06, 0.74, 0.06, 0.80), "dark"),
    ("head", _rect(-0.21, 0.79, 0.21, 1.06), "main"),
    ("eye_l", _rect(-0.14, 0.88, -0.04, 0.97), "glow"),
    ("eye_r", _rect(0.04, 0.88, 0.14, 0.97), "glow"),
    ("ant", _rect(-0.02, 1.06, 0.02, 1.18), "dark"),
    ("ant_t", _rect(-0.05, 1.17, 0.05, 1.25), "glow"),
]


def _make_robot(win, visual, main, dark, glow):
    cols = {"main": main, "dark": dark, "glow": glow}
    return [
        visual.ShapeStim(win, vertices=v, fillColor=cols[c], lineColor=None,
                         closeShape=True, autoLog=False)
        for _, v, c in ROBOT_PARTS
    ]


def _place_robot(parts, x, y, scale, flip=1):
    for p in parts:
        p.pos = (x, y)
        p.size = (scale * flip, scale)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_levelup(csv_path: Path, side: str, new_level: int, volume_idx: int,
                    session_time_s: float) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["side", "new_level", "volume_idx", "session_time_s", "timestamp"]
        )
        if not exists:
            writer.writeheader()
        writer.writerow({
            "side": side,
            "new_level": new_level,
            "volume_idx": volume_idx,
            "session_time_s": f"{session_time_s:.3f}",
            "timestamp": _iso_now(),
        })


def _append_outcome(csv_path: Path, outcome: str, our_level: int, enemy_level: int,
                    streak_after: int, wins_total: int, losses_total: int,
                    volume_idx: int, session_time_s: float) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "outcome", "our_level", "enemy_level", "streak_after",
            "wins_total", "losses_total", "volume_idx", "session_time_s", "timestamp",
        ])
        if not exists:
            writer.writeheader()
        writer.writerow({
            "outcome": outcome,
            "our_level": our_level,
            "enemy_level": enemy_level,
            "streak_after": streak_after,
            "wins_total": wins_total,
            "losses_total": losses_total,
            "volume_idx": volume_idx,
            "session_time_s": f"{session_time_s:.3f}",
            "timestamp": _iso_now(),
        })


def _append_robot_timeseries(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "session_time_s", "volume_idx", "score_raw", "score_z", "reg_ready",
            "control", "displayed", "meter", "our_level", "enemy_level",
            "streak", "our_hp", "opp_hp", "wins", "losses",
        ])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _append_trial_score(csv_path: Path, volume_idx: int, trial_score, score_raw,
                        score_z, phase: str = "active") -> None:
    """One row per volume. During the active phase, `trial_score` is the on-screen
    feedback value (0-100) the subject saw. During the post-active fixation phase,
    scores are still computed and saved (for analysis) but NOT shown, so
    `trial_score` is left blank and `phase` marks it as 'post_active'.
    Kept readable by print_mean.py (which reads the `trial_score` column)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["volume_idx", "trial_score", "score_raw", "score_z", "phase", "timestamp"]
        )
        if not exists:
            writer.writeheader()
        writer.writerow({
            "volume_idx": volume_idx,
            "trial_score": "" if trial_score is None else f"{float(trial_score):.4f}",
            "score_raw": "" if score_raw is None else f"{float(score_raw):.4f}",
            "score_z": "" if score_z is None else f"{float(score_z):.4f}",
            "phase": phase,
            "timestamp": _iso_now(),
        })


def _write_robot_summary(run_dir: Path, subject: str, day: str, run: str,
                         duration_s: float, volumes_seen: int, our_level: int,
                         enemy_level: int, wins: int, losses: int,
                         feedback_signal: str, feedback_delay: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "robot_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "day", "run", "duration_s", "volumes_seen",
                         "our_level", "enemy_level", "wins", "losses",
                         "feedback_signal", "feedback_delay"])
        writer.writerow([subject, day, run, f"{duration_s:.1f}", volumes_seen,
                         our_level, enemy_level, wins, losses,
                         feedback_signal, feedback_delay])


def _build_robot_window(visual):
    """Build the presentation window exactly like affectivestroop.py:
    size=[1920,1080], units='height', monitor='testMonitor', with fullscr set as a
    separate line after construction. The named monitor spec plus explicit size give
    PsychoPy a proper coordinate frame, so pos=(0,0) is truly centered and the layout
    does not bunch to the left on the scanner display (the issue seen with the
    pixel/no-monitor setup)."""
    win = visual.Window(
        size=[1920, 1080],
        units="height",
        color=(-0.85, -0.85, -0.8),
        colorSpace="rgb",
        pos=(0, 0),
        allowGUI=True,
        checkTiming=False,
        screen=0,
        monitor="testMonitor",
    )  # size in pix, units in height
    win.fullscr = True
    try:
        win.getActualFrameRate()
    except Exception as exc:
        log.warning("getActualFrameRate failed: %s", exc)
    win.mouseVisible = False
    _release_mouse_cursor(win)
    return win


def _make_unit_scalers(win):
    """Identity scalers for the height-units window: the robot layout is authored
    directly in height-fractions, so H/X/XY pass values through unchanged. Kept as
    functions so the call sites (H(...), XY(...)) don't need rewriting."""
    w_px, h_px = float(win.size[0]), float(win.size[1])

    def H(v):
        return v

    def X(v):
        return v

    def XY(x, y):
        return (x, y)

    return H, X, XY, (w_px, h_px)


def _robot_show_message(win, visual, event, core, text, H) -> bool:
    """Show a centred message; SPACE continues, ESCAPE aborts. Returns False on abort."""
    msg = visual.TextStim(win, text=text, height=H(0.040), color="white",
                          wrapWidth=H(1.5), alignText="center")
    msg.draw()
    win.flip()
    event.clearEvents()
    while True:
        keys = event.getKeys()
        if "space" in keys:
            return True
        if "escape" in keys:
            return False
        core.wait(0.02)


def run_robot_task_presentation(
    score_queue,
    condition,
    condition_scores_path,
    max_trs,
    duration_s=None,
    subject="",
    day="",
    run="",
    tr=1.0,
    feedback_delay=8.0,
    feedback_signal="z",
    z_gain=10.0,
    z_center=50.0,
    active_feedback_trs=None,
    rest_baseline_s=20.0,
) -> None:
    from psychopy import core, event, visual
    import math
    import random
    from collections import deque

    run_dir = Path(condition_scores_path).parent
    levelup_path = run_dir / "robot_levelups.csv"
    outcome_path = run_dir / "robot_outcomes.csv"
    robot_ts_path = run_dir / "robot_timeseries.csv"
    trial_score_path = run_dir / "trial_scores.csv"
    acquisition_speed_path = run_dir / "acquisition_speed_rt.csv"
    logged_trial_vols: set[int] = set()

    win = _build_robot_window(visual)
    H, X, XY, (w_px, h_px) = _make_unit_scalers(win)
    RIGHT = (w_px / h_px) / 2.0
    LEFT = -RIGHT
    _mouse = event.Mouse(visible=False, win=win)

    # ---- top ECG-style plot geometry (authored in height-fraction, drawn in px) ----
    PLOT_BOT, PLOT_TOP = -0.14, 0.46
    GROUND = -0.46

    def sy(score):
        return PLOT_BOT + (score / 100.0) * (PLOT_TOP - PLOT_BOT)

    frame_top = visual.Line(win, start=XY(LEFT + 0.02, sy(100)), end=XY(RIGHT - 0.02, sy(100)), lineColor=(-0.4, -0.4, -0.3))
    frame_bot = visual.Line(win, start=XY(LEFT + 0.02, sy(0)), end=XY(RIGHT - 0.02, sy(0)), lineColor=(-0.4, -0.4, -0.3))
    mid_line = visual.Line(win, start=XY(LEFT + 0.02, sy(50)), end=XY(RIGHT - 0.02, sy(50)), lineColor=(-0.6, -0.6, -0.5))
    xs = np.linspace(X(LEFT + 0.02), X(RIGHT - 0.02), N_TRACE)
    trace = visual.ShapeStim(win, vertices=np.column_stack([xs, np.full(N_TRACE, H(sy(50)))]),
                             closeShape=False, fillColor=None, lineColor="white", lineWidth=3)

    # ---- POWER meter (fills OUR level) ----
    BAR_X, BAR_W = LEFT + 0.07, 0.05
    BAR_BOT, BAR_TOP = GROUND + 0.02, GROUND + 0.24
    BAR_H = BAR_TOP - BAR_BOT
    bar_out = visual.Rect(win, width=H(BAR_W), height=H(BAR_H), pos=XY(BAR_X, (BAR_BOT + BAR_TOP) / 2),
                          fillColor=(-0.75, -0.75, -0.65), lineColor=(0.1, 0.3, 0.45), lineWidth=2)
    bar_fill = visual.Rect(win, width=H(BAR_W - 0.010), height=0.0, fillColor=(0.0, 0.75, 0.85), lineColor=None)
    pow_lbl = visual.TextStim(win, text="POWER", pos=XY(BAR_X, BAR_TOP + 0.03), height=H(0.022), color=(0.3, 0.6, 0.7))
    our_lvl_lbl = visual.TextStim(win, text="LV 1", pos=XY(BAR_X, BAR_BOT - 0.045), height=H(0.030), color=(0.3, 0.85, 1.0), bold=True)

    # ---- win / loss counter ----
    wl_lbl = visual.TextStim(win, text="W 0   L 0", pos=XY(RIGHT - 0.14, 0.485), height=H(0.030), color=(0.75, 0.75, 0.6), bold=True)

    # ---- robots ----
    ours = _make_robot(win, visual, (0.05, 0.55, 0.95), (-0.35, 0.05, 0.45), (0.35, 0.95, 1.0))
    opp = _make_robot(win, visual, (0.85, 0.15, 0.10), (0.35, -0.35, -0.4), (1.0, 0.6, 0.0))
    OUR_X, OPP_X = -0.14, 0.30
    BASE_SCALE = 0.16
    ground_line = visual.Line(win, start=XY(LEFT + 0.02, GROUND), end=XY(RIGHT - 0.02, GROUND), lineColor=(-0.5, -0.5, -0.4))

    HP_Y, HP_W = -0.20, 0.22

    def hp_bar(x, col):
        back = visual.Rect(win, width=H(HP_W), height=H(0.020), pos=XY(x, HP_Y),
                           fillColor=(-0.7, -0.7, -0.6), lineColor=(-0.3, -0.3, -0.2))
        fill = visual.Rect(win, width=H(HP_W), height=H(0.015), pos=XY(x, HP_Y), fillColor=col, lineColor=None)
        return back, fill

    our_hp_bg, our_hp_fg = hp_bar(OUR_X, (0.2, 0.85, 0.3))
    opp_hp_bg, opp_hp_fg = hp_bar(OPP_X, (0.9, 0.35, 0.2))
    flash = visual.TextStim(win, text="", pos=XY(0.05, -0.30), height=H(0.055), color=(1.0, 0.9, 0.2), bold=True)
    phase_msg = visual.TextStim(win, text="", pos=XY(0, -0.05), height=H(0.045), color=(0.9, 0.9, 0.7), bold=True)
    you_lbl = visual.TextStim(win, text="YOU  LV 1", pos=XY(OUR_X, HP_Y + 0.032), height=H(0.022), color=(0.4, 0.8, 1.0))
    opp_lbl = visual.TextStim(win, text="ENEMY  LV 1", pos=XY(OPP_X, HP_Y + 0.032), height=H(0.022), color=(0.9, 0.5, 0.4))
    pips = [visual.Circle(win, radius=H(0.010), pos=XY(OPP_X - 0.02 + i * 0.02, HP_Y - 0.035),
                          lineColor=(0.6, 0.35, 0.3), fillColor=(-0.6, -0.6, -0.5)) for i in range(WIN_STREAK)]
    pips_lbl = visual.TextStim(win, text="enemy powers up in", pos=XY(OPP_X, HP_Y - 0.065), height=H(0.016), color=(0.6, 0.4, 0.35))

    # ---- end-of-run fixation cross (shown during the rest baseline) ----
    fixation = visual.TextStim(win, text="+", pos=(0, 0), height=H(0.10), color="white", bold=True)

    # ---- instructions ----
    proceed = _robot_show_message(
        win, visual, event, core,
        "Keep the LINE HIGH to charge your POWER meter and LEVEL UP your robot "
        "(bigger, stronger).\n\n"
        "The enemy is even at first; win 3 fights in a row and it levels up too.\n\n"
        "Press SPACE when ready.",
        H,
    )
    if not proceed:
        win.close()
        return

    # ---- wait for scanner trigger 's' (drain-and-discard scores meanwhile) ----
    waiting_text = visual.TextStim(win, text="Waiting for scanner trigger ('s')...",
                                   pos=(0, 0), color="white", height=H(0.040))
    first_trigger_timestamp = None
    waiting = True
    while waiting:
        try:
            while True:
                score_queue.get_nowait()
        except Empty:
            pass
        waiting_text.draw()
        win.flip()
        keys = event.getKeys()
        if "s" in keys:
            first_trigger_timestamp = time.time()
            waiting = False
        if "escape" in keys:
            win.close()
            return
        core.wait(0.01)

    # ---- state ----
    clock = core.Clock()
    clock.reset()
    last_t = 0.0
    control_target = float(z_center)
    control = float(z_center)
    ys = deque([H(sy(control))] * N_TRACE, maxlen=N_TRACE)
    next_sample = 0.0
    next_log = 0.0

    meter = 0.0
    our_level = 1
    enemy_level = 1
    streak = 0
    wins, losses = 0, 0
    our_hp, opp_hp = GAME_HP, GAME_HP
    our_atk_t, opp_atk_t = ATTACK_INTERVAL, ATTACK_INTERVAL
    our_lunge, opp_lunge = 0.0, 0.0
    flash_until, flash_txt = 0.0, ""

    seen_vols: set[int] = set()
    current_vol = 0
    reg_ready_seen = False
    last_score_raw = None
    last_score_z = None

    # ---- phase machine --------------------------------------------------
    # active : each score is plotted the instant the pipeline emits it (like
    #          rt_psychopy_parallel; no artificial hold). A feedback volume is
    #          counted the moment its score is shown, so reaching
    #          `active_feedback_trs` means all that many volumes have already had
    #          their feedback displayed.
    # rest   : `rest_baseline_s` of fixation cross only.
    #
    # active_feedback_trs is counted from the FIRST reg_ready volume, so the ~40
    # warmup TRs before feedback do not eat into the 6-minute window.
    phase = "active"
    active_feedback_count = 0        # unique reg_ready volumes shown as feedback
    first_feedback_vol = None
    rest_start_time = None

    safety_s = None if duration_s is None else (float(duration_s) + 180.0)

    # Idle guard: if no NEW volume arrives for this long, assume the archive is
    # exhausted (or acquisition stopped) and end the run instead of hanging with
    # the line pinned. Generous enough not to fire between real TRs.
    idle_timeout_s = 30.0
    last_new_vol_time = None   # wall-clock (clock.getTime) of the last new volume

    while True:
        now = clock.getTime()
        dt = min(now - last_t, 0.05)
        last_t = now

        # ---- phase transitions ----
        if phase == "active" and active_feedback_trs is not None \
                and active_feedback_count >= active_feedback_trs:
            # all `active_feedback_trs` volumes have had their score shown -> rest
            phase = "rest"
            rest_start_time = now
        elif phase == "rest" and rest_start_time is not None \
                and (now - rest_start_time) >= float(rest_baseline_s):
            break

        over_time = (safety_s is not None) and (now >= safety_s)
        # Idle: at least one volume seen, but nothing new for idle_timeout_s
        # (archive exhausted / acquisition stopped) -> stop instead of hanging.
        idle_stop = (last_new_vol_time is not None) \
            and ((now - last_new_vol_time) >= idle_timeout_s)
        if idle_stop:
            log.warning(
                "No new volume for %.0fs (last vol %d, active feedback shown %d). "
                "Ending run (archive exhausted or acquisition stopped).",
                idle_timeout_s, current_vol, active_feedback_count,
            )
        # Fallback when active_feedback_trs is not set: original max_trs behaviour.
        reached_vols = (active_feedback_trs is None) and (max_trs is not None) \
            and (len(seen_vols) >= max_trs)
        run_out_no_trs = (active_feedback_trs is None) and (max_trs is None) \
            and (duration_s is not None) and (now >= duration_s)
        if over_time or idle_stop or reached_vols or run_out_no_trs or ("escape" in event.getKeys()):
            break

        # ---- drain the score queue ----
        try:
            while True:
                message = score_queue.get_nowait()
                vol_idx = int(message.get("volume_idx", 0))
                if vol_idx:
                    if vol_idx not in seen_vols:
                        last_new_vol_time = now
                    seen_vols.add(vol_idx)
                    if vol_idx > current_vol:
                        current_vol = vol_idx
                if first_trigger_timestamp is not None and vol_idx > 0:
                    estimated_trigger_timestamp = first_trigger_timestamp + ((vol_idx - 1) * float(tr))
                    watchdog_timestamp = message.get("watchdog_timestamp")
                    analysis_timestamp = message.get("analysis_timestamp", message.get("timestamp"))
                    if watchdog_timestamp is not None and analysis_timestamp is not None:
                        _append_acquisition_speed(
                            acquisition_speed_path,
                            {
                                "volume_idx": vol_idx,
                                "estimated_trigger_timestamp": f"{estimated_trigger_timestamp:.6f}",
                                "watchdog_timestamp": f"{float(watchdog_timestamp):.6f}",
                                "analysis_timestamp": f"{float(analysis_timestamp):.6f}",
                                "trigger_to_watchdog_s": f"{(float(watchdog_timestamp) - estimated_trigger_timestamp):.6f}",
                                "watchdog_to_analysis_s": f"{(float(analysis_timestamp) - float(watchdog_timestamp)):.6f}",
                            },
                        )
                has_score = "score_raw" in message
                if has_score and message.get("reg_ready", True):
                    reg_ready_seen = True
                    sr = float(message["score_raw"])
                    sz = message.get("score_z")
                    if phase == "active":
                        last_score_raw = sr
                        last_score_z = sz
                        # Plot each score the instant the pipeline emits it (no hold).
                        if feedback_signal == "z" and sz is not None:
                            control_target = max(0.0, min(100.0, float(z_center) + float(sz) * float(z_gain)))
                        else:
                            control_target = max(0.0, min(100.0, sr))
                        # count as one shown feedback volume (unique vols only)
                        if vol_idx and vol_idx not in logged_trial_vols:
                            logged_trial_vols.add(vol_idx)
                            active_feedback_count += 1
                            if first_feedback_vol is None:
                                first_feedback_vol = vol_idx
                            _append_condition_score(condition_scores_path, message, condition)
                            _append_trial_score(trial_score_path, vol_idx, control_target, sr, sz, phase="active")
                    else:
                        # post-active (fixation) phase: keep SCORING and saving for
                        # analysis, but do NOT display it (no control_target update).
                        if vol_idx and vol_idx not in logged_trial_vols:
                            logged_trial_vols.add(vol_idx)
                            _append_condition_score(condition_scores_path, message, condition)
                            _append_trial_score(trial_score_path, vol_idx, None, sr, sz, phase="post_active")
        except Empty:
            pass

        # ---- feedback is live as soon as the first score has arrived ----
        feedback_live = reg_ready_seen

        # ---- displayed value = current score, no artificial delay ----
        control = control_target if reg_ready_seen else float(z_center)
        disp = control

        g = 0
        while now >= next_sample and g < 5:
            ys.append(H(sy(disp)))
            next_sample += SAMPLE_DT
            g += 1

        # ---- active game only after the feedback delay has fully elapsed ----
        if feedback_live:
            if our_level < MAX_LEVEL:
                meter += disp * dt
                need = our_cost(our_level)
                if meter >= need:
                    meter -= need
                    our_level += 1
                    flash_txt, flash_until = "YOU LEVEL UP!", now + 1.3
                    _append_levelup(levelup_path, "our", our_level, current_vol, now)
                meter_prog = min(1.0, meter / our_cost(our_level)) if our_level < MAX_LEVEL else 1.0
            else:
                meter_prog = 1.0

            our_str = strength(our_level)
            opp_str = strength(enemy_level)
            our_atk_t -= dt
            if our_atk_t <= 0:
                our_atk_t = ATTACK_INTERVAL * random.uniform(0.85, 1.15)
                opp_hp -= DMG_K * our_str * random.uniform(1 - DMG_JITTER, 1 + DMG_JITTER)
                our_lunge = 0.15
            opp_atk_t -= dt
            if opp_atk_t <= 0:
                opp_atk_t = ATTACK_INTERVAL * random.uniform(0.85, 1.15)
                our_hp -= DMG_K * opp_str * random.uniform(1 - DMG_JITTER, 1 + DMG_JITTER)
                opp_lunge = 0.15
            our_lunge = max(0.0, our_lunge - dt)
            opp_lunge = max(0.0, opp_lunge - dt)

            if opp_hp <= 0:
                wins += 1
                streak += 1
                our_hp = opp_hp = GAME_HP
                our_atk_t = opp_atk_t = ATTACK_INTERVAL
                if streak >= WIN_STREAK and enemy_level < MAX_LEVEL:
                    enemy_level += 1
                    streak = 0
                    flash_txt, flash_until = "ENEMY LEVELS UP!", now + 1.3
                    _append_outcome(outcome_path, "win", our_level, enemy_level, streak, wins, losses, current_vol, now)
                    _append_levelup(levelup_path, "enemy", enemy_level, current_vol, now)
                else:
                    flash_txt, flash_until = "WIN!", now + 0.8
                    _append_outcome(outcome_path, "win", our_level, enemy_level, streak, wins, losses, current_vol, now)
            elif our_hp <= 0:
                losses += 1
                streak = 0
                our_hp = opp_hp = GAME_HP
                our_atk_t = opp_atk_t = ATTACK_INTERVAL
                flash_txt, flash_until = "LOST!", now + 0.8
                _append_outcome(outcome_path, "loss", our_level, enemy_level, streak, wins, losses, current_vol, now)
        else:
            meter_prog = 0.0
            our_hp, opp_hp = GAME_HP, GAME_HP
            our_atk_t = opp_atk_t = ATTACK_INTERVAL

        # ---- draw ----
        if phase == "rest":
            # fixation cross only; no feedback elements
            fixation.draw()
            win.flip()
        else:
            frame_top.draw(); frame_bot.draw(); mid_line.draw()
            trace.vertices = np.column_stack([xs, np.array(ys)])
            trace.draw()
            ground_line.draw()

            bar_out.draw()
            fh = (BAR_H - 0.010) * meter_prog
            bar_fill.height = max(0.0, H(fh))
            bar_fill.pos = XY(BAR_X, BAR_BOT + 0.005 + fh / 2.0)
            bar_fill.draw()
            pow_lbl.draw()
            our_lvl_lbl.text = "LV %d" % our_level
            our_lvl_lbl.draw()

            wl_lbl.text = "W %d   L %d" % (wins, losses)
            wl_lbl.draw()

            ox = 0.035 * (our_lunge / 0.15)
            px = -0.035 * (opp_lunge / 0.15)
            bob = 0.003 * math.sin(now * 4.0)
            _place_robot(ours, X(OUR_X + ox), H(GROUND + bob), H(scale_for(our_level, BASE_SCALE)), flip=1)
            _place_robot(opp, X(OPP_X + px), H(GROUND - bob), H(scale_for(enemy_level, BASE_SCALE)), flip=-1)
            for p in ours:
                p.draw()
            for p in opp:
                p.draw()

            for bg, fg, hp, x in ((our_hp_bg, our_hp_fg, our_hp, OUR_X),
                                  (opp_hp_bg, opp_hp_fg, opp_hp, OPP_X)):
                bg.draw()
                frac = max(0.0, min(1.0, hp / GAME_HP))
                fg.width = max(0.001, H(HP_W * frac))
                fg.pos = XY(x - HP_W / 2.0 + (HP_W * frac) / 2.0, HP_Y)
                fg.draw()
            you_lbl.text = "YOU  LV %d" % our_level
            opp_lbl.text = "ENEMY  LV %d" % enemy_level
            you_lbl.draw(); opp_lbl.draw()
            pips_lbl.draw()
            for i, pip in enumerate(pips):
                pip.fillColor = (0.9, 0.5, 0.2) if i < streak else (-0.6, -0.6, -0.5)
                pip.draw()

            if now < flash_until:
                flash.text = flash_txt
                flash.draw()

            if not reg_ready_seen:
                phase_msg.text = "Waiting for scanner..."
                phase_msg.draw()

            win.flip()

        # ---- robot timeseries log (~10 Hz) ----
        if now >= next_log:
            _append_robot_timeseries(robot_ts_path, {
                "session_time_s": f"{now:.2f}",
                "volume_idx": current_vol,
                "score_raw": "" if last_score_raw is None else f"{last_score_raw:.4f}",
                "score_z": "" if last_score_z is None else f"{float(last_score_z):.4f}",
                "reg_ready": int(reg_ready_seen),
                "control": f"{control:.2f}",
                "displayed": f"{disp:.2f}",
                "meter": f"{meter:.1f}",
                "our_level": our_level,
                "enemy_level": enemy_level,
                "streak": streak,
                "our_hp": f"{our_hp:.1f}",
                "opp_hp": f"{opp_hp:.1f}",
                "wins": wins,
                "losses": losses,
            })
            next_log += 0.1

    total_time = clock.getTime()
    _write_robot_summary(run_dir, subject, day, run, total_time, len(seen_vols),
                         our_level, enemy_level, wins, losses, feedback_signal, feedback_delay)
    win.close()


def main() -> None:
    mp.set_start_method("spawn", force=True)  # <-- IMPORTANT: before CUDA touches anything
    parser = argparse.ArgumentParser(
        description=(
            "Run rt_pipeline in parallel with the robot-fight neurofeedback display."
        )
    )
    parser.add_argument("--sub", required=True, help="Subject ID, e.g. 00086")
    parser.add_argument("--day", required=True, help="Day/session, e.g. 3")
    parser.add_argument("--run", required=True, help="Run number, e.g. 4")
    parser.add_argument(
        "--ap-block",
        default=None,
        type=int,
        help="AP block to use for b0",
    )
    parser.add_argument(
        "--pa-block",
        default=None,
        type=int,
        help="PA block to use for b0",
    )
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
        help="Optional decoder template path to override the default for decoder scoring.",
    )
    parser.add_argument(
        "--decoder-roi-txt",
        required=False,
        default=None,
        help=(
            "Optional ROI_DECODER-style text file for decoder scoring. "
            "Ignored when using PCA scoring."
        ),
    )
    parser.add_argument(
        "--score-source",
        choices=["decoder", "pca"],
        default=None,
        help=(
            "Score backend for feedback. Defaults to decoder unless --pca-mode "
            "is supplied; --pca-mode is kept as an alias for --score-source=pca."
        ),
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
        "--duration-min",
        type=float,
        default=6.0,
        help="Stop after this many minutes (default 6), using TR from settings. Ignored if --max-trs or --active-feedback-trs is set.",
    )
    parser.add_argument(
        "--active-feedback-trs",
        type=int,
        default=None,
        help=(
            "Number of ACTIVE feedback volumes, counted from the first reg_ready "
            "volume (so the ~40 TR pipeline warmup does NOT count). At TR=1s, 360 "
            "= 6 min of feedback. After this many volumes: freeze new input, drain "
            "the feedback-delay line, then show a fixation rest. Recommended over "
            "--max-trs / --duration-min for this paradigm."
        ),
    )
    parser.add_argument(
        "--rest-baseline-sec",
        type=float,
        default=20.0,
        help="Seconds of fixation-cross rest after the delay flush (only used with --active-feedback-trs).",
    )
    parser.add_argument(
        "--feedback-delay",
        type=float,
        default=8.0,
        help="Seconds of feedback delay for the robot line (delay-training).",
    )
    parser.add_argument(
        "--feedback-signal",
        choices=["raw", "z"],
        default="z",
        help="Drive the robot line from the RS-normalized z-score (default; needs --rs) or the raw decoder score.",
    )
    parser.add_argument(
        "--z-gain",
        type=float,
        default=10.0,
        help="Display units per z-unit when --feedback-signal=z (z=+/-5 spans the 0-100 line at the default). Tune from pilot data.",
    )
    parser.add_argument(
        "--z-center",
        type=float,
        default=50.0,
        help="Line centre (0-100) for z mapping and for the pre-feedback baseline.",
    )
    parser.add_argument(
        "--rs",
        dest="reference_score_run",
        help="Reference run ID for z-scoring (uses scores.csv from that run).",
    )
    parser.add_argument(
        "--pca-mode",
        action="store_true",
        help=(
            "Deprecated alias for --score-source=pca: disable decoder scoring "
            "and drive feedback from PCA outputs."
        ),
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
        help="PCA decoder/output mode folder name.",
    )
    parser.add_argument(
        "--pca-space",
        choices=["epi", "t1", "mni"],
        default="t1",
        help="rt_pipeline output space to create for PCA feedback.",
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
        "--pca-reference-scores",
        type=Path,
        default=None,
        help="Daily RS raw PCA scores CSV from rs_pca_score_all_rois.py.",
    )
    parser.add_argument(
        "--pca-reference-stats",
        type=Path,
        default=None,
        help="Deprecated explicit daily RS reference stats JSON.",
    )
    parser.add_argument(
        "--pca-reference-day",
        default=None,
        help="Day/session of the daily RS stats run. Defaults to --day when --pca-reference-run is set.",
    )
    parser.add_argument(
        "--pca-reference-run",
        default=None,
        help="Run whose daily RS PCA scores should be used for feedback.",
    )
    parser.add_argument(
        "--pca-volume-kind",
        choices=["reg", "mc", "unwarped", "t1", "mni"],
        default=None,
        help="Realtime volume folder to score with PCA.",
    )
    parser.add_argument(
        "--pca-target-pc",
        default=None,
        help="Legacy single PC to modulate when --pca-score-metric is projection/cosine.",
    )
    parser.add_argument(
        "--pca-normalization",
        choices=["zscore", "demean", "none"],
        default="zscore",
        help="Voxel normalization before PCA projection.",
    )
    parser.add_argument(
        "--pca-score-metric",
        choices=sorted(pca_rt.ALL_SCORE_METRICS),
        default="subspace_cosine",
        help="PCA scalar score. Default is cosine closeness to the top-PC subspace.",
    )
    parser.add_argument(
        "--pca-top-pcs",
        default="auto",
        help=(
            "Top PCs for multi-PC metrics: auto uses a cumulative explained-variance "
            "target; also accepts integer count, var:0.01, or all."
        ),
    )
    parser.add_argument(
        "--pca-top-pc-variance",
        type=float,
        default=0.10,
        help=(
            "Cumulative explained-variance threshold used only when "
            "--pca-top-pcs=auto. If the saved PCs do not reach it, auto uses all saved PCs."
        ),
    )
    parser.add_argument(
        "--pca-poll-interval",
        type=float,
        default=0.05,
        help="Seconds between checks for newly processed realtime volumes.",
    )
    parser.add_argument(
        "--condition-index",
        type=int,
        default=None,
        help="1-based condition index for this NF run. Defaults to numeric --run.",
    )
    parser.add_argument(
        "--condition-private-key",
        type=Path,
        default=None,
        help="Private A-F to ROI/direction mapping. Keep hidden from subject/experimenter.",
    )
    parser.add_argument(
        "--condition-public-schedule",
        type=Path,
        default=None,
        help="Blinded A-F schedule shown/recorded for the run.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum parallel processing workers for DICOM handling.",
    )
    parser.add_argument(
        "--rt-max-scan-length",
        type=int,
        default=None,
        help="Maximum TR count preallocated by RTPSpy regression.",
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
    max_trs = args.max_trs
    if args.active_feedback_trs is not None:
        # Phase machine controls termination; max_trs is only a hard safety ceiling.
        # Ceiling = generous warmup allowance + active + rest + margin.
        tr_val = float(REGRESSOR_SETTINGS.TR)
        warmup_allow = int(REGRESSOR_SETTINGS.skip_first_trs) + int(REGRESSOR_SETTINGS.voxel_norm_ref_volumes)
        rest_trs = int(round(float(args.rest_baseline_sec) / tr_val))
        safety_ceiling = warmup_allow + int(args.active_feedback_trs) + rest_trs + 60
        max_trs = safety_ceiling if max_trs is None else max_trs
    elif max_trs is None and args.duration_min is not None:
        max_trs = int(round((float(args.duration_min) * 60.0) / float(REGRESSOR_SETTINGS.TR)))
    from biopac_rt.biopac_receiver import BiopacReceiverConfig
    base_data = Path(args.base_data)
    subject_root = base_data / f"sub-{args.sub}"
    pca_reference_image = None
    score_source = args.score_source or ("pca" if args.pca_mode else "decoder")
    use_pca = score_source == "pca"
    if args.pca_mode and args.score_source == "decoder":
        raise ValueError("--pca-mode conflicts with --score-source=decoder")
    if use_pca and args.decoder_roi_txt:
        log.info("PCA score source enabled: ignoring --decoder-roi-txt=%s", args.decoder_roi_txt)

    cfg_decoder_template = (
        None
        if use_pca
        else Path(args.decoder_template) if args.decoder_template else None
    )
    cfg = RTSessionConfig(
        subject=args.sub,
        day=args.day,
        run=args.run,
        incoming_root=Path(args.incoming_root),
        base_data=base_data,
        decoder_template=cfg_decoder_template,
        decoder_roi_txt=None if use_pca else Path(args.decoder_roi_txt) if args.decoder_roi_txt else None,
        reference_score_run=None if use_pca else args.reference_score_run,
        enable_scoring=not use_pca,
        ap_block=args.ap_block,
        pa_block=args.pa_block,
    )

    if use_pca and args.reference_score_run:
        log.info("PCA score source enabled: ignoring --rs=%s", args.reference_score_run)

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
    if args.rt_max_scan_length is not None:
        settings_payload["rt_max_scan_length"] = max(1, int(args.rt_max_scan_length))
    if use_pca:
        settings_payload["analysis_space"] = args.pca_space
    pca_volume_kind = args.pca_volume_kind
    if use_pca and pca_volume_kind is None:
        pca_volume_kind = "reg" if args.pca_space == "epi" else args.pca_space
    pca_score_label = None
    if use_pca:
        pca_score_label = pca_rt.resolve_score_label(
            score_metric=args.pca_score_metric,
            target_pc=args.pca_target_pc,
            top_pcs=args.pca_top_pcs,
            top_pc_variance=args.pca_top_pc_variance,
        )

    roi_labels = _parse_csv_list(args.roi_labels)
    direction_labels = _parse_csv_list(args.direction_labels)
    symbols = _parse_csv_list(args.condition_symbols)
    symbol_seed = args.symbol_seed
    if symbol_seed is None and not use_pca:
        try:
            symbol_seed = int(args.sub)
        except ValueError:
            symbol_seed = None
    run_dir = cfg.rt_work_dir
    pca_day = None
    pca_run = None
    pca_root = None
    if use_pca:
        default_private_key, default_public_schedule = pca_rt.default_condition_paths(
            cfg.subject_root
        )
        private_key_path = args.condition_private_key or default_private_key
        public_schedule_path = args.condition_public_schedule or default_public_schedule
        schedule = pca_rt.load_or_create_condition_schedule(
            private_path=private_key_path,
            public_path=public_schedule_path,
            roi_labels=roi_labels,
            direction_labels=direction_labels,
            symbols=symbols,
            target_pc=pca_score_label,
            condition_seed=args.condition_seed,
            symbol_seed=symbol_seed,
        )
        if args.condition_index is not None:
            condition_index = args.condition_index
        else:
            try:
                condition_index = int(args.run)
            except ValueError:
                condition_index = 1
        condition = pca_rt.condition_for_index(
            schedule,
            condition_index,
            score_label=pca_score_label,
        )
        _write_pca_run_assignment(
            run_dir,
            condition,
            public_schedule_path,
        )
        pca_run = args.pca_run or args.run
        pca_day = args.pca_day or args.day
        pca_run_dir = pca_rt.build_run_dir(base_data, args.sub, pca_day, pca_run)
        pca_day_dir = pca_run_dir.parent.parent
        pca_root = args.pca_root or pca_rt.build_pca_root(
            pca_day_dir,
            pca_run_dir.name,
            args.pca_input,
        )
        if not pca_root.exists():
            raise FileNotFoundError(f"PCA decoder root not found: {pca_root}")
        if args.pca_space in {"t1", "mni"}:
            decoder_trans_dir = subject_root / pca_day / "func" / "trans"
            pca_reference_image = pca_rt.resolve_pca_transform_reference(
                subject_root=subject_root,
                decoder_trans_dir=decoder_trans_dir,
                pca_root=pca_root,
                explicit_path=args.pca_reference_image,
                resolution=args.pca_reference_resolution,
                truncate_to_epi_fov=bool(REGRESSOR_SETTINGS.truncate_t1_to_epi_fov),
                padding_vox=int(REGRESSOR_SETTINGS.truncate_t1_padding_vox),
            )
            cfg.decoder_template = pca_reference_image
            log.info("Using PCA transform reference: %s", pca_reference_image)
    else:
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
        _write_run_assignment(run_dir, condition, schedule_path)
    condition_scores_path = run_dir / "scores_with_conditions.csv"
    reference_scores_path = args.pca_reference_scores
    reference_stats_path = args.pca_reference_stats
    reference_stats = {}
    reference_stats_day = args.pca_reference_day
    reference_stats_run = args.pca_reference_run
    if use_pca:
        if reference_scores_path is not None and reference_stats_path is not None:
            raise ValueError("Use either --pca-reference-scores or --pca-reference-stats, not both.")
        if reference_stats_path is not None:
            if not reference_stats_path.exists():
                raise FileNotFoundError(f"PCA daily RS reference stats not found: {reference_stats_path}")
            reference_stats = pca_rt.load_reference_stats(reference_stats_path)
        elif reference_stats_run is not None:
            reference_stats_day = reference_stats_day or args.day
            reference_scores_path = pca_rt.build_reference_scores_path(
                base_data,
                args.sub,
                reference_stats_day,
                reference_stats_run,
                args.pca_input,
            )
        elif reference_stats_day is not None and reference_scores_path is None:
            raise ValueError(
                "--pca-reference-day requires --pca-reference-run unless "
                "--pca-reference-scores is provided explicitly."
            )
        if reference_scores_path is not None:
            if not reference_scores_path.exists():
                raise FileNotFoundError(
                    f"PCA daily RS reference scores not found: {reference_scores_path}. "
                    "Run rs_pca_score_all_rois.py for that daily RS run first."
                )
            reference_stats = pca_rt.load_reference_stats_from_score_csv(
                reference_scores_path,
                [condition.score_column],
            )

    condition_payload = {
        "condition_id": condition.condition_id,
        "symbol": condition.symbol,
    }
    if not use_pca:
        condition_payload.update(
            {
                "roi": condition.roi,
                "direction": condition.direction,
            }
        )

    _merge_session_metadata(
        run_dir,
        {
            "psychopy": {
                "max_points": args.max_points,
                "max_trs": max_trs,
                "ap_block": args.ap_block,
                "pa_block": args.pa_block,
                "fmap_dir": str(cfg.fmap_dir),
                "duration_min": args.duration_min,
                "active_feedback_trs": args.active_feedback_trs,
                "feedback_delay_s": args.feedback_delay,
                "rest_baseline_sec": args.rest_baseline_sec,
                "roi_labels": roi_labels,
                "direction_labels": direction_labels,
                "condition_symbols": symbols,
                "condition_seed": args.condition_seed,
                "symbol_seed": symbol_seed,
                "condition_schedule": str(
                    public_schedule_path if use_pca else schedule_path
                ),
                "condition_assignment": condition_payload,
                "score_source": score_source,
                "pca_mode": use_pca,
                "decoder_template": str(cfg.decoder_template) if cfg.decoder_template else None,
                "decoder_roi_txt": str(cfg.decoder_roi_txt) if cfg.decoder_roi_txt else None,
                "pca_decoder": {
                    "day": pca_day,
                    "run": pca_run,
                    "root": str(pca_root) if pca_root else None,
                    "input": args.pca_input,
                    "space": args.pca_space,
                    "reference_image": (
                        str(pca_reference_image)
                        if pca_reference_image
                        else None
                    ),
                    "reference_resolution": args.pca_reference_resolution,
                    "volume_kind": pca_volume_kind,
                    "target_pc": args.pca_target_pc,
                    "score_label": pca_score_label,
                    "top_pcs": args.pca_top_pcs,
                    "top_pc_variance": args.pca_top_pc_variance,
                    "normalization": args.pca_normalization,
                    "score_metric": args.pca_score_metric,
                    "reference_scores": str(reference_scores_path) if reference_scores_path else None,
                    "reference_stats": str(reference_stats_path) if reference_stats_path else None,
                    "reference_stats_day": reference_stats_day,
                    "reference_stats_run": reference_stats_run,
                }
                if use_pca
                else None,
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
    pca_stop = ctx.Event()
    pca_process = None
    if use_pca:
        pca_process = ctx.Process(
            target=pca_rt.run_realtime_pca_scorer,
            kwargs={
                "run_dir": run_dir,
                "pca_root": pca_root,
                "condition": condition,
                "score_queue": score_queue,
                "stop_event": pca_stop,
                "reference_stats_path": reference_stats_path,
                "reference_stats": reference_stats,
                "volume_kind": pca_volume_kind,
                "normalization": args.pca_normalization,
                "score_metric": args.pca_score_metric,
                "target_pc": args.pca_target_pc,
                "top_pcs": args.pca_top_pcs,
                "top_pc_variance": args.pca_top_pc_variance,
                "max_trs": max_trs,
                "poll_interval": args.pca_poll_interval,
                "qc_plot_path": run_dir / "qc_pca_scores_motion.png",
                "qc_update_every": 5,
            },
        )
        pca_process.start()

    if args.feedback_signal == "z" and not args.reference_score_run:
        log.warning(
            "Feedback signal is z (normalized to resting state) but no --rs was given: "
            "the line will fall back to RAW decoder values, which for many decoders are "
            "far outside 0-100 and will pin the line. Pass --rs <RS run> for normalized feedback."
        )
    log.info("[DISPLAY] Robot line driven by %s score (z_center=%s, z_gain=%s, delay=%ss).",
             args.feedback_signal, args.z_center, args.z_gain, args.feedback_delay)
    try:
        run_robot_task_presentation(
            score_queue=score_queue,
            condition=condition,
            condition_scores_path=condition_scores_path,
            max_trs=max_trs,
            duration_s=(float(max_trs) * float(REGRESSOR_SETTINGS.TR) if max_trs else None),
            subject=args.sub,
            day=args.day,
            run=args.run,
            tr=float(REGRESSOR_SETTINGS.TR),
            feedback_delay=args.feedback_delay,
            feedback_signal=args.feedback_signal,
            z_gain=args.z_gain,
            z_center=args.z_center,
            active_feedback_trs=args.active_feedback_trs,
            rest_baseline_s=args.rest_baseline_sec,
        )
    finally:
        pca_stop.set()
        if pca_process is not None:
            if pca_process.is_alive():
                pca_process.terminate()
            pca_process.join(timeout=5)
        if pipeline_process.is_alive():
            pipeline_process.terminate()
        pipeline_process.join(timeout=5)
        if biopac_stop is not None:
            biopac_stop.set()
        if biopac_process is not None:
            biopac_process.join(timeout=5)
        if use_pca:
            pca_rt.plot_pca_scores_motion(
                run_dir=run_dir,
                scores_csv=run_dir / "pca_realtime_scores.csv",
                out_png=run_dir / "qc_pca_scores_motion.png",
                score_columns=[
                    "raw_component_score",
                    "directed_score",
                    "score_z",
                    "feedback_score",
                ],
                title="PCA neurofeedback scores and motion",
            )
        else:
            _plot_qc(run_dir)


if __name__ == "__main__":
    main()