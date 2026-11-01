#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import multiprocessing as mp
import random
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty
from typing import Any, Optional

from brainpong_adapter import BrainPongScoreAdapter, BrainPongScoreState, PaddleDynamics, BinaryBrainPongScoreAdapter
from task.Minjoo.brain_pong_inputs.binary import BalancedTargets, decoder_state, target_position
from rt_global_settings import load_regressor_settings


log = logging.getLogger(__name__)
CONTACT_FIELDS = [
    "timestamp", "trial_timestamp", "run_id", "block_id", "trial_id", "rally",
    "approach_id", "target_side", "decoder_state_at_contact", "z_score_at_contact",
    "participant_hit", "target_aligned_z", "level", "raw_decoder_score",
    "source_volume_index", "score_missing",
]


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def load_brainpong_settings(settings_file: Optional[Path]) -> dict[str, Any]:
    default_path = Path(__file__).with_name("brainpong_settings.json")

    if settings_file is not None:
        settings_file = Path(settings_file).expanduser()
        if not settings_file.is_absolute():
            settings_file = Path.cwd() / settings_file

        if not settings_file.exists():
            raise FileNotFoundError(
                f"Brain-pong settings file not found: {settings_file}"
            )

        if default_path.exists():
            cfg = _merge_dict(_load_json(default_path), _load_json(settings_file))
        else:
            cfg = _load_json(settings_file)
    else:
        if not default_path.exists():
            raise FileNotFoundError(
                f"No Brain-pong settings file provided and default is missing: {default_path}\n"
                "Use --brainpong-settings /path/to/settings.json"
            )
        cfg = _load_json(default_path)

    feedback = cfg.setdefault("feedback", {})
    if not int(feedback.get("feedback_delay_volumes", 0) or 0):
        delay_sec = float(feedback.get("feedback_delay_sec", 0.0) or 0.0)
        tr = float(cfg.get("TR", 1.0) or 1.0)
        if delay_sec > 0 and tr > 0:
            feedback["feedback_delay_volumes"] = int(round(delay_sec / tr))

    return binary_pilot_settings(cfg)


def binary_pilot_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """Make the effective binary policy explicit in the saved config snapshot."""
    return _merge_dict(cfg, {
        "decoder": {"category_a_direction": "up", "category_b_direction": "down", "invert_score": False},
        "feedback": {"zscore_mode": "reference", "smoothing_window": 1,
                     "initial_ignore_volumes": 0, "feedback_delay_volumes": 0,
                     "feedback_delay_sec": 0.0, "feedback_transform": "identity",
                     "dead_zone": 0.0, "gain": 1.0},
        "task": {"movement_mode": "binary", "binary_threshold": 0.0,
                 "top_position": target_position("TOP"), "bottom_position": target_position("BOTTOM")},
    })


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _append_csv(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _merge_session_metadata(run_dir: Path, payload: dict[str, Any]) -> None:
    metadata_path = run_dir / "session_metadata.json"
    data: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            data = _load_json(metadata_path)
        except (OSError, json.JSONDecodeError):
            data = {}
    data.update(payload)
    _write_json(metadata_path, data)


def _run_pipeline_with_settings(cfg: "RTSessionConfig", score_queue: mp.Queue, settings: dict[str, Any]) -> None:
    from rt_pipeline import REGRESSOR_SETTINGS, run_rt_pipeline

    for key, value in settings.items():
        if hasattr(REGRESSOR_SETTINGS, key):
            setattr(REGRESSOR_SETTINGS, key, value)
    run_rt_pipeline(cfg, score_queue)


class SyntheticScoreSource:
    def __init__(self, mode: str, tr: float, replay_path: Optional[Path] = None):
        self.mode = mode
        self.tr = tr
        self.replay_rows = self._load_replay(replay_path) if replay_path else []
        self.index = 0

    def next_score(self) -> Optional[dict[str, Any]]:
        self.index += 1
        timestamp = time.time()
        if self.mode == "sine":
            score = math.sin((self.index - 1) / 8.0)
        elif self.mode == "random":
            score = random.uniform(-1.0, 1.0)
        elif self.mode == "step":
            score = 1.0 if ((self.index - 1) // 12) % 2 == 0 else -1.0
        elif self.mode == "replay":
            if not self.replay_rows:
                return None
            row = self.replay_rows[(self.index - 1) % len(self.replay_rows)]
            score = float(row.get("score_raw") or row.get("raw_score") or row.get("feedback_value") or 0.0)
            score_z = float(row.get("score_z") or score)
        else:
            return None
        if self.mode != "replay":
            score_z = score
        return {
            "volume_idx": self.index,
            "timestamp": timestamp,
            "analysis_timestamp": timestamp,
            "watchdog_timestamp": timestamp,
            "score_raw": score,
            "score_z": score_z,
            "reg_ready": True,
            "event_type": "simulation",
        }

    @staticmethod
    def _load_replay(path: Optional[Path]) -> list[dict[str, str]]:
        if path is None or not path.exists():
            return []
        with open(path, newline="", encoding="utf-8") as f:
            sample = f.read(4096)
            f.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
            return list(csv.DictReader(f, dialect=dialect))


class FileScoreTailer:
    def __init__(self, path: Path):
        self.path = path
        self.seen_vols: set[int] = set()

    def drain(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with open(self.path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                try:
                    vol = int(row.get("volume_idx") or row.get("volume") or 0)
                except ValueError:
                    continue
                if vol <= 0 or vol in self.seen_vols:
                    continue
                self.seen_vols.add(vol)
                try:
                    score = float(row.get("score_raw") or row.get("raw_score") or row.get("score") or 0.0)
                except ValueError:
                    continue
                try:
                    score_z = float(row["score_z"]) if row.get("score_z") not in {None, ""} else None
                except ValueError:
                    score_z = None
                rows.append(
                    {
                        "volume_idx": vol,
                        "timestamp": float(row.get("timestamp") or time.time()),
                        "score_raw": score,
                        "score_z": score_z,
                        "reg_ready": bool(int(row.get("reg_ready", "1") or 1)),
                        "event_type": row.get("event_type", "file"),
                    }
                )
            return rows


class BrainPongLogger:
    def __init__(self, output_dir: Path, cfg: dict[str, Any]):
        self.output_dir = _mkdir(output_dir)
        self.cfg = cfg
        self.run_start = datetime.now(timezone.utc).isoformat()
        self.frame_path = output_dir / "brainpong_frames.csv"
        self.score_path = output_dir / "brainpong_scores.csv"
        self.trial_path = output_dir / "brainpong_trials.csv"
        self.contact_path = output_dir / "brainpong_contacts.csv"
        if not self.contact_path.exists():
            with self.contact_path.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow(CONTACT_FIELDS)
        _write_json(output_dir / "brainpong_config_snapshot.json", {"run_start": self.run_start, "config": cfg})

    def log_score(self, state: BrainPongScoreState, message: dict[str, Any], block_idx: int, trial_idx: int, target_direction: str, game_context: Optional[dict[str, Any]] = None) -> None:
        if not self.cfg.get("logging", {}).get("save_score_log", True):
            return
        row = {
            **asdict(state),
            "block_index": block_idx,
            "trial_index": trial_idx,
            "target_direction": target_direction,
            "category_pair": self.cfg["decoder"]["pair"],
            "category_a": self.cfg["decoder"]["category_a"],
            "category_b": self.cfg["decoder"]["category_b"],
            "category_a_direction": self.cfg["decoder"]["category_a_direction"],
            "category_b_direction": self.cfg["decoder"]["category_b_direction"],
            "event_type": message.get("event_type"),
            **(game_context or {}),
        }
        row["details"] = json.dumps(row.get("details") or {})
        _append_csv(
            self.score_path,
            [
                "volume_index",
                "timestamp",
                "raw_score",
                "score_z",
                "signed_score",
                "normalized_score",
                "feedback_value",
                "source_volume_index",
                "score_missing",
                "score_delayed",
                "reg_ready",
                "block_index",
                "trial_index",
                "target_direction",
                "category_pair",
                "category_a",
                "category_b",
                "category_a_direction",
                "category_b_direction",
                "event_type",
                "details",
                "run_id", "trial_id", "block_id", "received_timestamp", "stage",
                "raw_decoder_score", "z_score", "decoder_state", "target_side",
                "participant_paddle_y", "ball_y", "time_to_arrival", "level",
                "approach_id", "correct_state", "target_aligned_z",
            ],
            row,
        )

    def log_frame(self, row: dict[str, Any]) -> None:
        if not self.cfg.get("logging", {}).get("save_frame_log", True):
            return
        _append_csv(
            self.frame_path,
            [
                "timestamp",
                "elapsed_sec",
                "block_index",
                "trial_index",
                "stage",
                "trial_elapsed_sec",
                "target_direction",
                "feedback_value",
                "raw_score",
                "source_volume_index",
                "score_missing",
                "score_delayed",
                "paddle_position",
                "paddle_velocity",
                "ball_x",
                "ball_y",
                "points",
                "agent_points",
                "rally",
                "participant_response",
                "z_score", "decoder_state", "target_side", "time_to_arrival", "level", "approach_id",
            ],
            row,
        )

    def log_trial(self, row: dict[str, Any]) -> None:
        if not self.cfg.get("logging", {}).get("save_trial_log", True):
            return
        _append_csv(
            self.trial_path,
            [
                "block_index",
                "trial_index",
                "target_direction",
                "trial_start_time",
                "trial_end_time",
                "active_duration_sec",
                "duration_sec",
                "start_points",
                "end_points",
                "agent_points",
                "player_won",
                "total_hits",
                "total_possible",
                "performance",
                "level",
                "paddle_size",
                "point_increment",
                "final_paddle_position",
                "max_abs_paddle_position",
                "n_score_updates",
                "n_missing_frames",
                "completed",
                "participant_hits", "participant_misses", "participant_hit_rate",
                "opponent_hits", "opponent_misses", "participant_score", "opponent_score",
                "round_result", "level_before", "level_after", "agent_miss_prob",
            ],
            row,
        )

    def log_contact(self, row: dict[str, Any]) -> None:
        _append_csv(self.contact_path, CONTACT_FIELDS, row)

    def finish(self, final_state: Optional[dict[str, Any]] = None) -> None:
        summary = {
            "run_start": self.run_start,
            "run_end": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(self.output_dir),
        }
        if final_state is not None:
            summary["final_state"] = final_state
            _write_json(self.output_dir / "brainpong_next_state.json", final_state)
        _write_json(self.output_dir / "brainpong_run_summary.json", summary)


def _build_window(visual, task_cfg: dict[str, Any]):
    fullscr = bool(task_cfg.get("fullscr", False))
    return visual.Window(
        size=[1920, 1080],
        units="height",
        fullscr=fullscr,
        winType="pyglet",
        color="black",
        screen=0,
    )


def _target_direction_for_trial(task_cfg: dict[str, Any], block_idx: int, trial_idx: int) -> str:
    schedule = task_cfg.get("target_schedule")
    if schedule:
        return str(schedule[(block_idx * int(task_cfg.get("trials_per_block", 1)) + trial_idx) % len(schedule)]).lower()
    if str(task_cfg.get("target_mode", "up_down")).lower() == "up_down":
        return "up" if (block_idx + trial_idx) % 2 == 0 else "down"
    if str(task_cfg.get("target_mode", "")).lower() == "pong_ball":
        return "ball"
    return str(task_cfg.get("target_direction", "up")).lower()


def _run_legacy_brainpong_presentation(
    cfg: dict[str, Any],
    score_queue: Optional[mp.Queue] = None,
    score_file: Optional[Path] = None,
) -> None:
    from psychopy import core, event, visual
    from task.Minjoo.brain_pong_inputs.pong import Pong

    feedback_cfg = cfg["feedback"]
    task_cfg = cfg["task"]
    game_cfg = dict(cfg.get("game", {}))
    sim_cfg = cfg.get("simulation", {})
    logger = BrainPongLogger(Path(cfg["logging"]["output_dir"]), cfg)
    adapter = BrainPongScoreAdapter(cfg["decoder"], feedback_cfg)
    dynamics = PaddleDynamics(feedback_cfg, task_cfg)
    source = SyntheticScoreSource(
        str(sim_cfg.get("score_source", "sine")),
        float(cfg.get("TR", 1.0)),
        Path(sim_cfg["replay_path"]) if sim_cfg.get("replay_path") else None,
    ) if sim_cfg.get("enabled") else None
    tailer = FileScoreTailer(score_file) if score_file else None

    win = _build_window(visual, task_cfg)
    fixation = visual.ShapeStim(
        win,
        units="norm",
        vertices="cross",
        size=(0.07, 0.07),
        fillColor="white",
        lineColor="white",
    )
    waiting_text = visual.TextStim(win, units="norm", text="Waiting for scanner trigger ('s')...", height=0.06, color="white")
    sim_mouse = event.Mouse(win=win) if sim_cfg.get("enabled") and str(sim_cfg.get("input_method", "")).lower() in {"mouse_wheel", "wheel"} else None

    first_trigger_timestamp: Optional[float] = None
    if not sim_cfg.get("enabled") or bool(task_cfg.get("wait_for_trigger_in_simulation", False)):
        while True:
            waiting_text.draw()
            win.flip()
            keys = event.getKeys()
            if "s" in keys:
                first_trigger_timestamp = time.time()
                break
            if "escape" in keys:
                win.close()
                logger.finish()
                return
            core.wait(0.02)
    else:
        first_trigger_timestamp = time.time()

    def drain_scores(block_idx: int, trial_idx: int, target_direction: str) -> BrainPongScoreState:
        messages: list[dict[str, Any]] = []
        if source and (time.time() - drain_scores.last_sim_time) >= float(cfg.get("TR", 1.0)):
            msg = source.next_score()
            drain_scores.last_sim_time = time.time()
            if msg:
                messages.append(msg)
        if tailer:
            messages.extend(tailer.drain())
        if score_queue is not None:
            try:
                while True:
                    messages.append(score_queue.get_nowait())
            except Empty:
                pass

        state = adapter.current_state()
        for message in messages:
            if not bool(message.get("reg_ready", True)):
                continue
            if message.get("score_raw") is None:
                continue
            state = adapter.update(
                raw_score=float(message["score_raw"]),
                score_z=(
                    float(message["score_z"])
                    if message.get("score_z") is not None
                    else None
                ),
                volume_index=int(message.get("volume_idx") or message.get("volume_index")),
                timestamp=float(message.get("timestamp") or time.time()),
                reg_ready=message.get("reg_ready"),
                event_type=message.get("event_type"),
            )
            logger.log_score(state, message, block_idx, trial_idx, target_direction)
        return state

    drain_scores.last_sim_time = 0.0

    trial_duration = float(task_cfg.get("trial_duration_sec", 20.0))
    initial_baseline_sec = float(task_cfg.get("initial_baseline_sec", 0.0))
    iti_sec = float(task_cfg.get("iti_sec", 0.0))
    if trial_duration <= 0:
        raise ValueError("trial_duration_sec must be > 0")
    if initial_baseline_sec < 0 or iti_sec < 0:
        raise ValueError("initial_baseline_sec and iti_sec must be >= 0")
    n_blocks = int(task_cfg.get("n_blocks", 1))
    trials_per_block = int(task_cfg.get("trials_per_block", 10))
    global_start = time.time()
    clock = core.Clock()

    try:
        for block_idx in range(n_blocks):
            block_points = int(game_cfg.get("init_points", 0))
            baseline_clock = core.Clock()
            while baseline_clock.getTime() < initial_baseline_sec:
                state = drain_scores(block_idx, -1, "")
                if "escape" in event.getKeys():
                    raise KeyboardInterrupt

                remaining = initial_baseline_sec - baseline_clock.getTime()
                fixation.fillColor = "red" if remaining <= 1.0 else "white"
                fixation.lineColor = fixation.fillColor
                fixation.draw()
                win.flip()

                logger.log_frame(
                    {
                        "timestamp": time.time(),
                        "elapsed_sec": time.time() - global_start,
                        "block_index": block_idx,
                        "trial_index": -1,
                        "stage": "initial_baseline",
                        "trial_elapsed_sec": "",
                        "target_direction": "",
                        "feedback_value": state.feedback_value,
                        "raw_score": state.raw_score,
                        "source_volume_index": state.source_volume_index,
                        "score_missing": state.score_missing,
                        "score_delayed": state.score_delayed,
                        "paddle_position": dynamics.position,
                        "paddle_velocity": 0.0,
                        "ball_x": "",
                        "ball_y": "",
                        "points": "",
                        "agent_points": "",
                        "rally": "",
                        "participant_response": "",
                    }
                )
                core.wait(0.001)

            for trial_idx in range(trials_per_block):
                target_direction = _target_direction_for_trial(task_cfg, block_idx, trial_idx)
                if bool(task_cfg.get("reset_position_each_trial", True)):
                    dynamics.reset()
                trial_game_cfg = dict(game_cfg)
                start_points = int(trial_game_cfg.pop("init_points", 0))
                if trial_idx > 0:
                    start_points = block_points
                game = Pong(
                    win=win,
                    logger=log,
                    init_points=start_points,
                    save_file=(
                        logger.output_dir
                        / f"brainpong_game_block{block_idx + 1:02d}_trial{trial_idx + 1:02d}.tsv"
                    ),
                    time_limit=trial_duration,
                    pre_delay=0.0,
                    input_method="brain",
                    **trial_game_cfg,
                )
                game.start_time = core.getTime()
                trial_start = time.time()
                trial_clock = core.Clock()
                clock.reset()
                max_abs_pos = 0.0
                score_updates = 0
                missing_frames = 0
                completed = True

                while trial_clock.getTime() < trial_duration:
                    state = drain_scores(block_idx, trial_idx, target_direction)
                    keys = event.getKeys()
                    participant_response = ""
                    if sim_cfg.get("enabled") and str(sim_cfg.get("input_method", "")).lower() == "keyboard":
                        if "up" in keys:
                            state.feedback_value = 1.0
                            participant_response = "up"
                        elif "down" in keys:
                            state.feedback_value = -1.0
                            participant_response = "down"
                    if sim_mouse is not None:
                        wheel = sim_mouse.getWheelRel()
                        if wheel[1] > 0:
                            state.feedback_value = 1.0
                            participant_response = "wheel_up"
                        elif wheel[1] < 0:
                            state.feedback_value = -1.0
                            participant_response = "wheel_down"
                    if "escape" in keys:
                        completed = False
                        raise KeyboardInterrupt

                    dt = max(clock.getTime(), 1e-4)
                    clock.reset()
                    pos, vel = dynamics.update(state.feedback_value, dt, target_direction=target_direction)
                    game.set_brain_position(pos)
                    game.update(dt)
                    max_abs_pos = max(max_abs_pos, abs(pos))
                    score_updates += int(not state.score_missing)
                    missing_frames += int(state.score_missing)

                    game.draw()
                    win.flip()

                    logger.log_frame(
                        {
                            "timestamp": time.time(),
                            "elapsed_sec": time.time() - global_start,
                            "block_index": block_idx,
                            "trial_index": trial_idx,
                            "stage": "feedback",
                            "trial_elapsed_sec": trial_clock.getTime(),
                            "target_direction": target_direction,
                            "feedback_value": state.feedback_value,
                            "raw_score": state.raw_score,
                            "source_volume_index": state.source_volume_index,
                            "score_missing": state.score_missing,
                            "score_delayed": state.score_delayed,
                            "paddle_position": pos,
                            "paddle_velocity": vel,
                            "ball_x": float(game.ball.pos[0]),
                            "ball_y": float(game.ball.pos[1]),
                            "points": game.points,
                            "agent_points": game.agent_points,
                            "rally": game.rally,
                            "participant_response": participant_response,
                        }
                    )
                    core.wait(0.001)

                trial_end = time.time()
                game.export()
                game.compute_performance()
                block_points = game.points
                logger.log_trial(
                    {
                        "block_index": block_idx,
                        "trial_index": trial_idx,
                        "target_direction": target_direction,
                        "trial_start_time": trial_start,
                        "trial_end_time": trial_end,
                        "active_duration_sec": trial_end - trial_start,
                        "duration_sec": trial_end - trial_start,
                        "start_points": start_points,
                        "end_points": game.points,
                        "agent_points": game.agent_points,
                        "player_won": (game.player_points > game.agent_points) - (game.player_points < game.agent_points),
                        "total_hits": game.total_hits,
                        "total_possible": game.total_possible,
                        "performance": game.performance,
                        "final_paddle_position": dynamics.position,
                        "max_abs_paddle_position": max_abs_pos,
                        "n_score_updates": score_updates,
                        "n_missing_frames": missing_frames,
                        "completed": completed,
                    }
                )

                if trial_idx < trials_per_block - 1:
                    iti_clock = core.Clock()
                    while iti_clock.getTime() < iti_sec:
                        state = drain_scores(block_idx, trial_idx, target_direction)
                        if "escape" in event.getKeys():
                            raise KeyboardInterrupt
                        fixation.fillColor = "white"
                        fixation.lineColor = "white"
                        fixation.draw()
                        win.flip()
                        logger.log_frame(
                            {
                                "timestamp": time.time(),
                                "elapsed_sec": time.time() - global_start,
                                "block_index": block_idx,
                                "trial_index": trial_idx,
                                "stage": "iti",
                                "trial_elapsed_sec": "",
                                "target_direction": "",
                                "feedback_value": state.feedback_value,
                                "raw_score": state.raw_score,
                                "source_volume_index": state.source_volume_index,
                                "score_missing": state.score_missing,
                                "score_delayed": state.score_delayed,
                                "paddle_position": dynamics.position,
                                "paddle_velocity": 0.0,
                                "ball_x": "",
                                "ball_y": "",
                                "points": block_points,
                                "agent_points": "",
                                "rally": "",
                                "participant_response": "",
                            }
                        )
                        core.wait(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        win.close()
        logger.finish()


def _resolve_task_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def run_brainpong_presentation(
    cfg: dict[str, Any],
    score_queue: Optional[mp.Queue] = None,
    score_file: Optional[Path] = None,
) -> None:
    """Run the shared behavioral Pong task with realtime neural control."""
    from psychopy import core, event, visual
    from task.Minjoo.brain_pong_inputs.block import TrialBlock
    from task.Minjoo.brain_pong_inputs.main import (
        handle_game_config,
        handle_levels,
        load_resume_state,
    )
    from task.Minjoo.brain_pong_inputs.utils import get_game_params

    cfg = binary_pilot_settings(cfg)
    feedback_cfg = cfg["feedback"]
    task_cfg = cfg["task"]
    sim_cfg = cfg.get("simulation", {})
    config_path = _resolve_task_path(task_cfg["config_path"])
    levels_path = _resolve_task_path(task_cfg["levels_path"])
    params = handle_game_config(config_path)
    levels = handle_levels(levels_path)
    game_params = get_game_params(params)
    game_params["input_method"] = "brain"
    cfg["binary_game_params"] = dict(game_params)
    cfg["binary_levels"] = levels
    game_params["target_schedule"] = BalancedTargets(task_cfg.get("target_seed"))
    game_params["run_id"] = str(cfg.get("session", {}).get("scanner_run", task_cfg.get("nf_run", 1)))

    logger = BrainPongLogger(Path(cfg["logging"]["output_dir"]), cfg)
    adapter = BinaryBrainPongScoreAdapter(cfg["decoder"], feedback_cfg)
    source = (
        SyntheticScoreSource(
            str(sim_cfg.get("score_source", "sine")),
            float(cfg.get("TR", 1.0)),
            Path(sim_cfg["replay_path"]) if sim_cfg.get("replay_path") else None,
        )
        if sim_cfg.get("enabled")
        else None
    )
    tailer = FileScoreTailer(score_file) if score_file else None

    win = visual.Window(
        size=tuple(params.get("win_size", [1920, 1080])),
        units="pix",
        color="black",
        pos=(0, 0),
        allowGUI=True,
        checkTiming=False,
        screen=int(params.get("screen", 0)),
        monitor=str(params.get("monitor", "testMonitor")),
        fullscr=False,
        winType="pyglet",
    )

    win.getActualFrameRate()
    win.fullscr = bool(params.get("fullscr", True))
    win.mouseVisible = bool(params.get("mouseVisible", False))
    win.flip()
    waiting_text = visual.TextStim(
        win,
        units="norm",
        text="Waiting for scanner trigger ('s')...",
        height=0.06,
        color="white",
    )
    sim_mouse = (
        event.Mouse(win=win)
        if sim_cfg.get("enabled")
        and str(sim_cfg.get("input_method", "")).lower() in {"mouse_wheel", "wheel"}
        else None
    )

    if not sim_cfg.get("enabled") or bool(task_cfg.get("wait_for_trigger_in_simulation", False)):
        while True:
            waiting_text.draw()
            win.flip()
            keys = event.getKeys()
            if "s" in keys:
                break
            if "escape" in keys:
                win.close()
                logger.finish()
                return
            core.wait(0.02)

    global_start = time.time()
    latest_state = adapter.current_state()
    context: dict[str, Any] = {
        "block": 0,
        "trial": 0,
        "stage": "idle",
        "position": target_position("BOTTOM"),
        "game": None,
        "velocity": 0.0,
        "participant_response": "",
        "score_updates": 0,
        "missing_frames": 0,
        "max_abs_position": 0.0,
    }

    def drain_scores() -> BrainPongScoreState:
        nonlocal latest_state
        messages: list[dict[str, Any]] = []
        if sim_cfg.get("enabled"):
            manual = None
            response = ""
            if str(sim_cfg.get("input_method", "")).lower() == "keyboard":
                keys = event.getKeys(keyList=["up", "down"])
                if "up" in keys:
                    manual, response = 1.0, "up"
                elif "down" in keys:
                    manual, response = -1.0, "down"
            if sim_mouse is not None:
                wheel = sim_mouse.getWheelRel()[1]
                if wheel:
                    manual = 1.0 if wheel > 0 else -1.0
                    response = "wheel_up" if wheel > 0 else "wheel_down"
            if manual is not None:
                context["manual_z"] = manual
                context["participant_response"] = response
        if source and (time.time() - drain_scores.last_sim_time) >= float(cfg.get("TR", 1.0)):
            message = source.next_score()
            drain_scores.last_sim_time = time.time()
            if message:
                if "manual_z" in context:
                    message["score_raw"] = message["score_z"] = context["manual_z"]
                    message["event_type"] = "manual_simulation"
                messages.append(message)
        if tailer:
            messages.extend(tailer.drain())
        if score_queue is not None:
            try:
                while True:
                    messages.append(score_queue.get_nowait())
            except Empty:
                pass

        # Pipeline completion order need not equal acquisition order.
        messages.sort(key=lambda m: int(m.get("volume_idx") or m.get("volume_index") or 0))
        for message in messages:
            if not bool(message.get("reg_ready", True)) or message.get("score_raw") is None:
                continue
            volume = int(message.get("volume_idx") or message.get("volume_index"))
            if latest_state.volume_index is not None and volume <= latest_state.volume_index:
                continue
            latest_state = adapter.update(
                raw_score=float(message["score_raw"]),
                score_z=(float(message["score_z"]) if message.get("score_z") is not None else None),
                volume_index=int(message.get("volume_idx") or message.get("volume_index")),
                timestamp=float(message.get("timestamp") or time.time()),
                reg_ready=message.get("reg_ready"),
                event_type=message.get("event_type"),
            )
            context["score_updates"] += 1
            logger.log_score(
                latest_state,
                message,
                int(context["block"]),
                int(context["trial"]),
                "ball",
                score_context(latest_state),
            )
        return latest_state

    drain_scores.last_sim_time = 0.0

    def score_context(state):
        game = context["game"]
        side = decoder_state(state.score_z if state.score_z is not None else 0.0)
        approaching = game is not None and context["stage"] == "feedback" and game.time_to_arrival() is not None
        target = game.target_side if approaching else ""
        return {
            "run_id": game_params["run_id"], "block_id": context["block"], "trial_id": context["trial"],
            "received_timestamp": time.time(), "stage": context["stage"],
            "raw_decoder_score": state.raw_score, "z_score": state.score_z, "decoder_state": side,
            "target_side": target, "participant_paddle_y": target_position(side),
            "ball_y": float(game.ball.pos[1]) if game else None,
            "time_to_arrival": game.time_to_arrival() if approaching else None,
            "level": game.level if game else None,
            "approach_id": game.approach_id if approaching else None,
            "correct_state": side == target if target else None,
            "target_aligned_z": (state.score_z if target == "TOP" else -state.score_z)
                if target and state.score_z is not None else None,
        }

    def neural_controller(dt: float) -> float:
        context["stage"] = "feedback"
        state = drain_scores()
        z = state.score_z if state.score_z is not None else 0.0
        position = target_position(decoder_state(z))
        context["missing_frames"] += int(state.score_missing)
        context["max_abs_position"] = max(float(context["max_abs_position"]), abs(position))
        context.update({"position": position, "velocity": 0.0})
        return z

    def log_contact(game, record):
        logger.log_contact({
            **record, "trial_timestamp": record["timestamp"], "timestamp": time.time(),
            "block_id": context["block"], "raw_decoder_score": latest_state.raw_score,
            "source_volume_index": latest_state.source_volume_index, "score_missing": latest_state.score_missing,
        })

    def log_game_frame(game, dt: float, trial_timestamp: float) -> None:
        state = latest_state
        logger.log_frame(
            {
                "timestamp": time.time(),
                "elapsed_sec": time.time() - global_start,
                "block_index": context["block"],
                "trial_index": context["trial"],
                "stage": "feedback",
                "trial_elapsed_sec": trial_timestamp,
                "target_direction": "ball",
                "feedback_value": state.feedback_value,
                "raw_score": state.raw_score,
                "source_volume_index": state.source_volume_index,
                "score_missing": state.score_missing,
                "score_delayed": state.score_delayed,
                "paddle_position": context["position"],
                "paddle_velocity": context["velocity"],
                "ball_x": float(game.ball.pos[0]),
                "ball_y": float(game.ball.pos[1]),
                "points": game.points,
                "agent_points": game.agent_points,
                "rally": game.rally,
                "participant_response": context["participant_response"],
                "z_score": game.z_score, "decoder_state": game.decoder_state,
                "target_side": game.target_side if game.ball_vel[0] < 0 else "",
                "time_to_arrival": game.time_to_arrival(), "level": game.level,
                "approach_id": game.approach_id,
            }
        )

    def log_idle_frame(game, elapsed: float, remaining: float) -> None:
        stage = "initial_baseline" if int(context["trial"]) == 1 else "iti"
        context["stage"] = stage
        state = drain_scores()
        game.set_brain_position(state.score_z if state.score_z is not None else 0.0)
        context["position"] = float(game.left_paddle.pos[1])
        logger.log_frame(
            {
                "timestamp": time.time(),
                "elapsed_sec": time.time() - global_start,
                "block_index": context["block"],
                "trial_index": context["trial"],
                "stage": stage,
                "trial_elapsed_sec": "",
                "target_direction": "",
                "feedback_value": state.feedback_value,
                "raw_score": state.raw_score,
                "source_volume_index": state.source_volume_index,
                "score_missing": state.score_missing,
                "score_delayed": state.score_delayed,
                "paddle_position": context["position"],
                "paddle_velocity": 0.0,
                "ball_x": "",
                "ball_y": "",
                "points": game.points,
                "agent_points": game.agent_points,
                "rally": "",
                "participant_response": "",
            }
        )

    def run_neural_game(game, block_num: int, trial_num: int, refresh_rate: float) -> bool:
        context.update(
            {
                "block": block_num,
                "trial": trial_num,
                "stage": "feedback",
                "score_updates": 0,
                "missing_frames": 0,
                "max_abs_position": 0.0,
            }
        )
        context["game"] = game
        start_points = game.points
        game.contact_callback = log_contact
        game.brain_controller = neural_controller
        game.frame_callback = log_game_frame
        game.idle_callback = log_idle_frame
        completed = game.run(show_instructions=False, refresh_rate=refresh_rate)
        total_duration = max(0.0, core.getTime() - game.start_time)
        logger.log_trial(
            {
                "block_index": block_num,
                "trial_index": trial_num,
                "target_direction": "ball",
                "trial_start_time": game.start_time,
                "trial_end_time": core.getTime(),
                "active_duration_sec": max(0.0, total_duration - game.pre_delay),
                "duration_sec": total_duration,
                "start_points": start_points,
                "end_points": game.points,
                "agent_points": game.agent_points,
                "player_won": (game.player_points > game.agent_points) - (game.player_points < game.agent_points),
                "total_hits": game.total_hits,
                "total_possible": game.total_possible,
                "performance": game.performance,
                "participant_hits": game.participant_hits, "participant_misses": game.participant_misses,
                "participant_hit_rate": game.performance,
                "opponent_hits": game.agent_hits, "opponent_misses": game.agent_misses,
                "participant_score": game.player_points, "opponent_score": game.agent_points,
                "agent_miss_prob": game.agent_miss_prob,
                "round_result": "WIN" if game.player_points > game.agent_points else ("LOSS" if game.player_points < game.agent_points else "DRAW"),
                "level_before": game.level,
                "level_after": min(game.level + 1, max(levels)) if game.player_points > game.agent_points else (max(0, game.level - 1) if game.player_points < game.agent_points else game.level),
                "final_paddle_position": context["position"],
                "max_abs_paddle_position": context["max_abs_position"],
                "n_score_updates": context["score_updates"],
                "n_missing_frames": context["missing_frames"],
                "completed": completed,
                "level": game.level,
                "paddle_size": game.paddle_height,
                "point_increment": game.point_increment,
            }
        )
        return completed

    n_blocks = int(task_cfg.get("n_blocks", 1))
    n_trials = int(task_cfg.get("trials_per_block", 6))
    start_block = int(task_cfg.get("start_block", 1))
    nf_run = int(task_cfg.get("nf_run", start_block))
    start_level = task_cfg.get("start_level")
    current_level = int(params["init_level"] if start_level is None else start_level)
    total_points = int(params["init_points"])
    resume_from = task_cfg.get("resume_from")
    if resume_from and start_level is not None:
        raise ValueError("Use either resume_from or start_level, not both.")
    if resume_from:
        resume_path = Path(resume_from)
        if not resume_path.is_absolute():
            resume_path = Path.cwd() / resume_path
        current_level = load_resume_state(resume_path, max(levels.keys()))
        log.info(
            "Resuming Brain-pong from %s at block %d: level=%d points reset to %d",
            resume_path,
            start_block,
            current_level,
            total_points,
        )
    else:
        log.info(
            "Starting Brain-pong NF run %d (internal block %d): level=%d points=%d",
            nf_run,
            start_block,
            current_level,
            total_points,
        )
    if current_level not in levels:
        raise ValueError(
            f"Start level {current_level} is not defined in {levels_path}. "
            f"Available levels: {min(levels.keys())}-{max(levels.keys())}"
        )
    initial_level = current_level
    completed_blocks = 0
    attempted_blocks: list[int] = []
    last_trial_summary: dict[str, Any] | None = None
    final_state: dict[str, Any] | None = None
    try:
        for block_num in range(start_block, start_block + n_blocks):
            trial_block = TrialBlock(
                block_num=block_num,
                n_trials=n_trials,
                data_dir=logger.output_dir,
                level_dict=levels,
                init_level=current_level,
                init_points=total_points,
                block_pre_delay=float(params["block_pre_delay"]),
                performance_criteria=float(params["performance_criteria"]),
                win=win,
                game_params=game_params,
                use_aggressive_leveling=bool(params["use_aggressive_leveling"]),
                game_runner=run_neural_game,
                show_instructions=False,
            )
            completed = trial_block.run()
            attempted_blocks.append(block_num)
            block_data = getattr(trial_block, "block_data", [])
            if block_data:
                last_trial = block_data[-1]
                last_trial_summary = {
                    "block_number": int(last_trial["block_number"]),
                    "trial_number": int(last_trial["trial_number"]),
                    "end_level": int(last_trial["diff_level"]),
                    "end_points": int(last_trial["end_points"]),
                    "performance": float(last_trial["performance"]),
                }
            total_points = trial_block.total_points
            current_level = trial_block.current_level
            if completed:
                completed_blocks += 1
            if not completed:
                break
    finally:
        final_state = {
            "nf_run": nf_run,
            "internal_block_number": start_block,
            "attempted_internal_blocks": attempted_blocks,
            "completed": completed_blocks == n_blocks,
            "completed_internal_blocks": completed_blocks,
            "movement_mode": task_cfg.get("movement_mode"),
            "start_level": initial_level,
            "next_start_level": current_level,
            "points_reset_each_run": True,
            "end_points": total_points,
            "output_dir": str(logger.output_dir),
        }
        session = cfg.get("session")
        if isinstance(session, dict):
            final_state["session"] = session
        if last_trial_summary is not None:
            final_state["end_level"] = last_trial_summary["end_level"]
            final_state["end_performance"] = last_trial_summary["performance"]
            final_state["last_trial"] = last_trial_summary
        try:
            logger.finish(final_state)
            log.info(
                "Brain-pong next_start_level=%d written to %s",
                current_level,
                logger.output_dir / "brainpong_next_state.json",
            )
        finally:
            win.close()


def _prepare_rt_config(args: argparse.Namespace, cfg: dict[str, Any]):
    from rt_pipeline import RTSessionConfig

    decoder = cfg["decoder"]
    return RTSessionConfig(
        subject=args.sub or str(cfg.get("subject_id", "sub-XX")).replace("sub-", ""),
        day=args.day or str(cfg.get("day_id", "day-XX")),
        run=args.run or str(cfg.get("run_id", "run-XX")),
        incoming_root=Path(args.incoming_root),
        base_data=Path(args.base_data),
        decoder_template=Path(args.decoder_template or decoder.get("decoder_path")) if (args.decoder_template or decoder.get("decoder_path")) else None,
        decoder_roi_txt=Path(args.decoder_roi_txt) if args.decoder_roi_txt else None,
        reference_score_run=args.reference_score_run,
        enable_scoring=True,
        ap_block=args.ap_block,
        pa_block=args.pa_block,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Run Brain-pong with DecNef realtime decoder feedback.")
    parser.add_argument("--brainpong-settings", type=Path, default=None, help="Brain-pong JSON settings file.")
    parser.add_argument("--settings-file", default=None, help="Optional DecNef realtime settings JSON.")
    parser.add_argument("--no-pipeline", action="store_true", help="Do not spawn rt_pipeline; use simulation or --score-file.")
    parser.add_argument("--score-file", type=Path, default=None, help="Tail an existing scores.csv-style file.")
    parser.add_argument("--sub", default=None)
    parser.add_argument("--day", default=None)
    parser.add_argument("--run", default=None)
    parser.add_argument("--incoming-root", default="/home/sin/DecNef_pain_Dec23/realtime/incoming/pain7T/20251105.20251105_00085.Kostya")
    parser.add_argument("--base-data", default="/SSD2/DecNef_py/data")
    parser.add_argument("--decoder-template", default=None)
    parser.add_argument("--decoder-roi-txt", default=None)
    parser.add_argument("--rs", dest="reference_score_run", default=None)
    parser.add_argument(
        "--ap-block",
        type=int,
        default=None,
        help="AP block to use for b0.",
    )
    parser.add_argument(
        "--pa-block",
        type=int,
        default=None,
        help="PA block to use for b0.",
    )
    parser.add_argument(
        "--nf-run",
        type=int,
        default=None,
        help="Neurofeedback run number. Used for Brain-pong output naming.",
    )
    parser.add_argument(
        "--start-level",
        type=int,
        default=None,
        help="Difficulty level to start this NF run at. Use the previous run's next_start_level.",
    )
    parser.add_argument(
        "--start-block",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    cfg = load_brainpong_settings(args.brainpong_settings)
    if args.resume_from is not None and args.start_level is not None:
        parser.error("--start-level replaces --resume-from; use only one.")
    if args.resume_from is not None and args.start_block is None and args.nf_run is None:
        parser.error("--resume-from requires --nf-run or --start-block")
    if args.nf_run is not None:
        if args.nf_run < 1:
            parser.error("--nf-run must be 1 or greater")
        if args.start_block is not None and args.start_block != args.nf_run:
            parser.error("--nf-run and --start-block must match when both are provided")
        cfg["task"]["start_block"] = args.nf_run
        cfg["task"]["nf_run"] = args.nf_run
    elif args.start_block is not None:
        if args.start_block < 1:
            parser.error("--start-block must be 1 or greater")
        cfg["task"]["start_block"] = args.start_block
        cfg["task"]["nf_run"] = args.start_block
    else:
        cfg["task"]["nf_run"] = int(cfg["task"].get("start_block", 1))
    if args.start_level is not None:
        if args.start_level < 0:
            parser.error("--start-level must be 0 or greater")
        cfg["task"]["start_level"] = args.start_level
    if args.resume_from is not None:
        cfg["task"]["resume_from"] = str(args.resume_from)
    cfg["session"] = {
        "subject": args.sub or str(cfg.get("subject_id", "sub-XX")).replace("sub-", ""),
        "day": args.day or str(cfg.get("day_id", "day-XX")),
        "scanner_run": args.run or str(cfg.get("run_id", "run-XX")),
        "nf_run": cfg["task"].get("nf_run"),
        "reference_score_run": args.reference_score_run,
        "ap_block": args.ap_block,
        "pa_block": args.pa_block,
    }
    zscore_mode = str(cfg.get("feedback", {}).get("zscore_mode", "none")).lower()
    uses_reference_score = zscore_mode in {"reference", "pipeline"}
    runs_pipeline = not args.no_pipeline and not cfg.get("simulation", {}).get("enabled", False)
    if uses_reference_score and runs_pipeline and not args.reference_score_run:
        parser.error("Brain-pong zscore_mode='reference' requires --rs REFERENCE_RUN")
    rt_settings = load_regressor_settings(args.settings_file) if args.settings_file else load_regressor_settings()
    for key in ["TR", "analysis_space"]:
        if key in cfg and hasattr(rt_settings, key):
            setattr(rt_settings, key, cfg[key])

    score_queue = None
    pipeline_process = None
    rt_cfg = None
    run_dir = Path(cfg["logging"]["output_dir"])

    if runs_pipeline:
        rt_cfg = _prepare_rt_config(args, cfg)
        if uses_reference_score:
            from rt_pipeline import load_reference_score_stats

            reference_stats = load_reference_score_stats(rt_cfg, rt_cfg.reference_score_run)
            if reference_stats is None:
                raise FileNotFoundError(
                    "Could not load reference-run score statistics. Check --rs, "
                    "subject/day, and the reference run scores.csv."
                )
        run_dir = rt_cfg.rt_work_dir
        cfg["logging"]["output_dir"] = str(run_dir / "brainpong")
        if uses_reference_score:
            cfg["rest_calibration"] = reference_stats
            _write_json(run_dir / "brainpong" / "brainpong_rest_calibration.json", reference_stats)
            log.info("Binary pilot resting calibration: %s", reference_stats)
            if reference_stats.get("binary_side_bias_flag"):
                log.warning("Resting z sign occupancy exceeds 75%% on one side; threshold remains z=0.")
        _merge_session_metadata(
            run_dir,
            {
                "psychopy": {
                    "script": "rt_brainpong_parallel_old.py",
                    "brainpong_config": cfg,
                    "decoder_template": str(rt_cfg.decoder_template) if rt_cfg.decoder_template else None,
                    "ap_block": args.ap_block,
                    "pa_block": args.pa_block,
                    "fmap_dir": str(rt_cfg.fmap_dir),
                }
            },
        )
        ctx = mp.get_context("spawn")
        score_queue = ctx.Queue(maxsize=200)
        pipeline_process = ctx.Process(target=_run_pipeline_with_settings, args=(rt_cfg, score_queue, vars(rt_settings).copy()))
        pipeline_process.start()

    try:
        run_brainpong_presentation(cfg, score_queue=score_queue, score_file=args.score_file)
    finally:
        if pipeline_process is not None:
            if pipeline_process.is_alive():
                pipeline_process.terminate()
            pipeline_process.join(timeout=5)
        if rt_cfg is not None:
            try:
                from brainpong_qc import generate_scores_motion_qc

                generate_scores_motion_qc(rt_cfg.rt_work_dir)
            except Exception:
                log.exception("Failed to create qc_scores_motion.png for %s", rt_cfg.rt_work_dir)


if __name__ == "__main__":
    main()
