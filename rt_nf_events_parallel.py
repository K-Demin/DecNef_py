#!/usr/bin/env python
import argparse
import csv
import json
import logging
from multiprocessing import Queue
from queue import Empty
from pathlib import Path
import multiprocessing as mp
import time
from typing import Optional

import numpy as np

from rt_global_settings import load_regressor_settings


log = logging.getLogger(__name__)

# python rt_nf_events_parallel.py \
#   --sub 00086 \
#   --day 3 \
#   --run 4 \
#   --incoming-root /path/to/incoming \
#   --base-data /SSD2/DecNef \
#   --n-trials 20 \
#   --baseline-trs 20 \
#   --iti-trs 3 \
#   --cue-trs 4 \
#   --scans-trs 3 \
#   --delay-trs 3 \
#   --feedback-trs 3 \
#   --score-delay 3 \
#   --settings-file jane_settings.json
#   --rs XXX\


#

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


def _append_trial_score(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trial",
                "score_window_start_tr",
                "score_window_end_tr",
                "n_scores_used",
                "trial_score",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_nf_events_presentation(
    score_queue: Queue,
    baseline_trs: int,
    n_trials: int,
    iti_trs: int,
    cue_trs: int,
    scans_trs: int,
    delay_trs: int,
    feedback_trs: int,
    score_delay: int,
    trial_scores_path: Path,
) -> None:
    from psychopy import core, event, visual

    stage_sum = iti_trs + cue_trs + scans_trs + delay_trs + feedback_trs
    if stage_sum <= 0:
        raise ValueError("At least one per-trial stage duration must be > 0")

    win = _build_presentation_window(visual, color=[0.0, 0.0, 0.0])
    waiting_text = visual.TextStim(
        win,
        text="Waiting for scanner trigger ('s')...",
        color="black",
        height=36,
    )
    stage_text = visual.TextStim(win, text="", pos=(0, 280), color="black", height=34)
    trial_text = visual.TextStim(win, text="", pos=(0, 220), color="black", height=28)
    fixation = visual.TextStim(win, text="+", color="black", height=80)
    score_text = visual.TextStim(win, text="", pos=(0, -280), color="black", height=28)
    feedback_circle = visual.Circle(
        win,
        radius=80,
        fillColor=[0.0, 0.0, 0.0],
        lineColor=[0.0, 0.0, 0.0],
        pos=(0, 0),
    )

    max_seen_vol = 0
    score_by_exp_tr: dict[int, float] = {}
    start_vol = None

    def drain_queue() -> None:
        nonlocal max_seen_vol
        try:
            while True:
                message = score_queue.get_nowait()
                vol_idx = int(message.get("volume_idx", 0) or 0)
                if vol_idx > 0:
                    max_seen_vol = max(max_seen_vol, vol_idx)
                if start_vol is None:
                    continue
                if vol_idx <= start_vol:
                    continue
                exp_tr = vol_idx - start_vol
                if message.get("reg_ready", True) and message.get("score_raw") is not None:
                    try:
                        score_by_exp_tr[exp_tr] = float(message["score_raw"])
                    except (TypeError, ValueError):
                        continue
        except Empty:
            pass

    # Wait for manual trigger.
    while True:
        drain_queue()
        waiting_text.draw()
        win.flip()
        keys = event.getKeys()
        if "s" in keys:
            start_vol = max_seen_vol
            break
        if "escape" in keys:
            win.close()
            return
        core.wait(0.02)

    current_end = 0
    schedule: list[dict] = []
    if baseline_trs > 0:
        current_end += baseline_trs
        schedule.append({"trial": 0, "stage": "baseline", "start_tr": 1, "end_tr": current_end})

    trial_windows: dict[int, tuple[int, int]] = {}
    for trial in range(1, n_trials + 1):
        iti_start = current_end + 1
        current_end += iti_trs
        iti_end = current_end

        cue_start = current_end + 1
        current_end += cue_trs
        cue_end = current_end

        scans_start = current_end + 1
        current_end += scans_trs
        scans_end = current_end

        delay_start = current_end + 1
        current_end += delay_trs
        delay_end = current_end

        fb_start = current_end + 1
        current_end += feedback_trs
        fb_end = current_end

        schedule.extend(
            [
                {"trial": trial, "stage": "iti", "start_tr": iti_start, "end_tr": iti_end},
                {"trial": trial, "stage": "cue", "start_tr": cue_start, "end_tr": cue_end},
                {"trial": trial, "stage": "scans", "start_tr": scans_start, "end_tr": scans_end},
                {"trial": trial, "stage": "delay", "start_tr": delay_start, "end_tr": delay_end},
                {"trial": trial, "stage": "feedback", "start_tr": fb_start, "end_tr": fb_end},
            ]
        )

        score_start = cue_start + max(0, score_delay)
        score_start = min(score_start, scans_end)
        trial_windows[trial] = (score_start, scans_end)

    trial_scores: dict[int, float] = {}

    for stage in schedule:
        trial = stage["trial"]
        stage_name = stage["stage"]
        stage_end = stage["end_tr"]

        while True:
            drain_queue()
            current_exp_tr = max(0, max_seen_vol - (start_vol or 0))

            win.color = [0.0, 0.0, 0.0]  # gray background
            if trial == 0:
                stage_text.text = f"BASELINE ({baseline_trs} TRs)"
                trial_text.text = ""
            else:
                stage_text.text = f"Trial {trial}/{n_trials} — {stage_name.upper()}"
                trial_text.text = ""

            if stage_name == "cue":
                fixation.draw()
            elif stage_name == "feedback" and trial > 0:
                if trial not in trial_scores:
                    s_start, s_end = trial_windows[trial]
                    used = [score_by_exp_tr[tr] for tr in range(s_start, s_end + 1) if tr in score_by_exp_tr]
                    trial_score = float(np.mean(used)) if used else float("nan")
                    trial_scores[trial] = trial_score
                    _append_trial_score(
                        trial_scores_path,
                        {
                            "trial": trial,
                            "score_window_start_tr": s_start,
                            "score_window_end_tr": s_end,
                            "n_scores_used": len(used),
                            "trial_score": trial_score,
                        },
                    )
                    print(
                        f"Trial {trial:02d} score over TRs [{s_start}, {s_end}] "
                        f"from {len(used)} samples: {trial_score:.4f}"
                    )

                t_score = trial_scores[trial]
                if np.isnan(t_score):
                    feedback_circle.radius = 70
                    feedback_circle.fillColor = [0.2, 0.2, 0.2]
                    feedback_circle.lineColor = [0.2, 0.2, 0.2]
                    score_text.text = "Score: NaN"
                else:
                    clipped = float(np.clip(t_score, -2.0, 2.0))
                    feedback_circle.radius = 70 + 60 * (abs(clipped) / 2.0)
                    if clipped >= 0:
                        feedback_circle.fillColor = [-0.5, 0.7, -0.5]
                        feedback_circle.lineColor = [-0.5, 0.7, -0.5]
                    else:
                        feedback_circle.fillColor = [0.8, -0.5, -0.5]
                        feedback_circle.lineColor = [0.8, -0.5, -0.5]
                    score_text.text = f"Score: {t_score:.3f}"
                feedback_circle.draw()
                score_text.draw()

            stage_text.draw()
            trial_text.draw()
            win.flip()

            if "escape" in event.getKeys():
                win.close()
                return

            if current_exp_tr >= stage_end:
                break
            core.wait(0.02)

    win.close()


def main() -> None:
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(
        description="Run rt_pipeline in parallel with an event-based NF PsychoPy presentation."
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
        "--n-trials",
        type=int,
        default=20,
        help="Number of NF trials to run.",
    )
    parser.add_argument("--baseline-trs", type=int, default=None, help="Baseline duration before trial 1 (TRs).")
    parser.add_argument("--iti-trs", type=int, default=3, help="ITI duration (TRs).")
    parser.add_argument("--cue-trs", type=int, default=4, help="Cue duration (TRs).")
    parser.add_argument("--scans-trs", type=int, default=3, help="Scans-collection duration (TRs).")
    parser.add_argument("--delay-trs", type=int, default=3, help="Delay duration (TRs).")
    parser.add_argument("--feedback-trs", type=int, default=3, help="Feedback duration (TRs).")
    parser.add_argument(
        "--rs",
        dest="reference_score_run",
        help="Reference run ID for z-scoring (uses scores.csv from that run).",
    )
    parser.add_argument(
        "--score-delay",
        type=int,
        default=3,
        help="TRs to wait after cue onset before score-window accumulation starts.",
    )
    parser.add_argument(
        "--biopac-enable",
        action="store_true",
        help="Enable BIOPAC RetroTS regressors via TCP/file input.",
    )
    parser.add_argument("--biopac-host", default="0.0.0.0", help="Host to bind BIOPAC receiver.")
    parser.add_argument("--biopac-port", type=int, default=15000, help="Port to bind BIOPAC receiver.")
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
        "--settings-file",
        default=None,
        help="Optional JSON file with global runtime settings (TR, censor thresholds, BIOPAC defaults, etc.).",
    )

    args = parser.parse_args()
    from rt_pipeline import RTSessionConfig, REGRESSOR_SETTINGS

    if args.settings_file:
        loaded = load_regressor_settings(args.settings_file)
        REGRESSOR_SETTINGS.update(vars(loaded))

    baseline_trs = (
        int(args.baseline_trs)
        if args.baseline_trs is not None
        else int(REGRESSOR_SETTINGS.voxel_norm_ref_volumes)
    )
    if baseline_trs < 0:
        raise ValueError("--baseline-trs must be >= 0")

    for name in ["n_trials", "iti_trs", "cue_trs", "scans_trs", "delay_trs", "feedback_trs", "score_delay"]:
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")

    from biopac_rt.biopac_receiver import BiopacReceiverConfig

    cfg = RTSessionConfig(
        subject=args.sub,
        day=args.day,
        run=args.run,
        incoming_root=Path(args.incoming_root),
        base_data=Path(args.base_data),
        decoder_template=Path(args.decoder_template) if args.decoder_template else None,
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

    run_dir = cfg.rt_work_dir
    trial_scores_path = run_dir / "trial_scores.csv"
    _merge_session_metadata(
        run_dir,
        {
            "psychopy": {
                "script": "rt_nf_events_parallel.py",
                "n_trials": args.n_trials,
                "baseline_trs": baseline_trs,
                "iti_trs": args.iti_trs,
                "cue_trs": args.cue_trs,
                "scans_trs": args.scans_trs,
                "delay_trs": args.delay_trs,
                "feedback_trs": args.feedback_trs,
                "score_delay": args.score_delay,
                "trial_scores_csv": str(trial_scores_path),
                "decoder_template": str(args.decoder_template) if args.decoder_template else None,
            }
        },
    )

    ctx = mp.get_context("spawn")
    score_queue = ctx.Queue(maxsize=200)
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
        expected_regressors = {"RICOR8": 8, "RVT5": 5, "RVT+RICOR13": 13}.get(args.biopac_phys_reg, 8)
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
        biopac_process = ctx.Process(target=_run_biopac_listener, args=(biopac_cfg, biopac_stop))
        biopac_process.start()

    pipeline_process = ctx.Process(target=_run_pipeline_with_settings, args=(cfg, score_queue, settings_payload))
    pipeline_process.start()

    try:
        run_nf_events_presentation(
            score_queue=score_queue,
            baseline_trs=baseline_trs,
            n_trials=args.n_trials,
            iti_trs=args.iti_trs,
            cue_trs=args.cue_trs,
            scans_trs=args.scans_trs,
            delay_trs=args.delay_trs,
            feedback_trs=args.feedback_trs,
            score_delay=args.score_delay,
            trial_scores_path=trial_scores_path,
        )
    finally:
        if pipeline_process.is_alive():
            pipeline_process.terminate()
        pipeline_process.join(timeout=5)
        if biopac_stop is not None:
            biopac_stop.set()
        if biopac_process is not None:
            biopac_process.join(timeout=5)


if __name__ == "__main__":
    main()
