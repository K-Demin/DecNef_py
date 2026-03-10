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
        else:
            window_kwargs.update(
                {
                    "screen": 0,
                    "fullscr": True,
                }
            )

    except Exception as exc:
        log.warning("Could not detect external monitor; using default window size: %s", exc)

    try:
        return visual.Window(**window_kwargs)
    except Exception as exc:
        # PsychoPy can fail on fullscreen/monitor-size combinations depending on
        # backend and local monitor settings. Retry with a conservative setup so
        # the presentation can still run.
        log.warning("Primary PsychoPy window config failed; retrying with safe defaults: %s", exc)
        return visual.Window(
            size=default_size,
            color=color,
            units="pix",
            screen=0,
            fullscr=False,
        )


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

    win = _build_presentation_window(visual, color=[-0.004, -0.004, -0.004])
    waiting_text = visual.TextStim(
        win,
        text="Waiting for scanner trigger ('s')...",
        color="black",
        height=36,
    )
    fixation = visual.ShapeStim(
        win,
        vertices="cross",
        size=(0.05, 0.05),
        lineWidth=1.0,
        fillColor=[0.0, 0.0, 0.0],
        lineColor=[0.0, 0.0, 0.0],
        pos=(0, 0),
    )
    feedback_circle = visual.Circle(
        win,
        radius=0,
        fillColor=[0.0, 0.0, 0.0],
        lineColor=[0.0, 0.0, 0.0],
        pos=(0, 0),
    )
    # Scale circles relative to the presentation window; reference and
    # feedback share the same maximum radius at a perfect score.
    max_reference_radius = 0.30 * min(win.size)
    max_feedback_radius = max_reference_radius
    max_reference_circle = visual.Circle(
        win,
        radius=max_reference_radius,
        fillColor=None,
        lineColor=[0.85, 0.85, 0.85],
        lineWidth=4,
        pos=(0, 0),
    )

    max_seen_vol = 0
    score_by_exp_tr: dict[int, float] = {}
    start_vol = None
    tr_count_start_vol: Optional[int] = None
    first_trigger_timestamp: Optional[float] = None
    acquisition_speed_path = trial_scores_path.parent / "acquisition_speed_rt.csv"

    def _exp_tr_for_volume(vol_idx: int) -> Optional[int]:
        if tr_count_start_vol is None:
            return None
        return max(0, vol_idx - tr_count_start_vol)

    def drain_queue() -> None:
        nonlocal max_seen_vol
        nonlocal tr_count_start_vol
        try:
            while True:
                message = score_queue.get_nowait()
                vol_idx = int(message.get("volume_idx", 0) or 0)
                if vol_idx > 0:
                    max_seen_vol = max(max_seen_vol, vol_idx)
                if first_trigger_timestamp is not None and vol_idx > 0:
                    estimated_trigger_timestamp = first_trigger_timestamp + ((vol_idx - 1) * 1.4)
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
                if start_vol is None:
                    continue
                if vol_idx <= start_vol:
                    continue
                if message.get("reg_ready", True) and tr_count_start_vol is None:
                    # Do not start the experiment/TR clock until regression/background
                    # warm-up scans are complete.
                    tr_count_start_vol = vol_idx - 1

                exp_tr = _exp_tr_for_volume(vol_idx)
                if exp_tr is None:
                    continue

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
            first_trigger_timestamp = time.time()
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
            current_exp_tr = _exp_tr_for_volume(max_seen_vol)
            if current_exp_tr is None:
                current_exp_tr = 0

            win.color = [-0.004,-0.004,-0.004]  # gray background
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
                    radius = 0.0
                else:
                    rating = float(np.clip(t_score, 0.0, 100.0))
                    radius = max_feedback_radius * (rating / 100.0)
                feedback_circle.radius = radius
                feedback_circle.fillColor = [-0.5, 0.7, -0.5]
                feedback_circle.lineColor = [-0.5, 0.7, -0.5]
                # Draw both circles on every feedback frame. Draw the filled
                # feedback first, then the outline reference on top so both are
                # visible even when the score reaches the maximum radius.
                feedback_circle.draw()
                max_reference_circle.draw()

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
        _plot_qc(run_dir)


if __name__ == "__main__":
    main()
