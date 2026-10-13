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
from scipy.stats import norm

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

# ---- block-design game constants (safe to tune) ---------------------------
# A run is: rest lead, then N_ROUNDS x (ACTIVE_S active + REST_S rest).
ACTIVE_S = 45.0
REST_S = 15.0                # line still shown and scored; robots still; meter at 50
N_ROUNDS_DEFAULT = 6
RESULT_S = 3.0               # "ROUND WON!/LOST" shown at the start of each rest, then "REST"

# Meter: restarts at METER_START every round. Every second it moves by
# METER_GAIN x (displayed - 50), clipped to 0..100 -- up while the line is
# above the midline, down while it is below. It sets YOUR robot's size and
# the strength of every hit it lands.
METER_START = 50.0
METER_GAIN = 0.10

# Fight: both robots hit every ATTACK_INTERVAL s for the whole active period.
# Your hit = the meter at that moment; the enemy's hit = ENEMY_STRENGTH[level].
# Each hit is multiplied by U(1-DMG_JITTER, 1+DMG_JITTER). Whoever deals more
# total damage over the active period wins the round.
ATTACK_INTERVAL = 1.2
DMG_JITTER = 0.30

# Enemy level = staircase in blocks of up to 3 rounds ("best of 3"). Within a
# block, the first side to reach LEVEL_STREAK decides it: LEVEL_STREAK round
# wins -> level up, LEVEL_STREAK round losses -> level down, then a new block
# starts. WW, WLW, LWW = up; LL, LWL, WLL = down. So every block ends with a
# level change, after 2 or 3 rounds. The level and the running block tally
# carry over between runs of the same day (robot_level_state.json in the day
# folder); each new day starts at level 1.
# (Checked by simulation against the old 2-in-a-row rule: same settled levels
# and accuracy, ~2.3 instead of ~2.0 level changes per run.)
N_LEVELS = 10
LEVEL_STREAK = 2
ENEMY_STRENGTH = {1: 38.7, 2: 44.4, 3: 49.9, 4: 55.7, 5: 61.3,
                  6: 67.0, 7: 72.1, 8: 76.7, 9: 80.5, 10: 83.1}
# Calibration, by simulation with realistic noise: t(5) innovations matched to
# 4 subjects' measured display SD 28 / 3.6% pinned and z MAD-SD 1.375, lag-1
# autocorrelation 0.35 (ASSUMED -- run measure_autocorr.py), arcsine scale 1.0,
# no smoothing, 8 runs x 6 rounds per day.
#   Each level L is set so that a participant whose mean displayed score is
#   45 + 2.5*(L-1) wins exactly 50% of rounds against it. The level a
#   participant settles at therefore tracks their mean: 50 -> 3, 52.5 -> 4,
#   57.5 -> 6, 67.5 -> 10. Averaged over the last 4 runs of a day, a
#   participant at chance ends at 3.01 (L2/L3/L4 = 22/55/22%).
#   Re-simulate if METER_GAIN, DMG_JITTER, ATTACK_INTERVAL, the mapping or the
#   smoothing change -- the strengths are only valid for these settings.

# trace geometry:
WINDOW_SEC = 14.0           # seconds of signal across the screen
N_TRACE = 240
SAMPLE_DT = WINDOW_SEC / N_TRACE


def _map_z_to_display(z, mapping, z_center=50.0, z_gain=10.0, scale=1.0):
    """
    Map a (possibly smoothed) z-score onto the 0-100 line.

    arcsine (default)
        arcsin(sqrt(norm.cdf(z/scale))) * 2/pi * 100.
        Variance-stabilizing transform of the percentile. Compared with a
        plain cdf it gives up some steepness at the centre in exchange for
        far less pinning at 0 and 100. Measured over 4 subjects (13k volumes,
        per TR, no smoothing): SD 28, 3.6% pinned, 25 display units per z at
        the centre, against the cdf's SD 34, 13.1% pinned, 39 units per z.
        `scale` divides z first; 1.0 is plain arcsine, larger values shrink
        the spread further (1.2 roughly halves the pinning).

    cdf
        norm.cdf(z/scale) * 100 -- the displayed number is a percentile
        against the resting-state reference. Same mapping as
        rt_nf_events_parallel.py, so event and continuous scores are directly
        comparable. Note the event task feeds it a 4-TR mean, which is why it
        does not pin there; per TR it pins ~13% of the time at the measured
        z spread of ~1.4, so use scale ~2 if you pick this.
    """
    zz = float(z) / max(1e-6, float(scale))
    if mapping == "cdf":
        return max(0.0, min(100.0, float(norm.cdf(zz)) * 100.0))
    p = min(1.0, max(0.0, float(norm.cdf(zz))))
    return max(0.0, min(100.0, float(np.arcsin(np.sqrt(p))) * (2.0 / np.pi) * 100.0))


def robot_scale(power, base):
    """Robot size from fighting power on the 0-100 scale. Both robots use the
    same rule, so equal power means equal size: your robot looking bigger than
    the enemy means you are currently out-hitting it."""
    return base * (0.55 + 0.9 * max(0.0, min(100.0, float(power))) / 100.0)


def _wav_with_channels(src, n_ch, out_dir):
    """Write a copy of `src` with exactly `n_ch` channels, return its path.

    Mono -> stereo duplicates the channel; stereo -> mono averages them.
    Uses the stdlib `wave` module, so it only handles PCM wav (what
    make_jingle.py writes). Returns None if the file cannot be converted.
    """
    import wave
    try:
        with wave.open(str(src), "rb") as w:
            ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(n)
        if ch == n_ch:
            return Path(src)
        dt = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sw)
        if dt is None:
            return None
        a = np.frombuffer(raw, dtype=dt).reshape(-1, ch).astype(np.float64)
        if n_ch == 1:
            out = a.mean(axis=1, keepdims=True)
        elif n_ch <= ch:
            out = a[:, :n_ch]
        else:
            # pad up: repeat the existing channels so every output channel has
            # audio and the frame count (and so the duration) is unchanged
            idx = [i % ch for i in range(n_ch)]
            out = a[:, idx]
        out = out.astype(dt)
        dest = Path(out_dir) / (Path(src).stem + "_%dch.wav" % n_ch)
        with wave.open(str(dest), "wb") as w:
            w.setnchannels(n_ch); w.setsampwidth(sw); w.setframerate(sr)
            w.writeframes(out.ravel().tobytes())
        return dest
    except Exception as exc:
        log.warning("[AUDIO] Could not convert %s to %d channel(s): %s", src, n_ch, exc)
        return None


def _probe_output_channels_ptb():
    """Output channel counts reported by PsychToolbox, if it is installed."""
    from psychtoolbox import audio
    return [int(d["NrOutputChannels"]) for d in audio.get_devices()
            if int(d.get("NrOutputChannels", 0)) > 0]


def _probe_output_channels_sd():
    """Output channel counts reported by sounddevice, if it is installed."""
    import sounddevice as sd
    out = []
    try:
        out.append(int(sd.query_devices(kind="output")["max_output_channels"]))
    except Exception:
        pass
    for d in sd.query_devices():
        if int(d.get("max_output_channels", 0)) > 0:
            out.append(int(d["max_output_channels"]))
    return out


class _AplaySound:
    """A Sound-like object (.play() / .stop()) that pipes a wav straight to
    ALSA with `aplay`, bypassing PsychoPy, PortAudio and PsychToolbox.

    Needed on machines where PortAudio cannot open the card the headphones are
    actually on -- on the SIN RT PC it only enumerates the NVIDIA HDMI outputs
    and the USB CODEC, never the motherboard analog jack, so PsychoPy cannot
    reach the headphone amp even though `aplay -D plughw:0,0` plays fine.

    The jingle is background music, not a timed stimulus, so the extra ~20-50 ms
    of process-start latency does not matter. Volume is baked into the samples
    because aplay has no volume control of its own.
    """

    def __init__(self, cmd, device):
        self.cmd = cmd
        self.device = device
        self.proc = None
        self.pgid = None

    def play(self):
        import os
        import subprocess
        self.stop()
        # a shell loop so the music continues even if a round overruns the
        # pre-built file; start_new_session puts the shell and every aplay it
        # spawns in their own process group, so stop() can kill all of them
        self.proc = subprocess.Popen(
            ["/bin/sh", "-c", self.cmd],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            self.pgid = os.getpgid(self.proc.pid)
        except Exception:
            self.pgid = None

    def stop(self):
        import os
        import signal
        p, pgid = self.proc, self.pgid
        self.proc = self.pgid = None
        if p is None:
            return
        # The shell may be between two aplay calls, so killing it is not enough
        # -- and if it survives the TERM it starts another one. Signal the whole
        # group, then make sure nothing is left.
        def killgrp(sig):
            try:
                if pgid is not None:
                    os.killpg(pgid, sig)
                else:
                    p.send_signal(sig)
                return True
            except Exception:
                return False

        killgrp(signal.SIGTERM)
        try:
            p.wait(timeout=1.0)
        except Exception:
            pass
        for _ in range(10):                      # up to ~0.5 s
            if pgid is None or not killgrp(0):   # signal 0 = does the group exist
                break
            time.sleep(0.05)
        else:
            killgrp(signal.SIGKILL)
            try:
                p.wait(timeout=1.0)
            except Exception:
                pass

    # PsychoPy's Sound has these; the task never calls them, but be safe
    def setVolume(self, *a, **k):
        pass

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass


def _aplay_jingle(jingle_path, volume, out_dir, device, active_s=None):
    """Build the looped, volume-scaled wav and return (_AplaySound, how)."""
    import shlex
    import shutil
    import subprocess
    import tempfile
    import wave

    if shutil.which("aplay") is None:
        raise RuntimeError("aplay not found on this machine (install alsa-utils)")
    dev = device or "default"

    with wave.open(str(jingle_path), "rb") as w:
        ch, sw, sr, n = (w.getnchannels(), w.getsampwidth(),
                         w.getframerate(), w.getnframes())
        raw = w.readframes(n)
    if sw != 2:
        raise RuntimeError("aplay backend needs a 16-bit wav, got %d-bit" % (sw * 8))
    a = np.frombuffer(raw, dtype=np.int16).reshape(-1, ch)

    # enough repeats to cover one active period with slack, so a round plays
    # through without the seam of a restart
    want_s = float(active_s or ACTIVE_S) + 15.0
    reps = max(1, int(np.ceil(want_s / (n / float(sr)))))
    v = float(np.clip(volume, 0.0, 1.0))
    scaled = np.clip(a.astype(np.float64) * v, -32768, 32767).astype(np.int16)
    # Keep the big looped file out of the run folder (it is ~12 MB and is not
    # data), and reuse it between runs instead of rebuilding it every time.
    cache = Path(tempfile.gettempdir()) / (
        "robot_%s_v%03d_x%d.wav" % (Path(jingle_path).stem, int(round(v * 100)), reps))
    if not (cache.is_file() and cache.stat().st_size > 44):
        out = np.tile(scaled, (reps, 1))
        tmp = cache.with_suffix(".tmp.wav")
        with wave.open(str(tmp), "wb") as w:
            w.setnchannels(ch); w.setsampwidth(sw); w.setframerate(sr)
            w.writeframes(out.ravel().tobytes())
        tmp.replace(cache)
    dest = cache

    # Check the device opens now, while an error can still be shown, instead of
    # finding out mid-run. This plays 0.1 s of SILENCE: `--dump-hw-params` is
    # not usable for this, because on some alsa-utils versions it dumps the
    # parameters and then goes on to play the whole file.
    sil = Path(tempfile.gettempdir()) / ("robot_silence_%dhz_%dch.wav" % (sr, ch))
    if not sil.is_file():
        with wave.open(str(sil), "wb") as w:
            w.setnchannels(ch); w.setsampwidth(sw); w.setframerate(sr)
            w.writeframes(np.zeros((int(sr * 0.1), ch), dtype=np.int16).ravel().tobytes())
    try:
        probe = subprocess.run(["aplay", "-q", "-D", dev, str(sil)],
                               capture_output=True, text=True, timeout=8)
        rc, err = probe.returncode, (probe.stderr or "").strip().splitlines()
    except subprocess.TimeoutExpired:
        raise RuntimeError("aplay did not return within 8 s on %s -- the device "
                           "is probably blocked by another program" % dev)
    if rc != 0:
        raise RuntimeError("aplay could not open %s: %s"
                           % (dev, err[-1] if err else "exit code %d" % rc))

    cmd = 'while :; do aplay -q -D %s %s || exit 1; done' % (
        shlex.quote(dev), shlex.quote(str(dest)))
    snd = _AplaySound(cmd, dev)
    return snd, "aplay -D %s, %d ch, %d x %.1f s (volume baked in at %.2f)" % (
        dev, ch, reps, n / float(sr), v)


def _load_jingle(jingle_path, volume, out_dir, audio_lib=None, audio_device=None,
                 active_s=None):
    """Open the looping jingle, trying the variants that differ between audio
    devices, and return (sound_object, description) or (None, error_text).

    PsychToolbox opens the output device with a fixed number of channels and
    refuses a file that does not match ("Number of columns of audio data matrix
    doesn't match number of output channels"). Which count it picks depends on
    the machine, so instead of guessing we try the file as it is, as mono and
    as stereo, under each requested audio backend, and keep the first that
    loads.
    """
    import importlib

    if audio_lib == "aplay":
        try:
            return _aplay_jingle(jingle_path, volume, out_dir, audio_device, active_s)
        except Exception as exc:
            log.error("[AUDIO] aplay backend failed: %s", exc)
            return None, "aplay: %s" % exc

    from psychopy import prefs

    libs = [audio_lib] if audio_lib else ["ptb", "sounddevice", "pygame"]

    # Channel counts to try: mono, stereo, then whatever the output devices
    # actually report (an HDMI or surround-capable card can want 4, 6 or 8).
    counts = [1, 2]
    for probe in (_probe_output_channels_ptb, _probe_output_channels_sd):
        try:
            for c in probe():
                if 0 < c <= 8 and c not in counts:
                    counts.append(c)
        except Exception:
            pass
    log.info("[AUDIO] Channel counts to try, in order: %s", counts)

    attempts = [(jingle_path, {}, "as-is")]
    for c in counts:
        conv = _wav_with_channels(jingle_path, c, out_dir)
        if conv is None:
            continue
        kw = {"stereo": False} if c == 1 else ({"stereo": True} if c == 2 else {})
        attempts.append((conv, kw, "%d channel%s" % (c, "" if c == 1 else "s")))
    last = "no audio backend available"

    for lib in libs:
        try:
            prefs.hardware["audioLib"] = [lib]
            if audio_device:
                prefs.hardware["audioDevice"] = audio_device
            import psychopy.sound as psound
            importlib.reload(psound)
        except Exception as exc:
            last = "%s: %s" % (lib, exc)
            log.warning("[AUDIO] backend %s unavailable: %s", lib, exc)
            continue
        for path, kwargs, how in attempts:
            try:
                snd = psound.Sound(str(path), loops=-1, volume=float(volume), **kwargs)
                return snd, "%s, %s (%s)" % (lib, how, Path(path).name)
            except Exception as exc:
                last = "%s / %s: %s" % (lib, how, exc)
                log.info("[AUDIO] %s with %s failed: %s", lib, how, exc)
    return None, last


def _load_level_state(path, start_level=None) -> dict:
    """Enemy level + current block tally carried between runs of the same day."""
    state = {"level": 1, "block_wins": 0, "block_losses": 0, "last_run": None}
    if start_level is not None:
        state["level"] = int(start_level)
        log.info("[LEVEL] --start-level %d given: ignoring any saved state.", state["level"])
    elif path is not None and Path(path).exists():
        try:
            saved = json.loads(Path(path).read_text())
            for k in ("level", "block_wins", "block_losses", "last_run"):
                if k in saved:
                    state[k] = saved[k]
            if "block_wins" not in saved and "win_streak" in saved:
                # file written by the old 2-in-a-row version: keep the level,
                # start a fresh block
                log.info("[LEVEL] Old-format state file; keeping the level, starting a new block.")
            log.info("[LEVEL] Continuing from %s: level %s (block W%s/L%s, last run %s).",
                     path, state["level"], state["block_wins"], state["block_losses"],
                     state["last_run"])
        except Exception as exc:
            log.warning("[LEVEL] Could not read %s (%s); starting at level 1.", path, exc)
    else:
        log.info("[LEVEL] No saved state for this day; starting at level 1.")
    state["level"] = max(1, min(N_LEVELS, int(state["level"])))
    state["block_wins"] = int(state.get("block_wins") or 0)
    state["block_losses"] = int(state.get("block_losses") or 0)
    return state


def _save_level_state(path, state: dict) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({**state, "updated_at": _iso_now()}, indent=2))
    tmp.replace(path)          # atomic: a crash never leaves a half-written file


ROUND_FIELDS = [
    "subject", "day", "run", "round", "active_start_s", "active_end_s",
    "level_before", "enemy_strength", "our_damage", "enemy_damage", "n_hits",
    "n_hits_enemy",
    "result", "block_wins", "block_losses", "level_after", "level_change",
    "meter_mean", "meter_end", "displayed_mean", "n_score_volumes",
]


def _append_round(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ROUND_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in ROUND_FIELDS})


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


def _append_robot_timeseries(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "session_time_s", "volume_idx", "phase", "round", "score_raw", "score_z",
            "reg_ready", "displayed", "meter", "enemy_level", "our_damage",
            "enemy_damage", "jingle_on",
        ])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _append_trial_score(csv_path: Path, volume_idx: int, trial_score, score_raw,
                        score_z, phase: str = "active", score_z_smoothed=None,
                        round_idx=None) -> None:
    """One row per volume. During the active phase, `trial_score` is the on-screen
    feedback value (0-100) the subject saw. During the post-active fixation phase,
    scores are still computed and saved (for analysis) but NOT shown, so
    `trial_score` is left blank and `phase` marks it as 'post_active'.
    Kept readable by print_mean.py (which reads the `trial_score` column)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["volume_idx", "trial_score", "score_raw", "score_z",
                           "score_z_smoothed", "phase", "round", "timestamp"]
        )
        if not exists:
            writer.writeheader()
        writer.writerow({
            "volume_idx": volume_idx,
            "trial_score": "" if trial_score is None else f"{float(trial_score):.4f}",
            "score_raw": "" if score_raw is None else f"{float(score_raw):.4f}",
            "score_z": "" if score_z is None else f"{float(score_z):.4f}",
            "score_z_smoothed": "" if score_z_smoothed is None else f"{float(score_z_smoothed):.4f}",
            "phase": phase,
            "round": "" if round_idx is None else int(round_idx),
            "timestamp": _iso_now(),
        })


def _write_robot_summary(run_dir: Path, subject: str, day: str, run: str,
                         duration_s: float, volumes_seen: int, rounds: list,
                         level_start: int, level_end: int, wins: int, losses: int,
                         feedback_signal: str, feedback_mapping: str,
                         mapping_scale: float, z_smooth_trs: int,
                         mean_displayed) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    levels = [r["level_after"] for r in rounds]
    with open(run_dir / "robot_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "day", "run", "duration_s", "volumes_seen",
                         "rounds_completed", "level_start", "level_end",
                         "level_mean_after_each_round", "wins", "losses",
                         "mean_displayed_active", "feedback_signal",
                         "feedback_mapping", "mapping_scale", "z_smooth_trs"])
        writer.writerow([subject, day, run, f"{duration_s:.1f}", volumes_seen,
                         len(rounds), level_start, level_end,
                         f"{np.mean(levels):.3f}" if levels else "", wins, losses,
                         "" if mean_displayed is None else f"{mean_displayed:.2f}",
                         feedback_signal, feedback_mapping, mapping_scale, z_smooth_trs])


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


def _print_run_report(subject, day, run, shown_scores, rounds, level_start,
                      level_end, wins, losses, total_time) -> None:
    """End-of-run summary to the console and the log. `shown_scores` holds the
    values that were on screen during the active periods only."""
    lines = ["", "=" * 62, f" RUN COMPLETE   sub-{subject}  day {day}  run {run}", "=" * 62]
    if shown_scores:
        a = np.asarray(shown_scores, dtype=float)
        lines += [
            f"  displayed score   mean {a.mean():6.2f}   median {np.median(a):6.2f}"
            f"   sd {a.std(ddof=1) if len(a) > 1 else float('nan'):5.2f}   n {len(a)}",
            f"                    above 50: {100.0 * (a > 50).mean():5.1f}% of volumes",
        ]
    else:
        lines.append("  displayed score   (no feedback volumes were shown)")
    lines.append(f"  enemy level       start {level_start}  ->  end {level_end}")
    if rounds:
        lines.append("  rounds            " + "  ".join(
            f"R{r['round']}:{'W' if r['result'] == 'win' else 'L'}{r['level_after']}"
            for r in rounds))
        lv = [r["level_after"] for r in rounds]
        lines.append(f"  mean level        {np.mean(lv):.2f}  (after each of {len(lv)} rounds)")
    total = wins + losses
    lines += [
        f"  rounds won/lost   {wins} / {losses}"
        + (f"   ({100.0 * wins / total:.0f}% won)" if total else ""),
        f"  run time          {total_time:.1f}s",
        "=" * 62, "",
    ]
    text = "\n".join(lines)
    print(text, flush=True)
    log.info("Run report:%s", text)


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
    rest_lead_s=45.0,
    feedback_signal="z",
    feedback_mapping="arcsine",
    mapping_scale=1.0,
    z_smooth_trs=1,
    z_gain=10.0,
    z_center=50.0,
    n_rounds=N_ROUNDS_DEFAULT,
    active_s=ACTIVE_S,
    rest_s=REST_S,
    result_s=RESULT_S,
    rest_baseline_s=0.0,
    level_state_path=None,
    start_level=None,
    jingle_path=None,
    jingle_volume=0.8,
    audio_lib=None,
    audio_device=None,
) -> None:
    """Block-design robot task.

    Timeline (all times locked to the scanner trigger):
        rest_lead  fixation, `rest_lead_s`
        round r    active `active_s` (line + fight + jingle), then rest `rest_s`
                   (line still shown and scored; robots stand still; the meter
                   returns to the middle and stays there; "REST" shown)
        tail       optional extra fixation, `rest_baseline_s` (default 0)

    Within a round the meter starts at METER_START and integrates
    METER_GAIN x (displayed - 50). Both robots land exactly the same number of
    hits; your hit = meter at that moment, the enemy's = ENEMY_STRENGTH[level].
    More total damage wins the round. The enemy level is a staircase
    (best of 3: 2 wins -> up, 2 losses -> down), saved after every round to
    `level_state_path` so it carries over to the next run of the same day.
    """
    from psychopy import core, event, visual
    import math

    rng = random.Random()
    run_dir = Path(condition_scores_path).parent
    robot_ts_path = run_dir / "robot_timeseries.csv"
    trial_score_path = run_dir / "trial_scores.csv"
    rounds_path = run_dir / "robot_rounds.csv"
    day_rounds_path = (Path(level_state_path).parent / "robot_rounds_day.csv"
                       if level_state_path is not None else None)
    acquisition_speed_path = run_dir / "acquisition_speed_rt.csv"
    logged_trial_vols: set[int] = set()

    # Hits per side per round. Both sides land exactly this many, so neither
    # robot gets an extra hit from frame timing. Ours fall at k*I, the enemy's
    # at (k-0.5)*I -- offset only so the lunges alternate on screen.
    n_hits = int(math.floor((float(active_s) - 1e-6) / ATTACK_INTERVAL))

    win = _build_robot_window(visual)
    H, X, XY, (w_px, h_px) = _make_unit_scalers(win)
    RIGHT = (w_px / h_px) / 2.0
    LEFT = -RIGHT
    _mouse = event.Mouse(visible=False, win=win)

    # ---- feedback line geometry (height-fraction units, drawn in px) ----
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

    # ---- POWER meter: 0..100, starts at the middle mark every round ----
    BAR_X, BAR_W = LEFT + 0.07, 0.05
    BAR_BOT, BAR_TOP = GROUND + 0.02, GROUND + 0.24
    BAR_H = BAR_TOP - BAR_BOT
    bar_out = visual.Rect(win, width=H(BAR_W), height=H(BAR_H), pos=XY(BAR_X, (BAR_BOT + BAR_TOP) / 2),
                          fillColor=(-0.75, -0.75, -0.65), lineColor=(0.1, 0.3, 0.45), lineWidth=2)
    bar_fill = visual.Rect(win, width=H(BAR_W - 0.010), height=0.0, fillColor=(0.0, 0.75, 0.85), lineColor=None)
    bar_mid = visual.Line(win, start=XY(BAR_X - BAR_W / 2 - 0.010, BAR_BOT + BAR_H / 2),
                          end=XY(BAR_X + BAR_W / 2 + 0.010, BAR_BOT + BAR_H / 2),
                          lineColor=(0.9, 0.9, 0.6), lineWidth=2)
    pow_lbl = visual.TextStim(win, text="POWER", pos=XY(BAR_X, BAR_TOP + 0.03), height=H(0.022), color=(0.3, 0.6, 0.7))

    # ---- header ----
    round_lbl = visual.TextStim(win, text="", pos=XY(LEFT + 0.16, 0.485), height=H(0.030), color=(0.75, 0.75, 0.6), bold=True)
    wl_lbl = visual.TextStim(win, text="", pos=XY(RIGHT - 0.14, 0.485), height=H(0.030), color=(0.75, 0.75, 0.6), bold=True)

    # ---- robots ----
    ours = _make_robot(win, visual, (0.05, 0.55, 0.95), (-0.35, 0.05, 0.45), (0.35, 0.95, 1.0))
    opp = _make_robot(win, visual, (0.85, 0.15, 0.10), (0.35, -0.35, -0.4), (1.0, 0.6, 0.0))
    OUR_X, OPP_X = -0.14, 0.30
    BASE_SCALE = 0.16
    ground_line = visual.Line(win, start=XY(LEFT + 0.02, GROUND), end=XY(RIGHT - 0.02, GROUND), lineColor=(-0.5, -0.5, -0.4))

    # ---- battle bar: your share of the damage dealt so far this round ----
    BATTLE_Y, BATTLE_W = -0.20, 0.50
    BATTLE_X = (OUR_X + OPP_X) / 2.0
    battle_bg = visual.Rect(win, width=H(BATTLE_W), height=H(0.024), pos=XY(BATTLE_X, BATTLE_Y),
                            fillColor=(0.85, 0.30, 0.20), lineColor=(-0.3, -0.3, -0.2))
    battle_fg = visual.Rect(win, width=H(BATTLE_W / 2), height=H(0.020), pos=XY(BATTLE_X, BATTLE_Y),
                            fillColor=(0.15, 0.65, 1.0), lineColor=None)
    battle_mid = visual.Line(win, start=XY(BATTLE_X, BATTLE_Y - 0.022), end=XY(BATTLE_X, BATTLE_Y + 0.022),
                             lineColor="white", lineWidth=2)
    you_lbl = visual.TextStim(win, text="YOU", pos=XY(BATTLE_X - BATTLE_W / 2, BATTLE_Y + 0.036),
                              height=H(0.022), color=(0.4, 0.8, 1.0), alignText="left")
    opp_lbl = visual.TextStim(win, text="", pos=XY(BATTLE_X + BATTLE_W / 2 - 0.10, BATTLE_Y + 0.036),
                              height=H(0.022), color=(0.9, 0.5, 0.4))
    phase_msg = visual.TextStim(win, text="", pos=XY(0, -0.05), height=H(0.045), color=(0.9, 0.9, 0.7), bold=True)
    rest_txt = visual.TextStim(win, text="REST", pos=XY(BATTLE_X, BATTLE_Y), height=H(0.050),
                               color=(0.85, 0.85, 0.7), bold=True)
    result_txt = visual.TextStim(win, text="", pos=XY(BATTLE_X, BATTLE_Y), height=H(0.055),
                                 color=(1.0, 0.9, 0.3), bold=True)
    fixation = visual.TextStim(win, text="+", pos=(0, 0), height=H(0.10), color="white", bold=True)

    # ---- enemy level, carried over from earlier runs today ----
    state = _load_level_state(level_state_path, start_level)
    level_start = state["level"]

    # ---- jingle ----
    jingle = None
    if jingle_path:
        jingle, how = _load_jingle(jingle_path, jingle_volume, run_dir,
                                   audio_lib=audio_lib, audio_device=audio_device,
                                   active_s=active_s)
        if jingle is not None:
            log.info("[AUDIO] Jingle loaded: %s (volume %.2f) via %s",
                     jingle_path, float(jingle_volume), how)
        else:
            log.error("[AUDIO] Could not load jingle %s. Last error: %s", jingle_path, how)
            ok = _robot_show_message(
                win, visual, event, core,
                "THE JINGLE FAILED TO LOAD:\n\n%s\n\n"
                "SPACE = continue WITHOUT sound\nESCAPE = abort" % how, H)
            if not ok:
                win.close()
                return
            jingle = None
    jingle_on = False

    def jingle_start():
        nonlocal jingle_on
        if jingle is not None and not jingle_on:
            try:
                jingle.play()
                jingle_on = True
            except Exception as exc:
                log.error("[AUDIO] play failed: %s", exc)

    def jingle_stop():
        nonlocal jingle_on
        if jingle is not None and jingle_on:
            try:
                jingle.stop()
            except Exception as exc:
                log.error("[AUDIO] stop failed: %s", exc)
            jingle_on = False

    # ---- instructions ----
    proceed = _robot_show_message(
        win, visual, event, core,
        "The run starts with a cross - just rest until the robots appear.\n\n"
        "In each round, keep the LINE ABOVE THE MIDDLE to charge your POWER.\n"
        "More power = a bigger robot that hits harder.\n\n"
        "The robots fight while the music plays.\n"
        "Deal more damage than the enemy to win the round.\n\n"
        "Win 2 out of 3 rounds and the enemy gets stronger.\n"
        "Lose 2 out of 3 and it gets weaker.\n\n"
        "When the music stops the robots stop fighting - rest\n"
        "until the music comes back.\n\n"
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
    while True:
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
            break
        if "escape" in keys:
            win.close()
            return
        core.wait(0.01)

    # ---- run state ----
    clock = core.Clock()
    clock.reset()
    last_t = 0.0
    control_target = float(z_center)
    control = float(z_center)
    ys = deque([H(sy(control))] * N_TRACE, maxlen=N_TRACE)
    next_sample = 0.0
    next_log = 0.0

    seen_vols: set[int] = set()
    current_vol = 0
    reg_ready_seen = False
    last_score_raw = None
    last_score_z = None
    z_hist: deque = deque(maxlen=max(1, int(z_smooth_trs)))
    shown_scores: list[float] = []
    wins, losses = 0, 0
    rounds_done: list[dict] = []

    # ---- per-round state ----
    meter = METER_START
    our_dmg = opp_dmg = 0.0
    our_hits = opp_hits = 0
    meter_area = meter_time = 0.0
    round_disp: list[float] = []
    level_before = state["level"]
    our_lunge = opp_lunge = 0.0

    def start_round(t0):
        # The line is NOT reset: it runs continuously through the rests.
        nonlocal meter, our_dmg, opp_dmg, our_hits, opp_hits, meter_area, meter_time
        nonlocal round_disp, level_before
        meter = METER_START
        our_dmg = opp_dmg = 0.0
        our_hits = opp_hits = 0
        meter_area = meter_time = 0.0
        round_disp = []
        level_before = state["level"]
        jingle_start()
        log.info("[ROUND %d/%d] start at %.2fs, enemy level %d",
                 round_idx, n_rounds, t0, state["level"])

    def land_hits(until, round_start):
        """Land every hit scheduled up to `until` (both sides capped at n_hits)."""
        nonlocal our_dmg, opp_dmg, our_hits, opp_hits, our_lunge, opp_lunge
        while our_hits < n_hits and round_start + (our_hits + 1) * ATTACK_INTERVAL <= until:
            our_dmg += meter * rng.uniform(1 - DMG_JITTER, 1 + DMG_JITTER)
            our_hits += 1
            our_lunge = 0.15
        while opp_hits < n_hits and round_start + (opp_hits + 0.5) * ATTACK_INTERVAL <= until:
            opp_dmg += ENEMY_STRENGTH[state["level"]] * rng.uniform(1 - DMG_JITTER, 1 + DMG_JITTER)
            opp_hits += 1
            opp_lunge = 0.15

    def end_round(t0, t1):
        nonlocal wins, losses
        land_hits(float("inf"), t0)               # land anything still due
        won = our_dmg > opp_dmg
        if won:
            wins += 1
            state["block_wins"] += 1
        else:
            losses += 1
            state["block_losses"] += 1
        change = 0
        block_done = False
        if state["block_wins"] >= LEVEL_STREAK:          # 2 of up to 3 won
            block_done = True
            if state["level"] < N_LEVELS:
                state["level"] += 1
                change = 1
        elif state["block_losses"] >= LEVEL_STREAK:      # 2 of up to 3 lost
            block_done = True
            if state["level"] > 1:
                state["level"] -= 1
                change = -1
        # log the tally that decided this round, then start a new block
        tally_w, tally_l = state["block_wins"], state["block_losses"]
        if block_done:
            state["block_wins"] = state["block_losses"] = 0
        state["last_run"] = str(run)
        _save_level_state(level_state_path, state)

        row = {
            "subject": subject, "day": day, "run": run, "round": round_idx,
            "active_start_s": f"{t0:.2f}", "active_end_s": f"{t1:.2f}",
            "level_before": level_before,
            "enemy_strength": ENEMY_STRENGTH[level_before],
            "our_damage": f"{our_dmg:.1f}", "enemy_damage": f"{opp_dmg:.1f}",
            "n_hits": our_hits, "n_hits_enemy": opp_hits,
            "result": "win" if won else "loss",
            "block_wins": tally_w, "block_losses": tally_l,
            "level_after": state["level"], "level_change": change,
            "meter_mean": f"{(meter_area / meter_time) if meter_time > 0 else meter:.2f}",
            "meter_end": f"{meter:.2f}",
            "displayed_mean": f"{np.mean(round_disp):.2f}" if round_disp else "",
            "n_score_volumes": len(round_disp),
        }
        _append_round(rounds_path, row)
        if day_rounds_path is not None:
            _append_round(day_rounds_path, row)
        rounds_done.append({**row, "level_after": state["level"]})

        result_txt.text = "ROUND WON!" if won else "ROUND LOST"
        result_txt.color = (0.4, 1.0, 0.5) if won else (1.0, 0.55, 0.4)
        jingle_stop()
        log.info("[ROUND %d/%d] %s  our %.0f vs enemy %.0f  (%d hits each)  level %d -> %d",
                 round_idx, n_rounds, "WIN" if won else "LOSS", our_dmg, opp_dmg,
                 our_hits, level_before, state["level"])

    # ---- phase machine (times are scheduled, not taken from the frame, so the
    #      rounds stay locked to the trigger and never drift) ----
    phase = "rest_lead" if float(rest_lead_s) > 0 else "active"
    round_idx = 0 if phase == "rest_lead" else 1
    phase_start = 0.0
    if phase == "active":
        start_round(phase_start)

    safety_s = None if duration_s is None else (float(duration_s) + 180.0)
    idle_timeout_s = 30.0
    last_new_vol_time = None
    aborted = False

    while True:
        now = clock.getTime()
        dt = max(0.0, min(now - last_t, 0.5))
        last_t = now

        # ---- phase transitions ----
        if phase == "rest_lead" and now - phase_start >= float(rest_lead_s):
            phase_start += float(rest_lead_s)
            phase, round_idx = "active", 1
            log.info("[PHASE] rest lead done -> round 1")
            next_sample = phase_start
            start_round(phase_start)
        elif phase == "active" and now - phase_start >= float(active_s):
            end_round(phase_start, phase_start + float(active_s))
            meter = METER_START          # back to the middle, frozen through the rest
            phase_start += float(active_s)
            phase = "rest"
        elif phase == "rest" and now - phase_start >= float(rest_s):
            phase_start += float(rest_s)
            if round_idx < int(n_rounds):
                round_idx += 1
                phase = "active"
                start_round(phase_start)
            elif float(rest_baseline_s) > 0:
                phase = "tail"
            else:
                break
        elif phase == "tail" and now - phase_start >= float(rest_baseline_s):
            break

        over_time = (safety_s is not None) and (now >= safety_s)
        idle_stop = (last_new_vol_time is not None) and ((now - last_new_vol_time) >= idle_timeout_s)
        if idle_stop:
            log.warning("No new volume for %.0fs (last vol %d). Ending run.", idle_timeout_s, current_vol)
        if over_time or idle_stop or ("escape" in event.getKeys()):
            aborted = True
            if phase == "active":
                log.warning("[ROUND %d] interrupted -- not counted, level not changed.", round_idx)
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
                    current_vol = max(current_vol, vol_idx)
                if first_trigger_timestamp is not None and vol_idx > 0:
                    est_trig = first_trigger_timestamp + ((vol_idx - 1) * float(tr))
                    wd_ts = message.get("watchdog_timestamp")
                    an_ts = message.get("analysis_timestamp", message.get("timestamp"))
                    if wd_ts is not None and an_ts is not None:
                        _append_acquisition_speed(acquisition_speed_path, {
                            "volume_idx": vol_idx,
                            "estimated_trigger_timestamp": f"{est_trig:.6f}",
                            "watchdog_timestamp": f"{float(wd_ts):.6f}",
                            "analysis_timestamp": f"{float(an_ts):.6f}",
                            "trigger_to_watchdog_s": f"{(float(wd_ts) - est_trig):.6f}",
                            "watchdog_to_analysis_s": f"{(float(an_ts) - float(wd_ts)):.6f}",
                        })
                if "score_raw" in message and message.get("reg_ready", True):
                    reg_ready_seen = True
                    sr = float(message["score_raw"])
                    sz = message.get("score_z")
                    if phase in ("active", "rest"):
                        last_score_raw, last_score_z = sr, sz
                        if feedback_signal == "z":
                            if sz is None:
                                raise RuntimeError(
                                    "feedback_signal='z' but the pipeline emitted no score_z. "
                                    "Check that --rs names a run with a usable scores.csv. "
                                    "Refusing to silently fall back to raw decoder values.")
                            z_hist.append(float(sz))
                            sz_used = float(np.mean(z_hist))
                            control_target = _map_z_to_display(sz_used, feedback_mapping,
                                                               z_center, z_gain, mapping_scale)
                        else:
                            sz_used = None
                            control_target = max(0.0, min(100.0, sr))
                        if vol_idx and vol_idx not in logged_trial_vols:
                            logged_trial_vols.add(vol_idx)
                            _append_condition_score(condition_scores_path, message, condition)
                            if phase == "active":          # round stats: active only
                                shown_scores.append(float(control_target))
                                round_disp.append(float(control_target))
                            _append_trial_score(trial_score_path, vol_idx, control_target, sr, sz,
                                                phase=phase, score_z_smoothed=sz_used,
                                                round_idx=round_idx)
                    else:
                        # rest lead / tail: scored and saved for analysis, not displayed
                        if vol_idx and vol_idx not in logged_trial_vols:
                            logged_trial_vols.add(vol_idx)
                            _append_condition_score(condition_scores_path, message, condition)
                            _append_trial_score(trial_score_path, vol_idx, None, sr, sz,
                                                phase=phase,
                                                round_idx=round_idx if phase == "rest" else None)
        except Empty:
            pass

        control = control_target if reg_ready_seen else float(z_center)
        disp = control

        # ---- line scrolls during active AND rest ----
        if phase in ("active", "rest"):
            while now >= next_sample:
                ys.append(H(sy(disp)))
                next_sample += SAMPLE_DT
        # ---- fight: active only ----
        if phase == "active":
            if reg_ready_seen:
                meter = max(0.0, min(100.0, meter + METER_GAIN * (disp - 50.0) * dt))
            meter_area += meter * dt
            meter_time += dt
            land_hits(now, phase_start)
            our_lunge = max(0.0, our_lunge - dt)
            opp_lunge = max(0.0, opp_lunge - dt)

        # ---- draw ----
        if phase in ("rest_lead", "tail"):
            fixation.draw()
        else:
            frame_top.draw(); frame_bot.draw(); mid_line.draw()
            trace.vertices = np.column_stack([xs, np.array(ys)])
            trace.draw()
            ground_line.draw()

            bar_out.draw()
            fh = (BAR_H - 0.010) * (meter / 100.0)
            bar_fill.height = max(0.0, H(fh))
            bar_fill.pos = XY(BAR_X, BAR_BOT + 0.005 + fh / 2.0)
            bar_fill.draw()
            bar_mid.draw()
            pow_lbl.draw()

            round_lbl.text = "ROUND %d / %d" % (round_idx, int(n_rounds))
            round_lbl.draw()
            wl_lbl.text = "W %d   L %d" % (wins, losses)
            wl_lbl.draw()

            ox = 0.035 * (our_lunge / 0.15)
            px = -0.035 * (opp_lunge / 0.15)
            bob = 0.003 * math.sin(now * 4.0)
            _place_robot(ours, X(OUR_X + ox), H(GROUND + bob), H(robot_scale(meter, BASE_SCALE)), flip=1)
            _place_robot(opp, X(OPP_X + px), H(GROUND - bob),
                         H(robot_scale(ENEMY_STRENGTH[state["level"]], BASE_SCALE)), flip=-1)
            for part in ours:
                part.draw()
            for part in opp:
                part.draw()

            if phase == "active":
                total = our_dmg + opp_dmg
                share = 0.5 if total <= 0 else our_dmg / total
                battle_bg.draw()
                battle_fg.width = max(0.001, H(BATTLE_W * share))
                battle_fg.pos = XY(BATTLE_X - BATTLE_W / 2.0 + (BATTLE_W * share) / 2.0, BATTLE_Y)
                battle_fg.draw()
                battle_mid.draw()
                you_lbl.draw()
            elif now - phase_start < float(result_s):
                result_txt.draw()                  # win/lose, in place of the battle bar
            else:
                rest_txt.draw()
            opp_lbl.text = "ENEMY  LV %d" % state["level"]
            opp_lbl.draw()

            if not reg_ready_seen:
                phase_msg.text = "Waiting for scanner..."
                phase_msg.draw()
        win.flip()

        # ---- timeseries log (~10 Hz) ----
        if now >= next_log:
            _append_robot_timeseries(robot_ts_path, {
                "session_time_s": f"{now:.2f}",
                "volume_idx": current_vol,
                "phase": phase,
                "round": round_idx,
                "score_raw": "" if last_score_raw is None else f"{last_score_raw:.4f}",
                "score_z": "" if last_score_z is None else f"{float(last_score_z):.4f}",
                "reg_ready": int(reg_ready_seen),
                "displayed": f"{disp:.2f}" if phase in ("active", "rest") else "",
                "meter": f"{meter:.2f}" if phase in ("active", "rest") else "",
                "enemy_level": state["level"],
                "our_damage": f"{our_dmg:.1f}",
                "enemy_damage": f"{opp_dmg:.1f}",
                "jingle_on": int(jingle_on),
            })
            next_log = now + 0.1

    jingle_stop()
    total_time = clock.getTime()
    if aborted:
        log.warning("Run ended early after %d of %d rounds.", len(rounds_done), int(n_rounds))
    _write_robot_summary(run_dir, subject, day, run, total_time, len(seen_vols), rounds_done,
                         level_start, state["level"], wins, losses, feedback_signal,
                         feedback_mapping, mapping_scale, z_smooth_trs,
                         float(np.mean(shown_scores)) if shown_scores else None)
    _print_run_report(subject, day, run, shown_scores, rounds_done, level_start,
                      state["level"], wins, losses, total_time)
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
        "--rounds",
        type=int,
        default=N_ROUNDS_DEFAULT,
        help="Rounds per run. Each round = --active-sec of feedback + --rest-sec "
             "of rest (default %d)." % N_ROUNDS_DEFAULT,
    )
    parser.add_argument(
        "--active-sec", type=float, default=ACTIVE_S,
        help="Active (feedback + fight + jingle) seconds per round (default %g)." % ACTIVE_S,
    )
    parser.add_argument(
        "--rest-sec", type=float, default=REST_S,
        help="Rest seconds after each round (default %g). The line is still "
             "shown and scored; the robots stop and the meter sits at the middle." % REST_S,
    )
    parser.add_argument(
        "--result-sec", type=float, default=RESULT_S,
        help="Seconds 'ROUND WON!/LOST' is shown at the start of each rest before "
             "'REST' (default %g)." % RESULT_S,
    )
    parser.add_argument(
        "--start-level", type=int, default=None,
        help="Force the enemy level at the start of this run (1-%d), ignoring the "
             "level saved from earlier runs today. Omit to continue from the saved "
             "level (or level 1 on the first run of the day)." % N_LEVELS,
    )
    parser.add_argument(
        "--jingle", type=str, default=None,
        help="Audio file (wav/ogg/mp3) looped during every active period and "
             "stopped during rest. Omit for no sound.",
    )
    parser.add_argument(
        "--audio-lib", choices=["ptb", "sounddevice", "pygame", "aplay"], default=None,
        help="Force one audio backend for the jingle. Default: try ptb, then "
             "sounddevice, then pygame, and keep the first that works. "
             "'aplay' skips PsychoPy and sends the wav straight to ALSA -- use "
             "it when PortAudio cannot open the card the headphones are on.",
    )
    parser.add_argument(
        "--audio-device", type=str, default=None,
        help="Force a specific output device. For the PsychoPy backends, the "
             "device NAME as listed by list_audio_devices.py. For --audio-lib "
             "aplay, the ALSA device string, e.g. plughw:0,0 (see aplay -l). "
             "Default: the system default output.",
    )
    parser.add_argument(
        "--jingle-volume", type=float, default=0.8,
        help="Jingle volume, 0-1 (default 0.8). Check it against scanner noise.",
    )
    parser.add_argument(
        "--active-feedback-trs",
        type=int,
        default=None,
        help="IGNORED in the block design -- run length is set by --rounds, "
             "--active-sec and --rest-sec. Kept so old commands still run.",
    )
    parser.add_argument(
        "--rest-baseline-sec",
        type=float,
        default=0.0,
        help="Extra fixation at the very end of the run, after the last round's "
             "rest (default 0: the last round's rest ends the run).",
    )
    parser.add_argument(
        "--rest-lead-sec",
        type=float,
        default=45.0,
        help=(
            "Seconds of fixation-cross rest at the START of the run, before any "
            "feedback. This is the run's rest baseline: the pipeline builds its "
            "voxel-normalization reference from the first "
            "skip_first_trs + voxel_norm_ref_volumes volumes, so those must be "
            "acquired at rest, not while the subject is already regulating. "
            "Values below that warmup are raised automatically. 0 disables."
        ),
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
        help="Line centre (0-100) for the pre-feedback baseline and the "
             "post-run rest display. Not used by the arcsine or cdf mappings, "
             "which are both centred on 50 by construction.",
    )
    parser.add_argument(
        "--feedback-mapping",
        choices=["arcsine", "cdf"],
        default="arcsine",
        help="How z becomes the 0-100 line. arcsine (default): "
             "arcsin(sqrt(cdf(z)))*2/pi*100 -- less pinning at the ends than a "
             "plain cdf while keeping good resolution near the middle. "
             "cdf: norm.cdf(z)*100, the percentile against resting state; pins "
             "~13%% of volumes per TR at the measured z spread, so pair it with "
             "--mapping-scale 2.",
    )
    parser.add_argument(
        "--mapping-scale",
        type=float,
        default=1.0,
        help="Divide z by this before mapping. 1.0 = off. Raising it narrows "
             "the displayed spread and reduces pinning at 0/100 (1.2 roughly "
             "halves it for arcsine; use ~2 with --feedback-mapping cdf).",
    )
    parser.add_argument(
        "--z-smooth-trs",
        type=int,
        default=1,
        help="Average the last N z values before mapping (1 = off, the default). "
             "Smoothing narrows the displayed distribution and makes the power "
             "meter far better at separating regulation from noise, but changes "
             "what the line shows -- pilot both before committing.",
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
    if args.active_feedback_trs is not None:
        log.warning("--active-feedback-trs is ignored in the block design; run length "
                    "comes from --rounds (%d) x (--active-sec %g + --rest-sec %g).",
                    args.rounds, args.active_sec, args.rest_sec)
    if not (1 <= args.rounds <= 50):
        parser.error("--rounds must be between 1 and 50")
    if args.start_level is not None and not (1 <= args.start_level <= N_LEVELS):
        parser.error("--start-level must be between 1 and %d" % N_LEVELS)
    if args.jingle is not None and not Path(args.jingle).is_file():
        parser.error("--jingle file not found: %s" % args.jingle)
    if not (0.0 <= args.jingle_volume <= 1.0):
        parser.error("--jingle-volume must be between 0 and 1")
    # The phase machine ends the run; max_trs is only a hard safety ceiling.
    tr_val = float(REGRESSOR_SETTINGS.TR)
    planned_s = (float(args.rest_lead_sec)
                 + args.rounds * (float(args.active_sec) + float(args.rest_sec))
                 + float(args.rest_baseline_sec))
    max_trs = args.max_trs
    if max_trs is None:
        max_trs = int(round(planned_s / tr_val)) + 60
    log.info("[RUN] planned length %.0fs = %.0fs rest lead + %d x (%.0f active + %.0f rest)%s "
             "-> acquire at least %d volumes (safety ceiling %d).",
             planned_s, float(args.rest_lead_sec), args.rounds, float(args.active_sec),
             float(args.rest_sec),
             (" + %.0fs tail" % float(args.rest_baseline_sec)) if args.rest_baseline_sec else "",
             int(np.ceil(planned_s / tr_val)), max_trs)
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

    # The rest lead must cover the pipeline warmup, otherwise the voxel-norm
    # reference is built from volumes acquired during regulation. Mirrors the
    # voxel_norm_ref_volumes = baseline_trs lock in rt_nf_events_parallel.py.
    rest_lead_s = float(args.rest_lead_sec)
    if rest_lead_s > 0:
        tr_s = float(REGRESSOR_SETTINGS.TR)
        warmup_s = (int(REGRESSOR_SETTINGS.skip_first_trs)
                    + int(REGRESSOR_SETTINGS.voxel_norm_ref_volumes)) * tr_s
        if rest_lead_s < warmup_s:
            log.warning(
                "--rest-lead-sec %.1fs is shorter than the pipeline warmup "
                "(skip_first_trs %d + voxel_norm_ref_volumes %d = %.1fs at TR=%.2f). "
                "Raising it to %.1fs so the normalization reference is collected at rest.",
                rest_lead_s, int(REGRESSOR_SETTINGS.skip_first_trs),
                int(REGRESSOR_SETTINGS.voxel_norm_ref_volumes), warmup_s, tr_s, warmup_s)
            rest_lead_s = warmup_s
        log.info("[PHASE] Rest lead %.1fs (%d TRs) before feedback.",
                 rest_lead_s, int(round(rest_lead_s / tr_s)))

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
                "rest_lead_s": rest_lead_s,
                "feedback_mapping": args.feedback_mapping,
                "mapping_scale": args.mapping_scale,
                "z_smooth_trs": args.z_smooth_trs,
                "rest_baseline_sec": args.rest_baseline_sec,
                "design": "block",
                "rounds": args.rounds,
                "active_sec": args.active_sec,
                "rest_sec": args.rest_sec,
                "meter_start": METER_START,
                "meter_gain": METER_GAIN,
                "attack_interval": ATTACK_INTERVAL,
                "dmg_jitter": DMG_JITTER,
                "n_levels": N_LEVELS,
                "level_rule": "best of 3 (2 wins up, 2 losses down)",
                "result_sec": args.result_sec,
                "enemy_strength": ENEMY_STRENGTH,
                "start_level_arg": args.start_level,
                "jingle": args.jingle,
                "jingle_volume": args.jingle_volume,
                "audio_lib": args.audio_lib,
                "audio_device": args.audio_device,
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
        parser.error(
            "--feedback-signal=z requires --rs <RS run>. Without it the pipeline emits "
            "no score_z and the display has nothing valid to show."
        )
    log.info("[DISPLAY] Robot line: signal=%s mapping=%s scale=%.2f smooth=%d TR.",
             args.feedback_signal, args.feedback_mapping, args.mapping_scale,
             args.z_smooth_trs)
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
            rest_lead_s=rest_lead_s,
            feedback_signal=args.feedback_signal,
            feedback_mapping=args.feedback_mapping,
            mapping_scale=args.mapping_scale,
            z_smooth_trs=args.z_smooth_trs,
            z_gain=args.z_gain,
            z_center=args.z_center,
            n_rounds=args.rounds,
            active_s=args.active_sec,
            rest_s=args.rest_sec,
            result_s=args.result_sec,
            rest_baseline_s=args.rest_baseline_sec,
            level_state_path=subject_root / str(args.day) / "robot_level_state.json",
            start_level=args.start_level,
            jingle_path=args.jingle,
            jingle_volume=args.jingle_volume,
            audio_lib=args.audio_lib,
            audio_device=args.audio_device,
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