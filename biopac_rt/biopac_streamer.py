import argparse
import csv
import json
import logging
import multiprocessing as mp
import queue as queue_mod
import socket
import time
from collections import deque
from pathlib import Path
"""
BIOPAC RetroTS streamer.

USAGE OVERVIEW
==============
1) Run on the BIOPAC PC.
   - Connects to BIOPAC MP device (or sim/CSV mode).
   - Computes RetroTS regressors per TR.
   - Streams JSON lines to the RT PC over TCP.

2) Synchronization options
   - Preferred: provide a trigger channel that receives scanner TTL/TR pulses.
   - The streamer detects rising edges on the trigger channel and estimates TR
     from inter-trigger intervals. Regressors are emitted on each trigger.
   - If no trigger channel is supplied, the streamer emits regressors on a
     fixed TR schedule (using --tr as the period). For better alignment, the
     RT PC can send a handshake message with the TR; the streamer will wait for
     that handshake before starting its fixed-TR schedule.

3) Optional logging
   - Use --log-samples-csv to save raw resp/card/trigger samples.
   - Use --log-sent-csv to save the regressors that were transmitted.
   - Use --log-regressors-csv to save computed regressors locally.
   - Use --live-plot for a small live plot window on the BIOPAC PC.

EXAMPLES
========
BIOPAC device + trigger:
  python -m biopac_rt.biopac_streamer \
    --host 115.145.189.30 --port 15000 --tr 0.9 --phys-fs 100 \
    --mode biopac --mpdev-dll "C:\\Program Files\\BIOPAC Systems, Inc\\...\\mpdev.dll" \
    --resp-channel 1 --card-channel 2 --trigger-channel 3 \
    --log-samples-csv biopac_samples.csv --log-sent-csv biopac_regressors.csv

    # to run with TR from the data
    python -m biopac_rt.biopac_streamer --host 115.145.189.30 --port 15000 --tr 0.9 --phys-fs 1000 --mode biopac --triger-channel 1 --mpdev-dll "D:\SIN_LAB RT-BIOPAC\DecNef_py\BIOPAC Hardware API 2.2.5 Research\VC10\x64\mpdev.dll" --downsample-hz 100 --live-plot
    # to run with exact TR
    python -m biopac_rt.biopac_streamer --host 115.145.189.30 --port 15000 --tr 0.9 --phys-fs 1000 --mode biopac --mpdev-dll "D:\SIN_LAB RT-BIOPAC\DecNef_py\BIOPAC Hardware API 2.2.5 Research\VC10\x64\mpdev.dll" --downsample-hz 100 --live-plot

Simulated stream (no trigger channel):
  python -m biopac_rt.biopac_streamer \
    --host 115.145.189.30 --port 15000 --tr 0.9 --phys-fs 100 --mode sim
"""

from dataclasses import dataclass
from typing import Deque, Iterator, List, Optional, Tuple

import numpy as np
from ctypes import POINTER, c_bool, c_double, c_int, c_char_p, c_uint32, cdll

from fmri_rt_preproc.RTPSpy_tools.rtp_retrots import RtpRetroTS

MP150 = 101
MP160 = 103

MPUDP = 11
MPSUCCESS = 1

log = logging.getLogger("biopac_streamer")


@dataclass
class RunInfo:
    subject: str
    day: str
    run: str


def _poll_control(
    sock: socket.socket,
    buffer: str,
    fallback_tr: float,
) -> Tuple[Optional[float], Optional[RunInfo], bool, str]:
    tr_value = None
    run_info = None
    run_end = False
    try:
        chunk = sock.recv(4096)
    except (BlockingIOError, socket.timeout):
        return tr_value, run_info, run_end, buffer
    if not chunk:
        return tr_value, run_info, run_end, buffer
    buffer += chunk.decode("utf-8")
    while "\n" in buffer:
        line, buffer = buffer.split("\n", 1)
        line = line.strip()
        if not line:
            continue
        if line == "s":
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = payload.get("kind")
        if kind == "handshake":
            try:
                tr_value = float(payload.get("tr", fallback_tr))
            except (TypeError, ValueError):
                tr_value = fallback_tr
        elif kind == "run_start":
            subject = payload.get("subject")
            day = payload.get("day")
            run = payload.get("run")
            if subject and day and run:
                run_info = RunInfo(str(subject), str(day), str(run))
        elif kind == "run_end":
            run_end = True
    return tr_value, run_info, run_end, buffer


class RunDataLogger:
    def __init__(self, base_dir: Path):
        self._base_dir = Path(base_dir)
        self._run_dir: Optional[Path] = None
        self._raw_handle = None
        self._raw_writer = None
        self._reg_handle = None
        self._reg_writer = None
        self._last_flush = 0.0

    @property
    def active(self) -> bool:
        return self._raw_handle is not None or self._reg_handle is not None

    def start_run(self, run_info: Optional[RunInfo]) -> Path:
        self.stop_run()
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        if run_info is None:
            run_dir = self._base_dir / timestamp
        else:
            run_dir = (
                self._base_dir
                / run_info.subject
                / run_info.day
                / f"run{run_info.run}_{timestamp}"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        self._run_dir = run_dir
        self._raw_handle = open(run_dir / "raw_channels.csv", "w", newline="")
        self._raw_writer = csv.writer(self._raw_handle)
        self._raw_writer.writerow(["timestamp", "resp", "card", "trigger"])
        self._reg_handle = open(run_dir / "regressors.csv", "w", newline="")
        self._reg_writer = csv.writer(self._reg_handle)
        self._reg_writer.writerow(
            [
                "timestamp",
                "volume_idx",
                "tr",
                "sample_idx",
                "nsamp_total",
                "samples_per_tr",
                "regressors",
            ]
        )
        self._last_flush = time.monotonic()
        return run_dir

    def stop_run(self):
        if self._raw_handle is not None:
            self._raw_handle.close()
        if self._reg_handle is not None:
            self._reg_handle.close()
        self._raw_handle = None
        self._raw_writer = None
        self._reg_handle = None
        self._reg_writer = None
        self._run_dir = None

    def log_sample(self, resp: float, card: float, trigger: float):
        if self._raw_writer is None:
            return
        self._raw_writer.writerow([time.time(), resp, card, trigger])
        self._maybe_flush()

    def log_regressors(self, vol_idx: int, meta: dict, regressors: List[float]):
        if self._reg_writer is None:
            return
        self._reg_writer.writerow(
            [
                time.time(),
                vol_idx,
                meta["tr"],
                meta["sample_idx"],
                meta["nsamp_total"],
                meta["samples_per_tr"],
                regressors,
            ]
        )
        self._maybe_flush()

    def _maybe_flush(self):
        now = time.monotonic()
        if now - self._last_flush >= 1.0:
            if self._raw_handle is not None:
                self._raw_handle.flush()
            if self._reg_handle is not None:
                self._reg_handle.flush()
            self._last_flush = now


@dataclass
class StreamerConfig:
    # If you want to change the default values make sure
    # you also change parser.add_argument section cause it
    # doesn't copy it directly at the moment
    host: str
    port: int
    tr: float
    phys_fs: float
    mode: str
    resp_channel: int = 12
    card_channel: int = 14
    csv_path: Optional[str] = None
    mpdev_dll: Optional[str] = None
    mp_device: int = MP150
    mp_comm: int = MPUDP
    mp_chunk_samples: int = 10
    mp_poll_sleep_ms: float = 0.0
    trigger_channel: Optional[int] = None
    trigger_threshold: float = 1.0
    trigger_min_interval: float = 0.0
    log_samples_csv: Optional[str] = None
    log_sent_csv: Optional[str] = None
    log_regressors_csv: Optional[str] = None
    wait_for_handshake: bool = False
    live_plot: bool = False
    plot_window_s: float = 0.0
    plot_update_hz: float = 10.0
    connect_retry_s: float = 2.0
    print_every: int = 1
    card_source: str = "ecg"
    downsample_hz: float = 0.0
    data_dir: str = "biopac_rt/data"
    run_control: bool = True
    use_multiprocess: bool = True
    queue_max: int = 5000
    plot_queue_max: int = 1000
    net_queue_max: int = 500
    # --- Offline reporting / behavior ---
    offline_status_every_s: float = 1.0   # print "I'm alive" line every N seconds
    handshake_grace_s: float = 2.0        # how long to wait for handshake after connect
    allow_offline_fixed_tr: bool = True   # if wait_for_handshake but no RT PC, still emit locally
    idle_stop_after_s: float = 20.0       # stop run if no regressors emitted for this long

class _IntFactorDownsampler:
    """
    Very small causal downsampler for streaming:
      - optional 1-pole lowpass (anti-alias-ish)
      - keep every Nth sample (integer factor)
    Returns (y or None) per input sample.

    We DO NOT downsample the trigger (TTL) because you can miss narrow pulses.
    """
    def __init__(self, fs_in: float, fs_out: float, alpha: float = 0.2):
        if fs_out <= 0 or fs_out >= fs_in:
            raise ValueError("fs_out must be >0 and < fs_in")
        ratio = fs_in / fs_out
        factor = int(round(ratio))
        if not np.isfinite(ratio) or factor < 1 or abs(ratio - factor) > 1e-3:
            raise ValueError(f"Downsample requires near-integer factor: fs_in/fs_out={ratio:.6f} (rounded={factor})")
        self.fs_in = float(fs_in)
        self.fs_out = float(fs_out)
        self.factor = factor
        self.alpha = float(alpha)  # 0..1 ; smaller = smoother
        self._k = 0
        self._y = 0.0
        self._init = False

    def step(self, x: float) -> Optional[float]:
        # one-pole lowpass (cheap, causal)
        if not self._init:
            self._y = float(x)
            self._init = True
        else:
            self._y = (1.0 - self.alpha) * self._y + self.alpha * float(x)

        self._k += 1
        if self._k >= self.factor:
            self._k = 0
            return float(self._y)
        return None


class _TimeBucketDownsampler:
    def __init__(self, target_hz: float, mode: str = "mean"):
        if target_hz <= 0:
            raise ValueError("target_hz must be > 0")
        if mode not in ("mean", "max"):
            raise ValueError("mode must be 'mean' or 'max'")
        self._dt = 1.0 / float(target_hz)
        self._mode = mode
        self._t0 = None
        self._bucket_idx = None
        self._sum = 0.0
        self._count = 0
        self._max = None

    def step(self, t: float, x: float) -> List[Tuple[float, float]]:
        t = float(t)
        x = float(x)
        if self._t0 is None:
            self._t0 = t
            self._bucket_idx = 0

        bucket_idx = int((t - self._t0) / self._dt)
        outputs: List[Tuple[float, float]] = []

        while self._bucket_idx is not None and bucket_idx > self._bucket_idx:
            if self._count > 0:
                value = (
                    (self._sum / self._count)
                    if self._mode == "mean"
                    else float(self._max)
                )
                t_out = self._t0 + (self._bucket_idx + 1) * self._dt
                outputs.append((t_out, value))
            self._sum = 0.0
            self._count = 0
            self._max = None
            self._bucket_idx += 1

        self._sum += x
        self._count += 1
        if self._mode == "max":
            self._max = x if self._max is None else max(self._max, x)

        return outputs


class RetroTSStreamer:
    def __init__(self, config: StreamerConfig):
        self.config = config
        self._retrots = RtpRetroTS()
        self._resp = []
        self._card = []
        self._vol_idx = 0
        self._start_time = time.monotonic()
        self._sample_idx = 0  # how many samples have been ingested
        self._proc_fs = float(self.config.downsample_hz) if (self.config.downsample_hz and self.config.downsample_hz > 0) else float(self.config.phys_fs)

        self._samples_per_tr = int(max(1, round(self._proc_fs * self.config.tr)))

        # Minimum samples needed before we attempt RetroTS.
        # One TR worth is usually the bare minimum; 2 TRs is safer.
        self._min_samples = int(max(1, round(self._proc_fs * self.config.tr * 5)))

    def reset_start_time(self):
        self._start_time = time.monotonic()

    def reset_for_run(self, tr_value: Optional[float] = None):
        if tr_value is not None:
            self.config.tr = float(tr_value)
        self._resp = []
        self._card = []
        self._vol_idx = 0
        self._sample_idx = 0
        self._start_time = time.monotonic()
        self._samples_per_tr = int(max(1, round(self._proc_fs * self.config.tr)))
        self._min_samples = int(max(1, round(self._proc_fs * self.config.tr * 5)))

    def add_sample(self, resp: float, card: float):
        self._resp.append(resp)
        self._card.append(card)
        self._sample_idx += 1


    def maybe_emit(self):
        elapsed = time.monotonic() - self._start_time
        while elapsed >= (self._vol_idx + 1) * self.config.tr:
            # Don't compute RetroTS until we have enough samples
            if len(self._resp) < self._min_samples or len(self._card) < self._min_samples:
                return  # wait for more samples

            self._vol_idx += 1
            regressors = self._compute_retrots(self._vol_idx, self.config.tr)
            meta = self._emit_meta(self.config.tr)
            yield self._vol_idx, regressors, meta
            elapsed = time.monotonic() - self._start_time

    def emit_on_trigger(self, tr_value: float) -> Tuple[int, List[float], dict]:
        self._vol_idx += 1
        regressors = self._compute_retrots(self._vol_idx, tr_value)
        meta = self._emit_meta(tr_value)
        return self._vol_idx, regressors, meta



    def _compute_retrots(self, n_vol: int, tr_value: float) -> List[float]:
        resp = np.asarray(self._resp, dtype=np.float32)
        card = np.asarray(self._card, dtype=np.float32)

        # --- EARLY GUARDS (prevents zero-size nanmin/nanmax + unstable early RetroTS) ---
        if resp.size == 0 or card.size == 0:
            return [0.0] * 8
        if resp.size < self._min_samples or card.size < self._min_samples:
            return [0.0] * 8
        # -----------------------------------------------------------------------------


        # ---- Robustify inputs (helps RetroTS peak/phase detection) ----
        def _z(x: np.ndarray) -> np.ndarray:
            x = x - np.nanmean(x)
            sd = np.nanstd(x)
            if not np.isfinite(sd) or sd < 1e-6:
                return np.zeros_like(x)
            return x / sd

        # If ECG is inverted (stronger negative excursions), flip it
        # (safe now because card.size > 0)
        if np.nanmin(card) < 0 and abs(np.nanmin(card)) > abs(np.nanmax(card)):
            card = -card

        # Optional: rectify ECG to emphasize R peaks (try if still unstable)
        # card = np.abs(card)

        resp = _z(resp)
        card = _z(card)
        # --------------------------------------------------------------
        log.info("[DBG] resp: min=%.4g max=%.4g std=%.4g | card: min=%.4g max=%.4g std=%.4g",
                float(resp.min()), float(resp.max()), float(resp.std()),
                float(card.min()), float(card.max()), float(card.std()))
        try:
            reg = self._retrots.RetroTs(
                resp,
                card,
                TR=tr_value,
                physFS=self._proc_fs,
                Nvol=n_vol,
            )
        except Exception as exc:
            # Never crash streaming because RetroTS had a short-window edge case
            log.warning("[RETROTS] RetroTs failed (n_vol=%d, nsamp=%d): %s",
                        n_vol, resp.size, exc)
            return [0.0] * 8  # typical RetroTS is 8 regressors (4 resp + 4 card)

        # reg can be shorter than Nvol early on or if detection failed
        if reg is None or len(reg) == 0:
            return [0.0] * 8

        # If fewer rows returned than requested, use the last available row
        idx = min(n_vol - 1, reg.shape[0] - 1)
        row = reg[idx]

        # Defensive: if shape is weird, fall back
        try:
            out = np.asarray(row, dtype=float).tolist()
        except Exception:
            out = [0.0] * 8

        # Ensure exactly 8 floats (pad/truncate)
        if len(out) < 8:
            out = out + [0.0] * (8 - len(out))
        elif len(out) > 8:
            out = out[:8]

        return out

    def add_sample_and_maybe_emit_by_samples(self, resp: float, card: float):
    # CSV/offline mode: drive volume emission purely from sample count.
    # Every samples_per_tr samples => one TR.
        self._resp.append(resp)
        self._card.append(card)
        self._sample_idx += 1

        # Emit volumes whenever we've crossed the next TR boundary in sample space
        while self._sample_idx >= (self._vol_idx + 1) * self._samples_per_tr:
            # Need enough historical data before computing RetroTS
            if len(self._resp) < self._min_samples or len(self._card) < self._min_samples:
                return

            self._vol_idx += 1
            regressors = self._compute_retrots(self._vol_idx, self.config.tr)
            meta = self._emit_meta(self.config.tr)
            yield self._vol_idx, regressors, meta
    def _emit_meta(self, tr_value: float) -> dict:
        return {
            "phys_fs": float(self._proc_fs),
            "tr": float(tr_value),
            "samples_per_tr": int(max(1, round(self._proc_fs * tr_value))),
            "sample_idx": int(self._sample_idx),
            "nsamp_total": int(len(self._resp)),
        }




class LivePlotter:
    """
    Live plot: separate axes for resp / card / trigger (no saving).
    Call:
        plotter = LivePlotter(window_s, update_hz)   # if enabled
        plotter.add_sample(t, resp, card, trigger)   # each sample
    """
    def __init__(self, window_s: float, update_hz: float):
        self._window_s = float(window_s)
        self._update_every = 1.0 / max(float(update_hz), 0.1)
        self._last_update = 0.0

        self._time: Deque[float] = deque()
        self._resp: Deque[float] = deque()
        self._card: Deque[float] = deque()
        self._trigger: Deque[float] = deque()

        self._enabled = False
        self._fig = None
        self._axes = None
        self._lines = None

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            log.warning("[PLOT] matplotlib unavailable (%s); live plot disabled.", exc)
            return

        self._plt = plt
        self._plt.ion()

        # 3 rows, shared X axis (time)
        self._fig, self._axes = self._plt.subplots(
            3, 1, sharex=True, num="BIOPAC Live", figsize=(10, 7)
        )

        # Create one line per axis
        self._lines = {
            "resp": self._axes[0].plot([], [])[0],
            "card": self._axes[1].plot([], [])[0],
            "trigger": self._axes[2].plot([], [])[0],
        }

        self._axes[0].set_title("Respiration")
        self._axes[0].set_ylabel("Resp")

        self._axes[1].set_title("Cardiac")
        self._axes[1].set_ylabel("Card")

        self._axes[2].set_title("Trigger")
        self._axes[2].set_ylabel("Trig")
        self._axes[2].set_xlabel("Time (s)")

        for ax in self._axes:
            ax.grid(True, alpha=0.3)

        self._enabled = True

    def add_sample(self, t: float, resp: float, card: float, trigger: float):
        if not self._enabled:
            return

        t = float(t)
        self._time.append(t)
        self._resp.append(float(resp))
        self._card.append(float(card))
        self._trigger.append(float(trigger))

        # keep last window_s seconds
        while self._time and (self._time[-1] - self._time[0]) > self._window_s:
            self._time.popleft()
            self._resp.popleft()
            self._card.popleft()
            self._trigger.popleft()

        now = time.monotonic()
        if now - self._last_update < self._update_every:
            return
        self._last_update = now

        self._lines["resp"].set_data(self._time, self._resp)
        self._lines["card"].set_data(self._time, self._card)
        self._lines["trigger"].set_data(self._time, self._trigger)

        # autoscale each axis independently
        for ax in self._axes:
            ax.relim()
            ax.autoscale_view()

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()


def _put_with_drop(target_queue: mp.Queue, item) -> None:
    try:
        target_queue.put_nowait(item)
    except queue_mod.Full:
        try:
            target_queue.get_nowait()
        except queue_mod.Empty:
            return
        try:
            target_queue.put_nowait(item)
        except queue_mod.Full:
            return


def _drain_latest(source_queue: mp.Queue):
    latest = None
    try:
        while True:
            latest = source_queue.get_nowait()
    except queue_mod.Empty:
        return latest


def _sample_source(config: StreamerConfig) -> Iterator[Tuple[float, float, float]]:
    if config.mode == "sim":
        return sim_samples(config.phys_fs, config.tr)
    if config.mode == "csv":
        if not config.csv_path:
            raise ValueError("--csv-path is required for csv mode.")
        return csv_samples(config.csv_path, config.phys_fs, card_source=config.card_source)
    if config.mode == "biopac":
        return biopac_samples(config)
    raise ValueError(f"Unknown mode: {config.mode}")


def _acquisition_worker(config: StreamerConfig, sample_queue: mp.Queue, stop_event: mp.Event):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    log = logging.getLogger("biopac_streamer.acq")
    try:
        source = _sample_source(config)
    except Exception as exc:
        log.error("[ACQ] Failed to start source: %s", exc)
        return

    sample_idx = 0
    try:
        for resp, card, trigger in source:
            if stop_event.is_set():
                break
            sample_idx += 1
            if config.mode == "csv":
                timestamp = sample_idx / float(config.phys_fs)
            else:
                timestamp = time.monotonic()
            _put_with_drop(sample_queue, (sample_idx, timestamp, resp, card, trigger))
    except Exception as exc:
        log.error("[ACQ] Acquisition stopped: %s", exc)


def _plotter_worker(
    plot_queue: mp.Queue,
    stop_event: mp.Event,
    window_s: float,
    update_hz: float,
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    plotter = LivePlotter(window_s, update_hz)
    try:
        while not stop_event.is_set():
            try:
                t, resp, card, trigger = plot_queue.get(timeout=0.1)
            except queue_mod.Empty:
                continue
            plotter.add_sample(t, resp, card, trigger)
    except Exception as exc:
        log = logging.getLogger("biopac_streamer.plot")
        log.warning("[PLOT] Plotter exited: %s", exc)


def _net_worker(
    host: str,
    port: int,
    connect_retry_s: float,
    payload_queue: mp.Queue,
    control_queue: mp.Queue,
    stop_event: mp.Event,
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    log = logging.getLogger("biopac_streamer.net")
    sock = None
    last_connect_attempt = 0.0
    handshake_buffer = ""
    net_connected = False

    def _connect():
        nonlocal sock, last_connect_attempt, handshake_buffer
        now = time.monotonic()
        if now - last_connect_attempt < connect_retry_s:
            return
        last_connect_attempt = now
        try:
            sock = socket.create_connection((host, port), timeout=2.0)
            sock.settimeout(0.0)
            log.info("[NET] Connected to RT PC %s:%s", host, port)
            handshake_buffer = ""
            _put_with_drop(control_queue, {"kind": "net_status", "connected": True})
        except OSError as exc:
            if sock is not None:
                sock.close()
            sock = None
            log.warning("[NET] Connection failed: %s", exc)
            _put_with_drop(control_queue, {"kind": "net_status", "connected": False})

    try:
        while not stop_event.is_set():
            if sock is None:
                _connect()
                time.sleep(0.05)
                continue

            tr_value, run_info, run_end, handshake_buffer = _poll_control(
                sock, handshake_buffer, fallback_tr=0.0
            )
            if tr_value is not None:
                _put_with_drop(control_queue, {"kind": "handshake", "tr": tr_value})
            if run_info is not None:
                _put_with_drop(
                    control_queue,
                    {
                        "kind": "run_start",
                        "subject": run_info.subject,
                        "day": run_info.day,
                        "run": run_info.run,
                    },
                )
            if run_end:
                _put_with_drop(control_queue, {"kind": "run_end"})

            payload = _drain_latest(payload_queue)
            if payload is None:
                time.sleep(0.005)
                continue

            try:
                sock.sendall((payload + "\n").encode("utf-8"))
            except OSError as exc:
                log.warning("[NET] Send failed: %s", exc)
                try:
                    sock.close()
                except OSError:
                    pass
                sock = None
                _put_with_drop(control_queue, {"kind": "net_status", "connected": False})
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

def _plotter_worker(
    plot_queue: mp.Queue,
    stop_event: mp.Event,
    window_s: float,
    update_hz: float,
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    plotter = LivePlotter(window_s, update_hz)
    try:
        while not stop_event.is_set():
            try:
                t, resp, card, trigger = plot_queue.get(timeout=0.1)
            except queue_mod.Empty:
                continue
            plotter.add_sample(t, resp, card, trigger)
    except Exception as exc:
        log = logging.getLogger("biopac_streamer.plot")
        log.warning("[PLOT] Plotter exited: %s", exc)


def _net_worker(
    host: str,
    port: int,
    connect_retry_s: float,
    payload_queue: mp.Queue,
    control_queue: mp.Queue,
    stop_event: mp.Event,
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    log = logging.getLogger("biopac_streamer.net")
    sock = None
    last_connect_attempt = 0.0
    handshake_buffer = ""
    net_connected = False

    def _connect():
        nonlocal sock, last_connect_attempt, handshake_buffer
        now = time.monotonic()
        if now - last_connect_attempt < connect_retry_s:
            return
        last_connect_attempt = now
        try:
            sock = socket.create_connection((host, port), timeout=2.0)
            sock.settimeout(0.0)
            log.info("[NET] Connected to RT PC %s:%s", host, port)
            handshake_buffer = ""
            _put_with_drop(control_queue, {"kind": "net_status", "connected": True})
        except OSError as exc:
            if sock is not None:
                sock.close()
            sock = None
            log.warning("[NET] Connection failed: %s", exc)
            _put_with_drop(control_queue, {"kind": "net_status", "connected": False})

    try:
        while not stop_event.is_set():
            if sock is None:
                _connect()
                time.sleep(0.05)
                continue

            tr_value, run_info, run_end, handshake_buffer = _poll_control(
                sock, handshake_buffer, fallback_tr=0.0
            )
            if tr_value is not None:
                _put_with_drop(control_queue, {"kind": "handshake", "tr": tr_value})
            if run_info is not None:
                _put_with_drop(
                    control_queue,
                    {
                        "kind": "run_start",
                        "subject": run_info.subject,
                        "day": run_info.day,
                        "run": run_info.run,
                    },
                )
            if run_end:
                _put_with_drop(control_queue, {"kind": "run_end"})

            payload = _drain_latest(payload_queue)
            if payload is None:
                time.sleep(0.005)
                continue

            try:
                sock.sendall((payload + "\n").encode("utf-8"))
            except OSError as exc:
                log.warning("[NET] Send failed: %s", exc)
                try:
                    sock.close()
                except OSError:
                    pass
                sock = None
                _put_with_drop(control_queue, {"kind": "net_status", "connected": False})
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

def sim_samples(sample_rate: float, tr: float) -> Iterator[Tuple[float, float, float]]:
    t0 = time.monotonic()
    idx = 0
    next_trigger = tr * 5
    while True:
        now = time.monotonic()
        t = now - t0
        resp = np.sin(2 * np.pi * 0.25 * t) + 0.02 * np.random.randn()
        card = np.sin(2 * np.pi * 1.1 * t) + 0.01 * np.random.randn()
        trigger = 5.0 if t >= next_trigger else 0.0
        if trigger > 0:
            next_trigger += tr
        yield resp, card, trigger
        idx += 1
        next_time = t0 + idx / sample_rate
        sleep_for = next_time - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)


def csv_samples(path: str, sample_rate: float, card_source: str = "ecg") -> Iterator[Tuple[float, float, float]]:
    """
    Headerless CSV/TSV/whitespace file with 5 columns:
      0: trigger (V)
      1: resp (cmH2O)
      2: ppg (V)
      3: ecg (mV)
      4: eda (uS)
    Returns (resp, card, trigger) with NO pacing/sleep.
    Timing is defined by --phys-fs (sample_rate) in the consumer.
    """
    card_source = card_source.lower().strip()
    if card_source not in ("ppg", "ecg"):
        raise ValueError("card_source must be 'ppg' or 'ecg'")

    with open(path, "r", newline="") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.replace(",", " ").split()
            if len(parts) < 5:
                continue
            try:
                vals = [float(x) for x in parts[:5]]
            except ValueError:
                continue

            trigger = vals[0]
            resp = vals[1]
            ppg = vals[2]
            ecg = vals[3]
            card = ppg if card_source == "ppg" else ecg
            yield resp, card, trigger



def biopac_samples(config: StreamerConfig) -> Iterator[Tuple[float, float, float]]:
    if not config.mpdev_dll:
        raise ValueError("mpdev.dll path required for biopac mode.")

    mpdev = cdll.LoadLibrary(config.mpdev_dll)
    mpdev.connectMPDev.argtypes = [c_int, c_int, c_char_p]
    retval = mpdev.connectMPDev(config.mp_device, config.mp_comm, b"auto")
    if retval != MPSUCCESS:
        raise RuntimeError(f"connectMPDev failed with code {retval}")

    mpdev.setSampleRate.argtypes = [c_double]
    retval = mpdev.setSampleRate(config.phys_fs/1000) # X = msec per sample, e.g. 2 = 500 Hz, 1 = 1000 Hz
    if retval != MPSUCCESS:
        raise RuntimeError(f"setSampleRate failed with code {retval}")

    arr_type = c_bool * 16
    channels = [False] * 16
    channels[config.resp_channel - 1] = True
    channels[config.card_channel - 1] = True
    if config.trigger_channel is not None:
        channels[config.trigger_channel - 1] = True
    retval = mpdev.setAcqChannels(arr_type(*channels))
    if retval != MPSUCCESS:
        raise RuntimeError(f"setAcqChannels failed with code {retval}")

    active_channels = [idx for idx, enabled in enumerate(channels) if enabled]
    if not active_channels:
        raise RuntimeError("No active BIOPAC channels selected.")
    channel_index = {ch: i for i, ch in enumerate(active_channels)}
    resp_idx = channel_index[config.resp_channel - 1]
    card_idx = channel_index[config.card_channel - 1]
    trigger_idx = channel_index[config.trigger_channel - 1] if config.trigger_channel is not None else None

    mpdev.startMPAcqDaemon.argtypes = []
    retval = mpdev.startMPAcqDaemon()
    if retval != MPSUCCESS:
        raise RuntimeError(f"startMPAcqDaemon failed with code {retval}")

    retval = mpdev.startAcquisition()
    if retval != MPSUCCESS:
        raise RuntimeError(f"startAcquisition failed with code {retval}")

    mpdev.receiveMPData.argtypes = [POINTER(c_double), c_uint32, POINTER(c_uint32)]
    samples_per_read = max(1, int(config.mp_chunk_samples))
    points_per_read = samples_per_read * len(active_channels)
    arr_type_double = c_double * points_per_read
    leftover: List[float] = []
    last_rate_t = time.monotonic()
    last_rate_warn_t = last_rate_t
    samples = 0
    points = 0
    reads = 0

    try:
        while True:
            buffer = arr_type_double()
            received = c_uint32(0)
            retval = mpdev.receiveMPData(buffer, points_per_read, received)
            reads += 1
            if retval == MPSUCCESS and received.value > 0:
                points += received.value
                data = list(buffer[:received.value])
                if leftover:
                    data = leftover + data
                    leftover = []

                full_samples = len(data) // len(active_channels)
                remainder = len(data) % len(active_channels)
                if remainder:
                    leftover = data[-remainder:]
                    data = data[:-remainder]

                for i in range(full_samples):
                    base = i * len(active_channels)
                    resp = data[base + resp_idx]
                    card = data[base + card_idx]
                    trigger = data[base + trigger_idx] if trigger_idx is not None else 0.0
                    yield resp, card, trigger
                samples += full_samples
            else:
                if config.mp_poll_sleep_ms > 0:
                    time.sleep(config.mp_poll_sleep_ms / 1000.0)

            now = time.monotonic()
            if now - last_rate_t >= 1.0:
                log.info("[BIOPAC] samples/s=%d  points/s=%d  reads/s=%d",
                         samples, points, reads)
                expected_samples = int(round(config.phys_fs))
                expected_points = int(round(config.phys_fs * len(active_channels)))
                if (now - last_rate_warn_t) >= 5.0:
                    if samples < expected_samples * 0.9:
                        log.warning(
                            "[BIOPAC] Low throughput: %d samples/s (expected ~%d). Consider lowering --mp-chunk-samples or setting --mp-poll-sleep-ms 0.",
                            samples,
                            expected_samples,
                        )
                    if points < expected_points * 0.9:
                        log.warning(
                            "[BIOPAC] Low throughput: %d points/s (expected ~%d). Active channels=%d.",
                            points,
                            expected_points,
                            len(active_channels),
                        )
                    last_rate_warn_t = now
                samples = 0
                points = 0
                reads = 0
                last_rate_t = now

            time.sleep(0.0)  # yield to scheduler
    finally:
        mpdev.stopAcquisition()
        mpdev.disconnectMPDev()


def run_streamer(config: StreamerConfig):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    source = None
    mp_context = None
    acq_process = None
    plot_process = None
    net_process = None
    stop_event = None
    sample_queue = None
    plot_queue = None
    payload_queue = None
    control_queue = None
    plot_window_s = config.plot_window_s if config.plot_window_s > 0 else config.tr

    if config.use_multiprocess:
        mp_context = mp.get_context("spawn")
        stop_event = mp_context.Event()
        sample_queue = mp_context.Queue(maxsize=max(1, int(config.queue_max)))
        acq_process = mp_context.Process(
            target=_acquisition_worker,
            args=(config, sample_queue, stop_event),
            daemon=True,
        )
        acq_process.start()
        if config.live_plot:
            plot_queue = mp_context.Queue(maxsize=max(1, int(config.plot_queue_max)))
            plot_process = mp_context.Process(
                target=_plotter_worker,
                args=(plot_queue, stop_event, plot_window_s, config.plot_update_hz),
                daemon=True,
            )
            plot_process.start()
        payload_queue = mp_context.Queue(maxsize=max(1, int(config.net_queue_max)))
        control_queue = mp_context.Queue(maxsize=max(1, int(config.queue_max)))
        net_process = mp_context.Process(
            target=_net_worker,
            args=(
                config.host,
                config.port,
                config.connect_retry_s,
                payload_queue,
                control_queue,
                stop_event,
            ),
            daemon=True,
        )
        net_process.start()
    else:
        source = _sample_source(config)

    retro = RetroTSStreamer(config)
    prev_trigger = 0.0
    last_trigger_time = None
    handshake_done = False
    run_active = False
    run_info: Optional[RunInfo] = None
    run_control_enabled = bool(config.run_control)
    rt_control_seen = False
    data_logger = RunDataLogger(Path(config.data_dir))
    samples_writer = None
    sent_writer = None
    regressors_writer = None
    samples_handle = None
    sent_handle = None
    regressors_handle = None
    # Optional downsampling for resp/card using timestamps
    ds_resp = None
    ds_card = None
    if config.downsample_hz and config.downsample_hz > 0:
        ds_resp = _TimeBucketDownsampler(target_hz=config.downsample_hz, mode="mean")
        ds_card = _TimeBucketDownsampler(target_hz=config.downsample_hz, mode="mean")
        log.info(
            "[DS] Timestamp downsample enabled: target %.1f Hz",
            config.downsample_hz,
        )
    if config.log_samples_csv:
        samples_handle = open(config.log_samples_csv, "a", newline="")
        samples_writer = csv.writer(samples_handle)
        if samples_handle.tell() == 0:
            samples_writer.writerow(["timestamp", "resp", "card", "trigger"])
    if config.log_sent_csv:
        sent_handle = open(config.log_sent_csv, "a", newline="")
        sent_writer = csv.writer(sent_handle)
        if sent_handle.tell() == 0:
           sent_writer.writerow(["timestamp", "volume_idx", "tr", "sample_idx", "nsamp_total", "samples_per_tr", "regressors"])
    if config.log_regressors_csv:
        regressors_handle = open(config.log_regressors_csv, "a", newline="")
        regressors_writer = csv.writer(regressors_handle)
        if regressors_handle.tell() == 0:
            regressors_writer.writerow(["timestamp", "volume_idx", "tr", "sample_idx", "nsamp_total", "samples_per_tr", "regressors"])


    plotter = None if config.use_multiprocess else (
        LivePlotter(plot_window_s, config.plot_update_hz) if config.live_plot else None
    )
    plot_sample_hz = min(
        config.downsample_hz if config.downsample_hz and config.downsample_hz > 0 else config.phys_fs,
        100.0,
    )
    plot_resp_ds = None
    plot_card_ds = None
    plot_trigger_ds = None
    if config.live_plot:
        plot_resp_ds = _TimeBucketDownsampler(target_hz=plot_sample_hz, mode="mean")
        plot_card_ds = _TimeBucketDownsampler(target_hz=plot_sample_hz, mode="mean")
        plot_trigger_ds = _TimeBucketDownsampler(target_hz=plot_sample_hz, mode="max")
    last_status_t = 0.0
    last_handshake_wait_start = None
    fixed_tr_allowed = True  # will be controlled below
    sock = None
    last_connect_attempt = 0.0
    handshake_buffer = ""
    run_log_dir = None
    last_regressor_t = None

    def _start_run(info: Optional[RunInfo], reason: str):
        nonlocal run_active, run_info, prev_trigger, last_trigger_time, run_log_dir, last_regressor_t
        retro.reset_for_run()
        prev_trigger = 0.0
        last_trigger_time = None
        run_info = info
        run_log_dir = data_logger.start_run(info)
        last_regressor_t = time.monotonic()
        run_active = True
        label = "offline" if info is None else f"{info.subject}/{info.day}/run{info.run}"
        log.info("[RUN] Started (%s) for %s in %s", reason, label, run_log_dir)

    def _stop_run(reason: str):
        nonlocal run_active, run_info, run_log_dir, last_regressor_t
        if not run_active and not data_logger.active:
            return
        data_logger.stop_run()
        run_active = False
        log.info("[RUN] Stopped (%s).", reason)
        run_info = None
        run_log_dir = None
        last_regressor_t = None

    def _maybe_connect():
        nonlocal sock, last_connect_attempt, handshake_done, handshake_buffer
        if config.use_multiprocess:
            return
        if sock is not None:
            return
        now = time.monotonic()
        if now - last_connect_attempt < config.connect_retry_s:
            return
        last_connect_attempt = now
        try:
            sock = socket.create_connection((config.host, config.port), timeout=2.0)
            sock.settimeout(0.0)
            log.info("[NET] Connected to RT PC %s:%s", config.host, config.port)
            handshake_buffer = ""
            if config.wait_for_handshake and config.trigger_channel is None:
                log.info("[NET] Awaiting handshake (non-blocking).")
        except OSError as exc:
            log.warning("[NET] Connection failed: %s", exc)
            sock = None

    def _send_payload(payload: str):
        nonlocal sock
        if payload_queue is not None:
            _put_with_drop(payload_queue, payload)
            return
        if sock is None:
            return
        try:
            sock.sendall((payload + "\n").encode("utf-8"))
        except OSError as exc:
            log.warning("[NET] Send failed: %s", exc)
            sock.close()
            sock = None

    if not run_control_enabled:
        _start_run(None, "no-run-control")

    try:
        sample_idx = 0  # raw sample counter (always raw-rate)
        start_ts = None
        while True:
            if config.use_multiprocess:
                assert sample_queue is not None
                try:
                    sample_idx, ts, resp, card, trigger = sample_queue.get(timeout=0.1)
                except queue_mod.Empty:
                    if stop_event is not None and stop_event.is_set():
                        break
                    continue
            else:
                try:
                    resp, card, trigger = next(source)
                except StopIteration:
                    break
                sample_idx += 1
                if config.mode == "csv":
                    ts = sample_idx / float(config.phys_fs)
                else:
                    ts = time.monotonic()
            if start_ts is None:
                start_ts = ts
            raw_time = ts - start_ts

            if control_queue is not None:
                while True:
                    try:
                        msg = control_queue.get_nowait()
                    except queue_mod.Empty:
                        break
                    kind = msg.get("kind")
                    if kind == "handshake":
                        try:
                            config.tr = float(msg.get("tr", config.tr))
                        except (TypeError, ValueError):
                            config.tr = config.tr
                        if retro._vol_idx == 0:
                            retro.reset_start_time()
                        handshake_done = True
                        fixed_tr_allowed = True
                        log.info("[NET] Handshake received (TR=%.3f).", config.tr)
                    elif kind == "run_start":
                        rt_control_seen = True
                        info = RunInfo(
                            str(msg.get("subject", "")),
                            str(msg.get("day", "")),
                            str(msg.get("run", "")),
                        )
                        if info.subject and info.day and info.run:
                            _start_run(info, "rt")
                    elif kind == "run_end":
                        rt_control_seen = True
                        _stop_run("rt")
                    elif kind == "net_status":
                        net_connected = bool(msg.get("connected", False))

            if control_queue is not None:
                while True:
                    try:
                        msg = control_queue.get_nowait()
                    except queue_mod.Empty:
                        break
                    kind = msg.get("kind")
                    if kind == "handshake":
                        try:
                            config.tr = float(msg.get("tr", config.tr))
                        except (TypeError, ValueError):
                            config.tr = config.tr
                        if retro._vol_idx == 0:
                            retro.reset_start_time()
                        handshake_done = True
                        fixed_tr_allowed = True
                        log.info("[NET] Handshake received (TR=%.3f).", config.tr)
                    elif kind == "run_start":
                        rt_control_seen = True
                        info = RunInfo(
                            str(msg.get("subject", "")),
                            str(msg.get("day", "")),
                            str(msg.get("run", "")),
                        )
                        if info.subject and info.day and info.run:
                            _start_run(info, "rt")
                    elif kind == "run_end":
                        rt_control_seen = True
                        _stop_run("rt")
                    elif kind == "net_status":
                        net_connected = bool(msg.get("connected", False))

            # Downsample resp/card if requested (trigger remains raw)
            downsampled_points: List[Tuple[float, float, float]] = []
            if ds_resp is not None and ds_card is not None:
                resp_out = ds_resp.step(ts, resp)
                card_out = ds_card.step(ts, card)
                if resp_out and card_out:
                    for idx in range(min(len(resp_out), len(card_out))):
                        t_out, resp_use = resp_out[idx]
                        _, card_use = card_out[idx]
                        downsampled_points.append((t_out, resp_use, card_use))
            else:
                downsampled_points.append((ts, resp, card))
            have_phys = bool(downsampled_points)

            _maybe_connect()
            if sock is not None:
                tr_value, new_run_info, run_end, handshake_buffer = _poll_control(
                    sock, handshake_buffer, config.tr
                )
                if tr_value is not None:
                    config.tr = tr_value
                    if retro._vol_idx == 0:
                        retro.reset_start_time()
                    handshake_done = True
                    fixed_tr_allowed = True
                    log.info("[NET] Handshake received (TR=%.3f).", config.tr)
                if new_run_info is not None:
                    rt_control_seen = True
                    _start_run(new_run_info, "rt")
                if run_end:
                    rt_control_seen = True
                    _stop_run("rt")

            if run_control_enabled and not run_active:
                continue

            # Ingest resp/card (possibly downsampled) into RetroTS buffers
            if have_phys:
                for _, resp_use, card_use in downsampled_points:
                    if ds_resp is None:
                        # old path (no DS): keep your original semantics
                        if config.mode == "csv":
                            retro._resp.append(resp_use)
                            retro._card.append(card_use)
                            retro._sample_idx += 1
                        else:
                            retro.add_sample(resp_use, card_use)
                    else:
                        # DS path: treat all modes the same on ingest (we are already controlling rate)
                        retro._resp.append(resp_use)
                        retro._card.append(card_use)
                        retro._sample_idx += 1

            if (
                run_active
                and config.idle_stop_after_s
                and last_regressor_t is not None
                and (time.monotonic() - last_regressor_t) >= config.idle_stop_after_s
            ):
                log.warning(
                    "[RUN] No regressors emitted for %.1fs; stopping run.",
                    config.idle_stop_after_s,
                )
                _stop_run("idle-timeout")
                continue

            # -----------------------------
            # Offline heartbeat / warmup
            # -----------------------------
            now_m = time.monotonic()
            if config.offline_status_every_s and config.offline_status_every_s > 0:
                if (now_m - last_status_t) >= config.offline_status_every_s:
                    last_status_t = now_m
                    if control_queue is not None:
                        conn_state = "up" if net_connected else "down"
                    else:
                        conn_state = "up" if sock is not None else "down"
                    # Show warmup progress for RetroTS
                    nsamp = len(retro._resp)
                    warm = f"warmup {nsamp}/{retro._min_samples}" if nsamp < retro._min_samples else "warm"
                    log.info(
                        "[LIVE] conn=%s | raw(resp=%.4g card=%.4g trig=%.3g) | ds_ingest=%s | %s | vol=%d",
                        conn_state,
                        float(resp),
                        float(card),
                        float(trigger),
                        "yes" if have_phys else "no",
                        warm,
                        retro._vol_idx,
                    )

            # Logging/plot stays RAW (so you can inspect real signals)
            if run_active:
                data_logger.log_sample(resp, card, trigger)
            if samples_writer is not None:
                samples_writer.writerow([time.time(), resp, card, trigger])
            plot_points: List[Tuple[float, float, float, float]] = []
            if plot_resp_ds is not None and plot_card_ds is not None and plot_trigger_ds is not None:
                resp_plot = plot_resp_ds.step(ts, resp)
                card_plot = plot_card_ds.step(ts, card)
                trig_plot = plot_trigger_ds.step(ts, trigger)
                if resp_plot and card_plot and trig_plot:
                    for idx in range(
                        min(len(resp_plot), len(card_plot), len(trig_plot))
                    ):
                        t_out, resp_p = resp_plot[idx]
                        _, card_p = card_plot[idx]
                        _, trig_p = trig_plot[idx]
                        plot_points.append((t_out - start_ts, resp_p, card_p, trig_p))
            if plotter is not None:
                for t_out, resp_p, card_p, trig_p in plot_points:
                    plotter.add_sample(t_out, resp_p, card_p, trig_p)
            if plot_queue is not None:
                for point in plot_points:
                    _put_with_drop(plot_queue, point)

            # ---------------------------------------------------------
            # Handshake gating (only relevant if NO trigger channel)
            # ---------------------------------------------------------
            if config.trigger_channel is None and config.wait_for_handshake and not handshake_done:
                if (net_connected if control_queue is not None else sock is not None):
                    # Start grace timer on first connect
                    if last_handshake_wait_start is None:
                        last_handshake_wait_start = time.monotonic()

                    waited = time.monotonic() - last_handshake_wait_start
                    if waited >= config.handshake_grace_s:
                        fixed_tr_allowed = bool(config.allow_offline_fixed_tr)
                    else:
                        fixed_tr_allowed = False
                else:
                    # Not connected at all -> either allow offline fixed TR or fully gate
                    fixed_tr_allowed = bool(config.allow_offline_fixed_tr)
            else:
                fixed_tr_allowed = True


            if config.trigger_channel is not None:
                trigger_now = trigger >= config.trigger_threshold
                trigger_prev = prev_trigger >= config.trigger_threshold
                if trigger_now and not trigger_prev:
                    now = raw_time

                    if last_trigger_time is None:
                        tr_value = config.tr
                    else:
                        tr_value = now - last_trigger_time
                        if tr_value <= 0:
                            tr_value = config.tr
                    if last_trigger_time is None or tr_value >= config.trigger_min_interval:
                        last_trigger_time = now
                        vol_idx, regressors, meta = retro.emit_on_trigger(tr_value)
                        if run_active:
                            data_logger.log_regressors(vol_idx, meta, regressors)
                        if regressors_writer is not None:
                            regressors_writer.writerow([time.time(), vol_idx, tr_value, regressors])
                        if config.print_every > 0 and (vol_idx % config.print_every == 0):
                            log.info(
                                "[RETROTS] vol=%s tr=%.3f samp=%d (perTR=%d, total=%d) \n reg=%s conn=%s",
                                vol_idx,
                                meta["tr"],
                                meta["sample_idx"],
                                meta["samples_per_tr"],
                                meta["nsamp_total"],
                                regressors,
                                "up"
                                if (net_connected if control_queue is not None else sock is not None)
                                else "down",
                            )
                        payload = json.dumps({
                            "kind": "retrots",
                            "volume_idx": vol_idx,
                            "n_regressors": len(regressors),
                            "regressors": regressors,
                            "timestamp": time.time(),
                            "tr": meta["tr"],
                            "phys_fs": meta["phys_fs"],
                            "sample_idx": meta["sample_idx"],
                            "nsamp_total": meta["nsamp_total"],
                            "samples_per_tr": meta["samples_per_tr"],
                        })
                        _send_payload(payload)
                        if sent_writer is not None:
                            sent_writer.writerow([time.time(), vol_idx, meta["tr"], meta["sample_idx"], meta["nsamp_total"], meta["samples_per_tr"], regressors])
                        last_regressor_t = time.monotonic()
                prev_trigger = trigger
            else:
                if config.mode == "csv":
                    if not have_phys or not fixed_tr_allowed:
                        continue
                    # Emit based on sample counts, not wall time
                    for vol_idx, regressors, meta in retro.add_sample_and_maybe_emit_by_samples(resp_use, card_use):
                        if run_active:
                            data_logger.log_regressors(vol_idx, meta, regressors)
                        if regressors_writer is not None:
                            regressors_writer.writerow([time.time(), vol_idx, config.tr, regressors])
                        if config.print_every > 0 and (vol_idx % config.print_every == 0):
                            log.info(
                                "[RETROTS] vol=%s tr=%.3f samp=%d (perTR=%d, total=%d) \n reg=%s conn=%s",
                                vol_idx,
                                meta["tr"],
                                meta["sample_idx"],
                                meta["samples_per_tr"],
                                meta["nsamp_total"],
                                regressors,
                                "up"
                                if (net_connected if control_queue is not None else sock is not None)
                                else "down",
                            )
                        payload = json.dumps({
                            "kind": "retrots",
                            "volume_idx": vol_idx,
                            "n_regressors": len(regressors),
                            "regressors": regressors,
                            "timestamp": time.time(),
                            "tr": meta["tr"],
                            "phys_fs": meta["phys_fs"],
                            "sample_idx": meta["sample_idx"],
                            "nsamp_total": meta["nsamp_total"],
                            "samples_per_tr": meta["samples_per_tr"],
                        })
                        _send_payload(payload)
                        if sent_writer is not None:
                            sent_writer.writerow([time.time(), vol_idx, meta["tr"], meta["sample_idx"], meta["nsamp_total"], meta["samples_per_tr"], regressors])
                        last_regressor_t = time.monotonic()

                else:
                    if not fixed_tr_allowed:
                        continue
                    for vol_idx, regressors, meta in retro.maybe_emit():
                        if run_active:
                            data_logger.log_regressors(vol_idx, meta, regressors)
                        if regressors_writer is not None:
                            regressors_writer.writerow([time.time(), vol_idx, config.tr, regressors])
                        if config.print_every > 0 and (vol_idx % config.print_every == 0):
                            log.info(
                                "[RETROTS] vol=%s tr=%.3f samp=%d (perTR=%d, total=%d) \n reg=%s conn=%s",
                                vol_idx,
                                meta["tr"],
                                meta["sample_idx"],
                                meta["samples_per_tr"],
                                meta["nsamp_total"],
                                regressors,
                                "up"
                                if (net_connected if control_queue is not None else sock is not None)
                                else "down",
                            )
                        payload = json.dumps({
                            "kind": "retrots",
                            "volume_idx": vol_idx,
                            "n_regressors": len(regressors),
                            "regressors": regressors,
                            "timestamp": time.time(),
                            "tr": meta["tr"],
                            "phys_fs": meta["phys_fs"],
                            "sample_idx": meta["sample_idx"],
                            "nsamp_total": meta["nsamp_total"],
                            "samples_per_tr": meta["samples_per_tr"],
                        })
                        _send_payload(payload)
                        if sent_writer is not None:
                            sent_writer.writerow([time.time(), vol_idx, meta["tr"], meta["sample_idx"], meta["nsamp_total"], meta["samples_per_tr"], regressors])
                        last_regressor_t = time.monotonic()

    finally:
        if stop_event is not None:
            stop_event.set()
        if acq_process is not None:
            acq_process.join(timeout=2.0)
        if plot_process is not None:
            plot_process.join(timeout=2.0)
        if net_process is not None:
            net_process.join(timeout=2.0)
        data_logger.stop_run()
        if samples_handle is not None:
            samples_handle.close()
        if sent_handle is not None:
            sent_handle.close()
        if regressors_handle is not None:
            regressors_handle.close()
        if sock is not None:
            sock.close()


def main():
    parser = argparse.ArgumentParser(description="BIOPAC RetroTS streamer")
    parser.add_argument("--host", required=True, help="RT PC host to connect.")
    parser.add_argument("--port", type=int, default=15000, help="RT PC port.")
    parser.add_argument(
        "--tr",
        type=float,
        required=True,
        help="Fallback fMRI TR (s) used before trigger-derived TR.",
    )
    parser.add_argument("--phys-fs", type=float, default=100.0, help="Physio sampling rate (Hz).")
    parser.add_argument("--resp-channel", type=int, default=12, help="BIOPAC resp channel (1-16).")
    parser.add_argument("--card-channel", type=int, default=14, help="BIOPAC card channel (1-16).")
    parser.add_argument("--trigger-channel", type=int, default = None, help="BIOPAC trigger channel (1-16).")
    parser.add_argument(
        "--trigger-threshold",
        type=float,
        default=1.0,
        help="Threshold for trigger edge detection.",
    )
    parser.add_argument(
        "--trigger-min-interval",
        type=float,
        default=None,
        help="Minimum seconds between trigger edges. Defaults to TR*0.9.",
    )
    parser.add_argument("--mode", choices=("sim", "csv", "biopac"), default="sim")
    parser.add_argument("--csv-path", help="CSV path with resp/card columns.")
    parser.add_argument("--mpdev-dll", help="Path to mpdev.dll for biopac mode.")
    parser.add_argument("--mp-device", type=int, default=MP150, help="BIOPAC device enum.")
    parser.add_argument("--mp-comm", type=int, default=MPUDP, help="BIOPAC comm enum.")
    parser.add_argument(
        "--mp-chunk-samples",
        type=int,
        default=10,
        help="BIOPAC daemon read size in samples per channel (used with receiveMPData).",
    )
    parser.add_argument(
        "--mp-poll-sleep-ms",
        type=float,
        default=0.0,
        help="Sleep duration (ms) when receiveMPData returns no data. 0 disables sleeping.",
    )
    parser.add_argument("--log-samples-csv", help="Optional CSV for raw samples.")
    parser.add_argument("--log-sent-csv", help="Optional CSV for sent regressors.")
    parser.add_argument("--log-regressors-csv", help="Optional CSV for computed regressors.")
    parser.add_argument(
        "--wait-for-handshake",
        action="store_true",
        help="Wait for RT PC handshake before fixed-TR streaming.",
    )
    parser.add_argument(
        "--live-plot",
        action="store_true",
        help="Show a live matplotlib plot of incoming physio data.",
    )
    parser.add_argument(
        "--plot-window-s",
        type=float,
        default=0.0,
        help="Seconds of data to show in the live plot window (<=0 uses TR).",
    )
    parser.add_argument(
        "--plot-update-hz",
        type=float,
        default=10.0,
        help="Plot refresh rate in Hz.",
    )
    parser.add_argument(
        "--connect-retry-s",
        type=float,
        default=2.0,
        help="Seconds between RT PC connection attempts.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print regressor status every N volumes (0 to disable).",
    )
    parser.add_argument(
        "--card-source",
        choices=("ppg", "ecg"),
        default="ecg",
        help="Which CSV column to use as cardiac: PPG or ECG.",
    )
    parser.add_argument(
        "--downsample-hz",
        type=float,
        default=0.0,
        help="If >0, software-downsample resp/card to this Hz before RetroTS (trigger stays raw). Requires near-integer factor vs --phys-fs.",
    )
    parser.add_argument(
        "--no-multiprocess",
        action="store_true",
        help="Disable multiprocessing (run acquisition/plotting in-process).",
    )
    parser.add_argument(
        "--queue-max",
        type=int,
        default=5000,
        help="Max samples to buffer between acquisition and processing.",
    )
    parser.add_argument(
        "--plot-queue-max",
        type=int,
        default=1000,
        help="Max samples to buffer for plotting.",
    )
    parser.add_argument(
        "--net-queue-max",
        type=int,
        default=500,
        help="Max messages to buffer for network sending.",
    )

    parser.add_argument("--offline-status-every-s", type=float, default=1.0,
                        help="Print local status every N seconds even if RT PC is down (0 disables).")
    parser.add_argument("--handshake-grace-s", type=float, default=2.0,
                        help="Seconds to wait for handshake after connect (no trigger mode).")
    parser.add_argument("--no-offline-fixed-tr", action="store_true",
                        help="If set, and --wait-for-handshake is enabled (no trigger), do not emit fixed-TR volumes until handshake arrives.")
    parser.add_argument("--data-dir", default="biopac_rt/data",
                        help="Base directory for run data logs.")
    parser.add_argument("--idle-stop-after-s", type=float, default=20.0,
                        help="Stop run if no regressors are emitted for this long (0 disables).")
    parser.add_argument("--no-run-control", action="store_true",
                        help="Disable run start/stop control messages from the RT PC.")


    args = parser.parse_args()

    config = StreamerConfig(
        host=args.host,
        port=args.port,
        tr=args.tr,
        phys_fs=args.phys_fs,
        resp_channel=args.resp_channel,
        card_channel=args.card_channel,
        mode=args.mode,
        csv_path=args.csv_path,
        mpdev_dll=args.mpdev_dll,
        mp_device=args.mp_device,
        mp_comm=args.mp_comm,
        mp_chunk_samples=args.mp_chunk_samples,
        mp_poll_sleep_ms=args.mp_poll_sleep_ms,
        trigger_channel=args.trigger_channel, 
        trigger_threshold=args.trigger_threshold,
        trigger_min_interval=(args.trigger_min_interval if args.trigger_min_interval is not None else args.tr * 0.9),
        log_samples_csv=args.log_samples_csv,
        log_sent_csv=args.log_sent_csv,
        log_regressors_csv=args.log_regressors_csv,
        wait_for_handshake=args.wait_for_handshake,
        live_plot=args.live_plot,
        plot_window_s=args.plot_window_s,
        plot_update_hz=args.plot_update_hz,
        connect_retry_s=args.connect_retry_s,
        print_every=args.print_every,
        card_source=args.card_source,
        downsample_hz=args.downsample_hz,
        use_multiprocess=(not args.no_multiprocess),
        queue_max=args.queue_max,
        plot_queue_max=args.plot_queue_max,
        net_queue_max=args.net_queue_max,
        offline_status_every_s=args.offline_status_every_s,
        handshake_grace_s=args.handshake_grace_s,
        allow_offline_fixed_tr=(not args.no_offline_fixed_tr),
        data_dir=args.data_dir,
        idle_stop_after_s=args.idle_stop_after_s,
        run_control=(not args.no_run_control),

    )
    run_streamer(config)


if __name__ == "__main__":
    main()
