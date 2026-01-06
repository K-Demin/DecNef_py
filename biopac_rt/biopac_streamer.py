import argparse
import csv
import json
import logging
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
    mp_chunk_samples: int = 50
    trigger_channel: Optional[int] = 1
    trigger_threshold: float = 0.5
    trigger_min_interval: float = 0.3
    log_samples_csv: Optional[str] = None
    log_sent_csv: Optional[str] = None
    log_regressors_csv: Optional[str] = None
    wait_for_handshake: bool = False
    live_plot: bool = False
    plot_window_s: float = 10.0
    plot_update_hz: float = 10.0
    connect_retry_s: float = 2.0
    print_every: int = 1
    card_source: str = "ecg"
    downsample_hz: float = 0.0
    data_dir: str = "biopac_rt/data"
    run_control: bool = True
    # --- Offline reporting / behavior ---
    offline_status_every_s: float = 1.0   # print "I'm alive" line every N seconds
    handshake_grace_s: float = 2.0        # how long to wait for handshake after connect
    allow_offline_fixed_tr: bool = True   # if wait_for_handshake but no RT PC, still emit locally

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
                time.sleep(0.0001)

            now = time.monotonic()
            if now - last_rate_t >= 1.0:
                log.info("[BIOPAC] samples/s=%d  points/s=%d  reads/s=%d",
                         samples, points, reads)
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
    if config.mode == "sim":
        source = sim_samples(config.phys_fs, config.tr)
    elif config.mode == "csv":
        if not config.csv_path:
            raise ValueError("--csv-path is required for csv mode.")
        source = csv_samples(config.csv_path, config.phys_fs, card_source=config.card_source)
    elif config.mode == "biopac":
        source = biopac_samples(config)
    else:
        raise ValueError(f"Unknown mode: {config.mode}")

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
    # Optional downsampling for resp/card (trigger stays raw)
    ds_resp = None
    ds_card = None
    if config.downsample_hz and config.downsample_hz > 0:
        ds_resp = _IntFactorDownsampler(fs_in=config.phys_fs, fs_out=config.downsample_hz, alpha=0.2)
        ds_card = _IntFactorDownsampler(fs_in=config.phys_fs, fs_out=config.downsample_hz, alpha=0.8)
        log.info("[DS] Software downsample enabled: %.1f -> %.1f Hz (factor=%d)",
                config.phys_fs, config.downsample_hz, ds_resp.factor)
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


    plotter = LivePlotter(config.plot_window_s, config.plot_update_hz) if config.live_plot else None
    last_status_t = 0.0
    last_handshake_wait_start = None
    fixed_tr_allowed = True  # will be controlled below
    sock = None
    last_connect_attempt = 0.0
    handshake_buffer = ""
    run_log_dir = None

    def _start_run(info: Optional[RunInfo], reason: str):
        nonlocal run_active, run_info, prev_trigger, last_trigger_time, run_log_dir
        retro.reset_for_run()
        prev_trigger = 0.0
        last_trigger_time = None
        run_info = info
        run_log_dir = data_logger.start_run(info)
        run_active = True
        label = "offline" if info is None else f"{info.subject}/{info.day}/run{info.run}"
        log.info("[RUN] Started (%s) for %s in %s", reason, label, run_log_dir)

    def _stop_run(reason: str):
        nonlocal run_active, run_info, run_log_dir
        if not run_active and not data_logger.active:
            return
        data_logger.stop_run()
        run_active = False
        log.info("[RUN] Stopped (%s).", reason)
        run_info = None
        run_log_dir = None

    def _maybe_connect():
        nonlocal sock, last_connect_attempt, handshake_done, handshake_buffer
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

    if not run_control_enabled:
        _start_run(None, "no-run-control")

    try:
        sample_idx = 0  # raw sample counter (always raw-rate)
        for resp, card, trigger in source:
            sample_idx += 1

            # Downsample resp/card if requested (trigger remains raw)
            have_phys = True
            if ds_resp is not None and ds_card is not None:
                resp_ds = ds_resp.step(resp)
                card_ds = ds_card.step(card)
                if resp_ds is None or card_ds is None:
                    have_phys = False
                else:
                    resp_use, card_use = resp_ds, card_ds
            else:
                resp_use, card_use = resp, card

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
                if sock is None and not data_logger.active and not rt_control_seen:
                    _start_run(None, "offline")
                else:
                    continue

            # Ingest resp/card (possibly downsampled) into RetroTS buffers
            if have_phys:
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

            # -----------------------------
            # Offline heartbeat / warmup
            # -----------------------------
            now_m = time.monotonic()
            if config.offline_status_every_s and config.offline_status_every_s > 0:
                if (now_m - last_status_t) >= config.offline_status_every_s:
                    last_status_t = now_m
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
            if plotter is not None:
                plotter.add_sample(time.monotonic(), resp, card, trigger)

            # ---------------------------------------------------------
            # Handshake gating (only relevant if NO trigger channel)
            # ---------------------------------------------------------
            if config.trigger_channel is None and config.wait_for_handshake and not handshake_done:
                if sock is not None:
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
                    if config.mode == "csv":
                        now = sample_idx / config.phys_fs   # raw trigger timebase
                    else:
                        now = time.monotonic()

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
                                "up" if sock is not None else "down",
                            )
                        if sock is not None:
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
                            try:
                                sock.sendall((payload + "\n").encode("utf-8"))
                            except OSError as exc:
                                log.warning("[NET] Send failed: %s", exc)
                                sock.close()
                                sock = None
                            else:
                                if sent_writer is not None:
                                    sent_writer.writerow([time.time(), vol_idx, meta["tr"], meta["sample_idx"], meta["nsamp_total"], meta["samples_per_tr"], regressors])
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
                                "up" if sock is not None else "down",
                            )
                        if sock is not None:
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

                            try:
                                sock.sendall((payload + "\n").encode("utf-8"))
                            except OSError as exc:
                                log.warning("[NET] Send failed: %s", exc)
                                sock.close()
                                sock = None
                            else:
                                if sent_writer is not None:
                                    sent_writer.writerow([time.time(), vol_idx, meta["tr"], meta["sample_idx"], meta["nsamp_total"], meta["samples_per_tr"], regressors])

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
                                "up" if sock is not None else "down",
                            )
                        if sock is not None:
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

                            try:
                                sock.sendall((payload + "\n").encode("utf-8"))
                            except OSError as exc:
                                log.warning("[NET] Send failed: %s", exc)
                                sock.close()
                                sock = None
                            else:
                                if sent_writer is not None:
                                    sent_writer.writerow([time.time(), vol_idx, meta["tr"], meta["sample_idx"], meta["nsamp_total"], meta["samples_per_tr"], regressors])

    finally:
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
    parser.add_argument("--trigger-channel", type=int, default = 1, help="BIOPAC trigger channel (1-16).")
    parser.add_argument(
        "--trigger-threshold",
        type=float,
        default=0.5,
        help="Threshold for trigger edge detection.",
    )
    parser.add_argument(
        "--trigger-min-interval",
        type=float,
        default=0.3,
        help="Minimum seconds between trigger edges.",
    )
    parser.add_argument("--mode", choices=("sim", "csv", "biopac"), default="sim")
    parser.add_argument("--csv-path", help="CSV path with resp/card columns.")
    parser.add_argument("--mpdev-dll", help="Path to mpdev.dll for biopac mode.")
    parser.add_argument("--mp-device", type=int, default=MP150, help="BIOPAC device enum.")
    parser.add_argument("--mp-comm", type=int, default=MPUDP, help="BIOPAC comm enum.")
    parser.add_argument(
        "--mp-chunk-samples",
        type=int,
        default=500,
        help="BIOPAC daemon read size in samples per channel (used with receiveMPData).",
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
        default=10.0,
        help="Seconds of data to show in the live plot window.",
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

    parser.add_argument("--offline-status-every-s", type=float, default=1.0,
                        help="Print local status every N seconds even if RT PC is down (0 disables).")
    parser.add_argument("--handshake-grace-s", type=float, default=2.0,
                        help="Seconds to wait for handshake after connect (no trigger mode).")
    parser.add_argument("--no-offline-fixed-tr", action="store_true",
                        help="If set, and --wait-for-handshake is enabled (no trigger), do not emit fixed-TR volumes until handshake arrives.")
    parser.add_argument("--data-dir", default="biopac_rt/data",
                        help="Base directory for run data logs.")
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
        trigger_channel=args.trigger_channel,
        trigger_threshold=args.trigger_threshold,
        trigger_min_interval=args.trigger_min_interval,
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
        offline_status_every_s=args.offline_status_every_s,
        handshake_grace_s=args.handshake_grace_s,
        allow_offline_fixed_tr=(not args.no_offline_fixed_tr),
        data_dir=args.data_dir,
        run_control=(not args.no_run_control),

    )
    run_streamer(config)


if __name__ == "__main__":
    main()
