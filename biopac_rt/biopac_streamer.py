import argparse
import csv
import json
import logging
import socket
import time
from collections import deque
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

Simulated stream (no trigger channel):
  python -m biopac_rt.biopac_streamer \
    --host 115.145.189.30 --port 15000 --tr 0.9 --phys-fs 100 --mode sim
"""

from dataclasses import dataclass
from typing import Deque, Iterator, List, Optional, Tuple

import numpy as np
from ctypes import c_bool, c_double, c_int, c_char_p, cdll

from fmri_rt_preproc.RTPSpy_tools.rtp_retrots import RtpRetroTS

MP150 = 101
MP160 = 103
MPUDP = 11
MPSUCCESS = 1

log = logging.getLogger("biopac_streamer")


def _wait_for_handshake(sock: socket.socket, fallback_tr: float) -> Optional[float]:
    sock.settimeout(0.5)
    buffer = ""
    while True:
        try:
            chunk = sock.recv(4096)
        except socket.timeout:
            continue
        if not chunk:
            return None
        buffer += chunk.decode("utf-8")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            if line == "s":
                return None
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("kind") == "handshake":
                try:
                    tr_value = float(payload.get("tr", fallback_tr))
                except (TypeError, ValueError):
                    tr_value = fallback_tr
                return tr_value


def _poll_handshake(sock: socket.socket, buffer: str, fallback_tr: float) -> Tuple[Optional[float], str]:
    try:
        chunk = sock.recv(4096)
    except (BlockingIOError, socket.timeout):
        return None, buffer
    if not chunk:
        return None, buffer
    buffer += chunk.decode("utf-8")
    while "\n" in buffer:
        line, buffer = buffer.split("\n", 1)
        line = line.strip()
        if not line:
            continue
        if line == "s":
            return None, buffer
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("kind") == "handshake":
            try:
                tr_value = float(payload.get("tr", fallback_tr))
            except (TypeError, ValueError):
                tr_value = fallback_tr
            return tr_value, buffer
    return None, buffer


@dataclass
class StreamerConfig:
    host: str
    port: int
    tr: float
    phys_fs: float
    resp_channel: int
    card_channel: int
    mode: str
    csv_path: Optional[str] = None
    mpdev_dll: Optional[str] = None
    mp_device: int = MP150
    mp_comm: int = MPUDP
    trigger_channel: Optional[int] = None
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


class RetroTSStreamer:
    def __init__(self, config: StreamerConfig):
        self.config = config
        self._retrots = RtpRetroTS()
        self._resp = []
        self._card = []
        self._vol_idx = 0
        self._start_time = time.monotonic()

    def reset_start_time(self):
        self._start_time = time.monotonic()

    def add_sample(self, resp: float, card: float):
        self._resp.append(resp)
        self._card.append(card)

    def maybe_emit(self):
        elapsed = time.monotonic() - self._start_time
        while elapsed >= (self._vol_idx + 1) * self.config.tr:
            self._vol_idx += 1
            regressors = self._compute_retrots(self._vol_idx, self.config.tr)
            yield self._vol_idx, regressors
            elapsed = time.monotonic() - self._start_time

    def emit_on_trigger(self, tr_value: float) -> Tuple[int, List[float]]:
        self._vol_idx += 1
        regressors = self._compute_retrots(self._vol_idx, tr_value)
        return self._vol_idx, regressors

    def _compute_retrots(self, n_vol: int, tr_value: float) -> List[float]:
        resp = np.asarray(self._resp, dtype=np.float32)
        card = np.asarray(self._card, dtype=np.float32)
        reg = self._retrots.RetroTs(
            resp,
            card,
            TR=tr_value,
            physFS=self.config.phys_fs,
            Nvol=n_vol,
        )
        return reg[n_vol - 1].astype(float).tolist()


class LivePlotter:
    def __init__(self, window_s: float, update_hz: float):
        self._window_s = window_s
        self._update_every = 1.0 / max(update_hz, 0.1)
        self._last_update = 0.0
        self._resp: Deque[float] = deque()
        self._card: Deque[float] = deque()
        self._trigger: Deque[float] = deque()
        self._time: Deque[float] = deque()
        self._enabled = False
        self._fig = None
        self._ax = None
        self._lines = None

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            log.warning("[PLOT] matplotlib unavailable (%s); live plot disabled.", exc)
            return

        self._plt = plt
        self._plt.ion()
        self._fig, self._ax = self._plt.subplots(num="BIOPAC Live")
        self._lines = {
            "resp": self._ax.plot([], [], label="resp")[0],
            "card": self._ax.plot([], [], label="card")[0],
            "trigger": self._ax.plot([], [], label="trigger")[0],
        }
        self._ax.set_xlabel("Time (s)")
        self._ax.set_ylabel("Signal")
        self._ax.legend(loc="upper right")
        self._enabled = True

    def add_sample(self, t: float, resp: float, card: float, trigger: float):
        if not self._enabled:
            return
        self._time.append(t)
        self._resp.append(resp)
        self._card.append(card)
        self._trigger.append(trigger)
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
        self._ax.relim()
        self._ax.autoscale_view()
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()


def sim_samples(sample_rate: float, tr: float) -> Iterator[Tuple[float, float, float]]:
    t0 = time.monotonic()
    idx = 0
    next_trigger = tr
    while True:
        now = time.monotonic()
        t = now - t0
        resp = np.sin(2 * np.pi * 0.25 * t) + 0.02 * np.random.randn()
        card = np.sin(2 * np.pi * 1.1 * t) + 0.01 * np.random.randn()
        trigger = 1.0 if t >= next_trigger else 0.0
        if trigger > 0:
            next_trigger += tr
        yield resp, card, trigger
        idx += 1
        next_time = t0 + idx / sample_rate
        sleep_for = next_time - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)


def csv_samples(path: str, sample_rate: float) -> Iterator[Tuple[float, float, float]]:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return
    t0 = time.monotonic()
    for idx, row in enumerate(rows):
        resp = float(row.get("resp", row.get("respiration", 0.0)))
        card = float(row.get("card", row.get("cardiac", 0.0)))
        trigger = float(row.get("trigger", row.get("ttl", 0.0)))
        yield resp, card, trigger
        next_time = t0 + (idx + 1) / sample_rate
        sleep_for = next_time - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)


def biopac_samples(config: StreamerConfig) -> Iterator[Tuple[float, float, float]]:
    if not config.mpdev_dll:
        raise ValueError("mpdev.dll path required for biopac mode.")

    mpdev = cdll.LoadLibrary(config.mpdev_dll)
    mpdev.connectMPDev.argtypes = [c_int, c_int, c_char_p]
    retval = mpdev.connectMPDev(config.mp_device, config.mp_comm, b"auto")
    if retval != MPSUCCESS:
        raise RuntimeError(f"connectMPDev failed with code {retval}")

    mpdev.setSampleRate.argtypes = [c_double]
    retval = mpdev.setSampleRate(1.0 / config.phys_fs)
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

    retval = mpdev.startAcquisition()
    if retval != MPSUCCESS:
        raise RuntimeError(f"startAcquisition failed with code {retval}")

    arr_type_double = c_double * 16
    try:
        while True:
            samples = arr_type_double(0.0)
            retval = mpdev.getMostRecentSample(samples)
            if retval == MPSUCCESS:
                resp = samples[config.resp_channel - 1]
                card = samples[config.card_channel - 1]
                trigger = 0.0
                if config.trigger_channel is not None:
                    trigger = samples[config.trigger_channel - 1]
                yield resp, card, trigger
            time.sleep(1.0 / config.phys_fs)
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
        source = csv_samples(config.csv_path, config.phys_fs)
    elif config.mode == "biopac":
        source = biopac_samples(config)
    else:
        raise ValueError(f"Unknown mode: {config.mode}")

    retro = RetroTSStreamer(config)
    prev_trigger = 0.0
    last_trigger_time = None
    handshake_done = False
    samples_writer = None
    sent_writer = None
    regressors_writer = None
    samples_handle = None
    sent_handle = None
    regressors_handle = None
    if config.log_samples_csv:
        samples_handle = open(config.log_samples_csv, "a", newline="")
        samples_writer = csv.writer(samples_handle)
        if samples_handle.tell() == 0:
            samples_writer.writerow(["timestamp", "resp", "card", "trigger"])
    if config.log_sent_csv:
        sent_handle = open(config.log_sent_csv, "a", newline="")
        sent_writer = csv.writer(sent_handle)
        if sent_handle.tell() == 0:
            sent_writer.writerow(["timestamp", "volume_idx", "tr", "regressors"])
    if config.log_regressors_csv:
        regressors_handle = open(config.log_regressors_csv, "a", newline="")
        regressors_writer = csv.writer(regressors_handle)
        if regressors_handle.tell() == 0:
            regressors_writer.writerow(["timestamp", "volume_idx", "tr", "regressors"])

    plotter = LivePlotter(config.plot_window_s, config.plot_update_hz) if config.live_plot else None
    sock = None
    last_connect_attempt = 0.0
    handshake_buffer = ""

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

    try:
        for resp, card, trigger in source:
            retro.add_sample(resp, card)
            if samples_writer is not None:
                samples_writer.writerow([time.time(), resp, card, trigger])
            if plotter is not None:
                plotter.add_sample(time.monotonic(), resp, card, trigger)

            _maybe_connect()
            if (
                sock is not None
                and config.wait_for_handshake
                and config.trigger_channel is None
                and not handshake_done
            ):
                tr_value, handshake_buffer = _poll_handshake(sock, handshake_buffer, config.tr)
                if tr_value is not None:
                    config.tr = tr_value
                    if retro._vol_idx == 0:
                        retro.reset_start_time()
                    handshake_done = True
                    log.info("[NET] Handshake received (TR=%.3f).", config.tr)

            if config.trigger_channel is not None:
                trigger_now = trigger >= config.trigger_threshold
                trigger_prev = prev_trigger >= config.trigger_threshold
                if trigger_now and not trigger_prev:
                    now = time.monotonic()
                    if last_trigger_time is None:
                        tr_value = config.tr
                    else:
                        tr_value = now - last_trigger_time
                        if tr_value <= 0:
                            tr_value = config.tr
                    if last_trigger_time is None or tr_value >= config.trigger_min_interval:
                        last_trigger_time = now
                        vol_idx, regressors = retro.emit_on_trigger(tr_value)
                        if regressors_writer is not None:
                            regressors_writer.writerow([time.time(), vol_idx, tr_value, regressors])
                        if config.print_every > 0 and (vol_idx % config.print_every == 0):
                            log.info(
                                "[RETROTS] vol=%s tr=%.3f reg=%s conn=%s",
                                vol_idx,
                                tr_value,
                                regressors,
                                "up" if sock is not None else "down",
                            )
                        if sock is not None:
                            payload = json.dumps(
                                {
                                    "kind": "retrots",
                                    "volume_idx": vol_idx,
                                    "n_regressors": len(regressors),
                                    "regressors": regressors,
                                    "timestamp": time.time(),
                                    "tr": tr_value,
                                }
                            )
                            try:
                                sock.sendall((payload + "\n").encode("utf-8"))
                            except OSError as exc:
                                log.warning("[NET] Send failed: %s", exc)
                                sock.close()
                                sock = None
                            else:
                                if sent_writer is not None:
                                    sent_writer.writerow([time.time(), vol_idx, tr_value, regressors])
                prev_trigger = trigger
            else:
                for vol_idx, regressors in retro.maybe_emit():
                    if regressors_writer is not None:
                        regressors_writer.writerow([time.time(), vol_idx, config.tr, regressors])
                    if config.print_every > 0 and (vol_idx % config.print_every == 0):
                        log.info(
                            "[RETROTS] vol=%s tr=%.3f reg=%s conn=%s",
                            vol_idx,
                            config.tr,
                            regressors,
                            "up" if sock is not None else "down",
                        )
                    if sock is not None:
                        payload = json.dumps(
                            {
                                "kind": "retrots",
                                "volume_idx": vol_idx,
                                "n_regressors": len(regressors),
                                "regressors": regressors,
                                "timestamp": time.time(),
                                "tr": config.tr,
                            }
                        )
                        try:
                            sock.sendall((payload + "\n").encode("utf-8"))
                        except OSError as exc:
                            log.warning("[NET] Send failed: %s", exc)
                            sock.close()
                            sock = None
                        else:
                            if sent_writer is not None:
                                sent_writer.writerow([time.time(), vol_idx, config.tr, regressors])
    finally:
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
    parser.add_argument("--resp-channel", type=int, default=1, help="BIOPAC resp channel (1-16).")
    parser.add_argument("--card-channel", type=int, default=2, help="BIOPAC card channel (1-16).")
    parser.add_argument("--trigger-channel", type=int, help="BIOPAC trigger channel (1-16).")
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
    )
    run_streamer(config)


if __name__ == "__main__":
    main()
