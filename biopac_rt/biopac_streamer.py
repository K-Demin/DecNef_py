import argparse
import csv
import json
import socket
import time
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

EXAMPLES
========
BIOPAC device + trigger:
  python -m biopac_rt.biopac_streamer \
    --host 10.0.0.5 --port 15000 --tr 0.9 --phys-fs 100 \
    --mode biopac --mpdev-dll "C:\\Program Files\\BIOPAC Systems, Inc\\...\\mpdev.dll" \
    --resp-channel 1 --card-channel 2 --trigger-channel 3 \
    --log-samples-csv biopac_samples.csv --log-sent-csv biopac_regressors.csv

Simulated stream (no trigger channel):
  python -m biopac_rt.biopac_streamer \
    --host 10.0.0.5 --port 15000 --tr 0.9 --phys-fs 100 --mode sim
"""

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np
from ctypes import c_bool, c_double, c_int, c_char_p, cdll

from fmri_rt_preproc.RTPSpy_tools.rtp_retrots import RtpRetroTS


MP160 = 103
MPUDP = 11
MPSUCCESS = 1


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
    mp_device: int = MP160
    mp_comm: int = MPUDP
    trigger_channel: Optional[int] = None
    trigger_threshold: float = 0.5
    trigger_min_interval: float = 0.3
    log_samples_csv: Optional[str] = None
    log_sent_csv: Optional[str] = None
    wait_for_handshake: bool = False


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

    def emit_on_trigger(self, tr_value: float) -> Tuple[int, list[float]]:
        self._vol_idx += 1
        regressors = self._compute_retrots(self._vol_idx, tr_value)
        return self._vol_idx, regressors

    def _compute_retrots(self, n_vol: int, tr_value: float) -> list[float]:
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


def sim_samples(sample_rate: float, tr: float) -> Iterator[tuple[float, float, float]]:
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


def csv_samples(path: str, sample_rate: float) -> Iterator[tuple[float, float, float]]:
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


def biopac_samples(config: StreamerConfig) -> Iterator[tuple[float, float, float]]:
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
    samples_writer = None
    sent_writer = None
    samples_handle = None
    sent_handle = None
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

    with socket.create_connection((config.host, config.port)) as sock:
        if config.wait_for_handshake and config.trigger_channel is None:
            tr_value = _wait_for_handshake(sock, config.tr)
            if tr_value is not None:
                config.tr = tr_value
            retro.reset_start_time()
        try:
            for resp, card, trigger in source:
                retro.add_sample(resp, card)
                if samples_writer is not None:
                    samples_writer.writerow([time.time(), resp, card, trigger])
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
                            sock.sendall((payload + "\n").encode("utf-8"))
                            if sent_writer is not None:
                                sent_writer.writerow([time.time(), vol_idx, tr_value, regressors])
                    prev_trigger = trigger
                else:
                    for vol_idx, regressors in retro.maybe_emit():
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
                        sock.sendall((payload + "\n").encode("utf-8"))
                        if sent_writer is not None:
                            sent_writer.writerow([time.time(), vol_idx, config.tr, regressors])
        finally:
            if samples_handle is not None:
                samples_handle.close()
            if sent_handle is not None:
                sent_handle.close()


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
    parser.add_argument("--mp-device", type=int, default=MP160, help="BIOPAC device enum.")
    parser.add_argument("--mp-comm", type=int, default=MPUDP, help="BIOPAC comm enum.")
    parser.add_argument("--log-samples-csv", help="Optional CSV for raw samples.")
    parser.add_argument("--log-sent-csv", help="Optional CSV for sent regressors.")
    parser.add_argument(
        "--wait-for-handshake",
        action="store_true",
        help="Wait for RT PC handshake before fixed-TR streaming.",
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
        wait_for_handshake=args.wait_for_handshake,
    )
    run_streamer(config)


if __name__ == "__main__":
    main()
