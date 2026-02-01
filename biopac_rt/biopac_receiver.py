"""
BIOPAC RetroTS receiver.

USAGE OVERVIEW
==============
1) Run on the real-time PC (RT PC).
   - Listens for BIOPAC RetroTS JSON lines via TCP.
   - Buffers per-volume regressors.
   - Provides a get_retrots() method compatible with RTPSpy's RtpRegress.

2) Integration
   - In rt_pipeline.py, enable via --biopac-enable and set host/port.
   - The receiver will zero-fill regressors if a volume times out, allowing
     regression to proceed while tracking which volumes were missing.

3) Expected data format (JSON per line)
   {"kind":"retrots","volume_idx":12,"n_regressors":8,"regressors":[...],"timestamp":...,"tr":...}

EXAMPLE (manual use)
====================
from biopac_rt.biopac_receiver import BiopacReceiverConfig, BiopacRetroTSReceiver
cfg = BiopacReceiverConfig(host="0.0.0.0", port=15000, timeout=0.3)
rx = BiopacRetroTSReceiver(cfg)
rx.start()
... pass rx into RtpRegress via rtp_physio ...
"""

import json
import logging
import socket
import threading
import time
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import numpy as np


log = logging.getLogger("biopac_receiver")


@dataclass
class BiopacReceiverConfig:
    host: str = "115.145.189.30"
    port: int = 15000
    timeout: float = 0.3
    expected_regressors: Optional[int] = None
    handshake_tr: Optional[float] = None
    subject: Optional[str] = None
    day: Optional[str] = None
    run: Optional[str] = None
    output_path: Optional[Path] = None


class BiopacRetroTSReceiver:
    def __init__(self, config: BiopacReceiverConfig):
        self.config = config
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._regressors_by_vol: dict[int, np.ndarray] = {}
        self._n_reg: Optional[int] = None
        self._missing_vols: set[int] = set()
        self._server_sock: Optional[socket.socket] = None
        self._conn: Optional[socket.socket] = None
        self._output_ready = False
        self._output_lock = threading.Lock()

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        log.info("BIOPAC receiver listening on %s:%s", self.config.host, self.config.port)

    def stop(self):
        self._stop.set()
        self._send_run_end()
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def get_retrots(self, TR: float, vol_idx: int, tshift: float, timeout: Optional[float] = None):
        wait_time = self.config.timeout if timeout is None else timeout
        deadline = time.monotonic() + max(0.0, wait_time)
        with self._cond:
            while vol_idx not in self._regressors_by_vol and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(timeout=remaining)

            n_reg = self._ensure_regressor_count()
            retro = np.zeros((vol_idx, n_reg), dtype=np.float32)
            for idx, reg in self._regressors_by_vol.items():
                if 1 <= idx <= vol_idx:
                    retro[idx - 1, : reg.shape[0]] = reg

            if vol_idx not in self._regressors_by_vol:
                if vol_idx not in self._missing_vols:
                    log.warning(
                        "[BIOPAC] Missing physio regressors for vol %s; using zeros.",
                        vol_idx,
                    )
                self._missing_vols.add(vol_idx)

            return retro


    def was_missing(self, vol_idx: int) -> bool:
        with self._lock:
            return vol_idx in self._missing_vols

    def _ensure_regressor_count(self) -> int:
        if self._n_reg is not None:
            return self._n_reg
        if self.config.expected_regressors is not None:
            self._n_reg = self.config.expected_regressors
            return self._n_reg
        self._n_reg = 8
        return self._n_reg

    def _run_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.config.host, self.config.port))
            server.listen(1)
            server.settimeout(0.5)
            self._server_sock = server
            while not self._stop.is_set():
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break

                log.info("[BIOPAC] Connected from %s:%s", addr[0], addr[1])
                with self._lock:
                    self._conn = conn
                if self.config.handshake_tr is not None:
                    payload = json.dumps(
                        {
                            "kind": "handshake",
                            "tr": self.config.handshake_tr,
                            "timestamp": time.time(),
                        }
                    )
                    try:
                        conn.sendall((payload + "\n").encode("utf-8"))
                    except OSError:
                        log.warning("[BIOPAC] Failed to send handshake.")
                with conn:
                    conn.settimeout(0.5)
                    buffer = ""
                    while not self._stop.is_set():
                        try:
                            chunk = conn.recv(4096)
                        except socket.timeout:
                            continue
                        except OSError:
                            break
                        if not chunk:
                            break
                        buffer += chunk.decode("utf-8")
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            self._handle_line(line)
                log.info("[BIOPAC] Connection closed.")
                with self._lock:
                    self._conn = None

    def _send_run_start(self):
        if not (self.config.subject and self.config.day and self.config.run):
            return
        payload = {
            "kind": "run_start",
            "subject": self.config.subject,
            "day": self.config.day,
            "run": self.config.run,
            "timestamp": time.time(),
        }
        self._send_control(payload)

    def send_run_start(self):
        self._send_run_start()

    def _send_run_end(self):
        payload = {
            "kind": "run_end",
            "timestamp": time.time(),
        }
        self._send_control(payload)

    def _send_control(self, payload: dict):
        with self._lock:
            conn = self._conn
        if conn is None:
            return
        try:
            conn.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        except OSError:
            log.warning("[BIOPAC] Failed to send control message.")

    def _handle_line(self, line: str):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            log.warning("[BIOPAC] Malformed JSON message ignored.")
            return
        if payload.get("kind") != "retrots":
            return
        vol_idx = payload.get("volume_idx")
        regressors = payload.get("regressors")
        if not isinstance(vol_idx, int) or not isinstance(regressors, list):
            return
        reg = np.asarray(regressors, dtype=np.float32)
        if reg.ndim != 1:
            return
        with self._cond:
            if self._n_reg is None:
                self._n_reg = int(payload.get("n_regressors", reg.shape[0]))
            self._regressors_by_vol[vol_idx] = reg
            self._cond.notify_all()
        self._append_received(vol_idx, reg, payload.get("timestamp"))
        log.info("[BIOPAC] Received regressors for vol %s (%d values).", vol_idx, reg.shape[0])

    def _append_received(self, vol_idx: int, reg: np.ndarray, timestamp: Optional[float]) -> None:
        if self.config.output_path is None:
            return
        with self._output_lock:
            path = Path(self.config.output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            exists = path.exists()
            try:
                with open(path, "a", newline="") as f:
                    if not exists:
                        header = ["volume_idx", "timestamp"] + [f"reg_{i+1:02d}" for i in range(reg.shape[0])]
                        f.write(",".join(header) + "\n")
                    ts = time.time() if timestamp is None else float(timestamp)
                    row = [str(vol_idx), f"{ts:.6f}"] + [f"{v:.6f}" for v in reg.tolist()]
                    f.write(",".join(row) + "\n")
            except OSError as exc:
                log.warning("[BIOPAC] Failed to write regressors CSV: %s", exc)

    def wait_for_volume(self, vol_idx: int, timeout: Optional[float] = None) -> bool:
        """
        Block until `vol_idx` exists in the buffer.

        Returns True if available, False if timed out or stopped.
        - timeout=None means wait indefinitely.
        - NO side effects: does not zero-fill and does not mark missing.
        """
        with self._cond:
            if timeout is None:
                while vol_idx not in self._regressors_by_vol and not self._stop.is_set():
                    self._cond.wait(timeout=0.5)  # periodic wake to re-check stop
                return (vol_idx in self._regressors_by_vol) and (not self._stop.is_set())

            deadline = time.monotonic() + max(0.0, timeout)
            while vol_idx not in self._regressors_by_vol and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(timeout=min(0.5, remaining))
            return vol_idx in self._regressors_by_vol


class BiopacRetroTSFileBuffer:
    """Read RetroTS regressors from a continuously written CSV file."""

    def __init__(
        self,
        path: Path,
        timeout: float = 0.3,
        expected_regressors: Optional[int] = None,
        poll_interval: float = 0.05,
    ) -> None:
        self.path = Path(path)
        self.timeout = timeout
        self.expected_regressors = expected_regressors
        self.poll_interval = poll_interval
        self._lock = threading.Lock()
        self._regressors_by_vol: dict[int, np.ndarray] = {}
        self._missing_vols: set[int] = set()
        self._n_reg: Optional[int] = None
        self._file_pos = 0
        self._header_skipped = False
        self._remainder = ""

    def _ensure_regressor_count(self) -> int:
        if self._n_reg is not None:
            return self._n_reg
        if self.expected_regressors is not None:
            self._n_reg = self.expected_regressors
            return self._n_reg
        self._n_reg = 8
        return self._n_reg

    def _update_from_file(self) -> None:
        if not self.path.exists():
            return
        with self._lock:
            with open(self.path, "r", encoding="utf-8") as f:
                f.seek(self._file_pos)
                chunk = f.read()
                self._file_pos = f.tell()
        if not chunk:
            return
        chunk = f"{self._remainder}{chunk}"
        if chunk and not chunk.endswith("\n"):
            chunk, self._remainder = chunk.rsplit("\n", 1)
        else:
            self._remainder = ""
        for line in chunk.splitlines():
            line = line.strip()
            if not line:
                continue
            if not self._header_skipped:
                if line.lower().startswith("volume_idx"):
                    self._header_skipped = True
                    continue
                self._header_skipped = True
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) < 3:
                continue
            try:
                vol_idx = int(parts[0])
            except ValueError:
                continue
            try:
                reg_vals = [float(v) for v in parts[2:]]
            except ValueError:
                continue
            reg = np.asarray(reg_vals, dtype=np.float32)
            if reg.ndim != 1:
                continue
            if self._n_reg is None:
                self._n_reg = reg.shape[0]
            self._regressors_by_vol[vol_idx] = reg

    def wait_for_volume(self, vol_idx: int, timeout: Optional[float] = None) -> bool:
        wait_time = self.timeout if timeout is None else timeout
        deadline = time.monotonic() + max(0.0, wait_time)
        while time.monotonic() < deadline:
            self._update_from_file()
            if vol_idx in self._regressors_by_vol:
                return True
            time.sleep(self.poll_interval)
        return vol_idx in self._regressors_by_vol

    def get_retrots(self, TR: float, vol_idx: int, tshift: float, timeout: Optional[float] = None):
        self.wait_for_volume(vol_idx, timeout=timeout)
        self._update_from_file()
        n_reg = self._ensure_regressor_count()
        retro = np.zeros((vol_idx, n_reg), dtype=np.float32)
        for idx, reg in self._regressors_by_vol.items():
            if 1 <= idx <= vol_idx:
                retro[idx - 1, : reg.shape[0]] = reg
        if vol_idx not in self._regressors_by_vol:
            if vol_idx not in self._missing_vols:
                log.warning(
                    "[BIOPAC] Missing physio regressors for vol %s in file; using zeros.",
                    vol_idx,
                )
            self._missing_vols.add(vol_idx)
        return retro

    def was_missing(self, vol_idx: int) -> bool:
        with self._lock:
            return vol_idx in self._missing_vols
