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
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


log = logging.getLogger("biopac_receiver")


@dataclass
class BiopacReceiverConfig:
    host: str = "0.0.0.0"
    port: int = 15000
    timeout: float = 0.3
    expected_regressors: Optional[int] = None
    handshake_tr: Optional[float] = None


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

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        log.info("BIOPAC receiver listening on %s:%s", self.config.host, self.config.port)

    def stop(self):
        self._stop.set()
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
