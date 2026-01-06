"""BIOPAC receiver test harness.

Runs a local BiopacRetroTSReceiver and optionally sends simulated RetroTS
messages so the receiver can be validated without the full RT pipeline.

Example:
    python -m biopac_rt.biopac_receiver_tester --volumes 5
"""

import argparse
import json
import logging
import socket
import threading
import time
from typing import Iterable, Optional

from biopac_rt.biopac_receiver import BiopacReceiverConfig, BiopacRetroTSReceiver


log = logging.getLogger("biopac_receiver_tester")


def _iter_regressors(vol_idx: int, n_regressors: int) -> Iterable[float]:
    base = float(vol_idx)
    return [base + (idx / 10.0) for idx in range(n_regressors)]


def _send_simulated(
    host: str,
    port: int,
    n_vols: int,
    n_regressors: int,
    tr: float,
    connect_timeout: float = 2.0,
) -> None:
    payloads = [
        {
            "kind": "retrots",
            "volume_idx": vol_idx,
            "n_regressors": n_regressors,
            "regressors": list(_iter_regressors(vol_idx, n_regressors)),
            "timestamp": time.time(),
            "tr": tr,
        }
        for vol_idx in range(1, n_vols + 1)
    ]
    with socket.create_connection((host, port), timeout=connect_timeout) as sock:
        sock.settimeout(0.2)
        for payload in payloads:
            try:
                sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
            except OSError as exc:
                log.error("Failed sending simulated payload: %s", exc)
                break
            time.sleep(tr)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="BIOPAC receiver test harness",
    )
    parser.add_argument("--host", default="115.145.189.30")
    parser.add_argument("--port", type=int, default=15000)
    parser.add_argument("--timeout", type=float, default=0.3)
    parser.add_argument("--expected-regressors", type=int, default=None)
    parser.add_argument("--tr", type=float, default=0.9)
    parser.add_argument("--volumes", type=int, default=5)
    parser.add_argument(
        "--send-sim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send simulated RetroTS messages to the receiver.",
    )
    parser.add_argument("--poll", type=float, default=0.002)
    parser.add_argument("--max-wait", type=float, default=None,
                        help="Max seconds to wait per volume. Default: wait forever.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = BiopacReceiverConfig(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        expected_regressors=args.expected_regressors,
    )
    receiver = BiopacRetroTSReceiver(config)
    receiver.start()

    sender_thread = None
    if args.send_sim:
        sender_thread = threading.Thread(
            target=_send_simulated,
            args=(
                args.host,
                args.port,
                args.volumes,
                config.expected_regressors or 8,
                args.tr,
            ),
            daemon=True,
        )
        sender_thread.start()

    missing_count = 0
    for vol_idx in range(1, args.volumes + 1):
        waited_s, missing = wait_for_volume(
            receiver=receiver,
            tr=args.tr,
            vol_idx=vol_idx,
            default_fill=0.0,
            poll_s=0.002,  # tighter polling for RT; adjust if you want less CPU
            max_wait_s=None,  # None = wait indefinitely
        )

        if missing:
            missing_count += 1
            log.info("Volume %s STILL missing after waiting %.3fs (zero-filled).", vol_idx, waited_s)
        else:
            log.info("Volume %s received regressors after waiting %.3fs.", vol_idx, waited_s)

    receiver.stop()
    if sender_thread is not None:
        sender_thread.join(timeout=1.0)

    log.info("Test complete. Missing volumes: %s/%s", missing_count, args.volumes)
    return 0 if missing_count == 0 else 1

def wait_for_volume(
    receiver: BiopacRetroTSReceiver,
    tr: float,
    vol_idx: int,
    default_fill: float = 0.0,
    poll_s: float = 0.005,
    max_wait_s: Optional[float] = None,  # None = wait forever
) -> tuple[float, bool]:
    """
    Block until regressors for `vol_idx` are available, then fetch them.

    Returns:
        waited_s: seconds spent waiting
        missing:  True if receiver still returned missing (should be False if we truly waited)
                 (kept as a sanity check in case receiver logic changes)
    """
    t0 = time.time()
    while True:
        # If your receiver exposes a "has(vol_idx)" or similar, use it here.
        # Otherwise we poll by attempting to fetch but *not* accepting missing results:
        receiver.get_retrots(tr, vol_idx, default_fill)
        missing = receiver.was_missing(vol_idx)

        if not missing:
            return (time.time() - t0), False

        # Not available yet → keep waiting
        if max_wait_s is not None and (time.time() - t0) >= max_wait_s:
            # Give up: last call already produced a missing/zero-filled result
            return (time.time() - t0), True

        time.sleep(poll_s)


if __name__ == "__main__":
    raise SystemExit(main())
