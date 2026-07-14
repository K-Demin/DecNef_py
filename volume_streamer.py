"""Lightweight live NIfTI volume QC viewer for real-time runs.

The streamer is intentionally isolated from the analytic pipeline: the pipeline only
sends file paths over a small multiprocessing queue and the viewer process loads
and renders those paths on a best-effort basis. If the display backend or an
individual NIfTI load fails, the run continues unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import multiprocessing as mp
from pathlib import Path
import queue
import time
from typing import Literal, Optional

import numpy as np


VolumeStreamKind = Literal["raw", "mc", "unwarped", "reg", "score_input"]


@dataclass
class VolumeStreamerConfig:
    """Runtime options for the live QC volume viewer."""

    enabled: bool = False
    kind: VolumeStreamKind = "unwarped"
    every_n: int = 1
    max_queue: int = 2
    window_title: str = "RT volume QC"
    cmap: str = "gray"
    percentile_low: float = 2.0
    percentile_high: float = 98.0
    poll_interval_s: float = 0.05

    def __post_init__(self) -> None:
        self.kind = str(self.kind).lower()  # type: ignore[assignment]
        allowed = {"raw", "mc", "unwarped", "reg", "score_input"}
        if self.kind not in allowed:
            raise ValueError(f"Unsupported volume stream kind {self.kind!r}; choose one of {sorted(allowed)}")
        self.every_n = max(1, int(self.every_n))
        self.max_queue = max(1, int(self.max_queue))
        self.poll_interval_s = max(0.01, float(self.poll_interval_s))


class VolumeStreamerHandle:
    """Small parent-process handle used by the real-time pipeline."""

    def __init__(self, config: VolumeStreamerConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.log = logger or logging.getLogger(__name__)
        self._queue: Optional[mp.Queue] = None
        self._stop_event: Optional[mp.Event] = None
        self._process: Optional[mp.Process] = None

    def start(self) -> None:
        if not self.config.enabled or self._process is not None:
            return
        ctx = mp.get_context("spawn")
        self._queue = ctx.Queue(maxsize=self.config.max_queue)
        self._stop_event = ctx.Event()
        self._process = ctx.Process(
            target=_viewer_main,
            args=(self.config, self._queue, self._stop_event),
            daemon=True,
        )
        self._process.start()
        self.log.info("[VOLQC] started %s streamer (pid=%s)", self.config.kind, self._process.pid)

    def publish(self, volume_idx: int, path: Path | None) -> None:
        if not self.config.enabled or self._queue is None or path is None:
            return
        if volume_idx % self.config.every_n != 0:
            return
        try:
            # Keep the parent non-blocking. Drop stale frames before publishing
            # the newest one so the GUI does not lag behind real time.
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._queue.put_nowait((int(volume_idx), str(path)))
        except queue.Full:
            self.log.debug("[VOLQC] dropping volume %05d because viewer queue is full", volume_idx)
        except Exception as exc:
            self.log.warning("[VOLQC] could not publish volume %05d: %s", volume_idx, exc)

    def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._process is not None:
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
            self.log.info("[VOLQC] stopped streamer")
        self._process = None
        self._queue = None
        self._stop_event = None


def _viewer_main(config: VolumeStreamerConfig, q: mp.Queue, stop_event: mp.Event) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    log = logging.getLogger("volume_streamer")
    try:
        import nibabel as nib
        import matplotlib.pyplot as plt
    except Exception as exc:  # GUI/QC dependency failure must not affect RT analysis.
        log.warning("[VOLQC] viewer disabled; could not import GUI dependencies: %s", exc)
        return

    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(6, 2.2), num=config.window_title)
    try:
        fig.canvas.manager.set_window_title(config.window_title)
    except Exception:
        pass
    fig.tight_layout()

    last_path = None
    while not stop_event.is_set():
        try:
            volume_idx, path_text = q.get(timeout=config.poll_interval_s)
        except queue.Empty:
            plt.pause(config.poll_interval_s)
            continue
        path = Path(path_text)
        if path == last_path or not path.exists():
            continue
        last_path = path
        try:
            data = np.asanyarray(nib.load(str(path)).dataobj, dtype=np.float32)
            if data.ndim > 3:
                data = np.squeeze(data)
            if data.ndim != 3:
                raise ValueError(f"expected 3D volume, got shape {data.shape}")
            _draw_volume(fig, axes, data, volume_idx, path.name, config)
        except Exception as exc:
            log.warning("[VOLQC] could not render %s: %s", path, exc)

    plt.close(fig)


def _draw_volume(fig, axes, data: np.ndarray, volume_idx: int, label: str, config: VolumeStreamerConfig) -> None:
    finite = data[np.isfinite(data)]
    if finite.size:
        vmin, vmax = np.percentile(finite, [config.percentile_low, config.percentile_high])
        if vmin == vmax:
            vmin, vmax = float(finite.min()), float(finite.max())
    else:
        vmin, vmax = 0.0, 1.0

    mids = [s // 2 for s in data.shape]
    slices = [
        np.rot90(data[mids[0], :, :]),
        np.rot90(data[:, mids[1], :]),
        np.rot90(data[:, :, mids[2]]),
    ]
    titles = ["sag", "cor", "axi"]
    for ax, slc, title in zip(axes, slices, titles):
        ax.clear()
        ax.imshow(slc, cmap=config.cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.suptitle(f"{config.kind} vol {volume_idx:05d}: {label}", fontsize=9)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    time.sleep(0.001)
