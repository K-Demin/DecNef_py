"""Lightweight live NIfTI volume QC viewer for real-time runs.

The streamer is intentionally isolated from the analytic pipeline: the pipeline
only sends file paths over a small multiprocessing queue and the viewer process
loads and renders those paths on a best-effort basis. Viewer failures do not
stop the realtime analysis.
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
    window_x: int = 50
    window_y: int = 50
    cmap: str = "gray"
    percentile_low: float = 2.0
    percentile_high: float = 98.0
    poll_interval_s: float = 0.05

    def __post_init__(self) -> None:
        self.kind = str(self.kind).lower()  # type: ignore[assignment]
        allowed = {"raw", "mc", "unwarped", "reg", "score_input"}
        if self.kind not in allowed:
            raise ValueError(
                f"Unsupported volume stream kind {self.kind!r}; "
                f"choose one of {sorted(allowed)}"
            )
        self.every_n = max(1, int(self.every_n))
        self.max_queue = max(1, int(self.max_queue))
        self.window_x = int(self.window_x)
        self.window_y = int(self.window_y)
        self.poll_interval_s = max(0.01, float(self.poll_interval_s))


def _move_figure_window(fig, x: int, y: int) -> bool:
    """Move a Matplotlib GUI window to global desktop coordinates."""

    log = logging.getLogger("volume_streamer")
    manager = getattr(fig.canvas, "manager", None)
    window = getattr(manager, "window", None)

    if window is None:
        log.warning(
            "[VOLQC] backend %s does not expose a movable native window",
            type(fig.canvas).__name__,
        )
        return False

    try:
        # QtAgg / Qt5Agg / Qt6Agg
        if hasattr(window, "move"):
            window.move(int(x), int(y))
            return True

        # TkAgg
        if hasattr(window, "wm_geometry"):
            window.wm_geometry(f"+{int(x)}+{int(y)}")
            return True

        # WXAgg
        if hasattr(window, "SetPosition"):
            window.SetPosition((int(x), int(y)))
            return True

        log.warning(
            "[VOLQC] unsupported Matplotlib window object %s",
            type(window).__name__,
        )
    except Exception as exc:
        log.warning(
            "[VOLQC] could not move viewer window to (%d, %d): %s",
            x,
            y,
            exc,
        )
    return False


def _disable_matplotlib_keymaps(matplotlib_module) -> None:
    """Disable Matplotlib shortcuts such as scanner trigger 's' -> Save."""
    for name in tuple(matplotlib_module.rcParams):
        if name.startswith("keymap."):
            matplotlib_module.rcParams[name] = []


def _make_figure_non_focusable(fig) -> bool:
    """Best-effort: show the QC window without stealing keyboard focus."""
    log = logging.getLogger("volume_streamer")
    manager = getattr(fig.canvas, "manager", None)
    window = getattr(manager, "window", None)

    if window is None:
        return False

    try:
        # QtAgg / Qt5Agg / Qt6Agg
        if hasattr(window, "setWindowFlag"):
            from matplotlib.backends.qt_compat import QtCore

            qt = QtCore.Qt
            window_type = getattr(qt, "WindowType", qt)
            focus_policy = getattr(qt, "FocusPolicy", qt)
            widget_attribute = getattr(qt, "WidgetAttribute", qt)

            no_focus_flag = getattr(
                window_type,
                "WindowDoesNotAcceptFocus",
                None,
            )
            show_without_activation = getattr(
                widget_attribute,
                "WA_ShowWithoutActivating",
                None,
            )
            no_focus_policy = getattr(
                focus_policy,
                "NoFocus",
                None,
            )

            if no_focus_flag is not None:
                window.setWindowFlag(no_focus_flag, True)
            if show_without_activation is not None and hasattr(window, "setAttribute"):
                window.setAttribute(show_without_activation, True)
            if no_focus_policy is not None and hasattr(window, "setFocusPolicy"):
                window.setFocusPolicy(no_focus_policy)

            canvas_widget = getattr(fig.canvas, "setFocusPolicy", None)
            if canvas_widget is not None and no_focus_policy is not None:
                fig.canvas.setFocusPolicy(no_focus_policy)

            return True

        # TkAgg: disable traversal/focus requests. This is weaker than Qt's
        # WindowDoesNotAcceptFocus, but keymaps are already disabled above.
        if hasattr(window, "configure"):
            try:
                window.configure(takefocus=False)
            except Exception:
                pass
            return True

    except Exception as exc:
        log.warning("[VOLQC] could not make viewer non-focusable: %s", exc)

    return False


class VolumeStreamerHandle:
    """Small parent-process handle used by the realtime pipeline."""

    def __init__(
        self,
        config: VolumeStreamerConfig,
        logger: Optional[logging.Logger] = None,
    ):
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
            name="DecNefVolumeStreamer",
        )
        self._process.start()
        self.log.info(
            "[VOLQC] started %s streamer (pid=%s, position=%d,%d)",
            self.config.kind,
            self._process.pid,
            self.config.window_x,
            self.config.window_y,
        )

    def publish(self, volume_idx: int, path: Path | None) -> None:
        if not self.config.enabled or self._queue is None or path is None:
            return
        if volume_idx % self.config.every_n != 0:
            return

        try:
            # Keep only the newest pending frame.
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._queue.put_nowait((int(volume_idx), str(path)))
        except queue.Full:
            self.log.debug(
                "[VOLQC] dropping volume %05d because viewer queue is full",
                volume_idx,
            )
        except Exception as exc:
            self.log.warning(
                "[VOLQC] could not publish volume %05d: %s",
                volume_idx,
                exc,
            )

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


def _viewer_main(
    config: VolumeStreamerConfig,
    q: mp.Queue,
    stop_event: mp.Event,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    log = logging.getLogger("volume_streamer")

    try:
        import nibabel as nib
        import matplotlib

        _disable_matplotlib_keymaps(matplotlib)
        matplotlib.rcParams["figure.raise_window"] = False

        import matplotlib.pyplot as plt
    except Exception as exc:
        log.warning(
            "[VOLQC] viewer disabled; could not import GUI dependencies: %s",
            exc,
        )
        return

    try:
        log.info("[VOLQC] Matplotlib backend: %s", matplotlib.get_backend())

        plt.ion()
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(9, 3),
            num=config.window_title,
        )

        try:
            fig.canvas.manager.set_window_title(config.window_title)
        except Exception:
            pass

        # Prevent scanner key triggers from activating Matplotlib actions,
        # and avoid stealing focus from the PsychoPy experiment where supported.
        non_focusable = _make_figure_non_focusable(fig)

        # Create/map the native window before asking the window manager to move it.
        plt.show(block=False)
        fig.canvas.draw()
        plt.pause(0.10)
        moved = _move_figure_window(fig, config.window_x, config.window_y)
        plt.pause(0.05)

        log.info(
            "[VOLQC] viewer ready at (%d, %d), moved=%s, non_focusable=%s",
            config.window_x,
            config.window_y,
            moved,
            non_focusable,
        )

        fig.tight_layout()

        last_path: Optional[Path] = None
        while not stop_event.is_set():
            # Closing the window manually should end only the viewer process.
            if not plt.fignum_exists(fig.number):
                log.info("[VOLQC] viewer window was closed")
                return

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
                data = np.asanyarray(
                    nib.load(str(path)).dataobj,
                    dtype=np.float32,
                )
                if data.ndim > 3:
                    data = np.squeeze(data)
                if data.ndim != 3:
                    raise ValueError(
                        f"expected 3D volume, got shape {data.shape}"
                    )
                _draw_volume(
                    fig,
                    axes,
                    data,
                    volume_idx,
                    path.name,
                    config,
                )
            except Exception as exc:
                log.warning(
                    "[VOLQC] could not render %s: %s",
                    path,
                    exc,
                )
    except Exception:
        # Make child-process failures visible in the parent terminal.
        log.exception("[VOLQC] viewer process crashed")
    finally:
        try:
            plt.close("all")
        except Exception:
            pass


def _draw_volume(
    fig,
    axes,
    data: np.ndarray,
    volume_idx: int,
    label: str,
    config: VolumeStreamerConfig,
) -> None:
    finite = data[np.isfinite(data)]
    if finite.size:
        vmin, vmax = np.percentile(
            finite,
            [config.percentile_low, config.percentile_high],
        )
        if vmin == vmax:
            vmin, vmax = float(finite.min()), float(finite.max())
    else:
        vmin, vmax = 0.0, 1.0

    mids = [size // 2 for size in data.shape]
    slices = [
        np.rot90(data[mids[0], :, :]),
        np.rot90(data[:, mids[1], :]),
        np.rot90(data[:, :, mids[2]]),
    ]

    for ax, slc, title in zip(axes, slices, ["sag", "cor", "axi"]):
        ax.clear()
        ax.imshow(
            slc,
            cmap=config.cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    fig.suptitle(
        f"{config.kind} vol {volume_idx:05d}: {label}",
        fontsize=9,
    )
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    time.sleep(0.001)
