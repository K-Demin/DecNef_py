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

    def publish(
        self,
        volume_idx: int,
        primary_path: Path | None,
        t1_path: Path | None = None,
    ) -> None:
        """Send same-volume paths atomically; all image IO stays in the viewer."""
        if not self.config.enabled or self._queue is None:
            return

        try:
            message = _decode_message((volume_idx, primary_path, t1_path))
            if message[0] % self.config.every_n != 0:
                return
            # Keep only the newest pending frame.
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._queue.put_nowait(message)
        except queue.Full:
            self.log.debug(
                "[VOLQC] dropping volume %05d because viewer queue is full",
                volume_idx,
            )
        except Exception as exc:
            self.log.warning(
                "[VOLQC] could not publish volume %s: %s",
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


def _decode_message(message) -> tuple[int, str | None, str | None]:
    """Validate the small path payload, including legacy single-row messages."""
    if not isinstance(message, (tuple, list)) or len(message) not in (2, 3):
        raise ValueError("Expected (volume_index, primary_path[, t1_path]).")
    volume_idx, primary_path = message[:2]
    t1_path = message[2] if len(message) == 3 else None
    if (isinstance(volume_idx, bool)
            or not isinstance(volume_idx, (int, np.integer))
            or volume_idx < 0):
        raise ValueError("Volume index must be a nonnegative integer.")

    def checked_path(value):
        if value is None:
            return None
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError("Volume paths must be nonempty strings or Path objects.")
        return str(value)

    return int(volume_idx), checked_path(primary_path), checked_path(t1_path)


def _load_display_volume(path_text: str | None):
    """Reorient to canonical RAS without resampling; preserve each row's grid."""
    if path_text is None:
        return None, None, None
    try:
        import nibabel as nib

        img = nib.as_closest_canonical(nib.load(path_text))
        data = np.asanyarray(img.dataobj, dtype=np.float32)
        if data.ndim > 3 and all(size == 1 for size in data.shape[3:]):
            data = data.reshape(data.shape[:3])
        if data.ndim != 3 or any(size == 0 for size in data.shape):
            raise ValueError(f"expected a 3D volume, got {data.shape}")
        return data, img.header.get_zooms()[:3], None
    except Exception as exc:
        logging.getLogger("volume_streamer").warning("[VOLQC] could not load %s: %s", path_text, exc)
        return None, None, "File unavailable or unreadable"


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
            2,
            3,
            figsize=(10, 6.5),
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

        while not stop_event.is_set():
            # Closing the window manually should end only the viewer process.
            if not plt.fignum_exists(fig.number):
                log.info("[VOLQC] viewer window was closed")
                return

            try:
                message = q.get(timeout=config.poll_interval_s)
            except queue.Empty:
                plt.pause(config.poll_interval_s)
                continue

            try:
                volume_idx, primary_path, t1_path = _decode_message(message)
                data, voxel_sizes, primary_error = _load_display_volume(primary_path)
                t1_data, t1_voxel_sizes, t1_error = _load_display_volume(t1_path)
                _draw_volume(
                    fig, axes, data, volume_idx,
                    Path(primary_path).name if primary_path else "Unavailable",
                    config, voxel_sizes=voxel_sizes, primary_error=primary_error,
                    t1_data=t1_data, t1_voxel_sizes=t1_voxel_sizes,
                    t1_label=Path(t1_path).name if t1_path else None,
                    t1_error=t1_error,
                )
            except Exception as exc:
                log.warning("[VOLQC] could not render viewer message: %s", exc)
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
    data: np.ndarray | None,
    volume_idx: int,
    label: str,
    config: VolumeStreamerConfig,
    *,
    voxel_sizes=None,
    primary_error: str | None = None,
    t1_data: np.ndarray | None = None,
    t1_voxel_sizes=None,
    t1_label: str | None = None,
    t1_error: str | None = None,
) -> None:
    """Draw both rows anew so an unavailable stage cannot retain an older image."""
    axes = np.asarray(axes).reshape(2, 3)
    row_labels = getattr(fig, "_volqc_row_labels", None)
    if row_labels is None:
        row_labels = [fig.text(.04, .91, "", fontsize=11, weight="bold"),
                      fig.text(.04, .455, "", fontsize=11, weight="bold")]
        fig._volqc_row_labels = row_labels
    row_labels[0].set_text(f"{config.kind} — {label}")
    row_labels[1].set_text("T1 — before smoothing" + (f"  |  {t1_label}" if t1_label else ""))
    _draw_row(axes[0], data, voxel_sizes, config, volume_idx,
              primary_error or "Selected stream unavailable")
    _draw_row(axes[1], t1_data, t1_voxel_sizes, config, volume_idx,
              t1_error or "No T1 stage in this processing path")
    fig.suptitle(f"{config.window_title}  |  Volume {volume_idx:05d}", fontsize=12)
    fig.subplots_adjust(left=.04, right=.98, top=.85, bottom=.04, hspace=.53, wspace=.08)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def _draw_row(axes, data, voxel_sizes, config, volume_idx, unavailable_reason):
    for ax in axes:
        ax.clear()
        ax.axis("off")
    if data is None:
        axes[1].text(.5, .5, f"Unavailable for volume {volume_idx:05d}\n{unavailable_reason}",
                     ha="center", va="center", transform=axes[1].transAxes,
                     fontsize=10, color="0.35", linespacing=1.6)
        return

    finite = data[np.isfinite(data)]
    # Sparse T1 grids can contain mostly background. Do not let those zeros
    # collapse the contrast range of the acquired slab.
    contrast_values = finite[finite != 0]
    if not contrast_values.size:
        contrast_values = finite
    if contrast_values.size:
        vmin, vmax = np.percentile(
            contrast_values,
            [config.percentile_low, config.percentile_high],
        )
        if vmin == vmax:
            vmin, vmax = float(finite.min()), float(finite.max())
        if vmin == vmax:
            vmin, vmax = float(vmin) - .5, float(vmax) + .5
    else:
        vmin, vmax = 0.0, 1.0

    mids = [size // 2 for size in data.shape]
    # In a full T1 grid the geometric midpoint can miss a partial EPI slab.
    # Select orthogonal centers of the displayed signal's bounding box.
    support = np.isfinite(data) & (data != 0)
    if support.any():
        for axis in range(3):
            occupied = np.flatnonzero(support.any(axis=tuple(i for i in range(3) if i != axis)))
            mids[axis] = int((occupied[0] + occupied[-1]) // 2)
    sizes = voxel_sizes if voxel_sizes is not None else (1., 1., 1.)
    slices = [
        np.rot90(data[mids[0], :, :]),
        np.rot90(data[:, mids[1], :]),
        np.rot90(data[:, :, mids[2]]),
    ]

    planes = [("Sagittal", "x", "P", "A", "S", "I", 1, 2),
              ("Coronal", "y", "L", "R", "S", "I", 0, 2),
              ("Axial", "z", "L", "R", "A", "P", 0, 1)]
    for ax, slc, index, plane in zip(axes, slices, mids, planes):
        title, coordinate, left, right, top, bottom, horizontal, vertical = plane
        ax.imshow(
            slc,
            cmap=config.cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            extent=(0, data.shape[horizontal] * sizes[horizontal],
                    0, data.shape[vertical] * sizes[vertical]),
            aspect="equal",
        )
        ax.set_title(f"{title}  |  {coordinate} = {index}", fontsize=9)
        for text, x, y in ((left, .025, .5), (right, .975, .5), (top, .5, .97), (bottom, .5, .03)):
            ax.text(x, y, text, ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="white", bbox=dict(facecolor="black", edgecolor="none", alpha=.45, pad=1))
