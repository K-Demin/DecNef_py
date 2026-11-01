"""The live viewer must not stall acquisition or pair different TRs."""

from pathlib import Path
import queue

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pytest

from volume_streamer import (
    VolumeStreamerConfig,
    VolumeStreamerHandle,
    _decode_message,
    _draw_volume,
    _load_display_volume,
)


@pytest.fixture
def display():
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
    yield fig, axes
    plt.close(fig)


def test_publish_replaces_pending_pair_and_preserves_legacy_call(monkeypatch):
    handle = VolumeStreamerHandle(VolumeStreamerConfig(enabled=True))
    handle._queue = queue.Queue(maxsize=2)
    handle._queue.put_nowait((1, "old_primary.nii", "old_t1.nii"))

    def no_io(*args, **kwargs):
        raise AssertionError("Image IO must remain in the child process")

    monkeypatch.setattr(nib, "load", no_io)
    handle.publish(2, Path("primary.nii"), Path("t1.nii"))
    assert handle._queue.get_nowait() == (2, "primary.nii", "t1.nii")
    assert handle._queue.empty()
    handle.publish(3, Path("legacy.nii"))
    assert handle._queue.get_nowait() == (3, "legacy.nii", None)
    # Missing stages still replace an old frame with an explicit unavailable TR.
    handle.publish(4, None)
    assert handle._queue.get_nowait() == (4, None, None)


def test_publish_throttles_and_drops_full_queue_without_blocking(caplog):
    class FullQueue:
        def get_nowait(self):
            raise queue.Empty

        def put_nowait(self, message):
            raise queue.Full

        def put(self, *args, **kwargs):
            pytest.fail("Publishing must never use blocking queue operations")

    handle = VolumeStreamerHandle(VolumeStreamerConfig(enabled=True, every_n=2))
    handle._queue = queue.Queue(maxsize=1)
    handle.publish(1, Path("skip.nii"), Path("skip_t1.nii"))
    assert handle._queue.empty()
    handle.publish("invalid", Path("bad.nii"))
    assert "could not publish" in caplog.text
    assert handle._queue.empty()
    handle._queue = FullQueue()
    handle.publish(2, Path("drop.nii"), Path("drop_t1.nii"))
    handle.config.enabled = False
    handle.publish(4, Path("disabled.nii"))


@pytest.mark.parametrize("message", [
    None, (1,), (1, "a", "b", "c"), (-1, "a"), (True, "a"),
    (1.5, "a"), ("1", "a"), (1, ""), (1, "a", 42),
])
def test_decode_rejects_malformed_payloads(message):
    with pytest.raises(ValueError):
        _decode_message(message)


def test_decode_accepts_legacy_and_optional_t1_payloads():
    assert _decode_message((np.int64(7), Path("epi.nii"))) == (7, "epi.nii", None)
    assert _decode_message((8, None, "t1.nii")) == (8, None, "t1.nii")


def test_loader_canonicalizes_orientation_without_resampling(tmp_path):
    original = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    # LAS storage: x decreases physically, so the display must reverse x.
    affine = np.diag([-2.0, 3.0, 4.0, 1.0])
    path = tmp_path / "las.nii.gz"
    nib.save(nib.Nifti1Image(original[..., None], affine), path)
    displayed, voxel_sizes, error = _load_display_volume(str(path))
    np.testing.assert_array_equal(displayed, original[::-1])
    assert tuple(voxel_sizes) == (2.0, 3.0, 4.0)
    assert error is None


def test_loader_rejects_timeseries_and_reports_unreadable_stage(tmp_path):
    path = tmp_path / "series.nii"
    nib.save(nib.Nifti1Image(np.ones((3, 4, 5, 2)), np.eye(4)), path)
    data, sizes, error = _load_display_volume(str(path))
    assert data is None and sizes is None and error
    assert _load_display_volume(str(tmp_path / "missing.nii"))[0] is None
    assert _load_display_volume(None) == (None, None, None)


def test_rows_use_independent_grid_slices_and_grayscale(display):
    fig, axes = display
    primary = np.arange(1, 61, dtype=np.float32).reshape(3, 4, 5)
    t1 = np.zeros((10, 12, 14), dtype=np.float32)
    t1[2:6, 3:9, 9:13] = np.linspace(1000, 2000, 96).reshape(4, 6, 4)
    config = VolumeStreamerConfig(kind="raw")
    _draw_volume(fig, axes, primary, 12, "raw.nii", config,
                 voxel_sizes=(2, 3, 4), t1_data=t1,
                 t1_voxel_sizes=(1, 1, 1), t1_label="t1.nii")
    assert axes[0, 0].images[0].get_array().shape == (5, 4)
    assert axes[1, 0].images[0].get_array().shape == (14, 12)
    assert axes[0, 0].images[0].get_extent() == [0, 12, 0, 20]
    assert axes[1, 0].images[0].get_extent() == [0, 12, 0, 14]
    assert "z = 10" in axes[1, 2].get_title()  # Center of actual slab, not empty T1 center.
    np.testing.assert_allclose(axes[0, 0].images[0].get_clim(), np.percentile(primary, [2, 98]))
    np.testing.assert_allclose(axes[1, 0].images[0].get_clim(), np.percentile(t1[t1 != 0], [2, 98]))
    assert any("T1 — before smoothing" in text.get_text() for text in fig.texts)
    assert {text.get_text() for text in axes[1, 2].texts} == {"L", "R", "A", "P"}


def test_unavailable_stage_clears_previous_volume(display):
    fig, axes = display
    config = VolumeStreamerConfig()
    data = np.ones((5, 6, 7), dtype=np.float32)
    _draw_volume(fig, axes, data, 1, "first.nii", config, t1_data=data)
    text_count = len(fig.texts)
    _draw_volume(fig, axes, data * 2, 2, "second.nii", config)
    assert all(len(ax.images) == 0 for ax in axes[1])
    assert "Unavailable for volume 00002" in axes[1, 1].texts[0].get_text()
    assert "No T1 stage" in axes[1, 1].texts[0].get_text()
    assert len(fig.texts) == text_count
    # One failed row must not prevent the other current-volume row from updating.
    _draw_volume(fig, axes, None, 3, "missing.nii", config,
                 primary_error="File unavailable or unreadable", t1_data=data * 3)
    assert all(len(ax.images) == 0 for ax in axes[0])
    assert all(len(ax.images) == 1 for ax in axes[1])
    assert "Unavailable for volume 00003" in axes[0, 1].texts[0].get_text()


@pytest.mark.parametrize("fill", [0.0, np.nan])
def test_empty_signal_renders_with_finite_contrast(display, fill):
    fig, axes = display
    _draw_volume(fig, axes, np.full((2, 3, 4), fill), 1, "empty.nii", VolumeStreamerConfig())
    limits = axes[0, 0].images[0].get_clim()
    assert all(np.isfinite(limits)) and limits[0] < limits[1]
