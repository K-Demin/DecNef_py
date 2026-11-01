"""Timing semantics, independent of scanner/ANTs/CUDA availability."""
import csv
import subprocess
import sys

import numpy as np
import pytest

from rt_speed import FIELDS, SpeedRecorder, STAGES, median_ci, plot_speed, select_profile_rows


class Clock:
    now = 0.

    def __call__(self):
        return self.now


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_separates_wait_processing_latency_and_live_cadence(tmp_path):
    clock = Clock()
    speed = SpeedRecorder(tmp_path / "speed.csv", clock)
    speed.arrive(1)
    clock.now = .1
    speed.arrive(1)  # duplicate file-close notification cannot reset detection
    clock.now = .2
    started = speed.conversion_start(1)
    clock.now = .5
    speed.conversion_end(1, started)
    clock.now = .6
    started = speed.process_start(1)
    speed.stage(1, "REG", .1)
    clock.now = .9
    speed.process_end(1, started)
    speed.readiness(1, True)
    speed.finish(1, True)
    speed.finish(1, True)  # one durable row per completed volume
    row = read_rows(speed.path)[0]
    assert float(row["latency_ms"]) == pytest.approx(900)
    assert float(row["processing_ms"]) == pytest.approx(600)
    assert float(row["worker_queue_ms"]) == pytest.approx(200)
    assert float(row["order_queue_ms"]) == pytest.approx(100)
    assert sum(float(row[f"{s}_ms"]) for s in STAGES) == pytest.approx(600)
    clock.now = 1.2
    speed.arrive(2)
    assert speed.rows[2]["arrival_interval_ms"] == pytest.approx(1200)
    assert speed.rows[2]["idle_wait_ms"] == pytest.approx(300)
    clock.now = 1.3
    speed.arrive(3)  # pending volume 2 means this is not idle pipeline time
    assert speed.rows[3]["idle_wait_ms"] == 0
    assert len(read_rows(speed.path)) == 1


def test_backlog_cached_inputs_and_retries_are_explicit(tmp_path):
    clock = Clock()
    speed = SpeedRecorder(tmp_path / "speed.csv", clock)
    speed.arrive(1, backlog=True)
    speed.arrive(1)  # a later watcher notification preserves backlog provenance
    started = speed.conversion_start(1)
    clock.now = .1
    speed.conversion_end(1, started)
    clock.now = .3  # retry backoff is latency, not active conversion time
    started = speed.conversion_start(1, cached=True)
    clock.now = .4
    speed.conversion_end(1, started)
    speed.finish(1, False)
    row = read_rows(speed.path)[0]
    assert row["source"] == "backlog"
    assert row["arrival_interval_ms"] == row["idle_wait_ms"] == ""
    assert row["cached_input"] == "True"
    assert row["attempts"] == "2"
    assert row["status"] == "failed"
    assert float(row["processing_ms"]) == pytest.approx(200)
    assert float(row["latency_ms"]) == pytest.approx(400)
    speed.arrive(2)
    assert speed.rows[2]["arrival_interval_ms"] == ""


def test_out_of_order_conversion_records_wait_for_ordered_processing(tmp_path):
    clock = Clock()
    speed = SpeedRecorder(tmp_path / "speed.csv", clock)
    speed.arrive(1, backlog=True)
    speed.arrive(2, backlog=True)
    start1 = speed.conversion_start(1)
    start2 = speed.conversion_start(2)
    clock.now = .1
    speed.conversion_end(2, start2)
    clock.now = .3
    speed.conversion_end(1, start1)
    started = speed.process_start(1)
    clock.now = .4
    speed.process_end(1, started)
    speed.finish(1, True)
    started = speed.process_start(2)
    clock.now = .5
    speed.process_end(2, started)
    speed.finish(2, True)
    first, second = read_rows(speed.path)
    assert float(first["order_queue_ms"]) == 0
    assert float(second["order_queue_ms"]) == pytest.approx(300)
    assert float(second["processing_ms"]) == pytest.approx(200)
    assert float(second["latency_ms"]) == pytest.approx(500)


def test_plot_handles_warmup_only_and_no_data(tmp_path):
    clock = Clock()
    (tmp_path / "speed.png").write_bytes(b"stale previous run")
    speed = SpeedRecorder(tmp_path / "speed.csv", clock)
    assert not speed.path.with_suffix(".png").exists()
    assert plot_speed(speed.path) is None
    speed.arrive(1, backlog=True)
    started = speed.conversion_start(1, cached=True)
    clock.now = .1
    speed.conversion_end(1, started)
    speed.readiness(1, False)
    speed.finish(1, True)
    result = plot_speed(speed.path)
    assert result.read_bytes().startswith(b"\x89PNG")


def test_recording_module_does_not_import_matplotlib():
    subprocess.run([sys.executable, "-c", "import sys, rt_speed; "
                    "assert 'matplotlib' not in sys.modules"], check=True)


def test_plot_keeps_completed_rows_after_interrupted_csv_write(tmp_path, caplog):
    clock = Clock()
    speed = SpeedRecorder(tmp_path / "speed.csv", clock)
    speed.arrive(1)
    started = speed.conversion_start(1)
    clock.now = .1
    speed.conversion_end(1, started)
    speed.finish(1, True)
    with speed.path.open("a", encoding="utf-8") as f:
        f.write("2,live,False,1,")
    result = plot_speed(speed.path)
    assert "incomplete/invalid CSV rows" in caplog.text
    assert result.read_bytes().startswith(b"\x89PNG")


def test_enabled_commit_failure_persists_timing_and_advances(tmp_path, monkeypatch):
    import test_rt_pipeline_parallel as headless
    import rt_pipeline as rt
    clock = Clock()
    handler = headless._mk_handler(tmp_path)
    handler.speed = SpeedRecorder(tmp_path / "speed.csv", clock)
    handler.speed.arrive(1)

    def failing_commit(*args):
        clock.now = .1
        handler.speed.stage(1, "REG", .1)
        raise RuntimeError("synthetic processing failure")

    monkeypatch.setattr(rt, "commit_stage", failing_commit)
    env = rt.ResultEnvelope(scan=1, volume_idx=1, dicom_path=tmp_path / "1.dcm",
                            volume_timestamp=0., success=True)
    rt.DICOMHandler._commit_scan(handler, env)
    row = read_rows(handler.speed.path)[0]
    assert row["status"] == "failed"
    assert float(row["REG_ms"]) == float(row["processing_ms"]) == 100
    assert 1 in handler._processed_scans


def test_detection_is_before_biopac_start_and_close_is_deduplicated(tmp_path):
    import queue
    import test_rt_pipeline_parallel as headless
    import rt_pipeline as rt
    clock = Clock()
    handler = headless._mk_handler(tmp_path)
    handler.speed = SpeedRecorder(tmp_path / "speed.csv", clock)
    handler.current_run = 4
    handler._online_mode = True
    handler._biopac_run_started = handler._biopac_started = False
    handler._pending = queue.PriorityQueue()

    def start_biopac():
        clock.now = 2.
    handler.start_biopac = start_biopac
    path = tmp_path / "001_000004_000001.dcm"
    rt.DICOMHandler.enqueue_path(handler, path)
    rt.DICOMHandler.enqueue_path(handler, path)
    assert handler.speed.rows[1]["detected"] == 0
    assert handler._pending.qsize() == 1


def test_file_write_uses_first_observed_create_and_close(tmp_path):
    clock = Clock()
    speed = SpeedRecorder(tmp_path / "speed.csv", clock)
    speed.arrive(1)  # generic detection is not a file-created observation
    clock.now = .1
    speed.file_event(1, "created")
    clock.now = .3
    speed.file_event(1, "created")
    clock.now = .6
    speed.file_event(1, "closed")
    clock.now = .9
    speed.file_event(1, "closed")
    speed.finish(1, True)
    row = read_rows(speed.path)[0]
    assert float(row["file_write_ms"]) == pytest.approx(500)
    assert speed.rows[1]["detected"] == 0


@pytest.mark.parametrize("events", [
    [], ["created"], ["closed"], ["closed", "created", "closed"],
])
def test_file_write_is_unknown_without_ordered_create_and_close(tmp_path, events):
    clock = Clock()
    speed = SpeedRecorder(tmp_path / "speed.csv", clock)
    speed.arrive(1)
    for event in events:
        clock.now += .1
        speed.file_event(1, event)
    speed.finish(1, True)
    assert read_rows(speed.path)[0]["file_write_ms"] == ""


def test_file_events_do_not_invent_backlog_transfer_or_rewrite_finished_rows(tmp_path):
    clock = Clock()
    speed = SpeedRecorder(tmp_path / "speed.csv", clock)
    speed.arrive(1, backlog=True)
    speed.file_event(1, "created")
    clock.now = .1
    speed.file_event(1, "closed")
    speed.finish(1, True)
    speed.file_event(2, "created")
    clock.now = .2
    speed.finish(2, True)
    before = speed.path.read_bytes()
    clock.now = .3
    speed.file_event(2, "closed")
    assert speed.path.read_bytes() == before
    rows = read_rows(speed.path)
    assert rows[0]["source"] == "backlog"
    assert rows[0]["file_write_ms"] == rows[1]["file_write_ms"] == ""
    assert speed.rows[2]["file_write_ms"] == ""


def test_watchdog_close_is_measured_even_when_file_is_already_queued(tmp_path):
    import queue
    from types import SimpleNamespace
    import test_rt_pipeline_parallel as headless
    import rt_pipeline as rt
    clock = Clock()
    handler = headless._mk_handler(tmp_path)
    handler.speed = SpeedRecorder(tmp_path / "speed.csv", clock)
    handler.current_run = 4
    handler._online_mode = False
    handler._pending = queue.PriorityQueue()
    handler.enqueue_path = rt.DICOMHandler.enqueue_path.__get__(handler)
    event = SimpleNamespace(is_directory=False, src_path=str(tmp_path / "001_000004_000001.dcm"))
    rt.DICOMHandler.on_created(handler, event)
    clock.now = .25
    rt.DICOMHandler.on_closed(handler, event)
    assert handler._pending.qsize() == 1
    assert handler.speed.rows[1]["file_write_ms"] == 250
    event.src_path = str(tmp_path / "001_000005_000002.dcm")
    rt.DICOMHandler.on_created(handler, event)
    assert 2 not in handler.speed.rows


@pytest.mark.parametrize("values, expected", [
    ([], np.nan), ([np.nan, np.inf, -np.inf], np.nan),
    ([3.], 3.), ([1., 2., 3., 4., 5., 6., 7.], 4.),
])
def test_median_ci_does_not_claim_uncertainty_from_too_few_samples(values, expected):
    median, low, high = median_ci(values)
    if np.isnan(expected):
        assert np.isnan(median)
    else:
        assert median == expected
    assert np.isnan(low) and np.isnan(high)


def test_median_ci_is_reproducible_and_keeps_the_observed_median():
    values = np.linspace(10., 30., 64) + np.sin(np.arange(64))
    result = median_ci(values, n_resamples=500, seed=73)
    assert result == median_ci(values, n_resamples=500, seed=73)
    median, low, high = result
    assert median == np.median(values)
    assert np.all(np.isfinite(result))
    assert values.min() <= low < high <= values.max()


@pytest.mark.parametrize("count", [8, 64])
def test_median_ci_constant_timings_have_zero_width(count):
    assert median_ci(np.full(count, 12.5)) == (12.5, 12.5, 12.5)


def test_median_ci_filters_nonfinite_values_without_reordering_valid_samples():
    values = np.array([2., 8., 1., 6., 5., 3., 7., 4.] * 5)
    # Unknown values before/after the retained block do not break adjacency.
    assert median_ci(np.concatenate(([np.inf], values, [-np.inf]))) == median_ci(values)
    contaminated = np.column_stack((values, np.full(values.size, np.nan))).ravel()
    median, low, high = median_ci(contaminated)
    assert median == np.median(values)
    assert np.isnan(low) and np.isnan(high)  # Missing interior TRs must not become adjacent.
    # Filtering can also leave an insufficient sample despite a longer input.
    median, low, high = median_ci([1., 2., np.nan, 3., np.inf, 4., 5., 6., 7.])
    assert median == 4.
    assert np.isnan(low) and np.isnan(high)


def test_median_ci_requires_contiguous_original_volume_indices():
    values = np.arange(16, dtype=float)
    consecutive = np.arange(30, 46)
    assert median_ci(values, sample_indices=consecutive) == median_ci(values)
    # Removing failed/nonselected volumes leaves a gap between otherwise valid timings.
    gapped = np.concatenate((np.arange(30, 38), np.arange(40, 48)))
    median, low, high = median_ci(values, sample_indices=gapped)
    assert median == 7.5
    assert np.isnan(low) and np.isnan(high)


def test_median_ci_respects_temporal_dependence_instead_of_iid_resampling():
    # Smoothly varying timings and their shuffled copy have the same values and
    # median. Preserving adjacent observations should increase uncertainty for
    # this strongly autocorrelated fixture relative to its shuffled copy.
    ordered = 20. + 5. * np.sin(np.linspace(0., 8. * np.pi, 256))
    shuffled = np.random.default_rng(19).permutation(ordered)
    median, low, high = median_ci(ordered, n_resamples=1000)
    shuffled_median, shuffled_low, shuffled_high = median_ci(shuffled, n_resamples=1000)
    assert median == shuffled_median
    assert high - low > shuffled_high - shuffled_low


@pytest.mark.parametrize("first_available", [0, 1, 2, 3])
def test_profile_summary_selects_one_preferred_acquisition_cache_cohort(first_available):
    cohorts = [("live", False), ("backlog", False), ("live", True), ("backlog", True)]
    rows = [dict(volume=i, source=source, cached_input=str(cached),
                 reg_ready="True", status="ok")
            for i, (source, cached) in enumerate(cohorts) if i >= first_available]
    # Failed fresh/live observations must never affect either selection or estimates.
    rows.insert(0, dict(volume=99, source="live", cached_input="False",
                        reg_ready="True", status="failed"))
    selected, label = select_profile_rows(rows)
    assert [row["volume"] for row in selected] == [first_available]
    source, cached = cohorts[first_available]
    assert source in label
    assert ("reused NIfTI" if cached else "fresh conversion") in label
    assert "ready" in label


def test_profile_summary_prefers_ready_data_and_preserves_its_temporal_order():
    rows = [
        dict(volume=1, source="live", cached_input="False", reg_ready="False", status="ok"),
        dict(volume=2, source="backlog", cached_input="True", reg_ready="True", status="ok"),
        dict(volume=3, source="live", cached_input="False", reg_ready="True", status="failed"),
        dict(volume=4, source="backlog", cached_input="True", reg_ready="True", status="ok"),
    ]
    selected, label = select_profile_rows(rows)
    assert [row["volume"] for row in selected] == [2, 4]
    assert "ready" in label
    rows[1]["reg_ready"] = rows[3]["reg_ready"] = "False"
    selected, label = select_profile_rows(rows)
    assert [row["volume"] for row in selected] == [1]
    assert "warm-up only" in label


def test_profile_summary_excludes_failed_only_input():
    assert select_profile_rows([]) == ([], "no successful volumes")
    selected, label = select_profile_rows([
        dict(source="live", cached_input="False", reg_ready="True", status="failed")
    ])
    assert selected == [] and label == "no successful volumes"


def test_speed_plot_is_one_stage_chart_and_supports_csv_before_file_timing(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    path = tmp_path / "old_speed.csv"
    old_fields = [field for field in FIELDS if field != "file_write_ms"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, old_fields)
        writer.writeheader()
        for volume in range(1, 9):
            row = dict.fromkeys(old_fields, 1.)
            row.update(volume=volume, source="live", cached_input="False",
                       reg_ready="True", status="ok", processing_ms=100., latency_ms=120.)
            writer.writerow(row)
    captured = {}

    def capture_figure(fig, *args, **kwargs):
        captured["axes_count"] = len(fig.axes)
        captured["labels"] = [label.get_text() for label in fig.axes[0].get_yticklabels()]
        captured["annotations"] = [text.get_text() for text in fig.axes[0].texts]
        captured["figure_text"] = [text.get_text() for text in fig.texts]
    monkeypatch.setattr(Figure, "savefig", capture_figure)
    assert plot_speed(path) == path.with_suffix(".png")
    assert captured["axes_count"] == 1
    labels = "\n".join(captured["labels"])
    assert all(term in labels.lower() for term in ["denoising", "dvars", "watchdog", "file"])
    assert "cumulative glm" not in "\n".join([labels, *captured["figure_text"]]).lower()
    assert captured["annotations"].count("not observed") == 1  # FILE, not a fabricated zero
