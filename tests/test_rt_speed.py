"""Timing semantics, independent of scanner/ANTs/CUDA availability."""
import csv
import subprocess
import sys

import pytest

from rt_speed import SpeedRecorder, STAGES, plot_speed


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
