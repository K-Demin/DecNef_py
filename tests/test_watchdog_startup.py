"""Files arriving during native handler/watchdog startup must not disappear."""
import ast
from collections import Counter
import logging
from pathlib import Path
from queue import PriorityQueue
import queue
import sys
from threading import RLock
from types import ModuleType, SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "rt_pipeline.py"


def nodes():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    handler = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DICOMHandler")
    methods = {node.name: node for node in handler.body if isinstance(node, ast.FunctionDef)}
    return functions, methods


def execute(body, namespace):
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), *body], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)


def harness(monkeypatch, tmp_path, *, recorded_only=False, speed_enabled=False):
    functions, methods = nodes()
    chronology, ingested, handlers, snapshots = [], [], [], []
    state = dict(observer_started=False, live_sent=False)

    def create(volume, run=7):
        path = tmp_path / f"001_{run:06d}_{volume:06d}.dcm"
        path.write_bytes(f"volume-{volume}".encode())
        return path

    create(1)
    create(20, run=8)
    (tmp_path / "not_a_volume.dcm").write_bytes(b"ignore")

    class Incoming:
        def glob(self, pattern):
            paths = list(tmp_path.glob(pattern))
            snapshots.append({"observer_started": state["observer_started"], "paths": paths})
            chronology.append("snapshot_after_start" if state["observer_started"] else "snapshot_before_start")
            return iter(paths)
        def __str__(self): return str(tmp_path)

    class Handler:
        def __init__(self, cfg, **kwargs):
            chronology.append("handler_init")
            self.current_run = int(cfg.run)
            self._pending, self._pending_scans, self._processed_scans = PriorityQueue(), set(), set()
            self._lock = RLock()
            self._inflight_scans, self._scan_first_seen = set(), {}
            self._next_scan_to_commit = None
            self._engine_mode = "parallel_ordered"
            self._executor = SimpleNamespace(submit=lambda compute, *args: Future(compute(*args)))
            self._online_mode = False
            self._biopac_run_started = True
            self.speed = (SimpleNamespace(arrive=lambda *args, **kwargs: None,
                                          file_event=lambda *args: None) if speed_enabled else None)
            handlers.append(self)
            # This file has no watcher notification: it is the real startup gap.
            create(2)

        def _compute_scan(self, path, scan, volume_idx):
            chronology.append(f"submit_{scan}")
            if not ingested:
                assert self._next_scan_to_commit == 1
            ingested.append(scan)
            return SimpleNamespace(scan=scan)

        def _on_compute_future_done(self, future):
            scan = future.result().scan
            self._inflight_scans.discard(scan)
            self.mark_processed(scan)

        def enable_online_mode(self): self._online_mode = True
        def stop(self): chronology.append("handler_stopped")

    class Future:
        def __init__(self, result): self._result = result
        def result(self): return self._result
        def add_done_callback(self, callback): callback(self)

    class Observer:
        def __init__(self):
            if recorded_only:
                pytest.fail("Recorded-only replay must never construct a watchdog")
        def schedule(self, handler, path, recursive):
            self.handler = handler
            assert path == str(tmp_path) and recursive is False
        def start(self):
            state["observer_started"] = True
            chronology.append("observer_started")
            create(3)  # Startup before watcher delivery: rescan must find it.
            path = create(4)
            event = SimpleNamespace(src_path=str(path), is_directory=False)
            # This volume overlaps live events and the post-start snapshot.
            self.handler.on_created(event)
            self.handler.on_closed(event)
        def stop(self): chronology.append("observer_stopped")
        def join(self): chronology.append("observer_joined")

    def sleep(seconds):
        if recorded_only:
            pytest.fail("A complete recorded snapshot must not wait for live files")
        if not state["live_sent"]:
            assert state["observer_started"]
            state["live_sent"] = True
            event = SimpleNamespace(src_path=str(create(5)), is_directory=False)
            handlers[0].on_created(event)
            handlers[0].on_closed(event)
            duplicate = SimpleNamespace(src_path=str(tmp_path / "001_000007_000001.dcm"), is_directory=False)
            handlers[0].on_created(duplicate)
            return
        raise KeyboardInterrupt

    alignment = ModuleType("fmri_rt_preproc.session_alignment")
    alignment.ensure_nf_session = lambda *args, **kwargs: chronology.append("reference_validated")
    monkeypatch.setitem(sys.modules, alignment.__name__, alignment)
    namespace = dict(Path=Path, queue=queue, log=logging.getLogger("watchdog-startup-test"),
        REGRESSOR_SETTINGS=SimpleNamespace(analysis_space="t1", enable_biopac_physio=False,
                                           biopac_start_online_only=False),
        DICOMHandler=Handler, Observer=Observer, time=SimpleNamespace(sleep=sleep, time=lambda: 123.),
        resolve_decoder_template=lambda cfg: None, validate_regression_status_schema=lambda *args: None,
        ensure_fieldmap_only=lambda *args: None, validate_run_provenance=lambda *args: None,
        load_reference_score_stats=lambda *args: None, write_session_metadata=lambda *args: None)
    execute([functions["parse_dicom_name"], methods["enqueue_path"], methods["on_created"],
             methods["on_closed"], methods["next_pending"], methods["mark_processed"],
             methods["submit_pending"], functions["_process_recorded_snapshot"], functions["run_rt_pipeline"]], namespace)
    # Use the real enqueue and event handling, including pending/processed dedup.
    for name in ("enqueue_path", "on_created", "on_closed", "next_pending", "mark_processed", "submit_pending"):
        setattr(Handler, name, namespace[name])
    cfg = SimpleNamespace(recorded_only=recorded_only, enable_scoring=False, decoder_template=None,
        incoming_dir=Incoming(), rt_work_dir=tmp_path, run="7", reference_score_run=None)
    return namespace["run_rt_pipeline"], cfg, chronology, ingested, snapshots


@pytest.mark.parametrize("speed_enabled", [False, True])
def test_observer_then_rescan_recovers_startup_files_without_duplicate_processing(
        monkeypatch, tmp_path, speed_enabled):
    run, cfg, chronology, ingested, snapshots = harness(monkeypatch, tmp_path, speed_enabled=speed_enabled)
    run(cfg)
    assert Counter(ingested) == Counter({volume: 1 for volume in range(1, 6)})
    assert ingested == [1, 2, 3, 4, 5]
    assert chronology.index("observer_started") < chronology.index("snapshot_after_start")
    assert chronology.index("snapshot_after_start") < chronology.index("submit_1")
    rescan = next(snapshot for snapshot in snapshots if snapshot["observer_started"])
    assert {path.name for path in rescan["paths"]} >= {
        f"001_000007_{volume:06d}.dcm" for volume in range(1, 5)}
    assert chronology[-3:] == ["observer_stopped", "handler_stopped", "observer_joined"]


def test_recorded_only_keeps_fixed_original_snapshot_without_watchdog_or_rescan(monkeypatch, tmp_path):
    run, cfg, chronology, ingested, snapshots = harness(monkeypatch, tmp_path, recorded_only=True)
    run(cfg)
    # A file created during handler construction is outside this fixed snapshot.
    assert ingested == [1]
    assert len(snapshots) == 1 and not snapshots[0]["observer_started"]
    assert "observer_started" not in chronology
    assert chronology[-1] == "handler_stopped"
