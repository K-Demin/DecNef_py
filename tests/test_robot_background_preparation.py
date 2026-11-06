"""The robot window must be visible while scanner-dependent preparation waits."""
import ast
import logging
from pathlib import Path
from queue import Empty, Queue
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "rt_robot_parallel.py"


def function(name):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    return next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def worker(monkeypatch, process):
    runtime = ModuleType("rt_pipeline")
    runtime.REGRESSOR_SETTINGS = {}
    runtime.run_rt_pipeline = process
    monkeypatch.setitem(sys.modules, "rt_pipeline", runtime)
    namespace = dict(Queue=Queue, log=logging.getLogger("robot-startup-test"))
    exec(compile(ast.Module(body=[function("_run_pipeline_with_settings")], type_ignores=[]),
                 str(SOURCE), "exec"), namespace)
    return namespace["_run_pipeline_with_settings"], runtime


def test_worker_starts_immediately_without_waiting_for_presentation(monkeypatch):
    cfg, queue, settings = object(), Queue(), {"analysis_space": "t1"}
    calls = []

    def process(actual_cfg, actual_queue):
        calls.append((actual_cfg, actual_queue))

    run_worker, runtime = worker(monkeypatch, process)
    run_worker(cfg, queue, settings)
    assert calls == [(cfg, queue)]
    assert runtime.REGRESSOR_SETTINGS == settings


def test_worker_keeps_original_three_argument_api(monkeypatch):
    signature = function("_run_pipeline_with_settings").args
    assert [arg.arg for arg in signature.args] == ["cfg", "score_queue", "settings"]
    assert signature.defaults == []
    calls = []
    run_worker, runtime = worker(monkeypatch, lambda *args: calls.append(args))
    cfg, queue = object(), Queue()
    run_worker(cfg, queue, {"TR": 1.})
    assert calls == [(cfg, queue)]
    assert runtime.REGRESSOR_SETTINGS == {"TR": 1.}


def test_actual_wait_screen_draws_before_scanner_trigger_without_pipeline_gate():
    presentation = function("run_robot_task_presentation")
    loop = next(node for node in presentation.body if isinstance(node, ast.While))
    events = []

    class EmptyQueue:
        def get_nowait(self):
            raise Empty

    keys = iter([[], ["space"], ["s"]])

    def get_keys():
        value = next(keys)
        events.append(tuple(value))
        return value

    namespace = dict(score_queue=EmptyQueue(), Empty=Empty,
        waiting_text=SimpleNamespace(draw=lambda: events.append("draw")),
        win=SimpleNamespace(flip=lambda: events.append("flip"), close=lambda: None),
        event=SimpleNamespace(getKeys=get_keys),
        time=SimpleNamespace(time=lambda: 123.),
        core=SimpleNamespace(wait=lambda value: events.append(value)))
    exec(compile(ast.Module(body=[function("_raise_if_pipeline_failed")], type_ignores=[]),
                 str(SOURCE), "exec"), namespace)
    harness = ast.FunctionDef(name="run_wait", args=ast.arguments(
        posonlyargs=[], args=[],
        kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[loop, ast.Return(value=ast.Name(id="first_trigger_timestamp", ctx=ast.Load()))],
        decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[harness], type_ignores=[])),
                 str(SOURCE), "exec"), namespace)
    assert namespace["run_wait"]() == 123.
    assert events == ["draw", "flip", (), .01, "draw", "flip", ("space",), .01,
                      "draw", "flip", ("s",)]


def test_worker_error_is_forwarded_to_presentation_and_original_failure_propagates(monkeypatch):
    def fail(cfg, queue):
        raise ValueError("reference registration failed")

    run_worker, _ = worker(monkeypatch, fail)
    queue = Queue()
    with pytest.raises(ValueError, match="reference registration failed"):
        run_worker(object(), queue, {})
    message = queue.get_nowait()
    namespace = {}
    exec(compile(ast.Module(body=[function("_raise_if_pipeline_failed")], type_ignores=[]),
                 str(SOURCE), "exec"), namespace)
    with pytest.raises(RuntimeError, match="Real-time processing failed: ValueError: reference registration failed"):
        namespace["_raise_if_pipeline_failed"](message)
    namespace["_raise_if_pipeline_failed"]({"score_raw": 1.})


def test_presentation_checks_worker_errors_in_wait_and_active_loops():
    presentation = function("run_robot_task_presentation")
    loops = [node for node in presentation.body if isinstance(node, ast.While)]
    assert len(loops) == 2
    for loop in loops:
        assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == "_raise_if_pipeline_failed" for node in ast.walk(loop))


def test_actual_main_opens_presentation_while_background_preparation_is_blocked(monkeypatch, tmp_path):
    main = function("main")
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == "prepare_nf_run" for node in ast.walk(main))
    # Execute the production process-start and presentation block, replacing
    # only native processes/UI with threads and a deterministic window stand-in.
    first = next(index for index, node in enumerate(main.body)
                 if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                    and target.id == "pipeline_process" for target in node.targets))
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    chronology, failures = [], []

    def process(cfg, queue):
        chronology.append("preparation")
        entered.set()
        if not release.wait(2):
            raise AssertionError("Presentation did not remain responsive during preparation")
        chronology.append("processing")
        done.set()

    run_worker, _ = worker(monkeypatch, process)

    class Process:
        def __init__(self, target, args):
            def invoke():
                try:
                    target(*args)
                except BaseException as exc:
                    failures.append(exc)
            self.thread = threading.Thread(target=invoke)

        def start(self):
            self.thread.start()
            assert entered.wait(2)

        def is_alive(self):
            return self.thread.is_alive()

        def terminate(self):
            release.set()

        def join(self, timeout):
            self.thread.join(timeout)

    def present(**kwargs):
        assert entered.is_set()
        chronology.append("visible")
        assert "preparation_start_event" not in kwargs
        assert not done.is_set()
        chronology.append("background")
        release.set()
        assert done.wait(2)

    args = SimpleNamespace(sub="55555", day="1", run="7", feedback_signal="z",
        reference_score_run="reference", feedback_mapping="arcsine", mapping_scale=1.,
        z_smooth_trs=1, z_gain=10., z_center=50., rounds=6, active_sec=45., rest_sec=15.,
        result_sec=3., rest_baseline_sec=0., start_level=None, jingle=None,
        jingle_volume=.8, audio_lib=None, audio_device=None)
    namespace = dict(ctx=SimpleNamespace(Process=Process, Event=threading.Event),
        _run_pipeline_with_settings=run_worker, cfg=object(), score_queue=Queue(),
        settings_payload={"TR": 1.}, use_pca=False, args=args,
        run_robot_task_presentation=present, condition=object(),
        condition_scores_path=tmp_path / "condition.csv", max_trs=100,
        REGRESSOR_SETTINGS=SimpleNamespace(TR=1.), rest_lead_s=45., subject_root=tmp_path,
        log=logging.getLogger("robot-startup-test"), biopac_stop=None, biopac_process=None,
        _plot_qc=lambda path: None, run_dir=tmp_path)
    try:
        exec(compile(ast.Module(body=main.body[first:], type_ignores=[]), str(SOURCE), "exec"), namespace)
    finally:
        release.set()
        if "pipeline_process" in namespace:
            namespace["pipeline_process"].join(timeout=2)
    assert not failures
    assert chronology == ["preparation", "visible", "background", "processing"]
