"""NF validates existing spatial assets; scanner-dependent work stays in its worker."""
import ast
from collections import deque
import logging
from pathlib import Path
from queue import Empty, Queue
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from rt_global_settings import RegressorSettings


ROOT = Path(__file__).resolve().parents[1]
CONSUMERS = ("rt_nf_events_parallel.py", "rt_nf_events_prob_parallel.py", "rt_psychopy_parallel.py")


def function(filename, name):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8-sig"))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def execute(filename, nodes, namespace):
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(ROOT / filename), "exec"), namespace)


@pytest.mark.parametrize("explicit_settings", [True, False])
def test_validator_delegates_without_preparing_fieldmaps_or_waiting(monkeypatch, explicit_settings):
    calls = []
    alignment = ModuleType("fmri_rt_preproc.session_alignment")
    alignment.validate_prepared_reference = lambda *args: calls.append(args)
    alignment.ensure_nf_session = lambda *args, **kwargs: pytest.fail("Validation must not prepare")
    monkeypatch.setitem(sys.modules, "fmri_rt_preproc.session_alignment", alignment)
    cfg, default_settings, supplied_settings = object(), object(), object()
    namespace = dict(REGRESSOR_SETTINGS=default_settings,
        ensure_fieldmap_only=lambda *args: pytest.fail("Validation must not wait for fieldmaps"))
    execute("rt_pipeline.py", [function("rt_pipeline.py", "validate_nf_reference")], namespace)
    if explicit_settings:
        namespace["validate_nf_reference"](cfg, supplied_settings)
    else:
        namespace["validate_nf_reference"](cfg)
    assert calls == [(cfg, supplied_settings if explicit_settings else default_settings)]


@pytest.mark.parametrize("filename", CONSUMERS)
@pytest.mark.parametrize("preparation_available", [True, False])
@pytest.mark.parametrize("offline_assets", [None, "/offline/prepared-reference"])
def test_consumer_validation_fails_before_ui_or_opens_while_fieldmap_waits(
        monkeypatch, tmp_path, filename, preparation_available, offline_assets):
    main = function(filename, "main")
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == "prepare_nf_run" for node in ast.walk(main))
    validation = next(index for index, node in enumerate(main.body)
        if isinstance(node, ast.ImportFrom) and node.module == "rt_pipeline"
        and any(name.name == "validate_nf_reference" for name in node.names))
    start = next(index for index, node in enumerate(main.body)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
        and target.id == "pipeline_process" for target in node.targets))
    # Execute the real validation and worker/UI boundary. The independent task
    # condition/metadata setup between them is supplied as context.
    boundary = [*main.body[validation:validation + 2], *main.body[start:]]
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    chronology, failures = [], []
    cfg = SimpleNamespace(reference_score_run="4", reference_score_stats={"mean": 0., "std": 1.})
    settings = RegressorSettings(TR=1., analysis_space="epi", offline_assets=offline_assets)
    runtime = ModuleType("rt_pipeline")
    runtime.REGRESSOR_SETTINGS = settings

    def validate(actual_cfg):
        assert actual_cfg is cfg
        chronology.append("validate_reference")
        if not preparation_available:
            raise FileNotFoundError("Prepared reference missing; run reference preparation first")

    def backend(actual_cfg, queue):
        assert actual_cfg is cfg
        chronology.append("fieldmap_waiting")
        entered.set()
        if not release.wait(3):
            raise AssertionError("UI did not open while fieldmap/backend was waiting")
        done.set()

    runtime.validate_nf_reference = validate
    runtime.run_rt_pipeline = backend
    runtime.prepare_nf_run = lambda *args: pytest.fail("Parent cannot prepare spatial assets")
    monkeypatch.setitem(sys.modules, "rt_pipeline", runtime)

    class Process:
        def __init__(self, target, args):
            def invoke():
                try:
                    target(*args)
                except BaseException as exc:
                    failures.append(exc)
                    entered.set()
            self.thread = threading.Thread(target=invoke)
        def start(self):
            self.thread.start()
            assert entered.wait(3)
        def is_alive(self): return self.thread.is_alive()
        def terminate(self): release.set()
        def join(self, timeout): self.thread.join(timeout)

    def present(*args, **kwargs):
        assert entered.is_set() and not done.is_set()
        chronology.append("presentation")
        release.set()
        assert done.wait(3)

    args = SimpleNamespace(skip_first_trs=2, n_trials=4, iti_trs=1, cue_trs=2,
        scans_trs=3, delay_trs=4, feedback_trs=5, score_delay=3, max_points=20)
    namespace = dict(ctx=SimpleNamespace(Process=Process, Event=threading.Event),
        cfg=cfg, score_queue=Queue(), settings_payload={**vars(settings), "_decoder_intercept": 0.},
        REGRESSOR_SETTINGS=settings, args=args, baseline_trs=30, run_dir=tmp_path,
        trial_scores_path=tmp_path / "trial_scores.csv", condition=object(),
        condition_scores_path=tmp_path / "condition.csv", max_trs=200, use_pca=False,
        biopac_stop=None, biopac_process=None, _plot_qc=lambda *args: None,
        _make_direct_scorer=lambda intercept: object(), run_nf_events_presentation=present,
        run_psychopy_presentation=present, log=logging.getLogger("nf-startup-test"))
    execute(filename, [function(filename, "_run_pipeline_with_settings")], namespace)
    try:
        if preparation_available:
            execute(filename, boundary, namespace)
        else:
            with pytest.raises(FileNotFoundError, match="run reference preparation first"):
                execute(filename, boundary, namespace)
    finally:
        release.set()
        if "pipeline_process" in namespace:
            namespace["pipeline_process"].join(timeout=3)
    assert not failures
    assert not hasattr(cfg, "reuse_prepared_reference")
    assert chronology == (["validate_reference", "fieldmap_waiting", "presentation"]
                          if preparation_available else ["validate_reference"])
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("filename", CONSUMERS)
def test_worker_failure_reaches_parent_protocol_and_keeps_original_exception(monkeypatch, filename):
    runtime = ModuleType("rt_pipeline")
    runtime.REGRESSOR_SETTINGS = RegressorSettings(TR=1.)
    failure = ValueError("fieldmap could not be prepared")
    def backend(*args): raise failure
    runtime.run_rt_pipeline = backend
    monkeypatch.setitem(sys.modules, "rt_pipeline", runtime)
    namespace = dict(_make_direct_scorer=lambda intercept: object())
    execute(filename, [function(filename, "_run_pipeline_with_settings"),
                       function(filename, "_raise_if_pipeline_failed")], namespace)
    queue = Queue()
    with pytest.raises(ValueError) as caught:
        namespace["_run_pipeline_with_settings"](object(), queue,
                                                  {"TR": 1., "_decoder_intercept": 0.})
    assert caught.value is failure
    closed = []
    with pytest.raises(RuntimeError, match="Real-time processing failed: ValueError: fieldmap"):
        namespace["_raise_if_pipeline_failed"](queue.get_nowait(),
            SimpleNamespace(close=lambda: closed.append(True)))
    assert closed == [True]


@pytest.mark.parametrize("filename", CONSUMERS[:2])
def test_actual_event_wait_surfaces_worker_failure_before_trigger(filename):
    presentation = function(filename, "run_nf_events_presentation")
    nested = [node for node in presentation.body if isinstance(node, ast.FunctionDef)]
    wait_loop = next(node for node in presentation.body if isinstance(node, ast.While))
    queue, closed = Queue(), []
    queue.put_nowait({"pipeline_error": "RuntimeError: missing fieldmap"})
    namespace = dict(score_queue=queue, Empty=Empty,
        win=SimpleNamespace(close=lambda: closed.append(True)), _release_mouse_cursor=lambda win: None,
        event=SimpleNamespace(getKeys=lambda: pytest.fail("Must fail before scanner trigger")))
    harness = ast.FunctionDef(name="wait", args=ast.arguments(posonlyargs=[], args=[],
        kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[ast.Assign(targets=[ast.Name(id="max_seen_vol", ctx=ast.Store())], value=ast.Constant(value=0)),
              *nested, wait_loop], decorator_list=[])
    execute(filename, [function(filename, "_raise_if_pipeline_failed"), harness], namespace)
    with pytest.raises(RuntimeError, match="missing fieldmap"):
        namespace["wait"]()
    assert closed == [True]


@pytest.mark.parametrize("failure", [False, True])
def test_psychopy_wait_preserves_scores_and_reports_errors_before_trigger(failure):
    filename = "rt_psychopy_parallel.py"
    presentation = function(filename, "run_psychopy_presentation")
    wait_loop = next(node for node in presentation.body if isinstance(node, ast.While))
    queue, pending, closed = Queue(), deque(), []
    score = {"volume_idx": 12, "reg_ready": True, "score_raw": .25}
    queue.put_nowait({"pipeline_error": "RuntimeError: missing fieldmap"} if failure else score)
    namespace = dict(score_queue=queue, Empty=Empty, pending_messages=pending, waiting=True,
        win=SimpleNamespace(close=lambda: closed.append(True), flip=lambda: None),
        waiting_text=SimpleNamespace(draw=lambda: None), _release_mouse_cursor=lambda win: None,
        time=SimpleNamespace(time=lambda: 123.), core=SimpleNamespace(wait=lambda seconds: None),
        event=SimpleNamespace(getKeys=lambda: ["s"]))
    execute(filename, [function(filename, "_raise_if_pipeline_failed")], namespace)
    harness = ast.FunctionDef(name="wait", args=ast.arguments(posonlyargs=[], args=[],
        kwonlyargs=[], kw_defaults=[], defaults=[]), body=[wait_loop], decorator_list=[])
    # waiting is assigned by the actual loop, so retain its production initial value.
    harness.body.insert(0, ast.Assign(targets=[ast.Name(id="waiting", ctx=ast.Store())], value=ast.Constant(value=True)))
    execute(filename, [harness], namespace)
    if failure:
        with pytest.raises(RuntimeError, match="missing fieldmap"):
            namespace["wait"]()
        assert closed == [True] and not pending
    else:
        namespace["wait"]()
        assert list(pending) == [score] and pending[0] is score
        assert closed == []
