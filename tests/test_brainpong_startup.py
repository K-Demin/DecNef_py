"""Exercise BrainPong's actual launcher without requiring PsychoPy/native tasks."""
import ast
from copy import deepcopy
import logging
from pathlib import Path
from queue import Empty, Queue
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from rt_global_settings import RegressorSettings


SOURCE = Path(__file__).resolve().parents[1] / "rt_brainpong_parallel.py"


def function(name):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    return next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def execute(nodes, namespace):
    # Keep production annotations without importing PsychoPy or the task package.
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)


@pytest.mark.parametrize("offline_assets", [None, "/prepared/offline/reference-v1"])
@pytest.mark.parametrize("reference", ["4", "4/reference_gameplay", "/reference/pooled"])
def test_actual_main_opens_game_while_backend_waits_and_preserves_reference(
        monkeypatch, tmp_path, offline_assets, reference):
    main = function("main")
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == "prepare_nf_run" for node in ast.walk(main))
    first = next(index for index, node in enumerate(main.body)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
        and target.id == "score_queue" for target in node.targets))
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    failures, chronology, received, created = [], [], [], []
    rt_settings = RegressorSettings(offline_assets=offline_assets,
        voxel_norm_mode="mean", voxel_norm_reference="running",
        enable_motion_regression=False, analysis_space="t1")
    runtime = ModuleType("rt_pipeline")
    runtime.REGRESSOR_SETTINGS = RegressorSettings()

    def make_config(**kwargs):
        cfg = SimpleNamespace(**kwargs, rt_work_dir=tmp_path / "run",
                              fmap_dir=tmp_path / "fmap", reuse_prepared_reference=False)
        created.append(cfg)
        return cfg

    runtime.RTSessionConfig = make_config
    reference_stats = {"baseline_mean": 2.5, "baseline_std": .7}

    def load_reference(cfg, rs):
        assert rs == reference
        assert cfg.reference_score_run == reference
        # Loading happens in the parent, so changing only the child settings
        # would still load the reference under the wrong normalization policy.
        assert vars(runtime.REGRESSOR_SETTINGS) == vars(rt_settings)
        chronology.append("load_reference")
        return reference_stats

    def backend(cfg, queue):
        assert vars(runtime.REGRESSOR_SETTINGS) == vars(rt_settings)
        chronology.append("backend_waiting")
        entered.set()
        if not release.wait(3):
            raise AssertionError("Presentation was blocked by preparation")
        done.set()

    runtime.load_reference_score_stats = load_reference
    runtime.run_rt_pipeline = backend
    runtime.prepare_nf_run = lambda *args: pytest.fail("Parent must not prepare synchronously")
    monkeypatch.setitem(sys.modules, "rt_pipeline", runtime)
    qc = ModuleType("rs_realtime_parallel")
    qc._plot_qc = lambda directory: received.append(("qc", directory))
    monkeypatch.setitem(sys.modules, "rs_realtime_parallel", qc)

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

        def is_alive(self):
            return self.thread.is_alive()

        def terminate(self):
            release.set()

        def join(self, timeout):
            self.thread.join(timeout)

    cfg = {"task": {"n_blocks": 2, "trials_per_block": 7, "start_block": 3,
                   "start_level": 2, "wait_for_trigger_in_simulation": True},
           "feedback": {"zscore_mode": "reference"},
           "decoder": {"decoder_path": "/decoder/weights.nii.gz"},
           "logging": {"output_dir": str(tmp_path / "original")}}
    original_task = deepcopy(cfg["task"])
    args = SimpleNamespace(sub="88888", day="2", run="8", incoming_root="/incoming",
        base_data=str(tmp_path), decoder_template=None, decoder_roi_txt=None,
        reference_score_run=reference, ap_block=2, pa_block=3,
        score_file=tmp_path / "external_scores.csv")

    def present(actual_cfg, *, score_queue, score_file):
        assert threading.current_thread() is threading.main_thread()
        assert entered.is_set() and not done.is_set()
        assert actual_cfg["task"] == original_task
        assert actual_cfg["rest_calibration"] is reference_stats
        assert score_file is args.score_file
        assert isinstance(score_queue, Queue)
        chronology.append("presentation")
        release.set()
        assert done.wait(3)

    namespace = dict(Path=Path, rt_settings=rt_settings, cfg=cfg, args=args,
        runs_pipeline=True, uses_reference_score=True,
        mp=SimpleNamespace(get_context=lambda method: SimpleNamespace(
            Process=Process, Queue=lambda maxsize: Queue(maxsize=maxsize))),
        run_brainpong_presentation=present,
        _write_json=lambda *values: received.append(("write", values)),
        _merge_session_metadata=lambda *values: received.append(("metadata", values)),
        log=logging.getLogger("brainpong-startup-test"))
    execute([function("_prepare_rt_config"), function("_run_pipeline_with_settings")], namespace)
    try:
        execute(main.body[first:], namespace)
    finally:
        release.set()
    assert not failures
    assert chronology == ["load_reference", "backend_waiting", "presentation"]
    assert created[0].reuse_prepared_reference is (offline_assets is None)
    assert cfg["logging"]["output_dir"] == str(tmp_path / "run" / "brainpong")
    assert received[-1] == ("qc", tmp_path / "run")


def test_no_pipeline_opens_presentation_without_reference_or_preparation(tmp_path):
    main = function("main")
    first = next(index for index, node in enumerate(main.body)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
        and target.id == "score_queue" for target in node.targets))
    cfg = {"logging": {"output_dir": str(tmp_path)}}
    shown = []
    execute(main.body[first:], dict(Path=Path, cfg=cfg, runs_pipeline=False,
        args=SimpleNamespace(score_file=None),
        run_brainpong_presentation=lambda *args, **kwargs: shown.append((args, kwargs))))
    assert shown == [((cfg,), {"score_queue": None, "score_file": None})]


@pytest.mark.parametrize("presentation_name", [
    "run_brainpong_presentation", "_run_legacy_brainpong_presentation"])
def test_actual_trigger_wait_still_waits_for_scanner_and_preserves_early_scores(presentation_name):
    presentation = function(presentation_name)
    wait = next(node for node in presentation.body if isinstance(node, ast.If)
                and any(isinstance(child, ast.While) for child in node.body))
    events = []
    keys = iter([[], ["space"], ["s"]])
    score = {"volume_idx": 1, "score_raw": .5}
    queue, pending = Queue(), []
    queue.put_nowait(score)
    namespace = dict(sim_cfg={}, task_cfg={}, score_queue=queue, pending_scores=pending,
        Empty=Empty, time=SimpleNamespace(time=lambda: 123.),
        logger=SimpleNamespace(finish=lambda: events.append("finish")),
        waiting_text=SimpleNamespace(draw=lambda: events.append("draw")),
        win=SimpleNamespace(flip=lambda: events.append("flip")),
        event=SimpleNamespace(getKeys=lambda: next(keys)),
        core=SimpleNamespace(wait=lambda seconds: events.append(seconds)))
    harness = ast.FunctionDef(name="wait_for_trigger", args=ast.arguments(
        posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[wait], decorator_list=[])
    execute([function("_drain_pipeline_scores"), harness], namespace)
    namespace["wait_for_trigger"]()
    assert events == ["draw", "flip", .02, "draw", "flip", .02, "draw", "flip"]
    assert pending == [score]
    assert pending[0] is score
    assert queue.empty()


def test_worker_reports_preparation_error_and_reraises_original_exception(monkeypatch):
    runtime = ModuleType("rt_pipeline")
    runtime.REGRESSOR_SETTINGS = RegressorSettings()
    failure = ValueError("could not load prepared transforms")

    def backend(cfg, queue):
        raise failure

    runtime.run_rt_pipeline = backend
    monkeypatch.setitem(sys.modules, "rt_pipeline", runtime)
    namespace, queue = {}, Queue()
    execute([function("_run_pipeline_with_settings")], namespace)
    with pytest.raises(ValueError) as caught:
        namespace["_run_pipeline_with_settings"](object(), queue, {})
    assert caught.value is failure
    assert queue.get_nowait() == {
        "pipeline_error": "ValueError: could not load prepared transforms"}


def test_queue_helper_preserves_score_objects_without_waiting_and_surfaces_errors():
    namespace = {"Empty": Empty}
    execute([function("_drain_pipeline_scores")], namespace)
    drain = namespace["_drain_pipeline_scores"]
    assert drain(None) == []
    queue = Queue()
    scores = [{"volume_idx": 1, "score_raw": .5}, {"volume_idx": 2, "score_raw": -.2}]
    for score in scores:
        queue.put_nowait(score)
    result = drain(queue)
    assert result == scores
    assert all(actual is expected for actual, expected in zip(result, scores))
    assert drain(queue) == []
    queue.put_nowait({"pipeline_error": "ValueError: reference missing"})
    with pytest.raises(RuntimeError, match="Real-time processing failed: ValueError: reference missing"):
        drain(queue)


@pytest.mark.parametrize("presentation_name", [
    "run_brainpong_presentation", "_run_legacy_brainpong_presentation"])
def test_actual_wait_screen_closes_and_reports_worker_error_before_scanner_trigger(presentation_name):
    presentation = function(presentation_name)
    wait = next(node for node in presentation.body if isinstance(node, ast.If)
                and any(isinstance(child, ast.While) for child in node.body))
    events, queue = [], Queue()
    queue.put_nowait({"pipeline_error": "FileNotFoundError: prepared mask missing"})

    def get_keys():
        pytest.fail("Worker error must surface without waiting for a scanner trigger")

    namespace = dict(sim_cfg={}, task_cfg={}, Empty=Empty,
        score_queue=queue, pending_scores=[],
        waiting_text=SimpleNamespace(draw=lambda: events.append("draw")),
        win=SimpleNamespace(flip=lambda: events.append("flip"), close=lambda: events.append("close")),
        logger=SimpleNamespace(finish=lambda: events.append("finish")),
        event=SimpleNamespace(getKeys=get_keys))
    harness = ast.FunctionDef(name="wait_for_trigger", args=ast.arguments(
        posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[wait], decorator_list=[])
    execute([function("_drain_pipeline_scores"), harness], namespace)
    with pytest.raises(RuntimeError, match="prepared mask missing"):
        namespace["wait_for_trigger"]()
    assert events == ["draw", "flip", "close", "finish"]


@pytest.mark.parametrize("presentation_name", [
    "run_brainpong_presentation", "_run_legacy_brainpong_presentation"])
def test_actual_active_score_drain_uses_buffer_once_and_surfaces_worker_errors(presentation_name):
    presentation = function(presentation_name)
    drain = next(node for node in presentation.body
                 if isinstance(node, ast.FunctionDef) and node.name == "drain_scores")
    pending = [{"volume_idx": 1, "score_raw": .1}, {"volume_idx": 2, "score_raw": .2}]
    queue, updates = Queue(), []
    queue.put_nowait({"volume_idx": 3, "score_raw": .3})

    def update(**kwargs):
        updates.append(kwargs["volume_index"])
        return SimpleNamespace(volume_index=kwargs["volume_index"])

    namespace = dict(Empty=Empty, pending_scores=pending, score_queue=queue,
        source=None, tailer=None, sim_cfg={},
        adapter=SimpleNamespace(current_state=lambda: SimpleNamespace(volume_index=None), update=update),
        context={"score_updates": 0, "block": 1, "trial": 1},
        time=SimpleNamespace(time=lambda: 123.), score_context=lambda state: {},
        logger=SimpleNamespace(log_score=lambda *args: None))
    # The real binary drain closes over latest_state; preserve that closure
    # while supplying only the adapter/logger and a deterministic queue.
    harness = ast.FunctionDef(name="make_drain", args=ast.arguments(
        posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[ast.Assign(targets=[ast.Name(id="latest_state", ctx=ast.Store())],
            value=ast.Call(func=ast.Attribute(value=ast.Name(id="adapter", ctx=ast.Load()),
                attr="current_state", ctx=ast.Load()), args=[], keywords=[])),
            drain, ast.Return(value=ast.Name(id="drain_scores", ctx=ast.Load()))],
        decorator_list=[])
    execute([function("_drain_pipeline_scores"), harness], namespace)
    actual_drain = namespace["make_drain"]()
    arguments = (1, 1, "TOP") if presentation_name.startswith("_run_legacy") else ()
    actual_drain(*arguments)
    assert updates == [1, 2, 3]
    assert pending == []
    actual_drain(*arguments)
    assert updates == [1, 2, 3]
    queue.put_nowait({"pipeline_error": "RuntimeError: worker failed"})
    with pytest.raises(RuntimeError, match="worker failed"):
        actual_drain(*arguments)


def test_actual_task_tail_preserves_blocks_trials_and_behavioral_timing(tmp_path):
    presentation = function("run_brainpong_presentation")
    first = next(index for index, node in enumerate(presentation.body)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
        and target.id == "n_blocks" for target in node.targets))
    blocks, finished = [], []
    params = {"init_level": 0, "init_points": 0, "block_pre_delay": 12.5,
              "performance_criteria": .8, "use_aggressive_leveling": False}
    game_params = {"initial_baseline": 31., "iti": 17., "trial_duration": 43.}

    class TrialBlock:
        def __init__(self, **kwargs):
            blocks.append(kwargs)
            self.total_points = kwargs["init_points"]
            self.current_level = kwargs["init_level"]

        def run(self):
            return True

    namespace = dict(task_cfg={"n_blocks": 2, "trials_per_block": 7,
        "start_block": 3, "start_level": 2}, params=params, cfg={},
        levels={0: {}, 1: {}, 2: {}}, levels_path=tmp_path / "levels.yaml",
        logger=SimpleNamespace(output_dir=tmp_path, finish=lambda state: finished.append(state)),
        win=SimpleNamespace(close=lambda: None), TrialBlock=TrialBlock,
        game_params=game_params, run_neural_game=object(),
        log=logging.getLogger("brainpong-startup-test"))
    execute(presentation.body[first:], namespace)
    assert [block["block_num"] for block in blocks] == [3, 4]
    assert all(block["n_trials"] == 7 and block["init_level"] == 2 for block in blocks)
    assert all(block["block_pre_delay"] == 12.5 and block["performance_criteria"] == .8
               and block["game_params"] is game_params for block in blocks)
    assert game_params == {"initial_baseline": 31., "iti": 17., "trial_duration": 43.}
    assert finished[0]["completed_internal_blocks"] == 2
