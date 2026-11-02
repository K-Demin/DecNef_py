import queue
import threading
import time
from pathlib import Path
import sys
import types

import numpy as np
import pytest

watchdog_mod = types.ModuleType("watchdog")
watchdog_observers = types.ModuleType("watchdog.observers")
watchdog_events = types.ModuleType("watchdog.events")
watchdog_observers.Observer = object
watchdog_events.FileSystemEventHandler = object
sys.modules.setdefault("watchdog", watchdog_mod)
sys.modules.setdefault("watchdog.observers", watchdog_observers)
sys.modules.setdefault("watchdog.events", watchdog_events)
pyqt5_mod = types.ModuleType("PyQt5")
pyqt5_widgets = types.ModuleType("PyQt5.QtWidgets")
pyqt5_core = types.ModuleType("PyQt5.QtCore")
sys.modules.setdefault("PyQt5", pyqt5_mod)
sys.modules.setdefault("PyQt5.QtWidgets", pyqt5_widgets)
sys.modules.setdefault("PyQt5.QtCore", pyqt5_core)
volreg_mod = types.ModuleType("fmri_rt_preproc.RTPSpy_tools.rtp_volreg")
regress_mod = types.ModuleType("fmri_rt_preproc.RTPSpy_tools.rtp_regress")


class _StubVolreg:
    def __init__(self, *args, **kwargs):
        self._motion = []

    def set_ref_vol(self, *args, **kwargs):
        return None

    def do_proc(self, *args, **kwargs):
        return None


class _StubRegress:
    def __init__(self, *args, **kwargs):
        self._vol_num = 0
        self.wait_num = 0
        self.desMtx = None
        self.reg_names = []

    def set_param(self, *args, **kwargs):
        return None

    def ready_proc(self):
        return True

    def do_proc(self, *args, **kwargs):
        self._vol_num += 1
        return None


volreg_mod.RtpVolreg = _StubVolreg
regress_mod.RtpRegress = _StubRegress
sys.modules.setdefault("fmri_rt_preproc.RTPSpy_tools.rtp_volreg", volreg_mod)
sys.modules.setdefault("fmri_rt_preproc.RTPSpy_tools.rtp_regress", regress_mod)

import rt_pipeline


class DummyCfg:
    def __init__(self, tmp_path: Path):
        self.rt_work_dir = tmp_path
        self.enable_scoring = False


class DummyRegressor:
    def __init__(self):
        self.calls = 0

    def apply(self, mc_img, volume_idx, fd_censor=0, dvars_censor=0, support_mask=None):
        self.calls += 1
        # This fixture isolates orchestration, not cumulative-GLM warmup.
        return np.asanyarray(mc_img.dataobj), True


class IdentityVoxelNormalizer:
    """Explicit identity stage for tests of unrelated processing operators."""

    mode = "zscore"
    reference = "fixed"
    ref_volumes = 2
    reference_count = 2
    ready = True
    brain_mask = None
    mean = std = count = None

    def __init__(self):
        self.output_support = None

    def transform(self, data, support_mask=None):
        data = np.asarray(data, dtype=np.float32)
        self.output_support = (np.ones(data.shape, dtype=bool) if support_mask is None
                               else np.asarray(support_mask, dtype=bool).copy())
        return np.where(self.output_support, data, 0.).astype(np.float32)

    def observe(self, data, support_mask=None):
        return None


def _mk_handler(tmp_path: Path):
    h = type("H", (), {})()
    h._next_scan_to_commit = 1
    h._last_committed_scan = 0
    h._timed_out_scans = set()
    h._inflight_scans = {1, 2, 3}
    h._pending_scans = set()
    h._result_buffer = {}
    h._scan_first_seen = {1: time.time() - 10, 2: time.time() - 10, 3: time.time() - 10}
    h._order_cv = threading.Condition()
    h._lock = threading.Lock()
    h._commit_wait_timeout_s = 0.01
    h._processed_scans = set()
    h.mark_processed = lambda s: h._processed_scans.add(s)
    h._advance_expected_scan_locked = lambda: None
    h.prev_motion = None
    h.brain_radius_mm = 50.0
    h.pre_trial_scans = 0
    h.censor_next_fd = False
    h.censor_next_dvars = False
    h.prev_mc_for_dvars = None
    h.dvars_hist = []
    h.last_dvars_val = float("nan")
    h.last_dvars_z = float("nan")
    h.dvars_mask = np.ones((2, 2, 2), dtype=bool)
    h.motion_file = tmp_path / "motion_rt.1D"
    h.fd_file = tmp_path / "fd_rt.csv"
    h.cfg = DummyCfg(tmp_path)
    h.motion_regressor = DummyRegressor()
    h.proc_src = rt_pipeline.ProcSrc()
    h.voxel_normalizer = IdentityVoxelNormalizer()
    h.scorer = None
    h.score_event_tracker = None
    h.score_queue = queue.Queue()
    return h


def test_ordered_commit_invariant_out_of_order(tmp_path):
    handler = _mk_handler(tmp_path)
    committed = []

    def fake_commit(env):
        committed.append(env.scan)
        handler._inflight_scans.discard(env.scan)
        handler.mark_processed(env.scan)
        handler._last_committed_scan = env.scan
        handler._next_scan_to_commit = env.scan + 1

    handler._commit_scan = fake_commit
    handler._drain_commit_ready = rt_pipeline.DICOMHandler._drain_commit_ready.__get__(handler)

    for scan in [3, 1, 2]:
        handler._result_buffer[scan] = rt_pipeline.ResultEnvelope(
            scan=scan,
            volume_idx=scan,
            dicom_path=Path(f"{scan}.dcm"),
            volume_timestamp=time.time(),
            success=True,
        )
    handler._drain_commit_ready()
    assert committed == [1, 2, 3]


def test_failure_path_timeout_advances_cursor(tmp_path):
    handler = _mk_handler(tmp_path)
    committed = []

    def fake_commit(env):
        committed.append((env.scan, env.success))
        handler._inflight_scans.discard(env.scan)
        handler._last_committed_scan = env.scan
        handler._next_scan_to_commit = env.scan + 1

    handler._commit_scan = fake_commit
    handler._drain_commit_ready = rt_pipeline.DICOMHandler._drain_commit_ready.__get__(handler)
    handler._result_buffer[2] = rt_pipeline.ResultEnvelope(
        scan=2,
        volume_idx=2,
        dicom_path=Path("2.dcm"),
        volume_timestamp=time.time(),
            success=True,
        )
    handler._drain_commit_ready()
    assert committed == []

    handler._result_buffer[1] = rt_pipeline.ResultEnvelope(
        scan=1,
        volume_idx=1,
        dicom_path=Path("1.dcm"),
        volume_timestamp=time.time(),
        success=False,
        error="compute failed",
    )
    handler._drain_commit_ready()
    assert committed[0] == (1, False)
    assert committed[1] == (2, True)


def test_commit_stage_fd_dvars_and_regression_progression(tmp_path):
    handler = _mk_handler(tmp_path)
    cfg = type("C", (), {})()
    called = {}

    def fake_process_volume(cfg_obj, handler_obj, dicom_path, volume_idx, raw_nii=None, volume_timestamp=None):
        called["dicom_path"] = dicom_path
        called["volume_idx"] = volume_idx
        called["raw_nii"] = raw_nii
        called["volume_timestamp"] = volume_timestamp
        return True

    orig = rt_pipeline.process_volume
    rt_pipeline.process_volume = fake_process_volume
    try:
        env = rt_pipeline.ResultEnvelope(
            scan=4,
            volume_idx=7,
            dicom_path=Path("4.dcm"),
            volume_timestamp=123.45,
            success=True,
            raw_nii=Path("raw.nii"),
        )
        assert rt_pipeline.commit_stage(cfg, handler, env) is True
    finally:
        rt_pipeline.process_volume = orig

    assert called["volume_idx"] == 7
    assert called["raw_nii"] == Path("raw.nii")


@pytest.mark.parametrize("outcome", [False, "exception", True])
def test_legacy_retries_conversion_but_never_repeats_stateful_processing(tmp_path, monkeypatch, outcome):
    handler = _mk_handler(tmp_path)
    handler.current_run = 8
    handler._next_scan_to_process = 1
    advanced, timing, prepared, processed = [], [], [], []
    handler._advance_expected_scan_locked = lambda: advanced.append(1)
    handler.speed = types.SimpleNamespace(
        process_start=lambda volume: timing.append(("start", volume)) or 12.,
        process_end=lambda volume, started: timing.append(("end", volume, started)),
        finish=lambda volume, success: timing.append(("finish", volume, success)),
    )
    monkeypatch.setattr(rt_pipeline.REGRESSOR_SETTINGS, "max_retries", 3)
    monkeypatch.setattr(rt_pipeline.time, "sleep", lambda seconds: None)

    def prepare(cfg, path, volume, speed):
        prepared.append(volume)
        return (None, None) if len(prepared) < 3 else (Path("raw.nii"), 123.)

    def process(cfg, h, path, volume, **kwargs):
        processed.append((volume, kwargs))
        if outcome == "exception":
            raise RuntimeError("Failure after normalization advanced")
        return outcome

    monkeypatch.setattr(rt_pipeline, "prepare_volume_input", prepare)
    monkeypatch.setattr(rt_pipeline, "process_volume", process)
    rt_pipeline.DICOMHandler._process_scan(handler, Path("1.dcm"), 1, 1)

    assert prepared == [1, 1, 1]
    assert processed == [(1, {"raw_nii": Path("raw.nii"), "volume_timestamp": 123.})]
    assert timing == [("start", 1), ("end", 1, 12.), ("finish", 1, outcome is True)]
    assert handler._processed_scans == {1}
    assert 1 not in handler._inflight_scans
    assert advanced == [1]


def test_legacy_conversion_exhaustion_finishes_without_entering_stateful_processing(tmp_path, monkeypatch):
    handler = _mk_handler(tmp_path)
    handler.current_run = 8
    handler._next_scan_to_process = 1
    handler.speed = None
    prepared, advanced = [], []
    handler._advance_expected_scan_locked = lambda: advanced.append(1)
    monkeypatch.setattr(rt_pipeline.REGRESSOR_SETTINGS, "max_retries", 2)
    monkeypatch.setattr(rt_pipeline.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(rt_pipeline, "prepare_volume_input",
                        lambda *args: prepared.append(1) or (None, None))

    def forbidden(*args, **kwargs):
        raise AssertionError("No converted input must not enter stateful processing")

    monkeypatch.setattr(rt_pipeline, "process_volume", forbidden)
    rt_pipeline.DICOMHandler._process_scan(handler, Path("1.dcm"), 1, 1)
    assert len(prepared) == 2
    assert handler._processed_scans == {1}
    assert 1 not in handler._inflight_scans
    assert advanced == [1]
