import queue
import threading
import time
from pathlib import Path
import sys
import types

import numpy as np

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

    def apply(self, mc_img, volume_idx, fd_censor=0, dvars_censor=0):
        self.calls += 1
        return np.asanyarray(mc_img.dataobj), self.calls >= 2


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
    h.voxel_normalizer = rt_pipeline.RTPStyleVoxelNormalizer(ref_volumes=1)
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
