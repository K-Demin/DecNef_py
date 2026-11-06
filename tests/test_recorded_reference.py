"""Finite normal-runtime processing and recorded wrapper boundaries."""
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import pytest

import test_rt_pipeline_parallel  # Existing headless dependency stand-ins.
import rt_pipeline as rt
import rs_offline_reference as recorded
from test_rt_pipeline_cli_settings import run_main


class Handler:
    def __init__(self):
        self._lock = RLock()
        self._processed_scans = set()
        self.pending = []
        self.events = []
        self._qc_export_thread = SimpleNamespace(join=lambda: self.events.append("export_joined"))
    def enqueue_path(self, path, backlog=False):
        assert backlog
        self.pending.append(path)
    def submit_pending(self):
        if not self.pending: return False
        path = self.pending.pop(0)
        self._processed_scans.add(rt.parse_dicom_name(path.name)[2])
        self.events.append(path.name)
        return True
    def stop(self): self.events.append("stopped")


def test_recorded_snapshot_uses_ordered_workers_and_joins_final_export():
    handler = Handler()
    paths = [Path(f"001_000008_{index:06d}.dcm") for index in range(1, 4)]
    rt._process_recorded_snapshot(handler, paths)
    assert handler._processed_scans == {1, 2, 3}
    assert handler.events[-2:] == ["stopped", "export_joined"]


@pytest.mark.parametrize("indices", [[], [1, 3], [1, 1]])
def test_invalid_fixed_snapshot_still_stops_and_joins(indices):
    handler = Handler()
    with pytest.raises(ValueError, match="nonempty, unique and contiguous"):
        rt._process_recorded_snapshot(handler, [Path(f"001_000008_{i:06d}.dcm") for i in indices])
    assert handler.events == ["stopped", "export_joined"]


def test_recorded_runtime_never_constructs_observer_or_activates_online(tmp_path, monkeypatch):
    for i in range(1, 4): (tmp_path / f"001_000008_{i:06d}.dcm").write_bytes(b"recorded-input")
    settings = SimpleNamespace(analysis_space="t1", enable_biopac_physio=False,
                               biopac_start_online_only=False)
    monkeypatch.setattr(rt, "REGRESSOR_SETTINGS", settings)
    for name in ("ensure_fieldmap_only", "validate_run_provenance", "validate_regression_status_schema"):
        monkeypatch.setattr(rt, name, lambda *args: None)
    monkeypatch.setattr(rt, "resolve_decoder_template", lambda cfg: tmp_path / "decoder.nii")
    monkeypatch.setattr(rt, "load_reference_score_stats", lambda *args: None)
    monkeypatch.setattr(rt, "write_session_metadata", lambda *args: None)
    import fmri_rt_preproc.session_alignment as alignment
    def prepare_session(cfg, settings, prepare_fieldmap=None):
        if prepare_fieldmap is not None:
            prepare_fieldmap(cfg)
    monkeypatch.setattr(alignment, "ensure_nf_session", prepare_session)
    handler = Handler()
    calls = []
    monkeypatch.setattr(rt, "DICOMHandler", lambda cfg, **kwargs: (calls.append(kwargs), handler)[1])
    monkeypatch.setattr(rt, "Observer", lambda: pytest.fail("Recorded mode must not construct a watchdog"))
    cfg = SimpleNamespace(recorded_only=True, enable_scoring=True, decoder_template=Path("decoder.nii"),
        incoming_dir=tmp_path, rt_work_dir=tmp_path, run="8", reference_score_run=None)
    rt.run_rt_pipeline(cfg)
    assert calls[0]["start_biopac"] is False
    assert handler._processed_scans == {1, 2, 3}
    assert handler.events[-1] == "export_joined"


def test_recorded_runtime_rejects_live_physio_before_processing(monkeypatch):
    monkeypatch.setattr(rt, "REGRESSOR_SETTINGS", SimpleNamespace(analysis_space="t1", enable_biopac_physio=True))
    with pytest.raises(ValueError, match="live BIOPAC"):
        rt.run_rt_pipeline(SimpleNamespace(recorded_only=True))


def test_recorded_cli_is_opt_in(tmp_path, monkeypatch):
    _, default = run_main(tmp_path, monkeypatch)
    assert default.recorded_only is False
    _, batch = run_main(tmp_path, monkeypatch, flags=("--recorded-only",))
    assert batch.recorded_only is True


def test_wrapper_inventory_requires_complete_contiguous_available_snapshot(tmp_path):
    for i in (1, 2, 3): (tmp_path / f"001_000008_{i:06d}.dcm").write_bytes(bytes([i]))
    (tmp_path / "001_000006_000001.dcm").write_bytes(b"fieldmap")
    rows = recorded.snapshot(tmp_path, 8)
    assert len(rows) == 3
    assert [Path(row["path"]).name for row in rows] == [f"001_000008_{i:06d}.dcm" for i in (1, 2, 3)]
    (tmp_path / "001_000008_000002.dcm").unlink()
    with pytest.raises(ValueError, match="contiguous"):
        recorded.snapshot(tmp_path, 8)


def test_wrapper_explicit_fieldmap_series_and_no_implicit_baseline_rewrite():
    args = recorded.build_parser().parse_args([
        "--sub", "55555", "--day", "offline", "--incoming-root", "dcm",
        "--settings-file", "sara_settings_v2.json", "--decoder-template", "decoder.nii",
        "--run", "8", "6", "7", "--run", "13", "11", "12"])
    assert args.run == [[8, 6, 7], [13, 11, 12]]
    assert args.pool_name == "pooled_reference"
    assert args.test_speed is False
    assert not hasattr(args, "baseline_trs")


def test_batch_worker_disables_only_viewer_without_mutating_live_settings():
    settings = SimpleNamespace(enable_volume_streamer=True, voxel_norm_ref_volumes=30,
                               voxel_norm_mode="zscore", analysis_space="t1")
    original = vars(settings).copy()
    payload = recorded._worker_settings(settings)
    assert payload == {**original, "enable_volume_streamer": False}
    assert vars(settings) == original
