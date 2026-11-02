"""Probability feedback consumes pipeline-normalized features without a second z-score."""
import math
import sys
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

import rt_nf_events_prob_parallel as probability


@pytest.fixture
def scorer(tmp_path):
    weights = np.zeros((2, 2, 2), dtype=np.float32)
    weights[0, 0, 0], weights[1, 0, 0] = 2, -3
    path = tmp_path / "weights.nii"
    nib.save(nib.Nifti1Image(weights, np.eye(4)), path)
    return probability._make_direct_scorer(0.7)(path)


def features(a, b):
    result = np.zeros((2, 2, 2), dtype=np.float32)
    result[0, 0, 0], result[1, 0, 0] = a, b
    return result


def test_direct_scoring_keeps_feature_values_without_warmup_or_renormalization(scorer):
    # Consecutive identical features still produce the same decision; a second
    # z-score would suppress them or require another baseline period.
    for _ in range(12):
        data = features(2.5, -1.2)
        scorer.accumulate_baseline(data)
        assert scorer.score_from_array(data) == pytest.approx(2 * 2.5 - 3 * -1.2 + .7)
    assert scorer.baseline_mean is None
    assert scorer.baseline_std is None


def test_nonready_and_repeated_calls_do_not_score(scorer):
    data = features(1, 2)
    assert math.isnan(scorer.score_from_array(data))
    scorer.accumulate_baseline(data)
    assert scorer.score_from_array(data) == pytest.approx(-3.3)
    assert math.isnan(scorer.score_from_array(data))


def test_original_stream_cannot_consume_pending_normalized_features(scorer):
    normalized = features(.2, -.4)
    original = features(800, 1100)
    assert math.isnan(scorer.score_from_array(original))
    scorer.accumulate_baseline(normalized)
    assert math.isnan(scorer.score_from_array(original))
    assert scorer.score_from_array(normalized) == pytest.approx(2 * .2 - 3 * -.4 + .7)


def test_new_ready_volume_replaces_an_unconsumed_previous_one(scorer):
    previous, current = features(1, 2), features(3, 4)
    scorer.accumulate_baseline(previous)
    scorer.accumulate_baseline(current)
    assert math.isnan(scorer.score_from_array(previous))
    assert scorer.score_from_array(current) == pytest.approx(-5.3)


@pytest.mark.parametrize("mode", ["zscore", "mean"])
def test_spawned_pipeline_uses_direct_scorer_and_preserves_central_json_mode(monkeypatch, mode):
    settings = SimpleNamespace(voxel_norm_mode="untouched", voxel_norm_ref_volumes=30)
    observed = []
    direct_type = object()
    fake_pipeline = SimpleNamespace(REGRESSOR_SETTINGS=settings, DecoderScorer=None)
    fake_pipeline.run_rt_pipeline = lambda cfg, queue: observed.append((cfg, queue, fake_pipeline.DecoderScorer))
    monkeypatch.setitem(sys.modules, "rt_pipeline", fake_pipeline)
    monkeypatch.setattr(probability, "_make_direct_scorer", lambda bias: direct_type if bias == .7 else None)

    def forbidden(*args, **kwargs):
        raise AssertionError("Historical second voxel z-score must never be active")

    monkeypatch.setattr(probability, "_make_runwise_zscore_scorer", forbidden)
    payload = {"_decoder_intercept": .7, "voxel_norm_mode": mode, "voxel_norm_ref_volumes": 41}
    probability._run_pipeline_with_settings("cfg", "queue", payload)
    assert observed == [("cfg", "queue", direct_type)]
    assert settings.voxel_norm_mode == mode
    assert settings.voxel_norm_ref_volumes == 41
    assert payload["_decoder_intercept"] == .7


@pytest.mark.parametrize("intercept", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_intercept_is_rejected(intercept):
    with pytest.raises(ValueError, match="finite"):
        probability._make_direct_scorer(intercept)


def test_feedback_still_maps_the_mean_decision_once():
    decisions = np.array([-1., 3.])
    expected = 100 / (1 + math.exp(-decisions.mean()))
    result = probability._score_to_percentile(float(decisions.mean()), {"mean": 999, "std": .001})
    assert result == pytest.approx(expected)
    assert result != pytest.approx(np.mean([100 / (1 + math.exp(-value)) for value in decisions]))
    assert math.isnan(probability._score_to_percentile(float("nan"), None))
