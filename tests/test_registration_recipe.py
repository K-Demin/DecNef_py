"""Contracts for registration recipes; these tests do not execute native ANTs."""
import shlex

import pytest

from fmri_rt_preproc.registration import (
    ANATOMICAL_RECIPE, BRIDGE_RECIPE,
    build_epi_bridge_registration, build_epi_t1_registration,
)


def values(args, flag):
    return [args[i + 1] for i, value in enumerate(args) if value == flag]


@pytest.fixture
def paths(tmp_path):
    return dict(epi=tmp_path / "session EPI.nii", t1=tmp_path / "anatomical T1.nii",
                epi_mask=tmp_path / "EPI mask.nii", t1_mask=tmp_path / "T1 mask.nii",
                prefix=tmp_path / "native fit_")


@pytest.mark.parametrize("direction", ["epi_to_t1", "t1_to_epi"])
def test_direction_consistent_for_metrics_masks_initialization_and_output_grids(paths, direction):
    plan = build_epi_t1_registration(**paths, direction=direction, initialization="center_of_mass")
    fixed, moving = (paths["t1"], paths["epi"]) if direction == "epi_to_t1" else (paths["epi"], paths["t1"])
    fixed_mask, moving_mask = (paths["t1_mask"], paths["epi_mask"]) if direction == "epi_to_t1" else (paths["epi_mask"], paths["t1_mask"])
    assert (plan.fixed, plan.moving) == (fixed, moving)
    assert values(plan.args, "--initial-moving-transform") == [f"[{fixed},{moving},1]"]
    assert values(plan.args, "--metric") == [f"MI[{fixed},{moving},1,32,None,1]"] * 3
    assert values(plan.args, "--masks") == [f"[{fixed_mask},{moving_mask}]"] * 3
    fwd, inv, warped, inverse_warped = plan.native_outputs
    assert plan.output_references == {warped: fixed, inverse_warped: moving}
    canonical = plan.semantic_outputs
    assert canonical["epi2t1_Composite.h5"] == (fwd if direction == "epi_to_t1" else inv)
    assert canonical["epi2t1_InverseComposite.h5"] == (inv if direction == "epi_to_t1" else fwd)
    assert plan.output_references[canonical["epi2t1_Warped.nii"]] == paths["t1"]
    assert plan.output_references[canonical["epi2t1_InverseWarped.nii"]] == paths["epi"]
    assert shlex.split(shlex.join(plan.args)) == plan.args  # Space-containing paths stay single arguments.


@pytest.mark.parametrize("direction", ["epi_to_t1", "t1_to_epi"])
def test_historical_affine_and_full_syn_preserved_in_both_directions(paths, direction):
    plan = build_epi_t1_registration(**paths, direction=direction)
    assert values(plan.args, "--transform") == ["Rigid[0.1]", "Affine[0.1]", "SyN[0.1,1,0]"]
    assert values(plan.args, "--convergence") == ["[1000x500x250x100,1e-7,15]",
        "[1000x500x250x100,1e-7,15]", "[100x70x50x20,1e-7,15]"]
    assert values(plan.args, "--shrink-factors") == ["8x4x2x1", "8x4x2x1", "6x4x2x1"]
    assert values(plan.args, "--smoothing-sigmas") == ["3x2x1x0vox"] * 3
    assert plan.provenance["recipe"] == ANATOMICAL_RECIPE
    assert ANATOMICAL_RECIPE == "rigid_affine_syn_dense_v1"
    for index, stage in enumerate(plan.provenance["stages"]):
        assert stage["sampling"] == "None,1"
        assert stage["transform"] == values(plan.args, "--transform")[index]
        assert stage["convergence"] == values(plan.args, "--convergence")[index]
        assert stage["shrink_factors"] == values(plan.args, "--shrink-factors")[index]
        assert stage["smoothing_sigmas"] == values(plan.args, "--smoothing-sigmas")[index]


@pytest.mark.parametrize("initialization", ["identity", "auto", "IDENTITY"])
@pytest.mark.parametrize("direction", ["epi_to_t1", "t1_to_epi"])
def test_identity_and_compatibility_alias_with_syn_disabled(paths, initialization, direction):
    plan = build_epi_t1_registration(**paths, initialization=initialization, direction=direction, syn=False)
    assert values(plan.args, "--initial-moving-transform") == ["Identity"]
    assert plan.provenance["initialization"] == "identity"
    assert values(plan.args, "--transform") == ["Rigid[0.1]", "Affine[0.1]"]
    assert all(metric.endswith(",1,32,None,1]") for metric in values(plan.args, "--metric"))
    assert plan.fixed == (paths["t1"] if direction == "epi_to_t1" else paths["epi"])


@pytest.mark.parametrize("direction,expected", [("epi_to_t1", "[NULL,{epi_mask}]"),
                                               ("t1_to_epi", "[{epi_mask},NULL]")])
def test_optional_anatomical_mask_retains_correct_position(paths, direction, expected):
    paths["t1_mask"] = None
    plan = build_epi_t1_registration(**paths, direction=direction)
    assert values(plan.args, "--masks") == [expected.format(**paths)] * 3


@pytest.mark.parametrize("overrides", [{"direction": "wrong"}, {"initialization": "wrong"},
                                       {"syn": "false"}, {"syn": 1}])
def test_invalid_settings_fail_before_command_execution(paths, overrides):
    with pytest.raises(ValueError):
        build_epi_t1_registration(**paths, **overrides)


def test_bridge_fixes_session_slab_and_exposes_inverse_as_session_to_reference(paths):
    plan = build_epi_bridge_registration(paths["epi"], paths["t1"], paths["epi_mask"], None,
                                         paths["prefix"], initialization="center_of_mass")
    assert (plan.fixed, plan.moving) == (paths["epi"], paths["t1"])
    assert values(plan.args, "--masks") == [f"[{paths['epi_mask']},NULL]"]
    assert values(plan.args, "--transform") == ["Rigid[0.1]"]
    assert values(plan.args, "--shrink-factors") == ["2x2x1"]
    assert values(plan.args, "--initial-moving-transform") == [f"[{paths['epi']},{paths['t1']},1]"]
    assert plan.semantic_outputs["session_to_reference_Composite.h5"] == plan.native_outputs[1]
    assert plan.semantic_outputs["session_to_reference_InverseComposite.h5"] == plan.native_outputs[0]
    assert plan.output_references[plan.semantic_outputs["session_to_reference_Warped.nii"]] == paths["t1"]
    assert plan.output_references[plan.semantic_outputs["session_to_reference_InverseWarped.nii"]] == paths["epi"]
    assert plan.provenance["recipe"] == BRIDGE_RECIPE
    assert values(plan.args, "--metric") == [f"MI[{paths['epi']},{paths['t1']},1,32,Regular,0.2]"]
    assert plan.provenance["stages"][0]["sampling"] == "Regular,0.2"
