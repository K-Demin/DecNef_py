"""Keep nuisance sampling on native MC despite final-grid temporal responses."""
import ast
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

import test_rt_pipeline_parallel  # Install the existing headless native-library stubs.
import rt_pipeline as rt


def test_first_mc_nonzero_fallback_freezes_native_selection_only_once():
    source_roi = np.ones((1, 1, 4), bool)
    handler = SimpleNamespace(native_nuisance_masks={"WM": source_roi.copy()},
        _infer_native_nuisance_mask=True,
        # These unrelated final-grid masks must not select native nuisance data.
        final_brain_mask=np.zeros((2, 3, 4), bool),
        processing_mask=np.zeros((2, 3, 4), bool),
        motion_regressor=SimpleNamespace(output_support=np.zeros((2, 3, 4), bool)))
    acquired = np.ones(source_roi.shape, bool)
    first = np.array([[[0., 10., 20., 0.]]], np.float32)
    assert rt.native_nuisance_means(handler, first, acquired) == {"WM": 15.}
    frozen = np.array([[[False, True, True, False]]])
    np.testing.assert_array_equal(handler.native_nuisance_masks["WM"], frozen)
    assert not handler._infer_native_nuisance_mask

    later = np.array([[[100., 12., 24., 99.]]], np.float32)
    assert rt.native_nuisance_means(handler, later, acquired) == {"WM": 18.}
    partial = acquired.copy()
    partial[0, 0, 1] = False
    assert rt.native_nuisance_means(handler, later, partial) == {"WM": 24.}
    # Current coverage can return, independently of conservative GLM history.
    assert rt.native_nuisance_means(handler, later, acquired) == {"WM": 18.}
    later[0, 0, 1] = 0
    assert rt.native_nuisance_means(handler, later, acquired) == {"WM": 12.}
    np.testing.assert_array_equal(handler.native_nuisance_masks["WM"], frozen)
    np.testing.assert_array_equal(source_roi, True)


def test_native_means_use_current_observations_and_preserve_invalid_roi_policy():
    masks = {"GS": np.array([[[True, True, True, True]]]),
             "WM": np.array([[[True, True, False, False]]]),
             "Vent": np.array([[[False, False, True, False]]])}
    handler = SimpleNamespace(native_nuisance_masks={k: v.copy() for k, v in masks.items()},
        _infer_native_nuisance_mask=False,
        final_brain_mask=np.zeros((3, 2, 5), bool),
        processing_mask=np.zeros((3, 2, 5), bool))
    data = np.array([[[0., 10., np.nan, 30.]]], np.float32)
    support = np.array([[[True, True, False, True]]])
    result = rt.native_nuisance_means(handler, data, support)
    assert result["GS"] == pytest.approx(40 / 3)
    assert result["WM"] == 5.  # Acquired zeros contribute to the mean.
    assert np.isnan(result["Vent"])
    support[0, 0, 2] = True
    result = rt.native_nuisance_means(handler, data, support)
    assert np.isnan(result["GS"]) and np.isnan(result["Vent"])
    assert result["WM"] == 5.
    for label, mask in masks.items():
        np.testing.assert_array_equal(handler.native_nuisance_masks[label], mask)
    assert rt.native_nuisance_means(SimpleNamespace(), data, support) is None
    assert rt.native_nuisance_means(SimpleNamespace(native_nuisance_masks={}), data, support) == {}


@pytest.mark.parametrize("regression", [False, True])
def test_nuisance_initialization_ignores_stale_unused_masks_without_glm(tmp_path, monkeypatch, regression):
    """Execute the actual initializer's nuisance-loading block without workers/UI."""
    filename = Path(rt.__file__)
    tree = ast.parse(filename.read_text(encoding="utf-8"))
    handler_class = next(node for node in tree.body
                         if isinstance(node, ast.ClassDef) and node.name == "DICOMHandler")
    initializer = next(node for node in handler_class.body
                       if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    start = next(index for index, node in enumerate(initializer.body)
                 if isinstance(node, ast.Assign) and any(
                     isinstance(target, ast.Attribute) and target.attr == "native_nuisance_masks"
                     for target in node.targets))
    block = ast.Module(body=initializer.body[start:start + 3], type_ignores=[])
    assert isinstance(block.body[-1], ast.If)
    native, stale = tmp_path / "epi.nii", tmp_path / "WM.nii"
    nib.save(nib.Nifti1Image(np.ones((3, 4, 5)), np.eye(4)), native)
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)), stale)
    loaded = []
    real_load = nib.load

    def load(path, *args, **kwargs):
        loaded.append(Path(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(nib, "load", load)
    handler = SimpleNamespace(anatomical_brain_mask=None)
    namespace = dict(self=handler, cfg=SimpleNamespace(rt_motion_ref_epi=native),
        used={"WM": stale}, REGRESSOR_SETTINGS=SimpleNamespace(enable_motion_regression=regression),
        nib=nib, np=np, require_same_grid=rt.require_same_grid)
    compiled = compile(block, str(filename), "exec")
    if regression:
        with pytest.raises(ValueError, match="grids differ"):
            exec(compiled, namespace)
        assert loaded == [native, stale]
    else:
        exec(compiled, namespace)
        assert loaded == []
        assert handler.native_nuisance_masks == {}
