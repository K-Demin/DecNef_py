"""Exercise the actual support method without requiring the RTPSpy Qt UI.

Native AFNI checks additionally run when the bundled shared library is usable.
"""
import ast
import ctypes
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "fmri_rt_preproc" / "RTPSpy_tools" / "rtp_volreg.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
volreg_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RtpVolreg")
methods = [node for node in volreg_class.body if isinstance(node, ast.FunctionDef)
           and node.name in {"resample_support", "setup_libfuncs"}]
namespace = {"np": np, "ctypes": ctypes,
             "librtp_path": str(SOURCE.with_name("librtp.dylib" if sys.platform == "darwin" else "librtp.so"))}
exec(compile(ast.Module(body=methods, type_ignores=[]), str(SOURCE), "exec"), namespace)
resample_support = namespace["resample_support"]


def _volreg():
    return SimpleNamespace(nx=4, ny=5, nz=6, dx=2., dy=3., dz=4.,
                           ax1=2, ax2=0, ax3=1, regmode="heptic",
                           regmode_dict={"linear": 1, "heptic": 5},
                           _motion=np.array([[10, 20, 30, 12, 2, 6]], dtype=np.float32))


def test_support_reuses_estimated_native_motion_and_preserves_input():
    volreg = _volreg()
    mode_calls = []
    volreg.THD_rota_method = mode_calls.append
    source = np.ones((4, 5, 6), np.float32)
    source[2, 3, 4] = .5

    def rotate(*args):
        assert args[:6] == (4, 5, 6, 2., 3., 4.)
        assert args[7:13:2] == (2, 0, 1)
        np.testing.assert_allclose(args[8:13:2], np.deg2rad([10, 20, 30]))
        assert args[13:] == (-1, 2., 6., 12.)
        buffer = np.ctypeslib.as_array(args[6], shape=(4 * 5 * 6,)).reshape(6, 5, 4)
        np.testing.assert_array_equal(buffer, source.transpose(2, 1, 0))
        buffer[:, :, 0] = 0  # Simulated exterior from native interpolation.

    volreg.THD_rota_vol = rotate
    result = resample_support(volreg, source, 0)
    assert mode_calls == [1, 5]
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result[0], 0)
    np.testing.assert_array_equal(result[1:], source[1:])
    np.testing.assert_array_equal(source[0], 1)
    np.testing.assert_array_equal(volreg._motion[0], [10, 20, 30, 12, 2, 6])


def test_support_restores_interpolator_after_native_failure():
    volreg = _volreg()
    modes = []
    volreg.THD_rota_method = modes.append

    def fail(*args):
        raise RuntimeError("native failure")

    volreg.THD_rota_vol = fail
    with pytest.raises(RuntimeError, match="native failure"):
        resample_support(volreg, np.ones((4, 5, 6), np.float32), 0)
    assert modes == [1, 5]


@pytest.mark.parametrize("problem", ["shape", "nan", "range", "index", "motion"])
def test_invalid_support_or_motion_is_rejected_before_native_call(problem):
    volreg = _volreg()
    source = np.ones((4, 5, 6), np.float32)
    index = 0
    if problem == "shape":
        source = source[:, :, :-1]
    elif problem == "nan":
        source[0, 0, 0] = np.nan
    elif problem == "range":
        source[0, 0, 0] = -1
    elif problem == "index":
        index = -1
    else:
        volreg._motion[0, 0] = np.nan
    with pytest.raises((ValueError, IndexError)):
        resample_support(volreg, source, index)


def test_native_afni_support_identity_and_integer_translation():
    if sys.platform not in {"linux", "darwin"}:
        pytest.skip("The bundled AFNI shared library requires Linux or macOS.")
    volreg = _volreg()
    try:
        namespace["setup_libfuncs"](volreg)
    except OSError as exc:
        pytest.skip(f"Bundled AFNI library unavailable: {exc}")
    volreg.nx, volreg.ny, volreg.nz = 16, 18, 12
    source = np.ones((16, 18, 12), np.float32)
    volreg._motion[:] = 0
    np.testing.assert_allclose(resample_support(volreg, source, 0), 1., atol=1e-5)
    volreg._motion[0, 3] = volreg.dz  # Native z translation of one voxel.
    result = resample_support(volreg, source, 0)
    assert np.count_nonzero(result >= 1. - 1e-5) == 16 * 18 * 11
    assert np.count_nonzero(result <= 1e-5) == 16 * 18
