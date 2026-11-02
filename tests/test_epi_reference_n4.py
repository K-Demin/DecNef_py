"""Reference-only N4 must not change the motion target, grid or cache meaning."""
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import fmri_rt_preproc.epi_reference as reference


@pytest.fixture
def images(tmp_path):
    epi, mask, output = (tmp_path / name for name in
                         ("mean.nii.gz", "brain_mask.nii.gz", "prep/epi_registration_reference.nii"))
    affine = np.diag([1.5, 1.5, 2., 1.])
    affine[:3, 3] = [-80., 15., 9.]
    values = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3) - 3.
    image = nib.Nifti1Image(values, affine)
    qform = affine.copy()
    qform[0, 3] += .5
    image.set_qform(qform, 1)
    image.set_sform(affine, 4)
    nib.save(image, epi)
    selected = np.zeros(values.shape, np.uint8)
    selected[1:3, 1:4, :] = 1
    nib.save(nib.Nifti1Image(selected, affine), mask)
    return epi, mask, output


def fake_n4(commands, mutate=None):
    def run(command):
        commands.append(command)
        original = nib.load(command[command.index("-i") + 1])
        output = Path(command[command.index("-o") + 1])
        values = np.asarray(original.dataobj) / 1.25
        affine = original.affine.copy()
        if mutate is not None:
            values, affine = mutate(values, affine)
        # Simulate a native writer with default form codes instead of inheriting
        # the source's distinct qform/sform: the helper restores those exactly.
        nib.save(nib.Nifti1Image(values, affine), output)
    return run


def test_n4_is_a_separate_float_copy_preserving_original_data_and_forms(images):
    epi, mask, output = images
    originals = {path: path.read_bytes() for path in (epi, mask)}
    commands = []
    assert reference.prepare_epi_registration_reference(epi, mask, output, fake_n4(commands)) == output
    source, corrected = nib.load(epi), nib.load(output)
    assert len(commands) == 1
    assert commands[0] == ["N4BiasFieldCorrection", "-d", "3", "-i", str(epi),
                           "-x", str(mask), "-s", "2", "-b", "[200]", "-c",
                           "[50x50x30x20,1e-7]", "-r", "0", "-o", str(output)]
    assert all(path.read_bytes() == content for path, content in originals.items())
    np.testing.assert_allclose(np.asarray(corrected.dataobj), np.asarray(source.dataobj) / 1.25)
    assert np.asarray(source.dataobj).min() < 0
    assert corrected.get_data_dtype() == np.dtype("float32")
    np.testing.assert_array_equal(corrected.affine, source.affine)
    np.testing.assert_array_equal(corrected.get_qform(), source.get_qform())
    np.testing.assert_array_equal(corrected.get_sform(), source.get_sform())
    assert corrected.header["qform_code"] == 1
    assert corrected.header["sform_code"] == 4
    provenance = reference.validate_epi_registration_reference(epi, mask, output)
    assert provenance["recipe"] == reference.EPI_REFERENCE_N4_RECIPE
    assert provenance["source"]["path"] == str(epi.resolve())
    assert reference.prepare_epi_registration_reference(
        epi, mask, output, lambda command: pytest.fail("Valid cache reran N4")) == output


@pytest.mark.parametrize("changed", ["source", "mask", "output", "recipe"])
def test_stale_reference_is_rejected_without_overwrite(images, monkeypatch, changed):
    epi, mask, output = images
    reference.prepare_epi_registration_reference(epi, mask, output, fake_n4([]))
    if changed == "recipe":
        monkeypatch.setattr(reference, "EPI_REFERENCE_N4_RECIPE", "different_recipe")
    else:
        path = {"source": epi, "mask": mask, "output": output}[changed]
        image = nib.load(path)
        values = np.asarray(image.dataobj).copy()
        values[0, 0, 0] = 1. if changed == "mask" else 321.
        nib.save(nib.Nifti1Image(values, image.affine), path)
    before = output.read_bytes()
    with pytest.raises(ValueError, match="changed.*fresh preparation"):
        reference.prepare_epi_registration_reference(epi, mask, output, lambda c: pytest.fail("Stale cache reran"))
    assert output.read_bytes() == before


@pytest.mark.parametrize("state", ["image_only", "sidecar_only", "malformed_sidecar"])
def test_incomplete_cache_is_not_reused(images, state):
    epi, mask, output = images
    reference.prepare_epi_registration_reference(epi, mask, output, fake_n4([]))
    sidecar = reference.epi_reference_metadata_path(output)
    if state == "image_only":
        sidecar.unlink()
    elif state == "sidecar_only":
        output.unlink()
    else:
        sidecar.write_text("not json")
    with pytest.raises(ValueError, match="fresh preparation"):
        reference.prepare_epi_registration_reference(epi, mask, output, lambda c: pytest.fail("Incomplete cache reran"))


@pytest.mark.parametrize("invalid", ["4d_epi", "nan_epi", "4d_mask", "nan_mask", "fractional_mask", "empty_mask", "wrong_grid", "no_positive_signal"])
def test_invalid_inputs_do_not_reach_n4(images, invalid):
    epi, mask, output = images
    path = epi if invalid in {"4d_epi", "nan_epi", "no_positive_signal"} else mask
    image = nib.load(path)
    values = np.asarray(image.dataobj, dtype=np.float32).copy()
    affine = image.affine.copy()
    if invalid.startswith("4d"):
        values = values[..., None]
    elif invalid.startswith("nan"):
        values[0, 0, 0] = np.nan
    elif invalid == "fractional_mask":
        values[0, 0, 0] = .5
    elif invalid == "empty_mask":
        values.fill(0)
    elif invalid == "no_positive_signal":
        values.fill(-1)
    elif invalid == "wrong_grid":
        affine[0, 3] += 2.
    nib.save(nib.Nifti1Image(values, affine), path)
    with pytest.raises(ValueError):
        reference.prepare_epi_registration_reference(epi, mask, output, lambda c: pytest.fail("Invalid input reached N4"))
    assert not output.parent.exists()


@pytest.mark.parametrize("target", ["epi", "mask"])
def test_input_cannot_be_replaced(images, target):
    epi, mask, _ = images
    output = epi if target == "epi" else mask
    with pytest.raises(ValueError, match="separate"):
        reference.prepare_epi_registration_reference(epi, mask, output, lambda c: pytest.fail("Overwrote input"))


@pytest.mark.parametrize("invalid", ["nonfinite", "wrong_grid", "4d"])
def test_bad_native_output_cannot_be_marked_complete(images, invalid):
    epi, mask, output = images
    def mutate(values, affine):
        if invalid == "nonfinite":
            values[0, 0, 0] = np.nan
        elif invalid == "wrong_grid":
            affine[1, 3] += 10.
        else:
            values = values[..., None]
        return values, affine
    with pytest.raises(ValueError):
        reference.prepare_epi_registration_reference(epi, mask, output, fake_n4([], mutate))
    assert not reference.epi_reference_metadata_path(output).exists()


def test_missing_native_output_cannot_be_marked_complete(images):
    epi, mask, output = images
    with pytest.raises(RuntimeError, match="did not produce"):
        reference.prepare_epi_registration_reference(epi, mask, output, lambda c: None)
    assert not reference.epi_reference_metadata_path(output).exists()


def test_metadata_basename_accepts_compressed_images():
    assert reference.epi_reference_metadata_path("ref.nii.gz") == Path("ref.json")
    assert reference.epi_reference_metadata_path("ref.nii") == Path("ref.json")
