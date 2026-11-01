"""Cached per-volume equivalent of smooth_masked in the supplied volume_smooth.py."""
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
from .analysis_mask import final_output_kind, prepare_analysis_mask


class MaskedGaussianSmoother:
    """Normalize convolution within the experiment mask and available coverage.

    Buffers are reused, so call only from the ordered processing stage. The
    returned array belongs to this instance and is overwritten by the next call.
    Acquisition support is supplied independently of signal values: an acquired
    residual may legitimately be zero or negative.
    """

    def __init__(self, mask, affine, fwhm_mm):
        self.mask = np.asarray(mask, dtype=bool).copy()
        self.affine = np.asarray(affine, dtype=float).copy()
        fwhm = float(fwhm_mm)
        if not np.isfinite(fwhm) or fwhm < 0:
            raise ValueError("FWHM must be finite and nonnegative.")
        if self.mask.ndim != 3 or not self.mask.any():
            raise ValueError("Smoothing mask must be a nonempty 3D binary mask.")
        sizes = np.linalg.norm(self.affine[:3, :3], axis=0)
        if not np.isfinite(sizes).all() or np.any(sizes <= 0):
            raise ValueError("Invalid smoothing voxel sizes.")
        self.output_kind = "smooth" if fwhm > 0 else "masked"
        self.output = np.zeros(self.mask.shape, dtype=np.float64)
        self._effective_mask = np.empty_like(self.mask)
        if fwhm == 0:
            return  # Mask-only processing: no Gaussian setup or filtering.
        self.sigma = fwhm / (np.sqrt(8 * np.log(2)) * sizes)
        density = self._filter(self.mask.astype(np.float64))
        self.bad = density[self.mask] < 1e-6
        self.denominator = np.where(self.bad, 1., density[self.mask])
        self.buffer = np.zeros(self.mask.shape, dtype=np.float64)
        self.filtered = np.empty_like(self.buffer)
        self._support_density = np.empty_like(self.buffer)

    def _filter(self, data, output=None):
        return gaussian_filter(data, sigma=self.sigma, output=output,
                               mode="constant", cval=0., truncate=4.)

    def apply(self, data, support_mask=None):
        """Process one volume; optional binary support must use the same grid.

        Missing samples are excluded from both the Gaussian numerator and its
        density normalization, and remain zero in the output. With no support
        supplied, the original cached experiment-mask normalization is reused.
        """
        data = np.asarray(data)
        if data.shape != self.mask.shape:
            raise ValueError("Smoothing data and mask shapes differ.")
        active = self.mask
        if support_mask is not None:
            support = np.asarray(support_mask)
            if support.shape != self.mask.shape:
                raise ValueError("Smoothing support and mask shapes differ.")
            if support.dtype.kind != "b" and (
                    support.dtype.kind not in "iuf" or
                    not np.isfinite(support).all() or
                    not np.all((support == 0) | (support == 1))):
                raise ValueError("Smoothing support mask must be finite and binary (0/1).")
            np.logical_and(self.mask, support, out=self._effective_mask)
            active = self._effective_mask

        values = data[active]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite values inside smoothing mask.")
        self.output.fill(0.)
        if self.output_kind == "masked":
            self.output[active] = values
            return self.output
        self.buffer.fill(0.)
        if values.size == 0:
            return self.output

        bad, denominator = self.bad, self.denominator
        if support_mask is not None and not np.array_equal(active, self.mask):
            self.buffer[active] = 1.
            self._filter(self.buffer, output=self._support_density)
            density = self._support_density[active]
            bad = density < 1e-6
            denominator = np.where(bad, 1., density)
            self.buffer.fill(0.)

        self.buffer[active] = values
        self._filter(self.buffer, output=self.filtered)
        result = self.filtered[active] / denominator
        result[bad] = values[bad]
        self.output[active] = result
        return self.output


def prepare_smoother(cfg, fwhm_mm, analysis_space, final_reference, run_command,
                     mask_type="whole_brain", custom_file=None, custom_space="final"):
    """Prepare the experiment mask once; zero FWHM may still apply masking."""
    kind = final_output_kind(fwhm_mm, mask_type)
    if kind is None:
        return None
    folder = cfg.rt_work_dir / kind
    folder.mkdir(parents=True, exist_ok=True)
    mask, affine = prepare_analysis_mask(cfg, mask_type, analysis_space, final_reference,
                                        folder, run_command, custom_file, custom_space)
    processor = MaskedGaussianSmoother(mask, affine, fwhm_mm)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), str(folder / "mask.nii"))
    return processor


def smooth_file(smoother, source, destination, support_mask=None):
    """Save atomically, using optional acquisition support in the source grid."""
    from pathlib import Path

    img = nib.load(str(source))
    if img.shape != smoother.mask.shape or not np.allclose(img.affine, smoother.affine, atol=1e-4, rtol=0):
        raise ValueError("Final BOLD and smoothing mask grids differ.")
    data = smoother.apply(np.asarray(img.dataobj), support_mask=support_mask)
    destination = Path(destination)
    temporary = destination.with_name("." + destination.name)
    try:
        nib.save(nib.Nifti1Image(data, img.affine), str(temporary))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
