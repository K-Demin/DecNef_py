"""Cached per-volume equivalent of smooth_masked in the supplied volume_smooth.py."""
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
from .native_fieldmap import require_same_grid


class MaskedGaussianSmoother:
    """Fixed binary mask; float64 normalized convolution, zero outside the mask.

    Buffers are reused, so call only from the ordered processing stage. The
    returned array belongs to this instance and is overwritten by the next call.
    """

    def __init__(self, mask, affine, fwhm_mm):
        self.mask = np.asarray(mask, dtype=bool).copy()
        self.affine = np.asarray(affine, dtype=float).copy()
        fwhm = float(fwhm_mm)
        if not np.isfinite(fwhm) or fwhm <= 0:
            raise ValueError("Smoother requires finite positive FWHM; bypass it for 0 mm.")
        if self.mask.ndim != 3 or not self.mask.any():
            raise ValueError("Smoothing mask must be a nonempty 3D binary mask.")
        sizes = np.linalg.norm(self.affine[:3, :3], axis=0)
        if not np.isfinite(sizes).all() or np.any(sizes <= 0):
            raise ValueError("Invalid smoothing voxel sizes.")
        self.sigma = fwhm / (np.sqrt(8 * np.log(2)) * sizes)
        density = self._filter(self.mask.astype(np.float64))
        self.bad = density[self.mask] < 1e-6
        self.denominator = np.where(self.bad, 1., density[self.mask])
        self.buffer = np.zeros(self.mask.shape, dtype=np.float64)
        self.filtered = np.empty_like(self.buffer)
        self.output = np.zeros_like(self.buffer)

    def _filter(self, data, output=None):
        return gaussian_filter(data, sigma=self.sigma, output=output,
                               mode="constant", cval=0., truncate=4.)

    def apply(self, data):
        data = np.asarray(data)
        if data.shape != self.mask.shape:
            raise ValueError("Smoothing data and mask shapes differ.")
        values = data[self.mask]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite values inside smoothing mask.")
        self.buffer[self.mask] = values
        self._filter(self.buffer, output=self.filtered)
        result = self.filtered[self.mask] / self.denominator
        result[self.bad] = values[self.bad]
        self.output[self.mask] = result
        return self.output


def prepare_smoother(cfg, fwhm_mm, analysis_space, final_reference, run_command):
    """Prepare the whole-brain mask once, using the same chain as final BOLD."""
    fwhm = float(fwhm_mm)
    if not np.isfinite(fwhm) or fwhm < 0:
        raise ValueError("smoothing_fwhm_mm must be finite and >= 0.")
    if fwhm == 0:
        return None
    source = cfg.rt_unwarped_analysis_ref_mask
    reference = nib.load(str(final_reference))
    folder = cfg.rt_work_dir / "smooth"
    folder.mkdir(parents=True, exist_ok=True)
    mask_path = folder / "mask.nii"
    if analysis_space == "epi":
        mask_img = nib.load(str(source))
    else:
        transforms = [cfg.trans_dir / "epi2t1_Composite.h5"]
        if analysis_space == "mni":
            transforms.insert(0, cfg.subject_root / "anat" / "warp_T1_to_MNI_synth.nii")
        elif analysis_space != "t1":
            raise ValueError(f"Unsupported smoothing space: {analysis_space}")
        for path in [source, *transforms]:
            if not path.exists():
                raise FileNotFoundError(path)
        cmd = ["antsApplyTransforms", "-d", "3", "-i", str(source),
               "-r", str(final_reference), "-o", str(mask_path),
               "-n", "NearestNeighbor", "--float", "1"]
        for transform in transforms:
            cmd.extend(["-t", str(transform)])
        run_command(cmd)
        mask_img = nib.load(str(mask_path))
    require_same_grid(mask_img, reference)
    values = np.asarray(mask_img.dataobj).copy()
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("Smoothing mask must be a finite 3D image.")
    mask = values > 0.5
    smoother = MaskedGaussianSmoother(mask, reference.affine, fwhm)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), reference.affine), str(mask_path))
    return smoother


def smooth_file(smoother, source, destination):
    """Save atomically: PCA consumers poll for complete per-volume files."""
    from pathlib import Path

    img = nib.load(str(source))
    if img.shape != smoother.mask.shape or not np.allclose(img.affine, smoother.affine, atol=1e-4, rtol=0):
        raise ValueError("Final BOLD and smoothing mask grids differ.")
    data = smoother.apply(np.asarray(img.dataobj))
    destination = Path(destination)
    temporary = destination.with_name("." + destination.name)
    try:
        nib.save(nib.Nifti1Image(data, img.affine), str(temporary))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
