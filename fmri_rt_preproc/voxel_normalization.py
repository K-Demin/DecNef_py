"""Voxel-wise temporal normalization with explicit observation support.

Statistics require O(number of voxels) memory. Call ``transform`` before
``observe`` when the current volume must be excluded from its own reference.
The caller selects what to observe: denoised values for temporal z-scoring,
or raw motion-corrected intensities for percent-of-raw-mean residual scaling.
"""

import numpy as np


class VoxelNormalizer:
    """Normalize using a fixed startup reference or an expanding past reference.

    ``reference_count`` counts observed volumes with at least one available
    brain voxel. ``count`` instead records each voxel's actual observations;
    missing acquisition coverage never contributes a zero measurement. A fixed
    reference freezes after ``ref_volumes`` observed volumes. Running references
    continue updating, but use the same startup requirement.

    ``mean`` and ``std`` are float64 arrays, and ``count`` is int64, on the input
    grid. They are None until a grid is established (by a brain mask or data).
    The sample SD is zero where fewer than two observations exist. Copy these
    arrays when retaining a calibration snapshot; mean/count update in place.
    ``output_support`` describes only the most recent transformed volume.
    This stateful object must be called in processing order, not concurrently.
    """

    def __init__(self, mode="zscore", reference="fixed", ref_volumes=30,
                 brain_mask=None):
        if mode not in {"zscore", "mean"}:
            raise ValueError("Voxel normalization mode must be 'zscore' or 'mean'.")
        if reference not in {"fixed", "running"}:
            raise ValueError("Voxel normalization reference must be 'fixed' or 'running'.")
        try:
            n_reference = int(ref_volumes)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Reference volume count must be an integer.") from exc
        if isinstance(ref_volumes, (bool, np.bool_)) or n_reference != ref_volumes:
            raise ValueError("Reference volume count must be an integer.")
        minimum = 2 if mode == "zscore" else 1
        if n_reference < minimum:
            raise ValueError(f"{mode} normalization needs at least {minimum} reference volumes.")
        self.mode = mode
        self.reference = reference
        self.ref_volumes = n_reference
        self.reference_count = 0
        self.brain_mask = None
        self.mean = None
        self.count = None
        self._m2 = None
        self.output_support = None
        if brain_mask is not None:
            self.brain_mask = self._binary_mask(brain_mask, "Brain mask").copy()
            if self.brain_mask.ndim != 3:
                raise ValueError("Brain mask must be a 3D array.")
            self._initialize(self.brain_mask.shape)

    @staticmethod
    def _binary_mask(mask, label):
        array = np.asarray(mask)
        if array.dtype.kind != "b" and (
                array.dtype.kind not in "iuf" or not np.isfinite(array).all() or
                not np.all((array == 0) | (array == 1))):
            raise ValueError(f"{label} must be finite and binary (0/1).")
        return array.astype(bool, copy=False)

    def _initialize(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.count = np.zeros(shape, dtype=np.int64)
        self._m2 = np.zeros(shape, dtype=np.float64)
        self.output_support = np.zeros(shape, dtype=bool)

    def _input(self, data, support_mask):
        values = np.asarray(data)
        if values.ndim != 3 or values.dtype.kind not in "biuf":
            raise ValueError("Voxel normalization data must be a real 3D numeric array.")
        if self.mean is not None and values.shape != self.mean.shape:
            raise ValueError("Voxel normalization data shape differs from its reference grid.")
        if support_mask is None:
            observed = np.ones(values.shape, dtype=bool)
        else:
            observed = self._binary_mask(support_mask, "Normalization support")
            if observed.shape != values.shape:
                raise ValueError("Normalization support and data shapes differ.")
        if self.brain_mask is not None:
            observed = observed & self.brain_mask
        if not np.isfinite(values[observed]).all():
            raise ValueError("Nonfinite acquired values inside the normalization mask.")
        if self.mean is None:
            self._initialize(values.shape)
        return values, observed

    @property
    def ready(self):
        """Whether the reference has received enough observed volumes."""
        return self.reference_count >= self.ref_volumes

    @property
    def std(self):
        """Current sample SD; zero for voxels with fewer than two observations."""
        if self.mean is None:
            return None
        variance = np.divide(self._m2, self.count - 1,
                             out=np.zeros_like(self._m2), where=self.count >= 2)
        np.maximum(variance, 0., out=variance)
        return np.sqrt(variance, out=variance)

    def observe(self, data, support_mask=None):
        """Update reference statistics from available samples, without scoring.

        Completely unavailable volumes do not advance the reference. Input is
        still validated after a fixed reference freezes, but its statistics
        remain unchanged. Acquired zeros and negative values are observations.
        """
        values, observed = self._input(data, support_mask)
        if (self.reference == "fixed" and self.ready) or not observed.any():
            return
        sample = np.asarray(values[observed], dtype=np.float64)
        counts = self.count[observed] + 1
        delta = sample - self.mean[observed]
        new_mean = self.mean[observed] + delta / counts
        self._m2[observed] += delta * (sample - new_mean)
        self.mean[observed] = new_mean
        self.count[observed] = counts
        self.reference_count += 1

    def transform(self, data, support_mask=None):
        """Return float32 normalized values using existing statistics only.

        Before startup completes, and wherever calibration or current coverage
        is insufficient, output is zero and output_support is false. No valid
        normalized values are clipped. Calling this method never updates the
        reference, so the caller controls causal observation order explicitly.
        """
        values, observed = self._input(data, support_mask)
        out = np.zeros(values.shape, dtype=np.float32)
        self.output_support.fill(False)
        if not self.ready:
            return out
        if self.mode == "zscore":
            scale = self.std
            valid = observed & (self.count >= 2) & np.isfinite(self.mean) & np.isfinite(scale) & (scale > 1e-6)
            normalized = (values[valid].astype(np.float64) - self.mean[valid]) / scale[valid]
        else:
            valid = observed & (self.count >= 1) & np.isfinite(self.mean) & (np.abs(self.mean) > 1e-6)
            normalized = values[valid].astype(np.float64) / self.mean[valid] * 100.
        out[valid] = normalized
        self.output_support[:] = valid
        return out
