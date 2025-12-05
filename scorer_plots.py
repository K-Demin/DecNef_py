# decoder_scorer.py
import numpy as np
import nibabel as nib
from pathlib import Path


import numpy as np
import nibabel as nib
from pathlib import Path

class DecoderScorerPlots:
    """
    Single-ROI, single-TR decoder scorer, ATR-style:

      score_t = sum_i ( z_t[i] * w[i] ) + bias

    where z_t[i] is voxel-wise z-score relative to baseline TRs
    (first `baseline_n_tr` volumes).
    """

    def __init__(self, decoder_path: str | Path,
                 baseline_n_tr: int = 20,
                 intercept: float | None = 0.0):
        decoder_path = Path(decoder_path)
        dec_img = nib.load(str(decoder_path))
        dec_data = np.asanyarray(dec_img.dataobj).astype(np.float32)

        # ROI = non-zero & non-NaN decoder weights
        mask = (~np.isnan(dec_data)) & (dec_data != 0)
        self.mask = mask
        self.w_vec_full = dec_data[mask]  # weights for ROI voxels
        self.bias = 0.0 if intercept is None else float(intercept)

        self.baseline_n_tr = baseline_n_tr
        self._baseline_buffer = []  # list of 1D arrays (ROI voxels)
        self._mu = None
        self._sigma = None
        self._valid = None  # mask within ROI voxels where sigma>0

    def _finalize_baseline(self):
        """Compute voxel-wise baseline mean/std and a valid-mask."""
        baseline = np.stack(self._baseline_buffer, axis=0)  # (T0, N_roi)
        mu = baseline.mean(axis=0)
        sigma = baseline.std(axis=0, ddof=1)

        valid = sigma > 0
        if not np.any(valid):
            raise RuntimeError("No voxels with non-zero baseline std in ROI.")

        # restrict everything to voxels with nonzero std
        self._mu = mu[valid]
        self._sigma = sigma[valid]
        self._valid = valid
        self.w_vec = self.w_vec_full[valid]

    def score_volume(self, vol_path: str | Path) -> float:
        """
        Score a single volume NIfTI.

        - During the first `baseline_n_tr` calls, *only* collects baseline.
        - On the call that reaches N baseline TRs, computes baseline stats and returns NaN.
        - After that, returns the decoder dot-product for each new volume.
        """
        img = nib.load(str(vol_path))
        vol_data = np.asanyarray(img.dataobj).astype(np.float32)

        # Extract ROI voxels as 1D
        roi_vals = vol_data[self.mask]

        # --- Baseline phase ---
        if self._mu is None:
            self._baseline_buffer.append(roi_vals)
            if len(self._baseline_buffer) < self.baseline_n_tr:
                # still building baseline; no score yet
                return np.nan
            # This call completed the baseline set
            self._finalize_baseline()
            return np.nan

        # --- Scoring phase ---
        roi_vals = roi_vals[self._valid]
        z = (roi_vals - self._mu) / self._sigma
        score = float(np.dot(z, self.w_vec) + self.bias)
        return score
