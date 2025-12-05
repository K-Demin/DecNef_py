from pathlib import Path
import nibabel as nib
import numpy as np


class DecoderScorer:
    def __init__(
        self,
        decoder_path: str | Path,
        roi_txt: str | Path | None = None,
        n_baseline: int = 20,
    ):
        """
        Real-time decoder scorer.

        If roi_txt is provided and exists:
          - Use ROI_DECODER.txt in ATR-style format:
              line 1: nx ny nz
              line 2: N_voxels
              next N lines: x y z decoder(x,y,z) wmeanVol(x,y,z)
              last line: 0 0 0 bias 0
          - Mask and weights come from the ROI file (preferred).

        Otherwise:
          - Fallback to using the decoder NIfTI as a mask:
              mask = (decoder != 0 & !NaN)
              w    = decoder[mask]
          - Bias = 0 or ROI-derived if available.
        """
        self.decoder_path = Path(decoder_path)
        self.roi_txt = Path(roi_txt) if roi_txt is not None else None
        self.n_baseline = n_baseline

        # --- Core fields filled by _load_from_roi or _load_from_decoder ---
        self.mask = None          # boolean array (nx, ny, nz)
        self.w = None             # weights vector (n_vox,)
        self.bias = 0.0           # scalar bias
        self.template = None      # optional template from ROI (wmeanVol), not strictly needed
        self.dims = None          # (nx, ny, nz)

        # --- Baseline accumulators (for ATR-style normalization) ---
        self.baseline_count = 0
        self.baseline_sum = None
        self.baseline_sum_sq = None
        self.baseline_mean = None
        self.baseline_std = None

        # Try to load from ROI first, fallback to decoder NIfTI
        if self.roi_txt is not None and self.roi_txt.exists():
            self._load_from_roi()
        else:
            self._load_from_decoder_nii()

    # ---------- ROI & decoder loading ----------

    def _load_from_roi(self):
        """
        Parse ROI_DECODER.txt in the same format as export_decoder_roi_txt()
        and build mask, weights, bias, and optional template.
        """
        lines = []
        with open(self.roi_txt, "r") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    lines.append(ln)

        if len(lines) < 3:
            raise ValueError(f"ROI file seems too short: {self.roi_txt}")

        # Line 1: nx ny nz
        nx, ny, nz = map(int, lines[0].split()[:3])
        self.dims = (nx, ny, nz)

        # Line 2: number of voxels
        n_vox = int(lines[1].split()[0])

        if len(lines) < 2 + n_vox + 1:
            raise ValueError(
                f"ROI file {self.roi_txt} inconsistent: "
                f"expected {n_vox} voxel lines but found {len(lines)-2}"
            )

        voxel_lines = lines[2 : 2 + n_vox]
        bias_line = lines[2 + n_vox].split()

        coords = []
        weights = []
        template_vals = []

        for ln in voxel_lines:
            parts = ln.split()
            if len(parts) < 5:
                continue
            # x,y,z are 1-based indices
            x = int(parts[0]) - 1
            y = int(parts[1]) - 1
            z = int(parts[2]) - 1
            w = float(parts[3])
            t = float(parts[4])

            coords.append((x, y, z))
            weights.append(w)
            template_vals.append(t)

        coords = np.array(coords, dtype=int)
        weights = np.asarray(weights, dtype=np.float32)
        template_vals = np.asarray(template_vals, dtype=np.float32)

        mask = np.zeros(self.dims, dtype=bool)
        for (x, y, z) in coords:
            mask[x, y, z] = True

        # Bias from last line, 4th column
        bias = 0.0
        if len(bias_line) >= 4:
            try:
                bias = float(bias_line[3])
            except ValueError:
                bias = 0.0

        self.mask = mask
        self.w = weights
        self.bias = bias
        self.template = template_vals  # not strictly needed, but useful for debugging

    def _load_from_decoder_nii(self):
        """
        Fallback: use decoder NIfTI as mask and weights.

        This is the simpler dot-product version; used only if ROI file is missing.
        """
        dec_img = nib.load(str(self.decoder_path))
        dec_data = np.asanyarray(dec_img.dataobj).astype(np.float32)
        self.dims = dec_data.shape

        mask = (np.abs(dec_data) > 0) & ~np.isnan(dec_data)
        self.mask = mask
        self.w = dec_data[mask].ravel()

        # Bias is zero by default; can be overridden if ROI info exists
        self.bias = 0.0

    # ---------- Baseline handling ----------

    def _ensure_baseline_arrays(self, n_vox: int):
        if self.baseline_sum is None:
            self.baseline_sum = np.zeros(n_vox, dtype=np.float64)
            self.baseline_sum_sq = np.zeros(n_vox, dtype=np.float64)

    def accumulate_baseline(self, vol_arr: np.ndarray):
        """
        Add one baseline volume (already in decoder/template space).

        This mimics ATR-style baseline accumulation: we store per-voxel sums
        and sum-of-squares for the ROI voxels only.
        """
        vox = np.asanyarray(vol_arr, dtype=np.float32)[self.mask].ravel()
        self._ensure_baseline_arrays(len(vox))
        self.baseline_sum += vox
        self.baseline_sum_sq += vox ** 2
        self.baseline_count += 1

    def finalize_baseline(self):
        """
        Compute per-voxel baseline mean and std after enough volumes.
        """
        if self.baseline_count == 0:
            return

        n = float(self.baseline_count)
        mean = self.baseline_sum / n
        var = self.baseline_sum_sq / n - mean ** 2

        # avoid zeros / negative numerical noise
        var[var < 1e-6] = 1e-6

        self.baseline_mean = mean.astype(np.float32)
        self.baseline_std = np.sqrt(var).astype(np.float32)

    @property
    def baseline_ready(self) -> bool:
        return (
            self.baseline_count >= self.n_baseline
            and self.baseline_mean is not None
            and self.baseline_std is not None
        )

    # ---------- Scoring ----------

    def score_from_array(self, vol_arr: np.ndarray, use_z: bool = True):
        """
        vol_arr: 3D volume in the same space as the decoder/ROI.
        Returns (raw_score, z_score).

        - raw_score: dot(vox, w) + bias  (no baseline normalization)
        - z_score:   dot( (vox - mean)/std, w ) + bias
                     (only if baseline_ready and use_z=True; else NaN)
        """
        vox = np.asanyarray(vol_arr, dtype=np.float32)[self.mask].ravel()

        # Raw dot-product decoder value (no baseline normalization)
        raw_score = float(np.dot(vox, self.w) + self.bias)

        # ATR-style normalized decoder value
        if use_z and self.baseline_ready:
            z = (vox - self.baseline_mean) / self.baseline_std
            z_score = float(np.dot(z, self.w) + self.bias)
        else:
            z_score = float("nan")

        return raw_score, z_score

    def score_volume(self, vol_path: str | Path, use_z: bool = True):
        """
        Convenience: load 3D NIfTI from disk and score it.
        """
        img = nib.load(str(vol_path))
        data = np.asanyarray(img.dataobj)
        return self.score_from_array(data, use_z=use_z)
