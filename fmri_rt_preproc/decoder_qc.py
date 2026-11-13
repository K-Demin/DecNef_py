"""Spatial-pattern QC on a fixed decoder ROI, independent of decoder scoring.

Compare the current spatially aligned EPI, before smoothing, denoising and voxel
normalization, with a fixed mean EPI on that same grid. This is a spatial Pearson
correlation across voxels, not a temporal correlation or a weight dot product.
No result here changes a score, regressor, censor decision or processing mask.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json

import nibabel as nib
import numpy as np

from .run_qc_report import write_run_qc_workbook


QC_CSV_NAME = "decoder_qc_rt.csv"
QC_REFERENCE_NAME = "decoder_qc_reference.json"


def _image(source):
    return nib.load(str(source)) if isinstance(source, (str, Path)) else source


def _three_dimensional(image):
    shape = tuple(image.shape)
    if len(shape) == 4 and shape[-1] == 1:
        return np.asanyarray(image.dataobj)[..., 0]
    if len(shape) != 3:
        raise ValueError("Decoder correlation QC requires a 3D image or a singleton 4D image.")
    return np.asanyarray(image.dataobj)


def _grid_matches(image, shape, affine):
    image_shape = tuple(image.shape)
    if len(image_shape) == 4 and image_shape[-1] == 1:
        image_shape = image_shape[:3]
    return image_shape == tuple(shape) and np.allclose(image.affine, affine, rtol=0, atol=1e-4)


def _unit_centered(values):
    """Center in float64, then scale before the norm to avoid overflow.

    Subtracting one observed value before the mean also avoids summing a large
    common BOLD offset. Finite input and a nonconstant spatial pattern are
    checked by the caller; a zero returned norm denotes a constant pattern.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        centered = values - values[0]
    if not np.isfinite(centered).all():
        # Exceptional opposite-signed near-float-limit values: scale first.
        centered = values / np.max(np.abs(values))
    scale = float(np.max(np.abs(centered)))
    if scale == 0:
        return None
    centered /= scale
    centered -= centered.mean(dtype=np.float64)
    norm = float(np.sqrt(np.dot(centered, centered)))
    if norm <= 0 or not np.isfinite(norm):
        return None
    centered /= norm
    return centered


@dataclass(frozen=True)
class DecoderCorrelationResult:
    reference_correlation: float
    reference_coverage_fraction: float
    correlation_status: str
    decoder_weight_loss_fraction: float = float("nan")

    def as_dict(self):
        return asdict(self)


class DecoderSpatialCorrelation:
    """Cache one reference pattern and preserve one ROI for the entire run.

    ``decoder_mask`` is the effective decoder feature mask supplied by the
    scorer. Optional ``decoder_weights`` follow the same flattened mask order
    and measure unavailable squared weight independently of the Pearson r.
    ``fixed_roi_mask`` may restrict that mask once using prepared session brain
    and acquisition coverage. Its exclusions are recorded relative to the full
    decoder. An optional explicit ``roi_image`` is intersected once as well.

    A decoder-construction mean EPI is the preferred reference. The fallback
    ``reference_kind='prepared_session_reference'`` must remain labelled as
    session stability QC, rather than agreement with the training acquisition.
    """

    def __init__(self, reference_image, decoder_mask, *, analysis_image=None,
                 fixed_roi_mask=None, roi_image=None,
                 reference_kind="decoder_training_mean", reference_label=None,
                 fixed_roi_policy=None, decoder_weights=None):
        if reference_kind not in {"decoder_training_mean", "prepared_session_reference"}:
            raise ValueError("Unknown decoder correlation reference kind.")
        reference = _image(reference_image)
        data = _three_dimensional(reference)
        self.shape = tuple(data.shape)
        self.affine = np.asarray(reference.affine, dtype=np.float64).copy()
        if analysis_image is not None and not _grid_matches(_image(analysis_image), self.shape, self.affine):
            raise ValueError("Decoder correlation reference and analysis image grids differ; provide a reference already in the analysis space.")
        decoder = np.asarray(decoder_mask, dtype=bool)
        if decoder.shape != self.shape:
            raise ValueError("Decoder correlation reference and decoder mask shapes differ.")
        roi = decoder.copy()
        if fixed_roi_mask is not None:
            prepared = np.asarray(fixed_roi_mask, dtype=bool)
            if prepared.shape != self.shape:
                raise ValueError("Prepared decoder QC ROI and reference shapes differ.")
            roi &= prepared
        if roi_image is not None:
            explicit = _image(roi_image)
            if not _grid_matches(explicit, self.shape, self.affine):
                raise ValueError("Explicit decoder QC ROI and analysis grids differ.")
            labels = _three_dimensional(explicit)
            if not np.isfinite(labels).all():
                raise ValueError("Explicit decoder QC ROI contains nonfinite values.")
            roi &= labels > 0
        self.roi = roi
        self.roi.setflags(write=False)
        self._indices = np.flatnonzero(roi)
        values = data.ravel()[self._indices].astype(np.float64, copy=False)
        values.setflags(write=False)
        self._reference_values = values
        self._reference_unit = None
        self._reference_status = None
        if not values.size:
            self._reference_status = "empty_roi"
        elif values.size < 2:
            self._reference_status = "insufficient_voxels"
        elif not np.isfinite(values).all():
            self._reference_status = "nonfinite_reference"
        else:
            self._reference_unit = _unit_centered(values)
            if self._reference_unit is None:
                self._reference_status = "constant_reference"
            else:
                self._reference_unit.setflags(write=False)
        self.reference_kind = reference_kind
        self.reference_label = reference_label or (
            "Decoder-construction mean EPI" if reference_kind == "decoder_training_mean"
            else "Prepared session EPI reference (fallback)")
        full_count = int(decoder.sum())
        self._decoder_indices = None
        self._decoder_weight_energy = None
        if decoder_weights is not None:
            weights = np.asarray(decoder_weights, dtype=np.float64)
            if weights.shape != (full_count,):
                raise ValueError("Decoder QC weights must be a 1D array in decoder mask order.")
            scale = np.max(np.abs(weights), initial=0.)
            if np.isfinite(scale) and scale > 0:
                # Scaling before squaring keeps the fraction defined for large weights.
                energy = np.square(weights / scale)
                self._decoder_weight_energy = energy / energy.sum()
                self._decoder_indices = (
                    self._indices if full_count == self._indices.size else np.flatnonzero(decoder)
                )
        self.metadata = {
            "reference_kind": reference_kind, "reference_label": self.reference_label,
            "reference_path": str(Path(reference_image).resolve()) if isinstance(reference_image, (str, Path)) else None,
            "reference_shape": list(self.shape), "reference_affine": self.affine.tolist(),
            "full_decoder_voxel_count": full_count, "fixed_qc_voxel_count": int(self._indices.size),
            "excluded_decoder_fraction": 1 - self._indices.size / full_count if full_count else 0.0,
            "fixed_roi_policy": fixed_roi_policy or "Effective decoder mask intersected once with provided fixed session ROI and optional explicit QC ROI.",
            "explicit_roi_path": str(Path(roi_image).resolve()) if isinstance(roi_image, (str, Path)) else None,
            "comparison_stage": "Spatially transformed EPI before smoothing, denoising and voxel normalization.",
            "correlation_definition": "Unweighted spatial Pearson correlation on the available subset of the fixed QC ROI at each TR.",
            "incomplete_support_policy": "Use covered voxels with finite incoming values; defined correlations on a subset have partial_coverage status. Coverage is relative to the fixed QC ROI.",
            "decoder_weight_loss_definition": "Fraction of full decoder squared weight on features without current spatial coverage or with nonfinite incoming EPI values.",
            "score_and_censor_effect": "None; diagnostic only.",
        }

    def evaluate(self, volume, *, coverage=None, affine=None):
        """Return QC only, including warmup volumes; never alter input arrays.

        ``coverage`` describes the current spatially transformed EPI before
        temporal denoising. Do not substitute cumulative cGLM support here.
        Omitting it assumes acquired support at every decoder voxel; it does
        not infer support from nonzero image intensity.
        """
        if hasattr(volume, "dataobj") and hasattr(volume, "affine"):
            if not _grid_matches(volume, self.shape, self.affine):
                raise ValueError("Incoming EPI and decoder correlation reference grids differ.")
            data = _three_dimensional(volume)
        else:
            data = np.asarray(volume)
        if affine is not None and not np.allclose(affine, self.affine, rtol=0, atol=1e-4):
            raise ValueError("Incoming EPI and decoder correlation reference grids differ.")
        if data.shape != self.shape:
            raise ValueError("Incoming EPI and decoder correlation reference shapes differ.")
        flat_data = data.ravel()
        flat_coverage = None
        if coverage is not None:
            valid = np.asarray(coverage, dtype=bool)
            if valid.shape != self.shape:
                raise ValueError("Current coverage and decoder correlation reference shapes differ.")
            flat_coverage = valid.ravel()
        values = flat_data[self._indices].astype(np.float64, copy=False)
        available = np.isfinite(values)
        if flat_coverage is not None:
            available &= flat_coverage[self._indices]
        count = np.count_nonzero(available)
        fraction = float(count / values.size) if values.size else 0.0
        weight_loss = float("nan")
        if self._decoder_weight_energy is not None:
            decoder_available = available
            if self._decoder_indices is not self._indices:
                decoder_available = np.isfinite(flat_data[self._decoder_indices])
                if flat_coverage is not None:
                    decoder_available &= flat_coverage[self._decoder_indices]
            weight_loss = float(np.clip(self._decoder_weight_energy.sum(where=~decoder_available), 0., 1.))

        status = self._reference_status
        if status is not None:
            return DecoderCorrelationResult(float("nan"), fraction, status, weight_loss)
        if count < 2:
            return DecoderCorrelationResult(float("nan"), fraction, "insufficient_overlap", weight_loss)
        reference_unit = self._reference_unit
        status = "ok"
        if count < values.size:
            reference_unit = _unit_centered(self._reference_values[available])
            if reference_unit is None:
                return DecoderCorrelationResult(float("nan"), fraction, "constant_reference", weight_loss)
            values = values[available]
            status = "partial_coverage"
        unit = _unit_centered(values)
        if unit is None:
            return DecoderCorrelationResult(float("nan"), fraction, "constant_volume", weight_loss)
        correlation = float(np.dot(reference_unit, unit))
        if not np.isfinite(correlation):
            return DecoderCorrelationResult(float("nan"), fraction, "nonfinite_correlation", weight_loss)
        return DecoderCorrelationResult(float(np.clip(correlation, -1, 1)), fraction, status, weight_loss)


def has_decoder_qc(run_dir, *, csv_name=QC_CSV_NAME):
    return (Path(run_dir) / csv_name).is_file()


def add_reference_correlation_panel(ax, run_dir, *, csv_name=QC_CSV_NAME):
    """Draw all logged TRs, including warmup, on a caller-owned QC axis.

    Returns False when no QC CSV exists. Defined partial-coverage correlations
    share the main trace. Unavailable correlations appear as gaps.
    """
    run_dir = Path(run_dir)
    path = run_dir / csv_name
    ax.set_ylabel("Spatial Pearson r")
    ax.set_xlabel("Volume / TR")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=.2)
    if not path.is_file():
        ax.set_title("Decoder ROI spatial stability · unavailable")
        return False
    metadata = {}
    meta_path = run_dir / QC_REFERENCE_NAME
    if meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    points = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            try:
                point = (int(row["volume_idx"]), float(row["reference_correlation"]),
                         float(row["reference_coverage_fraction"]), row.get("correlation_status", ""))
            except (KeyError, TypeError, ValueError):
                continue
            try:
                weight_loss = float(row.get("decoder_weight_loss_fraction") or "nan")
            except (TypeError, ValueError):
                weight_loss = float("nan")
            points.append((*point, weight_loss))
    points.sort(key=lambda p: p[0])
    label = metadata.get("reference_label", "Fixed EPI reference")
    ax.set_title("Decoder ROI spatial stability · " + label)
    if metadata.get("reference_kind") == "prepared_session_reference":
        ax.set_title("Decoder ROI spatial stability · session-reference fallback")
    if not points:
        return False
    x = np.array([p[0] for p in points])
    y = np.array([p[1] if p[3] in {"ok", "partial_coverage"} and np.isfinite(p[1]) else np.nan
                  for p in points])
    ax.plot(x, y, color="#246a8c", linewidth=1.25, marker=".", markersize=3,
            label="EPI correlation on available QC voxels")
    coverage_axis = ax.twinx()
    coverage_axis.plot(x, [p[2] * 100 for p in points], color="#b88645", linestyle=":", linewidth=1., alpha=.85, label="Fixed QC ROI covered (%)")
    weight_loss_percent = np.array([p[4] * 100 for p in points])
    if np.isfinite(weight_loss_percent).any():
        coverage_axis.plot(x, weight_loss_percent, color="#974368", linestyle="--", linewidth=1.,
                           label="Full decoder squared weight missing (%)")
    coverage_axis.set_ylim(-2, 105)
    coverage_axis.set_ylabel("Percent", fontsize=9)
    coverage_axis.tick_params(axis="y", labelsize=8)
    unavailable = np.isnan(y)
    if unavailable.any():
        ax.scatter(x[unavailable], np.full(unavailable.sum(), -.99), marker="x", s=18, color="#b55b42", label="Correlation unavailable · see status/coverage")
    handles, labels = ax.get_legend_handles_labels()
    extra_handles, extra_labels = coverage_axis.get_legend_handles_labels()
    ax.legend(handles + extra_handles, labels + extra_labels, loc="lower left", fontsize=8)
    n = metadata.get("fixed_qc_voxel_count")
    total = metadata.get("full_decoder_voxel_count")
    if n is not None and total is not None:
        ax.text(.99, .02, f"Fixed QC ROI: {n:,}/{total:,} decoder voxels", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    return True
